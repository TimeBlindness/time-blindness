#!/usr/bin/env python3

import os
import sys
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model handling for various Qwen VL variants
class QwenVLModel:
    """Class to handle loading and running different variants of Qwen VL models."""
    
    # Map of model IDs to HuggingFace model paths
    MODEL_VARIANTS = {
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct"
    }
    
    # Default prompts for different categories
    CATEGORY_PROMPTS = {
        "words": "This video contains text encoded through temporal patterns. What specific word or phrase do you see? Respond with just the text.",
        "shapes": "This video contains a geometric object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "images": "This video contains a common object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "videos": "This video contains movement encoded through temporal patterns. What is shown? Respond with just 1-3 words describing what you see."
    }
    
    COT_PROMPTS = {
        "words": "This video encodes text through temporal patterns. To identify it: 1) Look for areas where opposing motion patterns reveal letters. 2) Focus on the overall word or phrase that emerges. 3) Read the specific text content. \n\nPlease respond with just the text you identify.",
        "shapes": "This video encodes a geometric object through temporal patterns. To identify it: 1) Look for areas where motion reveals contours and edges. 2) Focus on the geometric outline that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "images": "This video encodes an object through temporal patterns. To identify it: 1) Look for areas where motion reveals object outlines. 2) Focus on the object silhouette that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "videos": "This video encodes movement through temporal patterns. To identify it: 1) Look for areas where motion patterns reveal action. 2) Focus on the movement that emerges. 3) Determine what is being shown. \n\nPlease respond with just 1-3 words describing what you see."
    }
    
    def __init__(self, model_id, precision="bfloat16"):
        """Initialize the Qwen VL model.
        
        Args:
            model_id (str): ID of the model to load
            precision (str): Precision to use ("fp16", "bfloat16", or "fp32")
        """
        self.model_id = model_id.lower()
        self.precision = precision
        
        # Validate model ID
        if self.model_id not in self.MODEL_VARIANTS:
            raise ValueError(f"Unknown model ID: {model_id}. Available models: {list(self.MODEL_VARIANTS.keys())}")
        
        # Determine torch dtype based on precision
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # Set up device - use CUDA if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model variables to be loaded later
        self.model = None
        self.processor = None
    
    def load(self):
        """Load the model and processor."""
        print(f"Loading {self.model_id} from {self.MODEL_VARIANTS[self.model_id]}...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_VARIANTS[self.model_id],
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_VARIANTS[self.model_id],
            trust_remote_code=True
        )
        
        print(f"Successfully loaded {self.model_id}")
        return self
    
    def process_video(self, video_path, prompt=None, category=None, use_cot=False, fps=30.0, max_pixels=1280*28*28):
        """Process a video and run inference on it.
        
        Args:
            video_path (str): Path to the video file
            prompt (str, optional): Custom prompt to use. If None, uses a default prompt
            category (str, optional): Video category for selecting default prompt
            use_cot (bool): Whether to use chain-of-thought prompting
            fps (float): Frames per second to sample from the video
            max_pixels (int): Maximum number of pixels per frame
            
        Returns:
            str: Model response
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            raise ValueError("Video file not found")
        
        # Prepare message with video and prompt
        print(f"Processing video: {video_path}")
        
        # Use provided prompt or select based on category
        if prompt is None and category is not None:
            prompt_dict = self.COT_PROMPTS if use_cot else self.CATEGORY_PROMPTS
            if category in prompt_dict:
                prompt = prompt_dict[category]
            else:
                prompt = "What is shown in this video? Please describe in detail."
        elif prompt is None:
            prompt = "What is shown in this video? Please describe in detail."
            
        # Create message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels,
                        "fps": fps
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process message
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Run inference
        print(f"Running inference with prompt: {prompt}")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Return the raw output - let's be transparent about what the model actually returns
        return output_text.strip()

def process_dataset(model, csv_path, dataset_dir, categories=None, use_cot=False, sample_size=None, fps=10.0, max_pixels=1280*28*28):
    """Process videos in the dataset and collect model outputs.
    
    Args:
        model (QwenVLModel): Loaded model
        csv_path (str): Path to SpookyBench metadata CSV
        dataset_dir (str): Directory containing SpookyBench videos
        categories (list): List of categories to process (None for all)
        use_cot (bool): Whether to use chain-of-thought prompts
        sample_size (int): Number of videos to sample per category
        fps (float): Frames per second to sample from videos
        max_pixels (int): Maximum number of pixels per frame
        
    Returns:
        dict: Dictionary mapping video paths to model outputs
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} videos from SpookyBench metadata")
    
    # Filter by categories if specified
    if categories:
        df = df[df["Category"].isin(categories)]
        print(f"Filtered to {len(df)} videos in categories: {categories}")
    
    # Process all videos and collect outputs
    results = {}
    
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        
        # Sample if needed
        if sample_size and sample_size < len(category_df):
            category_df = category_df.sample(sample_size, random_state=42)
        
        print(f"Processing {len(category_df)} videos in category '{category}'...")
        
        for _, row in category_df.iterrows():
            video_path = row["Path"]
            full_video_path = os.path.join(dataset_dir, video_path)
            
            if not os.path.exists(full_video_path):
                print(f"Video file not found: {full_video_path}")
                continue
                
            # Process video with model
            response = model.process_video(
                full_video_path,
                category=category,
                use_cot=use_cot,
                fps=fps,
                max_pixels=max_pixels
            )
            
            # Store result
            results[video_path] = response
    
    return results

def main():
    """Main function to run Qwen VL models with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Qwen VL models on SpookyBench videos")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        choices=list(QwenVLModel.MODEL_VARIANTS.keys()),
                        help="Qwen VL model variant to use")
    parser.add_argument("--precision", type=str, default="bfloat16", 
                        choices=["fp16", "bfloat16", "fp32"],
                        help="Model precision to use")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to SpookyBench dataset directory or a single video file")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to SpookyBench metadata CSV")
    
    # Processing arguments
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=["words", "shapes", "images", "videos"],
                        help="Categories to process (if not specified, all categories are used)")
    parser.add_argument("--use_cot", action="store_true",
                        help="Use chain-of-thought prompting instead of direct prompting")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample per category (if not specified, all videos are used)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second to sample from video (lower = faster processing)")
    parser.add_argument("--max_pixels", type=int, default=1280*28*28,
                        help="Maximum number of pixels per frame (lower = faster processing)")
    
    # Custom prompt
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to use for all videos (overrides category-specific prompts)")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Metadata CSV file not found: {args.csv}")
        return
    
    # Initialize and load model
    model = QwenVLModel(model_id=args.model, precision=args.precision).load()
    print(f"Using device: {model.device}")
    
    results = {}
    
    # Process videos based on input type
    if os.path.isfile(args.dataset):
        # Single video processing
        video_path = args.dataset
        print(f"Processing single video: {video_path}")
        
        # Use custom prompt if provided, otherwise use a generic prompt
        prompt = args.prompt if args.prompt else "What is shown in this video? Please describe in detail."
        
        response = model.process_video(
            video_path,
            prompt=prompt,
            fps=args.fps,
            max_pixels=args.max_pixels
        )
        results[os.path.basename(video_path)] = response
    else:
        # Dataset processing
        if not os.path.isdir(args.dataset):
            print(f"Dataset directory not found: {args.dataset}")
            return
            
        print(f"Processing videos from dataset: {args.dataset}")
        
        # Process with custom prompt if provided
        if args.prompt:
            print(f"Using custom prompt for all videos: {args.prompt}")
            
            # Load metadata and filter by categories if needed
            df = pd.read_csv(args.csv)
            if args.categories:
                df = df[df["Category"].isin(args.categories)]
                
            # Sample if needed
            if args.sample_size:
                sample_dfs = []
                for category in df["Category"].unique():
                    cat_df = df[df["Category"] == category]
                    if len(cat_df) > args.sample_size:
                        cat_df = cat_df.sample(args.sample_size, random_state=42)
                    sample_dfs.append(cat_df)
                df = pd.concat(sample_dfs)
                
            # Process each video with the custom prompt
            for _, row in df.iterrows():
                video_path = row["Path"]
                full_video_path = os.path.join(args.dataset, video_path)
                
                if not os.path.exists(full_video_path):
                    print(f"Video file not found: {full_video_path}")
                    continue
                    
                response = model.process_video(
                    full_video_path,
                    prompt=args.prompt,
                    fps=args.fps,
                    max_pixels=args.max_pixels
                )
                results[video_path] = response
        else:
            # Process with category-specific prompts
            results = process_dataset(
                model=model,
                csv_path=args.csv,
                dataset_dir=args.dataset,
                categories=args.categories,
                use_cot=args.use_cot,
                sample_size=args.sample_size,
                fps=args.fps,
                max_pixels=args.max_pixels
            )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_type = "cot" if args.use_cot else "direct"
    output_name = f"{args.model}_{prompt_type}_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, output_name)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. {len(results)} videos processed.")
    print(f"Results saved to {output_path}")
    
    # Print command to analyze results
    print(f"To analyze results, run: python analyze_results.py --results {output_path} --csv {args.csv} --output {args.output}/analysis_{timestamp}")

if __name__ == "__main__":
    main()