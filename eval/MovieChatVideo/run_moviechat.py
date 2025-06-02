#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import json
import time
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import gc
import torch
# Import MovieChat modules
from MovieChat.processors.video_processor import AlproVideoEvalProcessor
from MovieChat.models.chat_model import Chat
from MovieChat.models.moviechat import MovieChat

class MovieChatModel:
    """Class to handle running MovieChat on videos."""
    
    # Default prompts for different categories
    CATEGORY_PROMPTS = {
        "words": "This video contains text encoded through temporal patterns. What specific word or phrase do you see? Respond with just the text.",
        "shapes": "This video contains a object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "images": "This video contains a common object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "videos": "This video contains movement encoded through temporal patterns. What is shown? Respond with just 1-3 words describing what you see."
    }
    
    COT_PROMPTS = {
        "words": "This video encodes text through temporal patterns. To identify it: 1) Look for areas where opposing motion patterns reveal letters. 2) Focus on the overall word or phrase that emerges. 3) Read the specific text content. \n\nPlease respond with just the text you identify.",
        "shapes": "This video encodes a object through temporal patterns. To identify it: 1) Look for areas where motion reveals contours and edges. 2) Focus on the geometric outline that emerges. 3) Determine what object is shown. \n\nPlease respond with just the object name.",
        "images": "This video encodes an object through temporal patterns. To identify it: 1) Look for areas where motion reveals object outlines. 2) Focus on the object silhouette that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "videos": "This video encodes movement through temporal patterns. To identify it: 1) Look for areas where motion patterns reveal action. 2) Focus on the movement that emerges. 3) Determine what is being shown. \n\nPlease respond with just 1-3 words describing what you see."
    }
    
    def __init__(self, temp_dir=None, n_frames=8, image_size=224):
        """Initialize the MovieChat model.
        
        Args:
            temp_dir (str): Directory to store temporary fragments
            n_frames (int): Number of frames to process
            image_size (int): Size of input frames
        """
        # Automatically select the device (CUDA if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "temp_fragments")
        self.n_frames = n_frames
        self.image_size = image_size
        self.model = None
        self.chat = None
    
    def load(self):
        """Load the MovieChat model."""
        print(f"Loading MovieChat model...")
        
        # Initialize MovieChat
        self.model = MovieChat.from_config(device=self.device).to(self.device)
        vis_processor_cfg = {
            'name': 'alpro_video_eval', 
            'n_frms': self.n_frames, 
            'image_size': self.image_size
        }
        frame_processor = AlproVideoEvalProcessor.from_config(vis_processor_cfg)
        self.chat = Chat(self.model, frame_processor, device=self.device)
        
        print("MovieChat model loaded successfully")
        return self
    
    def process_video(self, video_path, prompt=None, category=None, use_cot=False):
        """Process a video with MovieChat.
        
        Args:
            video_path (str): Path to the video file
            prompt (str, optional): Custom prompt to use
            category (str, optional): Video category for selecting default prompt
            use_cot (bool): Whether to use chain-of-thought prompting
            
        Returns:
            str: Model response
        """
        if self.model is None or self.chat is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return "ERROR: Video file not found"
        
        # Use provided prompt or select based on category
        if prompt is None and category is not None:
            prompt_dict = self.COT_PROMPTS if use_cot else self.CATEGORY_PROMPTS
            if category in prompt_dict:
                prompt = prompt_dict[category]
            else:
                prompt = "What is shown in this video? Please describe in detail."
        elif prompt is None:
            prompt = "What is shown in this video? Please describe in detail."
            
        # Create a fragment path for temporary storage
        os.makedirs(self.temp_dir, exist_ok=True)
        basename = os.path.basename(video_path)
        fragment_path = os.path.join(self.temp_dir, f"temp_{int(time.time())}_{basename}")
        
        # Copy the video to fragment path
        shutil.copy2(video_path, fragment_path)
        print(f"Copied video to temporary location: {fragment_path}")
        
        # Process video
        print(f"Processing video: {video_path}")
        
        # Initialize video processing
        cap = cv2.VideoCapture(video_path)
        cur_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_fps)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from {video_path}")
            
        # Process frame and prepare for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image = self.chat.image_vis_processor(pil_image).unsqueeze(0).unsqueeze(2).half().to(self.device)
        cur_image = self.model.encode_image(image)
        
        # Run inference
        print(f"Running inference with prompt: {prompt}")
        img_list = []
        
        # Note about frame limit: MovieChat has a limit of 128 frames but we can't control it directly
        # The upload_video_without_audio() doesn't accept a max_frames parameter
        
        msg = self.chat.upload_video_without_audio(
            video_path=video_path, 
            fragment_video_path=fragment_path,
            cur_min=0, 
            cur_sec=0, 
            cur_image=cur_image, 
            img_list=img_list, 
            middle_video=False,
            question=prompt
        )
        
        # Generate response
        answer = self.chat.answer(
            img_list=img_list,
            input_text=prompt,
            msg=msg,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000
        )[0]
        
        # Clean up the fragment file
        if os.path.exists(fragment_path):
            os.remove(fragment_path)
            print(f"Removed temporary file: {fragment_path}")
            
        # Clean up model caches to avoid memory issues
        torch.cuda.empty_cache()
        
        return answer

def is_match(prediction, ground_truth_labels):
    """Check if any ground truth label is contained in the prediction.
    
    Args:
        prediction (str): Model prediction
        ground_truth_labels (list): List of acceptable labels
        
    Returns:
        bool: True if any ground truth label is found in prediction
    """
    # Clean and normalize prediction
    prediction = prediction.lower().strip()
    
    # Check if any ground truth label is in the prediction
    for label in ground_truth_labels:
        # For single label
        if isinstance(label, str) and label.lower() in prediction:
            return True
        
        # For list of acceptable labels
        if isinstance(label, list):
            for sublabel in label:
                if sublabel.lower() in prediction:
                    return True
    
    return False

def process_dataset(model, csv_path, dataset_dir, categories=None, use_cot=False, sample_size=None):
    """Process videos in the dataset and collect model outputs.
    
    Args:
        model (MovieChatModel): Loaded model
        csv_path (str): Path to SpookyBench metadata CSV
        dataset_dir (str): Directory containing SpookyBench videos
        categories (list): List of categories to process (None for all)
        use_cot (bool): Whether to use chain-of-thought prompts
        sample_size (int): Number of videos to sample per category
        
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
        
        for _, row in tqdm(category_df.iterrows(), total=len(category_df)):
            video_path = row["Path"]
            full_video_path = os.path.join(dataset_dir, video_path)
            
            if not os.path.exists(full_video_path):
                print(f"Video file not found: {full_video_path}")
                continue
                
            # Process video with model - no try/except to let errors propagate naturally
            response = model.process_video(
                full_video_path,
                category=category,
                use_cot=use_cot
            )
            
            # Store result
            results[video_path] = response
            
            # Wait between videos
            time.sleep(5)
            
            # Aggressive memory cleanup
            print("Performing memory cleanup...")
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reset CUDA context if there are multiple GPUs (helps with memory fragmentation)
            if torch.cuda.device_count() > 1:
                torch.cuda.synchronize()
                
            # Wait between videos
            print("Waiting between videos...")
            time.sleep(5)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run MovieChat on SpookyBench videos")
    
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
    # Removed --device argument to let PyTorch automatically select an available GPU
    parser.add_argument("--n_frames", type=int, default=8,
                        help="Number of frames to process")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size of input frames")
    parser.add_argument("--reset_per_category", action="store_true",
                        help="Reset model between categories to prevent memory issues")
    
    # Custom prompt
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to use for all videos (overrides category-specific prompts)")
    
    # Temporary directory
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Directory to store temporary fragments (default: ./temp_fragments)")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Metadata CSV file not found: {args.csv}")
        return
    
    # Setting reset_per_category to True by default since it helps with memory issues
    reset_per_category = True
    
    results = {}
    
    # Process videos based on input type
    if os.path.isfile(args.dataset):
        # Single video processing
        video_path = args.dataset
        print(f"Processing single video: {video_path}")
        
        # Initialize and load model
        model = MovieChatModel(
            temp_dir=args.temp_dir,
            n_frames=args.n_frames,
            image_size=args.image_size
        ).load()
        print(f"Using device: {model.device}")
        
        # Use custom prompt if provided, otherwise use a generic prompt
        prompt = args.prompt if args.prompt else "What is shown in this video? Please describe in detail."
        
        response = model.process_video(
            video_path,
            prompt=prompt
        )
        results[os.path.basename(video_path)] = response
    else:
        # Dataset processing
        if not os.path.isdir(args.dataset):
            print(f"Dataset directory not found: {args.dataset}")
            return
            
        print(f"Processing videos from dataset: {args.dataset}")
        
        # Load metadata
        df = pd.read_csv(args.csv)
        
        # Filter by categories if specified
        categories = args.categories if args.categories else df["Category"].unique()
        print(f"Will process these categories: {categories}")
        
        # Process each category separately to allow model reset between categories
        for category in categories:
            print(f"\n{'='*30}\nProcessing category: {category}\n{'='*30}")
            
            # Reset model for each category to avoid memory issues
            # This is crucial to prevent the "index out of bounds" error
            print("Initializing fresh model for this category...")
            model = MovieChatModel(
                temp_dir=args.temp_dir,
                n_frames=args.n_frames,
                image_size=args.image_size
            ).load()
            print(f"Using device: {model.device}")
            
            # Clear memory before starting a new category
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)  # Give system time to stabilize
            
            # Filter for current category
            category_df = df[df["Category"] == category]
            
            # Sample if needed
            if args.sample_size and args.sample_size < len(category_df):
                category_df = category_df.sample(args.sample_size, random_state=42)
            
            print(f"Processing {len(category_df)} videos in category '{category}'...")
            
            # Process all videos in this category
            for _, row in tqdm(category_df.iterrows(), total=len(category_df)):
                video_path = row["Path"]
                full_video_path = os.path.join(args.dataset, video_path)
                
                if not os.path.exists(full_video_path):
                    print(f"Video file not found: {full_video_path}")
                    continue
                
                # The key issue appears to be the model not properly resetting between videos
                # Let's fully reinitialize the model for EACH video to prevent the index error
                print(f"\n{'*'*30}\nREINITIALIZING MODEL FOR NEW VIDEO\n{'*'*30}\n")
                
                # First, clean up the old model
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(3)  # Give system time to fully release resources
                
                # Create a fresh model instance for this video
                model = MovieChatModel(
                    temp_dir=args.temp_dir,
                    n_frames=args.n_frames,
                    image_size=args.image_size
                ).load()
                
                # Process the video with appropriate prompt - no try/except to let errors propagate naturally
                if args.prompt:
                    response = model.process_video(
                        full_video_path,
                        prompt=args.prompt
                    )
                else:
                    response = model.process_video(
                        full_video_path,
                        category=category,
                        use_cot=args.use_cot
                    )
                
                # Store result
                results[video_path] = response
                
                # Explicitly destroy the model after each video
                del model
                model = None
                
                # Aggressive memory cleanup after each video
                print("Performing aggressive memory cleanup...")
                gc.collect()
                torch.cuda.empty_cache()
                
                # Extended wait between videos
                print("Waiting between videos...")
                time.sleep(10)  # Longer wait to ensure full cleanup
            
            # Wait longer between categories and release model resources
            print(f"Finished category: {category}. Cleaning up before next category...")
            model = None  # Release model resources
            gc.collect()
            torch.cuda.empty_cache()
            print("Waiting between categories...")
            time.sleep(15)  # Longer wait between categories
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_type = "cot" if args.use_cot else "direct"
    output_name = f"moviechat_{prompt_type}_{timestamp}.json"
    
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

