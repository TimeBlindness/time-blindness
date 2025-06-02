#!/usr/bin/env python3

import os
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_video_info_cv2(video_path):
    """Get video information using OpenCV when Decord fails."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def read_frame_cv2(video_path, frame_idx):
    """Read a specific frame using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_index(bound, fps, max_frame, first_idx=0, num_segments=8):
    """Calculate frame indices for sampling."""
    import numpy as np
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = 0, float(max_frame) / fps
    
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def build_transform(input_size=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """Build image transformation pipeline."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for image processing."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Preprocess images with dynamic tiling based on aspect ratio."""
    from PIL import Image
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the possible aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load video frames for InternVL model processing.
    
    Args:
        video_path (str): Path to the video file
        bound (tuple, optional): Time bounds (start, end) in seconds
        input_size (int): Size of the input frames
        max_num (int): Maximum number of tiles per frame
        num_segments (int): Number of video segments to sample
        
    Returns:
        tuple: (pixel_values, num_patches_list)
    """
    try:
        import numpy as np
        import decord
        from decord import VideoReader, cpu
        from PIL import Image
        
        # First try Decord
        vr = None
        if os.path.exists(video_path):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
            except Exception as e:
                logger.warning(f"Decord failed to open {video_path}, using OpenCV instead: {str(e)}")
        else:
            raise ValueError(f"Video file not found: {video_path}")
        
        if vr is None:
            # Fallback to OpenCV
            max_frame, fps = get_video_info_cv2(video_path)
        else:
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
        
        logger.info(f"Video has {max_frame+1} frames at {fps} FPS")
        
        if max_frame <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

        for frame_index in frame_indices:
            if vr is not None:
                frame = vr[frame_index].asnumpy()
            else:
                frame = read_frame_cv2(video_path, frame_index)
                
            img = Image.fromarray(frame).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        if not pixel_values_list:
            raise ValueError("Failed to load any frames from the video")

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
        
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {str(e)}")
        raise

class InternVLModel:
    """Class to handle loading and running InternVL models on videos."""
    
    # Map of model IDs to HuggingFace model paths
    MODEL_VARIANTS = {
        "internvl2-40b": "OpenGVLab/InternVL2-40B",
        "internvl2-8b": "OpenGVLab/InternVL2-8B",
        "internvl2.5-78b": "OpenGVLab/InternVL2_5-78B",
        "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
        "internvideo2-chat-8b": "OpenGVLab/InternVideo2-Chat-8B",
        "internvl2.5-2b": "OpenGVLab/InternVL2_5-2B"
    }
    
    # Default prompts for different categories
    CATEGORY_PROMPTS = {
        "words": "This video contains text encoded through temporal patterns. What specific word or phrase do you see? Respond with just the text.",
        "shapes": "This video contains a object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "images": "This video contains a common object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "videos": "This video contains movement encoded through temporal patterns. What is shown? Respond with just 1-3 words describing what you see."
    }
    
    COT_PROMPTS = {
        "words": "This video encodes text through temporal patterns. Look for words that emerge from the motion and identify the specific text shown. Respond with just the text you identify.",
        "shapes": "This video encodes a object through temporal patterns. Look for the object outline that emerges from the motion and identify it. Respond with just the object name.",
        "images": "This video encodes an object through temporal patterns. Look for the object silhouette that emerges from the motion and identify it. Respond with just the object name.",
        "videos": "This video encodes movement through temporal patterns. Look for the action or object that emerges from the motion and identify it. Respond with just 1-3 words describing what you see."
    }
    
    def __init__(self, model_id, precision="bfloat16"):
        """Initialize the InternVL model.
        
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
        self.tokenizer = None
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    def load(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading {self.model_id} from {self.MODEL_VARIANTS[self.model_id]}...")
        
        self.model = AutoModel.from_pretrained(
            self.MODEL_VARIANTS[self.model_id],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_VARIANTS[self.model_id], 
            trust_remote_code=True, 
            use_fast=False
        )
        
        logger.info(f"Successfully loaded {self.model_id}")
        return self
    
    def process_video(self, video_path, prompt=None, category=None, use_cot=False, num_segments=32, max_num=1, input_size=448):
        """Process a video and run inference on it.
        
        Args:
            video_path (str): Path to the video file
            prompt (str, optional): Custom prompt to use. If None, uses a default prompt
            category (str, optional): Video category for selecting default prompt
            use_cot (bool): Whether to use chain-of-thought prompting
            num_segments (int): Number of video segments to sample
            
        Returns:
            str: Model response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return "ERROR: Video file not found"
        
        # Load video frames
        logger.info(f"Processing video: {video_path}")
        pixel_values, num_patches_list = load_video(
            video_path, 
            num_segments=num_segments, 
            max_num=max_num,
            input_size=input_size
        )
        pixel_values = pixel_values.to(self.dtype).to(self.device)
        
        # Create frame prefix for prompt
        video_prefix = ''.join([f'Frame-{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        # Use provided prompt or select based on category
        if prompt is None and category is not None:
            prompt_dict = self.COT_PROMPTS if use_cot else self.CATEGORY_PROMPTS
            if category in prompt_dict:
                prompt = prompt_dict[category]
            else:
                prompt = "What is shown in this video? Please describe in detail."
        elif prompt is None:
            prompt = "What is shown in this video? Please describe in detail."
            
        question = video_prefix + prompt
        
        # Run inference
        logger.info(f"Running inference with prompt: {prompt}")
        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, self.generation_config,
            num_patches_list=num_patches_list, history=None, return_history=True
        )
        
        return response

def process_dataset(model, csv_path, dataset_dir, categories=None, use_cot=False, sample_size=None, num_segments=8, max_num=4, input_size=224):
    """Process videos in the dataset and collect model outputs.
    
    Args:
        model (InternVLModel): Loaded model
        csv_path (str): Path to SpookyBench metadata CSV
        dataset_dir (str): Directory containing SpookyBench videos
        categories (list): List of categories to process (None for all)
        use_cot (bool): Whether to use chain-of-thought prompts
        sample_size (int): Number of videos to sample per category
        num_segments (int): Number of video segments to sample
        
    Returns:
        dict: Dictionary mapping video paths to model outputs
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} videos from SpookyBench metadata")
    
    # Filter by categories if specified
    if categories:
        df = df[df["Category"].isin(categories)]
        logger.info(f"Filtered to {len(df)} videos in categories: {categories}")
    
    # Process all videos and collect outputs
    results = {}
    
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        
        # Sample if needed
        if sample_size and sample_size < len(category_df):
            category_df = category_df.sample(sample_size, random_state=42)
        
        logger.info(f"Processing {len(category_df)} videos in category '{category}'...")
        
        for _, row in tqdm(category_df.iterrows(), total=len(category_df)):
            video_path = row["Path"]
            full_video_path = os.path.join(dataset_dir, video_path)
            
            if not os.path.exists(full_video_path):
                logger.warning(f"Video file not found: {full_video_path}")
                continue
                
            # Process video with model
            response = model.process_video(
                full_video_path,
                category=category,
                use_cot=use_cot,
                num_segments=num_segments,
                max_num=max_num,
                input_size=input_size
            )
            
            # Store result
            results[video_path] = response
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run InternVL on SpookyBench videos")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        choices=list(InternVLModel.MODEL_VARIANTS.keys()),
                        help="InternVL model variant to use")
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
    parser.add_argument("--segments", type=int, default=32,
                        help="Number of video segments to sample")
    parser.add_argument("--max_num", type=int, default=12,
                        help="Maximum number of tiles per frame")
    parser.add_argument("--input_size", type=int, default=448,
                        help="Size of input frames")
    
    # Custom prompt
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to use for all videos (overrides category-specific prompts)")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        logger.error(f"Metadata CSV file not found: {args.csv}")
        return
    
    # Initialize and load model
    model = InternVLModel(model_id=args.model, precision=args.precision).load()
    logger.info(f"Using device: {model.device}")
    
    results = {}
    
    # Process videos based on input type
    if os.path.isfile(args.dataset):
        # Single video processing
        video_path = args.dataset
        logger.info(f"Processing single video: {video_path}")
        
        # Use custom prompt if provided, otherwise use a generic prompt
        prompt = args.prompt if args.prompt else "What is shown in this video? Please describe in detail."
        
        response = model.process_video(
            video_path,
            prompt=prompt,
            num_segments=args.segments,
            max_num=args.max_num,
            input_size=args.input_size
        )
        results[os.path.basename(video_path)] = response
    else:
        # Dataset processing
        if not os.path.isdir(args.dataset):
            logger.error(f"Dataset directory not found: {args.dataset}")
            return
            
        logger.info(f"Processing videos from dataset: {args.dataset}")
        
        # Process with custom prompt if provided
        if args.prompt:
            logger.info(f"Using custom prompt for all videos: {args.prompt}")
            
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
            for _, row in tqdm(df.iterrows(), total=len(df)):
                video_path = row["Path"]
                full_video_path = os.path.join(args.dataset, video_path)
                
                if not os.path.exists(full_video_path):
                    logger.warning(f"Video file not found: {full_video_path}")
                    continue
                    
                response = model.process_video(
                    full_video_path,
                    prompt=args.prompt,
                    num_segments=args.segments,
                    max_num=args.max_num,
                    input_size=args.input_size
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
                num_segments=args.segments,
                max_num=args.max_num,
                input_size=args.input_size
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
    
    logger.info(f"Processing complete. {len(results)} videos processed.")
    logger.info(f"Results saved to {output_path}")
    
    # Print command to analyze results
    logger.info(f"To analyze results, run: python analyze_results.py --results {output_path} --csv {args.csv} --output {args.output}/analysis_{timestamp}")

if __name__ == "__main__":
    main()
