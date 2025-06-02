#!/usr/bin/env python3

import os
import sys
import json
import argparse
import pandas as pd
import base64
import requests
import cv2
import io
from tqdm import tqdm
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Load environment variables
load_dotenv()

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class GPT4oProcessor:
    CATEGORY_PROMPTS = {
        "words": "This video contains text encoded through temporal patterns. What specific word or phrase do you see? Respond with just the text.",
        "shapes": "This video contains a object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "images": "This video contains a common object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "videos": "This video contains movement encoded through temporal patterns. What is shown? Respond with just 1-3 words describing what you see."
    }
    
    COT_PROMPTS = {
        "words": "This video encodes text through temporal patterns. To identify it: 1) Look for areas where opposing motion patterns reveal letters. 2) Focus on the overall word or phrase that emerges. 3) Read the specific text content. \n\nPlease respond with just the text you identify.",
        "shapes": "This video encodes a object through temporal patterns. To identify it: 1) Look for areas where motion reveals contours and edges. 2) Focus on the geometric outline that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "images": "This video encodes an object through temporal patterns. To identify it: 1) Look for areas where motion reveals object outlines. 2) Focus on the object silhouette that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "videos": "This video encodes movement through temporal patterns. To identify it: 1) Look for areas where motion patterns reveal action. 2) Focus on the movement that emerges. 3) Determine what is being shown. \n\nPlease respond with just 1-3 words describing what you see."
    }
    
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize the GPT-4o processor
        
        Args:
            model_name (str): The OpenAI model to use (default: 'gpt-4o')
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"Using OpenAI model: {self.model_name}")

    def extract_frames(self, video_path, stride=3):
        """
        Extract frames from a video file with a specific stride
        
        Args:
            video_path (str): Path to the video file
            stride (int): Take every Nth frame (lower means more frames)
            
        Returns:
            list: List of base64-encoded frames
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
        
        # Calculate a reasonable stride for SpookyBench videos
        # Use smaller stride for shorter videos to get more frames
        if frame_count < 100:
            actual_stride = 1  # Take every frame for very short videos
        elif frame_count < 200:
            actual_stride = 2  # Take every other frame
        else:
            actual_stride = stride
            
        # Calculate which frames to extract
        indices = range(0, frame_count, actual_stride)
        
        print(f"Extracting frames with stride {actual_stride} (approx. {len(indices)} frames)")
        
        # Extract the frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_img = Image.fromarray(frame)
                # Resize to reasonable dimensions to limit token usage
                pil_img = pil_img.resize((384, 384))
                # Convert to base64
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=80)
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                frames.append(img_str)
        
        # Release the video capture object
        cap.release()
        
        print(f"Successfully extracted {len(frames)} frames")
        return frames

    def process_video(self, video_path, prompt):
        """
        Process a video with GPT-4o API by extracting frames
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt to send to the model
            
        Returns:
            str: Model response
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames from the video
        frames = self.extract_frames(video_path)
        
        if not frames:
            raise ValueError(f"Could not extract any frames from video: {video_path}")
        
        # Create content list starting with the prompt
        content = [
            {"type": "input_text", "text": prompt}
        ]
        
        # Add each frame as an image
        for i, frame_b64 in enumerate(frames):
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame_b64}"
            })
        
        # Send the frames to GPT-4o
        response = self.client.responses.create(
            model=self.model_name,
            input=[{
                "role": "user",
                "content": content
            }]
        )
        
        # Extract the response text
        answer = response.output_text.strip()
        print(f"Response: {answer}")
        return answer

def process_dataset(model, csv_path, dataset_dir, categories=None, use_cot=False, sample_size=None):
    """
    Process videos in the dataset and collect model outputs
    
    Args:
        model (GPT4oProcessor): Initialized model
        csv_path (str): Path to SpookyBench metadata CSV
        dataset_dir (str): Directory containing SpookyBench videos
        categories (list): List of categories to process (None for all)
        use_cot (bool): Whether to use chain-of-thought prompts
        sample_size (int): Number of videos to sample per category
        
    Returns:
        dict: Dictionary mapping video paths to model outputs
    """
    # Load dataset metadata
    df = pd.read_csv(csv_path)
    
    # Print the column names to debug
    print(f"CSV columns: {df.columns.tolist()}")
    
    # If categories not specified, use all available
    if not categories:
        categories = ["words", "shapes", "images", "videos"]
    
    # Validate categories
    for category in categories:
        if category not in ["words", "shapes", "images", "videos"]:
            raise ValueError(f"Invalid category: {category}. Must be one of: words, shapes, images, videos")
    
    results = {}
    
    # Process each category
    for category in categories:
        print(f"\n{'='*30}\nProcessing category: {category}\n{'='*30}")
        
        # The CSV might not have a 'category' column but instead rely on directories
        # We'll process videos based on provided categories instead of filtering the CSV
        
        # Get the prompt based on category
        prompt_dict = model.COT_PROMPTS if use_cot else model.CATEGORY_PROMPTS
        prompt = prompt_dict[category]
        
        # Get all video files in this category directory
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}")
            continue
            
        # Get all video files in this directory
        video_files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]
        
        # Sample if requested
        if sample_size and sample_size < len(video_files):
            import random
            random.seed(42)
            video_files = random.sample(video_files, sample_size)
        
        # Process videos in this category
        print(f"Processing {len(video_files)} videos in category '{category}'...")
        for video_file in tqdm(video_files, total=len(video_files)):
            video_path = os.path.join(category, video_file)
            full_video_path = os.path.join(category_dir, video_file)
            
            # Check if video exists
            if not os.path.exists(full_video_path):
                print(f"Video file not found: {full_video_path}")
                continue
                
            # Process video with model
            response = model.process_video(
                full_video_path,
                prompt
            )
            
            # Store result
            results[video_path] = response
            
            # Wait between videos to avoid rate limits
            time.sleep(5)
        
        # Wait longer between categories
        print(f"Finished category: {category}")
        time.sleep(10)
    
    return results

def save_results(results, output_dir, model_name):
    """
    Save results to a JSON file
    
    Args:
        results (dict): Dictionary of results
        output_dir (str): Directory to save results
        model_name (str): Name of the model used
    
    Returns:
        str: Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_name}_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process SpookyBench videos with GPT-4o")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to SpookyBench dataset directory")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to SpookyBench metadata CSV")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini"],
                        help="OpenAI model to use")
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=["words", "shapes", "images", "videos"],
                        help="Categories to process (if not specified, all categories are used)")
    parser.add_argument("--use_cot", action="store_true",
                        help="Use chain-of-thought prompting instead of direct prompting")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample per category (if not specified, all videos are used)")
    
    args = parser.parse_args()
    
    print(f"Processing videos from dataset: {args.dataset}")
    if args.categories:
        print(f"Will process these categories: {args.categories}")
    
    # Initialize model
    model = GPT4oProcessor(model_name=args.model)
    
    # Process dataset
    results = process_dataset(
        model=model,
        csv_path=args.csv,
        dataset_dir=args.dataset,
        categories=args.categories,
        use_cot=args.use_cot,
        sample_size=args.sample_size
    )
    
    # Save results
    save_results(results, args.output, args.model)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
