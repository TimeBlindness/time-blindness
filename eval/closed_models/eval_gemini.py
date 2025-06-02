#!/usr/bin/env python3

import os
import sys
import json
import argparse
import pandas as pd
import cv2
from tqdm import tqdm
import time
from pathlib import Path
from dotenv import load_dotenv

# Import Google's Generative AI client
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
# Configure the Gemini API with the key
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiProcessor:
    CATEGORY_PROMPTS = {
        "words": "This video contains text encoded through temporal patterns. What specific word or phrase do you see? Respond with just the text.",
        "shapes": "This video contains a objects encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "images": "This video contains a common object encoded through temporal patterns. What object do you see? Respond with just the object name.",
        "videos": "This video contains movement encoded through temporal patterns. What is shown? Respond with just 1-3 words describing what you see."
    }
    
    COT_PROMPTS = {
        "words": "This video encodes text through temporal patterns. To identify it: 1) Look for areas where opposing motion patterns reveal letters. 2) Focus on the overall word or phrase that emerges. 3) Read the specific text content. \n\nPlease respond with just the text you identify.",
        "shapes": "This video encodes a objects through temporal patterns. To identify it: 1) Look for areas where motion reveals contours and edges. 2) Focus on the geometric outline that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "images": "This video encodes an object through temporal patterns. To identify it: 1) Look for areas where motion reveals object outlines. 2) Focus on the object silhouette that emerges. 3) Determine what specific object is shown. \n\nPlease respond with just the object name.",
        "videos": "This video encodes movement through temporal patterns. To identify it: 1) Look for areas where motion patterns reveal action. 2) Focus on the movement that emerges. 3) Determine what is being shown. \n\nPlease respond with just 1-3 words describing what you see."
    }
    
    def __init__(self, model_name="gemini-1.5-pro"):
        """
        Initialize the Gemini processor
        
        Args:
            model_name (str): The Gemini model to use ('gemini-1.5-pro' or 'gemini-2.0-flash')
        """
        self.model_name = model_name
        # Use the correct model format for the API
        if "gemini" not in model_name:
            self.model_name = f"models/{model_name}"
        elif not model_name.startswith("models/"):
            self.model_name = f"models/{model_name}"
        print(f"Using Gemini model: {self.model_name}")

    def upload_video_file(self, video_path, display_name=None):
        """
        Upload a video file to Gemini File API
        
        Args:
            video_path (str): Path to the video file
            display_name (str): Display name for the uploaded file
            
        Returns:
            genai.File: Uploaded file object
        """
        if display_name is None:
            display_name = os.path.basename(video_path)
            
        print(f"Uploading video: {video_path}")
        
        # Check if file with same display name already exists
        file_list = genai.list_files(page_size=100)
        existing_file = next((f for f in file_list if f.display_name == display_name), None)
        
        if existing_file:
            print(f"File already exists: {existing_file.uri}")
            # Check if file is still processing
            while existing_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                # time.sleep(10)
                existing_file = genai.get_file(existing_file.name)
            
            if existing_file.state.name == "FAILED":
                print(f"\nExisting file failed, re-uploading...")
                # Delete failed file and re-upload
                genai.delete_file(existing_file.name)
            else:
                print(f"\nUsing existing file: {existing_file.uri}")
                return existing_file
        
        # Upload new file
        video_file = genai.upload_file(
            path=video_path, 
            display_name=display_name,
            resumable=True
        )
        print(f"Upload completed: {video_file.uri}")
        
        # Wait for file to be processed
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            # time.sleep(10)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"File processing failed: {video_file.state.name}")
        
        print(f"\nFile ready for use: {video_file.uri}")
        return video_file

    def process_video(self, video_path, prompt, fps=None):
        """
        Process a video with Gemini API
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt to send to the model
            fps (int, optional): Frames per second to sample, if provided (not used with File API)
            
        Returns:
            str: Model response
        """
        print(f"Processing video: {video_path}")
        
        # Create a unique display name for this video
        display_name = f"{os.path.basename(video_path)}_{int(time.time())}"
        
        # Upload video file using File API
        video_file = self.upload_video_file(video_path, display_name)
        
        # Create a GenerativeModel instance
        model = genai.GenerativeModel(model_name=self.model_name)
        
        # Generate content using the uploaded file
        print("Making inference request...")
        response = model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 600}
        )
        
        # Extract the text response
        answer = response.text
        print(f"Response: {answer}")
        
        # Clean up - delete the uploaded file to save quota
        genai.delete_file(video_file.name)
        print("Uploaded file cleaned up")
        
        return answer

def process_dataset(model, csv_path, dataset_dir, categories=None, use_cot=False, sample_size=None, fps=None):
    """
    Process videos in the dataset and collect model outputs
    
    Args:
        model (GeminiProcessor): Initialized model
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
                
            # Process video with model - NO ERROR HANDLING
            response = model.process_video(
                full_video_path,
                prompt,
                fps=fps
            )
            
            # Store result
            results[video_path] = response
            
            # Write partial results after each video to ensure we don't lose data
            print("Saving partial results...")
            with open("partial_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
            # Wait between videos to avoid rate limits and quota issues
            # print("Waiting 15 seconds before next video...")
            # time.sleep(15)
        
        # Wait longer between categories
        print(f"Finished category: {category}")
        # time.sleep(30)
    
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
    filename = f"{model_name.replace('/', '_')}_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process SpookyBench videos with Gemini")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to SpookyBench dataset directory")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to SpookyBench metadata CSV")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--model", type=str, default="gemini-1.5-pro",
                        choices=["gemini-1.5-pro", "gemini-2.0-flash"],
                        help="Gemini model to use")
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=["words", "shapes", "images", "videos"],
                        help="Categories to process (if not specified, all categories are used)")
    parser.add_argument("--use_cot", action="store_true",
                        help="Use chain-of-thought prompting instead of direct prompting")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample per category (if not specified, all videos are used)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second to sample from videos (note: File API handles this automatically)")
    
    args = parser.parse_args()
    
    print(f"Processing videos from dataset: {args.dataset}")
    if args.categories:
        print(f"Will process these categories: {args.categories}")
    
    # Initialize model
    model = GeminiProcessor(model_name=args.model)
    
    # Process dataset
    results = process_dataset(
        model,
        args.csv,
        args.dataset,
        categories=args.categories,
        use_cot=args.use_cot,
        sample_size=args.sample_size,
        fps=args.fps
    )
    
    # Save results
    save_results(results, args.output, args.model)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()