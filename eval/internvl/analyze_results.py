#!/usr/bin/env python3

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def analyze_results(results_file, metadata_csv, output_dir):
    """Analyze model prediction results against ground truth.
    
    Args:
        results_file (str): Path to JSON file containing model predictions
        metadata_csv (str): Path to SpookyBench metadata CSV
        output_dir (str): Directory to save analysis results
    """
    # Load results
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    # Process results by category
    categories = ['words', 'shapes', 'images', 'videos']
    category_results = {cat: [] for cat in categories}
    
    # Track total counts
    total_correct = 0
    total_count = 0
    
    # Process each video
    for _, row in df.iterrows():
        video_path = row["Path"]
        category = row["Category"]
        labels = eval(row["Label"]) if isinstance(row["Label"], str) else row["Label"]
        
        # Skip if video path not in results
        if video_path not in results:
            logger.warning(f"Video {video_path} not found in results")
            continue
        
        prediction = results[video_path]
        correct = is_match(prediction, labels)
        
        # Store result
        result = {
            "video_path": video_path,
            "ground_truth": labels,
            "prediction": prediction,
            "correct": correct
        }
        
        category_results[category].append(result)
        
        # Update counts
        if correct:
            total_correct += 1
        total_count += 1
    
    # Calculate metrics by category
    output = {
        "overall_accuracy": total_correct / total_count if total_count else 0,
        "category_results": {}
    }
    
    for category in categories:
        results = category_results[category]
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0
        
        logger.info(f"Category '{category}' accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
        
        output["category_results"][category] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results),
            "results": results
        }
    
    # Save analysis results
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Analysis saved to {analysis_path}")
    
    # Generate visualization
    visualize_results(output, os.path.join(output_dir, "accuracy.png"))

def visualize_results(results, output_path):
    """Create visualization of accuracy results.
    
    Args:
        results (dict): Analysis results
        output_path (str): Path to save visualization
    """
    # Extract category accuracies
    categories = []
    accuracies = []
    
    for category, result in results["category_results"].items():
        categories.append(category)
        accuracies.append(result["accuracy"])
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracies, color='skyblue')
    plt.axhline(y=results["overall_accuracy"], color='r', linestyle='-', 
               label=f'Overall: {results["overall_accuracy"]:.4f}')
    
    plt.title('InternVL Performance on SpookyBench')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend()
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze InternVL SpookyBench results")
    parser.add_argument("--results", type=str, required=True, help="Path to JSON results file")
    parser.add_argument("--csv", type=str, required=True, help="Path to SpookyBench metadata CSV")
    parser.add_argument("--output", type=str, default="./analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyze_results(args.results, args.csv, args.output)

if __name__ == "__main__":
    main()
