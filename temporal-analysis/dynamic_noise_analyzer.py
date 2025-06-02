#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Fixed-Mask Dynamic Noise SNR Calculator

This script analyzes videos with the following properties:
•⁠ Fixed mask shapes (letters, objects, etc.)
•⁠ Noise inside the mask moves in one direction
•⁠ Background noise moves in the opposite direction
•⁠ When paused, all noise appears uniform and indistinguishable
•⁠ When in motion, the differential movement reveals the shapes

The script calculates various SNR metrics to quantify how well the shapes
can be perceived in the dynamic sequence compared to static frames.

python dynamic_noise_analyzer.py \
    --directory "/path/to/spooky_bench/images" \
    --output "results_images"\
    --processes 6

"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import argparse
import os
import glob
import csv
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

class DynamicNoiseSNRAnalyzer:
    """Class to analyze and calculate SNR for fixed-mask dynamic noise videos."""
    
    def __init__(self, video_path, output_dir=None, frame_sample_rate=1, 
                 temporal_window=5, verbose=True):
        """
        Initialize the analyzer with video parameters.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file to analyze
        output_dir : str, optional
            Directory to save output visualizations and results
        frame_sample_rate : int, optional
            Process every nth frame (to reduce computation)
        temporal_window : int, optional
            Number of frames to use for temporal integration
        verbose : bool, optional
            Whether to print progress information
        """
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate
        self.temporal_window = temporal_window
        self.verbose = verbose
        
        # Create output directory if specified
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            self.output_dir = None
        
        # Initialize data containers
        self.frames = []
        self.flows = []
        self.motion_boundaries = []
        self.direction_coherence = None
        self.estimated_mask = None
        
        # Results storage
        self.metrics = {}
        
    def load_video(self):
        """Load and preprocess video frames."""
        if self.verbose:
            print(f"Loading video: {self.video_path}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"Video properties: {self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames")
            print(f"Sampling every {self.frame_sample_rate} frames")
        
        # Read frames
        frames_to_process = range(0, self.frame_count, self.frame_sample_rate)
        if self.verbose:
            frames_to_process = tqdm(frames_to_process, desc="Loading frames")
            
        for i in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.append(gray)
        
        cap.release()
        
        if self.verbose:
            print(f"Loaded {len(self.frames)} frames")
            
        return self.frames
    
    def compute_optical_flow(self):
        """Compute optical flow between consecutive frames."""
        if not self.frames:
            self.load_video()
            
        if self.verbose:
            print("Computing optical flow...")
            iterator = tqdm(range(1, len(self.frames)), desc="Computing flow")
        else:
            iterator = range(1, len(self.frames))
            
        for i in iterator:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.frames[i-1], self.frames[i], 
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self.flows.append(flow)
            
        return self.flows
    
    def compute_motion_boundaries(self):
        """Calculate motion boundary strength based on flow gradients."""
        if not self.flows:
            self.compute_optical_flow()
            
        if self.verbose:
            print("Computing motion boundaries...")
            iterator = tqdm(self.flows, desc="Processing flows")
        else:
            iterator = self.flows
            
        for flow in iterator:
            # Extract flow components
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Calculate spatial gradients of flow
            fx_x = cv2.Sobel(flow_x, cv2.CV_64F, 1, 0, ksize=3)
            fx_y = cv2.Sobel(flow_x, cv2.CV_64F, 0, 1, ksize=3)
            fy_x = cv2.Sobel(flow_y, cv2.CV_64F, 1, 0, ksize=3)
            fy_y = cv2.Sobel(flow_y, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute motion boundary strength
            boundary = np.sqrt(fx_x**2 + fx_y**2 + fy_x**2 + fy_y**2)
            self.motion_boundaries.append(boundary)
            
        return self.motion_boundaries
    
    def compute_temporally_integrated_boundaries(self):
        """Integrate motion boundaries over temporal window to enhance signal."""
        if not self.motion_boundaries:
            self.compute_motion_boundaries()
            
        if self.verbose:
            print(f"Computing temporally integrated boundaries (window={self.temporal_window})...")
            
        integrated_boundaries = []
        
        for i in range(len(self.motion_boundaries) - self.temporal_window + 1):
            # Average over window_size consecutive frames
            window_avg = np.mean(self.motion_boundaries[i:i+self.temporal_window], axis=0)
            integrated_boundaries.append(window_avg)
        
        return integrated_boundaries
    
    def analyze_directional_coherence(self):
        """
        Analyze the coherence of motion directions.
        
        Returns a map where high values indicate consistent motion direction over time,
        which helps identify regions with coherent motion.
        """
        if not self.flows:
            self.compute_optical_flow()
            
        if self.verbose:
            print("Analyzing directional coherence...")
            
        # Calculate direction of motion at each pixel
        directions = []
        magnitudes = []
        
        for flow in self.flows:
            # Convert to polar coordinates (magnitude, angle)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            directions.append(angle)
            magnitudes.append(magnitude)
        
        # Calculate mean direction and magnitude
        mean_direction = np.mean(directions, axis=0)
        mean_magnitude = np.mean(magnitudes, axis=0)
        
        # Calculate circular variance of direction (adapted for angles)
        direction_var = np.zeros_like(mean_direction)
        for angle in directions:
            # Calculate angular difference
            diff = np.minimum(
                np.abs(angle - mean_direction),
                2 * np.pi - np.abs(angle - mean_direction)
            )
            direction_var += diff**2
            
        direction_var /= len(directions)
        
        # Low variance indicates consistent direction
        # Weight by magnitude to emphasize areas with significant motion
        coherence = np.exp(-direction_var) * (mean_magnitude > np.percentile(mean_magnitude, 50))
        
        self.direction_coherence = coherence
        return coherence
    
    def estimate_mask(self, threshold_percentile=90):
        """
        Estimate the mask based on motion boundaries.
        
        Parameters:
        -----------
        threshold_percentile : float, optional
            Percentile threshold for motion boundary strength
            
        Returns:
        --------
        mask : ndarray
            Binary mask representing the estimated shape
        """
        if not self.motion_boundaries:
            self.compute_motion_boundaries()
            
        # Use temporally integrated boundaries for better results
        integrated = self.compute_temporally_integrated_boundaries()
        
        # Average integrated boundaries over time
        avg_boundary = np.mean(integrated, axis=0)
        
        # Threshold to create binary mask
        threshold = np.percentile(avg_boundary, threshold_percentile)
        raw_mask = avg_boundary > threshold
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = raw_mask.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.estimated_mask = mask > 0
        return self.estimated_mask
    
    def compute_motion_spectrum(self):
        """
        Compute the spatiotemporal frequency spectrum of the video.
        
        This reveals the frequency signature of the dynamic patterns,
        which can help separate signal from noise.
        """
        if self.verbose:
            print("Computing motion spectrum...")
            
        # Take a subset of frames to keep computation manageable
        max_frames = min(32, len(self.frames))
        frame_subset = self.frames[:max_frames]
        
        # Stack frames into a 3D array (x, y, t)
        video_cube = np.stack(frame_subset, axis=2).astype(float)
        
        # Compute 3D FFT
        spectrum = np.abs(np.fft.fftn(video_cube))
        
        # Shift zero frequency to center
        spectrum = np.fft.fftshift(spectrum)
        
        return spectrum
    
    def calculate_basic_snr(self):
        """
        Calculate the basic SNR using motion boundary strength vs. static noise.
        
        Returns:
        --------
        snr : float
            Signal-to-Noise Ratio in dB
        """
        if not self.motion_boundaries:
            self.compute_motion_boundaries()
            
        # Calculate signal power (motion boundary energy)
        signal_power = np.mean([np.mean(boundary**2) for boundary in self.motion_boundaries])
        
        # Calculate noise power (variance of a static frame)
        static_frame = self.frames[0].astype(float)
        noise_power = np.var(static_frame)
        
        # Calculate SNR
        if noise_power > 0 and signal_power > 0:
            snr = 10 * math.log10(signal_power / noise_power)
        else:
            snr = 0
            
        self.metrics['basic_snr'] = snr
        return snr
    
    def calculate_perceptually_weighted_snr(self):
        """
        Calculate SNR with perceptual weighting based on human visual sensitivity.
        
        Returns:
        --------
        weighted_snr : float
            Perceptually weighted Signal-to-Noise Ratio in dB
        """
        if not self.motion_boundaries:
            self.compute_motion_boundaries()
            
        # Use integrated boundaries for better perceptual relevance
        integrated = self.compute_temporally_integrated_boundaries()
        avg_boundary = np.mean(integrated, axis=0)
        
        # Static noise frame
        static_noise = self.frames[0].astype(float)
        
        # Convert to frequency domain
        boundary_spectrum = np.fft.fft2(avg_boundary)
        noise_spectrum = np.fft.fft2(static_noise)
        
        # Generate spatial frequency coordinates
        h, w = static_noise.shape
        fx = np.fft.fftfreq(w)
        fy = np.fft.fftfreq(h)
        fx, fy = np.meshgrid(fx, fy)
        spatial_freq = np.sqrt(fx**2 + fy**2)
        
        # Perceptual weighting function based on Contrast Sensitivity Function
        # Peak sensitivity at around 4-8 cycles per degree (assuming standard viewing distance)
        # This is a simplified model of human visual sensitivity
        csf_peak = 0.1  # Approximate peak frequency in normalized units
        weight = spatial_freq * np.exp(-spatial_freq / csf_peak)
        
        # Apply weighting
        weighted_boundary = boundary_spectrum * weight
        weighted_noise = noise_spectrum * weight
        
        # Calculate weighted powers
        signal_power = np.mean(np.abs(weighted_boundary)**2)
        noise_power = np.mean(np.abs(weighted_noise)**2)
        
        # Calculate weighted SNR
        if noise_power > 0 and signal_power > 0:
            weighted_snr = 10 * math.log10(signal_power / noise_power)
        else:
            weighted_snr = 0
            
        self.metrics['perceptual_snr'] = weighted_snr
        return weighted_snr
    
    def calculate_temporal_coherence_snr(self):
        """
        Calculate SNR based on temporal coherence of motion.
        
        This metric focuses on how consistently motion patterns are maintained
        over time, which strongly relates to perceptibility.
        
        Returns:
        --------
        coherence_snr : float
            Temporal coherence based Signal-to-Noise Ratio in dB
        """
        if self.direction_coherence is None:
            self.analyze_directional_coherence()
            
        # Signal power is the variance of the coherence map
        # High variance indicates clear boundaries between motion regions
        signal_power = np.var(self.direction_coherence)
        
        # Noise power is the variance within regions of similar coherence
        # We estimate this by calculating local variance in small patches
        kernel_size = 5
        local_mean = cv2.blur(self.direction_coherence, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(self.direction_coherence**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2
        
        # Average local variance as noise power
        noise_power = np.mean(local_var)
        
        # Calculate SNR
        if noise_power > 0 and signal_power > 0:
            coherence_snr = 10 * math.log10(signal_power / noise_power)
        else:
            coherence_snr = 0
            
        self.metrics['temporal_coherence_snr'] = coherence_snr
        return coherence_snr
    
    def calculate_motion_contrast_snr(self):
        """
        Calculate SNR based on the contrast between differently moving regions.
        
        This metric focuses on the perceptual edge that forms between regions with
        different motion directions, which is the key to shape visibility.
        
        Returns:
        --------
        contrast_snr : float
            Motion contrast based Signal-to-Noise Ratio in dB
        """
        # If no mask is estimated yet, create one
        if self.estimated_mask is None:
            self.estimate_mask()
            
        # Calculate mean motion vectors in mask and background regions
        mask_motion = []
        bg_motion = []
        
        for flow in self.flows:
            mask_vx = np.mean(flow[..., 0][self.estimated_mask])
            mask_vy = np.mean(flow[..., 1][self.estimated_mask])
            
            bg_vx = np.mean(flow[..., 0][~self.estimated_mask])
            bg_vy = np.mean(flow[..., 1][~self.estimated_mask])
            
            mask_motion.append((mask_vx, mask_vy))
            bg_motion.append((bg_vx, bg_vy))
        
        # Calculate average motion vectors
        mask_motion_avg = np.mean(mask_motion, axis=0)
        bg_motion_avg = np.mean(bg_motion, axis=0)
        
        # Calculate motion contrast (difference between regions)
        motion_contrast = np.sqrt(
            (mask_motion_avg[0] - bg_motion_avg[0])**2 +
            (mask_motion_avg[1] - bg_motion_avg[1])**2
        )
        
        # Calculate motion variance within each region (noise)
        mask_var = np.mean([
            (mx - mask_motion_avg[0])**2 + (my - mask_motion_avg[1])**2
            for mx, my in mask_motion
        ])
        
        bg_var = np.mean([
            (bx - bg_motion_avg[0])**2 + (by - bg_motion_avg[1])**2
            for bx, by in bg_motion
        ])
        
        # Average variance as noise power
        noise_power = (mask_var + bg_var) / 2
        
        # Signal power is the squared motion contrast
        signal_power = motion_contrast**2
        
        # Calculate SNR
        if noise_power > 0 and signal_power > 0:
            contrast_snr = 10 * math.log10(signal_power / noise_power)
        else:
            contrast_snr = 0
            
        self.metrics['motion_contrast_snr'] = contrast_snr
        return contrast_snr
    
    def calculate_all_metrics(self):
        """Calculate all SNR metrics."""
        if self.verbose:
            print("Calculating all SNR metrics...")
            
        # Basic motion boundary SNR
        basic_snr = self.calculate_basic_snr()
        
        # Perceptually weighted SNR
        perceptual_snr = self.calculate_perceptually_weighted_snr()
        
        # Temporal coherence SNR
        coherence_snr = self.calculate_temporal_coherence_snr()
        
        # Motion contrast SNR
        contrast_snr = self.calculate_motion_contrast_snr()
        
        # Combined metric (weighted average of all metrics)
        # Weights determined by empirical relevance to perception
        weights = {
            'basic_snr': 0.2,
            'perceptual_snr': 0.3,
            'temporal_coherence_snr': 0.2,
            'motion_contrast_snr': 0.3
        }
        
        combined_snr = sum(self.metrics[k] * weights[k] for k in weights)
        self.metrics['combined_snr'] = combined_snr
        
        if self.verbose:
            print("SNR Metrics:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.2f} dB")
                
        return self.metrics
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        if not self.output_dir:
            print("No output directory specified. Skipping visualizations.")
            return
            
        if self.verbose:
            print(f"Saving visualizations to {self.output_dir}...")
            
        # 1. Motion boundaries visualization
        if self.motion_boundaries:
            avg_boundary = np.mean(self.motion_boundaries, axis=0)
            plt.figure(figsize=(10, 8))
            plt.imshow(avg_boundary, cmap='viridis')
            plt.colorbar(label='Motion Boundary Strength')
            plt.title('Average Motion Boundary Strength')
            plt.savefig(os.path.join(self.output_dir, 'motion_boundaries.png'), dpi=150)
            plt.close()
            
            # Overlay on original frame
            plt.figure(figsize=(10, 8))
            normalized = (avg_boundary - avg_boundary.min()) / (avg_boundary.max() - avg_boundary.min())
            plt.imshow(self.frames[0], cmap='gray', alpha=0.7)
            plt.imshow(normalized, cmap='viridis', alpha=0.5)
            plt.title('Motion Boundaries Overlay')
            plt.savefig(os.path.join(self.output_dir, 'boundaries_overlay.png'), dpi=150)
            plt.close()
        
        # 2. Directional coherence visualization
        if self.direction_coherence is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.direction_coherence, cmap='plasma')
            plt.colorbar(label='Directional Coherence')
            plt.title('Motion Direction Coherence')
            plt.savefig(os.path.join(self.output_dir, 'direction_coherence.png'), dpi=150)
            plt.close()
        
        # 3. Estimated mask visualization
        if self.estimated_mask is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.estimated_mask, cmap='gray')
            plt.title('Estimated Mask')
            plt.savefig(os.path.join(self.output_dir, 'estimated_mask.png'), dpi=150)
            plt.close()
            
            # Overlay on original frame
            plt.figure(figsize=(10, 8))
            plt.imshow(self.frames[0], cmap='gray')
            mask_overlay = np.zeros((*self.estimated_mask.shape, 4))
            mask_overlay[self.estimated_mask, 3] = 0.5  # Alpha channel
            mask_overlay[self.estimated_mask, 0] = 1.0  # Red channel
            plt.imshow(mask_overlay)
            plt.title('Estimated Mask Overlay')
            plt.savefig(os.path.join(self.output_dir, 'mask_overlay.png'), dpi=150)
            plt.close()
        
        # 4. SNR metrics bar chart
        plt.figure(figsize=(12, 6))
        metrics = {k: v for k, v in self.metrics.items() if k != 'combined_snr'}
        plt.bar(metrics.keys(), metrics.values())
        plt.axhline(y=self.metrics.get('combined_snr', 0), color='r', linestyle='-', label='Combined SNR')
        plt.ylabel('SNR (dB)')
        plt.title('SNR Metrics Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'snr_metrics.png'), dpi=150)
        plt.close()
        
        # 5. Save numerical results to text file
        with open(os.path.join(self.output_dir, 'snr_results.txt'), 'w') as f:
            f.write("SNR Analysis Results\n")
            f.write("===================\n\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Dimensions: {self.width}x{self.height}\n")
            f.write(f"FPS: {self.fps}\n")
            f.write(f"Frames analyzed: {len(self.frames)}\n\n")
            f.write("SNR Metrics:\n")
            for metric, value in self.metrics.items():
                f.write(f"  {metric}: {value:.2f} dB\n")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        # 1. Load video
        self.load_video()
        
        # 2. Compute optical flow
        self.compute_optical_flow()
        
        # 3. Compute motion boundaries
        self.compute_motion_boundaries()
        
        # 4. Analyze directional coherence
        self.analyze_directional_coherence()
        
        # 5. Estimate mask
        self.estimate_mask()
        
        # 6. Calculate all metrics
        self.calculate_all_metrics()
        
        # 7. Create visualizations
        self.visualize_results()
        
        return self.metrics


def process_video(video_path, output_dir=None, sample_rate=1, window=5, verbose=True):
    """Process a single video and return metrics."""
    # Create video-specific output directory if needed
    video_output_dir = None
    if output_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
    
    # Create and run analyzer
    analyzer = DynamicNoiseSNRAnalyzer(
        video_path=video_path,
        output_dir=video_output_dir,
        frame_sample_rate=sample_rate,
        temporal_window=window,
        verbose=verbose
    )
    
    try:
        metrics = analyzer.run_full_analysis()
        
        # Print individual video results if verbose
        if verbose:
            print(f"\nResults for {os.path.basename(video_path)}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f} dB")
        
        return (os.path.basename(video_path), metrics, True)
    except Exception as e:
        if verbose:
            print(f"Error processing {video_path}: {str(e)}")
        return (os.path.basename(video_path), None, False)


def parallel_process_video(video_path, video_idx, output_dir, sample_rate, window, verbose, video_count):
    """Wrapper for process_video to support multiprocessing with progress updates."""
    try:
        if verbose and video_idx % 10 == 0:
            print(f"Processing video {video_idx+1}/{video_count}: {os.path.basename(video_path)}")
        
        # Process video with reduced verbosity
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_rate=sample_rate,
            window=window,
            verbose=False  # Turn off verbose within each process
        )
        
        # Print minimal success/error message
        if verbose:
            if result[2]:  # Success
                print(f"✓ Completed {video_idx+1}/{video_count}: {os.path.basename(video_path)}")
            else:  # Error
                print(f"✗ Failed {video_idx+1}/{video_count}: {os.path.basename(video_path)}")
                
        return result
    except Exception as e:
        print(f"Critical error processing {video_path}: {str(e)}")
        return (os.path.basename(video_path), None, False)


def process_directory(dir_path, output_dir, sample_rate=1, window=5, verbose=True, parallel=True, n_processes=None):
    """Process all videos in a directory and generate aggregate metrics."""
    # List of video file extensions to process
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    # Find all video files in the directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(dir_path, f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {dir_path}")
        return
    
    # Sort video files for consistent processing order
    video_files.sort()
    
    if verbose:
        print(f"Found {len(video_files)} video files to process")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Configure multiprocessing
    if n_processes is None:
        # Use 75% of available cores by default, but at least 1
        n_processes = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    if verbose:
        if parallel:
            print(f"Processing videos in parallel using {n_processes} processes")
        else:
            print("Processing videos sequentially")
    
    start_time = time.time()
    
    # Process videos (parallel or sequential)
    if parallel and len(video_files) > 1:
        # Create a partial function with fixed parameters
        process_func = partial(
            parallel_process_video,
            output_dir=output_dir,
            sample_rate=sample_rate,
            window=window,
            verbose=verbose,
            video_count=len(video_files)
        )
        
        # Create argument list with video paths and indices in the correct order
        # This must match the order of parameters in parallel_process_video function
        args_list = [(video_path, idx) for idx, video_path in enumerate(video_files)]
        
        # Process in parallel
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.starmap(process_func, args_list)
    else:
        # Process sequentially
        results = []
        for idx, video_path in enumerate(video_files):
            if verbose:
                print(f"\nProcessing video {idx+1}/{len(video_files)}: {os.path.basename(video_path)}")
            
            result = process_video(
                video_path=video_path,
                output_dir=output_dir,
                sample_rate=sample_rate,
                window=window,
                verbose=verbose
            )
            results.append(result)
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    avg_time_per_video = elapsed_time / len(video_files) if video_files else 0
    
    if verbose:
        print(f"\nTotal processing time: {elapsed_time:.1f} seconds")
        print(f"Average time per video: {avg_time_per_video:.1f} seconds")
    
    # Extract successful results
    all_metrics = {}
    successful_videos = []
    failed_videos = []
    
    for video_name, metrics, success in results:
        if success:
            successful_videos.append(video_name)
            all_metrics[video_name] = metrics
        else:
            failed_videos.append(video_name)
    
    # Report success rate
    success_rate = len(successful_videos) / len(video_files) * 100 if video_files else 0
    if verbose:
        print(f"\nSuccessfully processed {len(successful_videos)}/{len(video_files)} videos ({success_rate:.1f}%)")
        if failed_videos:
            print(f"Failed to process {len(failed_videos)} videos:")
            for video in failed_videos[:10]:  # Show first 10 failed videos
                print(f"  - {video}")
            if len(failed_videos) > 10:
                print(f"  ... and {len(failed_videos) - 10} more")
    
    # Generate aggregate metrics
    if successful_videos:
        generate_aggregate_metrics(all_metrics, output_dir, verbose)
    else:
        print("No videos were successfully processed.")


def generate_aggregate_metrics(all_metrics, output_dir, verbose=True):
    """Generate and save aggregate metrics across all videos."""
    if not all_metrics:
        return
    
    if verbose:
        print("\nGenerating aggregate metrics...")
    
    # Calculate aggregate statistics
    metric_names = next(iter(all_metrics.values())).keys()
    
    # Initialize storage for aggregate data
    aggregate = {
        'mean': {},
        'median': {},
        'min': {},
        'max': {},
        'std': {}
    }
    
    # Calculate statistics for each metric
    for metric in metric_names:
        # Extract this metric from all videos
        values = [metrics[metric] for metrics in all_metrics.values() if metric in metrics]
        
        if values:
            aggregate['mean'][metric] = np.mean(values)
            aggregate['median'][metric] = np.median(values)
            aggregate['min'][metric] = np.min(values)
            aggregate['max'][metric] = np.max(values)
            aggregate['std'][metric] = np.std(values)
    
    # Print aggregate results
    if verbose:
        print("\nAggregate Metrics:")
        for stat_name, metrics in aggregate.items():
            print(f"\n{stat_name.capitalize()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f} dB")
    
    # Save aggregate results to CSV
    if output_dir:
        # Prepare data for CSV
        rows = []
        # Header row
        header = ['Video'] + list(metric_names)
        rows.append(header)
        
        # Add each video's metrics
        for video_name, metrics in all_metrics.items():
            row = [video_name] + [f"{metrics.get(metric, 'N/A'):.2f}" if isinstance(metrics.get(metric, 'N/A'), (int, float)) else 'N/A' for metric in metric_names]
            rows.append(row)
        
        # Add aggregate statistics
        for stat_name, stat_metrics in aggregate.items():
            row = [f"{stat_name.capitalize()}"] + [f"{stat_metrics.get(metric, 'N/A'):.2f}" if isinstance(stat_metrics.get(metric, 'N/A'), (int, float)) else 'N/A' for metric in metric_names]
            rows.append(row)
        
        # Write to CSV
        csv_path = os.path.join(output_dir, 'aggregate_metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        # Generate comparison visualizations
        generate_comparison_visualizations(all_metrics, aggregate, output_dir)
        
        if verbose:
            print(f"\nAggregate results saved to: {csv_path}")


def generate_comparison_visualizations(all_metrics, aggregate, output_dir):
    """Generate visualizations comparing metrics across videos."""
    if not output_dir:
        return
    
    # Extract metric names
    metric_names = next(iter(all_metrics.values())).keys()
    
    # 1. Bar chart comparing each metric across videos
    for metric in metric_names:
        plt.figure(figsize=(12, 6))
        
        # Extract values and video names
        videos = []
        values = []
        for video, metrics in all_metrics.items():
            if metric in metrics:
                videos.append(video)
                values.append(metrics[metric])
        
        # If there are too many videos, sample or group them
        if len(videos) > 30:
            # Sample videos for visualization or use histograms instead
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=20)
            plt.xlabel(f'{metric} (dB)')
            plt.ylabel('Number of Videos')
            plt.title(f'Distribution of {metric} Across {len(videos)} Videos')
            if metric in aggregate['mean']:
                plt.axvline(x=aggregate['mean'][metric], color='r', linestyle='-',
                            label=f'Mean: {aggregate["mean"][metric]:.2f} dB')
                plt.axvline(x=aggregate['median'][metric], color='g', linestyle='--',
                            label=f'Median: {aggregate["median"][metric]:.2f} dB')
            plt.legend()
        else:
            # Sort by metric value for better visualization
            sorted_data = sorted(zip(videos, values), key=lambda x: x[1])
            videos = [x[0] for x in sorted_data]
            values = [x[1] for x in sorted_data]
            
            # Create bar chart
            bars = plt.bar(videos, values)
            
            # Add mean line
            if metric in aggregate['mean']:
                plt.axhline(y=aggregate['mean'][metric], color='r', linestyle='-', 
                            label=f'Mean: {aggregate["mean"][metric]:.2f} dB')
            
            plt.ylabel(f'{metric} (dB)')
            plt.title(f'Comparison of {metric} Across Videos')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=150)
        plt.close()
    
    # 2. Radar chart comparing all metrics for each video
    # Limit to a reasonable number of videos for readability
    max_videos = 5
    if len(all_metrics) > max_videos:
        # Select a subset of videos with diverse metrics
        sample_videos = {}
        
        # Include min and max for each metric
        for metric in metric_names:
            metric_values = [(video, metrics[metric]) for video, metrics in all_metrics.items() 
                            if metric in metrics]
            
            if metric_values:
                # Get videos with min and max values for this metric
                min_video = min(metric_values, key=lambda x: x[1])[0]
                max_video = max(metric_values, key=lambda x: x[1])[0]
                
                sample_videos[min_video] = all_metrics[min_video]
                sample_videos[max_video] = all_metrics[max_video]
                
                # If we still need more, add median
                if len(sample_videos) < max_videos:
                    sorted_values = sorted(metric_values, key=lambda x: x[1])
                    median_video = sorted_values[len(sorted_values) // 2][0]
                    sample_videos[median_video] = all_metrics[median_video]
        
        # If we still don't have enough, add random ones
        remaining_videos = [v for v in all_metrics.keys() if v not in sample_videos]
        while len(sample_videos) < max_videos and remaining_videos:
            random_video = remaining_videos.pop(0)
            sample_videos[random_video] = all_metrics[random_video]
        
        videos_to_plot = sample_videos
    else:
        videos_to_plot = all_metrics
    
    # Create radar chart
    plt.figure(figsize=(10, 10))
    
    # Number of metrics
    num_metrics = len(metric_names)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create radar plot
    ax = plt.subplot(111, polar=True)
    
    # Add metric labels
    metric_list = list(metric_names)
    plt.xticks(angles[:-1], metric_list, size=8)
    
    # Find max value for scaling
    max_value = max([max([metrics.get(metric, 0) for metric in metric_names]) 
                     for metrics in videos_to_plot.values()])
    plt.ylim(0, max_value * 1.1)
    
    # Plot each video
    for video, metrics in videos_to_plot.items():
        values = [metrics.get(metric, 0) for metric in metric_list]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=1, label=video)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Radar Chart of Metrics Across Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Correlation matrix between metrics
    # Create a DataFrame with all metrics
    data = []
    for video, metrics in all_metrics.items():
        row = {'video': video}
        for metric, value in metrics.items():
            row[metric] = value
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate correlation between metrics
    if len(df) > 1:  # Need at least 2 videos for correlation
        corr = df.drop('video', axis=1).corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlation.png'), dpi=150)
        plt.close()


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Analyze SNR in fixed-mask dynamic noise videos")
    
    # Create a mutually exclusive group for video path or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", "-v", help="Path to a single video file")
    input_group.add_argument("--directory", "-d", help="Path to directory containing video files")
    parser.add_argument("--output", "-o", help="Output directory for results and visualizations")
    parser.add_argument("--sample-rate", "-s", type=int, default=2, help="Process every nth frame")
    parser.add_argument("--window", "-w", type=int, default=5, help="Temporal integration window size")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--sequential", action="store_true", help="Process videos sequentially (no parallelization)")
    parser.add_argument("--processes", "-p", type=int, help="Number of parallel processes to use")
    parser.add_argument("--force-sequential", action="store_true", help="Force sequential processing even if parallel is supported")
    
    args = parser.parse_args()
    
    # Process based on input type
    if args.video:
        # Process a single video
        result = process_video(
            video_path=args.video,
            output_dir=args.output,
            sample_rate=args.sample_rate,
            window=args.window,
            verbose=not args.quiet
        )
        
        if result[2] and args.output:  # If success
            print(f"\nResults and visualizations saved to: {args.output}")
    
    elif args.directory:
        try:
            # Try with parallel processing first (if not disabled)
            if not args.sequential and not args.force_sequential:
                process_directory(
                    dir_path=args.directory,
                    output_dir=args.output,
                    sample_rate=args.sample_rate,
                    window=args.window,
                    verbose=not args.quiet,
                    parallel=True,
                    n_processes=args.processes
                )
            else:
                # Use sequential processing if requested
                raise ValueError("Sequential processing requested")
                
        except Exception as e:
            if not args.force_sequential:
                print(f"\nParallel processing failed with error: {str(e)}")
                print("Falling back to sequential processing...")
            
            # Fallback to sequential processing
            process_directory(
                dir_path=args.directory,
                output_dir=args.output,
                sample_rate=args.sample_rate,
                window=args.window,
                verbose=not args.quiet,
                parallel=False,
                n_processes=None
            )
        
        if args.output:
            print(f"\nAll results and visualizations saved to: {args.output}")


if __name__ == "__main__":
    try:
        # Fix for multiprocessing on macOS
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # If already set, ignore the error
        pass
    
    main()