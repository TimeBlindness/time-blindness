# Dynamic Noise SNR Analyzer

This code can be used for generating images shown in the paper for analyzing videos with fixed-mask dynamic noise patterns, where hidden shapes become visible through differential motion between foreground and background noise.

## Overview

This analyzer is designed for videos with unique properties:
- **Fixed mask shapes** (letters, objects, etc.) hidden within noise
- **Differential motion**: Noise inside the mask moves in one direction, background noise moves in the opposite direction
- **Static invisibility**: When paused, all noise appears uniform and indistinguishable
- **Dynamic visibility**: When in motion, the differential movement reveals the hidden shapes

The tool calculates various Signal-to-Noise Ratio (SNR) metrics to quantify how well shapes can be perceived in dynamic sequences compared to static frames.

## Features

- **Multiple SNR Metrics**: Basic, perceptual, temporal coherence, and motion contrast SNR calculations
- **Motion Analysis**: Optical flow computation and motion boundary detection
- **Automatic Shape Detection**: Estimates mask shapes from motion patterns
- **Batch Processing**: Process entire directories of videos in parallel
- **Rich Visualizations**: Motion boundaries, coherence maps, estimated masks, and comparison charts
- **Comprehensive Reports**: Detailed metrics and aggregate statistics

## Installation

### Requirements

```bash
pip install numpy opencv-python matplotlib scipy pandas seaborn tqdm
```

### Dependencies

- **numpy**: Numerical computations
- **opencv-python**: Video processing and optical flow
- **matplotlib**: Visualization and plotting
- **scipy**: Signal processing
- **pandas**: Data manipulation
- **seaborn**: Statistical visualizations
- **tqdm**: Progress bars

## Usage

### Single Video Analysis

```bash
python dynamic_noise_analyzer.py --video /path/to/video.mp4 --output results_single
```

### Directory Processing

```bash
python dynamic_noise_analyzer.py --directory /path/to/videos --output results_batch --processes 6
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--video` | `-v` | Path to single video file | - |
| `--directory` | `-d` | Path to directory with videos | - |
| `--output` | `-o` | Output directory for results | - |
| `--sample-rate` | `-s` | Process every nth frame | 2 |
| `--window` | `-w` | Temporal integration window | 5 |
| `--quiet` | `-q` | Suppress progress output | False |
| `--sequential` | - | Disable parallel processing | False |
| `--processes` | `-p` | Number of parallel processes | 75% of CPU cores |

### Examples

**Basic analysis with custom sampling:**
```bash
python dynamic_noise_analyzer.py -v input.mp4 -o results -s 3 -w 7
```

**Batch processing with specific process count:**
```bash
python dynamic_noise_analyzer.py -d /videos -o /results -p 4
```

**Sequential processing (for debugging):**
```bash
python dynamic_noise_analyzer.py -d /videos -o /results --sequential
```

## Output Structure

For each processed video, the analyzer generates:

```
output_directory/
├── video_name/
│   ├── motion_boundaries.png      # Motion boundary strength map
│   ├── boundaries_overlay.png     # Boundaries overlaid on original frame
│   ├── direction_coherence.png    # Motion direction coherence map
│   ├── estimated_mask.png         # Detected shape mask
│   ├── mask_overlay.png          # Mask overlaid on original frame
│   ├── snr_metrics.png           # Bar chart of all SNR metrics
│   └── snr_results.txt           # Numerical results
├── aggregate_metrics.csv          # All videos' metrics in CSV format
├── comparison_[metric].png        # Per-metric comparison charts
├── radar_comparison.png           # Radar chart comparing top videos
└── metric_correlation.png         # Correlation heatmap between metrics
```

## SNR Metrics Explained

### 1. Basic SNR
Compares motion boundary energy to static noise variance.
- **Good for**: Overall motion detection strength
- **Range**: Typically 0-20 dB

### 2. Perceptual SNR
Applies human visual sensitivity weighting based on spatial frequencies.
- **Good for**: Human-relevant shape visibility
- **Range**: Typically -5 to 15 dB

### 3. Temporal Coherence SNR
Measures consistency of motion patterns over time.
- **Good for**: Stability of shape visibility
- **Range**: Typically 0-25 dB

### 4. Motion Contrast SNR
Quantifies contrast between differently moving regions.
- **Good for**: Edge definition and shape clarity
- **Range**: Typically 0-30 dB

### 5. Combined SNR
Weighted average of all metrics for overall assessment.
- **Weights**: Basic (0.2), Perceptual (0.3), Coherence (0.2), Contrast (0.3)

## Technical Details

### Motion Analysis Pipeline

1. **Optical Flow Computation**: Dense optical flow using Farneback method
2. **Motion Boundary Detection**: Spatial gradients of flow fields
3. **Temporal Integration**: Averaging over sliding window for noise reduction
4. **Directional Coherence**: Consistency of motion directions over time
5. **Shape Estimation**: Thresholding and morphological operations on boundaries

### Algorithm Parameters

- **Optical Flow**: Farneback with pyramid scale 0.5, 3 levels, window 15
- **Boundary Detection**: Sobel gradients with 3x3 kernel
- **Morphological Cleanup**: 5x5 kernel for opening and closing operations
- **Mask Threshold**: 90th percentile of boundary strength

