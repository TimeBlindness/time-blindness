# Video Generation Scripts

Scripts for generating motion-based video datasets with temporal noise patterns.

## Main Scripts

- **generate_motion_only_videos.py** - Generates 500 videos with background noise only
- **generate_text_fg_bg_noise.py** - Generates 500 videos with text revealed through opposing motion
- **noise_generator.py** - Core noise animation engine
- **upload_videos_to_hf.py** - Upload generated datasets to HuggingFace

## dynamic_batch/

Advanced video generation with dynamic content and SNR gradients:
- **generate_dynamic_videos.py** - Video generator with smooth motion paths (smooth, circular, linear)
- **main.py** - Configuration and batch generation (SNR gradients, moving/static content)
- **noise_generator.py** - Noise engine copy

Features: temporal SNR control, content movement, smooth trajectories

## misc/

- **generate_image_mask.py** - Image segmentation using briaai/RMBG-1.4 for mask generation

## Configuration

Fixed parameters (motion_only, text_fg_bg):
- Noise density: 0.5
- Speckle size: 1
- Speed: 2
- Direction: horizontal
- Duration: 5.0s
- Resolution: 960x540
- FPS: 60

## Usage

```bash
python generate_motion_only_videos.py
python generate_text_fg_bg_noise.py
python dynamic_batch/main.py
```
