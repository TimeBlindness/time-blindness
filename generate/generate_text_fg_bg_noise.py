import numpy as np
import cv2
import random
import os
import csv
from pathlib import Path
from noise_generator import NoiseAnimator
from PIL import Image, ImageDraw, ImageFont

# Random text samples for generation (single words only)
RANDOM_TEXTS = [
    "hello", "machine", "learning", "neural", "computer", "vision", "artificial",
    "intelligence", "data", "science", "quantum", "cloud", "blockchain", "natural",
    "language", "image", "pattern", "algorithm", "system", "code", "parallel",
    "distributed", "analytics", "predictive", "gradient", "descent", "convolutional",
    "recurrent", "attention", "transformer", "generative", "reinforcement", "transfer",
    "ensemble", "dimensionality", "validation", "hyperparameter", "deployment", "edge",
    "internet", "cybersecurity", "encryption", "database", "query", "indexing",
    "microservices", "container", "orchestration", "continuous", "integration", "version",
    "agile", "methodology", "software", "testing", "documentation", "experience",
    "interface", "design", "responsive", "layout", "development", "mobile", "applications",
    "native", "platforms", "progressive", "rendering", "routing", "management",
    "async", "operations", "promise", "handling", "boundaries", "performance",
    "metrics", "balancing", "caching", "strategies", "content", "delivery", "network",
    "protocols", "socket", "programming", "thread", "process", "synchronization",
    "memory", "allocation", "garbage", "collection", "compiler", "optimization",
    "runtime", "efficiency", "debugging", "profiling", "analysis", "benchmark"
]


def generate_text_video_fg_bg_noise(output_path, text, width=960, height=540,
                                    fps=60, duration_sec=5, bg_noise=0.5,
                                    fg_noise=0.5, speckle_size=3, speed=2,
                                    direction="vertical"):
    """
    Generate video with text using same noise density but different patterns for FG/BG
    """
    animator = NoiseAnimator(width, height, fps)
    animator.bg_noise_density = bg_noise
    animator.fg_noise_density = fg_noise
    animator.use_same_noise = True  # SAME noise pattern, opposite motion makes text visible
    animator.speckle_size = speckle_size
    animator.animation_speed = speed
    animator.direction = direction

    animator.refresh_noise()

    total_frames = int(duration_sec * fps)

    # Static centered text
    position = (width // 2, height // 2)
    mask = animator.create_text_mask(text, position, font_size=80)

    # Use H.264 codec for web compatibility (VSCode, HuggingFace)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    bg_offset = 0
    fg_offset = 0

    for frame_idx in range(total_frames):
        # Roll noise patterns
        if direction == "vertical":
            bg_rolled = np.roll(animator.background_noise, int(bg_offset), axis=0)
            fg_rolled = np.roll(animator.foreground_noise, int(fg_offset), axis=0)
        else:
            bg_rolled = np.roll(animator.background_noise, int(bg_offset), axis=1)
            fg_rolled = np.roll(animator.foreground_noise, int(fg_offset), axis=1)

        # Apply mask: foreground noise on text, background noise elsewhere
        frame = np.where(mask, fg_rolled, bg_rolled)
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        bg_offset += animator.animation_speed
        fg_offset -= animator.animation_speed

    out.release()
    return str(output_path)


def generate_dataset(output_dir="text_fg_bg_noise_videos", num_videos=500,
                    bg_noise=0.5, fg_noise=0.5, speckle_size=1, speed=2,
                    direction="horizontal", duration=5.0):
    """Generate 500 text videos with fixed configuration"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {num_videos} text videos with BG={bg_noise}, FG={fg_noise}, "
          f"speckle={speckle_size}, speed={speed}, dir={direction}, duration={duration}s")

    # Create metadata CSV
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'text', 'duration', 'num_frames'])

        for i in range(num_videos):
            # Random text
            text = random.choice(RANDOM_TEXTS)
            filename = f"text_{i:04d}.mp4"
            output_path = output_dir / filename

            generate_text_video_fg_bg_noise(
                output_path=output_path,
                text=text,
                duration_sec=duration,
                bg_noise=bg_noise,
                fg_noise=fg_noise,
                speckle_size=speckle_size,
                speed=speed,
                direction=direction
            )

            num_frames = int(duration * 60)
            writer.writerow([filename, 'text_fg_bg', text, f"{duration:.2f}", num_frames])

            print(f"[{i+1}/{num_videos}] ✓ {filename} ('{text}')")


if __name__ == "__main__":
    generate_dataset(
        output_dir="text_fg_bg_noise_videos",
        num_videos=500,
        bg_noise=0.5,
        fg_noise=0.5,  # Same density as background
        speckle_size=1,
        speed=2,
        direction="horizontal",
        duration=5.0
    )
    print("\nComplete!")
