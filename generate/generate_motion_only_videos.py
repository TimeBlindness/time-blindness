import numpy as np
import cv2
import random
import os
import csv
from pathlib import Path
from noise_generator import NoiseAnimator

def generate_motion_only_video(output_path, width=960, height=540, fps=60,
                               duration_sec=5, bg_noise=0.5, speckle_size=3,
                               speed=2, direction="vertical"):
    """
    Generate video with only background noise - motion detected through noise patterns
    """
    animator = NoiseAnimator(width, height, fps)
    animator.bg_noise_density = bg_noise
    animator.speckle_size = speckle_size
    animator.animation_speed = speed
    animator.direction = direction

    animator.refresh_noise()

    total_frames = int(duration_sec * fps)

    # Use H.264 codec for web compatibility (VSCode, HuggingFace)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    bg_offset = 0

    for frame_idx in range(total_frames):
        # Roll noise pattern
        if direction == "vertical":
            bg_rolled = np.roll(animator.background_noise, int(bg_offset), axis=0)
        else:
            bg_rolled = np.roll(animator.background_noise, int(bg_offset), axis=1)

        frame_bgr = cv2.cvtColor(bg_rolled.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

        bg_offset += animator.animation_speed

    out.release()
    return str(output_path)


def generate_dataset(output_dir="motion_only_videos", num_videos=500,
                    bg_noise=0.5, speckle_size=1, speed=2, direction="horizontal",
                    duration=5.0):
    """Generate 500 motion-only videos with fixed configuration"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {num_videos} motion-only videos with noise={bg_noise}, "
          f"speckle={speckle_size}, speed={speed}, dir={direction}, duration={duration}s")

    # Create metadata CSV
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'duration', 'num_frames'])

        for i in range(num_videos):
            filename = f"motion_{i:04d}.mp4"
            output_path = output_dir / filename

            generate_motion_only_video(
                output_path=output_path,
                duration_sec=duration,
                bg_noise=bg_noise,
                speckle_size=speckle_size,
                speed=speed,
                direction=direction
            )

            num_frames = int(duration * 60)
            writer.writerow([filename, 'motion_only', f"{duration:.2f}", num_frames])

            print(f"[{i+1}/{num_videos}] ✓ {filename}")


if __name__ == "__main__":
    generate_dataset(
        output_dir="motion_only_videos",
        num_videos=500,
        bg_noise=0.5,
        speckle_size=1,
        speed=2,
        direction="horizontal",
        duration=5.0
    )
    print("\nComplete!")
