from generate_dynamic_videos import DynamicVideoGenerator
import random
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Output settings
    "output_dir": "./noise_video/dynamic_videos",
    "resolution": (960, 540),
    "fps": 60,
    
    # Noise mode control
    "use_same_noise": True,  # True = temporal SNR only (recommended)
    
    # Content mode
    "static_content": False,  # True = static content, False = moving content
    
    # SNR Gradient settings
    "snr_steps": 4,
    "bg_noise_start": 0.70,
    "bg_noise_end": 0.45,
    "fg_noise_start": 0.50,
    "fg_noise_end": 0.55,
    
    # Video duration
    "duration_min": 5.0,
    "duration_max": 10.0,
    
    # Animation parameters
    "speed_options": [1, 2],
    "direction_options": ["vertical", "horizontal"],
    "movement_options": ["smooth", "circular", "linear"],
    
    # Noise parameters
    "speckle_size": 1,
    
    # Wordlist
    "wordlist_file": "wordlist.txt",
}

# ============================================
# WORD-BASED VIDEOS
# ============================================

def generate_single_video():
    """Generate a single test video"""
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"],
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    print("Generating single test video...")
    generator.generate_text_video(
        text="HELLO",
        duration_sec=7.0,
        bg_noise=0.55,
        fg_noise=0.55,
        use_same_noise=CONFIG["use_same_noise"],
        speckle_size=CONFIG["speckle_size"],
        speed=2,
        direction="vertical",
        movement_type="smooth",
        static_content=CONFIG["static_content"]
    )
    print("Done!")

def batch_generate_from_wordlist():
    """Generate full dataset from wordlist with SNR gradient"""
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"] + "_wordlist",
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    # Read wordlist
    try:
        with open(CONFIG["wordlist_file"], "r") as f:
            words = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {CONFIG['wordlist_file']} not found!")
        print("Creating sample wordlist...")
        words = ["ALGORITHM", "NEURAL", "VISION", "COMPUTE", "TENSOR"]
        with open(CONFIG["wordlist_file"], "w") as f:
            f.write("\n".join(words))
    
    noise_mode = "TEMPORAL ONLY" if CONFIG["use_same_noise"] else "SPATIAL+TEMPORAL"
    content_mode = "STATIC" if CONFIG["static_content"] else "MOVING"
    
    print(f"Generating videos for {len(words)} words...")
    print(f"Noise Mode: {noise_mode}")
    print(f"Content Mode: {content_mode}")
    print(f"SNR steps: {CONFIG['snr_steps']}")
    print(f"Total videos: {len(words) * CONFIG['snr_steps']}\n")
    
    for i, word in enumerate(words):
        print(f"[{i+1}/{len(words)}] Processing: {word}")
        generator.generate_snr_gradient_videos(
            text=word,
            snr_steps=CONFIG["snr_steps"],
            use_same_noise=CONFIG["use_same_noise"],
            bg_noise_start=CONFIG["bg_noise_start"],
            bg_noise_end=CONFIG["bg_noise_end"],
            fg_noise_start=CONFIG["fg_noise_start"],
            fg_noise_end=CONFIG["fg_noise_end"],
            duration_range=(CONFIG["duration_min"], CONFIG["duration_max"]),
            speed_options=CONFIG["speed_options"],
            direction_options=CONFIG["direction_options"],
            movement_options=CONFIG["movement_options"],
            static_content=CONFIG["static_content"]
        )
    
    print(f"\n✓ All {len(words) * CONFIG['snr_steps']} videos generated!")


# ============================================
# DEPTH MAP VIDEOS
# ============================================

def generate_depth_map_videos(depth_video_folder):
    """Generate videos using depth maps from video files"""
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"] + "_depth",
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    depth_dir = Path(depth_video_folder)
    depth_videos = list(depth_dir.glob("*.mp4")) + list(depth_dir.glob("*.avi"))
    
    if not depth_videos:
        print(f"No depth videos found in {depth_video_folder}")
        return
    
    print(f"Found {len(depth_videos)} depth videos")
    
    for i, depth_video in enumerate(depth_videos):
        print(f"\n[{i+1}/{len(depth_videos)}] Processing: {depth_video.name}")
        
        # Generate with different SNR levels
        for step in range(CONFIG["snr_steps"]):
            progress = step / (CONFIG["snr_steps"] - 1) if CONFIG["snr_steps"] > 1 else 0
            
            bg_noise = CONFIG["bg_noise_start"] - progress * (CONFIG["bg_noise_start"] - CONFIG["bg_noise_end"])
            duration = random.uniform(CONFIG["duration_min"], CONFIG["duration_max"])
            speed = random.choice(CONFIG["speed_options"])
            direction = random.choice(CONFIG["direction_options"])
            
            print(f"  [{step+1}/{CONFIG['snr_steps']}] Noise={bg_noise:.1%} {duration:.1f}s", end=" ")
            
            generator.generate_video_with_depth(
                depth_video_path=depth_video,
                duration_sec=duration,
                bg_noise=bg_noise,
                fg_noise=bg_noise,
                use_same_noise=True,
                speckle_size=CONFIG["speckle_size"],
                speed=speed,
                direction=direction,
                depth_scale=2.0
            )
    
    print(f"\n✓ All depth map videos generated!")

# ============================================
# IMAGE VIDEOS
# ============================================

def generate_image_dataset(image_folder):
    """Generate videos from images in a folder"""
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"] + "_images",
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    image_dir = Path(image_folder)
    images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.webp"))
    
    print(f"Found {len(images)} images")
    
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] Processing: {image_path.name}")
        
        # Random parameters for this image
        direction = random.choice(CONFIG["direction_options"])
        speed = random.choice(CONFIG["speed_options"])
        if not CONFIG["static_content"]:
            movement_type = random.choice(CONFIG["movement_options"])  
        else:
            movement_type = "static"
        
        for step in range(CONFIG["snr_steps"]):
            progress = step / (CONFIG["snr_steps"] - 1) if CONFIG["snr_steps"] > 1 else 0
            bg_noise = CONFIG["bg_noise_start"] - progress * (CONFIG["bg_noise_start"] - CONFIG["bg_noise_end"])
            fg_noise = CONFIG["fg_noise_start"] + progress * (CONFIG["fg_noise_start"] - CONFIG["fg_noise_end"])
            duration = random.uniform(CONFIG["duration_min"], CONFIG["duration_max"])
            
            if CONFIG["use_same_noise"]:
                print(f"  [{step+1}/{CONFIG['snr_steps']}] Noise={bg_noise:.1%}", end=" ")
            else:
                print(f"  [{step+1}/{CONFIG['snr_steps']}] BG={bg_noise:.1%} FG={fg_noise:.1%}", end=" ")
            
            generator.generate_image_video(
                image_path=str(image_path),
                duration_sec=duration,
                bg_noise=bg_noise,
                fg_noise=fg_noise,
                use_same_noise=CONFIG["use_same_noise"],
                speckle_size=CONFIG["speckle_size"],
                speed=speed,
                direction=direction,
                movement_type=movement_type,
                static_content=CONFIG["static_content"]
            )

# ============================================
# DUAL IMAGE VIDEOS
# ============================================
def generate_dual_image_dataset(image_folder):
    """Generate videos with pairs of images"""
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"] + "_dual_images",
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    image_dir = Path(image_folder)
    images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.webp"))
    
    print(f"Found {len(images)} images")
    
    # Generate random pairs (non-repeating)
    num_pairs = min(25, len(images) * (len(images) - 1) // 2)  # Limit pairs
    all_pairs = [(images[i], images[j]) for i in range(len(images)) 
                 for j in range(i+1, len(images))]
    selected_pairs = random.sample(all_pairs, num_pairs)
    
    print(f"Generating {num_pairs} dual-image videos...")
    
    for idx, (img1, img2) in enumerate(selected_pairs):
        print(f"\n[{idx+1}/{num_pairs}] Processing: {img1.name} + {img2.name}")
        
        # Random parameters
        direction = random.choice(CONFIG["direction_options"])
        speed = random.choice(CONFIG["speed_options"])
        movement_type = random.choice(CONFIG["movement_options"]) if not CONFIG["static_content"] else "static"
        
        for step in range(CONFIG["snr_steps"]):
            progress = step / (CONFIG["snr_steps"] - 1) if CONFIG["snr_steps"] > 1 else 0
            bg_noise = CONFIG["bg_noise_start"] - progress * (CONFIG["bg_noise_start"] - CONFIG["bg_noise_end"])
            fg_noise = CONFIG["fg_noise_start"] + progress * (CONFIG["fg_noise_start"] - CONFIG["fg_noise_end"])
            duration = random.uniform(CONFIG["duration_min"], CONFIG["duration_max"])
            
            if CONFIG["use_same_noise"]:
                print(f"  [{step+1}/{CONFIG['snr_steps']}] Noise={bg_noise:.1%}", end=" ")
            else:
                print(f"  [{step+1}/{CONFIG['snr_steps']}] BG={bg_noise:.1%} FG={fg_noise:.1%}", end=" ")
            
            generator.generate_dual_image_video(
                image_path1=str(img1),
                image_path2=str(img2),
                duration_sec=duration,
                bg_noise=bg_noise,
                fg_noise=fg_noise,
                use_same_noise=CONFIG["use_same_noise"],
                speckle_size=CONFIG["speckle_size"],
                speed=speed,
                direction=direction,
                movement_type=movement_type,
                static_content=CONFIG["static_content"],
                max_size=400  # Smaller size since we have 2 objects
            )
    
    print(f"\n✓ All {num_pairs * CONFIG['snr_steps']} dual-image videos generated!")

# ============================================
# COMPARISON & DEMOS
# ============================================

def generate_static_vs_moving_comparison():
    """Compare static vs moving content side-by-side"""
    print("\n=== STATIC VS MOVING COMPARISON ===\n")
    
    generator = DynamicVideoGenerator(
        output_dir=CONFIG["output_dir"] + "_comparison",
        width=CONFIG["resolution"][0],
        height=CONFIG["resolution"][1],
        fps=CONFIG["fps"]
    )
    
    test_words = ["NEURAL", "VISION", "MOTION"]
    
    for word in test_words:
        print(f"\nGenerating {word}...")
        
        # Static version
        print("  - Static content")
        generator.generate_text_video(
            text=word,
            duration_sec=5.0,
            bg_noise=0.55,
            fg_noise=0.55,
            use_same_noise=True,
            speckle_size=2,
            speed=2,
            direction="vertical",
            movement_type="static",
            static_content=True
        )
        
        # Moving version
        print("  - Moving content")
        generator.generate_text_video(
            text=word,
            duration_sec=5.0,
            bg_noise=0.55,
            fg_noise=0.55,
            use_same_noise=True,
            speckle_size=2,
            speed=2,
            direction="vertical",
            movement_type="smooth",
            static_content=False
        )
    
    print("\n✓ Comparison videos generated!")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPOOKYBENCH DYNAMIC VIDEO GENERATOR")
    print("=" * 60)
    
    # Choose what to run:
    
    # === BASIC TESTS ===
    # generate_single_video()
    # generate_static_vs_moving_comparison()
    
    # === WORD DATASET ===
    # batch_generate_from_wordlist()
    
    # === DEPTH MAP VIDEOS ===
    # generate_depth_map_videos("path/to/depth/videos")
    
    # === DUAL IMAGE DATASET ===
    generate_dual_image_dataset("./noise_video/images_masks")
    
    # === IMAGE DATASET ===
    # generate_image_dataset("./noise_video/images_masks")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)