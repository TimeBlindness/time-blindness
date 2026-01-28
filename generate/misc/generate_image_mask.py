from transformers import pipeline
from PIL import Image
from pathlib import Path

# Initialize pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

input_folder = "./noise_video/dynamic_videos_images/temp_images/silhouette"
output_folder = "./noise_video/dynamic_videos_images/temp_images/masks"

Path(output_folder).mkdir(exist_ok=True)

for img_path in Path(input_folder).glob("*"):
    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        print(f"Processing {img_path.name}...")
        
        # Remove background
        result = pipe(str(img_path))  # Returns PIL image with alpha
        
        # Save as PNG with transparency
        output_path = Path(output_folder) / f"{img_path.stem}.png"
        result.save(output_path)
        print(f"Saved to {output_path}")

print("\nDone! All images processed.")