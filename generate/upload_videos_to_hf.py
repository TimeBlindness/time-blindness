import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
import shutil
import os

# Configuration
MOTION_DIR = "motion_only_videos"
TEXT_DIR = "text_fg_bg_noise_videos"
DATASET_FOLDER = "video_dataset"
REPO_ID = "mukul54/motion-text-classifier-v2"
NUM_WORKERS=4
HF_TOKEN = "xxxxx"

train_folder = f"{DATASET_FOLDER}/train"

# Create directories
os.makedirs(f"{train_folder}/motion_only", exist_ok=True)
os.makedirs(f"{train_folder}/text_fg_bg", exist_ok=True)

# Copy motion videos
motion_src_path = Path(MOTION_DIR)
motion_videos = sorted(motion_src_path.glob("motion_*.mp4"))
print(f"Copying {len(motion_videos)} motion videos...")
for src in motion_videos:
    dst = f"{train_folder}/motion_only/{src.name}"
    shutil.copy2(src, dst)

# Copy text videos
text_src_path = Path(TEXT_DIR)
text_videos = sorted(text_src_path.glob("text_*.mp4"))
print(f"Copying {len(text_videos)} text videos...")
for src in text_videos:
    dst = f"{train_folder}/text_fg_bg/{src.name}"
    shutil.copy2(src, dst)

# Load text metadata and create metadata.csv for text videos
text_metadata = pd.read_csv(f"{TEXT_DIR}/metadata.csv")

# Create metadata.csv for text videos with file_name column
text_meta_df = pd.DataFrame({
    'file_name': text_metadata['filename'],
    'text': text_metadata['text']
})
text_meta_df.to_csv(f"{train_folder}/text_fg_bg/metadata.csv", index=False)

print(f"\nUploading to HuggingFace: {REPO_ID}")
api = HfApi(token=HF_TOKEN)

# Create the repository first
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    exist_ok=True
)

# Use upload_large_folder for large video datasets
api.upload_large_folder(
    folder_path=DATASET_FOLDER,
    repo_id=REPO_ID,
    repo_type="dataset",
    num_workers=NUM_WORKERS
)

print("Upload complete!")
