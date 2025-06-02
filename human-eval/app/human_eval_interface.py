import os
import sys
# import json # Not used in the final version of the relevant code
import csv
import datetime
import argparse
import tempfile
# import functools # Not used in the final version of the relevant code

# Override default tempfile.gettempdir to use our own directory for Gradio's internal temp files
orig_gettempdir = tempfile.gettempdir

def patched_gettempdir():
    # This temp_dir is used by Gradio internally if it calls tempfile.gettempdir()
    temp_dir = os.path.join(os.getcwd(), "temp_gradio_internal") 
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

# Apply the patch
tempfile.gettempdir = patched_gettempdir

import gradio as gr
import cv2
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Video Annotation Tool for SpookyBench Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to SpookyBench dataset directory containing images/, shapes/, videos/, words/ folders'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./annotation_data',
        help='Directory to save annotation files (default: ./annotation_data)'
    )
    
    parser.add_argument(
        '--temp_dir',
        type=str,
        default='./temp_processed_videos', # Default for our processed videos
        help='Directory for temporary processed video files (default: ./temp_processed_videos)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for Gradio app (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    
    return parser.parse_args()

class VideoAnnotationTool:
    def __init__(self, data_path, output_dir, temp_dir_processed_videos):
        self.data_path = data_path
        self.output_dir = output_dir
        # This directory is for videos we process and output
        self.processed_videos_temp_dir = temp_dir_processed_videos 
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_videos_temp_dir, exist_ok=True)
        
        self.categories = {
            "images": os.path.join(data_path, "images"),
            "shapes": os.path.join(data_path, "shapes"), 
            "videos": os.path.join(data_path, "videos"),
            "words": os.path.join(data_path, "words")
        }
        
        self.frame_rates = [1, 5, 10, 20, 30, 60]
        self.validate_data_structure()
    
    def validate_data_structure(self):
        """Validate that the data directory has the expected structure"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        missing_categories = []
        for category, path in self.categories.items():
            if not os.path.exists(path):
                missing_categories.append(category)
        
        if missing_categories:
            print(f"Warning: Missing category folders: {missing_categories}")
            print("Expected folder structure:")
            print(f"{self.data_path}/")
            for category_name in self.categories.keys(): # Use category_name for clarity
                print(f"├── {category_name}/")
        
        total_videos = 0
        for category_path in self.categories.values():
            if os.path.exists(category_path): # Check if path exists before listing
                 total_videos += sum(1 for f in os.listdir(category_path) if f.endswith('.mp4'))
        
        if total_videos == 0:
            print("Warning: No .mp4 files found in any category folder. The video selection dropdown will be empty.")
        
        print(f"Found {total_videos} videos across all categories")

    def get_video_paths(self):
        """Get all video paths organized by category"""
        video_paths = {}
        for category, directory in self.categories.items():
            if os.path.exists(directory):
                video_paths[category] = []
                for filename in os.listdir(directory):
                    if filename.endswith('.mp4'):
                        video_id = Path(filename).stem  # Get the ID without extension
                        video_paths[category].append({
                            "id": video_id,
                            "path": os.path.join(directory, filename)
                        })
        return video_paths

    def get_all_videos(self):
        """Get all videos as a flat list for the dropdown"""
        video_paths = self.get_video_paths()
        all_videos = []
        for category, videos in video_paths.items():
            for video_item in videos: # Renamed to avoid confusion with category 'videos'
                all_videos.append(f"{category}/{video_item['id']}")
        return all_videos

    def video_path_from_selection(self, selection):
        """Convert selection string to actual file path"""
        if not selection:
            return None
        try:
            category, video_id = selection.split('/')
            if category not in self.categories:
                print(f"Error: Unknown category '{category}' in selection '{selection}'. Valid categories are: {list(self.categories.keys())}")
                return None
            return os.path.join(self.categories[category], f"{video_id}.mp4")
        except ValueError:
            print(f"Error: Invalid video selection format '{selection}'. Expected 'category/video_id'.")
            return None


    def save_annotation(self, video_selection, fps, annotation):
        """Save annotation to CSV file"""
        if not video_selection : # Check if selection is empty or None
            return "⚠️ Please select a video first."
        if not annotation or not annotation.strip(): # Check if annotation is empty or only whitespace
            return "⚠️ Please provide an annotation."
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        annotation_file = os.path.join(self.output_dir, "video_annotations.csv")
        file_exists = os.path.exists(annotation_file)
        
        try:
            with open(annotation_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header if file doesn't exist or is empty
                if not file_exists or os.path.getsize(annotation_file) == 0:
                    writer.writerow(["timestamp", "video", "fps", "annotation"])
                writer.writerow([timestamp, video_selection, fps, annotation])
            return f"✅ Annotation saved for {video_selection} at {fps} fps"
        except Exception as e:
            print(f"Error saving annotation: {e}") # Log to console for debugging
            return f"❌ Error saving annotation: {e}"


    def display_video_at_fps(self, video_selection, target_fps_from_dropdown):
        """Re-encodes the video to the selected frame rate and returns the path to the processed video."""
        if not video_selection:
            gr.Warning("No video selected. Please select a video from the dropdown.")
            return None # Return None to clear the video player or show an error
        
        if target_fps_from_dropdown is None:
            gr.Warning("No FPS selected. Please select a frame rate.")
            return None
        
        try:
            target_fps = float(target_fps_from_dropdown) # Ensure it's a float for cv2.VideoWriter
            if target_fps <= 0:
                gr.Error("FPS must be a positive value.")
                return None
        except ValueError:
            gr.Error(f"Invalid FPS value: {target_fps_from_dropdown}")
            return None

        original_video_path_str = self.video_path_from_selection(video_selection)
        if not original_video_path_str or not os.path.exists(original_video_path_str):
            gr.Error(f"Original video path not found or invalid: {original_video_path_str}")
            return None

        original_video_path_obj = Path(original_video_path_str)
        video_id = original_video_path_obj.stem

        os.makedirs(self.processed_videos_temp_dir, exist_ok=True) 
        
        temp_output_filename = f"{video_id}_processed_{int(target_fps)}fps.mp4" # Use int for filename
        processed_video_path = Path(self.processed_videos_temp_dir) / temp_output_filename
        processed_video_path_str = str(processed_video_path)

        cap = cv2.VideoCapture(original_video_path_str)
        if not cap.isOpened():
            gr.Error(f"Could not open video source: {original_video_path_str}")
            return None

        original_fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if original_fps_video == 0:
            gr.Warning(f"Original FPS of '{original_video_path_obj.name}' is reported as 0. Assuming 30 FPS for processing calculations. Playback might be unexpected if this assumption is incorrect.")
            # original_fps_video = 30.0 # Don't change original_fps_video, just use for logging/info

        # Try H.264 ('avc1'), then MPEG-4 ('mp4v') as fallback
        fourcc_options = {
            'avc1': cv2.VideoWriter_fourcc(*'avc1'), # H.264
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4
        }
        
        out = None
        selected_fourcc_str = ""

        for name, code in fourcc_options.items():
            # Using processed_video_path_str directly in VideoWriter
            temp_out_writer = cv2.VideoWriter(processed_video_path_str, code, target_fps, (frame_width, frame_height))
            if temp_out_writer.isOpened():
                out = temp_out_writer
                selected_fourcc_str = name
                print(f"Successfully opened VideoWriter with FourCC: {selected_fourcc_str} for '{processed_video_path.name}'")
                break
            else: # Release if it was somehow created but not "opened" in a usable state
                temp_out_writer.release() # Good practice though likely not strictly needed if not opened
                print(f"Failed to open VideoWriter with FourCC: {name}")
        
        if out is None or not out.isOpened():
            cap.release()
            gr.Error(f"Could not open video writer for '{processed_video_path.name}'. Tried codecs: {list(fourcc_options.keys())}. Ensure OpenCV has video writing support (e.g., FFmpeg backend installed and configured with OpenCV).")
            if processed_video_path.exists(): # Clean up potentially created 0-byte file
                try: os.remove(processed_video_path_str)
                except OSError as e: print(f"Warning: Could not remove temp file '{processed_video_path_str}': {e}")
            return None
            
        print(f"Processing '{original_video_path_obj.name}' (Res: {frame_width}x{frame_height}, OrigFPS reported: {original_fps_video:.2f}) -> Outputting at {target_fps:.2f} FPS as '{processed_video_path.name}' using {selected_fourcc_str}")
        
        frames_written = 0
        total_frames_read = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames_read +=1
            out.write(frame)
            frames_written += 1
        
        print(f"Finished processing. Read {total_frames_read} frames, wrote {frames_written} frames to '{processed_video_path.name}'.")

        cap.release()
        out.release() # Crucial: This finalizes the video file.

        if frames_written == 0:
            message = f"No frames were written to '{processed_video_path.name}'. "
            if total_frames_read > 0:
                message += f"Read {total_frames_read} frames, but writing failed. Check codec compatibility and disk space."
                gr.Error(message)
            else:
                message += f"Original video '{original_video_path_obj.name}' might be empty or unreadable by OpenCV."
                gr.Warning(message)
            
            if processed_video_path.exists(): # Clean up empty/failed file
                try: processed_video_path.unlink()
                except OSError as e: print(f"Warning: Could not remove failed processed file '{processed_video_path_str}': {e}")
            return None

        print(f"Successfully processed video. Output: {processed_video_path_str}")
        return processed_video_path_str
    
    def create_ui(self):
        """Create the Gradio UI for video annotation"""
        with gr.Blocks(title="SpookyBench Video Annotation Tool", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 👻 SpookyBench Video Annotation Tool")
            gr.Markdown("Annotate videos at different frame rates to study temporal pattern recognition.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎬 Video Selection & Settings")
                    video_dropdown = gr.Dropdown(
                        choices=self.get_all_videos(),
                        label="Select Video",
                        info="Choose a video to annotate (category/video_id)"
                    )
                    fps_dropdown = gr.Dropdown(
                        choices=self.frame_rates,
                        value=10, 
                        label="Frame Rate (FPS)",
                        info="Select target FPS for video processing and playback"
                    )
                    load_button = gr.Button("🔄 Process and Load Video", variant="primary")
                    
                    gr.Markdown("### ✍️ Annotation")
                    annotation_input = gr.Textbox(
                        lines=3,
                        label="Your Annotation",
                        placeholder="Enter what you see in the video (1 word to 1 sentence)",
                        info="Describe the temporal pattern you observe"
                    )
                    save_button = gr.Button("💾 Save Annotation", variant="secondary")
                    status_msg = gr.Textbox(label="Status", interactive=False, lines=2, show_label=True) # Ensure label is shown
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 📺 Video Preview")
                    video_output = gr.Video(label="Processed Video Preview", height=400) # Updated label
            
            load_button.click(
                fn=self.display_video_at_fps,
                inputs=[video_dropdown, fps_dropdown],
                outputs=video_output
            ).then(
                fn=lambda: "", # Clear status on new load attempt
                outputs=status_msg 
            )
            
            save_button.click(
                fn=self.save_annotation,
                inputs=[video_dropdown, fps_dropdown, annotation_input],
                outputs=status_msg
            ).then(
                fn=lambda: "", # Clear annotation input after saving
                inputs=[], 
                outputs=annotation_input
            )
        
        return app

def main():
    """Main function to run the application"""
    args = parse_arguments()
    
    print("🚀 Starting SpookyBench Video Annotation Tool...")
    print(f"📁 Data path: {args.data_path}")
    print(f"💾 Output directory (annotations): {args.output_dir}")
    print(f"📂 Temp directory (processed videos): {args.temp_dir}") 
    print(f"🌐 Port: {args.port}")
    
    try:
        tool = VideoAnnotationTool(
            data_path=args.data_path,
            output_dir=args.output_dir,
            temp_dir_processed_videos=args.temp_dir 
        )
        
        app = tool.create_ui()
        
        app.launch(
            server_port=args.port,
            share=args.share,
            server_name="0.0.0.0" 
        )
        
    except Exception as e:
        print(f"❌ Error during application startup or execution: {e}")
        import traceback
        traceback.print_exc() 
        sys.exit(1)

if __name__ == "__main__":
    main()