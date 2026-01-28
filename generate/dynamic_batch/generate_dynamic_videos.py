import numpy as np
import cv2
import random
import os
from pathlib import Path
from noise_generator import NoiseAnimator

class DynamicVideoGenerator:
    def __init__(self, output_dir="videos", width=960, height=540, fps=60):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.width = width
        self.height = height
        self.fps = fps
        
    def generate_smooth_path(self, duration_frames, margin=100):
        """Generate smooth random path"""
        start_x = random.randint(margin, self.width - margin)
        start_y = random.randint(margin, self.height - margin)
        end_x = random.randint(margin, self.width - margin)
        end_y = random.randint(margin, self.height - margin)
        
        num_control_points = 5
        control_x = [start_x] + [random.randint(margin, self.width - margin) 
                                 for _ in range(num_control_points - 2)] + [end_x]
        control_y = [start_y] + [random.randint(margin, self.height - margin) 
                                 for _ in range(num_control_points - 2)] + [end_y]
        
        t_values = np.linspace(0, 1, duration_frames)
        positions = []
        
        for t in t_values:
            idx = min(int(t * (num_control_points - 1)), num_control_points - 2)
            local_t = (t * (num_control_points - 1)) - idx
            
            x = self._cubic_interp(control_x, idx, local_t)
            y = self._cubic_interp(control_y, idx, local_t)
            
            x = np.clip(x, margin, self.width - margin)
            y = np.clip(y, margin, self.height - margin)
            
            positions.append((int(x), int(y)))
        
        return positions
    
    def _cubic_interp(self, points, idx, t):
        """Cubic interpolation"""
        if idx == 0:
            p0, p1, p2, p3 = points[0], points[0], points[1], points[2]
        elif idx >= len(points) - 2:
            p0, p1, p2, p3 = points[-3], points[-2], points[-1], points[-1]
        else:
            p0, p1, p2, p3 = points[idx-1], points[idx], points[idx+1], points[idx+2]
        
        return 0.5 * (
            2 * p1 +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t**3
        )
    
    def generate_text_video(self, text, duration_sec=5, 
                           bg_noise=0.5, fg_noise=0.5,
                           use_same_noise=True,
                           speckle_size=3, speed=2, direction="vertical",
                           movement_type="smooth", static_content=False):
        """Generate video with text - supports static or moving content"""
        animator = NoiseAnimator(self.width, self.height, self.fps)
        animator.bg_noise_density = bg_noise
        animator.fg_noise_density = fg_noise
        animator.use_same_noise = use_same_noise
        animator.speckle_size = speckle_size
        animator.animation_speed = speed
        animator.direction = direction
        
        animator.refresh_noise()
        
        total_frames = int(duration_sec * self.fps)
        
        # Generate movement path
        if static_content or movement_type == "static":
            positions = [(self.width // 2, self.height // 2)] * total_frames
            move_label = "static"
        elif movement_type == "smooth":
            positions = self.generate_smooth_path(total_frames)
            move_label = "smooth"
        elif movement_type == "circular":
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(self.width, self.height) // 3
            angles = np.linspace(0, 2*np.pi, total_frames)
            positions = [(int(center_x + radius * np.cos(a)), 
                         int(center_y + radius * np.sin(a))) for a in angles]
            move_label = "circular"
        elif movement_type == "linear":
            xs = np.linspace(100, self.width - 100, total_frames).astype(int)
            positions = [(x, self.height // 2) for x in xs]
            move_label = "linear"
        else:
            positions = [(self.width // 2, self.height // 2)] * total_frames
            move_label = "static"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if use_same_noise:
            filename = (f"{text}_noise{int(bg_noise*100)}_"
                       f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        else:
            filename = (f"{text}_bg{int(bg_noise*100)}_fg{int(fg_noise*100)}_"
                       f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        
        output_path = self.output_dir / filename
        
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, 
                             (self.width, self.height))
        
        mask_cache = {}
        bg_offset = 0
        fg_offset = 0
        
        for frame_idx, position in enumerate(positions):
            pos_key = position
            if pos_key not in mask_cache:
                mask_cache[pos_key] = animator.create_text_mask(text, position, font_size=80)
            
            mask = mask_cache[pos_key]
            frame = animator.animate_frame_vectorized(mask, bg_offset, fg_offset)
            out.write(frame)
            
            bg_offset += animator.animation_speed
            fg_offset -= animator.animation_speed
            
            if len(mask_cache) > 50:
                mask_cache.clear()
        
        out.release()
        print(f"✓ {filename}")
        return str(output_path)
    
    def generate_video_with_depth(self, depth_video_path, duration_sec=5,
                                  bg_noise=0.5, fg_noise=0.5,
                                  use_same_noise=True,
                                  speckle_size=3, speed=2, direction="vertical",
                                  depth_scale=2.0):
        """Generate video using depth map from another video"""
        animator = NoiseAnimator(self.width, self.height, self.fps)
        animator.bg_noise_density = bg_noise
        animator.fg_noise_density = fg_noise
        animator.use_same_noise = use_same_noise
        animator.speckle_size = speckle_size
        animator.animation_speed = speed
        animator.direction = direction
        animator.depth_scale = depth_scale
        
        animator.refresh_noise()
        
        # Load depth video
        depth_cap = cv2.VideoCapture(str(depth_video_path))
        if not depth_cap.isOpened():
            print(f"Error: Cannot open depth video {depth_video_path}")
            return None
        
        total_depth_frames = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(duration_sec * self.fps)
        
        # Setup video writer
        video_name = Path(depth_video_path).stem
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = (f"{video_name}_depth_noise{int(bg_noise*100)}_"
                   f"speckle{speckle_size}_speed{speed}_{direction}.mp4")
        output_path = self.output_dir / filename
        
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                             (self.width, self.height))
        
        bg_offset = 0
        
        for frame_idx in range(total_frames):
            # Read corresponding depth frame (loop if needed)
            depth_frame_idx = frame_idx % total_depth_frames
            depth_cap.set(cv2.CAP_PROP_POS_FRAMES, depth_frame_idx)
            ret, depth_frame = depth_cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale depth map
            if len(depth_frame.shape) == 3:
                depth_map = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            else:
                depth_map = depth_frame
            
            # Load depth map
            animator.load_depth_map(depth_map, depth_scale)
            
            # Generate frame
            frame = animator.animate_frame_vectorized(None, bg_offset, 0)
            out.write(frame)
            
            bg_offset += animator.animation_speed
        
        depth_cap.release()
        out.release()
        print(f"✓ {filename}")
        return str(output_path)
    
    def generate_image_video(self, image_path, duration_sec=5,
                        bg_noise=0.5, fg_noise=0.5,
                        use_same_noise=True,
                        speckle_size=3, speed=2, direction="vertical",
                        movement_type="smooth", static_content=False):
        """Generate image video with flexible noise control"""
        animator = NoiseAnimator(self.width, self.height, self.fps)
        animator.bg_noise_density = bg_noise
        animator.fg_noise_density = fg_noise
        animator.use_same_noise = use_same_noise
        animator.speckle_size = speckle_size
        animator.animation_speed = speed
        animator.direction = direction
        
        animator.refresh_noise()
        
        total_frames = int(duration_sec * self.fps)
        
        if static_content or movement_type == "static":
            positions = [(self.width//2, self.height//2)] * total_frames
            move_label = "static"
        elif movement_type == "smooth":
            positions = self.generate_smooth_path(total_frames)
            move_label = "smooth"
        elif movement_type == "circular":
            center_x, center_y = self.width // 2, self.height // 2
            radius = min(self.width, self.height) // 3
            angles = np.linspace(0, 2*np.pi, total_frames)
            positions = [(int(center_x + radius * np.cos(a)), 
                        int(center_y + radius * np.sin(a))) for a in angles]
            move_label = "circular"
        elif movement_type == "linear":
            xs = np.linspace(100, self.width - 100, total_frames).astype(int)
            positions = [(x, self.height // 2) for x in xs]
            move_label = "linear"
        else:
            raise ValueError("not a valid movement type")
        
        image_name = Path(image_path).stem
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if use_same_noise:
            filename = (f"{image_name}_noise{int(bg_noise*100)}_"
                    f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        else:
            filename = (f"{image_name}_bg{int(bg_noise*100)}_fg{int(fg_noise*100)}_"
                    f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        
        output_path = self.output_dir / filename
        
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                            (self.width, self.height))
        
        mask_cache = {}
        bg_offset = 0
        fg_offset = 0
        
        for frame_idx, position in enumerate(positions):
            pos_key = position
            if pos_key not in mask_cache:
                mask_cache[pos_key] = animator.create_image_mask(image_path, position, max_size=300)
            
            mask = mask_cache[pos_key]
            frame = animator.animate_frame_vectorized(mask, bg_offset, fg_offset)
            out.write(frame)
            
            bg_offset += animator.animation_speed
            fg_offset -= animator.animation_speed
            
            if len(mask_cache) > 50:
                mask_cache.clear()
        
        out.release()
        print(f"✓ {filename}")
        return str(output_path)
    
    def generate_snr_gradient_videos(self, text, snr_steps=100,
                                     use_same_noise=True,
                                     bg_noise_start=0.70, bg_noise_end=0.45,
                                     fg_noise_start=0.50, fg_noise_end=0.55,
                                     duration_range=(5.0, 10.0),
                                     speed_options=[1, 2],
                                     direction_options=["vertical", "horizontal"],
                                     movement_options=["smooth", "circular", "linear"],
                                     static_content=False):
        """Generate SNR gradient videos with full control"""
        direction = random.choice(direction_options)
        speed = random.choice(speed_options)
        movement_type = random.choice(movement_options) if not static_content else "static"
        
        mode = "TEMPORAL" if use_same_noise else "SPATIAL+TEMPORAL"
        content_mode = "STATIC" if static_content else "MOVING"
        print(f"\n[{text}] {mode} SNR | {content_mode} | dir={direction}, speed={speed}, move={movement_type}")
        
        for step in range(snr_steps):
            progress = step / (snr_steps - 1) if snr_steps > 1 else 0
            
            bg_noise = bg_noise_start - progress * (bg_noise_start - bg_noise_end)
            fg_noise = fg_noise_start + progress * (fg_noise_start - fg_noise_end)
            
            speckle_size = 1
            duration = random.uniform(*duration_range)
            
            if use_same_noise:
                print(f"  [{step+1}/{snr_steps}] Noise={bg_noise:.1%} {duration:.1f}s", end=" ")
            else:
                print(f"  [{step+1}/{snr_steps}] BG={bg_noise:.1%} FG={fg_noise:.1%} {duration:.1f}s", end=" ")
            
            self.generate_text_video(
                text=text,
                duration_sec=duration,
                bg_noise=bg_noise,
                fg_noise=fg_noise,
                use_same_noise=use_same_noise,
                speckle_size=speckle_size,
                speed=speed,
                direction=direction,
                movement_type=movement_type,
                static_content=static_content
            )

    
    def _avoid_collisions(self, positions1, positions2, min_distance=250):
        """Adjust paths to avoid collisions while maintaining smooth movement"""
        adjusted_pos1 = [positions1[0]]
        adjusted_pos2 = [positions2[0]]
        
        for i in range(1, len(positions1)):
            x1, y1 = positions1[i]
            x2, y2 = positions2[i]
            
            # Calculate distance
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if dist < min_distance:
                # Too close - push them apart
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    dx, dy = 100, 0
                
                # Normalize and scale to min_distance
                current_dist = np.sqrt(dx**2 + dy**2)
                if current_dist > 0:
                    scale = (min_distance - dist) / (2 * current_dist)
                    
                    # Push both away from each other
                    x1 = int(x1 - dx * scale)
                    y1 = int(y1 - dy * scale)
                    x2 = int(x2 + dx * scale)
                    y2 = int(y2 + dy * scale)
                    
                    # Keep in bounds
                    x1 = np.clip(x1, 150, self.width - 150)
                    y1 = np.clip(y1, 150, self.height - 150)
                    x2 = np.clip(x2, 150, self.width - 150)
                    y2 = np.clip(y2, 150, self.height - 150)
            
            adjusted_pos1.append((x1, y1))
            adjusted_pos2.append((x2, y2))
        
        return adjusted_pos1, adjusted_pos2

    def generate_dual_image_video(self, image_path1, image_path2, duration_sec=5,
                                bg_noise=0.5, fg_noise=0.5,
                                use_same_noise=True,
                                speckle_size=3, speed=2, direction="vertical",
                                movement_type="smooth", static_content=False,
                                max_size=200, min_distance=250):
        """Generate video with two images that move freely but don't overlap"""
        animator = NoiseAnimator(self.width, self.height, self.fps)
        animator.bg_noise_density = bg_noise
        animator.fg_noise_density = fg_noise
        animator.use_same_noise = use_same_noise
        animator.speckle_size = speckle_size
        animator.animation_speed = speed
        animator.direction = direction
        
        animator.refresh_noise()
        
        total_frames = int(duration_sec * self.fps)
        
        # Generate initial paths independently
        if static_content or movement_type == "static":
            positions1 = [(self.width//3, self.height//2)] * total_frames
            positions2 = [(2*self.width//3, self.height//2)] * total_frames
            move_label = "static"
        elif movement_type == "smooth":
            # Both move freely across entire canvas
            positions1 = self.generate_smooth_path(total_frames, margin=150)
            positions2 = self.generate_smooth_path(total_frames, margin=150)
            # Avoid collisions
            positions1, positions2 = self._avoid_collisions(positions1, positions2, min_distance)
            move_label = "smooth"
        elif movement_type == "circular":
            # Two circular paths with different centers/radii
            center1_x = self.width // 2 + random.randint(-100, 100)
            center1_y = self.height // 2 + random.randint(-100, 100)
            center2_x = self.width // 2 + random.randint(-100, 100)
            center2_y = self.height // 2 + random.randint(-100, 100)
            
            radius1 = min(self.width, self.height) // 4
            radius2 = min(self.width, self.height) // 5
            
            angles1 = np.linspace(0, 2*np.pi, total_frames)
            angles2 = np.linspace(np.pi, 3*np.pi, total_frames)
            
            positions1 = [(int(center1_x + radius1 * np.cos(a)), 
                        int(center1_y + radius1 * np.sin(a))) for a in angles1]
            positions2 = [(int(center2_x + radius2 * np.cos(a)), 
                        int(center2_y + radius2 * np.sin(a))) for a in angles2]
            move_label = "circular"
        elif movement_type == "linear":
            # Both moving in opposite directions
            xs1 = np.linspace(150, self.width - 150, total_frames).astype(int)
            xs2 = np.linspace(self.width - 150, 150, total_frames).astype(int)
            
            positions1 = [(x, self.height // 3) for x in xs1]
            positions2 = [(x, 2 * self.height // 3) for x in xs2]
            move_label = "linear"
        else:
            positions1 = [(self.width//3, self.height//2)] * total_frames
            positions2 = [(2*self.width//3, self.height//2)] * total_frames
            move_label = "static"
        
        # Create filename
        name1 = Path(image_path1).stem
        name2 = Path(image_path2).stem
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if use_same_noise:
            filename = (f"{name1}_{name2}_noise{int(bg_noise*100)}_"
                    f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        else:
            filename = (f"{name1}_{name2}_bg{int(bg_noise*100)}_fg{int(fg_noise*100)}_"
                    f"speckle{speckle_size}_speed{speed}_{direction}_{move_label}.mp4")
        
        output_path = self.output_dir / filename
        
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps,
                            (self.width, self.height))
        
        mask_cache1 = {}
        mask_cache2 = {}
        bg_offset = 0
        fg_offset = 0
        
        for frame_idx in range(total_frames):
            pos1 = positions1[frame_idx]
            pos2 = positions2[frame_idx]
            
            # Get or create masks
            if pos1 not in mask_cache1:
                mask_cache1[pos1] = animator.create_image_mask(image_path1, pos1, max_size=max_size)
            if pos2 not in mask_cache2:
                mask_cache2[pos2] = animator.create_image_mask(image_path2, pos2, max_size=max_size)
            
            mask1 = mask_cache1[pos1]
            mask2 = mask_cache2[pos2]
            
            # Combine masks - both are foreground
            combined_mask = np.logical_or(mask1, mask2)
            
            # Generate frame
            frame = animator.animate_frame_vectorized(combined_mask, bg_offset, fg_offset)
            out.write(frame)
            
            bg_offset += animator.animation_speed
            fg_offset -= animator.animation_speed
            
            if len(mask_cache1) > 30:
                mask_cache1.clear()
            if len(mask_cache2) > 30:
                mask_cache2.clear()
        
        out.release()
        print(f"✓ {filename}")
        return str(output_path)

