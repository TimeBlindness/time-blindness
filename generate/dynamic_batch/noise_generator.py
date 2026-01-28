import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random

class NoiseAnimator:
    def __init__(self, width=960, height=540, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Animation parameters
        self.animation_speed = 2
        self.speckle_size = 1
        self.bg_noise_density = 0.5
        self.fg_noise_density = 0.5
        self.direction = "vertical"
        self.use_same_noise = True
        
        # Depth map mode
        self.use_depth_map = False
        self.depth_map = None
        self.depth_scale = 2.0
        
        # Noise arrays
        self.background_noise = np.zeros((height, width), dtype=np.uint8)
        self.foreground_noise = np.zeros((height, width), dtype=np.uint8)
        
    def generate_binary_noise(self, density):
        """Generate binary noise pattern with given density - VECTORIZED"""
        h, w = self.height, self.width
        s = self.speckle_size
        
        grid_h = (h + s - 1) // s
        grid_w = (w + s - 1) // s
        
        random_grid = (np.random.random((grid_h, grid_w)) > density).astype(np.uint8) * 255
        noise = np.repeat(np.repeat(random_grid, s, axis=0), s, axis=1)
        
        return noise[:h, :w]
    
    def refresh_noise(self):
        """Regenerate noise patterns"""
        if self.use_same_noise:
            self.background_noise = self.generate_binary_noise(self.bg_noise_density)
            self.foreground_noise = self.background_noise.copy()
        else:
            self.background_noise = self.generate_binary_noise(self.bg_noise_density)
            self.foreground_noise = self.generate_binary_noise(self.fg_noise_density)
    
    def load_depth_map(self, depth_map, depth_scale=2.0):
        """Load a depth map for depth-based animation"""
        # Resize to match canvas size
        self.depth_map = cv2.resize(depth_map, (self.width, self.height), 
                                    interpolation=cv2.INTER_LINEAR)
        # Normalize to 0-1
        self.depth_map = self.depth_map.astype(np.float32) / 255.0
        self.depth_scale = depth_scale
        self.use_depth_map = True
    
    def create_text_mask(self, text, position, font_size=100):
        """Create a mask from text at given position"""
        mask = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = position[0] - text_width // 2
        y = position[1] - text_height // 2

        draw.text((x, y), text, fill=255, font=font)
        return np.array(mask) > 0
    
    def create_image_mask(self, image_path, position, max_size=400):
        """Create a mask from an image at given position"""
        img = Image.open(image_path).convert('RGBA')
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        mask = Image.new('L', (self.width, self.height), 0)
        
        x = position[0] - img.width // 2
        y = position[1] - img.height // 2

        alpha = img.split()[3]
        mask.paste(alpha, (x, y))

        # if img.mode == 'RGBA':
        #     alpha = img.split()[3]
        #     mask.paste(alpha, (x, y))
        # else:
        #     img_gray = img.convert('L')
        #     mask.paste(img_gray, (x, y))
        
        return np.array(mask) > 0
    
    def create_video_frame_mask(self, video_frame, position=(None, None), scale=5):
        """
        Create mask from a video frame (e.g., Moving MNIST frame)
        
        Args:
            video_frame: 2D numpy array (grayscale video frame)
            position: (x, y) position to place frame, or (None, None) for center
            scale: Scaling factor for the frame
        """
        # Scale up the frame
        scaled_height = video_frame.shape[0] * scale
        scaled_width = video_frame.shape[1] * scale
        
        import cv2
        frame_scaled = cv2.resize(video_frame, (scaled_width, scaled_height), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Create mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Determine position
        if position[0] is None or position[1] is None:
            # Center
            x = (self.width - scaled_width) // 2
            y = (self.height - scaled_height) // 2
        else:
            x = position[0] - scaled_width // 2
            y = position[1] - scaled_height // 2
        
        # Ensure bounds
        x = max(0, min(x, self.width - scaled_width))
        y = max(0, min(y, self.height - scaled_height))
        
        # Paste frame
        y_end = min(y + scaled_height, self.height)
        x_end = min(x + scaled_width, self.width)
        
        mask[y:y_end, x:x_end] = frame_scaled[:y_end-y, :x_end-x]
        
        return mask > 0  # Binary mask
    
    def create_shape_mask(self, shape_type, position, size=100):
        """Create a mask from a geometric shape"""
        mask = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        
        x, y = position
        
        if shape_type == "circle":
            r = size // 2
            draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        elif shape_type == "rectangle":
            half = size // 2
            draw.rectangle([x-half, y-half, x+half, y+half], fill=255)
        elif shape_type == "polygon":
            sides = 5
            points = []
            for i in range(sides):
                angle = (2 * np.pi * i / sides) - np.pi/2
                px = x + size/2 * np.cos(angle)
                py = y + size/2 * np.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=255)
        
        return np.array(mask) > 0
    
    def animate_frame_vectorized(self, mask_data, bg_offset, fg_offset):
        """Generate frame - supports both standard and depth map modes"""
        if self.use_depth_map:
            return self._animate_frame_depth_map(bg_offset)
        else:
            return self._animate_frame_standard(mask_data, bg_offset, fg_offset)
    
    def _animate_frame_standard(self, mask_data, bg_offset, fg_offset):
        """Standard mode: opposing motion based on mask"""
        if self.direction == "vertical":
            bg_rolled = np.roll(self.background_noise, int(bg_offset), axis=0)
            fg_rolled = np.roll(self.foreground_noise, int(fg_offset), axis=0)
        else:
            bg_rolled = np.roll(self.background_noise, int(bg_offset), axis=1)
            fg_rolled = np.roll(self.foreground_noise, int(fg_offset), axis=1)
        
        frame = np.where(mask_data, fg_rolled, bg_rolled)
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return frame_bgr
    
    def _animate_frame_depth_map(self, base_offset):
        """Depth map mode: speed varies by depth value"""
        # Calculate per-pixel speed based on depth
        speed_map = 1.0 + self.depth_map * self.depth_scale
        
        # Create output frame
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        
        if self.direction == "vertical":
            # For each row, calculate offset based on depth
            for y in range(self.height):
                for x in range(self.width):
                    offset = int(base_offset * speed_map[y, x])
                    src_y = (y + offset) % self.height
                    frame[y, x] = self.background_noise[src_y, x]
        else:
            # Horizontal
            for y in range(self.height):
                for x in range(self.width):
                    offset = int(base_offset * speed_map[y, x])
                    src_x = (x + offset) % self.width
                    frame[y, x] = self.background_noise[y, src_x]
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame_bgr