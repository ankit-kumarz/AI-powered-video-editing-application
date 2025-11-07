import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import os
import mediapipe as mp
from PIL import Image
import json

class BackgroundRemovalService:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = None  # lazy init to avoid import overhead at startup
    
    async def process(self, upload_id: str, processing_status: dict, replace_with_image=None, blur_background=False):
        """Process background removal/replacement"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            processing_status[upload_id]["progress"] = 30
            frames = self._extract_frames(str(file_path))
            
            # Process frames
            processing_status[upload_id]["progress"] = 50
            processed_frames = []
            
            for i, frame in enumerate(frames):
                mask = self._segment_person_mask(frame)

                if replace_with_image:
                    processed_frame = self._composite_with_image_background(frame, mask, replace_with_image)
                elif blur_background:
                    processed_frame = self._composite_with_blur_background(frame, mask)
                else:
                    processed_frame = self._composite_with_solid_background(frame, mask, color=(0, 255, 0))  # green background

                processed_frames.append(processed_frame)
                
                # Update progress
                if i % 10 == 0:
                    progress = 50 + (i / len(frames)) * 30
                    processing_status[upload_id]["progress"] = int(progress)
            
            # Recombine frames into video
            processing_status[upload_id]["progress"] = 80
            output_path = output_dir / f"{upload_id}_background_removed.mp4"
            
            self._create_video_from_frames(processed_frames, str(output_path), original_video_path=str(file_path))
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Background removal error: {e}")
    
    def _extract_frames(self, video_path: str, max_frames: int = 1000000):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

            frame_count += 1
        
        cap.release()
        return frames
    
    def _segment_person_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a foreground mask using MediaPipe Selfie Segmentation."""
        if self.segmentation is None:
            self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.segmentation.process(rgb)
        mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    def _composite_with_image_background(self, frame: np.ndarray, mask: np.ndarray, background_image_path: str) -> np.ndarray:
        """Replace background with uploaded image (path)."""
        try:
            bg_img = Image.open(background_image_path)
            frame_height, frame_width = frame.shape[:2]
            bg_img = bg_img.resize((frame_width, frame_height))
            bg_img = np.array(bg_img)
            if len(bg_img.shape) == 3 and bg_img.shape[2] == 3:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)

            inv = cv2.bitwise_not(mask)
            fg = cv2.bitwise_and(frame, frame, mask=mask)
            bg = cv2.bitwise_and(bg_img, bg_img, mask=inv)
            return cv2.add(fg, bg)
        except Exception as e:
            print(f"Background replacement error: {e}")
            return frame

    def _composite_with_blur_background(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply blurred background using mask."""
        blurred_bg = cv2.GaussianBlur(frame, (21, 21), 0)
        inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(blurred_bg, blurred_bg, mask=inv)
        return cv2.add(fg, bg)

    def _composite_with_solid_background(self, frame: np.ndarray, mask: np.ndarray, color=(0, 0, 0)) -> np.ndarray:
        bg = np.full_like(frame, color, dtype=np.uint8)
        inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(bg, bg, mask=inv)
        return cv2.add(fg, bg)

    def _replace_background_with_image(self, foreground_frame, background_image):
        """Replace background with uploaded image"""
        try:
            # Load background image
            if hasattr(background_image, 'file'):
                bg_img = Image.open(background_image.file)
            else:
                bg_img = Image.open(background_image)
            
            # Resize background to match frame size
            frame_height, frame_width = foreground_frame.shape[:2]
            bg_img = bg_img.resize((frame_width, frame_height))
            bg_img = np.array(bg_img)
            
            # Convert to BGR if needed
            if len(bg_img.shape) == 3 and bg_img.shape[2] == 3:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
            
            # Create mask from foreground (assuming white background after rembg)
            mask = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
            
            # Apply mask
            result = foreground_frame.copy()
            result[mask == 0] = bg_img[mask == 0]
            
            return result
            
        except Exception as e:
            print(f"Background replacement error: {e}")
            return foreground_frame
    
    def _apply_blurred_background(self, original_frame, foreground_frame):
        """Apply blurred background"""
        try:
            # Create mask from foreground
            mask = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)
            
            # Blur original frame
            blurred_bg = cv2.GaussianBlur(original_frame, (21, 21), 0)
            
            # Apply mask
            result = foreground_frame.copy()
            result[mask == 0] = blurred_bg[mask == 0]
            
            return result
            
        except Exception as e:
            print(f"Blurred background error: {e}")
            return foreground_frame
    
    def _create_video_from_frames(self, frames, output_path: str, fps: int = None, original_video_path: str = None):
        """Create video from processed frames"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        if fps is None and original_video_path:
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()
        if fps is None or fps <= 0:
            fps = 25
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Convert to MP4 using FFmpeg for better compatibility
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_path)
        # Ensure at least 720p height and merge original audio if present
        vf = "scale=-2:'if(gte(ih,720),ih,720)'"
        cmd = [
            'ffmpeg', '-i', temp_path,
            '-i', original_video_path if original_video_path else temp_path,
            '-vf', vf,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
        except subprocess.CalledProcessError:
            # If FFmpeg fails, keep the original
            os.rename(temp_path, output_path)
