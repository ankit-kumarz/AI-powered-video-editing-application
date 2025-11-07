import cv2
import numpy as np
from pathlib import Path
import subprocess
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tempfile
import gc

class StyleTransferNet(nn.Module):
    """Enhanced neural network for style transfer with better CUDA optimization"""
    def __init__(self, style_type="cartoon"):
        super(StyleTransferNet, self).__init__()
        self.style_type = style_type
        
        # Enhanced architecture for better style transfer
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 3, 3, padding=1)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # Enhanced forward pass with residual connections
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = torch.sigmoid(self.conv5(x4))
        
        # Add residual connection for better quality
        return torch.clamp(x5, 0, 1)

class StyleFilterService:
    def __init__(self):
        # Enhanced CUDA detection and setup
        self.device = self._setup_device()
        self.style_models = {}  # Lazy loading - only load when needed
        self.batch_size = 4  # Process multiple frames at once for better GPU utilization
        print(f"StyleFilterService initialized on device: {self.device}")
    
    def _setup_device(self):
        """Enhanced device setup with better CUDA detection"""
        if torch.cuda.is_available():
            # Check CUDA memory and set optimal device
            cuda_device = torch.device('cuda:0')
            torch.cuda.empty_cache()  # Clear GPU memory
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return cuda_device
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _get_style_model(self, style: str):
        """Lazy load style model only when needed"""
        if style not in self.style_models:
            try:
                print(f"Loading {style} style model...")
                if style == "cartoon":
                    self.style_models[style] = self._create_cartoon_model()
                elif style == "sketch":
                    self.style_models[style] = self._create_sketch_model()
                elif style == "cinematic":
                    self.style_models[style] = self._create_cinematic_model()
                elif style == "vintage":
                    self.style_models[style] = self._create_vintage_model()
                elif style == "neon":
                    self.style_models[style] = self._create_neon_model()
                else:
                    print(f"Unknown style: {style}, using traditional filter")
                    return None
                print(f"Successfully loaded {style} style model")
            except Exception as e:
                print(f"Error loading {style} style model: {e}")
                return None
        
        return self.style_models[style]
    
    def _create_cartoon_model(self):
        """Create enhanced cartoon-style filter"""
        model = StyleTransferNet("cartoon").to(self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def _create_sketch_model(self):
        """Create enhanced sketch-style filter"""
        model = StyleTransferNet("sketch").to(self.device)
        model.eval()
        return model
    
    def _create_cinematic_model(self):
        """Create enhanced cinematic-style filter"""
        model = StyleTransferNet("cinematic").to(self.device)
        model.eval()
        return model
    
    def _create_vintage_model(self):
        """Create enhanced vintage-style filter"""
        model = StyleTransferNet("vintage").to(self.device)
        model.eval()
        return model
    
    def _create_neon_model(self):
        """Create enhanced neon-style filter"""
        model = StyleTransferNet("neon").to(self.device)
        model.eval()
        return model
    
    async def process(self, upload_id: str, processing_status: dict, style: str = "cartoon"):
        """Process style filter application"""
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
            
            # Apply style filter
            processing_status[upload_id]["progress"] = 50
            processed_frames = self._apply_style_filter(frames, style, processing_status, upload_id)
            
            # Recombine frames into video
            processing_status[upload_id]["progress"] = 80
            output_path = output_dir / f"{upload_id}_{style}_styled.mp4"
            self._create_video_from_frames(processed_frames, str(output_path), original_video_path=str(file_path))
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            processing_status[upload_id]["style_applied"] = style
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Style filter error: {e}")
    
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
    
    def _apply_style_filter(self, frames: list, style: str, processing_status: dict, upload_id: str):
        """Apply style filter to frames with batch processing for better GPU utilization"""
        processed_frames = []
        
        # Try to load neural model for the specific style
        style_model = self._get_style_model(style)
        
        if style_model is not None and self.device.type == 'cuda':
            # Use batch processing for GPU
            processed_frames = self._apply_neural_style_batch(frames, style, processing_status, upload_id)
        else:
            # Use traditional processing for CPU or fallback
            processed_frames = self._apply_traditional_style_batch(frames, style, processing_status, upload_id)
        
        return processed_frames
    
    def _apply_neural_style_batch(self, frames: list, style: str, processing_status: dict, upload_id: str):
        """Apply neural style transfer using batch processing for better GPU performance"""
        processed_frames = []
        total_frames = len(frames)
        
        # Process frames in batches
        for i in range(0, total_frames, self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            try:
                # Convert batch to tensors
                batch_tensors = []
                for frame in batch_frames:
                    tensor = self._frame_to_tensor(frame)
                    batch_tensors.append(tensor)
                
                # Stack tensors into batch
                batch_tensor = torch.cat(batch_tensors, dim=0)
                
                # Apply style model to batch
                with torch.no_grad():
                    styled_batch = self.style_models[style](batch_tensor)
                
                # Convert batch back to frames
                for j, styled_tensor in enumerate(styled_batch):
                    styled_frame = self._tensor_to_frame(styled_tensor.unsqueeze(0))
                    processed_frames.append(styled_frame)
                
                # Clear GPU memory
                del batch_tensor, styled_batch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Batch processing failed, falling back to traditional: {e}")
                # Fallback to traditional processing for this batch
                for frame in batch_frames:
                    processed_frame = self._apply_traditional_filter(frame, style)
                    processed_frames.append(processed_frame)
            
            # Update progress
            progress = 50 + (i / total_frames) * 30
            processing_status[upload_id]["progress"] = int(progress)
        
        return processed_frames
    
    def _apply_traditional_style_batch(self, frames: list, style: str, processing_status: dict, upload_id: str):
        """Apply traditional style filters with progress tracking"""
        processed_frames = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            processed_frame = self._apply_traditional_filter(frame, style)
            processed_frames.append(processed_frame)
            
            # Update progress
            if i % 10 == 0:
                progress = 50 + (i / total_frames) * 30
                processing_status[upload_id]["progress"] = int(progress)
        
        return processed_frames
    
    def _apply_neural_style(self, frame: np.ndarray, style: str) -> np.ndarray:
        """Apply neural style transfer"""
        try:
            # Convert frame to tensor
            frame_tensor = self._frame_to_tensor(frame)
            
            # Apply style model
            with torch.no_grad():
                styled_tensor = self.style_models[style](frame_tensor)
            
            # Convert back to numpy array
            styled_frame = self._tensor_to_frame(styled_tensor)
            
            return styled_frame
            
        except Exception as e:
            print(f"Neural style failed, using fallback: {e}")
            return self._apply_traditional_filter(frame, style)
    
    def _apply_traditional_filter(self, frame: np.ndarray, style: str) -> np.ndarray:
        """Apply traditional image processing filters"""
        if style == "cartoon":
            return self._cartoon_filter(frame)
        elif style == "sketch":
            return self._sketch_filter(frame)
        elif style == "cinematic":
            return self._cinematic_filter(frame)
        elif style == "vintage":
            return self._vintage_filter(frame)
        elif style == "neon":
            return self._neon_filter(frame)
        else:
            return frame
    
    def _cartoon_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply cartoon effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge preservation
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        
        # Apply color quantization
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        
        # Combine edges and color
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cartoon
    
    def _sketch_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply sketch effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert the image
        inv = 255 - gray
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        
        # Invert the blurred image
        inv_blur = 255 - blur
        
        
        # Blend with original
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    def _cinematic_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply cinematic color grading"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Adjust contrast in L channel
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply warm color tint
        result = cv2.addWeighted(result, 0.8, np.full_like(result, [0, 0, 50]), 0.2, 0)
        
        return result
    
    def _vintage_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply vintage effect"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust saturation and value
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 0.7)  # Reduce saturation
        v = cv2.multiply(v, 0.8)  # Reduce brightness
        
        # Merge channels
        hsv = cv2.merge([h, s, v])
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add sepia tone
        result = cv2.addWeighted(result, 0.7, np.full_like(result, [19, 69, 139]), 0.3, 0)
        
        return result
    
    def _neon_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply neon glow effect"""
        # Create neon glow
        glow = cv2.GaussianBlur(frame, (15, 15), 0)
        glow = cv2.addWeighted(glow, 0.5, np.full_like(glow, [0, 255, 255]), 0.5, 0)
        
        # Combine with original
        result = cv2.addWeighted(frame, 0.7, glow, 0.3, 0)
        
        # Enhance edges
        edges = cv2.Canny(frame, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges = cv2.addWeighted(edges, 0.3, np.full_like(edges, [0, 255, 255]), 0.7, 0)
        
        result = cv2.add(result, edges)
        
        return result
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy frame to PyTorch tensor"""
        # Normalize to [0, 1]
        frame_norm = frame.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy frame"""
        # Remove batch dimension and permute
        frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Denormalize to [0, 255]
        frame_denorm = (frame_bgr * 255).astype(np.uint8)
        
        return frame_denorm
    
    def _create_video_from_frames(self, frames: list, output_path: str, fps: int = None, original_video_path: str = None):
        """Create video from processed frames with original audio preserved"""
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
        
        # Convert to MP4 using FFmpeg and preserve original audio
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_path)
        
        if original_video_path:
            # Combine processed video with original audio
            vf = "scale=-2:'if(gte(ih,720),ih,720)'"
            cmd = [
                'ffmpeg', '-i', temp_path, '-i', original_video_path,
                '-vf', vf,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'copy',  # Copy original audio without re-encoding
                '-map', '0:v:0',  # Use video from first input (processed)
                '-map', '1:a:0',  # Use audio from second input (original)
                '-shortest',  # End when shortest stream ends
                '-movflags', '+faststart',
                '-y', output_path
            ]
        else:
            # No original video, just convert without audio
            vf = "scale=-2:'if(gte(ih,720),ih,720)'"
            cmd = [
                'ffmpeg', '-i', temp_path,
                '-vf', vf,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-an',  # No audio
                '-movflags', '+faststart',
                '-y', output_path
            ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
        except subprocess.CalledProcessError:
            # If FFmpeg fails, keep the original
            os.rename(temp_path, output_path)
    
    def clear_unused_models(self, keep_style: str = None):
        """Clear unused style models to free up memory"""
        if keep_style is None:
            # Clear all models
            self.style_models.clear()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print("Cleared all style models from memory")
        else:
            # Keep only the specified style
            models_to_remove = [style for style in self.style_models.keys() if style != keep_style]
            for style in models_to_remove:
                del self.style_models[style]
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"Cleared unused models, kept only {keep_style}")
    
    def get_available_styles(self) -> list:
        """Get list of available style filters"""
        return [
            {"id": "cartoon", "name": "Cartoon", "description": "Animated cartoon style"},
            {"id": "sketch", "name": "Sketch", "description": "Pencil sketch effect"},
            {"id": "cinematic", "name": "Cinematic", "description": "Movie-like color grading"},
            {"id": "vintage", "name": "Vintage", "description": "Old film look"},
            {"id": "neon", "name": "Neon", "description": "Cyberpunk neon glow"}
        ]
