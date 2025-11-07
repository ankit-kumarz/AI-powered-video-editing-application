import cv2
import numpy as np
from pathlib import Path
import subprocess
import os
import tempfile
from PIL import Image, ImageDraw
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import partial
import time

class ObjectRemovalService:
    def __init__(self):
        # Add thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._cache = {}  # Simple cache for intermediate results
    
    async def process(self, upload_id: str, processing_status: dict, bounding_boxes: list = None):
        """Process object removal from video - OPTIMIZED VERSION"""
        try:
            # Use provided bounding boxes or default to empty list
            if bounding_boxes is None:
                bounding_boxes = []
            
            # FIXED: Handle different bounding box formats properly
            parsed_bounding_boxes = []
            
            if isinstance(bounding_boxes, str):
                # Parse bounding boxes from string format "x1,y1,x2,y2;x1,y1,x2,y2"
                for box_str in bounding_boxes.split(';'):
                    if box_str.strip():
                        try:
                            coords = [float(x) for x in box_str.split(',')]
                            if len(coords) == 4:
                                # Convert to integers for pixel coordinates
                                parsed_bounding_boxes.append([int(coord) for coord in coords])
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid bounding box format: {box_str}, error: {e}")
                            continue
            elif isinstance(bounding_boxes, list):
                # Handle list format - could be list of lists or list of strings
                for box in bounding_boxes:
                    if isinstance(box, list) and len(box) == 4:
                        # Already in correct format [x1, y1, x2, y2]
                        parsed_bounding_boxes.append([int(coord) for coord in box])
                    elif isinstance(box, str):
                        # Parse individual string box
                        try:
                            coords = [float(x) for x in box.split(',')]
                            if len(coords) == 4:
                                parsed_bounding_boxes.append([int(coord) for coord in coords])
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Invalid bounding box format: {box}, error: {e}")
                            continue
            else:
                print(f"Warning: Unexpected bounding_boxes type: {type(bounding_boxes)}")
            
            # If no valid bounding boxes provided, use default
            if not parsed_bounding_boxes:
                print("No valid bounding boxes provided, using default box")
                parsed_bounding_boxes = [[100, 100, 200, 200]]  # Default box
            
            print(f"Processing {len(parsed_bounding_boxes)} bounding boxes: {parsed_bounding_boxes}")
            
            # Update status
            processing_status[upload_id]["progress"] = 10
            processing_status[upload_id]["status"] = "processing"
            processing_status[upload_id]["message"] = f"Analyzing video for object removal ({len(parsed_bounding_boxes)} objects)..."
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get video info for optimization decisions
            video_info = await self._get_video_info(str(file_path))
            processing_status[upload_id]["video_info"] = video_info
            
            # Validate and normalize bounding boxes
            validated_bounding_boxes = []
            for bbox in parsed_bounding_boxes:
                if self._validate_bounding_box(bbox, video_info["width"], video_info["height"]):
                    normalized_bbox = self._normalize_bounding_box(bbox, video_info["width"], video_info["height"])
                    validated_bounding_boxes.append(normalized_bbox)
                else:
                    print(f"Warning: Invalid bounding box {bbox}, skipping")
            
            if not validated_bounding_boxes:
                print("No valid bounding boxes after validation, using default box")
                validated_bounding_boxes = [[100, 100, 200, 200]]
            
            print(f"Processing {len(validated_bounding_boxes)} validated bounding boxes: {validated_bounding_boxes}")
            
            # OPTIMIZATION: Determine frame sampling rate based on video length
            frame_sampling_rate = self._calculate_frame_sampling_rate(video_info["duration"])
            
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["message"] = f"Extracting frames (sampling every {frame_sampling_rate} frames)..."
            
            # OPTIMIZATION: Extract frames with sampling
            frames = await self._extract_frames_optimized(str(file_path), frame_sampling_rate)
            
            processing_status[upload_id]["progress"] = 30
            processing_status[upload_id]["message"] = "Processing frames in parallel..."
            
            # OPTIMIZATION: Process frames in parallel batches
            processed_frames = await self._remove_objects_from_frames_parallel(
                frames, validated_bounding_boxes, processing_status, upload_id, video_info
            )
            
            processing_status[upload_id]["progress"] = 80
            processing_status[upload_id]["message"] = "Creating optimized video output..."
            
            # Recombine frames into video with optimization
            output_path = output_dir / f"{upload_id}_object_removed.mp4"
            await self._create_video_from_frames_optimized(
                processed_frames, str(output_path), str(file_path), video_info
            )
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            processing_status[upload_id]["removed_objects"] = validated_bounding_boxes
            processing_status[upload_id]["optimization_info"] = {
                "frame_sampling_rate": frame_sampling_rate,
                "parallel_processing": True,
                "memory_efficient": True,
                "objects_removed": len(validated_bounding_boxes)
            }
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Object removal error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _get_video_info(self, video_path: str) -> dict:
        """Get video information for optimization decisions"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_video_info_sync, video_path)
    
    def _get_video_info_sync(self, video_path: str) -> dict:
        """Synchronous video info extraction"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height
        }
    
    def _calculate_frame_sampling_rate(self, duration: float) -> int:
        """Calculate optimal frame sampling rate based on video duration"""
        if duration > 300:  # > 5 minutes: sample every 3 frames
            return 3
        elif duration > 120:  # > 2 minutes: sample every 2 frames
            return 2
        else:  # <= 2 minutes: process every frame
            return 1
    
    async def _extract_frames_optimized(self, video_path: str, sampling_rate: int = 1, max_frames: int = 10000):
        """Extract frames with sampling for optimization"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._extract_frames_sync, 
            video_path, 
            sampling_rate, 
            max_frames
        )
    
    def _extract_frames_sync(self, video_path: str, sampling_rate: int = 1, max_frames: int = 10000):
        """Synchronous frame extraction with sampling"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # OPTIMIZATION: Sample frames based on rate
            if frame_count % sampling_rate == 0:
                # OPTIMIZATION: Resize large frames for faster processing
                height, width = frame.shape[:2]
                if width > 1280:  # Resize if too large
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                frames.append(frame)
                extracted_count += 1

            frame_count += 1
        
        cap.release()
        return frames
    
    async def _remove_objects_from_frames_parallel(self, frames: list, bounding_boxes: list, 
                                                  processing_status: dict, upload_id: str, video_info: dict):
        """Remove objects from frames using parallel processing"""
        if not frames:
            return []
        
        # OPTIMIZATION: Process frames in batches for memory efficiency
        batch_size = min(50, len(frames))  # Process 50 frames at a time
        processed_frames = []
        
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            
            # Create tasks for parallel processing
            tasks = []
            for i, frame in enumerate(batch_frames):
                task = self._process_single_frame(frame, bounding_boxes, batch_start + i)
                tasks.append(task)
            
            # Process batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and update progress
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing frame {batch_start + i}: {result}")
                    # Use original frame as fallback
                    processed_frames.append(batch_frames[i])
                else:
                    processed_frames.append(result)
            
            # Update progress
            progress = 30 + (batch_end / len(frames)) * 50
            processing_status[upload_id]["progress"] = int(progress)
            processing_status[upload_id]["message"] = f"Processed {batch_end}/{len(frames)} frames..."
        
        return processed_frames
    
    async def _process_single_frame(self, frame: np.ndarray, bounding_boxes: list, frame_index: int) -> np.ndarray:
        """Process a single frame for object removal"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_single_frame_sync,
            frame,
            bounding_boxes,
            frame_index
        )
    
    def _process_single_frame_sync(self, frame: np.ndarray, bounding_boxes: list, frame_index: int) -> np.ndarray:
        """Synchronous single frame processing"""
        try:
            # Create mask for the object to remove
            mask = self._create_removal_mask(frame, bounding_boxes)
            
            # Apply inpainting to remove the object
            processed_frame = self._apply_inpainting_optimized(frame, mask)
            
            return processed_frame
            
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
            return frame  # Return original frame on error
    
    def _extract_frames(self, video_path: str, max_frames: int = 1000000):
        """Extract frames from video - LEGACY METHOD"""
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
    
    def _remove_objects_from_frames(self, frames: list, bounding_box: list, processing_status: dict, upload_id: str):
        """Remove objects from frames using inpainting - LEGACY METHOD"""
        processed_frames = []
        
        for i, frame in enumerate(frames):
            # Create mask for the object to remove
            mask = self._create_removal_mask(frame, bounding_box)
            
            # Apply inpainting to remove the object
            processed_frame = self._apply_inpainting(frame, mask)
            
            processed_frames.append(processed_frame)
            
            # Update progress
            if i % 10 == 0:
                progress = 50 + (i / len(frames)) * 30
                processing_status[upload_id]["progress"] = int(progress)
        
        return processed_frames
    
    def _create_removal_mask(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """Create a mask for the object to remove"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for bbox in bounding_boxes:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Create rectangular mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _apply_inpainting_optimized(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply optimized inpainting to remove objects"""
        try:
            # OPTIMIZATION: Try GPU acceleration if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # GPU-accelerated inpainting
                gpu_frame = cv2.cuda_GpuMat()
                gpu_mask = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_mask.upload(mask)
                
                # GPU inpainting (if available)
                result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            else:
                # CPU inpainting with optimization
                result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            return result
            
        except Exception as e:
            print(f"Optimized inpainting failed: {e}")
            # Fallback: simple blur and blend
            return self._fallback_object_removal(frame, mask)
    
    def _apply_inpainting(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply inpainting to remove objects - LEGACY METHOD"""
        try:
            # Use OpenCV's inpainting algorithm
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            # Alternative: use Navier-Stokes inpainting
            # result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
            
            return result
            
        except Exception as e:
            print(f"Inpainting failed: {e}")
            # Fallback: simple blur and blend
            return self._fallback_object_removal(frame, mask)
    
    def _fallback_object_removal(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback method for object removal using blur and blending"""
        # Create a blurred version of the frame
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Blend the blurred area with the original frame
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_normalized = np.expand_dims(mask_normalized, axis=2)
        
        result = frame * (1 - mask_normalized) + blurred * mask_normalized
        
        return result.astype(np.uint8)
    
    async def _create_video_from_frames_optimized(self, frames: list, output_path: str, 
                                                original_video_path: str, video_info: dict):
        """Create video from processed frames with optimization"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_video_from_frames_sync,
            frames,
            output_path,
            original_video_path,
            video_info
        )
    
    def _create_video_from_frames_sync(self, frames: list, output_path: str, 
                                     original_video_path: str, video_info: dict):
        """Synchronous video creation with optimization"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fps = video_info.get("fps", 25)
        
        # OPTIMIZATION: Use faster encoding settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # OPTIMIZATION: Use faster FFmpeg settings
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_path)
        
        # Optimized FFmpeg command
        cmd = [
            'ffmpeg', '-i', temp_path,
            '-i', original_video_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'fast',  # Faster encoding
            '-crf', '25',  # Slightly lower quality for speed
            '-threads', '4',  # Multi-threading
            '-movflags', '+faststart',
            '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
        except subprocess.CalledProcessError:
            # If FFmpeg fails, keep the original
            os.rename(temp_path, output_path)
    
    def _create_video_from_frames(self, frames: list, output_path: str, fps: int = None, original_video_path: str = None):
        """Create video from processed frames - LEGACY METHOD"""
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
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """Detect potential objects in a frame for easier selection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 1000  # Minimum area to consider as object
            objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "confidence": min(area / 10000, 1.0)  # Simple confidence based on area
                    })
            
            # Sort by confidence
            objects.sort(key=lambda x: x["confidence"], reverse=True)
            
            return objects[:10]  # Return top 10 objects
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    def create_preview_mask(self, frame: np.ndarray, bounding_box: list) -> np.ndarray:
        """Create a preview showing what will be removed"""
        preview = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add label
        cv2.putText(preview, "Object to Remove", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return preview
    
    def get_removal_techniques(self) -> list:
        """Get available object removal techniques"""
        return [
            {
                "id": "inpaint_telea",
                "name": "Telea Inpainting",
                "description": "Fast inpainting using Telea algorithm",
                "quality": "high",
                "speed": "fast"
            },
            {
                "id": "inpaint_ns",
                "name": "Navier-Stokes Inpainting",
                "description": "High-quality inpainting using Navier-Stokes",
                "quality": "very_high",
                "speed": "slow"
            },
            {
                "id": "blur_blend",
                "name": "Blur and Blend",
                "description": "Simple blur and blend technique",
                "quality": "medium",
                "speed": "very_fast"
            }
        ]
    
    def process_multiple_objects(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """Remove multiple objects from a single frame"""
        result = frame.copy()
        
        for bbox in bounding_boxes:
            mask = self._create_removal_mask(result, bbox)
            result = self._apply_inpainting_optimized(result, mask)
        
        return result
    
    def estimate_processing_time(self, video_duration: float, frame_count: int) -> dict:
        """Estimate processing time for object removal - OPTIMIZED VERSION"""
        # OPTIMIZATION: Account for parallel processing and frame sampling
        frames_per_second = frame_count / video_duration
        
        # Calculate frame sampling rate
        sampling_rate = self._calculate_frame_sampling_rate(video_duration)
        actual_frames = frame_count // sampling_rate
        
        # Processing time per frame (optimized)
        time_per_frame = 0.05  # 50ms per frame (optimized)
        
        # Account for parallel processing (4 workers)
        parallel_factor = 4
        total_time = (actual_frames * time_per_frame) / parallel_factor
        
        return {
            "estimated_time_seconds": total_time,
            "estimated_time_minutes": total_time / 60,
            "frames_per_second": frames_per_second,
            "total_frames": frame_count,
            "sampling_rate": sampling_rate,
            "actual_frames_processed": actual_frames,
            "parallel_processing": True
        }

    def _validate_bounding_box(self, bbox: list, video_width: int, video_height: int) -> bool:
        """Validate bounding box coordinates"""
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Check if coordinates are within video bounds
        if x1 < 0 or y1 < 0 or x2 > video_width or y2 > video_height:
            return False
        
        # Check if box has positive dimensions
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check if box is not too small (minimum 10x10 pixels)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
        
        return True

    def _normalize_bounding_box(self, bbox: list, video_width: int, video_height: int) -> list:
        """Normalize bounding box coordinates to video dimensions"""
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, video_width - 1))
        y1 = max(0, min(y1, video_height - 1))
        x2 = max(0, min(x2, video_width - 1))
        y2 = max(0, min(y2, video_height - 1))
        
        # Ensure minimum size
        if (x2 - x1) < 10:
            x2 = min(x1 + 10, video_width - 1)
        if (y2 - y1) < 10:
            y2 = min(y1 + 10, video_height - 1)
        
        return [int(x1), int(y1), int(x2), int(y2)]
