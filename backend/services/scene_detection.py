import cv2
import numpy as np
from pathlib import Path
import subprocess
import os
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager

class SceneDetectionService:
    def __init__(self):
        pass
    
    async def process(self, upload_id: str, processing_status: dict, threshold: float = 30.0):
        """Process scene detection and split video into clips"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect scenes
            processing_status[upload_id]["progress"] = 40
            scenes = self._detect_scenes(str(file_path), threshold)
            
            # Split video into clips
            processing_status[upload_id]["progress"] = 60
            clip_paths = self._split_video_into_clips(str(file_path), scenes, upload_id, output_dir)
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["clips"] = clip_paths
            processing_status[upload_id]["scene_count"] = len(scenes)
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Scene detection error: {e}")
    
    def _detect_scenes(self, video_path: str, threshold: float) -> list:
        """Detect scene changes using PySceneDetect"""
        try:
            # Use ContentDetector for content-based scene detection
            scenes = detect(video_path, ContentDetector(threshold=threshold))
            
            # Convert to list of (start, end) tuples
            scene_list = []
            for i, scene in enumerate(scenes):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                scene_list.append((start_time, end_time))
            
            return scene_list
            
        except Exception as e:
            print(f"Scene detection failed, using fallback method: {e}")
            # Fallback: use OpenCV for basic scene detection
            return self._fallback_scene_detection(video_path, threshold)
    
    def _fallback_scene_detection(self, video_path: str, threshold: float) -> list:
        """Fallback scene detection using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        scenes = []
        prev_frame = None
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate difference between frames
                diff = cv2.absdiff(prev_frame, frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                mean_diff = np.mean(gray_diff)
                
                # If difference is above threshold, mark as scene change
                if mean_diff > threshold:
                    scene_start = frame_count / fps
                    if scenes and scenes[-1][1] == -1:
                        scenes[-1] = (scenes[-1][0], scene_start)
                    else:
                        scenes.append((scene_start, -1))
            
            prev_frame = frame.copy()
            frame_count += 1
        
        cap.release()
        
        # Close any open scenes
        if scenes and scenes[-1][1] == -1:
            scenes[-1] = (scenes[-1][0], frame_count / fps)
        
        # Add start scene if no scenes detected
        if not scenes:
            scenes = [(0, frame_count / fps)]
        
        return scenes
    
    def _split_video_into_clips(self, video_path: str, scenes: list, upload_id: str, output_dir: Path) -> list:
        """Split video into separate clip files"""
        clip_paths = []
        
        for i, (start_time, end_time) in enumerate(scenes):
            # Format timestamps for FFmpeg
            start_str = self._format_timestamp(start_time)
            duration = end_time - start_time
            
            # Generate output filename
            output_filename = f"{upload_id}_clip_{i+1:03d}.mp4"
            output_path = output_dir / output_filename
            
            # Use FFmpeg to extract clip
            cmd = [
                'ffmpeg', '-ss', start_str, '-t', str(duration),
                '-i', video_path,
                '-vf', "scale=-2:'if(gte(ih,720),ih,720)'",
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'medium',
                '-crf', '23',
                '-avoid_negative_ts', 'make_zero',
                '-y', str(output_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                clip_paths.append(str(output_path))
            except subprocess.CalledProcessError as e:
                print(f"Failed to create clip {i+1}: {e}")
                continue
        
        return clip_paths
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format for FFmpeg"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_scene_info(self, video_path: str) -> dict:
        """Get detailed information about detected scenes"""
        try:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            # Detect scenes
            scenes = self._detect_scenes(video_path, 30.0)
            
            # Calculate scene statistics
            scene_stats = []
            for i, (start, end) in enumerate(scenes):
                scene_duration = end - start
                scene_stats.append({
                    "scene_number": i + 1,
                    "start_time": start,
                    "end_time": end,
                    "duration": scene_duration,
                    "start_frame": int(start * fps),
                    "end_frame": int(end * fps)
                })
            
            return {
                "total_duration": duration,
                "total_frames": frame_count,
                "fps": fps,
                "scene_count": len(scenes),
                "scenes": scene_stats
            }
            
        except Exception as e:
            print(f"Error getting scene info: {e}")
            return {}
    
    def create_scene_preview(self, video_path: str, output_dir: Path, max_scenes: int = 10) -> list:
        """Create preview images for detected scenes"""
        try:
            scenes = self._detect_scenes(video_path, 30.0)
            preview_paths = []
            
            # Limit number of scenes for preview
            scenes = scenes[:max_scenes]
            
            for i, (start_time, end_time) in enumerate(scenes):
                # Extract frame from middle of scene
                mid_time = (start_time + end_time) / 2
                
                # Generate preview filename
                preview_filename = f"scene_{i+1:03d}_preview.jpg"
                preview_path = output_dir / preview_filename
                
                # Use FFmpeg to extract frame
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-ss', str(mid_time),
                    '-vframes', '1',
                    '-q:v', '2',
                    '-y', str(preview_path)
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    preview_paths.append(str(preview_path))
                except subprocess.CalledProcessError:
                    continue
            
            return preview_paths
            
        except Exception as e:
            print(f"Error creating scene previews: {e}")
            return []
