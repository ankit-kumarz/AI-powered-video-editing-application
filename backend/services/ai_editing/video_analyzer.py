"""
Video analysis functionality for AI editing suggestions.
Handles frame extraction, scene detection, and visual feature analysis.
"""
import os
import subprocess
import asyncio
import concurrent.futures
import gc
from typing import List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path

from .models import VideoFeature, AnalysisConfig


class VideoAnalyzer:
    """Handles video analysis and feature extraction"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.face_cascade = None
        self._load_face_cascade()
    
    def _load_face_cascade(self):
        """Load OpenCV face detection cascade"""
        try:
            # Try to load the cascade file
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                print("Warning: Face detection cascade not found")
        except Exception as e:
            print(f"Warning: Could not load face detection: {e}")

    async def analyze_video_features(self, video_path: str) -> List[VideoFeature]:
        """Analyze video and extract features for editing suggestions - ADVANCED OPTIMIZATION"""
        try:
            # Check cache first for longer videos
            if self.config.enable_cache and await self._get_video_duration(video_path) > 300:  # 5+ minutes
                cached_features = await self._load_cached_analysis(video_path)
                if cached_features:
                    print("✅ Using cached analysis for longer video")
                    return cached_features
            
            features = []
            
            # Get video duration
            duration = await self._get_video_duration(video_path)
            if not duration:
                return features
            
            # Check if video is too long
            if duration > self.config.max_video_duration:
                print(f"Warning: Video too long ({duration}s), limiting analysis")
                duration = self.config.max_video_duration
            
            # Calculate optimal frame sampling
            frame_interval = self._calculate_optimal_frame_interval(duration)
            print(f"Analyzing video: {duration:.1f}s, sampling every {frame_interval} frames")
            
            # Extract frames with memory optimization
            frames = await self._extract_frames_optimized(video_path, frame_interval)
            
            # Parallel frame analysis for optimal performance
            print(f"Processing {len(frames)} frames with {self.config.max_workers} workers")
            
            if self.config.enable_parallel and len(frames) > self.config.parallel_batch_size:
                # Use parallel processing for larger frame sets
                features = await self._analyze_frames_parallel(frames)
            else:
                # Fallback to sequential processing for small frame sets
                features = await self._analyze_frames_sequential(frames)
            
            # Analyze audio features (lightweight)
            audio_features = await self._analyze_audio_features_optimized(video_path)
            features.extend(audio_features)
            
            # Detect scene changes (lightweight)
            scene_changes = await self._detect_scene_changes_optimized(frames)
            features.extend(scene_changes)
            
            # Final garbage collection
            gc.collect()
            
            print(f"Analysis complete: {len(features)} features found")
            
            # Cache results for longer videos
            if self.config.enable_cache and duration > 300:  # 5+ minutes
                await self._save_cached_analysis(video_path, features)
            
            return features
            
        except Exception as e:
            print(f"Video analysis error: {e}")
            # Force cleanup on error
            gc.collect()
            raise RuntimeError(f"Video feature analysis failed: {e}")

    def _calculate_optimal_frame_interval(self, duration: float) -> int:
        """Calculate optimal frame interval based on video duration - ADVANCED OPTIMIZATION"""
        if duration <= 60:  # 1 minute or less
            return max(1, int(duration / 20))  # 20 samples max
        elif duration <= 300:  # 5 minutes or less
            return max(1, int(duration / 40))  # 40 samples max
        elif duration <= 600:  # 10 minutes or less
            return max(1, int(duration / 60))  # 60 samples max
        elif duration <= 1200:  # 20 minutes or less
            return max(2, int(duration / 80))  # 80 samples max
        elif duration <= 1800:  # 30 minutes or less
            return max(3, int(duration / 100))  # 100 samples max
        else:  # Very long videos
            return max(5, int(duration / 150))  # 150 samples max

    async def _extract_frames_optimized(self, video_path: str, interval: int) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video with memory optimization - ENHANCED FOR FULL VIDEO"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {total_frames} frames, {fps} fps, duration: {total_frames/fps:.1f}s")
            
            # Advanced frame extraction optimization for longer videos
            if fps > 0:
                video_duration = total_frames / fps
                
                # Dynamic sampling based on video length
                if video_duration <= 300:  # 5 minutes or less
                    sample_interval = 2  # Sample every 2 seconds
                elif video_duration <= 600:  # 10 minutes or less
                    sample_interval = 3  # Sample every 3 seconds
                elif video_duration <= 1200:  # 20 minutes or less
                    sample_interval = 5  # Sample every 5 seconds
                else:  # 30+ minutes
                    sample_interval = 8  # Sample every 8 seconds
                
                # Calculate frames needed for optimal coverage
                frames_needed = min(self.config.max_frames_to_analyze, int(video_duration / sample_interval))
                interval = max(1, total_frames // frames_needed) if frames_needed > 0 else 1
                
                print(f"Optimization: {sample_interval}s intervals, {frames_needed} samples, {interval} frame skip")
            else:
                interval = max(1, interval)
            
            print(f"Extraction settings: interval={interval}, max_frames={self.config.max_frames_to_analyze}")
            
            frame_count = 0
            extracted_count = 0
            
            while frame_count < total_frames and extracted_count < self.config.max_frames_to_analyze:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    
                    # Resize frame to save memory
                    height, width = frame.shape[:2]
                    new_width = int(width * self.config.frame_resize_factor)
                    new_height = int(height * self.config.frame_resize_factor)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                    
                    frames.append((timestamp, frame_resized))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"Extracted {len(frames)} frames from {frame_count} total frames")
            
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        return frames

    async def _analyze_frame_optimized(self, frame: np.ndarray, timestamp: float) -> List[VideoFeature]:
        """Analyze a single frame with advanced professional editing features"""
        features = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. ADVANCED FACE DETECTION - Character analysis
            if self.face_cascade:
                try:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        # Analyze face positions and sizes for shot composition
                        face_sizes = []
                        face_positions = []
                        
                        for (x, y, w, h) in faces:
                            face_sizes.append(w * h)
                            face_positions.append((x + w/2, y + h/2))  # Center point
                        
                        # Determine shot type based on face analysis
                        avg_face_size = sum(face_sizes) / len(face_sizes)
                        frame_area = frame.shape[0] * frame.shape[1]
                        face_ratio = avg_face_size / frame_area
                        
                        shot_type = 'wide_shot'
                        if face_ratio > 0.1:
                            shot_type = 'close_up'
                        elif face_ratio > 0.05:
                            shot_type = 'medium_shot'
                        
                        features.append(VideoFeature(
                            timestamp=timestamp,
                            feature_type='face_detected',
                            confidence=min(1.0, len(faces) * 0.3),
                            metadata={
                                'face_count': len(faces),
                                'shot_type': shot_type,
                                'face_ratio': face_ratio,
                                'face_positions': face_positions
                            }
                        ))
                except Exception as e:
                    print(f"Face detection failed: {e}")
            
            # 2. MOTION DETECTION - Action analysis
            try:
                # Simple motion detection using frame difference
                motion_score = self._detect_motion_lightweight(frame)
                if motion_score > self.config.motion_threshold:
                    features.append(VideoFeature(
                        timestamp=timestamp,
                        feature_type='motion',
                        confidence=motion_score,
                        metadata={'motion_score': motion_score}
                    ))
            except Exception as e:
                print(f"Motion detection failed: {e}")
            
            # 3. COMPOSITION ANALYSIS - Visual quality
            try:
                composition_score = self._analyze_composition_lightweight(frame)
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='composition',
                    confidence=composition_score,
                    metadata={'composition_score': composition_score}
                ))
            except Exception as e:
                print(f"Composition analysis failed: {e}")
            
        except Exception as e:
            print(f"Frame analysis failed at {timestamp}s: {e}")
        
        return features

    def _detect_motion_lightweight(self, frame: np.ndarray) -> float:
        """Lightweight motion detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate variance as motion indicator
            motion_score = np.var(gray) / 10000.0  # Normalize
            
            return min(1.0, motion_score)
        except Exception:
            return 0.0

    def _analyze_composition_lightweight(self, frame: np.ndarray) -> float:
        """Lightweight frame composition analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple brightness and contrast analysis instead of edge detection
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Normalize to 0-1 range
            brightness_score = mean_brightness / 255.0
            contrast_score = min(1.0, std_brightness / 50.0)
            
            # Combined score
            return (brightness_score + contrast_score) / 2.0
            
        except Exception:
            return 0.5

    async def _analyze_audio_features_optimized(self, video_path: str) -> List[VideoFeature]:
        """Lightweight audio analysis"""
        features = []
        
        try:
            # Use ffprobe for quick audio analysis instead of extracting audio
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Simple audio feature
                features.append(VideoFeature(
                    timestamp=0.0,
                    feature_type='audio_detected',
                    confidence=1.0,
                    metadata={'audio_info': 'Audio stream detected'}
                ))
                
        except Exception as e:
            print(f"Audio analysis failed: {e}")
        
        return features

    async def _detect_scene_changes_optimized(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Lightweight scene change detection - ENHANCED FOR FULL VIDEO"""
        scene_changes = []
        
        try:
            # Analyze all frame pairs for scene changes
            for i in range(1, len(frames)):
                prev_frame = frames[i-1][1]
                curr_frame = frames[i][1]
                timestamp = frames[i][0]
                
                # Simple frame difference (much faster than Canny)
                diff = cv2.absdiff(prev_frame, curr_frame)
                change_score = np.mean(diff) / 255.0
                
                if change_score > self.config.scene_change_threshold:
                    scene_changes.append(VideoFeature(
                        timestamp=timestamp,
                        feature_type='scene_change',
                        confidence=min(1.0, change_score),
                        metadata={'change_score': change_score}
                    ))
                    
        except Exception as e:
            print(f"Scene change detection failed: {e}")
        
        return scene_changes

    async def _analyze_frames_parallel(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Analyze frames using parallel processing for optimal performance"""
        features = []
        
        try:
            # Split frames into parallel batches
            batches = [frames[i:i + self.config.parallel_batch_size] 
                      for i in range(0, len(frames), self.config.parallel_batch_size)]
            
            print(f"Processing {len(batches)} batches in parallel with {self.config.max_workers} workers")
            
            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit batch processing tasks
                future_to_batch = {
                    executor.submit(self._process_frame_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        batch_features = future.result()
                        features.extend(batch_features)
                        completed += 1
                        
                        # Progress update
                        progress = (completed / len(batches)) * 100
                        print(f"Parallel progress: {progress:.1f}% ({completed}/{len(batches)} batches)")
                        
                        # Memory cleanup every few batches
                        if completed % self.config.gc_interval == 0:
                            gc.collect()
                            print(f"Memory cleanup at {progress:.1f}%")
                            
                    except Exception as e:
                        print(f"Batch {batch_index} failed: {e}")
                        continue
            
            print(f"Parallel processing complete: {len(features)} features extracted")
            
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            features = await self._analyze_frames_sequential(frames)
        
        return features

    def _process_frame_batch(self, batch: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Process a batch of frames (for parallel execution)"""
        batch_features = []
        
        for timestamp, frame in batch:
            try:
                # Analyze frame (synchronous version for threading)
                frame_features = self._analyze_frame_sync(frame, timestamp)
                batch_features.extend(frame_features)
            except Exception as e:
                print(f"Frame analysis failed at {timestamp}s: {e}")
                continue
        
        return batch_features

    def _analyze_frame_sync(self, frame: np.ndarray, timestamp: float) -> List[VideoFeature]:
        """Synchronous version of frame analysis for parallel processing"""
        features = []
        
        try:
            # Face detection
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    features.append(VideoFeature(
                        timestamp=timestamp,
                        feature_type='face_detected',
                        confidence=min(1.0, len(faces) * 0.3),
                        metadata={'face_count': len(faces)}
                    ))
            
            # Motion detection
            motion_score = self._detect_motion_lightweight(frame)
            if motion_score > self.config.motion_threshold:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='motion',
                    confidence=motion_score,
                    metadata={'motion_score': motion_score}
                ))
            
            # Composition analysis
            composition_score = self._analyze_composition_lightweight(frame)
            features.append(VideoFeature(
                timestamp=timestamp,
                feature_type='composition',
                confidence=composition_score,
                metadata={'composition_score': composition_score}
            ))
            
        except Exception as e:
            print(f"Sync frame analysis failed at {timestamp}s: {e}")
        
        return features

    async def _analyze_frames_sequential(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Sequential frame analysis (fallback method)"""
        features = []
        
        try:
            for i, (timestamp, frame) in enumerate(frames):
                try:
                    # Analyze frame
                    frame_features = await self._analyze_frame_optimized(frame, timestamp)
                    features.extend(frame_features)
                    
                    # Progress update
                    if i % 10 == 0:
                        progress = (i / len(frames)) * 100
                        print(f"Sequential progress: {progress:.1f}% ({i}/{len(frames)} frames)")
                    
                    # Memory cleanup
                    if i % self.config.gc_interval == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"Frame analysis failed at {timestamp}s: {e}")
                    continue
            
            print(f"Sequential processing complete: {len(features)} features extracted")
            
        except Exception as e:
            print(f"Sequential processing failed: {e}")
        
        return features

    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"Duration detection failed: {e}")
        return None

    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file for caching"""
        try:
            import hashlib
            with open(video_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(hash(video_path))

    async def _load_cached_analysis(self, video_path: str) -> Optional[List[VideoFeature]]:
        """Load cached analysis results"""
        try:
            if not self.config.enable_cache:
                return None
            
            video_hash = self._get_video_hash(video_path)
            cache_file = Path(self.config.cache_dir) / f"{video_hash}.pkl"
            
            if cache_file.exists():
                import pickle
                import time
                
                # Check cache age
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age > self.config.cache_ttl:
                    cache_file.unlink()  # Remove expired cache
                    return None
                
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Cache load failed: {e}")
        return None

    async def _save_cached_analysis(self, video_path: str, features: List[VideoFeature]):
        """Save analysis results to cache"""
        try:
            if not self.config.enable_cache:
                return
            
            video_hash = self._get_video_hash(video_path)
            cache_file = Path(self.config.cache_dir) / f"{video_hash}.pkl"
            
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            
            print(f"Analysis cached: {cache_file}")
        except Exception as e:
            print(f"Cache save failed: {e}")
