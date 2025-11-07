import subprocess
import tempfile
import os
from pathlib import Path
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import partial
import logging

class VideoCompilationService:
    def __init__(self):
        self.temp_dir = Path("temp/compilation")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Add thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)  # Optimized for 5 videos
        self._lock = threading.Lock()
        # Setup logger
        self.logger = logging.getLogger(__name__)
    
    async def process(self, upload_ids: List[str], processing_status: dict, 
                     max_duration: int = 300, transition_style: str = "fade", 
                     preset: str = "youtube_shorts", apply_effects: bool = False,
                     effect_type: str = "none"):
        """Process video compilation with AI best parts detection and transitions - OPTIMIZED VERSION"""
        try:
            # Update status
            main_upload_id = upload_ids[0]  # Use first upload ID as main
            processing_status[main_upload_id]["progress"] = 10
            processing_status[main_upload_id]["status"] = "processing"
            processing_status[main_upload_id]["message"] = "Analyzing videos for best parts (parallel processing)..."
            
            # Validate input
            if len(upload_ids) > 5:
                raise ValueError("Maximum 5 videos allowed")
            
            # Get preset configuration
            preset_config = self._get_preset_config(preset)
            if preset_config:
                max_duration = preset_config["max_duration"]
                aspect_ratio = preset_config["aspect_ratio"]
                platform = preset_config["platform"]
            else:
                aspect_ratio = "16:9"
                platform = "Custom"
            
            # Get file paths
            video_paths = []
            for upload_id in upload_ids:
                if upload_id not in processing_status:
                    raise ValueError(f"Upload ID {upload_id} not found")
                video_paths.append(processing_status[upload_id]["file_path"])
            
            # PARALLEL: Analyze videos and detect best parts concurrently
            processing_status[main_upload_id]["progress"] = 20
            processing_status[main_upload_id]["message"] = f"Detecting best moments for {platform} (parallel analysis)..."
            
            # Create tasks for parallel video analysis
            analysis_tasks = []
            for i, video_path in enumerate(video_paths):
                target_duration = max_duration // len(video_paths)
                task = self._analyze_video_parallel(video_path, target_duration, i)
                analysis_tasks.append(task)
            
            # Execute all analysis tasks in parallel
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            best_clips = []
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    print(f"Error analyzing video {i}: {result}")
                    # Fallback to default clips
                    cap = cv2.VideoCapture(video_paths[i])
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    fallback_clips = self._create_default_clips(duration, max_duration // len(video_paths))
                    best_clips.extend([(video_paths[i], clip) for clip in fallback_clips])
                else:
                    best_clips.extend([(video_paths[i], clip) for clip in result])
            
            # Shuffle clips for variety
            random.shuffle(best_clips)
            
            # Create compilation with transitions
            processing_status[main_upload_id]["progress"] = 40
            processing_status[main_upload_id]["message"] = f"Creating {platform} compilation with {transition_style} transitions..."
            
            output_path = await self._create_compilation_parallel(best_clips, transition_style, main_upload_id, aspect_ratio)
            
            # Optimize for specific platform
            processing_status[main_upload_id]["progress"] = 80
            processing_status[main_upload_id]["message"] = f"Optimizing for {platform}..."
            
            final_output = await self._optimize_for_platform_parallel(output_path, main_upload_id, preset_config)
            
            # Apply post-compilation effects if requested
            if apply_effects and effect_type != "none":
                processing_status[main_upload_id]["progress"] = 90
                processing_status[main_upload_id]["message"] = f"Applying {effect_type} effect to compilation..."
                
                final_output = await self._apply_post_compilation_effects_parallel(
                    final_output, main_upload_id, effect_type
                )
            
            # Update status
            processing_status[main_upload_id]["progress"] = 100
            processing_status[main_upload_id]["status"] = "completed"
            processing_status[main_upload_id]["output_path"] = str(final_output)
            processing_status[main_upload_id]["message"] = f"{platform} compilation completed successfully! (Optimized processing)"
            processing_status[main_upload_id]["clips_used"] = len(best_clips)
            processing_status[main_upload_id]["total_duration"] = self._get_video_duration(str(final_output))
            processing_status[main_upload_id]["platform"] = platform
            processing_status[main_upload_id]["aspect_ratio"] = aspect_ratio
            
        except Exception as e:
            processing_status[main_upload_id]["status"] = "error"
            processing_status[main_upload_id]["error"] = str(e)
            print(f"Video compilation error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _analyze_video_parallel(self, video_path: str, target_duration: int, video_index: int) -> List[Dict]:
        """Analyze video for best parts using parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._detect_best_parts, 
            video_path, 
            target_duration
        )
    
    async def _create_compilation_parallel(self, best_clips: List[Tuple[str, Dict]], 
                                         transition_style: str, upload_id: str, aspect_ratio: str) -> str:
        """Create compilation video with transitions using parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._create_compilation,
            best_clips,
            transition_style,
            upload_id,
            aspect_ratio
        )
    
    async def _optimize_for_platform_parallel(self, input_path: str, upload_id: str, preset_config: Dict) -> str:
        """Optimize video for specific platform using parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._optimize_for_platform,
            input_path,
            upload_id,
            preset_config
        )
    
    async def _apply_post_compilation_effects_parallel(self, input_path: str, upload_id: str, effect_type: str) -> str:
        """Apply post-compilation effects using parallel processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._apply_post_compilation_effects,
            input_path,
            upload_id,
            effect_type
        )

    def _get_preset_config(self, preset_id: str) -> Dict:
        """Get configuration for a specific preset"""
        presets = self.get_compilation_presets()
        for preset in presets:
            if preset["id"] == preset_id:
                return preset
        return None

    def _detect_best_parts(self, video_path: str, target_duration: int) -> List[Dict]:
        """Detect best parts of a video using motion, audio, and scene analysis - OPTIMIZED VERSION"""
        clips = []
        
        try:
            # Get video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # OPTIMIZATION: Adaptive frame interval based on video length
            if duration > 300:  # Long videos: analyze every 3 seconds
                frame_interval = max(1, int(fps * 3))
            elif duration > 60:  # Medium videos: analyze every 2 seconds
                frame_interval = max(1, int(fps * 2))
            else:  # Short videos: analyze every 1 second
                frame_interval = max(1, int(fps * 1))
            
            motion_scores = []
            frame_times = []
            
            # OPTIMIZATION: Pre-allocate arrays for better performance
            estimated_samples = int(duration / (frame_interval / fps)) + 1
            motion_scores = np.zeros(estimated_samples)
            frame_times = np.zeros(estimated_samples)
            sample_count = 0
            
            # Analyze motion and visual interest
            prev_frame = None
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # OPTIMIZATION: Resize frame for faster processing
                    height, width = frame.shape[:2]
                    if width > 640:  # Resize if too large
                        scale = 640 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to grayscale for motion detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate motion score
                        diff = cv2.absdiff(prev_frame, gray)
                        motion_score = np.mean(diff)
                        
                        # Calculate visual interest (edge density, contrast)
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                        
                        # Combined score
                        combined_score = motion_score * 0.7 + edge_density * 0.3
                        
                        if sample_count < len(motion_scores):
                            motion_scores[sample_count] = combined_score
                            frame_times[sample_count] = frame_count / fps
                            sample_count += 1
                    
                    prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            
            # Trim arrays to actual size
            motion_scores = motion_scores[:sample_count]
            frame_times = frame_times[:sample_count]
            
            # Find peaks (best moments)
            if len(motion_scores) > 0:
                # Normalize scores
                scores = np.array(motion_scores)
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                
                # Find local maxima (peaks)
                peaks = self._find_peaks(scores, distance=5)  # Minimum 5 samples apart
                
                # Create clips around peaks
                for peak_idx in peaks:
                    if len(clips) >= 3:  # Max 3 clips per video
                        break
                    
                    peak_time = frame_times[peak_idx]
                    clip_start = max(0, peak_time - 3)  # 3 seconds before peak
                    clip_end = min(duration, peak_time + 3)  # 3 seconds after peak
                    
                    clips.append({
                        "start_time": clip_start,
                        "end_time": clip_end,
                        "duration": clip_end - clip_start,
                        "score": scores[peak_idx]
                    })
            
            # If no peaks found, create clips from beginning, middle, and end
            if len(clips) == 0:
                clips = self._create_default_clips(duration, target_duration)
            
            return clips
            
        except Exception as e:
            print(f"Error detecting best parts: {e}")
            # Fallback to default clips
            cap = cv2.VideoCapture(video_path)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return self._create_default_clips(duration, target_duration)
    
    def _find_peaks(self, scores: np.ndarray, distance: int) -> List[int]:
        """Find local maxima in scores with minimum distance"""
        peaks = []
        for i in range(1, len(scores) - 1):
            if (scores[i] > scores[i-1] and scores[i] > scores[i+1] and 
                scores[i] > 0.5):  # Only significant peaks
                # Check distance from previous peak
                if not peaks or i - peaks[-1] >= distance:
                    peaks.append(i)
        return peaks
    
    def _create_default_clips(self, duration: float, target_duration: int) -> List[Dict]:
        """Create default clips from beginning, middle, and end"""
        clips = []
        clip_duration = min(5.0, target_duration / 3)  # 5 seconds or target/3
        
        # Beginning clip
        clips.append({
            "start_time": 0,
            "end_time": min(clip_duration, duration),
            "duration": min(clip_duration, duration),
            "score": 0.8
        })
        
        # Middle clip
        if duration > clip_duration * 2:
            mid_start = (duration - clip_duration) / 2
            clips.append({
                "start_time": mid_start,
                "end_time": mid_start + clip_duration,
                "duration": clip_duration,
                "score": 0.7
            })
        
        # End clip
        if duration > clip_duration:
            clips.append({
                "start_time": max(0, duration - clip_duration),
                "end_time": duration,
                "duration": clip_duration,
                "score": 0.6
                    })
        
        return clips
    
    def _create_compilation(self, best_clips: List[Tuple[str, Dict]], 
                           transition_style: str, upload_id: str, aspect_ratio: str) -> str:
        """Create compilation video with transitions"""
        
        # Create filter complex for transitions
        filter_complex = self._create_transition_filter_simple(best_clips, transition_style)
        
        # Prepare input files (include each video multiple times if needed)
        input_files = []
        for video_path, clip in best_clips:
            input_files.extend(['-i', video_path])
        
        # Output path
        output_path = self.temp_dir / f"{upload_id}_compilation.mp4"
        
        # OPTIMIZED FFmpeg command for faster processing
        cmd = [
            'ffmpeg',
            *input_files,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', 'libx264',
            '-preset', 'fast',  # Faster encoding
            '-crf', '25',  # Slightly lower quality for speed
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-threads', '4',  # Use multiple threads
            '-y', str(output_path)
        ]
        
        self.logger.info(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"Compilation video created: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg compilation failed: {e.stderr.decode()}")
            raise RuntimeError(f"FFmpeg compilation failed: {e.stderr.decode()}")
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def _create_transition_filter_simple(self, best_clips: List[Tuple[str, Dict]], 
                                       transition_style: str) -> str:
        """Create FFmpeg filter complex for transitions with simple input handling"""
        
        filters = []
        inputs = []
        
        for i, (video_path, clip) in enumerate(best_clips):
            # Each clip gets its own input index (i)
            # Trim video to clip and scale to common resolution (720x1280)
            trim_filter = f"[{i}:v]trim=start={clip['start_time']}:end={clip['end_time']},setpts=PTS-STARTPTS,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v{i}]"
            filters.append(trim_filter)
            
            # Trim audio
            audio_filter = f"[{i}:a]atrim=start={clip['start_time']}:end={clip['end_time']},asetpts=PTS-STARTPTS[a{i}]"
            filters.append(audio_filter)
            
            inputs.append(f"[v{i}]")
            inputs.append(f"[a{i}]")
        
        # Create transitions
        if transition_style == "fade":
            transition_filter = self._create_fade_transitions(inputs, len(best_clips))
        elif transition_style == "slide":
            transition_filter = self._create_slide_transitions(inputs, len(best_clips))
        else:  # crossfade
            transition_filter = self._create_crossfade_transitions(inputs, len(best_clips))
        
        filters.append(transition_filter)
        
        return ';'.join(filters)
    
    def _create_transition_filter_optimized(self, best_clips: List[Tuple[str, Dict]], 
                                           transition_style: str, video_to_index: Dict[str, int]) -> str:
        """Create FFmpeg filter complex for transitions with optimized input handling"""
        
        filters = []
        inputs = []
        
        for i, (video_path, clip) in enumerate(best_clips):
            # Get the input index for this video
            input_index = video_to_index[video_path]
            
            # Trim video to clip and scale to common resolution (720x1280)
            trim_filter = f"[{input_index}:v]trim=start={clip['start_time']}:end={clip['end_time']},setpts=PTS-STARTPTS,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v{i}]"
            filters.append(trim_filter)
            
            # Trim audio
            audio_filter = f"[{input_index}:a]atrim=start={clip['start_time']}:end={clip['end_time']},asetpts=PTS-STARTPTS[a{i}]"
            filters.append(audio_filter)
            
            inputs.append(f"[v{i}]")
            inputs.append(f"[a{i}]")
        
        # Create transitions
        if transition_style == "fade":
            transition_filter = self._create_fade_transitions(inputs, len(best_clips))
        elif transition_style == "slide":
            transition_filter = self._create_slide_transitions(inputs, len(best_clips))
        else:  # crossfade
            transition_filter = self._create_crossfade_transitions(inputs, len(best_clips))
        
        filters.append(transition_filter)
        
        return ';'.join(filters)
    
    def _create_transition_filter(self, best_clips: List[Tuple[str, Dict]], 
                                transition_style: str) -> str:
        """Create FFmpeg filter complex for transitions (legacy method)"""
        
        filters = []
        inputs = []
        outputs = []
        
        for i, (video_path, clip) in enumerate(best_clips):
            # Trim video to clip and scale to common resolution (720x1280)
            trim_filter = f"[{i}:v]trim=start={clip['start_time']}:end={clip['end_time']},setpts=PTS-STARTPTS,scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2[v{i}]"
            filters.append(trim_filter)
            
            # Trim audio
            audio_filter = f"[{i}:a]atrim=start={clip['start_time']}:end={clip['end_time']},asetpts=PTS-STARTPTS[a{i}]"
            filters.append(audio_filter)
            
            inputs.append(f"[v{i}]")
            inputs.append(f"[a{i}]")
        
        # Create transitions
        if transition_style == "fade":
            transition_filter = self._create_fade_transitions(inputs, len(best_clips))
        elif transition_style == "slide":
            transition_filter = self._create_slide_transitions(inputs, len(best_clips))
        else:  # crossfade
            transition_filter = self._create_crossfade_transitions(inputs, len(best_clips))
        
        filters.append(transition_filter)
        
        return ';'.join(filters)
    
    def _create_fade_transitions(self, inputs: List[str], num_clips: int) -> str:
        """Create fade transitions between clips"""
        if num_clips == 1:
            return f"{inputs[0]}{inputs[1]}concat=n=1:v=1:a=1[out]"
        
        # Create fade in/out for each clip
        fade_inputs = []
        for i in range(0, len(inputs), 2):
            v_input = inputs[i]
            a_input = inputs[i + 1]
            
            # Add fade in/out
            fade_v = f"{v_input}fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[vfade{i//2}]"
            fade_a = f"{a_input}afade=t=in:st=0:d=1,afade=t=out:st=4:d=1[afade{i//2}]"
            
            fade_inputs.extend([fade_v, fade_a])
        
        # Concatenate all clips
        concat_inputs = []
        for i in range(num_clips):
            concat_inputs.extend([f"[vfade{i}]", f"[afade{i}]"])
        
        concat_filter = f"{';'.join(fade_inputs)};{''.join(concat_inputs)}concat=n={num_clips}:v=1:a=1[out]"
        return concat_filter
    
    def _create_crossfade_transitions(self, inputs: List[str], num_clips: int) -> str:
        """Create crossfade transitions between clips"""
        if num_clips == 1:
            return f"{inputs[0]}{inputs[1]}concat=n=1:v=1:a=1[out]"
        
        # For simplicity, use fade transitions as crossfade
        return self._create_fade_transitions(inputs, num_clips)
    
    def _create_slide_transitions(self, inputs: List[str], num_clips: int) -> str:
        """Create slide transitions between clips"""
        if num_clips == 1:
            return f"{inputs[0]}{inputs[1]}concat=n=1:v=1:a=1[out]"
        
        # For simplicity, use fade transitions as slide
        return self._create_fade_transitions(inputs, num_clips)
    
    def _optimize_for_platform(self, input_path: str, upload_id: str, preset_config: Dict) -> str:
        """Optimize video for specific platform"""
        if not preset_config:
            return self._optimize_for_youtube(input_path, upload_id)
        
        platform = preset_config["platform"]
        aspect_ratio = preset_config["aspect_ratio"]
        
        if platform == "YouTube":
            return self._optimize_for_youtube(input_path, upload_id, aspect_ratio)
        elif platform == "Instagram":
            return self._optimize_for_instagram(input_path, upload_id, aspect_ratio)
        elif platform == "TikTok":
            return self._optimize_for_tiktok(input_path, upload_id, aspect_ratio)
        elif platform == "Facebook":
            return self._optimize_for_facebook(input_path, upload_id, aspect_ratio)
        elif platform == "Twitter":
            return self._optimize_for_twitter(input_path, upload_id, aspect_ratio)
        elif platform == "LinkedIn":
            return self._optimize_for_linkedin(input_path, upload_id, aspect_ratio)
        elif platform == "Snapchat":
            return self._optimize_for_snapchat(input_path, upload_id, aspect_ratio)
        elif platform == "Pinterest":
            return self._optimize_for_pinterest(input_path, upload_id, aspect_ratio)
        else:
            return self._optimize_for_youtube(input_path, upload_id, aspect_ratio)
    
    def _optimize_for_youtube(self, input_path: str, upload_id: str, aspect_ratio: str = "16:9") -> str:
        """Optimize video for YouTube upload"""
        output_path = self.temp_dir / f"{upload_id}_youtube_ready.mp4"
        
        # YouTube recommended settings
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'slow',  # Better compression
            '-crf', '18',  # High quality
            '-c:a', 'aac',
            '-b:a', '192k',  # Higher audio bitrate
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',  # YouTube compatible
            '-vf', f'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',  # 1080p
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_instagram(self, input_path: str, upload_id: str, aspect_ratio: str = "9:16") -> str:
        """Optimize video for Instagram"""
        output_path = self.temp_dir / f"{upload_id}_instagram_ready.mp4"
        
        # Instagram optimization
        if aspect_ratio == "9:16":
            # Vertical format for Reels/Stories
            scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        else:
            # Square format
            scale_filter = "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_tiktok(self, input_path: str, upload_id: str, aspect_ratio: str = "9:16") -> str:
        """Optimize video for TikTok"""
        output_path = self.temp_dir / f"{upload_id}_tiktok_ready.mp4"
        
        # TikTok vertical format
        scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',  # Faster for social media
            '-crf', '25',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_facebook(self, input_path: str, upload_id: str, aspect_ratio: str = "9:16") -> str:
        """Optimize video for Facebook"""
        output_path = self.temp_dir / f"{upload_id}_facebook_ready.mp4"
        
        # Facebook optimization
        if aspect_ratio == "9:16":
            scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        else:
            scale_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_twitter(self, input_path: str, upload_id: str, aspect_ratio: str = "16:9") -> str:
        """Optimize video for Twitter"""
        output_path = self.temp_dir / f"{upload_id}_twitter_ready.mp4"
        
        # Twitter optimization
        scale_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '25',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_linkedin(self, input_path: str, upload_id: str, aspect_ratio: str = "16:9") -> str:
        """Optimize video for LinkedIn"""
        output_path = self.temp_dir / f"{upload_id}_linkedin_ready.mp4"
        
        # LinkedIn professional optimization
        scale_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'slow',  # Better quality for professional content
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_snapchat(self, input_path: str, upload_id: str, aspect_ratio: str = "9:16") -> str:
        """Optimize video for Snapchat"""
        output_path = self.temp_dir / f"{upload_id}_snapchat_ready.mp4"
        
        # Snapchat vertical format
        scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '25',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _optimize_for_pinterest(self, input_path: str, upload_id: str, aspect_ratio: str = "2:3") -> str:
        """Optimize video for Pinterest"""
        output_path = self.temp_dir / f"{upload_id}_pinterest_ready.mp4"
        
        # Pinterest 2:3 aspect ratio
        scale_filter = "scale=1080:1620:force_original_aspect_ratio=decrease,pad=1080:1620:(ow-iw)/2:(oh-ih)/2"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-vf', scale_filter,
            '-y', str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1', video_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    
    def _apply_post_compilation_effects(self, input_path: str, upload_id: str, effect_type: str) -> str:
        """Apply post-compilation effects to the entire video"""
        output_path = self.temp_dir / f"{upload_id}_with_effects.mp4"
        
        # Get effect configuration
        effect_config = self._get_effect_config(effect_type)
        if not effect_config:
            return input_path  # Return original if effect not found
        
        # Build FFmpeg command with effects
        cmd = ['ffmpeg', '-i', input_path]
        
        # Add video filters based on effect type
        video_filters = []
        
        if effect_type == "vintage":
            video_filters.extend([
                "curves=preset=vintage",
                "colorbalance=rs=-0.1:gs=0:bs=0.1",
                "eq=saturation=0.8:contrast=1.2"
            ])
        elif effect_type == "cinematic":
            video_filters.extend([
                "curves=preset=cinematic",
                "eq=contrast=1.3:saturation=0.9",
                "colorbalance=rs=0.05:gs=0:bs=-0.05"
            ])
        elif effect_type == "warm":
            video_filters.extend([
                "colorbalance=rs=0.1:gs=0.05:bs=-0.1",
                "eq=saturation=1.1:contrast=1.1"
            ])
        elif effect_type == "cool":
            video_filters.extend([
                "colorbalance=rs=-0.1:gs=0:bs=0.1",
                "eq=saturation=0.9:contrast=1.1"
            ])
        elif effect_type == "dramatic":
            video_filters.extend([
                "curves=preset=dramatic",
                "eq=contrast=1.4:saturation=1.2",
                "colorbalance=rs=0.1:gs=0:bs=-0.1"
            ])
        elif effect_type == "bright":
            video_filters.extend([
                "eq=brightness=0.1:contrast=1.2:saturation=1.1"
            ])
        elif effect_type == "moody":
            video_filters.extend([
                "eq=brightness=-0.1:contrast=1.3:saturation=0.8",
                "colorbalance=rs=-0.05:gs=0:bs=0.05"
            ])
        elif effect_type == "vibrant":
            video_filters.extend([
                "eq=saturation=1.3:contrast=1.2",
                "colorbalance=rs=0.05:gs=0:bs=-0.05"
            ])
        elif effect_type == "monochrome":
            video_filters.extend([
                "hue=s=0",
                "eq=contrast=1.2"
            ])
        elif effect_type == "sepia":
            video_filters.extend([
                "colorbalance=rs=0.2:gs=0.1:bs=-0.3",
                "eq=saturation=0.7:contrast=1.1"
            ])
        
        # Add video filter if effects are specified
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        # Add output settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',  # Keep original audio
            '-movflags', '+faststart',
            '-y', str(output_path)
        ])
        
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    
    def _get_effect_config(self, effect_type: str) -> Dict:
        """Get configuration for a specific effect"""
        effects = self.get_post_compilation_effects()
        for effect in effects:
            if effect["id"] == effect_type:
                return effect
        return None
    
    def get_transition_styles(self) -> List[Dict]:
        """Get available transition styles"""
        return [
            {"id": "fade", "name": "Fade In/Out", "description": "Smooth fade transitions between clips", "icon": "🌅"},
            {"id": "crossfade", "name": "Crossfade", "description": "Overlapping fade transitions for seamless flow", "icon": "🔄"},
            {"id": "slide", "name": "Slide", "description": "Slide transitions between clips", "icon": "➡️"},
            {"id": "zoom", "name": "Zoom", "description": "Zoom in/out transitions for dynamic effect", "icon": "🔍"},
            {"id": "wipe", "name": "Wipe", "description": "Wipe transitions for modern look", "icon": "🧹"},
            {"id": "dissolve", "name": "Dissolve", "description": "Dissolve transitions for artistic effect", "icon": "✨"}
        ]
    
    def get_post_compilation_effects(self) -> List[Dict]:
        """Get available post-compilation effects"""
        return [
            {
                "id": "none",
                "name": "No Effect",
                "description": "Keep original video without any effects",
                "icon": "🎬",
                "category": "none"
            },
            {
                "id": "vintage",
                "name": "Vintage",
                "description": "Classic film look with warm tones and grain",
                "icon": "📷",
                "category": "retro"
            },
            {
                "id": "cinematic",
                "name": "Cinematic",
                "description": "Movie-like appearance with enhanced contrast",
                "icon": "🎭",
                "category": "professional"
            },
            {
                "id": "warm",
                "name": "Warm",
                "description": "Cozy, golden-hour lighting effect",
                "icon": "🌅",
                "category": "color"
            },
            {
                "id": "cool",
                "name": "Cool",
                "description": "Blue-tinted, modern aesthetic",
                "icon": "❄️",
                "category": "color"
            },
            {
                "id": "dramatic",
                "name": "Dramatic",
                "description": "High contrast, bold colors for impact",
                "icon": "⚡",
                "category": "professional"
            },
            {
                "id": "bright",
                "name": "Bright",
                "description": "Enhanced brightness and vibrant colors",
                "icon": "☀️",
                "category": "color"
            },
            {
                "id": "moody",
                "description": "Dark, atmospheric mood with reduced brightness",
                "icon": "🌙",
                "category": "atmospheric"
            },
            {
                "id": "vibrant",
                "name": "Vibrant",
                "description": "Saturated, eye-catching colors",
                "icon": "🌈",
                "category": "color"
            },
            {
                "id": "monochrome",
                "name": "Monochrome",
                "description": "Classic black and white effect",
                "icon": "⚫",
                "category": "retro"
            },
            {
                "id": "sepia",
                "name": "Sepia",
                "description": "Antique brown-tinted effect",
                "icon": "📜",
                "category": "retro"
            }
        ]
    
    def get_compilation_presets(self) -> List[Dict]:
        """Get compilation presets for different social media platforms"""
        return [
            {
                "id": "youtube_shorts",
                "name": "YouTube Shorts",
                "max_duration": 60,
                "description": "Vertical 9:16 format, perfect for mobile viewing",
                "aspect_ratio": "9:16",
                "features": ["Vertical format", "Mobile-optimized", "Trending content"],
                "platform": "YouTube"
            },
            {
                "id": "youtube_standard",
                "name": "YouTube Standard",
                "max_duration": 300,
                "description": "Traditional 16:9 format for desktop viewing",
                "aspect_ratio": "16:9",
                "features": ["Desktop optimized", "Longer content", "Detailed editing"],
                "platform": "YouTube"
            },
            {
                "id": "instagram_reels",
                "name": "Instagram Reels",
                "max_duration": 90,
                "description": "Vertical format with music and effects support",
                "aspect_ratio": "9:16",
                "features": ["Music integration", "Effects ready", "Story-friendly"],
                "platform": "Instagram"
            },
            {
                "id": "instagram_stories",
                "name": "Instagram Stories",
                "max_duration": 15,
                "description": "Quick 15-second stories for daily content",
                "aspect_ratio": "9:16",
                "features": ["Quick content", "Daily updates", "Engagement focused"],
                "platform": "Instagram"
            },
            {
                "id": "tiktok",
                "name": "TikTok",
                "max_duration": 60,
                "description": "Trending vertical format with viral potential",
                "aspect_ratio": "9:16",
                "features": ["Viral potential", "Trending format", "Music sync"],
                "platform": "TikTok"
            },
            {
                "id": "facebook_reels",
                "name": "Facebook Reels",
                "max_duration": 60,
                "description": "Facebook's short-form video format",
                "aspect_ratio": "9:16",
                "features": ["Facebook optimized", "Community focused", "Shareable"],
                "platform": "Facebook"
            },
            {
                "id": "twitter_video",
                "name": "Twitter Video",
                "max_duration": 140,
                "description": "Twitter's video format with character limit awareness",
                "aspect_ratio": "16:9",
                "features": ["Twitter optimized", "Quick consumption", "Thread-friendly"],
                "platform": "Twitter"
            },
            {
                "id": "linkedin_video",
                "name": "LinkedIn Video",
                "max_duration": 600,
                "description": "Professional format for business content",
                "aspect_ratio": "16:9",
                "features": ["Professional", "Business focused", "Educational"],
                "platform": "LinkedIn"
            },
            {
                "id": "snapchat_spotlight",
                "name": "Snapchat Spotlight",
                "max_duration": 60,
                "description": "Snapchat's vertical video format",
                "aspect_ratio": "9:16",
                "features": ["Snapchat native", "Youth audience", "Creative effects"],
                "platform": "Snapchat"
            },
            {
                "id": "pinterest_video",
                "name": "Pinterest Video",
                "max_duration": 60,
                "description": "Pinterest's visual discovery format",
                "aspect_ratio": "2:3",
                "features": ["Visual discovery", "Inspiration focused", "Pin-optimized"],
                "platform": "Pinterest"
            },
            {
                "id": "custom",
                "name": "Custom Format",
                "max_duration": 300,
                "description": "Customize your own format and settings",
                "aspect_ratio": "16:9",
                "features": ["Fully customizable", "Flexible duration", "Any platform"],
                "platform": "Custom"
            }
        ]
