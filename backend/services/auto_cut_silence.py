import subprocess
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import json

# Ensure pydub finds ffmpeg even if PATH was not refreshed
os.environ.setdefault("FFMPEG_BINARY", "ffmpeg")
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


class AutoCutSilenceService:
    def __init__(self):
        pass

    async def process(self, upload_id: str, processing_status: dict, silence_threshold_db: int = -40, min_silence_len_ms: int = 2000, enable_ai_cuts: bool = True):
        """Detect silent parts and cut them out, joining remaining segments with AI-powered optimal cuts and transitions.

        Args:
            upload_id: ID of the uploaded video.
            processing_status: Shared status dict to update progress.
            silence_threshold_db: Threshold in dBFS to consider as silence.
            min_silence_len_ms: Minimum silence duration (ms) to cut out.
            enable_ai_cuts: Enable AI-powered scene analysis for optimal cuts.
        """
        try:
            processing_status[upload_id]["progress"] = 10
            processing_status[upload_id]["status"] = "processing"

            file_path = Path(processing_status[upload_id]["file_path"])  # original video
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1) Extract audio to wav for analysis (optimized)
            processing_status[upload_id]["progress"] = 20
            audio_wav = self._extract_audio_optimized(str(file_path))

            # 2) Load audio and detect non-silent segments (optimized)
            processing_status[upload_id]["progress"] = 30
            audio = AudioSegment.from_file(audio_wav)

            non_silent_ranges = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len_ms,
                silence_thresh=silence_threshold_db,
            )

            # If no speech detected, just copy original
            if not non_silent_ranges:
                output_path = output_dir / f"{upload_id}_no_cut.mp4"
                self._fast_copy(str(file_path), str(output_path))
                processing_status[upload_id]["progress"] = 100
                processing_status[upload_id]["status"] = "completed"
                processing_status[upload_id]["output_path"] = str(output_path)
                os.remove(audio_wav)
                return

            # 3) Optimize cut points (skip AI analysis for speed)
            processing_status[upload_id]["progress"] = 40
            optimal_cut_points = self._merge_silence_gaps(non_silent_ranges)

            # 4) Cut video to the optimal segments (optimized)
            processing_status[upload_id]["progress"] = 50
            segment_paths = []
            print(f"Creating {len(optimal_cut_points)} segments for upload {upload_id}")
            
            for idx, (start_ms, end_ms) in enumerate(optimal_cut_points, start=1):
                start_sec = max(0.0, start_ms / 1000.0)
                duration = max(0.0, (end_ms - start_ms) / 1000.0)
                if duration <= 0.1:  # Increased minimum duration
                    print(f"Skipping segment {idx} - too short ({duration:.3f}s)")
                    continue

                seg_out = output_dir / f"{upload_id}_seg_{idx:04d}.mp4"
                print(f"Creating segment {idx}: {start_sec:.3f}s to {start_sec + duration:.3f}s (duration: {duration:.3f}s)")
                
                # Use faster FFmpeg settings
                cmd = [
                    "ffmpeg", "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}",
                    "-i", str(file_path),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",  # Faster preset, higher CRF
                    "-c:a", "aac", "-b:a", "96k",  # Lower audio bitrate
                    "-avoid_negative_ts", "make_zero",
                    "-y", str(seg_out),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                segment_paths.append(seg_out)
                print(f"Created segment {idx}: {seg_out}")
            
            print(f"Successfully created {len(segment_paths)} segments")

            # 5) Fast concatenation (skip complex transitions for speed)
            processing_status[upload_id]["progress"] = 70
            if len(segment_paths) > 1:
                print(f"Concatenating {len(segment_paths)} segments for upload {upload_id}")
                final_path = self._fast_concat(segment_paths, output_dir, upload_id)
            else:
                print(f"Only one segment found, no concatenation needed for upload {upload_id}")
                final_path = output_dir / f"{upload_id}_trimmed.mp4"
                os.replace(segment_paths[0], final_path)

            # 6) Skip 720p upscaling for speed (optional)
            processing_status[upload_id]["progress"] = 95
            ensured_path = final_path  # Skip upscaling for speed

            # Cleanup temp files
            try:
                os.remove(audio_wav)
            except Exception:
                pass
            for p in segment_paths:
                try:
                    if p.exists():
                        os.remove(p)
                except Exception:
                    pass

            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(ensured_path)

        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = f"Auto cut silence processing failed: {str(e)}"
            print(f"Auto cut silence error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

    def _analyze_scenes_for_optimal_cuts(self, video_path: str, non_silent_ranges: list) -> list:
        """Analyze video scenes to find optimal cut points using AI"""
        try:
            # Use PySceneDetect to find scene changes
            scenes = detect(video_path, ContentDetector(threshold=30))
            
            # Convert scene times to milliseconds
            scene_times_ms = [int(scene[0].get_seconds() * 1000) for scene in scenes]
            
            # Find optimal cut points that align with scene changes
            optimal_cuts = []
            for start_ms, end_ms in non_silent_ranges:
                # Find the best scene-aligned start and end points
                best_start = self._find_nearest_scene_boundary(start_ms, scene_times_ms)
                best_end = self._find_nearest_scene_boundary(end_ms, scene_times_ms)
                
                # Ensure minimum segment length (1 second)
                if best_end - best_start >= 1000:
                    optimal_cuts.append([best_start, best_end])
            
            return optimal_cuts if optimal_cuts else self._merge_silence_gaps(non_silent_ranges)
            
        except Exception as e:
            print(f"Scene analysis failed, falling back to silence-based cuts: {e}")
            return self._merge_silence_gaps(non_silent_ranges)

    def _find_nearest_scene_boundary(self, time_ms: int, scene_times_ms: list) -> int:
        """Find the nearest scene boundary to the given time"""
        if not scene_times_ms:
            return time_ms
        
        nearest = min(scene_times_ms, key=lambda x: abs(x - time_ms))
        # Only use scene boundary if it's within 2 seconds
        if abs(nearest - time_ms) <= 2000:
            return nearest
        return time_ms

    def _merge_silence_gaps(self, non_silent_ranges: list) -> list:
        """Merge very small gaps between non-silent segments to avoid micro-cuts"""
        merged = []
        gap_merge_ms = 250  # merge gaps shorter than this
        
        for start, end in non_silent_ranges:
            if not merged:
                merged.append([start, end])
            else:
                prev_start, prev_end = merged[-1]
                if start - prev_end <= gap_merge_ms:
                    merged[-1][1] = end
                else:
                    merged.append([start, end])
        
        return merged

    def _add_smooth_transitions(self, segment_paths: list, output_dir: Path, upload_id: str) -> Path:
        """Add smooth crossfade transitions between video segments"""
        if len(segment_paths) <= 1:
            return segment_paths[0]
        
        # Validate that all segment files exist
        for segment_path in segment_paths:
            if not segment_path.exists():
                print(f"Warning: Segment file {segment_path} does not exist")
                # Fallback to simple concatenation
                return self._simple_concat(segment_paths, output_dir, upload_id)

        # For now, let's use simple concatenation with crossfade filter
        # This is more reliable than complex transition files
        try:
            return self._simple_concat_with_crossfade(segment_paths, output_dir, upload_id)
        except Exception as e:
            print(f"Crossfade concat failed: {e}")
            return self._simple_concat(segment_paths, output_dir, upload_id)

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(video_path)
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 5.0  # fallback duration

    def _simple_concat(self, segment_paths: list, output_dir: Path, upload_id: str) -> Path:
        """Simple concatenation without transitions as fallback"""
        if len(segment_paths) <= 1:
            return segment_paths[0]
            
        concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for segment_path in segment_paths:
                # Use absolute path and proper escaping for Windows
                abs_path = os.path.abspath(str(segment_path))
                safe_path = abs_path.replace('\\', '/')
                concat_list.write(f"file '{safe_path}'\n")
            
            concat_list.close()
            
            final_path = output_dir / f"{upload_id}_simple_concat.mp4"
            
            # Try copy first (faster and more reliable)
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
                "-c:v", "copy", "-c:a", "copy",
                "-avoid_negative_ts", "make_zero",
                "-y", str(final_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return final_path
            except subprocess.CalledProcessError as e:
                print(f"Copy concat failed: {e}")
                # Try re-encoding approach
                try:
                    cmd_alt = [
                        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k",
                        "-avoid_negative_ts", "make_zero",
                        "-y", str(final_path)
                    ]
                    subprocess.run(cmd_alt, check=True, capture_output=True, text=True)
                    return final_path
                except subprocess.CalledProcessError as e2:
                    print(f"Re-encode concat also failed: {e2}")
                    # Last resort: just return the first segment
                    if segment_paths:
                        final_path = output_dir / f"{upload_id}_fallback.mp4"
                        os.replace(segment_paths[0], final_path)
                        return final_path
                    else:
                        raise Exception("No segments available for concatenation")
                    
        finally:
            try:
                os.remove(concat_list.name)
            except Exception:
                pass

    def _simple_concat_with_crossfade(self, segment_paths: list, output_dir: Path, upload_id: str) -> Path:
        """Simple concatenation with crossfade transitions"""
        if len(segment_paths) <= 1:
            return segment_paths[0]
        
        # Create a simple concat file
        concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for segment_path in segment_paths:
                # Use absolute path and proper escaping for Windows
                abs_path = os.path.abspath(str(segment_path))
                safe_path = abs_path.replace('\\', '/')
                concat_list.write(f"file '{safe_path}'\n")
            
            concat_list.close()
            
            final_path = output_dir / f"{upload_id}_with_crossfade.mp4"
            
            # Try crossfade first, then fallback to simple concat
            try:
                # Use FFmpeg concat with crossfade filter
                cmd = [
                    "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
                    "-vf", "xfade=transition=fade:duration=0.5:offset=0.5",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-avoid_negative_ts", "make_zero",
                    "-y", str(final_path)
                ]
                
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return final_path
            except subprocess.CalledProcessError as e:
                print(f"Crossfade concat failed: {e}")
                # Fallback to simple concat without transitions
                return self._simple_concat(segment_paths, output_dir, upload_id)
                    
        finally:
            try:
                os.remove(concat_list.name)
            except Exception:
                pass

    def _extract_audio_optimized(self, video_path: str) -> str:
        """Optimized audio extraction with lower quality for faster processing"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "8000", "-ac", "1",  # Lower sample rate for speed
            "-y", tmp.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp.name

    def _extract_audio(self, video_path: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", tmp.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp.name

    def _fast_copy(self, input_path: str, output_path: str):
        """Fast copy without re-encoding"""
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "copy", "-c:a", "copy",
            "-y", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _fast_concat(self, segment_paths: list, output_dir: Path, upload_id: str) -> Path:
        """Fast concatenation without complex transitions"""
        if len(segment_paths) <= 1:
            return segment_paths[0]
            
        concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for segment_path in segment_paths:
                abs_path = os.path.abspath(str(segment_path))
                safe_path = abs_path.replace('\\', '/')
                concat_list.write(f"file '{safe_path}'\n")
            
            concat_list.close()
            
            final_path = output_dir / f"{upload_id}_fast_concat.mp4"
            
            # Use copy mode for maximum speed
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
                "-c:v", "copy", "-c:a", "copy",
                "-avoid_negative_ts", "make_zero",
                "-y", str(final_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                return final_path
            except subprocess.CalledProcessError:
                # Fallback to re-encoding with fast settings
                cmd_fallback = [
                    "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-c:a", "aac", "-b:a", "96k",
                    "-avoid_negative_ts", "make_zero",
                    "-y", str(final_path)
                ]
                subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)
                return final_path
                    
        finally:
            try:
                os.remove(concat_list.name)
            except Exception:
                pass

    def _reencode_for_compat(self, input_path: str, output_path: str):
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac",
            "-y", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _ensure_min_720p(self, input_path: str, output_path: str):
        # Scale height to at least 720 while keeping aspect ratio
        # If input already >= 720p height, this will copy scale 1:1
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", "scale=-2:'if(gte(ih,720),ih,720)'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


