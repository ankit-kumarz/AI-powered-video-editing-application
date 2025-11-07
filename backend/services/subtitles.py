import whisper
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import json
from googletrans import Translator
import shutil

class SubtitleService:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.translator = Translator()
    
    async def process(self, upload_id: str, processing_status: dict, burn_in: bool = True, language: str = "en"):
        """Process subtitle generation and optional burn-in"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths with better path resolution
            file_path_str = processing_status[upload_id]["file_path"]
            
            # Try multiple path resolution strategies
            file_path = None
            possible_paths = [
                Path("../" + file_path_str),  # Original approach
                Path(file_path_str),  # Direct path
                Path("temp/uploads") / Path(file_path_str).name,  # Just filename in uploads
                Path("backend/temp/uploads") / Path(file_path_str).name,  # Backend uploads
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    print(f"✅ Found video file at: {file_path}")
                    break
            
            if not file_path:
                # List available files for debugging
                print(f"❌ Video file not found. Tried paths:")
                for path in possible_paths:
                    print(f"   - {path} (exists: {path.exists()})")
                
                # Check what files are actually in the uploads directory
                uploads_dir = Path("temp/uploads")
                if uploads_dir.exists():
                    print(f"📁 Files in {uploads_dir}:")
                    for file in uploads_dir.glob("*"):
                        print(f"   - {file.name}")
                
                raise Exception(f"Video file not found. Tried: {[str(p) for p in possible_paths]}")
            
            # Ensure output directory exists
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            processing_status[upload_id]["progress"] = 30
            audio_path = self._extract_audio(str(file_path))
            
            # Transcribe audio
            processing_status[upload_id]["progress"] = 50
            transcription = self._transcribe_audio(audio_path)
            
            # Generate SRT file
            processing_status[upload_id]["progress"] = 70
            srt_path = output_dir / f"{upload_id}_subtitles.srt"
            self._generate_srt(transcription, str(srt_path))
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            if burn_in:
                # Burn subtitles into video
                processing_status[upload_id]["progress"] = 85
                output_path = output_dir / f"{upload_id}_with_subtitles.mp4"
                result_path = self.burn_subtitles(str(file_path), str(srt_path), str(output_path))
                
                # Update status with video output
                processing_status[upload_id]["progress"] = 100
                processing_status[upload_id]["status"] = "completed"
                processing_status[upload_id]["output_path"] = result_path
                processing_status[upload_id]["srt_path"] = str(srt_path)
            else:
                # Just return SRT file
                processing_status[upload_id]["progress"] = 100
                processing_status[upload_id]["status"] = "completed"
                processing_status[upload_id]["srt_path"] = str(srt_path)
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Subtitle generation error: {e}")
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg with improved error handling"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        # Find FFmpeg executable with multiple fallback options
        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        
        # Normalize video path for Windows compatibility
        video_path = str(Path(video_path).resolve())
        temp_audio_path = str(Path(temp_audio.name).resolve())
        
        # Verify input file exists before running FFmpeg
        if not os.path.exists(video_path):
            raise Exception(f"Input video file does not exist: {video_path}")
        
        print(f"🎵 Extracting audio from: {video_path}")
        print(f"🎵 Output audio to: {temp_audio_path}")
        
        cmd = [
            ffmpeg_path, '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', temp_audio_path
        ]
        
        try:
            # Run FFmpeg with detailed error handling
            print(f"🔧 Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                check=True
            )
            
            # Verify the output file was created and has content
            if not os.path.exists(temp_audio_path):
                raise Exception("Audio extraction failed - no output file created")
            
            file_size = os.path.getsize(temp_audio_path)
            if file_size == 0:
                raise Exception("Audio extraction failed - output file is empty")
            
            print(f"✅ Audio extraction successful. File size: {file_size} bytes")
            return temp_audio_path
            
        except subprocess.TimeoutExpired:
            raise Exception("Audio extraction timed out after 5 minutes")
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error (code {e.returncode}): {e.stderr}"
            print(f"❌ FFmpeg command failed: {' '.join(cmd)}")
            print(f"❌ FFmpeg stderr: {e.stderr}")
            print(f"❌ FFmpeg stdout: {e.stdout}")
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable with multiple fallback options"""
        # Common FFmpeg paths on Windows
        possible_paths = [
            'ffmpeg',  # If in PATH
            'C:\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Users\\HP\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffmpeg.EXE',  # From error message
        ]
        
        for path in possible_paths:
            try:
                if path == 'ffmpeg':
                    # Check if ffmpeg is in PATH
                    result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        return path
                else:
                    # Check specific path
                    if os.path.exists(path):
                        result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            return path
            except Exception:
                continue
        
        # Try using shutil.which as last resort
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        return None
    
    def _transcribe_audio(self, audio_path: str) -> list:
        """Transcribe audio using Whisper"""
        result = self.model.transcribe(audio_path)
        
        # Format transcription for SRT
        segments = []
        for i, segment in enumerate(result["segments"]):
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            segments.append({
                "index": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })
        
        return segments
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _generate_srt(self, segments: list, output_path: str):
        """Generate SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['index']}\n")
                f.write(f"{segment['start_time']} --> {segment['end_time']}\n")
                f.write(f"{segment['text']}\n\n")
    
    
    def _burn_subtitles_fallback(self, video_path: str, srt_path: str, output_path: str):
        """Fallback subtitle burning method using subprocess"""
        try:
            # Find FFmpeg executable
            ffmpeg_path = self._find_ffmpeg()
            if not ffmpeg_path:
                raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
            
            # Normalize paths for Windows compatibility
            video_path = str(Path(video_path).resolve())
            srt_path = str(Path(srt_path).resolve())
            output_path = str(Path(output_path).resolve())
            
            # Verify files exist
            if not os.path.exists(video_path):
                raise Exception(f"Input video file does not exist: {video_path}")
            if not os.path.exists(srt_path):
                raise Exception(f"SRT file does not exist: {srt_path}")
            
            # Use simple subtitle filter
            cmd = [
                ffmpeg_path, '-i', video_path,
                '-vf', f"subtitles={srt_path}",
                '-c:v', 'libx264',
                '-c:a', 'copy',
                '-y', output_path
            ]
            
            print(f"🔧 Fallback subtitle burning command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
            
            # Verify output
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Fallback subtitle burning failed - no output file created")
            
            print(f"✅ Fallback subtitle burning successful")
            
        except Exception as e:
            print(f"❌ Fallback method failed: {e}")
            # Final fallback to copying video without subtitles
            print("⚠️ All subtitle burning methods failed, copying video without subtitles")
            self._copy_video_without_subtitles(video_path, output_path, ffmpeg_path)
    
    def _copy_video_without_subtitles(self, video_path: str, output_path: str, ffmpeg_path: str):
        """Copy video without burning subtitles as fallback"""
        try:
            cmd = [
                ffmpeg_path, '-i', video_path,
                '-c', 'copy',  # Copy without re-encoding
                '-y', output_path
            ]
            
            print(f"📋 Copying video: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
            print(f"✅ Video copied successfully (without subtitles)")
            
        except Exception as e:
            raise Exception(f"Failed to copy video: {e}")
    
    def translate_subtitles(self, srt_path: str, target_language: str) -> str:
        """Translate subtitles to target language"""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT content
            segments = self._parse_srt(content)
            
            # Translate each text segment
            translated_segments = []
            for segment in segments:
                translated_text = self.translator.translate(
                    segment['text'], 
                    dest=target_language
                ).text
                
                segment['text'] = translated_text
                translated_segments.append(segment)
            
            # Generate new SRT file
            output_path = srt_path.replace('.srt', f'_{target_language}.srt')
            self._generate_srt(translated_segments, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Translation error: {e}")
            return srt_path
    
    def _parse_srt(self, srt_content: str) -> list:
        """Parse SRT content into segments"""
        segments = []
        lines = srt_content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.isdigit():  # Segment index
                # Find timestamp line
                timestamp_line = lines[i + 1].strip()
                start_time, end_time = timestamp_line.split(' --> ')
                
                # Find text lines
                text_lines = []
                j = i + 2
                while j < len(lines) and lines[j].strip():
                    text_lines.append(lines[j].strip())
                    j += 1
                
                text = ' '.join(text_lines)
                
                segments.append({
                    'index': int(line),
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
                
                i = j + 1
            else:
                i += 1
        
        return segments
    
    def generate_subtitles_from_text(self, text_content: str, video_path: str) -> str:
        """Generate SRT subtitles from text content"""
        segments = self._parse_text_to_segments(text_content)
        
        # Generate SRT file path
        srt_path = str(Path(video_path).parent / f"{Path(video_path).stem}_subtitles.srt")
        
        # Generate SRT content
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start_time_str = self._seconds_to_srt_time(segment['start_time'])
            end_time_str = self._seconds_to_srt_time(segment['end_time'])
            
            srt_content += f"{i}\n"
            srt_content += f"{start_time_str} --> {end_time_str}\n"
            srt_content += f"{segment['text']}\n\n"
        
        # Save SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        return srt_path
    
    def _parse_text_to_segments(self, text_content: str) -> list:
        """Parse text content into subtitle segments"""
        segments = []
        lines = text_content.split('\n')
        
        current_time = 0.0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Estimate duration based on text length
            duration = max(2.0, len(line) * 0.1)  # At least 2 seconds, 0.1s per character
            
            segments.append({
                'start_time': current_time,
                'end_time': current_time + duration,
                'text': line
            })
            
            current_time += duration
        
        return segments
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def burn_subtitles(self, video_path: str, srt_path: str, output_path: str) -> str:
        """Burn subtitles into video using FFmpeg"""
        try:
            # Import ffmpeg-python for subtitle burning
            import ffmpeg
            
            print(f"🔥 Burning subtitles from: {srt_path}")
            print(f"🔥 Input video: {video_path}")
            print(f"🔥 Output video: {output_path}")
            
            # Input the video file
            input_stream = ffmpeg.input(video_path)
            
            # Apply subtitle filter to video stream
            video_with_subtitles = ffmpeg.filter(input_stream, 'subtitles', srt_path,
                                               force_style='FontSize=24,PrimaryColour=&HFFFFFF,BackColour=&H000000,Bold=1')
            
            # Output with both video (with subtitles) and audio (original)
            stream = ffmpeg.output(video_with_subtitles, input_stream, output_path, 
                                 vcodec='libx264', acodec='copy')
            
            # Run FFmpeg
            print(f"🔧 Running ffmpeg-python subtitle burning...")
            ffmpeg.run(stream, overwrite_output=True)
            
            # Verify output
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Subtitle burning failed - no output file created")
            
            print(f"✅ Subtitle burning successful with ffmpeg-python")
            return output_path
            
        except ImportError:
            print("❌ ffmpeg-python not available, trying fallback method...")
            return self._burn_subtitles_fallback(video_path, srt_path, output_path)
        except Exception as e:
            print(f"❌ ffmpeg-python method failed: {e}")
            return self._burn_subtitles_fallback(video_path, srt_path, output_path)
    
    def _parse_srt_for_burning(self, srt_content: str) -> list:
        """Parse SRT content for subtitle burning with timing information"""
        segments = []
        lines = srt_content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.isdigit():  # Segment index
                # Find timestamp line
                timestamp_line = lines[i + 1].strip()
                start_time, end_time = timestamp_line.split(' --> ')
                
                # Find text lines
                text_lines = []
                j = i + 2
                while j < len(lines) and lines[j].strip():
                    text_lines.append(lines[j].strip())
                    j += 1
                
                text = ' '.join(text_lines)
                
                # Convert timestamps to seconds
                start_seconds = self._srt_time_to_seconds(start_time)
                end_seconds = self._srt_time_to_seconds(end_time)
                
                segments.append({
                    'start_seconds': start_seconds,
                    'end_seconds': end_seconds,
                    'text': text
                })
                
                i = j + 1
            else:
                i += 1
        
        return segments
    
    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT time format (HH:MM:SS,mmm) to seconds"""
        time_part, ms_part = srt_time.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        
        return h * 3600 + m * 60 + s + ms / 1000
