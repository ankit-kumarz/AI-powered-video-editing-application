import whisper
import subprocess
import tempfile
import os
from pathlib import Path
import json
from googletrans import Translator
from TTS.api import TTS
from gtts import gTTS
import numpy as np
import wave
import struct

class VoiceTranslationService:
    def __init__(self):
        self.whisper_model = None  # lazy load
        self.translator = Translator()
        self.tts = None  # lazy load
    
    async def process(self, upload_id: str, processing_status: dict, target_language: str = "es", voice_type: str = "female"):
        """Process voice translation and dubbing with subtitle translation"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 10
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            processing_status[upload_id]["progress"] = 20
            audio_path = self._extract_audio(str(file_path))
            
            # Transcribe audio
            processing_status[upload_id]["progress"] = 30
            transcription = self._transcribe_audio(audio_path)
            
            # Translate text
            processing_status[upload_id]["progress"] = 40
            translated_text = self._translate_text(transcription, target_language)
            
            # Generate new audio
            processing_status[upload_id]["progress"] = 50
            new_audio_path = self._generate_speech(translated_text, target_language, voice_type)
            
            # Check if video has existing subtitles and translate them
            processing_status[upload_id]["progress"] = 60
            subtitle_path = None
            if self._has_subtitles(str(file_path)):
                subtitle_path = self._extract_and_translate_subtitles(str(file_path), target_language, upload_id)
            else:
                # Generate new subtitles for the translated audio
                subtitle_path = self._generate_translated_subtitles(translated_text, target_language, upload_id)
            
            # Replace audio in video and add translated subtitles if available
            processing_status[upload_id]["progress"] = 70
            output_path = output_dir / f"{upload_id}_dubbed_{target_language}.mp4"
            self._replace_audio_with_subtitles(str(file_path), new_audio_path, str(output_path), subtitle_path)
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(new_audio_path):
                os.remove(new_audio_path)
            if subtitle_path and os.path.exists(subtitle_path):
                os.remove(subtitle_path)
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            processing_status[upload_id]["original_text"] = transcription
            processing_status[upload_id]["translated_text"] = translated_text
            processing_status[upload_id]["target_language"] = target_language
            processing_status[upload_id]["subtitle_translated"] = subtitle_path is not None
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Voice translation error: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            '-y', temp_audio.name
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_audio.name
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("base")
            result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text using Google Translate"""
        try:
            # Split text into sentences for better translation
            sentences = text.split('.')
            translated_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    translated = self.translator.translate(
                        sentence.strip(), 
                        dest=target_language
                    )
                    translated_sentences.append(translated.text)
            
            return '. '.join(translated_sentences)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def _generate_speech(self, text: str, language: str, voice_type: str) -> str:
        """Generate speech from translated text using TTS"""
        try:
            if self.tts is None:
                # Try to init Coqui lazily; if it fails, fall back to gTTS
                try:
                    self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                except Exception:
                    self.tts = None
                    return self._generate_speech_gtts(text, language)
            
            # Create temporary file for output
            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_output.close()
            
            # Generate speech
            self.tts.tts_to_file(text=text, file_path=temp_output.name)
            
            return temp_output.name
            
        except Exception as e:
            print(f"TTS error, falling back to gTTS: {e}")
            return self._generate_speech_gtts(text, language)

    def _generate_speech_gtts(self, text: str, language: str) -> str:
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output.close()
        # gTTS outputs mp3; create mp3 then convert to wav
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.close()
        try:
            gTTS(text=text or " ", lang=(language if len(language) == 2 else "en")).save(tmp_mp3.name)
        except Exception:
            # fallback to english
            gTTS(text=text or " ", lang="en").save(tmp_mp3.name)

        cmd = [
            'ffmpeg', '-i', tmp_mp3.name,
            '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
            '-y', temp_output.name
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        try:
            os.remove(tmp_mp3.name)
        except Exception:
            pass
        return temp_output.name
    
    def _generate_speech_espeak(self, text: str, language: str) -> str:
        """Generate speech using espeak as fallback"""
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output.close()
        
        # Map language codes to espeak voices
        voice_map = {
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
            "ja": "japanese",
            "ko": "korean",
            "zh": "mandarin"
        }
        
        voice = voice_map.get(language, "english")
        
        cmd = [
            'espeak', '-w', temp_output.name,
            '-v', voice,
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # If espeak fails, create a silent audio file
            self._create_silent_audio(temp_output.name, 3.0)  # 3 seconds of silence
        
        return temp_output.name
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """Create a silent audio file as fallback"""
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Create silent audio data
        silent_data = [0] * num_samples
        
        # Write WAV file
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert to bytes
            audio_bytes = struct.pack(f'<{num_samples}h', *silent_data)
            wav_file.writeframes(audio_bytes)
    
    def _has_subtitles(self, video_path: str) -> bool:
        """Check if video has embedded subtitles"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 's',
                '-show_entries', 'stream=index', '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except Exception:
            return False
    
    def _extract_and_translate_subtitles(self, video_path: str, target_language: str, upload_id: str) -> str:
        """Extract subtitles from video and translate them"""
        try:
            # Extract subtitles
            subtitle_temp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            subtitle_temp.close()
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-map', '0:s:0',  # Extract first subtitle stream
                '-c:s', 'srt',
                '-y', subtitle_temp.name
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Read and translate subtitles
            with open(subtitle_temp.name, 'r', encoding='utf-8') as f:
                subtitle_content = f.read()
            
            # Translate subtitle content
            translated_subtitle = self._translate_subtitle_content(subtitle_content, target_language)
            
            # Save translated subtitles
            translated_subtitle_path = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            translated_subtitle_path.close()
            
            with open(translated_subtitle_path.name, 'w', encoding='utf-8') as f:
                f.write(translated_subtitle)
            
            # Clean up original subtitle file
            os.remove(subtitle_temp.name)
            
            return translated_subtitle_path.name
            
        except Exception as e:
            print(f"Subtitle extraction/translation error: {e}")
            return None
    
    def _translate_subtitle_content(self, subtitle_content: str, target_language: str) -> str:
        """Translate SRT subtitle content"""
        try:
            lines = subtitle_content.split('\n')
            translated_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    translated_lines.append(line)
                    continue
                
                # Skip timestamp lines and subtitle numbers
                if (line.isdigit() or 
                    '-->' in line or 
                    line.startswith('WEBVTT') or
                    line.startswith('NOTE')):
                    translated_lines.append(line)
                    continue
                
                # Translate subtitle text
                if line:
                    translated_text = self._translate_text(line, target_language)
                    translated_lines.append(translated_text)
                else:
                    translated_lines.append(line)
            
            return '\n'.join(translated_lines)
            
        except Exception as e:
            print(f"Subtitle translation error: {e}")
            return subtitle_content
    
    def _generate_translated_subtitles(self, translated_text: str, target_language: str, upload_id: str) -> str:
        """Generate SRT subtitles for translated text"""
        try:
            # Split translated text into sentences for subtitle timing
            sentences = translated_text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create SRT content
            srt_content = []
            subtitle_number = 1
            start_time = 0
            duration_per_sentence = 3.0  # 3 seconds per sentence
            
            for sentence in sentences:
                if not sentence:
                    continue
                
                # Calculate timestamps
                start_seconds = start_time
                end_seconds = start_time + duration_per_sentence
                
                # Format timestamps
                start_timestamp = self._seconds_to_srt_time(start_seconds)
                end_timestamp = self._seconds_to_srt_time(end_seconds)
                
                # Add subtitle entry
                srt_content.append(f"{subtitle_number}")
                srt_content.append(f"{start_timestamp} --> {end_timestamp}")
                srt_content.append(sentence)
                srt_content.append("")
                
                subtitle_number += 1
                start_time += duration_per_sentence
            
            # Save to temporary file
            subtitle_path = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            subtitle_path.close()
            
            with open(subtitle_path.name, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            return subtitle_path.name
            
        except Exception as e:
            print(f"Subtitle generation error: {e}")
            return None
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _replace_audio_with_subtitles(self, video_path: str, audio_path: str, output_path: str, subtitle_path: str = None):
        """Replace audio in video and optionally burn in translated subtitles"""
        try:
            if subtitle_path and os.path.exists(subtitle_path):
                # Burn in translated subtitles
                cmd = [
                    'ffmpeg', '-i', video_path, '-i', audio_path,
                    '-vf', f"scale=-2:'if(gte(ih,720),ih,720)',subtitles={subtitle_path}:force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,BackColour=&H000000,Outline=2,Shadow=1'",
                    '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-c:a', 'aac',
                    '-shortest',
                    '-movflags', '+faststart',
                    '-y', output_path
                ]
            else:
                # Just replace audio without subtitles
                cmd = [
                    'ffmpeg', '-i', video_path, '-i', audio_path,
                    '-vf', "scale=-2:'if(gte(ih,720),ih,720)'",
                    '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                    '-c:a', 'aac',
                    '-shortest',
                    '-movflags', '+faststart',
                    '-y', output_path
                ]

            subprocess.run(cmd, check=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            # Fallback to simple audio replacement
            self._replace_audio(video_path, audio_path, output_path)
    
    def _replace_audio(self, video_path: str, audio_path: str, output_path: str):
        """Replace audio in video using FFmpeg and ensure at least 720p height."""
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-vf', "scale=-2:'if(gte(ih,720),ih,720)'",
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac',
            '-shortest',
            '-movflags', '+faststart',
            '-y', output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages for translation"""
        return [
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese (Mandarin)"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"},
            {"code": "tr", "name": "Turkish"}
        ]
    
    def get_voice_options(self, language: str) -> list:
        """Get available voice options for a language"""
        # This would typically return available TTS voices
        # For now, return basic options
        return [
            {"id": "female", "name": "Female Voice"},
            {"id": "male", "name": "Male Voice"}
        ]
    
    def preview_translation(self, text: str, target_language: str) -> str:
        """Preview translation without processing full video"""
        try:
            return self._translate_text(text, target_language)
        except Exception as e:
            return f"Translation error: {str(e)}"
