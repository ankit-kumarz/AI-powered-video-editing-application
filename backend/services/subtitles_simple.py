#!/usr/bin/env python3
"""
Simple Subtitle Service
Handles subtitle generation and burning without heavy dependencies
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ffmpeg

@dataclass
class SubtitleSegment:
    start_time: float
    end_time: float
    text: str
    speaker: str = ""

class SimpleSubtitleService:
    def __init__(self):
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        self.default_font_size = 24
        self.default_font_color = "white"
        self.default_background_color = "black"
        
    def generate_subtitles_from_text(self, text_content: str, output_path: str) -> str:
        """Generate SRT subtitles from text content"""
        segments = self._parse_text_to_segments(text_content)
        srt_content = self._segments_to_srt(segments)
        
        # Save SRT file
        srt_path = output_path.replace('.mp4', '.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        return srt_path
    
    def _parse_text_to_segments(self, text_content: str) -> List[SubtitleSegment]:
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
            
            # Extract speaker if present
            speaker = ""
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].strip().isupper():
                    speaker = parts[0].strip()
                    line = parts[1].strip()
            
            segments.append(SubtitleSegment(
                start_time=current_time,
                end_time=current_time + duration,
                text=line,
                speaker=speaker
            ))
            
            current_time += duration
        
        return segments
    
    def _segments_to_srt(self, segments: List[SubtitleSegment]) -> str:
        """Convert segments to SRT format"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time_str = self._seconds_to_srt_time(segment.start_time)
            end_time_str = self._seconds_to_srt_time(segment.end_time)
            
            srt_content += f"{i}\n"
            srt_content += f"{start_time_str} --> {end_time_str}\n"
            
            if segment.speaker:
                srt_content += f"{segment.speaker}: {segment.text}\n"
            else:
                srt_content += f"{segment.text}\n"
            
            srt_content += "\n"
        
        return srt_content
    
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
            # FFmpeg command to burn subtitles
            stream = ffmpeg.input(video_path)
            
            # Add subtitle filter
            stream = ffmpeg.filter(stream, 'subtitles', srt_path,
                                 force_style=f'FontSize={self.default_font_size},'
                                           f'PrimaryColour=&H{self.default_font_color},'
                                           f'BackColour=&H{self.default_background_color}')
            
            # Output
            stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='copy')
            
            # Run FFmpeg
            ffmpeg.run(stream, overwrite_output=True)
            
            return output_path
            
        except Exception as e:
            print(f"Error burning subtitles: {e}")
            return video_path
    
    def translate_subtitles(self, srt_path: str, target_language: str) -> str:
        """Translate subtitles to target language (placeholder implementation)"""
        if target_language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {target_language}")
        
        # Read original SRT
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # For now, just return the original content
        # In a real implementation, this would use a translation service
        translated_path = srt_path.replace('.srt', f'_{target_language}.srt')
        
        with open(translated_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return translated_path
    
    def get_subtitle_statistics(self, srt_path: str) -> Dict[str, Any]:
        """Get statistics about subtitle file"""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            segments = self._parse_srt_content(content)
            
            total_duration = sum(seg.end_time - seg.start_time for seg in segments)
            total_words = sum(len(seg.text.split()) for seg in segments)
            
            return {
                'segment_count': len(segments),
                'total_duration': total_duration,
                'total_words': total_words,
                'average_words_per_segment': total_words / len(segments) if segments else 0,
                'average_segment_duration': total_duration / len(segments) if segments else 0
            }
            
        except Exception as e:
            print(f"Error getting subtitle statistics: {e}")
            return {}
    
    def _parse_srt_content(self, content: str) -> List[SubtitleSegment]:
        """Parse SRT content into segments"""
        segments = []
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp line
                time_line = lines[1]
                start_time, end_time = self._parse_srt_timestamp(time_line)
                
                # Parse text (lines 2 onwards)
                text = '\n'.join(lines[2:])
                
                segments.append(SubtitleSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
        
        return segments
    
    def _parse_srt_timestamp(self, time_line: str) -> tuple[float, float]:
        """Parse SRT timestamp line"""
        start_str, end_str = time_line.split(' --> ')
        
        def time_str_to_seconds(time_str: str) -> float:
            time_part, ms_part = time_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            return h * 3600 + m * 60 + s + ms / 1000
        
        return time_str_to_seconds(start_str), time_str_to_seconds(end_str)
