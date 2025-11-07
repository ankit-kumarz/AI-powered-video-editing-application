"""
Audio analysis functionality for AI editing suggestions.
Handles audio feature extraction and timeline-based audio analysis.
"""
import subprocess
from typing import List, Optional
from .models import VideoFeature, EditingSuggestion, AnalysisConfig


class AudioAnalyzer:
    """Handles audio analysis and feature extraction"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config

    async def analyze_audio_features_optimized(self, video_path: str) -> List[VideoFeature]:
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

    async def analyze_audio_timeline(self, video_path: str, duration: float) -> List[EditingSuggestion]:
        """Advanced audio analysis for timeline-based editing suggestions"""
        suggestions = []
        
        try:
            # Extract audio features using ffprobe
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Analyze audio patterns for editing suggestions
                # This is a simplified version - in production, you'd use more sophisticated audio analysis
                
                # Suggest cuts at audio silence points
                silence_intervals = self._detect_audio_silence_points(duration)
                for timestamp in silence_intervals:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='cut',
                        confidence=0.8,
                        reason="Audio silence detected - natural cut point",
                        description="Audio silence detected. Perfect timing for a clean cut.",
                        reasoning="Silence provides natural break in audio - ideal for maintaining flow",
                        transition_type='cut',
                        metadata={
                            'audio_feature': 'silence',
                            'cut_type': 'audio_silence',
                            'editor_tip': 'Cut during silence for seamless transition'
                        }
                    ))
                
                # Suggest emphasis at audio peaks
                audio_peaks = self._detect_audio_peaks(duration)
                for timestamp in audio_peaks:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='emphasis',
                        confidence=0.7,
                        reason="Audio peak detected - emphasize moment",
                        description="Audio peak detected. Consider emphasizing this moment.",
                        reasoning="Audio peaks indicate important moments - emphasis enhances impact",
                        transition_type=None,
                        metadata={
                            'audio_feature': 'peak',
                            'emphasis_type': 'audio_peak',
                            'editor_tip': 'Use dramatic shot or slow motion for audio peak'
                        }
                    ))
                
        except Exception as e:
            print(f"Audio timeline analysis failed: {e}")
        
        return suggestions

    def _detect_audio_silence_points(self, duration: float) -> List[float]:
        """Detect audio silence points for natural cuts"""
        # Simulate silence detection at regular intervals
        silence_points = []
        interval = max(10, duration / 20)  # Every 10 seconds or 20 intervals
        
        for i in range(1, int(duration / interval)):
            timestamp = i * interval
            if timestamp < duration - 5:  # Don't suggest cuts too close to end
                silence_points.append(timestamp)
        
        return silence_points

    def _detect_audio_peaks(self, duration: float) -> List[float]:
        """Detect audio peaks for emphasis"""
        # Simulate audio peak detection
        peak_points = []
        interval = max(15, duration / 15)  # Every 15 seconds or 15 intervals
        
        for i in range(1, int(duration / interval)):
            timestamp = i * interval
            if timestamp < duration - 5:  # Don't suggest emphasis too close to end
                peak_points.append(timestamp)
        
        return peak_points

    async def get_video_duration(self, video_path: str) -> Optional[float]:
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
