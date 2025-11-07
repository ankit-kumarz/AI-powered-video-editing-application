"""
Base models and data structures for AI editing suggestions.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class VideoFeature:
    """Represents a video feature at a specific timestamp"""
    timestamp: float
    feature_type: str  # 'scene_change', 'motion', 'face_detected', 'silence', 'loud_audio'
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class EditingSuggestion:
    """Represents an AI editing suggestion"""
    timestamp: float
    suggestion_type: str  # 'cut', 'transition', 'pace_change', 'emphasis'
    confidence: float
    reason: str
    description: str = None  # Add description field
    reasoning: str = None    # Add reasoning field
    transition_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        # Set description and reasoning from reason if not provided
        if self.description is None:
            self.description = self.reason
        if self.reasoning is None:
            self.reasoning = self.reason
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalysisConfig:
    """Configuration for video analysis"""
    scene_change_threshold: float = 0.15
    motion_threshold: float = 0.1
    max_frames_to_analyze: int = 500
    max_video_duration: int = 1800  # 30 minutes max
    frame_resize_factor: float = 0.3
    batch_size: int = 20
    gc_interval: int = 5
    max_workers: int = 8
    parallel_batch_size: int = 10
    enable_parallel: bool = True
    enable_cache: bool = True
    cache_compression: bool = True
    cache_ttl: int = 86400 * 7  # 7 days
    cache_dir: str = "backend/temp/analysis_cache"  # Cache directory path
    timeline_analysis_enabled: bool = True
    advanced_audio_analysis: bool = True
    enable_optical_flow: bool = True
    emotion_detection_enabled: bool = True
    action_detection_enabled: bool = True
    genre_aware_analysis: bool = True
