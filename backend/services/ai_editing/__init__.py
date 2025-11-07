"""
AI Editing Suggestions Service - Refactored Architecture

This module provides AI-powered editing suggestions for video content.
The service has been refactored into smaller, focused modules for better maintainability.

Modules:
- models: Data structures and configuration
- video_analyzer: Video analysis and feature extraction
- audio_analyzer: Audio analysis and timeline optimization
- script_analyzer: Script text analysis and narrative structure
- suggestion_generator: Suggestion generation and timeline optimization
- service: Main orchestrating service class
"""

from .service import AIEditingSuggestionsService
from .models import VideoFeature, EditingSuggestion, AnalysisConfig

__all__ = [
    'AIEditingSuggestionsService',
    'VideoFeature', 
    'EditingSuggestion',
    'AnalysisConfig'
]
