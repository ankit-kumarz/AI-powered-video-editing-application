"""
Main AI Editing Suggestions Service - Orchestrates all analysis modules.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import VideoFeature, EditingSuggestion, AnalysisConfig
from .video_analyzer import VideoAnalyzer
from .audio_analyzer import AudioAnalyzer
from .script_analyzer import ScriptAnalyzer
from .suggestion_generator import SuggestionGenerator


class AIEditingSuggestionsService:
    """Main service class that orchestrates all AI editing analysis modules"""
    
    def __init__(self):
        # Initialize configuration
        self.config = AnalysisConfig()
        
        # Initialize analysis modules
        self.video_analyzer = VideoAnalyzer(self.config)
        self.audio_analyzer = AudioAnalyzer(self.config)
        self.script_analyzer = ScriptAnalyzer()
        self.suggestion_generator = SuggestionGenerator(self.config)
        
        # Set up cache directory
        self.cache_dir = Path("backend/temp/analysis_cache")
        self.cache_dir.mkdir(exist_ok=True)

    async def analyze_video_features(self, video_path: str) -> List[VideoFeature]:
        """Analyze video and extract features for editing suggestions"""
        return await self.video_analyzer.analyze_video_features(video_path)

    async def generate_editing_suggestions(self, video_path: str, script_content: str = "") -> List[EditingSuggestion]:
        """Generate AI editing suggestions based on video features and optional script analysis"""
        try:
            # Analyze video features
            video_features = await self.analyze_video_features(video_path)
            
            # Generate suggestions based on video features
            video_suggestions = self.suggestion_generator.generate_video_based_suggestions(video_features)
            
            # Generate suggestions based on script analysis (if provided)
            script_suggestions = []
            if script_content:
                script_suggestions = self.script_analyzer.generate_script_based_suggestions_from_text(script_content)
            
            # ULTRA-OPTIMIZED TIMELINE ANALYSIS
            if self.config.timeline_analysis_enabled:
                timeline_suggestions = await self.suggestion_generator.generate_ultra_optimized_timeline_suggestions(video_path, video_features)
                video_suggestions.extend(timeline_suggestions)
            
            # Combine and rank suggestions
            all_suggestions = video_suggestions + script_suggestions
            ranked_suggestions = self._rank_suggestions(all_suggestions)
            
            return ranked_suggestions
            
        except Exception as e:
            raise RuntimeError(f"Editing suggestions generation failed: {e}")

    def _rank_suggestions(self, suggestions: List[EditingSuggestion]) -> List[EditingSuggestion]:
        """Rank suggestions by confidence and importance"""
        # Sort by confidence (highest first)
        ranked = sorted(suggestions, key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates (same timestamp and type)
        unique_suggestions = []
        seen = set()
        
        for suggestion in ranked:
            key = (suggestion.timestamp, suggestion.suggestion_type)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions

    def _group_suggestions(self, suggestions: List[EditingSuggestion]) -> Dict[str, List[EditingSuggestion]]:
        """Group suggestions by type"""
        grouped = {
            'cuts': [],
            'transitions': [],
            'emphasis': [],
            'pace_changes': []
        }
        
        for suggestion in suggestions:
            if suggestion.suggestion_type == 'cut':
                grouped['cuts'].append(suggestion)
            elif suggestion.suggestion_type == 'transition':
                grouped['transitions'].append(suggestion)
            elif suggestion.suggestion_type == 'emphasis':
                grouped['emphasis'].append(suggestion)
            elif suggestion.suggestion_type == 'pace_change':
                grouped['pace_changes'].append(suggestion)
        
        return grouped

    def _suggestion_to_dict(self, suggestion: EditingSuggestion) -> Dict[str, Any]:
        """Convert suggestion to dictionary format"""
        return {
            'timestamp': suggestion.timestamp,
            'suggestion_type': suggestion.suggestion_type,
            'confidence': suggestion.confidence,
            'reason': suggestion.reason,
            'description': suggestion.description,
            'reasoning': suggestion.reasoning,
            'transition_type': suggestion.transition_type,
            'metadata': suggestion.metadata
        }

    def _feature_to_dict(self, feature: VideoFeature) -> Dict[str, Any]:
        """Convert feature to dictionary format"""
        return {
            'timestamp': feature.timestamp,
            'feature_type': feature.feature_type,
            'confidence': feature.confidence,
            'metadata': feature.metadata
        }

    async def process(self, upload_id: str, processing_status: dict, script_content: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Process video and generate editing suggestions (for compatibility with main.py)"""
        try:
            # Update processing status
            processing_status[upload_id]["status"] = "processing"
            processing_status[upload_id]["progress"] = 10
            processing_status[upload_id]["message"] = "Starting AI editing analysis..."
            
            # Find the video file
            if not file_path:
                from pathlib import Path
                upload_dir = Path("temp/uploads")
                video_files = list(upload_dir.glob(f"{upload_id}*"))
                if not video_files:
                    raise FileNotFoundError(f"Video file not found for upload ID: {upload_id}")
                file_path = str(video_files[0])
            
            processing_status[upload_id]["progress"] = 30
            processing_status[upload_id]["message"] = "Analyzing video features..."
            
            # Analyze video features
            video_features = await self.analyze_video_features(file_path)
            
            processing_status[upload_id]["progress"] = 60
            processing_status[upload_id]["message"] = "Generating editing suggestions..."
            
            # Generate editing suggestions
            suggestions = await self.generate_editing_suggestions(file_path, script_content or "")
            
            processing_status[upload_id]["progress"] = 90
            processing_status[upload_id]["message"] = "Finalizing results..."
            
            # Convert to dictionary format for JSON serialization
            suggestions_dict = [self._suggestion_to_dict(s) for s in suggestions]
            features_dict = [self._feature_to_dict(f) for f in video_features]
            
            # Count suggestions by type
            cut_suggestions = len([s for s in suggestions if s.suggestion_type == "cut"])
            transition_suggestions = len([s for s in suggestions if s.suggestion_type == "transition"])
            emphasis_suggestions = len([s for s in suggestions if s.suggestion_type == "emphasis"])
            
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["message"] = "AI editing analysis completed"
            processing_status[upload_id]["result"] = {
                "success": True,
                "suggestions": suggestions_dict,
                "video_features": features_dict,
                "total_suggestions": len(suggestions),
                "cut_suggestions": cut_suggestions,
                "transition_suggestions": transition_suggestions,
                "emphasis_suggestions": emphasis_suggestions,
                "upload_id": upload_id
            }
            
            return processing_status[upload_id]["result"]
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["message"] = f"AI editing analysis failed: {str(e)}"
            raise e
