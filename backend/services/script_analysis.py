#!/usr/bin/env python3
"""
Script Analysis Service
Analyzes script content to generate editing suggestions
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ScriptSegment:
    start_time: float
    end_time: float
    text: str
    emotion: str
    intensity: float
    speaker: str = ""

@dataclass
class ScriptSuggestion:
    timestamp: float
    suggestion_type: str
    description: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class ScriptAnalysisService:
    def __init__(self):
        self.emotion_keywords = {
            'excited': ['excited', 'thrilled', 'amazing', 'incredible', 'wow'],
            'calm': ['calm', 'peaceful', 'gentle', 'soft', 'quiet'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'emotional', 'moving'],
            'funny': ['funny', 'hilarious', 'comedy', 'joke', 'laugh'],
            'serious': ['serious', 'important', 'critical', 'crucial', 'vital']
        }
        
        self.transition_indicators = [
            'meanwhile', 'later', 'after', 'before', 'then', 'next',
            'suddenly', 'finally', 'eventually', 'meanwhile', 'however'
        ]

    def analyze_script(self, script_content: str) -> List[ScriptSuggestion]:
        """Analyze script content and generate editing suggestions"""
        suggestions = []
        
        # Parse script into segments
        segments = self._parse_script(script_content)
        
        # Generate suggestions based on segments
        for i, segment in enumerate(segments):
            # Emotion-based suggestions
            if segment.emotion and segment.intensity > 0.7:
                suggestions.append(ScriptSuggestion(
                    timestamp=segment.start_time,
                    suggestion_type="cut",
                    description=f"Emotional cut at {segment.start_time:.1f}s ({segment.emotion})",
                    confidence=0.8,
                    reasoning=f"High {segment.emotion} emotion detected",
                    metadata={"emotion": segment.emotion, "intensity": segment.intensity}
                ))
            
            # Transition suggestions
            if any(indicator in segment.text.lower() for indicator in self.transition_indicators):
                suggestions.append(ScriptSuggestion(
                    timestamp=segment.start_time,
                    suggestion_type="transition",
                    description=f"Script transition at {segment.start_time:.1f}s",
                    confidence=0.7,
                    reasoning="Transition indicator found in script",
                    metadata={"transition_type": "script_indicator"}
                ))
            
            # Pacing suggestions
            if i > 0 and (segment.start_time - segments[i-1].end_time) > 5:
                suggestions.append(ScriptSuggestion(
                    timestamp=segment.start_time,
                    suggestion_type="cut",
                    description=f"Pacing cut at {segment.start_time:.1f}s",
                    confidence=0.6,
                    reasoning="Natural break in script flow",
                    metadata={"break_duration": segment.start_time - segments[i-1].end_time}
                ))
        
        return suggestions

    def _parse_script(self, script_content: str) -> List[ScriptSegment]:
        """Parse script content into timed segments"""
        segments = []
        lines = script_content.split('\n')
        
        current_time = 0.0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Estimate duration (2-3 seconds per line)
            duration = 2.5 if len(line) > 50 else 2.0
            
            # Analyze emotion
            emotion, intensity = self._analyze_emotion(line)
            
            # Extract speaker if present
            speaker = self._extract_speaker(line)
            
            segments.append(ScriptSegment(
                start_time=current_time,
                end_time=current_time + duration,
                text=line,
                emotion=emotion,
                intensity=intensity,
                speaker=speaker
            ))
            
            current_time += duration
        
        return segments

    def _analyze_emotion(self, text: str) -> tuple[str, float]:
        """Analyze emotion in text"""
        text_lower = text.lower()
        
        max_intensity = 0.0
        detected_emotion = ""
        
        for emotion, keywords in self.emotion_keywords.items():
            intensity = sum(1 for keyword in keywords if keyword in text_lower)
            if intensity > max_intensity:
                max_intensity = intensity
                detected_emotion = emotion
        
        # Normalize intensity
        normalized_intensity = min(1.0, max_intensity / 3.0)
        
        return detected_emotion, normalized_intensity

    def _extract_speaker(self, text: str) -> str:
        """Extract speaker name from script line"""
        # Look for patterns like "SPEAKER: text" or "SPEAKER (description): text"
        match = re.match(r'^([A-Z][A-Z\s]+):', text)
        if match:
            return match.group(1).strip()
        return ""

    def get_script_statistics(self, script_content: str) -> Dict[str, Any]:
        """Get statistics about the script"""
        segments = self._parse_script(script_content)
        
        total_duration = sum(seg.end_time - seg.start_time for seg in segments)
        emotion_counts = {}
        
        for segment in segments:
            if segment.emotion:
                emotion_counts[segment.emotion] = emotion_counts.get(segment.emotion, 0) + 1
        
        return {
            'total_duration': total_duration,
            'segment_count': len(segments),
            'emotion_distribution': emotion_counts,
            'average_intensity': sum(seg.intensity for seg in segments) / len(segments) if segments else 0
        }
