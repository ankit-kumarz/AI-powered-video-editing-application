"""
Script analysis functionality for AI editing suggestions.
Handles script text analysis, narrative structure detection, and script-based editing suggestions.
"""
from typing import List, Dict, Any, Optional
from .models import EditingSuggestion


class ScriptAnalyzer:
    """Handles script analysis and script-based editing suggestions"""
    
    def __init__(self):
        pass

    def generate_script_based_suggestions(self, script_analysis: Dict) -> List[EditingSuggestion]:
        """Generate editing suggestions based on script analysis"""
        suggestions = []
        
        # Extract script-based suggestions
        script_suggestions = script_analysis.get('editing_suggestions', [])
        
        for script_sug in script_suggestions:
            suggestions.append(EditingSuggestion(
                timestamp=script_sug.get('timestamp', 0),
                suggestion_type=script_sug.get('suggestion_type', 'cut'),
                confidence=script_sug.get('confidence', 0.5),
                reason=script_sug.get('reason', 'Script-based suggestion'),
                transition_type=script_sug.get('transition_type'),
                metadata={'source': 'script_analysis'}
            ))
        
        return suggestions

    def generate_script_based_suggestions_from_text(self, script_content: str) -> List[EditingSuggestion]:
        """Generate advanced editing suggestions based on script text content - PROFESSIONAL EDITOR OPTIMIZED"""
        suggestions = []
        
        try:
            # Advanced script analysis for professional editing
            lines = script_content.split('\n')
            current_time = 0.0
            
            # Track narrative structure
            narrative_phases = []
            emotional_arc = []
            speaker_timeline = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Estimate duration based on text length and complexity
                words = line.split()
                word_count = len(words)
                duration = max(2.0, word_count * 0.3)  # More realistic timing
                
                # 1. NARRATIVE STRUCTURE ANALYSIS
                narrative_phase = self._analyze_narrative_phase(line, i, len(lines))
                if narrative_phase:
                    narrative_phases.append({
                        'timestamp': current_time,
                        'phase': narrative_phase,
                        'line': line[:100]
                    })
                
                # 2. EMOTIONAL CONTENT ANALYSIS
                emotional_analysis = self._analyze_emotional_content(line)
                if emotional_analysis['intensity'] > 0.5:
                    emotional_arc.append({
                        'timestamp': current_time,
                        'emotion': emotional_analysis['emotion'],
                        'intensity': emotional_analysis['intensity'],
                        'line': line[:100]
                    })
                
                # 3. SPEAKER AND DIALOGUE ANALYSIS
                speaker_analysis = self._analyze_speaker_content(line)
                if speaker_analysis['is_speaker']:
                    speaker_timeline.append({
                        'timestamp': current_time,
                        'speaker': speaker_analysis['speaker'],
                        'dialogue_type': speaker_analysis['dialogue_type'],
                        'line': line[:100]
                    })
                
                # 4. TRANSITION WORD ANALYSIS - Enhanced
                transition_analysis = self._analyze_transition_words(line)
                if transition_analysis['has_transition']:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='transition',
                        confidence=transition_analysis['confidence'],
                        reason=f"Transition word detected: {transition_analysis['word']}",
                        description=f"Script transition detected: '{transition_analysis['word']}'. Use {transition_analysis['transition_type']} for smooth narrative flow.",
                        reasoning=f"Transition word '{transition_analysis['word']}' indicates narrative shift - {transition_analysis['transition_type']} maintains story continuity",
                        transition_type=transition_analysis['transition_type'],
                        metadata={
                            'source': 'script_analysis',
                            'transition_word': transition_analysis['word'],
                            'transition_category': transition_analysis['category'],
                            'editor_tip': transition_analysis['editor_tip']
                        }
                    ))
                
                # 5. EMOTIONAL MOMENT SUGGESTIONS
                if emotional_analysis['intensity'] > 0.7:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='emphasis',
                        confidence=emotional_analysis['intensity'],
                        reason=f"High emotional content: {emotional_analysis['emotion']}",
                        description=f"Strong {emotional_analysis['emotion']} emotion detected. Consider dramatic emphasis or close-up.",
                        reasoning=f"High emotional intensity requires visual emphasis - dramatic treatment enhances impact",
                        transition_type=None,
                        metadata={
                            'source': 'script_analysis',
                            'emotion': emotional_analysis['emotion'],
                            'intensity': emotional_analysis['intensity'],
                            'suggested_shot': emotional_analysis['suggested_shot'],
                            'editor_tip': emotional_analysis['editor_tip']
                        }
                    ))
                
                # 6. SPEAKER CHANGE SUGGESTIONS
                if speaker_analysis['is_speaker'] and speaker_analysis['dialogue_type'] != 'monologue':
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='cut',
                        confidence=0.8,
                        reason=f"Speaker change: {speaker_analysis['speaker']}",
                        description=f"New speaker '{speaker_analysis['speaker']}' detected. Cut to show speaker or reaction.",
                        reasoning=f"Speaker changes are natural cut points - visual transition maintains dialogue flow",
                        transition_type='cut',
                        metadata={
                            'source': 'script_analysis',
                            'speaker': speaker_analysis['speaker'],
                            'dialogue_type': speaker_analysis['dialogue_type'],
                            'suggested_shot': 'close_up_or_reaction',
                            'editor_tip': 'Cut on natural speech rhythm or gesture'
                        }
                    ))
                
                # 7. ACTION AND MOVEMENT SUGGESTIONS
                action_analysis = self._analyze_action_content(line)
                if action_analysis['has_action']:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='emphasis',
                        confidence=action_analysis['confidence'],
                        reason=f"Action content: {action_analysis['action_type']}",
                        description=f"Action detected: {action_analysis['action_type']}. Consider dynamic camera movement or emphasis.",
                        reasoning=f"Action content benefits from dynamic editing - movement enhances viewer engagement",
                        transition_type=None,
                        metadata={
                            'source': 'script_analysis',
                            'action_type': action_analysis['action_type'],
                            'suggested_shot': action_analysis['suggested_shot'],
                            'editor_tip': action_analysis['editor_tip']
                        }
                    ))
                
                current_time += duration
            
            # 8. NARRATIVE STRUCTURE SUGGESTIONS
            structure_suggestions = self._generate_narrative_structure_suggestions(narrative_phases, current_time)
            suggestions.extend(structure_suggestions)
            
            # 9. EMOTIONAL ARC SUGGESTIONS
            arc_suggestions = self._generate_emotional_arc_suggestions(emotional_arc, current_time)
            suggestions.extend(arc_suggestions)
            
        except Exception as e:
            print(f"Advanced script analysis failed: {e}")
        
        return suggestions

    def _analyze_narrative_phase(self, line: str, line_index: int, total_lines: int) -> Optional[str]:
        """Analyze narrative phase based on content and position"""
        line_lower = line.lower()
        
        # Opening phase indicators
        opening_words = ['begin', 'start', 'first', 'introduction', 'meet', 'welcome']
        if any(word in line_lower for word in opening_words) or line_index < total_lines * 0.1:
            return 'opening'
        
        # Development phase indicators
        development_words = ['develop', 'grow', 'learn', 'discover', 'explore', 'build']
        if any(word in line_lower for word in development_words) or (total_lines * 0.1 <= line_index < total_lines * 0.8):
            return 'development'
        
        # Climax phase indicators
        climax_words = ['climax', 'peak', 'moment', 'finally', 'suddenly', 'dramatic', 'intense']
        if any(word in line_lower for word in climax_words) or (total_lines * 0.7 <= line_index < total_lines * 0.9):
            return 'climax'
        
        # Resolution phase indicators
        resolution_words = ['end', 'conclude', 'finally', 'result', 'outcome', 'resolution']
        if any(word in line_lower for word in resolution_words) or line_index >= total_lines * 0.9:
            return 'resolution'
        
        return None

    def _analyze_emotional_content(self, line: str) -> Dict[str, Any]:
        """Analyze emotional content for editing suggestions"""
        line_lower = line.lower()
        
        # Emotional word categories
        emotions = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'amazing', 'wonderful', 'fantastic'],
            'sadness': ['sad', 'depressed', 'melancholy', 'sorrow', 'grief', 'heartbroken'],
            'anger': ['angry', 'furious', 'rage', 'outraged', 'furious', 'mad'],
            'fear': ['afraid', 'scared', 'terrified', 'fearful', 'anxious', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'love': ['love', 'adore', 'passion', 'romantic', 'affection', 'tender'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'emotional', 'moving']
        }
        
        max_intensity = 0.0
        detected_emotion = 'neutral'
        suggested_shot = 'medium_shot'
        editor_tip = 'Standard emotional treatment'
        
        for emotion, words in emotions.items():
            intensity = sum(1 for word in words if word in line_lower) / len(words)
            if intensity > max_intensity:
                max_intensity = intensity
                detected_emotion = emotion
                
                # Suggest appropriate shots based on emotion
                if emotion in ['joy', 'love']:
                    suggested_shot = 'close_up'
                    editor_tip = 'Use close-up to capture emotional expression'
                elif emotion in ['anger', 'fear']:
                    suggested_shot = 'dramatic_angle'
                    editor_tip = 'Use dramatic angles for intense emotions'
                elif emotion == 'dramatic':
                    suggested_shot = 'hero_shot'
                    editor_tip = 'Use hero shot for maximum dramatic impact'
        
        return {
            'emotion': detected_emotion,
            'intensity': max_intensity,
            'suggested_shot': suggested_shot,
            'editor_tip': editor_tip
        }

    def _analyze_speaker_content(self, line: str) -> Dict[str, Any]:
        """Analyze speaker and dialogue content"""
        # Check for speaker format (NAME: dialogue)
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            dialogue = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if speaker is in caps (typical script format)
            if speaker.isupper() and len(speaker) > 1:
                # Analyze dialogue type
                dialogue_type = 'dialogue'
                if len(dialogue) > 100:
                    dialogue_type = 'monologue'
                elif len(dialogue) < 10:
                    dialogue_type = 'reaction'
                
                return {
                    'is_speaker': True,
                    'speaker': speaker,
                    'dialogue_type': dialogue_type,
                    'dialogue_length': len(dialogue)
                }
        
        return {
            'is_speaker': False,
            'speaker': None,
            'dialogue_type': None,
            'dialogue_length': 0
        }

    def _analyze_transition_words(self, line: str) -> Dict[str, Any]:
        """Analyze transition words for editing suggestions"""
        line_lower = line.lower()
        
        # Categorized transition words
        transitions = {
            'temporal': {
                'words': ['meanwhile', 'later', 'earlier', 'before', 'after', 'then', 'now'],
                'transition_type': 'cross_dissolve',
                'category': 'time_shift',
                'editor_tip': 'Use cross dissolve for smooth time transitions'
            },
            'contrast': {
                'words': ['however', 'but', 'although', 'despite', 'nevertheless'],
                'transition_type': 'cut',
                'category': 'contrast',
                'editor_tip': 'Use hard cut for contrast emphasis'
            },
            'causal': {
                'words': ['therefore', 'thus', 'consequently', 'as a result', 'because'],
                'transition_type': 'cross_dissolve',
                'category': 'cause_effect',
                'editor_tip': 'Use cross dissolve to show cause-effect relationship'
            },
            'dramatic': {
                'words': ['suddenly', 'dramatically', 'shockingly', 'unexpectedly'],
                'transition_type': 'dramatic_cut',
                'category': 'drama',
                'editor_tip': 'Use dramatic cut for sudden revelations'
            }
        }
        
        for category, data in transitions.items():
            for word in data['words']:
                if word in line_lower:
                    return {
                        'has_transition': True,
                        'word': word,
                        'category': category,
                        'transition_type': data['transition_type'],
                        'confidence': 0.8,
                        'editor_tip': data['editor_tip']
                    }
        
        return {
            'has_transition': False,
            'word': None,
            'category': None,
            'transition_type': None,
            'confidence': 0.0,
            'editor_tip': None
        }

    def _analyze_action_content(self, line: str) -> Dict[str, Any]:
        """Analyze action content for dynamic editing suggestions"""
        line_lower = line.lower()
        
        # Action word categories
        actions = {
            'movement': ['walk', 'run', 'move', 'travel', 'go', 'come'],
            'physical': ['fight', 'dance', 'jump', 'fall', 'climb', 'throw'],
            'gesture': ['point', 'wave', 'nod', 'shake', 'smile', 'frown'],
            'interaction': ['touch', 'hold', 'push', 'pull', 'embrace', 'kiss']
        }
        
        detected_actions = []
        for action_type, words in actions.items():
            if any(word in line_lower for word in words):
                detected_actions.append(action_type)
        
        if detected_actions:
            action_type = detected_actions[0]
            confidence = min(0.9, len(detected_actions) * 0.3)
            
            # Suggest appropriate shots based on action
            suggested_shot = 'medium_shot'
            editor_tip = 'Standard action treatment'
            
            if action_type == 'movement':
                suggested_shot = 'tracking_shot'
                editor_tip = 'Use tracking shot to follow movement'
            elif action_type == 'physical':
                suggested_shot = 'wide_shot'
                editor_tip = 'Use wide shot to show full action'
            elif action_type == 'gesture':
                suggested_shot = 'close_up'
                editor_tip = 'Use close-up to capture gesture detail'
            elif action_type == 'interaction':
                suggested_shot = 'two_shot'
                editor_tip = 'Use two-shot to show interaction'
            
            return {
                'has_action': True,
                'action_type': action_type,
                'confidence': confidence,
                'suggested_shot': suggested_shot,
                'editor_tip': editor_tip
            }
        
        return {
            'has_action': False,
            'action_type': None,
            'confidence': 0.0,
            'suggested_shot': None,
            'editor_tip': None
        }

    def _generate_narrative_structure_suggestions(self, narrative_phases: List[Dict], total_duration: float) -> List[EditingSuggestion]:
        """Generate editing suggestions based on narrative structure"""
        suggestions = []
        
        for phase in narrative_phases:
            if phase['phase'] == 'opening':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='transition',
                    confidence=0.8,
                    reason="Narrative opening - establish story",
                    description="Story opening detected. Use establishing shot or smooth transition.",
                    reasoning="Opening phase requires clear story establishment - smooth transition sets tone",
                    transition_type='cross_dissolve',
                    metadata={
                        'narrative_phase': 'opening',
                        'editor_tip': 'Use establishing shot to set scene and tone'
                    }
                ))
            
            elif phase['phase'] == 'climax':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='emphasis',
                    confidence=0.9,
                    reason="Narrative climax - maximum impact",
                    description="Story climax detected. Use dramatic emphasis for maximum impact.",
                    reasoning="Climax requires maximum visual impact - dramatic treatment enhances tension",
                    transition_type=None,
                    metadata={
                        'narrative_phase': 'climax',
                        'editor_tip': 'Use dramatic angles and close-ups for climax impact'
                    }
                ))
            
            elif phase['phase'] == 'resolution':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='transition',
                    confidence=0.7,
                    reason="Narrative resolution - story conclusion",
                    description="Story resolution detected. Use gentle transition for conclusion.",
                    reasoning="Resolution phase requires gentle treatment - smooth transition provides closure",
                    transition_type='fade_to_black',
                    metadata={
                        'narrative_phase': 'resolution',
                        'editor_tip': 'Use gentle fade for story conclusion'
                    }
                ))
        
        return suggestions

    def _generate_emotional_arc_suggestions(self, emotional_arc: List[Dict], total_duration: float) -> List[EditingSuggestion]:
        """Generate editing suggestions based on emotional arc"""
        suggestions = []
        
        if len(emotional_arc) < 2:
            return suggestions
        
        # Find emotional peaks
        high_emotions = [e for e in emotional_arc if e['intensity'] > 0.8]
        
        for emotion in high_emotions:
            suggestions.append(EditingSuggestion(
                timestamp=emotion['timestamp'],
                suggestion_type='emphasis',
                confidence=emotion['intensity'],
                reason=f"Emotional peak: {emotion['emotion']}",
                description=f"Emotional peak detected: {emotion['emotion']}. Use dramatic emphasis.",
                reasoning=f"Emotional peaks require visual emphasis - dramatic treatment enhances impact",
                transition_type=None,
                metadata={
                    'emotion': emotion['emotion'],
                    'intensity': emotion['intensity'],
                    'editor_tip': f'Use dramatic treatment for {emotion["emotion"]} emotion'
                }
            ))
        
        return suggestions
