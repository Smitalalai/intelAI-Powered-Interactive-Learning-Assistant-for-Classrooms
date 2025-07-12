"""
Personalization Service for Adaptive Learning
Implements learning style detection, difficulty adaptation, and progress tracking
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from configs.config import PERSONALIZATION_CONFIG, CONTENT_GENERATION_CONFIG

logger = logging.getLogger(__name__)


class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class StudentProfile:
    """Student learning profile with preferences and progress tracking"""
    student_id: str
    learning_style: LearningStyle = LearningStyle.READING
    difficulty_preference: DifficultyLevel = DifficultyLevel.MEDIUM
    subject_strengths: Dict[str, float] = field(default_factory=dict)
    interaction_patterns: Dict[str, int] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    session_count: int = 0
    total_time_spent: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class LearningSession:
    """Extended learning session with personalization tracking"""
    session_id: str
    student_id: str
    start_time: float
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    content_preferences: Dict[str, int] = field(default_factory=dict)


class PersonalizationService:
    """Service for providing personalized learning experiences"""
    
    def __init__(self):
        self.student_profiles: Dict[str, StudentProfile] = {}
        self.active_sessions: Dict[str, LearningSession] = {}
        self.learning_analytics = LearningAnalytics()
        
        logger.info("Personalization Service initialized")
    
    def create_student_profile(self, student_id: str, initial_data: Dict[str, Any] = None) -> StudentProfile:
        """Create a new student profile"""
        profile = StudentProfile(student_id=student_id)
        
        if initial_data:
            if "grade_level" in initial_data:
                profile.difficulty_preference = self._map_grade_to_difficulty(initial_data["grade_level"])
            if "preferred_subjects" in initial_data:
                for subject in initial_data["preferred_subjects"]:
                    profile.subject_strengths[subject] = 0.7  # Initial moderate strength
        
        self.student_profiles[student_id] = profile
        logger.info(f"Created profile for student: {student_id}")
        return profile
    
    def get_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        """Get existing student profile"""
        return self.student_profiles.get(student_id)
    
    def start_personalized_session(self, student_id: str, session_id: str) -> LearningSession:
        """Start a new personalized learning session"""
        session = LearningSession(
            session_id=session_id,
            student_id=student_id,
            start_time=time.time()
        )
        
        self.active_sessions[session_id] = session
        
        # Update student profile session count
        if student_id in self.student_profiles:
            self.student_profiles[student_id].session_count += 1
        
        logger.info(f"Started personalized session: {session_id} for student: {student_id}")
        return session
    
    def adapt_content_difficulty(self, student_id: str, content_type: str, base_content: str) -> Dict[str, Any]:
        """Adapt content difficulty based on student profile"""
        profile = self.get_student_profile(student_id)
        if not profile:
            return {"content": base_content, "difficulty": "medium"}
        
        # Determine appropriate difficulty
        current_difficulty = profile.difficulty_preference
        subject_strength = profile.subject_strengths.get(content_type, 0.5)
        
        # Adjust difficulty based on performance
        if subject_strength > 0.8:
            target_difficulty = DifficultyLevel.HARD
        elif subject_strength < 0.3:
            target_difficulty = DifficultyLevel.EASY
        else:
            target_difficulty = current_difficulty
        
        # Generate adapted content
        adapted_content = self._adapt_content_for_difficulty(base_content, target_difficulty)
        
        return {
            "content": adapted_content,
            "difficulty": target_difficulty.value,
            "reasoning": f"Adapted based on {subject_strength:.2f} strength in {content_type}"
        }
    
    def detect_learning_style(self, student_id: str) -> LearningStyle:
        """Detect student's preferred learning style based on interaction patterns"""
        profile = self.get_student_profile(student_id)
        if not profile or profile.session_count < 3:
            return LearningStyle.READING  # Default
        
        style_scores = {}
        config = PERSONALIZATION_CONFIG["learning_style_detection"]
        
        for style, style_config in config.items():
            score = 0.0
            for indicator in style_config["indicators"]:
                interaction_count = profile.interaction_patterns.get(indicator, 0)
                score += interaction_count * style_config["weight"]
            style_scores[style] = score
        
        # Find the style with highest score
        detected_style = max(style_scores.items(), key=lambda x: x[1])[0]
        profile.learning_style = LearningStyle(detected_style)
        
        logger.info(f"Detected learning style for {student_id}: {detected_style}")
        return profile.learning_style
    
    def generate_personalized_questions(self, student_id: str, topic: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate personalized quiz questions based on student profile"""
        profile = self.get_student_profile(student_id)
        learning_style = profile.learning_style if profile else LearningStyle.READING
        difficulty = profile.difficulty_preference if profile else DifficultyLevel.MEDIUM
        
        questions = []
        question_types = self._get_preferred_question_types(learning_style)
        
        for i in range(count):
            question = {
                "id": f"q_{i+1}",
                "topic": topic,
                "difficulty": difficulty.value,
                "type": question_types[i % len(question_types)],
                "learning_style": learning_style.value,
                "content": self._generate_question_content(topic, difficulty, learning_style)
            }
            questions.append(question)
        
        return questions
    
    def track_interaction(self, session_id: str, interaction_type: str, content_data: Dict[str, Any]) -> None:
        """Track student interaction for personalization"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        interaction = {
            "timestamp": time.time(),
            "type": interaction_type,
            "data": content_data
        }
        
        session.interactions.append(interaction)
        
        # Update student profile
        profile = self.get_student_profile(session.student_id)
        if profile:
            profile.interaction_patterns[interaction_type] = profile.interaction_patterns.get(interaction_type, 0) + 1
            profile.last_updated = time.time()
    
    def generate_study_plan(self, student_id: str, subject: str, duration_days: int = 7) -> Dict[str, Any]:
        """Generate personalized study plan"""
        profile = self.get_student_profile(student_id)
        if not profile:
            return self._default_study_plan(subject, duration_days)
        
        # Analyze strengths and weaknesses
        subject_strength = profile.subject_strengths.get(subject, 0.5)
        learning_style = profile.learning_style
        
        study_plan = {
            "student_id": student_id,
            "subject": subject,
            "duration_days": duration_days,
            "learning_style": learning_style.value,
            "current_level": subject_strength,
            "daily_activities": []
        }
        
        # Generate daily activities based on learning style and performance
        for day in range(1, duration_days + 1):
            activities = self._generate_daily_activities(subject, learning_style, subject_strength, day)
            study_plan["daily_activities"].append({
                "day": day,
                "activities": activities,
                "estimated_time": sum(activity["duration_minutes"] for activity in activities)
            })
        
        return study_plan
    
    def _map_grade_to_difficulty(self, grade_level: str) -> DifficultyLevel:
        """Map grade level to difficulty level"""
        grade_mapping = {
            "Elementary": DifficultyLevel.EASY,
            "Middle School": DifficultyLevel.MEDIUM,
            "High School": DifficultyLevel.HARD,
            "College": DifficultyLevel.HARD,
            "Adult Learning": DifficultyLevel.MEDIUM
        }
        return grade_mapping.get(grade_level, DifficultyLevel.MEDIUM)
    
    def _adapt_content_for_difficulty(self, content: str, difficulty: DifficultyLevel) -> str:
        """Adapt content based on difficulty level"""
        config = PERSONALIZATION_CONFIG["difficulty_adaptation"][difficulty.value]
        
        # This is a simplified adaptation - in production, use more sophisticated NLP
        if difficulty == DifficultyLevel.EASY:
            adapted = f"[SIMPLIFIED] {content}"
        elif difficulty == DifficultyLevel.HARD:
            adapted = f"[ADVANCED] {content}"
        else:
            adapted = content
        
        return adapted
    
    def _get_preferred_question_types(self, learning_style: LearningStyle) -> List[str]:
        """Get preferred question types based on learning style"""
        style_preferences = {
            LearningStyle.VISUAL: ["diagram_analysis", "image_based", "multiple_choice"],
            LearningStyle.AUDITORY: ["audio_response", "verbal_explanation", "discussion"],
            LearningStyle.KINESTHETIC: ["interactive", "step_by_step", "hands_on"],
            LearningStyle.READING: ["text_analysis", "essay", "short_answer"]
        }
        return style_preferences.get(learning_style, ["multiple_choice", "short_answer"])
    
    def _generate_question_content(self, topic: str, difficulty: DifficultyLevel, learning_style: LearningStyle) -> Dict[str, Any]:
        """Generate question content based on parameters"""
        return {
            "question": f"Sample {difficulty.value} question about {topic} for {learning_style.value} learner",
            "options": ["Option A", "Option B", "Option C", "Option D"] if learning_style in [LearningStyle.VISUAL, LearningStyle.READING] else [],
            "explanation": f"This question tests understanding of {topic} at {difficulty.value} level"
        }
    
    def _generate_daily_activities(self, subject: str, learning_style: LearningStyle, strength_level: float, day: int) -> List[Dict[str, Any]]:
        """Generate daily study activities"""
        activities = []
        
        # Base activities adjusted for learning style
        if learning_style == LearningStyle.VISUAL:
            activities.extend([
                {"type": "diagram_study", "duration_minutes": 20, "description": f"Visual study of {subject} concepts"},
                {"type": "infographic_review", "duration_minutes": 15, "description": "Review key concepts through infographics"}
            ])
        elif learning_style == LearningStyle.AUDITORY:
            activities.extend([
                {"type": "audio_lecture", "duration_minutes": 25, "description": f"Listen to {subject} explanations"},
                {"type": "discussion_simulation", "duration_minutes": 15, "description": "Practice explaining concepts aloud"}
            ])
        elif learning_style == LearningStyle.KINESTHETIC:
            activities.extend([
                {"type": "interactive_exercise", "duration_minutes": 30, "description": f"Hands-on {subject} activities"},
                {"type": "practice_problems", "duration_minutes": 20, "description": "Solve practice problems step-by-step"}
            ])
        else:  # READING
            activities.extend([
                {"type": "text_study", "duration_minutes": 25, "description": f"Read {subject} materials"},
                {"type": "note_taking", "duration_minutes": 15, "description": "Create study notes and summaries"}
            ])
        
        # Add review activity every few days
        if day % 3 == 0:
            activities.append({
                "type": "review_quiz", 
                "duration_minutes": 10, 
                "description": "Quick review quiz of previous topics"
            })
        
        return activities
    
    def _default_study_plan(self, subject: str, duration_days: int) -> Dict[str, Any]:
        """Generate default study plan when no profile exists"""
        return {
            "subject": subject,
            "duration_days": duration_days,
            "learning_style": "balanced",
            "daily_activities": [
                {
                    "day": day,
                    "activities": [
                        {"type": "study", "duration_minutes": 30, "description": f"Study {subject}"},
                        {"type": "practice", "duration_minutes": 20, "description": "Practice problems"}
                    ]
                } for day in range(1, duration_days + 1)
            ]
        }


class LearningAnalytics:
    """Analytics for learning patterns and performance"""
    
    def __init__(self):
        self.interaction_history = defaultdict(list)
        self.performance_trends = defaultdict(deque)
        
    def analyze_learning_patterns(self, student_id: str, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning patterns from session data"""
        patterns = {
            "interaction_frequency": self._calculate_interaction_frequency(session_data),
            "content_preferences": self._analyze_content_preferences(session_data),
            "performance_trend": self._calculate_performance_trend(student_id),
            "engagement_score": self._calculate_engagement_score(session_data)
        }
        
        return patterns
    
    def _calculate_interaction_frequency(self, session_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate how frequently different interaction types occur"""
        interactions = defaultdict(int)
        total = len(session_data)
        
        for interaction in session_data:
            interaction_type = interaction.get("type", "unknown")
            interactions[interaction_type] += 1
        
        return {k: v/total for k, v in interactions.items()} if total > 0 else {}
    
    def _analyze_content_preferences(self, session_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze what types of content the student prefers"""
        content_types = defaultdict(int)
        
        for interaction in session_data:
            if "content_type" in interaction.get("data", {}):
                content_type = interaction["data"]["content_type"]
                content_types[content_type] += 1
        
        total = sum(content_types.values())
        return {k: v/total for k, v in content_types.items()} if total > 0 else {}
    
    def _calculate_performance_trend(self, student_id: str) -> str:
        """Calculate if performance is improving, declining, or stable"""
        if student_id not in self.performance_trends:
            return "insufficient_data"
        
        recent_scores = list(self.performance_trends[student_id])[-5:]  # Last 5 sessions
        
        if len(recent_scores) < 3:
            return "insufficient_data"
        
        # Simple linear trend analysis
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_engagement_score(self, session_data: List[Dict[str, Any]]) -> float:
        """Calculate engagement score based on interaction patterns"""
        if not session_data:
            return 0.0
        
        # Factors for engagement calculation
        interaction_count = len(session_data)
        session_duration = 0
        
        if session_data:
            first_interaction = session_data[0].get("timestamp", 0)
            last_interaction = session_data[-1].get("timestamp", 0)
            session_duration = max(last_interaction - first_interaction, 1)
        
        # Calculate engagement based on interactions per minute
        interactions_per_minute = interaction_count / (session_duration / 60) if session_duration > 0 else 0
        
        # Normalize to 0-1 scale (assuming 10 interactions per minute is maximum engagement)
        engagement_score = min(interactions_per_minute / 10, 1.0)
        
        return engagement_score
