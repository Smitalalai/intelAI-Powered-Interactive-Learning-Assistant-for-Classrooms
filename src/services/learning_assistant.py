"""
Learning Assistant Service - Main orchestrator for AI models
Enhanced with personalization and content generation capabilities
"""
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..models.base_model import ModelManager
from ..models.question_answering import QuestionAnsweringModel
from ..models.summarization import SummarizationModel
from ..models.speech_recognition import SpeechRecognitionModel
from ..models.text_to_speech import TextToSpeechModel
from ..models.image_captioning import ImageCaptioningModel
from .personalization import PersonalizationService, StudentProfile, LearningStyle
from .content_generation import ContentGenerationService, ContentType
from configs.config import PERFORMANCE_THRESHOLDS, PERSONALIZATION_CONFIG

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Different modes of interaction with the learning assistant"""
    TEXT_ONLY = "text_only"
    VOICE_ONLY = "voice_only"
    MULTIMODAL = "multimodal"
    CLASSROOM = "classroom"


@dataclass
class LearningSession:
    """Represents a learning session with context and history"""
    session_id: str
    user_id: Optional[str] = None
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    interaction_mode: InteractionMode = InteractionMode.TEXT_ONLY
    context_history: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    personalization_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_history is None:
            self.context_history = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.personalization_data is None:
            self.personalization_data = {}


class PerformanceTracker:
    """Track performance metrics for AI operations"""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        self.total_operations = 0
        
    def track_operation(self, operation_name: str, processing_time: float) -> None:
        """Track an operation's performance"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(processing_time)
        self.operation_counts[operation_name] += 1
        self.total_operations += 1
        
        # Check performance thresholds
        threshold_key = f"{operation_name}_max_time"
        if threshold_key in PERFORMANCE_THRESHOLDS:
            max_time = PERFORMANCE_THRESHOLDS[threshold_key]
            if processing_time > max_time:
                logger.warning(f"{operation_name} took {processing_time:.2f}s, exceeding threshold of {max_time}s")
    
    def get_average_time(self, operation_name: str) -> float:
        """Get average processing time for an operation"""
        if operation_name not in self.operation_times:
            return 0.0
        
        times = self.operation_times[operation_name]
        return sum(times) / len(times) if times else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            "total_operations": self.total_operations,
            "operations": {}
        }
        
        for operation_name in self.operation_times:
            times = self.operation_times[operation_name]
            summary["operations"][operation_name] = {
                "count": self.operation_counts[operation_name],
                "average_time": self.get_average_time(operation_name),
                "min_time": min(times) if times else 0.0,
                "max_time": max(times) if times else 0.0,
                "total_time": sum(times)
            }
        
        return summary


class LearningAssistantService:
    """Main service class for the AI-powered learning assistant"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.sessions: Dict[str, LearningSession] = {}
        self.performance_tracker = PerformanceTracker()
        
        # Initialize new services
        self.personalization_service = PersonalizationService()
        self.content_generation_service = ContentGenerationService()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Enhanced Learning Assistant Service initialized with personalization and content generation")
    
    def _initialize_models(self) -> None:
        """Initialize and register all AI models"""
        try:
            # Register all models
            self.model_manager.register_model("qa", QuestionAnsweringModel())
            self.model_manager.register_model("summarization", SummarizationModel())
            self.model_manager.register_model("speech_recognition", SpeechRecognitionModel())
            self.model_manager.register_model("tts", TextToSpeechModel())
            self.model_manager.register_model("image_captioning", ImageCaptioningModel())
            
            # Load all models
            self.model_manager.load_all_models()
            
            # Connect content generation service with models
            self.content_generation_service.qa_model = self.model_manager.get_model("qa")
            self.content_generation_service.summarization_model = self.model_manager.get_model("summarization")
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def create_personalized_session(self, session_config: Dict[str, Any]) -> str:
        """Create a new personalized learning session"""
        session_id = f"session_{int(time.time())}"
        
        # Create or get student profile
        student_id = session_config.get("student_id")
        if student_id:
            profile = self.personalization_service.get_student_profile(student_id)
            if not profile:
                profile = self.personalization_service.create_student_profile(
                    student_id, session_config
                )
        
        session = LearningSession(
            session_id=session_id,
            user_id=session_config.get("user_id"),
            subject=session_config.get("subject"),
            grade_level=session_config.get("grade_level"),
            interaction_mode=InteractionMode(session_config.get("interaction_mode", "text_only")),
            personalization_data={
                "student_id": student_id,
                "learning_style": profile.learning_style.value if profile else "reading",
                "difficulty_preference": profile.difficulty_preference.value if profile else "medium"
            }
        )
        
        self.sessions[session_id] = session
        
        # Start personalized session tracking
        if student_id:
            self.personalization_service.start_personalized_session(student_id, session_id)
        
        logger.info(f"Created personalized learning session: {session_id}")
        
        return session_id

    def create_session(self, session_config: Dict[str, Any]) -> str:
        """Create a new learning session (backwards compatibility)"""
        return self.create_personalized_session(session_config)
    
    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """Get an existing learning session"""
        return self.sessions.get(session_id)
    
    def process_question(self, session_id: str, question: str, context: str = "", 
                        mode: str = "educational") -> Dict[str, Any]:
        """Process a student question with contextual understanding"""
        # Use personalized version for better results
        return self.process_personalized_question(session_id, question, context)
    
    def summarize_content(self, session_id: str, content: str, 
                         summary_type: str = "general") -> Dict[str, Any]:
        """Summarize educational content"""
        start_time = time.time()
        
        try:
            session = self.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get summarization model
            summarization_model = self.model_manager.get_model("summarization")
            if not summarization_model:
                return {"error": "Summarization model not available"}
            
            # Generate summary
            if summary_type in ["key_points", "review"]:
                result = summarization_model.summarize_lesson(content, summary_type)
            else:
                result = summarization_model.predict({"text": content})
            
            # Track performance
            processing_time = time.time() - start_time
            self.performance_tracker.track_operation("summarization", processing_time)
            
            # Update session history
            self._update_session_history(session, {
                "type": "summarization",
                "content_length": len(content),
                "summary": result.get("summary", ""),
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            result["processing_time"] = processing_time
            result["session_id"] = session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to summarize content: {str(e)}")
            return {"error": str(e)}
    
    def process_speech(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process speech input and return transcription"""
        start_time = time.time()
        
        try:
            session = self.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get speech recognition model
            speech_model = self.model_manager.get_model("speech_recognition")
            if not speech_model:
                return {"error": "Speech recognition model not available"}
            
            # Transcribe audio
            result = speech_model.transcribe(audio_data)
            
            # Track performance
            processing_time = time.time() - start_time
            self.performance_tracker.track_operation("speech_recognition", processing_time)
            
            # Update session history
            self._update_session_history(session, {
                "type": "speech_input",
                "transcription": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            result["processing_time"] = processing_time
            result["session_id"] = session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process speech: {str(e)}")
            return {"error": str(e)}
    
    def generate_speech(self, session_id: str, text: str) -> Dict[str, Any]:
        """Generate speech from text"""
        start_time = time.time()
        
        try:
            session = self.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get text-to-speech model
            tts_model = self.model_manager.get_model("tts")
            if not tts_model:
                return {"error": "Text-to-speech model not available"}
            
            # Generate speech
            result = tts_model.synthesize(text)
            
            # Track performance
            processing_time = time.time() - start_time
            self.performance_tracker.track_operation("text_to_speech", processing_time)
            
            # Update session history
            self._update_session_history(session, {
                "type": "speech_output",
                "text": text,
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            result["processing_time"] = processing_time
            result["session_id"] = session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate speech: {str(e)}")
            return {"error": str(e)}
    
    def analyze_image(self, session_id: str, image_data: bytes) -> Dict[str, Any]:
        """Analyze image and generate educational description"""
        start_time = time.time()
        
        try:
            session = self.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Get image captioning model
            image_model = self.model_manager.get_model("image_captioning")
            if not image_model:
                return {"error": "Image captioning model not available"}
            
            # Generate caption
            result = image_model.caption_image(image_data)
            
            # Track performance
            processing_time = time.time() - start_time
            self.performance_tracker.track_operation("image_captioning", processing_time)
            
            # Update session history
            self._update_session_history(session, {
                "type": "image_analysis",
                "caption": result.get("caption", ""),
                "confidence": result.get("confidence", 0.0),
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            result["processing_time"] = processing_time
            result["session_id"] = session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_metrics(self, session_id: str = None) -> Dict[str, Any]:
        """Get performance metrics for session or overall system"""
        try:
            metrics = {
                "system_metrics": self.performance_tracker.get_performance_summary(),
                "timestamp": time.time()
            }
            
            if session_id:
                session = self.get_session(session_id)
                if session:
                    metrics["session_metrics"] = {
                        "session_id": session_id,
                        "duration": time.time() - session.context_history[0]["timestamp"] if session.context_history else 0,
                        "interactions": len(session.context_history),
                        "performance_data": session.performance_metrics
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def _build_context(self, session: LearningSession, additional_context: str) -> str:
        """Build context string from session history and additional context"""
        return self._build_personalized_context(session, additional_context)
    
    def _update_session_history(self, session: LearningSession, interaction_data: Dict[str, Any]) -> None:
        """Update session history with new interaction"""
        session.context_history.append(interaction_data)
        
        # Keep only last 20 interactions to prevent memory issues
        if len(session.context_history) > 20:
            session.context_history = session.context_history[-20:]
        
        # Update performance metrics
        if "processing_time" in interaction_data:
            operation_type = interaction_data.get("type", "unknown")
            if operation_type not in session.performance_metrics:
                session.performance_metrics[operation_type] = []
            session.performance_metrics[operation_type].append(interaction_data["processing_time"])
