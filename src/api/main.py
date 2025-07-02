"""
FastAPI backend for the AI-Powered Interactive Learning Assistant
"""
import logging
import io
import time
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ..services.learning_assistant import LearningAssistantService, InteractionMode
from configs.config import API_CONFIG, UI_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
learning_service: Optional[LearningAssistantService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global learning_service
    
    # Startup
    logger.info("Starting AI Learning Assistant API")
    try:
        learning_service = LearningAssistantService()
        logger.info("Learning service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize learning service: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Learning Assistant API")
    if learning_service:
        learning_service.shutdown()


# Pydantic models for request/response
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    student_id: Optional[str] = None  # For personalization
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    interaction_mode: str = "text_only"
    preferred_subjects: Optional[List[str]] = None


class PersonalizedSessionRequest(BaseModel):
    student_id: str
    subject: str
    grade_level: str
    interaction_mode: str = "multimodal"
    learning_preferences: Optional[Dict[str, Any]] = None


class QuestionRequest(BaseModel):
    session_id: str
    question: str
    context: str = ""
    mode: str = "educational"


class LessonPlanRequest(BaseModel):
    session_id: str
    topic: str
    duration_minutes: int = 50
    learning_objectives: Optional[List[str]] = None


class StudyGuideRequest(BaseModel):
    session_id: str
    topic: str
    difficulty_level: str = "medium"


class QuizGenerationRequest(BaseModel):
    session_id: str
    topic: str
    num_questions: int = 10
    question_types: Optional[List[str]] = None


class StudyPlanRequest(BaseModel):
    session_id: str
    subject: str
    duration_days: int = 7
    goals: Optional[List[str]] = None


class AnalyticsRequest(BaseModel):
    session_id: str
    include_personalization: bool = True


class SummarizationRequest(BaseModel):
    session_id: str
    content: str
    summary_type: str = "general"


class LessonSummaryRequest(BaseModel):
    session_id: str
    title: str
    content: str


class SpeechRequest(BaseModel):
    session_id: str
    response_mode: str = "text"


class TextToSpeechRequest(BaseModel):
    text: str
    speed: float = 1.0
    pitch: float = 1.0
    content_type: str = "general"


# Create FastAPI app
app = FastAPI(
    title="AI-Powered Interactive Learning Assistant",
    description="An intelligent classroom companion for enhanced learning experiences",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_learning_service() -> LearningAssistantService:
    """Dependency to get the learning service"""
    if learning_service is None:
        raise HTTPException(status_code=503, detail="Learning service not available")
    return learning_service


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI-Powered Interactive Learning Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        service = get_learning_service()
        status = service.model_manager.get_models_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Session Management Endpoints
@app.post("/sessions")
async def create_session(
    request: SessionCreateRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Create a new learning session"""
    try:
        session_id = service.create_session({
            "user_id": request.user_id,
            "student_id": request.student_id,
            "subject": request.subject,
            "grade_level": request.grade_level,
            "interaction_mode": request.interaction_mode,
            "preferred_subjects": request.preferred_subjects
        })
        
        return {
            "session_id": session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/personalized")
async def create_personalized_session(
    request: PersonalizedSessionRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Create a new personalized learning session"""
    try:
        session_id = service.create_personalized_session({
            "student_id": request.student_id,
            "subject": request.subject,
            "grade_level": request.grade_level,
            "interaction_mode": request.interaction_mode,
            "learning_preferences": request.learning_preferences
        })
        
        return {
            "session_id": session_id,
            "message": "Personalized session created successfully",
            "personalization_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get session information"""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "subject": session.subject,
        "grade_level": session.grade_level,
        "interaction_mode": session.interaction_mode.value,
        "history_count": len(session.context_history)
    }


# Core AI Endpoints
@app.post("/question")
async def ask_question(
    request: QuestionRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Process a student question"""
    try:
        result = service.process_question(
            session_id=request.session_id,
            question=request.question,
            context=request.context,
            mode=request.mode
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_content(
    request: SummarizationRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Summarize educational content"""
    try:
        result = service.summarize_content(
            session_id=request.session_id,
            content=request.content,
            summary_type=request.summary_type
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech/upload")
async def process_speech_upload(
    session_id: str = Form(...),
    response_mode: str = Form("text"),
    audio_file: UploadFile = File(...),
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Process uploaded audio file"""
    try:
        # Validate file format
        if not audio_file.filename.lower().endswith(tuple(UI_CONFIG["supported_audio_formats"])):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported: {UI_CONFIG['supported_audio_formats']}"
            )
        
        # Check file size
        audio_bytes = await audio_file.read()
        if len(audio_bytes) > UI_CONFIG["max_file_size"]:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Process audio
        result = service.process_speech(
            session_id=session_id,
            audio_input=audio_bytes,
            response_mode=response_mode
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech/text-to-speech")
async def text_to_speech(
    request: TextToSpeechRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Convert text to speech"""
    try:
        tts_model = service.model_manager.get_model("tts")
        if not tts_model:
            raise HTTPException(status_code=503, detail="Text-to-speech service not available")
        
        result = tts_model.speak_educational_content({
            "text": request.text,
            "type": request.content_type,
            "speed": request.speed,
            "pitch": request.pitch
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speech/text-to-speech/audio")
async def get_speech_audio(
    text: str,
    speed: float = 1.0,
    pitch: float = 1.0,
    content_type: str = "general",
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get speech audio as streaming response"""
    try:
        tts_model = service.model_manager.get_model("tts")
        if not tts_model:
            raise HTTPException(status_code=503, detail="Text-to-speech service not available")
        
        result = tts_model.speak_educational_content({
            "text": text,
            "type": content_type,
            "speed": speed,
            "pitch": pitch
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Get audio bytes
        audio_bytes = tts_model.get_audio_bytes(result, format="wav")
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Features
@app.post("/lesson/summary")
async def generate_lesson_summary(
    request: LessonSummaryRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate comprehensive lesson summary"""
    try:
        result = service.generate_lesson_summary(
            session_id=request.session_id,
            lesson_data={
                "title": request.title,
                "content": request.content
            }
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Image Processing Endpoints
@app.post("/image/caption")
async def generate_image_caption(
    session_id: str = Form(...),
    context: str = Form(""),
    analysis_type: str = Form("educational"),
    file: UploadFile = File(...),
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate caption or educational analysis for an uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Process image
        result = service.process_image(session_id, image_bytes, context, analysis_type)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image/analyze")
async def analyze_educational_image(
    request: Dict[str, Any],
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Analyze image for educational content"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        image_data = request.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="image data is required")
        
        context = request.get("context", "")
        
        # Process image (assuming base64 encoded)
        result = service.process_image(session_id, image_data, context, "educational")
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Multimodal Processing Endpoints
@app.post("/multimodal/process")
async def process_multimodal_input(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Process multimodal input combining text, audio, and images"""
    try:
        inputs = {"context": context or ""}
        
        # Add text input
        if text:
            inputs["text"] = text
        
        # Add audio input
        if audio_file:
            if not audio_file.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Audio file must be an audio format")
            audio_bytes = await audio_file.read()
            inputs["audio"] = audio_bytes
        
        # Add image input
        if image_file:
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Image file must be an image format")
            image_bytes = await image_file.read()
            inputs["image"] = image_bytes
        
        # Validate that at least one input is provided
        if not any(key in inputs for key in ["text", "audio", "image"]):
            raise HTTPException(status_code=400, detail="At least one input (text, audio, or image) must be provided")
        
        # Process multimodal input
        result = service.process_multimodal_input(session_id, inputs)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multimodal/classroom")
async def process_classroom_interaction(
    request: Dict[str, Any],
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Process classroom interaction with multiple students and modalities"""
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        student_inputs = request.get("student_inputs", [])
        classroom_context = request.get("classroom_context", "")
        
        results = {"student_responses": [], "classroom_summary": ""}
        
        # Process each student's input
        for i, student_input in enumerate(student_inputs):
            student_input["context"] = classroom_context
            student_result = service.process_multimodal_input(session_id, student_input)
            student_result["student_id"] = student_input.get("student_id", f"student_{i+1}")
            results["student_responses"].append(student_result)
        
        # Generate classroom summary if multiple students
        if len(student_inputs) > 1:
            summary_text = f"Classroom interaction with {len(student_inputs)} students. "
            summary_text += "Multiple modalities processed including text, speech, and visual content."
            results["classroom_summary"] = summary_text
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Content Generation Endpoints
@app.post("/generate/lesson-plan")
async def generate_lesson_plan(
    request: LessonPlanRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate a comprehensive lesson plan"""
    try:
        result = service.generate_lesson_plan(
            request.session_id,
            request.topic,
            request.duration_minutes
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/study-guide")
async def generate_study_guide(
    request: StudyGuideRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate a personalized study guide"""
    try:
        result = service.generate_study_guide(
            request.session_id,
            request.topic
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/quiz")
async def generate_quiz(
    request: QuizGenerationRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate a personalized quiz"""
    try:
        result = service.generate_personalized_quiz(
            request.session_id,
            request.topic,
            request.num_questions
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/study-plan")
async def generate_study_plan(
    request: StudyPlanRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Generate a personalized study plan"""
    try:
        result = service.generate_study_plan(
            request.session_id,
            request.subject,
            request.duration_days
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced Question Answering
@app.post("/questions/personalized")
async def ask_personalized_question(
    request: QuestionRequest,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Ask a question with personalized response"""
    try:
        result = service.process_personalized_question(
            request.session_id,
            request.question,
            request.context
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Analytics and Insights
@app.get("/analytics/{session_id}")
async def get_learning_analytics(
    session_id: str,
    include_personalization: bool = True,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get comprehensive learning analytics"""
    try:
        result = service.get_learning_analytics(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/metrics")
async def get_performance_metrics(
    session_id: Optional[str] = None,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get system and session performance metrics"""
    try:
        result = service.get_performance_metrics(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/profile")
async def get_student_profile(
    session_id: str,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get student learning profile"""
    try:
        session = service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        student_id = session.personalization_data.get("student_id")
        if not student_id:
            raise HTTPException(status_code=400, detail="No student profile associated with session")
        
        profile = service.personalization_service.get_student_profile(student_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Student profile not found")
        
        return {
            "student_id": profile.student_id,
            "learning_style": profile.learning_style.value,
            "difficulty_preference": profile.difficulty_preference.value,
            "subject_strengths": profile.subject_strengths,
            "session_count": profile.session_count,
            "total_time_spent": profile.total_time_spent
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Performance and Analytics
@app.get("/performance")
async def get_performance_metrics(
    session_id: Optional[str] = None,
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get performance metrics"""
    try:
        return service.get_performance_report(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_models_status(
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Get status of all AI models"""
    try:
        return service.model_manager.get_models_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/benchmark")
async def benchmark_models(
    service: LearningAssistantService = Depends(get_learning_service)
):
    """Benchmark all models performance"""
    try:
        # Sample test inputs
        test_inputs = {
            "qa": {"question": "What is machine learning?", "context": "Machine learning is a branch of artificial intelligence."},
            "summarization": {"text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."},
            "speech_recognition": "test_audio_sample.wav",  # Would need actual audio
            "tts": {"text": "Hello, this is a test of the text-to-speech system."}
        }
        
        results = service.model_manager.benchmark_all_models(test_inputs)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        log_level=API_CONFIG["log_level"]
    )


if __name__ == "__main__":
    run_server()
