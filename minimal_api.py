#!/usr/bin/env python3
"""
Minimal FastAPI backend for the AI-Powered Interactive Learning Assistant
"""
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Learning Assistant API",
    description="Backend API for the AI-Powered Interactive Learning Assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    subject: Optional[str] = None
    grade_level: Optional[str] = None
    interaction_mode: str = "text_only"

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    context: str = ""

@app.get("/")
async def root():
    return {"message": "AI Learning Assistant API is running!", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": 1704067200.0,
        "models": {
            "qa": {"status": "not_loaded", "device": "cpu"},
            "summarization": {"status": "not_loaded", "device": "cpu"}
        },
        "message": "Minimal API version - models not loaded"
    }

@app.post("/sessions/")
async def create_session(request: SessionCreateRequest):
    """Create a new learning session"""
    session_id = f"session_{hash(str(request.dict()))}"[:16]
    return {
        "session_id": session_id,
        "user_id": request.user_id,
        "subject": request.subject,
        "grade_level": request.grade_level,
        "interaction_mode": request.interaction_mode,
        "created_at": "2025-07-01T00:00:00Z",
        "status": "active"
    }

@app.post("/question/")
async def ask_question(request: QuestionRequest):
    """Process a question and return an answer"""
    # Mock response for now
    return {
        "answer": f"This is a mock answer to your question: '{request.question}'. In a full implementation, this would be processed by AI models.",
        "confidence": 0.85,
        "processing_time": 0.5,
        "sources": ["mock_source_1", "mock_source_2"],
        "session_id": request.session_id
    }

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session history"""
    return {
        "session_id": session_id,
        "history": [
            {
                "timestamp": "2025-07-01T00:00:00Z",
                "question": "Sample question",
                "answer": "Sample answer",
                "type": "text"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
