"""
Configuration settings for the AI-Powered Interactive Learning Assistant
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "question_answering": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 512,
        "temperature": 0.7,
        "device": "CPU"  # Will be dynamically set based on OpenVINO capabilities
    },
    "summarization": {
        "model_name": "facebook/bart-large-cnn",
        "max_length": 150,
        "min_length": 30,
        "device": "CPU"
    },
    "speech_recognition": {
        "model_name": "openai/whisper-base",
        "language": "en",
        "device": "CPU"
    },
    "text_to_speech": {
        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
        "vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
        "device": "CPU"
    },
    "image_captioning": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "max_length": 50,
        "device": "CPU"
    }
}

# OpenVINO optimization settings
OPENVINO_CONFIG = {
    "precision": "FP16",  # Options: FP32, FP16, INT8
    "target_devices": ["CPU", "GPU", "NPU"],
    "performance_hint": "LATENCY",  # Options: LATENCY, THROUGHPUT
    "cache_dir": str(MODELS_DIR / "openvino_cache"),
    "intel_optimizations": {
        "cpu_threads": "auto",  # Use all available CPU threads
        "enable_cpu_pinning": True,
        "enable_dynamic_shapes": True,
        "inference_precision": "bf16",  # Brain float 16 for Intel CPUs
        "cpu_throughput_streams": "auto"
    },
    "npu_config": {
        "enable_npu": True,
        "npu_device_id": 0,
        "performance_hint": "LATENCY"
    }
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# UI settings
UI_CONFIG = {
    "title": "AI-Powered Interactive Learning Assistant",
    "theme": "light",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "supported_audio_formats": ["wav", "mp3", "m4a", "ogg", "flac"],
    "supported_image_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    "max_file_size_mb": 10,
    "default_grade_levels": ["Elementary", "Middle School", "High School", "College", "Adult Learning"],
    "default_subjects": ["Mathematics", "Science", "English", "History", "Geography", "Art", "Music", "Other"],
    "interaction_modes": ["text_only", "voice_only", "multimodal", "classroom"]
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Performance benchmarking thresholds
PERFORMANCE_THRESHOLDS = {
    "question_answering_max_time": 2.0,  # seconds
    "summarization_max_time": 3.0,  # seconds
    "speech_recognition_max_time": 1.0,  # seconds
    "text_to_speech_max_time": 2.0,  # seconds
    "image_captioning_max_time": 1.5,  # seconds
    "max_memory_usage": 4 * 1024 * 1024 * 1024  # 4GB in bytes
}

# Environment variables with defaults
def get_env_var(key: str, default: Any = None) -> str:
    """Get environment variable with default fallback"""
    return os.getenv(key, default)

# Database configuration (for future user progress tracking)
DATABASE_CONFIG = {
    "url": get_env_var("DATABASE_URL", "sqlite:///./learning_assistant.db"),
    "echo": get_env_var("DATABASE_ECHO", "False").lower() == "true"
}

# Security settings
SECURITY_CONFIG = {
    "secret_key": get_env_var("SECRET_KEY", "your-secret-key-here"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
}

# Personalization and adaptive learning configurations
PERSONALIZATION_CONFIG = {
    "learning_style_detection": {
        "visual": {"weight": 0.3, "indicators": ["image_requests", "diagram_questions"]},
        "auditory": {"weight": 0.3, "indicators": ["speech_input_frequency", "explanation_requests"]},
        "kinesthetic": {"weight": 0.2, "indicators": ["interactive_examples", "step_by_step_requests"]},
        "reading": {"weight": 0.2, "indicators": ["text_input_preference", "detailed_explanations"]}
    },
    "difficulty_adaptation": {
        "easy": {"vocabulary_level": "grade_appropriate", "complexity_score": 0.3},
        "medium": {"vocabulary_level": "advanced", "complexity_score": 0.6},
        "hard": {"vocabulary_level": "expert", "complexity_score": 0.9}
    },
    "progress_tracking": {
        "session_weight": 0.4,
        "historical_weight": 0.6,
        "min_sessions_for_adaptation": 3
    }
}

# Real-time performance monitoring
REALTIME_CONFIG = {
    "enable_performance_monitoring": True,
    "benchmark_intervals": 10,  # seconds
    "alert_thresholds": {
        "high_latency": 3.0,  # seconds
        "high_memory": 0.8,   # 80% of available memory
        "low_accuracy": 0.7   # 70% accuracy threshold
    },
    "auto_optimization": {
        "enabled": True,
        "fallback_to_cpu": True,
        "dynamic_batch_sizing": True
    }
}

# Classroom engagement features
CLASSROOM_CONFIG = {
    "max_concurrent_students": 30,
    "session_timeout_minutes": 60,
    "content_generation": {
        "quiz_questions_per_topic": 5,
        "summary_max_bullets": 7,
        "explanation_depth_levels": ["basic", "intermediate", "advanced"]
    },
    "accessibility": {
        "text_size_options": ["small", "medium", "large", "extra_large"],
        "high_contrast_mode": True,
        "screen_reader_support": True,
        "keyboard_navigation": True
    }
}

# Study material generation settings
CONTENT_GENERATION_CONFIG = {
    "study_guides": {
        "max_topics": 10,
        "include_examples": True,
        "include_practice_questions": True,
        "difficulty_levels": ["beginner", "intermediate", "advanced"]
    },
    "lesson_planning": {
        "duration_options": [15, 30, 45, 60, 90],  # minutes
        "include_activities": True,
        "include_assessments": True,
        "adaptive_pacing": True
    },
    "quiz_generation": {
        "question_types": ["multiple_choice", "true_false", "short_answer", "essay"],
        "auto_grading": True,
        "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    }
}
