#!/usr/bin/env python3
"""
Lightweight model setup script for the AI Learning Assistant
Downloads smaller, more efficient models for demonstration
"""
import os
import sys
import logging
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_lightweight_models():
    """Setup lightweight models for demonstration"""
    logger.info("🚀 Setting up lightweight AI models for demo...")
    
    try:
        # Import required libraries
        from transformers import (
            AutoTokenizer, AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM, pipeline
        )
        
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        
        # 1. Question Answering - Use a smaller, faster model
        logger.info("📚 Setting up Question Answering model (DistilBERT)...")
        qa_model_name = "distilbert-base-cased-distilled-squad"
        qa_dir = models_dir / "qa_model"
        qa_dir.mkdir(exist_ok=True)
        
        try:
            qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
            qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
            
            # Save locally
            qa_tokenizer.save_pretrained(qa_dir)
            qa_model.save_pretrained(qa_dir)
            
            # Test the model
            qa_pipeline = pipeline("question-answering", 
                                 model=qa_model, tokenizer=qa_tokenizer)
            test_result = qa_pipeline(
                question="What is AI?",
                context="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines."
            )
            logger.info(f"✅ QA Model test: {test_result['answer']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup QA model: {e}")
        
        # 2. Summarization - Use a smaller model
        logger.info("📝 Setting up Summarization model (DistilBART)...")
        summ_model_name = "sshleifer/distilbart-cnn-6-6"
        summ_dir = models_dir / "summarization_model"
        summ_dir.mkdir(exist_ok=True)
        
        try:
            summ_tokenizer = AutoTokenizer.from_pretrained(summ_model_name)
            summ_model = AutoModelForSeq2SeqLM.from_pretrained(summ_model_name)
            
            # Save locally
            summ_tokenizer.save_pretrained(summ_dir)
            summ_model.save_pretrained(summ_dir)
            
            # Test the model
            summ_pipeline = pipeline("summarization", 
                                   model=summ_model, tokenizer=summ_tokenizer)
            test_text = "Artificial Intelligence is revolutionizing education by providing personalized learning experiences. AI can adapt to individual student needs, provide instant feedback, and help teachers identify areas where students need additional support."
            test_result = summ_pipeline(test_text, max_length=50, min_length=10)
            logger.info(f"✅ Summarization test: {test_result[0]['summary_text']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Summarization model: {e}")
        
        # 3. Create a simple model registry
        model_registry = {
            "question_answering": {
                "path": str(qa_dir),
                "model_name": qa_model_name,
                "status": "ready",
                "type": "qa"
            },
            "summarization": {
                "path": str(summ_dir),
                "model_name": summ_model_name,
                "status": "ready", 
                "type": "seq2seq"
            }
        }
        
        registry_file = models_dir / "model_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(model_registry, f, indent=2)
        
        logger.info("✅ Model registry created")
        logger.info("🎉 Lightweight model setup completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model setup failed: {e}")
        return False

def create_sample_data():
    """Create sample data for the application"""
    logger.info("📊 Creating sample educational data...")
    
    data_dir = PROJECT_ROOT / "data" / "samples"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample lessons
    lessons = [
        {
            "id": "lesson_001",
            "title": "Introduction to Artificial Intelligence",
            "subject": "Computer Science",
            "grade_level": "High School",
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
            "questions": [
                {
                    "question": "What is Artificial Intelligence?",
                    "answer": "AI is a branch of computer science that aims to create intelligent machines that can think and learn like humans."
                },
                {
                    "question": "What tasks can AI systems perform?",
                    "answer": "AI systems can perform visual perception, speech recognition, decision-making, and language translation."
                }
            ]
        },
        {
            "id": "lesson_002", 
            "title": "Machine Learning Basics",
            "subject": "Computer Science",
            "grade_level": "High School",
            "content": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. There are three main types: supervised learning, unsupervised learning, and reinforcement learning.",
            "questions": [
                {
                    "question": "What is Machine Learning?",
                    "answer": "Machine Learning is a subset of AI that enables computers to learn from experience without being explicitly programmed."
                },
                {
                    "question": "What are the three main types of machine learning?",
                    "answer": "The three main types are supervised learning, unsupervised learning, and reinforcement learning."
                }
            ]
        }
    ]
    
    # Save lessons
    for lesson in lessons:
        lesson_file = data_dir / f"{lesson['id']}.json"
        with open(lesson_file, 'w') as f:
            json.dump(lesson, f, indent=2)
    
    logger.info(f"✅ Created {len(lessons)} sample lessons")

if __name__ == "__main__":
    logger.info("🚀 Starting Lightweight AI Learning Assistant Setup")
    
    # Create sample data
    create_sample_data()
    
    # Setup models
    success = setup_lightweight_models()
    
    if success:
        logger.info("✅ Setup completed successfully!")
        logger.info("🎯 You can now run the full application:")
        logger.info("   1. FastAPI Backend: python -m src.api.main")
        logger.info("   2. Streamlit Frontend: streamlit run src/ui/app.py")
    else:
        logger.error("❌ Setup failed. Check the logs above for details.")
        sys.exit(1)
