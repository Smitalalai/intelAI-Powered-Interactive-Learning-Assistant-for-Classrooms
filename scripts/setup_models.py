#!/usr/bin/env python3
"""
Model setup script for downloading and optimizing AI models
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.base_model import ModelManager
from src.models.question_answering import QuestionAnsweringModel
from src.models.summarization import SummarizationModel
from src.models.speech_recognition import SpeechRecognitionModel
from src.models.text_to_speech import TextToSpeechModel
from configs.config import MODELS_DIR, OPENVINO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_models():
    """Download and optimize all models"""
    logger.info("Starting model setup process...")
    
    try:
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # List of models to setup
        models_to_setup = [
            ("Question Answering", QuestionAnsweringModel),
            ("Summarization", SummarizationModel),
            ("Speech Recognition", SpeechRecognitionModel),
            ("Text-to-Speech", TextToSpeechModel)
        ]
        
        for model_name, model_class in models_to_setup:
            logger.info(f"Setting up {model_name} model...")
            
            try:
                # Initialize model
                model = model_class()
                
                # Register with manager
                model_manager.register_model(model_name.lower().replace(" ", "_"), model)
                
                # Load and optimize model
                model.load_model()
                
                logger.info(f"✅ {model_name} model setup completed")
                
            except Exception as e:
                logger.error(f"❌ Failed to setup {model_name} model: {str(e)}")
                # Continue with other models
                continue
        
        logger.info("Model setup process completed!")
        
        # Run a quick test
        logger.info("Running quick model tests...")
        test_models(model_manager)
        
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        sys.exit(1)


def test_models(model_manager: ModelManager):
    """Run quick tests on loaded models"""
    test_results = {}
    
    # Test Question Answering
    qa_model = model_manager.get_model("question_answering")
    if qa_model:
        try:
            result = qa_model.predict({
                "question": "What is machine learning?",
                "context": "Machine learning is a branch of artificial intelligence that uses algorithms to learn patterns from data."
            })
            test_results["Question Answering"] = "✅ Working" if result.get("answer") else "❌ Failed"
        except Exception as e:
            test_results["Question Answering"] = f"❌ Error: {str(e)}"
    
    # Test Summarization
    summarization_model = model_manager.get_model("summarization")
    if summarization_model:
        try:
            result = summarization_model.predict({
                "text": "Artificial intelligence is a rapidly growing field that encompasses machine learning, deep learning, and natural language processing. These technologies are transforming various industries including healthcare, finance, and education."
            })
            test_results["Summarization"] = "✅ Working" if result.get("summary") else "❌ Failed"
        except Exception as e:
            test_results["Summarization"] = f"❌ Error: {str(e)}"
    
    # Test Speech Recognition (basic initialization test)
    stt_model = model_manager.get_model("speech_recognition")
    if stt_model:
        test_results["Speech Recognition"] = "✅ Initialized" if stt_model.whisper_model else "❌ Not initialized"
    
    # Test Text-to-Speech (basic initialization test)
    tts_model = model_manager.get_model("text_to_speech")
    if tts_model:
        test_results["Text-to-Speech"] = "✅ Initialized" if tts_model.tts_model else "❌ Not initialized"
    
    # Print test results
    logger.info("Model Test Results:")
    for model_name, result in test_results.items():
        logger.info(f"  {model_name}: {result}")


def download_sample_data():
    """Download sample educational content for testing"""
    logger.info("Setting up sample data...")
    
    sample_data_dir = PROJECT_ROOT / "data" / "samples"
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample lesson content
    sample_lessons = {
        "mathematics_algebra.txt": """
        Introduction to Algebra
        
        Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations. The main purpose of algebra is to find unknown values, called variables.
        
        Key Concepts:
        1. Variables: Letters that represent unknown numbers (like x, y, z)
        2. Expressions: Combinations of variables, numbers, and operations (like 2x + 3)
        3. Equations: Mathematical statements that show two expressions are equal (like 2x + 3 = 7)
        
        Basic Operations:
        - Addition and subtraction of like terms
        - Multiplication and division of variables
        - Solving simple equations
        
        Example: If 2x + 3 = 7, then 2x = 4, so x = 2
        
        Practice Problems:
        1. Solve for x: 3x + 5 = 14
        2. Simplify: 4x + 2x - 3x
        3. If y = 2x + 1 and x = 3, what is y?
        """,
        
        "science_photosynthesis.txt": """
        Photosynthesis: How Plants Make Food
        
        Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose (sugar) and oxygen. This process is essential for life on Earth as it produces the oxygen we breathe and the food that forms the base of most food chains.
        
        The Photosynthesis Equation:
        6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
        
        Key Components:
        1. Chloroplasts: Cell structures where photosynthesis occurs
        2. Chlorophyll: Green pigment that captures light energy
        3. Stomata: Tiny pores on leaves that allow gas exchange
        
        Two Main Stages:
        1. Light Reactions: Convert light energy to chemical energy
        2. Calvin Cycle: Use chemical energy to make glucose
        
        Importance:
        - Produces oxygen for all living things
        - Creates food for plants and animals
        - Removes carbon dioxide from the atmosphere
        - Forms the foundation of most ecosystems
        
        Factors Affecting Photosynthesis:
        - Light intensity
        - Temperature
        - Carbon dioxide concentration
        - Water availability
        """,
        
        "history_world_war_ii.txt": """
        World War II: A Global Conflict (1939-1945)
        
        World War II was the largest and most destructive conflict in human history, involving more than 30 countries and resulting in 70-85 million deaths. The war was fought between the Axis powers (primarily Germany, Italy, and Japan) and the Allied forces (including the United States, Soviet Union, United Kingdom, and China).
        
        Major Causes:
        1. Rise of totalitarian regimes in Germany, Italy, and Japan
        2. Economic problems from the Great Depression
        3. Failure of the League of Nations to prevent aggression
        4. Unresolved issues from World War I
        
        Key Events:
        - September 1939: Germany invades Poland, war begins
        - December 1941: Japan attacks Pearl Harbor, US enters war
        - June 1944: D-Day invasion of Normandy
        - August 1945: Atomic bombs dropped on Japan
        - September 1945: Japan surrenders, war ends
        
        Significant Battles:
        - Battle of Britain (1940)
        - Battle of Stalingrad (1942-1943)
        - Battle of Midway (1942)
        - Battle of Normandy (1944)
        
        Consequences:
        - Establishment of the United Nations
        - Beginning of the Cold War
        - Decolonization movements
        - Technological advances
        - Human rights awareness
        """
    }
    
    for filename, content in sample_lessons.items():
        file_path = sample_data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    logger.info(f"Sample lesson files created in {sample_data_dir}")


def create_environment_file():
    """Create a sample .env file"""
    env_file = PROJECT_ROOT / ".env"
    
    if not env_file.exists():
        env_content = """
# AI Learning Assistant Configuration

# Database
DATABASE_URL=sqlite:///./learning_assistant.db
DATABASE_ECHO=False

# Security
SECRET_KEY=your-secret-key-change-this-in-production

# OpenVINO Settings
OPENVINO_CACHE_DIR=./models/openvino_cache

# Logging
LOG_LEVEL=INFO

# Development Settings
DEVELOPMENT_MODE=True
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content.strip())
        
        logger.info(f"Environment file created: {env_file}")


def main():
    """Main setup function"""
    logger.info("🚀 Starting AI Learning Assistant Setup")
    
    # Create environment file
    create_environment_file()
    
    # Download sample data
    download_sample_data()
    
    # Setup models
    setup_models()
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Start the API server: python src/api/main.py")
    logger.info("2. Start the UI: streamlit run src/ui/app.py")
    logger.info("3. Open your browser to http://localhost:8501")


if __name__ == "__main__":
    main()
