import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'smital20'
    
    # Database configuration for production
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///eduai_pro.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # Set this in environment for real AI
    AI_MODEL = 'gpt-3.5-turbo'  # Default model to use
    
    # Demo mode - when True, uses enhanced mock responses instead of real AI
    # Set to False in production to use real OpenVINO processing
    AI_DEMO_MODE = os.environ.get('AI_DEMO_MODE', 'true').lower() == 'true'
    
    # Production settings
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', '0.0.0.0')