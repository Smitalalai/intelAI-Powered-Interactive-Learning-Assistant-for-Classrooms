import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fallback-secret-key-change-in-production'
    
    # Use PostgreSQL for production (required for Vercel)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///eduai_pro.db'  # Fallback for local dev
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    AI_MODEL = 'gpt-3.5-turbo'
    
    # Demo mode - set to False in production with real API key
    AI_DEMO_MODE = os.environ.get('AI_DEMO_MODE', 'true').lower() == 'true'
    
    # Vercel-specific settings
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
class ProductionConfig(Config):
    DEBUG = False
    AI_DEMO_MODE = True  # Keep demo mode for Vercel unless API key provided

class DevelopmentConfig(Config):
    DEBUG = True
    AI_DEMO_MODE = True

# Choose config based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
