from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from models import db, User, Subject, Chapter, Quiz, QuizAttempt, QuestionPool, PoolQuestion, UserAnswer, LearningJournal, Flashcard, AIInteraction, ContentLibrary, StudentInsight
from ai_service import AIService
from datetime import datetime, timedelta
from functools import wraps
import random
import os
import json

# Lightweight versions of services for Vercel
class MockOpenVINOService:
    """Lightweight mock service for Vercel deployment"""
    def __init__(self):
        self.device = "CPU (Mock)"
    
    def text_generation(self, prompt):
        return f"AI Response: {prompt[:50]}..." if len(prompt) > 50 else f"AI Response: {prompt}"
    
    def generate_flashcards(self, topic, difficulty, count):
        cards = []
        for i in range(count):
            cards.append({
                'front': f"{topic} - Question {i+1}",
                'back': f"Answer about {topic} (difficulty: {difficulty})"
            })
        return cards
    
    def real_time_interaction(self, interaction_type, content):
        return {
            'content': f"Response to {interaction_type}: {str(content)[:100]}",
            'confidence': 0.85,
            'latency_ms': 45
        }
    
    def get_performance_metrics(self):
        return {
            'device': self.device,
            'inference_time_ms': 45,
            'model_load_time_ms': 1200,
            'memory_usage_mb': 150
        }

class MockMultimodalService:
    """Lightweight mock multimodal service"""
    def __init__(self, openvino_service):
        self.openvino_service = openvino_service
    
    def get_interaction_capabilities(self):
        return {
            'text': True,
            'voice': False,  # Disabled for Vercel
            'image': False,  # Disabled for Vercel
            'real_time': True
        }

# Import config based on environment
try:
    from config_vercel import config
    app_config = config[os.environ.get('FLASK_ENV', 'production')]
except ImportError:
    from config import Config
    app_config = Config

app = Flask(__name__, template_folder='templates')
app.config.from_object(app_config)

# Initialize lightweight AI Services for Vercel
ai_service = AIService(api_key=app.config.get('OPENAI_API_KEY'), demo_mode=app.config.get('AI_DEMO_MODE', True))
openvino_service = MockOpenVINOService()
multimodal_service = MockMultimodalService(openvino_service)

db.init_app(app)

# Helper functions
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        user = db.session.get(User, session['user_id'])
        if not user or not user.is_admin:
            flash('You do not have permission to access this page', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Create admin user and tables
def init_db():
    """Initialize database and admin user"""
    try:
        db.create_all()
        admin = User.query.filter_by(is_admin=True).first()
        if not admin:
            admin = User(
                username='admin',
                full_name='Administrator',
                email='admin@eduaipro.com',
                qualification='Admin',
                dob=datetime.strptime('2000-01-01', '%Y-%m-%d').date(),
                is_admin=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")

# Initialize database when app starts
with app.app_context():
    init_db()

# Main routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin
            flash(f'Welcome back, {user.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        qualification = request.form.get('qualification')
        dob = request.form.get('dob')
        password = request.form.get('password')
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('auth/register.html')
        
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('auth/register.html')
        
        try:
            # Create new user
            user = User(
                username=username,
                full_name=full_name,
                email=email,
                qualification=qualification,
                dob=datetime.strptime(dob, '%Y-%m-%d').date(),
                is_admin=False
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'danger')
    
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = db.session.get(User, session['user_id'])
    subjects = Subject.query.all()
    attempts = []
    ai_insights = {}
    
    if not user.is_admin:
        attempts = db.session.query(QuizAttempt).filter_by(user_id=user.id).order_by(QuizAttempt.start_time.desc()).limit(5).all()
        attempts_data = [{'percentage': attempt.percentage, 'subject': attempt.quiz.chapter.subject.name} for attempt in attempts]
        ai_insights = ai_service.analyze_learning_pattern(attempts_data)
        
        if attempts:
            avg_score = sum(attempt.percentage for attempt in attempts) / len(attempts)
            ai_insights['personalized_feedback'] = ai_service.generate_personalized_feedback({'percentage': avg_score})
    else:
        all_attempts = QuizAttempt.query.all()
        total_students = User.query.filter_by(is_admin=False).count()
        ai_insights = {
            'total_students': total_students,
            'total_attempts': len(all_attempts),
            'average_performance': sum(attempt.percentage for attempt in all_attempts) / len(all_attempts) if all_attempts else 0
        }
    
    return render_template('dashboard.html', user=user, subjects=subjects, attempts=attempts, ai_insights=ai_insights)

# API Routes (simplified for Vercel)
@app.route('/api/create_flashcards', methods=['POST'])
@login_required
def create_flashcards():
    """Create AI flashcards (Vercel-compatible)"""
    data = request.get_json()
    topic = data.get('topic', '')
    count = data.get('count', 5)
    difficulty = data.get('difficulty', 'medium')
    
    start_time = datetime.now()
    ai_flashcards = openvino_service.generate_flashcards(topic, difficulty, count)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds() * 1000
    
    return jsonify({
        'success': True,
        'flashcards_created': len(ai_flashcards),
        'processing_time_ms': processing_time,
        'device_used': openvino_service.device,
        'message': f'Created {len(ai_flashcards)} AI-powered flashcards for {topic}'
    })

@app.route('/api/openvino/performance_metrics')
@login_required
def openvino_performance_metrics():
    """Get performance metrics"""
    try:
        metrics = openvino_service.get_performance_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Vercel"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'app': 'EduAI Pro',
        'version': '1.0.0'
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

# Vercel entry point - the app object should be accessible directly
# No need for a handler function in Vercel with Python

if __name__ == '__main__':
    app.run(debug=True)
