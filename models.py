from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Date
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    qualification = db.Column(db.String(100), nullable=False)
    dob = Column(Date, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password, password)

class Subject(db.Model):
    __tablename__ = 'subjects'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    chapters = db.relationship('Chapter', backref='subject', lazy=True, cascade="all, delete-orphan")

class Chapter(db.Model):
    __tablename__ = 'chapters'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    quizzes = db.relationship('Quiz', backref='chapter', lazy=True, cascade="all, delete-orphan")
    pools = db.relationship('QuestionPool', backref='chapter', lazy=True, cascade="all, delete-orphan")

class QuestionPool(db.Model):
    __tablename__ = 'question_pools'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    chapter_id = db.Column(db.Integer, db.ForeignKey('chapters.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    questions = db.relationship('PoolQuestion', backref='pool', lazy=True, cascade="all, delete-orphan")
    quizzes = db.relationship('Quiz', backref='question_pool', lazy=True)

class PoolQuestion(db.Model):
    __tablename__ = 'pool_questions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pool_id = db.Column(db.Integer, db.ForeignKey('question_pools.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    option_a = db.Column(db.String(200), nullable=False)
    option_b = db.Column(db.String(200), nullable=False)
    option_c = db.Column(db.String(200), nullable=False)
    option_d = db.Column(db.String(200), nullable=False)
    correct_answer = db.Column(db.String(1), nullable=False)  # 'A', 'B', 'C', or 'D'
    user_answers = db.relationship('UserAnswer', backref='question', lazy=True)

class Quiz(db.Model):
    __tablename__ = 'quizzes'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    chapter_id = db.Column(db.Integer, db.ForeignKey('chapters.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    date_of_quiz = db.Column(db.Date, nullable=False)
    time_duration = db.Column(db.String(5), nullable=False)  # Format: "HH:MM"
    remarks = db.Column(db.Text)
    randomize_questions = db.Column(db.Boolean, default=False)
    num_questions_from_pool = db.Column(db.Integer, default=0)  # 0 means all questions
    show_correct_answers = db.Column(db.Boolean, default=True)
    pool_id = db.Column(db.Integer, db.ForeignKey('question_pools.id'))
    attempts = db.relationship('QuizAttempt', backref='quiz', lazy=True, cascade="all, delete-orphan")

class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempts'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quizzes.id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    percentage = db.Column(db.Float, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    answers = db.relationship('UserAnswer', backref='attempt', lazy=True, cascade="all, delete-orphan")
    user = db.relationship('User', backref='attempts')

class UserAnswer(db.Model):
    __tablename__ = 'user_answers'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey('quiz_attempts.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('pool_questions.id'), nullable=False)
    user_answer = db.Column(db.String(1))  # 'A', 'B', 'C', or 'D'
    is_correct = db.Column(db.Boolean, default=False)

# New AI-Enhanced Models

class LearningJournal(db.Model):
    __tablename__ = 'learning_journals'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow)
    content = db.Column(db.Text)
    mood = db.Column(db.String(50))
    difficulty_level = db.Column(db.String(50))
    ai_feedback = db.Column(db.Text)
    tags = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='journal_entries')

class Flashcard(db.Model):
    __tablename__ = 'flashcards'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    topic = db.Column(db.String(100), nullable=False)
    front_text = db.Column(db.Text, nullable=False)
    back_text = db.Column(db.Text, nullable=False)
    difficulty = db.Column(db.String(20), default='medium')
    next_review_date = db.Column(db.DateTime)
    review_count = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='flashcards')

class AIInteraction(db.Model):
    __tablename__ = 'ai_interactions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    interaction_type = db.Column(db.String(50))  # 'chat', 'hint_request', 'explanation', etc.
    content = db.Column(db.Text)  # User input content
    ai_response = db.Column(db.Text)
    context_metadata = db.Column(db.Text)  # JSON string for additional context
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='ai_interactions')

class ContentLibrary(db.Model):
    __tablename__ = 'content_library'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(50))  # 'pdf', 'ppt', 'image', 'text'
    file_path = db.Column(db.String(500))
    ai_analysis = db.Column(db.Text)  # AI analysis results
    content_metadata = db.Column(db.Text)  # JSON metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'))
    user = db.relationship('User', backref='content_files')
    subject = db.relationship('Subject', backref='content_files')

class StudentInsight(db.Model):
    __tablename__ = 'student_insights'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    insight_type = db.Column(db.String(50))  # 'performance', 'learning_pattern', 'suggestion'
    insight_data = db.Column(db.Text)  # JSON string with insight details
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    user = db.relationship('User', backref='insights')
