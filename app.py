from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from models import db, User, Subject, Chapter, Quiz, QuizAttempt, QuestionPool, PoolQuestion, UserAnswer, LearningJournal, Flashcard, AIInteraction, ContentLibrary, StudentInsight
from ai_service import AIService
from openvino_service import OpenVINOAIService
from multimodal_service import MultimodalService
from config import Config
from datetime import datetime, timedelta
from functools import wraps
import random
import os
import json
import asyncio
import base64

app = Flask(__name__,template_folder='templates')
app.config.from_object(Config)

# Initialize AI Services
ai_service = AIService(api_key=app.config.get('OPENAI_API_KEY'), demo_mode=app.config.get('AI_DEMO_MODE', True))

# Initialize OpenVINO Service for optimized inference
openvino_service = OpenVINOAIService()

# Initialize Multimodal Service
multimodal_service = MultimodalService(openvino_service)

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

# Create admin user
def create_admin():
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
        print("Admin user created")

with app.app_context():
    db.create_all()
    create_admin()

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        print(f"Login attempt: username={username}")
        
        user = User.query.filter_by(username=username).first()
        if user:
            print(f"User found: {user.username}, is_admin: {user.is_admin}")
            if user.check_password(password):
                session['user_id'] = user.id
                session['is_admin'] = user.is_admin
                print(f"Login successful, session: user_id={session['user_id']}, is_admin={session['is_admin']}")
                flash(f'Welcome back, {user.full_name}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                print("Password check failed")
                flash('Invalid username or password', 'danger')
        else:
            print("User not found")
            flash('Invalid username or password', 'danger')
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        full_name = request.form.get('fullname')
        email = request.form.get('email')
        qualification = request.form.get('qualification')
        dob = datetime.strptime(request.form.get('dob'), '%Y-%m-%d').date()
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username already exists', 'danger')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('auth/register.html')
        
        # Check if the date of birth is valid
        if dob > datetime.now().date():
            flash('Date of birth cannot be in the future', 'danger')
            return render_template('auth/register.html')

        new_user = User(
            username=username,
            full_name=full_name,
            email=email,
            qualification=qualification,
            dob=dob
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth/register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    print(f"Dashboard accessed, session: {session}")
    user = db.session.get(User, session['user_id'])
    print(f"User retrieved: {user.username if user else 'None'}")
    subjects = Subject.query.all()
    attempts = []
    ai_insights = {}
    
    if not user.is_admin:
        attempts = db.session.query(QuizAttempt).filter_by(user_id=user.id).order_by(QuizAttempt.start_time.desc()).limit(5).all()
        
        # Generate AI insights for student
        attempts_data = [{'percentage': attempt.percentage, 'subject': attempt.quiz.chapter.subject.name} for attempt in attempts]
        ai_insights = ai_service.analyze_learning_pattern(attempts_data)
        
        # Get recent AI interactions
        recent_interactions = AIInteraction.query.filter_by(user_id=user.id).order_by(AIInteraction.timestamp.desc()).limit(3).all()
        ai_insights['recent_interactions'] = recent_interactions
        
        # Get personalized suggestions
        if attempts:
            avg_score = sum(attempt.percentage for attempt in attempts) / len(attempts)
            ai_insights['personalized_feedback'] = ai_service.generate_personalized_feedback({'percentage': avg_score})
    else:
        # Admin insights
        all_attempts = QuizAttempt.query.all()
        total_students = User.query.filter_by(is_admin=False).count()
        ai_insights = {
            'total_students': total_students,
            'total_attempts': len(all_attempts),
            'average_performance': sum(attempt.percentage for attempt in all_attempts) / len(all_attempts) if all_attempts else 0
        }
    
    print(f"Rendering dashboard for user: {user.username}")
    return render_template('dashboard.html', user=user, subjects=subjects, attempts=attempts, ai_insights=ai_insights)

@app.route('/subject/<int:subject_id>')
@login_required
def subject_detail(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    chapters = Chapter.query.filter_by(subject_id=subject_id).all()
    return render_template('subject_details.html', subject=subject, chapters=chapters)


@app.route('/chapter/<int:chapter_id>')
@login_required
def chapter_detail(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    quizzes = Quiz.query.filter_by(chapter_id=chapter_id).all()
    return render_template('chapter_details.html', chapter=chapter, quizzes=quizzes)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('landing'))

# Admin routes
@app.route('/admin/subjects', methods=['GET', 'POST'])
@admin_required
def admin_subjects():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        new_subject = Subject(name=name, description=description)
        db.session.add(new_subject)
        db.session.commit()
        flash('Subject added successfully', 'success')
    
    subjects = Subject.query.all()
    return render_template('admin/subjects.html', subjects=subjects)

@app.route('/admin/subject/edit/<int:subject_id>', methods=['GET', 'POST'])
@admin_required
def edit_subject(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    
    if request.method == 'POST':
        subject.name = request.form.get('name')
        subject.description = request.form.get('description')
        db.session.commit()
        flash('Subject updated successfully', 'success')
        return redirect(url_for('admin_subjects'))
    
    return render_template('admin/edit_subject.html', subject=subject)

@app.route('/admin/subject/delete/<int:subject_id>')
@admin_required
def delete_subject(subject_id):
    subject = Subject.query.get_or_404(subject_id)
    db.session.delete(subject)
    db.session.commit()
    flash('Subject deleted successfully', 'success')
    return redirect(url_for('admin_subjects'))

@app.route('/admin/chapters', methods=['GET', 'POST'])
@admin_required
def admin_chapters():
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        name = request.form.get('name')
        description = request.form.get('description')
        new_chapter = Chapter(subject_id=subject_id, name=name, description=description)
        db.session.add(new_chapter)
        db.session.commit()
        flash('Chapter added successfully', 'success')
    
    subjects = Subject.query.all()
    chapters = Chapter.query.all()
    return render_template('admin/chapters.html', subjects=subjects, chapters=chapters)

@app.route('/admin/chapter/edit/<int:chapter_id>', methods=['GET', 'POST'])
@admin_required
def edit_chapter(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    
    if request.method == 'POST':
        chapter.subject_id = request.form.get('subject_id')
        chapter.name = request.form.get('name')
        chapter.description = request.form.get('description')
        db.session.commit()
        flash('Chapter updated successfully', 'success')
        return redirect(url_for('admin_chapters'))
    
    subjects = Subject.query.all()
    return render_template('admin/edit_chapter.html', chapter=chapter, subjects=subjects)

@app.route('/admin/chapter/delete/<int:chapter_id>')
@admin_required
def delete_chapter(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    db.session.delete(chapter)
    db.session.commit()
    flash('Chapter deleted successfully', 'success')
    return redirect(url_for('admin_chapters'))

@app.route('/admin/pools', methods=['GET', 'POST'])
@admin_required
def admin_pools():
    if request.method == 'POST':
        chapter_id = request.form.get('chapter_id')
        name = request.form.get('name')
        description = request.form.get('description')
        
        new_pool = QuestionPool(chapter_id=chapter_id, name=name, description=description)
        db.session.add(new_pool)
        db.session.commit()
        flash('Question pool added successfully', 'success')
    
    chapters = Chapter.query.all()
    pools = QuestionPool.query.all()
    return render_template('admin/pools.html', chapters=chapters, pools=pools)

@app.route('/admin/pool/edit/<int:pool_id>', methods=['GET', 'POST'])
@admin_required
def edit_pool(pool_id):
    pool = QuestionPool.query.get_or_404(pool_id)
    
    if request.method == 'POST':
        pool.chapter_id = request.form.get('chapter_id')
        pool.name = request.form.get('name')
        pool.description = request.form.get('description')
        db.session.commit()
        flash('Question pool updated successfully', 'success')
        return redirect(url_for('admin_pools'))
    
    chapters = Chapter.query.all()
    return render_template('admin/edit_pool.html', pool=pool, chapters=chapters)

@app.route('/admin/pool/delete/<int:pool_id>')
@admin_required
def delete_pool(pool_id):
    pool = QuestionPool.query.get_or_404(pool_id)
    db.session.delete(pool)
    db.session.commit()
    flash('Question pool deleted successfully', 'success')
    return redirect(url_for('admin_pools'))

@app.route('/admin/pool/<int:pool_id>/questions', methods=['GET', 'POST'])
@admin_required
def admin_pool_questions(pool_id):
    pool = QuestionPool.query.get_or_404(pool_id)
    
    if request.method == 'POST':
        question_text = request.form.get('question_text')
        option_a = request.form.get('option_a')
        option_b = request.form.get('option_b')
        option_c = request.form.get('option_c')
        option_d = request.form.get('option_d')
        correct_answer = request.form.get('correct_answer')
        
        new_question = PoolQuestion(
            pool_id=pool_id,
            question_text=question_text,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
            correct_answer=correct_answer
        )
        db.session.add(new_question)
        db.session.commit()
        flash('Question added to pool successfully', 'success')
    
    questions = PoolQuestion.query.filter_by(pool_id=pool_id).all()
    return render_template('admin/pool_questions.html', pool=pool, questions=questions)

@app.route('/admin/question/edit/<int:question_id>', methods=['GET', 'POST'])
@admin_required
def edit_question(question_id):
    question = PoolQuestion.query.get_or_404(question_id)
    
    if request.method == 'POST':
        question.question_text = request.form.get('question_text')
        question.option_a = request.form.get('option_a')
        question.option_b = request.form.get('option_b')
        question.option_c = request.form.get('option_c')
        question.option_d = request.form.get('option_d')
        question.correct_answer = request.form.get('correct_answer')
        db.session.commit()
        flash('Question updated successfully', 'success')
        return redirect(url_for('admin_pool_questions', pool_id=question.pool_id))
    
    return render_template('admin/edit_question.html', question=question)

@app.route('/admin/question/delete/<int:question_id>')
@admin_required
def delete_question(question_id):
    question = PoolQuestion.query.get_or_404(question_id)
    pool_id = question.pool_id
    db.session.delete(question)
    db.session.commit()
    flash('Question deleted successfully', 'success')
    return redirect(url_for('admin_pool_questions', pool_id=pool_id))

@app.route('/admin/quizzes', methods=['GET', 'POST'])
@admin_required
def admin_quizzes():
    if request.method == 'POST':
        chapter_id = request.form.get('chapter_id')
        name = request.form.get('name')
        date_of_quiz = datetime.strptime(request.form.get('date_of_quiz'), '%Y-%m-%d').date()
        time_duration = request.form.get('time_duration')
        remarks = request.form.get('remarks')
        randomize_questions = 'randomize_questions' in request.form
        num_questions = request.form.get('num_questions', 0, type=int)
        show_correct_answers = 'show_correct_answers' in request.form
        pool_id = request.form.get('pool_id')
        
        new_quiz = Quiz(
            chapter_id=chapter_id,
            name=name,
            date_of_quiz=date_of_quiz,
            time_duration=time_duration,
            remarks=remarks,
            randomize_questions=randomize_questions,
            num_questions_from_pool=num_questions,
            show_correct_answers=show_correct_answers,
            pool_id=pool_id
        )
        db.session.add(new_quiz)
        db.session.commit()
        flash('Quiz added successfully', 'success')
    
    chapters = Chapter.query.all()
    pools = QuestionPool.query.all()
    quizzes = Quiz.query.all()
    return render_template('admin/quizzes.html', chapters=chapters, pools=pools, quizzes=quizzes)

@app.route('/admin/quiz/edit/<int:quiz_id>', methods=['GET', 'POST'])
@admin_required
def edit_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    
    if request.method == 'POST':
        quiz.chapter_id = request.form.get('chapter_id')
        quiz.name = request.form.get('name')
        quiz.date_of_quiz = datetime.strptime(request.form.get('date_of_quiz'), '%Y-%m-%d').date()
        quiz.time_duration = request.form.get('time_duration')
        quiz.remarks = request.form.get('remarks')
        quiz.randomize_questions = 'randomize_questions' in request.form
        quiz.num_questions_from_pool = request.form.get('num_questions', 0, type=int)
        quiz.show_correct_answers = 'show_correct_answers' in request.form
        quiz.pool_id = request.form.get('pool_id')
        db.session.commit()
        flash('Quiz updated successfully', 'success')
        return redirect(url_for('admin_quizzes'))
    
    chapters = Chapter.query.all()
    pools = QuestionPool.query.all()
    return render_template('admin/edit_quiz.html', quiz=quiz, chapters=chapters, pools=pools)

@app.route('/admin/quiz/delete/<int:quiz_id>')
@admin_required
def delete_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    db.session.delete(quiz)
    db.session.commit()
    flash('Quiz deleted successfully', 'success')
    return redirect(url_for('admin_quizzes'))

@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.filter_by(is_admin=False).all()
    return render_template('admin/users.html', users=users)

@app.route('/quiz/<int:quiz_id>/take', methods=['GET', 'POST'])
@login_required
def take_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    
    # Check if user has already attempted this quiz
    existing_attempt = db.session.query(QuizAttempt).filter_by(user_id=user_id, quiz_id=quiz_id).first()

    if existing_attempt:
        flash('You have already attempted this quiz', 'info')
        return redirect(url_for('quiz_results', attempt_id=existing_attempt.id))
    
    # Get questions from the pool
    pool_questions = PoolQuestion.query.filter_by(pool_id=quiz.pool_id).all()
    
    # If number of questions is specified and less than total, select random subset
    if quiz.num_questions_from_pool > 0 and quiz.num_questions_from_pool < len(pool_questions):
        selected_questions = random.sample(pool_questions, quiz.num_questions_from_pool)
    else:
        selected_questions = pool_questions
    
    # Randomize question order if specified
    if quiz.randomize_questions:
        random.shuffle(selected_questions)
    
    if request.method == 'POST':
        hours, minutes = map(int, quiz.time_duration.split(':'))
        total_minutes = hours * 60 + minutes 
        attempt = QuizAttempt(
            user_id=user_id,
            quiz_id=quiz_id,
            score=0,
            total_questions=len(selected_questions),
            percentage=0,
            start_time=datetime.strptime(request.form.get('start_time'), '%Y-%m-%d %H:%M:%S'),
            end_time=datetime.utcnow() + timedelta(minutes=total_minutes)
        )
        db.session.add(attempt)
        db.session.commit()
        
        # Process answers and track AI interactions with OpenVINO enhancements
        score = 0
        hints_used = 0
        explanations_viewed = 0
        ai_interactions_count = 0
        
        for question in selected_questions:
            user_answer = request.form.get(f'question_{question.id}')
            is_correct = user_answer == question.correct_answer
            
            if is_correct:
                score += 1
            
            # Save user answer
            answer = UserAnswer(
                attempt_id=attempt.id,
                question_id=question.id,
                user_answer=user_answer,
                is_correct=is_correct
            )
            db.session.add(answer)
            
            # Track AI hint usage during quiz
            hint_requested = request.form.get(f'hint_used_{question.id}')
            if hint_requested:
                hints_used += 1
                # Generate OpenVINO-powered hint
                hint_content = openvino_service.text_generation(
                    f"Provide a helpful hint for: {question.question_text}"
                )
                
                # Log AI interaction
                ai_interaction = AIInteraction(
                    user_id=user_id,
                    interaction_type='openvino_hint',
                    content=f"Hint for: {question.question_text}",
                    ai_response=hint_content,
                    context_metadata=json.dumps({
                        'question_id': question.id,
                        'quiz_id': quiz_id,
                        'processing_device': openvino_service.device
                    })
                )
                db.session.add(ai_interaction)
                ai_interactions_count += 1
            
            # Track explanation requests
            explanation_requested = request.form.get(f'explanation_viewed_{question.id}')
            if explanation_requested:
                explanations_viewed += 1
                # Generate OpenVINO-powered explanation
                explanation_content = openvino_service.text_generation(
                    f"Explain why the correct answer to '{question.question_text}' is {question.correct_answer}"
                )
                
                # Log AI interaction
                ai_interaction = AIInteraction(
                    user_id=user_id,
                    interaction_type='openvino_explanation',
                    content=f"Explanation for: {question.question_text}",
                    ai_response=explanation_content,
                    context_metadata=json.dumps({
                        'question_id': question.id,
                        'quiz_id': quiz_id,
                        'user_answer': user_answer,
                        'correct_answer': question.correct_answer,
                        'processing_device': openvino_service.device
                    })
                )
                db.session.add(ai_interaction)
                ai_interactions_count += 1
        
        # Update the score and percentage
        attempt.score = score
        attempt.percentage = (score / len(selected_questions)) * 100
        db.session.commit()
        
        # Log AI interaction for analytics
        ai_interaction = AIInteraction(
            user_id=user_id,
            interaction_type='quiz_completion',
            content=json.dumps({
                'quiz_id': quiz_id,
                'score': score,
                'hints_used': hints_used,
                'explanations_viewed': explanations_viewed,
                'total_questions': len(selected_questions)
            }),
            ai_response='Quiz completed with AI assistance tracking'
        )
        db.session.add(ai_interaction)
        db.session.commit()
        
        return redirect(url_for('quiz_results', attempt_id=attempt.id))
    
    # Get AI-generated study tips for this quiz
    study_tips = ai_service.generate_study_tips(quiz.name, quiz.chapter.name if quiz.chapter else "General")
    
    return render_template('take_quiz_enhanced.html', 
                         quiz=quiz, 
                         questions=selected_questions, 
                         now=datetime.now(),
                         user=user,
                         study_tips=study_tips)


@app.route('/quiz/results/<int:attempt_id>')
@login_required
def quiz_results(attempt_id):
    attempt = QuizAttempt.query.get_or_404(attempt_id)
    
    # Ensure the user can only see their own results
    if attempt.user_id != session['user_id'] and not db.session.get(User, session['user_id']).is_admin:

        flash('You do not have permission to view these results', 'danger')
        return redirect(url_for('dashboard'))
    
    answers = UserAnswer.query.filter_by(attempt_id=attempt_id).all()
    
    return render_template('quiz_results.html', attempt=attempt, answers=answers)

# AI-Enhanced Routes

@app.route('/api/get_hint/<int:question_id>')
@login_required
def get_hint(question_id):
    """Get AI-generated hint for a specific question"""
    question = PoolQuestion.query.get_or_404(question_id)
    hint = ai_service.generate_hint(question.question_text, [
        question.option_a, question.option_b, question.option_c, question.option_d
    ])
    
    # Log hint request
    interaction = AIInteraction(
        user_id=session['user_id'],
        interaction_type='hint_request',
        content=f"Hint for question {question_id}",
        ai_response=hint,
        context_metadata=json.dumps({'question_id': question_id})
    )
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify({'hint': hint})

@app.route('/api/generate_questions', methods=['POST'])
@admin_required
def generate_ai_questions():
    """Generate AI questions for administrators"""
    data = request.get_json()
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'medium')
    count = data.get('count', 5)
    
    questions = ai_service.generate_quiz_questions(topic, difficulty, count)
    
    return jsonify({'questions': questions})

@app.route('/learning_journal')
@login_required
def learning_journal():
    """Display student's learning journal"""
    user_id = session['user_id']
    journal_entries = LearningJournal.query.filter_by(user_id=user_id).order_by(LearningJournal.created_at.desc()).limit(10).all()
    recent_attempts = QuizAttempt.query.filter_by(user_id=user_id).order_by(QuizAttempt.start_time.desc()).limit(5).all()
    
    # Generate AI insights
    attempts_data = [{'percentage': attempt.percentage, 'subject': attempt.quiz.chapter.subject.name} for attempt in recent_attempts]
    ai_insights = ai_service.analyze_learning_pattern(attempts_data)
    
    return render_template('learning_journal.html', 
                         journal_entries=journal_entries, 
                         ai_insights=ai_insights,
                         recent_attempts=recent_attempts)

@app.route('/flashcards')
@login_required
def flashcards():
    """Display and manage OpenVINO-powered flashcards"""
    user_id = session['user_id']
    user = db.session.get(User, user_id)
    flashcards = Flashcard.query.filter_by(user_id=user_id).all()
    
    # Generate OpenVINO-powered insights about flashcard usage
    if flashcards:
        topics = list(set([card.topic for card in flashcards]))
        performance_data = {
            'total_cards': len(flashcards),
            'topics_covered': len(topics),
            'average_success_rate': sum([card.success_rate for card in flashcards]) / len(flashcards),
            'review_frequency': sum([card.review_count for card in flashcards]) / len(flashcards)
        }
        
        # Use OpenVINO to generate personalized study recommendations
        study_recommendations = openvino_service.text_generation(
            f"Generate study recommendations for a student with {len(flashcards)} flashcards covering {topics}, "
            f"with average success rate of {performance_data['average_success_rate']:.1f}%"
        )
        
        # Log AI interaction for analytics
        ai_interaction = AIInteraction(
            user_id=user_id,
            interaction_type='openvino_flashcard_insights',
            content=f"Flashcard performance analysis",
            ai_response=study_recommendations,
            context_metadata=json.dumps({
                'performance_data': performance_data,
                'processing_device': openvino_service.device,
                'inference_type': 'study_recommendations'
            })
        )
        db.session.add(ai_interaction)
        db.session.commit()
    else:
        performance_data = {}
        study_recommendations = "Create your first flashcards to get personalized AI study recommendations!"
    
    return render_template('flashcards.html', 
                         flashcards=flashcards, 
                         user=user,
                         performance_data=performance_data,
                         study_recommendations=study_recommendations)

@app.route('/api/create_flashcards', methods=['POST'])
@login_required
def create_flashcards():
    """Create OpenVINO-powered AI flashcards from topics"""
    data = request.get_json()
    topic = data.get('topic', '')
    count = data.get('count', 5)
    difficulty = data.get('difficulty', 'medium')
    
    # Use OpenVINO to generate enhanced flashcards
    start_time = datetime.now()
    ai_flashcards = openvino_service.generate_flashcards(topic, difficulty, count)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds() * 1000
    
    # Save flashcards to database
    user_id = session['user_id']
    created_cards = []
    
    for card_data in ai_flashcards:
        flashcard = Flashcard(
            user_id=user_id,
            topic=topic,
            front_text=card_data.get('front', ''),
            back_text=card_data.get('back', ''),
            difficulty=difficulty,
            next_review_date=datetime.now() + timedelta(days=1),
            review_count=0,
            success_rate=0.0
        )
        db.session.add(flashcard)
        created_cards.append(flashcard)
    
    db.session.commit()
    
    # Log AI interaction
    ai_interaction = AIInteraction(
        user_id=user_id,
        interaction_type='openvino_flashcard_generation',
        content=f"Generated {count} flashcards for topic: {topic}",
        ai_response=f"Successfully created {len(created_cards)} flashcards using OpenVINO optimization",
        context_metadata=json.dumps({
            'topic': topic,
            'count': count,
            'difficulty': difficulty,
            'processing_time_ms': processing_time,
            'processing_device': openvino_service.device,
            'created_cards_count': len(created_cards)
        })
    )
    db.session.add(ai_interaction)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'flashcards_created': len(created_cards),
        'processing_time_ms': processing_time,
        'device_used': openvino_service.device,
        'message': f'Created {len(created_cards)} AI-powered flashcards for {topic}'
    })

@app.route('/admin/ai_insights')
@admin_required
def admin_ai_insights():
    """AI insights dashboard for educators"""
    # Get class performance data
    all_attempts = QuizAttempt.query.all()
    
    # Calculate insights
    subject_performance = {}
    for attempt in all_attempts:
        subject_name = attempt.quiz.chapter.subject.name
        if subject_name not in subject_performance:
            subject_performance[subject_name] = []
        subject_performance[subject_name].append(attempt.percentage)
    
    # Generate AI insights for each subject
    insights = {}
    for subject, scores in subject_performance.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        insights[subject] = {
            'average_score': avg_score,
            'student_count': len(scores),
            'mastery_level': 'high' if avg_score >= 80 else 'medium' if avg_score >= 60 else 'low',
            'ai_suggestion': ai_service.generate_personalized_feedback({'percentage': avg_score})
        }
    
    return render_template('admin/ai_insights.html', insights=insights)

@app.route('/admin/content_library')
@admin_required
def content_library():
    """Content library for educators"""
    import json
    user_id = session['user_id']
    content_files = ContentLibrary.query.filter_by(user_id=user_id).all()
    for content in content_files:
        if content.ai_analysis and isinstance(content.ai_analysis, str):
            try:
                content.ai_analysis = json.loads(content.ai_analysis)
            except Exception:
                content.ai_analysis = {}
    return render_template('admin/content_library.html', content_files=content_files)

@app.route('/api/upload_content', methods=['POST'])
@admin_required
def upload_content():
    """Upload and process content with AI analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file (implement proper file handling)
    filename = file.filename
    file_type = filename.split('.')[-1] if '.' in filename else 'unknown'
    
    # AI analysis of content
    ai_analysis = ai_service.process_multimodal_input(file_type, file.read())
    
    # Save to content library
    content = ContentLibrary(
        user_id=session['user_id'],
        title=request.form.get('title', filename),
        content_type=file_type,
        file_path=filename,  # In production, save to proper file storage
        ai_analysis=json.dumps(ai_analysis),
        content_metadata=json.dumps({
            'filename': filename,
            'upload_date': datetime.utcnow().isoformat()
        })
    )
    db.session.add(content)
    db.session.commit()
    
    return jsonify({'success': True, 'analysis': ai_analysis})

@app.route('/api/journal_entry', methods=['POST'])
@login_required
def save_journal_entry():
    """Save learning journal entry"""
    data = request.get_json()
    user_id = session['user_id']
    
    # Generate AI feedback on the journal entry
    ai_feedback = ai_service.generate_personalized_feedback({
        'content': data.get('content', ''),
        'mood': data.get('mood', 'neutral'),
        'difficulty': data.get('difficulty', 'medium')
    })
    
    journal_entry = LearningJournal(
        user_id=user_id,
        content=data.get('content', ''),
        mood=data.get('mood', 'neutral'),
        difficulty_level=data.get('difficulty', 'medium'),
        ai_feedback=ai_feedback,
        tags=data.get('tags', '')
    )
    
    db.session.add(journal_entry)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'ai_feedback': ai_feedback,
        'entry_id': journal_entry.id
    })

@app.route('/api/flashcard/<int:flashcard_id>/mark', methods=['POST'])
@login_required
def mark_flashcard(flashcard_id):
    """Mark flashcard as easy/hard and update spaced repetition"""
    flashcard = Flashcard.query.get_or_404(flashcard_id)
    
    # Ensure user owns this flashcard
    if flashcard.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    difficulty = data.get('difficulty', 'medium')
    
    # Update next review date based on spaced repetition algorithm
    days_to_add = {'easy': 7, 'medium': 3, 'hard': 1}.get(difficulty, 3)
    flashcard.next_review_date = datetime.utcnow() + timedelta(days=days_to_add)
    flashcard.difficulty = difficulty
    flashcard.review_count += 1
    
    db.session.commit()
    
    return jsonify({'success': True, 'next_review': flashcard.next_review_date.isoformat()})

@app.route('/api/flashcard/<int:flashcard_id>', methods=['DELETE'])
@login_required
def delete_flashcard(flashcard_id):
    """Delete a flashcard"""
    flashcard = Flashcard.query.get_or_404(flashcard_id)
    
    # Ensure user owns this flashcard
    if flashcard.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(flashcard)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/ai_chat', methods=['POST'])
@login_required
def ai_chat():
    """Enhanced AI chat with better responses"""
    data = request.get_json()
    user_message = data.get('message', '')
    context = data.get('context', {})
    
    # Use the improved AI service chat response
    ai_response = ai_service.chat_response(user_message, context)
    
    # Log the interaction
    interaction = AIInteraction(
        user_id=session['user_id'],
        interaction_type='chat',
        content=user_message,
        ai_response=ai_response,
        context_metadata=json.dumps(context)
    )
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify({
        'response': ai_response,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/generate_content_questions', methods=['POST'])
@admin_required
def generate_content_questions():
    """Generate quiz questions from uploaded content"""
    data = request.get_json()
    content_id = data.get('content_id')
    
    content = ContentLibrary.query.get_or_404(content_id)
    
    # For demo, use content title as topic
    topic = content.title or 'General'
    questions = ai_service.generate_quiz_questions(topic, count=5)
    
    # Optionally, save questions to DB (not implemented here)
    return jsonify({'success': True, 'count': len(questions), 'questions': questions})

@app.route('/api/create_content_flashcards', methods=['POST'])
@admin_required
def create_content_flashcards():
    """Create flashcards from uploaded content"""
    data = request.get_json()
    content_id = data.get('content_id')
    
    content = ContentLibrary.query.get_or_404(content_id)
    
    topic = content.title or 'General'
    flashcards = ai_service.generate_flashcards(topic, count=5)
    
    # Optionally, save flashcards to DB (not implemented here)
    return jsonify({'success': True, 'count': len(flashcards), 'flashcards': flashcards})

@app.route('/api/content/<int:content_id>', methods=['DELETE'])
@admin_required
def delete_content(content_id):
    """Delete content from library"""
    content = ContentLibrary.query.get_or_404(content_id)
    db.session.delete(content)
    db.session.commit()
    return jsonify({'success': True})

# Enhanced AI API Endpoints with OpenVINO Integration

@app.route('/api/multimodal/voice', methods=['POST'])
@login_required
def process_voice_input():
    """Process voice input for hands-free interaction"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        session_id = request.form.get('session_id', f"session_{session['user_id']}")
        
        # Process voice input using multimodal service
        result = asyncio.run(multimodal_service.process_voice_input(audio_data, session_id))
        
        if result['success']:
            # Log interaction
            interaction = AIInteraction(
                user_id=session['user_id'],
                interaction_type='voice_input',
                content=result.get('transcript', ''),
                ai_response=result.get('response', ''),
                context_metadata=json.dumps({'session_id': session_id})
            )
            db.session.add(interaction)
            db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/multimodal/image', methods=['POST'])
@login_required
def process_image_input():
    """Process image input for visual learning support"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        session_id = request.form.get('session_id', f"session_{session['user_id']}")
        context = request.form.get('context', '')
        
        # Process image using multimodal service
        result = asyncio.run(multimodal_service.process_image_input(image_data, session_id, context))
        
        if result['success']:
            # Log interaction
            interaction = AIInteraction(
                user_id=session['user_id'],
                interaction_type='image_analysis',
                content=f"Image analysis with context: {context}",
                ai_response=json.dumps(result.get('analysis', {})),
                context_metadata=json.dumps({'session_id': session_id})
            )
            db.session.add(interaction)
            db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/openvino/performance', methods=['GET'])
@login_required
def get_openvino_performance():
    """Get OpenVINO performance metrics for benchmarking"""
    try:
        metrics = openvino_service.get_performance_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/real_time_interaction', methods=['POST'])
@login_required
def real_time_interaction():
    """Handle real-time classroom interactions"""
    try:
        data = request.get_json()
        interaction_type = data.get('type', 'question_answering')
        content = data.get('content', {})
        
        # Process using OpenVINO optimized inference
        result = openvino_service.real_time_interaction(interaction_type, content)
        
        # Log interaction
        interaction = AIInteraction(
            user_id=session['user_id'],
            interaction_type=f'real_time_{interaction_type}',
            content=json.dumps(content),
            ai_response=json.dumps(result),
            context_metadata=json.dumps({'latency_ms': result.get('latency_ms', 0)})
        )
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_interactive_lesson', methods=['POST'])
@admin_required
def generate_interactive_lesson():
    """Generate interactive multimodal lesson content"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'medium')
        
        # Generate lesson using multimodal service
        lesson = multimodal_service.create_interactive_lesson(topic, difficulty)
        
        return jsonify(lesson)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/accessibility_features', methods=['POST'])
@login_required
def generate_accessibility_features():
    """Generate accessibility features for inclusive learning"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        user_needs = data.get('user_needs', [])
        
        # Generate accessibility features
        features = multimodal_service.generate_accessibility_features(content, user_needs)
        
        return jsonify(features)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session_analytics/<session_id>')
@login_required
def get_session_analytics(session_id):
    """Get analytics for a learning session"""
    try:
        analytics = multimodal_service.get_session_analytics(session_id)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/enhanced_chat', methods=['POST'])
@login_required
def enhanced_ai_chat():
    """Enhanced AI chat with multimodal support"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', f"session_{session['user_id']}")
        interaction_type = data.get('type', 'chat')
        
        # Process using multimodal service
        result = asyncio.run(multimodal_service.process_text_input(message, session_id, interaction_type))
        
        if result['success']:
            # Log the interaction
            interaction = AIInteraction(
                user_id=session['user_id'],
                interaction_type='enhanced_chat',
                content=message,
                ai_response=result.get('response', ''),
                context_metadata=json.dumps({
                    'session_id': session_id,
                    'suggestions': result.get('suggestions', [])
                })
            )
            db.session.add(interaction)
            db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# OpenVINO-Enhanced AI Endpoints
@app.route('/api/openvino/question_answering', methods=['POST'])
@login_required
def openvino_question_answering():
    """Real-time question answering using OpenVINO optimized models"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', {})
        
        # Use OpenVINO service for fast inference
        start_time = datetime.now()
        response = openvino_service.real_time_interaction("question_answering", {
            "question": question,
            "context": context
        })
        end_time = datetime.now()
        
        # Calculate actual processing time
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Log the interaction
        interaction = AIInteraction(
            user_id=session['user_id'],
            interaction_type='openvino_qa',
            content=question,
            ai_response=response.get('content', ''),
            context_metadata=json.dumps({
                'processing_time_ms': processing_time,
                'device': openvino_service.device,
                'confidence': response.get('confidence', 0.0)
            })
        )
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': response,
            'processing_time_ms': processing_time,
            'device_used': openvino_service.device,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/openvino/summarize', methods=['POST'])
@login_required  
def openvino_summarize():
    """Content summarization using OpenVINO optimized models"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        max_length = data.get('max_length', 200)
        
        start_time = datetime.now()
        summary = openvino_service.summarize_content(content, max_length)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return jsonify({
            'success': True,
            'summary': summary,
            'processing_time_ms': processing_time,
            'original_length': len(content),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(content) if content else 0
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/openvino/generate_quiz', methods=['POST'])
@admin_required
def openvino_generate_quiz():
    """Generate quiz questions using OpenVINO optimized models"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        difficulty = data.get('difficulty', 'medium')
        count = data.get('count', 5)
        
        start_time = datetime.now()
        questions = openvino_service.generate_quiz_questions(topic, difficulty, count)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return jsonify({
            'success': True,
            'questions': questions,
            'processing_time_ms': processing_time,
            'count_generated': len(questions),
            'topic': topic,
            'difficulty': difficulty
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/multimodal/voice_interaction', methods=['POST'])
@login_required
def multimodal_voice_interaction():
    """Handle voice-based multimodal interactions"""
    try:
        # Get audio data from request
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_data = audio_file.read()
        
        # Process using multimodal service
        result = multimodal_service.process_voice_interaction(audio_data)
        
        # Log the interaction
        interaction = AIInteraction(
            user_id=session['user_id'],
            interaction_type='voice_multimodal',
            content=result.get('transcript', ''),
            ai_response=result.get('text_response', ''),
            context_metadata=json.dumps({
                'processing_time_ms': result.get('processing_time_ms', 0),
                'confidence': result.get('confidence', 0.0),
                'modality': 'voice'
            })
        )
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/multimodal/image_analysis', methods=['POST'])
@login_required
def multimodal_image_analysis():
    """Analyze educational images using OpenVINO vision models"""
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        question = data.get('question', '')
        
        # Process using multimodal service
        result = multimodal_service.process_image_interaction(image_data, question)
        
        # Log the interaction
        interaction = AIInteraction(
            user_id=session['user_id'],
            interaction_type='image_analysis',
            content=question or 'Image analysis request',
            ai_response=json.dumps(result.get('analysis', {})),
            context_metadata=json.dumps({
                'processing_time_ms': result.get('processing_time_ms', 0),
                'has_question': bool(question),
                'modality': 'visual'
            })
        )
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/openvino/performance_metrics')
@admin_required
def openvino_performance_metrics():
    """Get OpenVINO performance metrics for benchmarking"""
    try:
        metrics = openvino_service.get_performance_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/multimodal_demo')
@login_required
def multimodal_demo():
    """Demo page for multimodal AI interactions"""
    capabilities = multimodal_service.get_interaction_capabilities()
    return render_template('multimodal_demo.html', capabilities=capabilities)

@app.route('/api/classroom/real_time_interaction', methods=['POST'])
@login_required
def classroom_real_time_interaction():
    """Handle real-time classroom interactions with low latency"""
    try:
        data = request.get_json()
        interaction_type = data.get('type', 'text')
        
        start_time = datetime.now()
        
        if interaction_type == 'live_qa':
            # Use OpenVINO for fast question answering
            question = data.get('question', '')
            response = openvino_service.real_time_interaction("question_answering", {
                "question": question
            })
        elif interaction_type == 'quick_explanation':
            # Generate quick explanations for concepts
            concept = data.get('concept', '')
            response = openvino_service.text_generation(f"Explain {concept} briefly for students")
        else:
            response = {'content': 'Unknown interaction type'}
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Ensure low latency for classroom use
        if processing_time > 100:  # If over 100ms, log for optimization
            app.logger.warning(f"High latency detected: {processing_time}ms for {interaction_type}")
        
        return jsonify({
            'success': True,
            'response': response,
            'processing_time_ms': processing_time,
            'interaction_type': interaction_type,
            'optimized': processing_time < 100,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        
        # Test OpenVINO service
        openvino_status = openvino_service.get_performance_metrics()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app': 'EduAI Pro',
            'version': '1.0.0',
            'database': 'connected',
            'openvino': 'loaded',
            'device': openvino_service.device,
            'ai_demo_mode': app.config.get('AI_DEMO_MODE', True)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

# ...existing code...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host=host, port=port, debug=debug)
