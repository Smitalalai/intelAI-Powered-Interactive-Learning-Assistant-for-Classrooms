from flask import Flask, render_template, request, redirect, url_for, flash, session
from models import db, User, Subject, Chapter, Quiz, QuizAttempt, QuestionPool, PoolQuestion, UserAnswer
from datetime import datetime, timedelta
from functools import wraps
import random
import os

app = Flask(__name__,template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quiz_master.db'
app.config['SECRET_KEY'] = 'smital20'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
            email='admin@quizmaster.com',
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
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash(f'Welcome back, {user.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
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
    user = db.session.get(User, session['user_id'])
    subjects = Subject.query.all()
    attempts = []
    
    if not user.is_admin:
        attempts = db.session.query(QuizAttempt).filter_by(user_id=user.id).order_by(QuizAttempt.start_time.desc()).limit(5).all()
    
    return render_template('dashboard.html', user=user, subjects=subjects, attempts=attempts)

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
        
        # Process answers
        score = 0
        for question in selected_questions:
            user_answer = request.form.get(f'question_{question.id}')
            is_correct = user_answer == question.correct_answer
            if is_correct:
                score += 1
            
            answer = UserAnswer(
                attempt_id=attempt.id,
                question_id=question.id,
                user_answer=user_answer,
                is_correct=is_correct
            )
            db.session.add(answer)
        
        # Update the score and percentage
        attempt.score = score
        attempt.percentage = (score / len(selected_questions)) * 100
        db.session.commit()
        
        return redirect(url_for('quiz_results', attempt_id=attempt.id))
    
    return render_template('take_quiz.html', quiz=quiz, questions=selected_questions, now=datetime.now())


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

with app.app_context():
    db.create_all()
    print("Database tables created")

if __name__ == '__main__':
    app.run(debug=True)
