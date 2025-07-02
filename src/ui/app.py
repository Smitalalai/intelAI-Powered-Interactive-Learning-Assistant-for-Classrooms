"""
Streamlit frontend for the AI-Powered Interactive Learning Assistant
"""
import streamlit as st
import requests
import io
import time
import json
import base64
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from configs.config import API_CONFIG, UI_CONFIG

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class LearningAssistantUI:
    """Main UI class for the learning assistant"""
    
    def __init__(self):
        self.session_id = None
        self.initialize_session()
    
    def initialize_session(self):
        """Initialize session state"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = None
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "performance_data" not in st.session_state:
            st.session_state.performance_data = {}
    
    def make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict[str, Any]:
        """Make API request with error handling"""
        try:
            url = f"{API_BASE_URL}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, params=data)
            elif method == "POST":
                if files:
                    response = requests.post(url, data=data, files=files)
                else:
                    response = requests.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return {"error": str(e)}
    
    def create_session(self, user_id: str, subject: str, grade_level: str, interaction_mode: str, 
                      student_id: str = None, personalized: bool = False) -> bool:
        """Create a new learning session with optional personalization"""
        
        if personalized and student_id:
            # Create personalized session
            data = {
                "student_id": student_id,
                "subject": subject,
                "grade_level": grade_level,
                "interaction_mode": interaction_mode
            }
            result = self.make_api_request("/sessions/personalized", "POST", data)
        else:
            # Create standard session
            data = {
                "user_id": user_id,
                "student_id": student_id,
                "subject": subject,
                "grade_level": grade_level,
                "interaction_mode": interaction_mode
            }
            result = self.make_api_request("/sessions", "POST", data)
        
        if "error" not in result:
            st.session_state.session_id = result["session_id"]
            st.session_state.personalized = personalized
            if personalized:
                st.success(f"Personalized session created! Session ID: {result['session_id']}")
            else:
                st.success(f"Session created successfully! Session ID: {result['session_id']}")
            return True
        else:
            st.error(f"Failed to create session: {result['error']}")
            return False
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🎓 AI-Powered Interactive Learning Assistant</h1>', unsafe_allow_html=True)
        
        # Sidebar for session management
        self.render_sidebar()
        
        # Main content area
        if st.session_state.session_id:
            self.render_main_interface()
        else:
            self.render_welcome_page()
    
    def render_sidebar(self):
        """Render sidebar with session management"""
        st.sidebar.header("Session Management")
        
        if st.session_state.session_id:
            st.sidebar.success(f"Active Session: {st.session_state.session_id}")
            
            # Show personalization status
            if hasattr(st.session_state, 'personalized') and st.session_state.personalized:
                st.sidebar.info("🎯 Personalization Active")
            
            if st.sidebar.button("End Session"):
                st.session_state.session_id = None
                st.session_state.conversation_history = []
                st.session_state.personalized = False
                st.rerun()
        else:
            st.sidebar.info("No active session")
        
        # Session creation form
        with st.sidebar.expander("Create New Session", expanded=not st.session_state.session_id):
            # User details
            user_id = st.text_input("User ID (optional)")
            
            # Personalization option
            enable_personalization = st.checkbox("Enable Personalization", value=True)
            student_id = None
            
            if enable_personalization:
                student_id = st.text_input("Student ID", help="Required for personalized learning")
            
            # Session configuration
            subject = st.selectbox("Subject", UI_CONFIG["default_subjects"])
            grade_level = st.selectbox("Grade Level", UI_CONFIG["default_grade_levels"])
            interaction_mode = st.selectbox(
                "Interaction Mode", 
                ["text_only", "voice_only", "multimodal", "classroom"]
            )
            
            if st.button("Create Session", type="primary"):
                if enable_personalization and not student_id:
                    st.error("Student ID is required for personalized sessions")
                else:
                    self.create_session(
                        user_id or "anonymous",
                        subject,
                        grade_level,
                        interaction_mode,
                        student_id,
                        enable_personalization
                    )
                    st.rerun()
    
    def render_main_interface(self):
        """Render the main interface with tabs"""
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💬 Chat", 
            "📚 Content Generation", 
            "🧠 AI Models", 
            "📊 Analytics", 
            "🎯 Personalization",
            "⚡ Performance"
        ])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_content_generation()
        
        with tab3:
            self.render_ai_models_interface()
        
        with tab4:
            self.render_analytics_dashboard()
        
        with tab5:
            self.render_personalization_dashboard()
        
        with tab6:
            self.render_performance_dashboard()
    
    def render_content_generation(self):
        """Render content generation interface"""
        st.header("📚 Content Generation")
        
        generation_type = st.selectbox(
            "What would you like to generate?",
            ["Lesson Plan", "Study Guide", "Quiz", "Study Plan"]
        )
        
        if generation_type == "Lesson Plan":
            self.render_lesson_plan_generator()
        elif generation_type == "Study Guide":
            self.render_study_guide_generator()
        elif generation_type == "Quiz":
            self.render_quiz_generator()
        elif generation_type == "Study Plan":
            self.render_study_plan_generator()
    
    def render_lesson_plan_generator(self):
        """Render lesson plan generation interface"""
        st.subheader("📝 Lesson Plan Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Lesson Topic", placeholder="e.g., Introduction to Photosynthesis")
            duration = st.slider("Duration (minutes)", 15, 120, 50, 5)
        
        with col2:
            st.info("💡 The lesson plan will be adapted based on your session settings and personalization preferences.")
        
        if st.button("Generate Lesson Plan", type="primary"):
            if not topic:
                st.error("Please enter a topic")
                return
            
            with st.spinner("Generating comprehensive lesson plan..."):
                result = self.generate_lesson_plan(topic, duration)
            
            if "error" not in result:
                lesson_plan = result["lesson_plan"]
                
                st.success("✅ Lesson plan generated successfully!")
                
                # Display lesson plan
                st.markdown(f"## {lesson_plan['title']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Subject", lesson_plan['subject'])
                with col2:
                    st.metric("Grade Level", lesson_plan['grade_level'])
                with col3:
                    st.metric("Duration", f"{lesson_plan['duration_minutes']} min")
                
                # Learning objectives
                st.markdown("### 🎯 Learning Objectives")
                for i, objective in enumerate(lesson_plan['learning_objectives'], 1):
                    st.markdown(f"{i}. {objective}")
                
                # Activities
                st.markdown("### 📋 Activities")
                for activity in lesson_plan['activities']:
                    with st.expander(f"{activity['name']} ({activity['duration_minutes']} min)"):
                        st.markdown(f"**Type:** {activity['type']}")
                        st.markdown(f"**Description:** {activity['description']}")
                        st.markdown(f"**Materials:** {', '.join(activity['materials'])}")
                
                # Assessments
                st.markdown("### 📊 Assessments")
                for assessment in lesson_plan['assessments']:
                    st.markdown(f"**{assessment['name']}** ({assessment['type']}): {assessment['description']}")
                
                # Materials needed
                st.markdown("### 📦 Materials Needed")
                st.markdown(", ".join(lesson_plan['materials_needed']))
                
                # Homework
                if lesson_plan.get('homework_assignment'):
                    st.markdown("### 🏠 Homework Assignment")
                    st.markdown(lesson_plan['homework_assignment'])
            else:
                st.error(f"Failed to generate lesson plan: {result['error']}")
    
    def render_study_guide_generator(self):
        """Render study guide generation interface"""
        st.subheader("📖 Study Guide Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Study Topic", placeholder="e.g., Algebra Fundamentals")
            difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"])
        
        with col2:
            st.info("📚 Study guides include key concepts, examples, and practice questions.")
        
        if st.button("Generate Study Guide", type="primary"):
            if not topic:
                st.error("Please enter a topic")
                return
            
            with st.spinner("Creating comprehensive study guide..."):
                result = self.generate_study_guide(topic, difficulty)
            
            if "error" not in result:
                study_guide = result["study_guide"]
                
                st.success("✅ Study guide generated successfully!")
                
                # Display study guide
                st.markdown(f"## {study_guide['title']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Subject", study_guide['subject'])
                with col2:
                    st.metric("Difficulty", study_guide['difficulty_level'])
                with col3:
                    st.metric("Study Time", f"{study_guide['estimated_study_time']} min")
                
                # Key concepts
                st.markdown("### 🔑 Key Concepts")
                for concept in study_guide['key_concepts']:
                    st.markdown(f"• {concept}")
                
                # Topics breakdown
                st.markdown("### 📋 Topics")
                for topic_item in study_guide['topics']:
                    with st.expander(f"{topic_item['name']} ({topic_item['estimated_time']} min)"):
                        st.markdown(f"**Difficulty:** {topic_item['difficulty']}")
                        st.markdown(f"**Description:** {topic_item['description']}")
                
                # Examples
                st.markdown("### 💡 Examples")
                for example in study_guide['examples']:
                    with st.expander(example['title']):
                        st.markdown(example['description'])
                        if 'solution' in example:
                            st.markdown(f"**Solution:** {example['solution']}")
                
                # Practice questions
                st.markdown("### ❓ Practice Questions")
                for i, question in enumerate(study_guide['practice_questions'], 1):
                    with st.expander(f"Question {i}"):
                        st.markdown(question['question'])
                        if st.button(f"Show Answer {i}"):
                            st.markdown(f"**Answer:** {question['answer']}")
                            st.markdown(f"**Explanation:** {question['explanation']}")
            else:
                st.error(f"Failed to generate study guide: {result['error']}")
    
    def render_quiz_generator(self):
        """Render quiz generation interface"""
        st.subheader("🧩 Quiz Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Quiz Topic", placeholder="e.g., World War II")
            num_questions = st.slider("Number of Questions", 5, 20, 10)
        
        with col2:
            if hasattr(st.session_state, 'personalized') and st.session_state.personalized:
                st.success("🎯 Personalized quiz will be generated based on your learning profile!")
            else:
                st.info("💡 Enable personalization for adaptive quiz generation.")
        
        if st.button("Generate Quiz", type="primary"):
            if not topic:
                st.error("Please enter a topic")
                return
            
            with st.spinner("Creating personalized quiz..."):
                result = self.generate_quiz(topic, num_questions)
            
            if "error" not in result:
                quiz = result["quiz"]
                
                st.success("✅ Quiz generated successfully!")
                
                if quiz.get('personalized'):
                    st.info("🎯 This quiz has been personalized for your learning style and difficulty level!")
                
                st.markdown(f"## {quiz['title']}")
                
                # Quiz metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions", quiz.get('total_questions', len(quiz['questions'])))
                with col2:
                    if 'total_points' in quiz:
                        st.metric("Total Points", quiz['total_points'])
                with col3:
                    if 'time_limit_minutes' in quiz:
                        st.metric("Time Limit", f"{quiz['time_limit_minutes']} min")
                
                # Display questions
                st.markdown("### 📝 Questions")
                for i, question in enumerate(quiz['questions'], 1):
                    with st.expander(f"Question {i} - {question.get('difficulty', 'medium')} ({question.get('points', 1)} pts)"):
                        st.markdown(question['question_text'] if 'question_text' in question else question['content']['question'])
                        
                        if question.get('type') == 'multiple_choice':
                            options = question.get('options', question.get('content', {}).get('options', []))
                            for opt in options:
                                st.markdown(f"  {opt}")
                        
                        if st.button(f"Show Answer {i}"):
                            if 'correct_answer' in question:
                                st.success(f"Answer: {question['correct_answer']}")
                            if 'explanation' in question:
                                st.info(f"Explanation: {question['explanation']}")
            else:
                st.error(f"Failed to generate quiz: {result['error']}")
    
    def render_analytics_dashboard(self):
        """Render learning analytics dashboard"""
        st.header("📊 Learning Analytics")
        
        if st.button("Refresh Analytics"):
            with st.spinner("Loading analytics..."):
                analytics = self.get_learning_analytics()
            
            if "error" not in analytics:
                # Session overview
                st.markdown("### 📈 Session Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Session Duration", f"{analytics.get('session_duration', 0):.1f} sec")
                with col2:
                    st.metric("Total Interactions", analytics.get('total_interactions', 0))
                with col3:
                    if 'detected_learning_style' in analytics:
                        st.metric("Learning Style", analytics['detected_learning_style'].title())
                with col4:
                    if 'performance_metrics' in analytics:
                        avg_time = sum(analytics['performance_metrics'].values()) / len(analytics['performance_metrics']) if analytics['performance_metrics'] else 0
                        st.metric("Avg Response Time", f"{avg_time:.2f}s")
                
                # Interaction types breakdown
                if 'interaction_types' in analytics:
                    st.markdown("### 🔄 Interaction Types")
                    
                    interaction_data = analytics['interaction_types']
                    if interaction_data:
                        df_interactions = pd.DataFrame(
                            list(interaction_data.items()),
                            columns=['Type', 'Count']
                        )
                        
                        fig = px.pie(
                            df_interactions, 
                            values='Count', 
                            names='Type',
                            title="Distribution of Interaction Types"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Personalized insights
                if 'personalized_insights' in analytics:
                    st.markdown("### 🎯 Personalized Insights")
                    insights = analytics['personalized_insights']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'engagement_score' in insights:
                            st.metric("Engagement Score", f"{insights['engagement_score']:.2f}")
                        
                        if 'performance_trend' in insights:
                            trend = insights['performance_trend']
                            if trend == "improving":
                                st.success("📈 Performance is improving!")
                            elif trend == "declining":
                                st.warning("📉 Performance needs attention")
                            else:
                                st.info("📊 Performance is stable")
                    
                    with col2:
                        if 'content_preferences' in insights:
                            prefs = insights['content_preferences']
                            if prefs:
                                st.markdown("**Content Preferences:**")
                                for content_type, preference in prefs.items():
                                    st.markdown(f"• {content_type}: {preference:.2f}")
            else:
                st.error(f"Failed to load analytics: {analytics['error']}")
    
    def render_personalization_dashboard(self):
        """Render personalization dashboard"""
        st.header("🎯 Personalization Dashboard")
        
        if not (hasattr(st.session_state, 'personalized') and st.session_state.personalized):
            st.warning("⚠️ Personalization is not enabled for this session. Create a personalized session to access this feature.")
            return
        
        # Get student profile
        profile_result = self.make_api_request(f"/sessions/{st.session_state.session_id}/profile")
        
        if "error" not in profile_result:
            st.success("🎯 Personalization is active!")
            
            # Profile overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Student ID", profile_result['student_id'])
            with col2:
                st.metric("Learning Style", profile_result['learning_style'].title())
            with col3:
                st.metric("Difficulty Level", profile_result['difficulty_preference'].title())
            
            # Session statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Sessions", profile_result['session_count'])
            with col2:
                st.metric("Total Time", f"{profile_result['total_time_spent']:.1f} hours")
            
            # Subject strengths
            if profile_result['subject_strengths']:
                st.markdown("### 📚 Subject Strengths")
                
                strengths_data = profile_result['subject_strengths']
                df_strengths = pd.DataFrame(
                    list(strengths_data.items()),
                    columns=['Subject', 'Strength']
                )
                
                fig = px.bar(
                    df_strengths,
                    x='Subject',
                    y='Strength',
                    title="Subject Performance Overview",
                    color='Strength',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to load profile: {profile_result['error']}")
    
    def render_performance_dashboard(self):
        """Render performance monitoring dashboard"""
        st.header("⚡ Performance Dashboard")
        
        if st.button("Refresh Performance Metrics"):
            with st.spinner("Loading performance data..."):
                metrics = self.make_api_request("/performance/metrics")
            
            if "error" not in metrics:
                system_metrics = metrics.get('system_metrics', {})
                
                # System overview
                st.markdown("### 🖥️ System Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Operations", system_metrics.get('total_operations', 0))
                
                with col2:
                    st.metric("Timestamp", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics.get('timestamp', 0))))
                
                # Operation performance breakdown
                if 'operations' in system_metrics:
                    st.markdown("### 📊 Operation Performance")
                    
                    operations = system_metrics['operations']
                    if operations:
                        # Create performance table
                        perf_data = []
                        for op_name, op_data in operations.items():
                            perf_data.append({
                                'Operation': op_name.replace('_', ' ').title(),
                                'Count': op_data['count'],
                                'Avg Time (s)': f"{op_data['average_time']:.3f}",
                                'Min Time (s)': f"{op_data['min_time']:.3f}",
                                'Max Time (s)': f"{op_data['max_time']:.3f}",
                                'Total Time (s)': f"{op_data['total_time']:.3f}"
                            })
                        
                        df_perf = pd.DataFrame(perf_data)
                        st.dataframe(df_perf, use_container_width=True)
                        
                        # Performance visualization
                        fig = px.bar(
                            df_perf,
                            x='Operation',
                            y='Avg Time (s)',
                            title="Average Response Time by Operation",
                            color='Avg Time (s)',
                            color_continuous_scale='reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Session-specific metrics if available
                if 'session_metrics' in metrics:
                    st.markdown("### 📱 Session Performance")
                    session_data = metrics['session_metrics']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Session Duration", f"{session_data.get('duration', 0):.1f}s")
                    with col2:
                        st.metric("Session Interactions", session_data.get('interactions', 0))
                    with col3:
                        st.metric("Session ID", session_data.get('session_id', 'N/A'))
            else:
                st.error(f"Failed to load performance metrics: {metrics['error']}")

    # ...existing methods continue...
