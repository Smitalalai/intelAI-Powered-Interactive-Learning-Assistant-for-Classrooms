"""
Content Generation Service for Educational Materials
Generates lesson plans, study guides, quizzes, and educational content
"""
import logging
import json
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from configs.config import CONTENT_GENERATION_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class ContentType(Enum):
    LESSON_PLAN = "lesson_plan"
    STUDY_GUIDE = "study_guide"
    QUIZ = "quiz"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    PRACTICE_PROBLEMS = "practice_problems"


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    FILL_BLANK = "fill_blank"


@dataclass
class LessonPlan:
    """Structured lesson plan with activities and assessments"""
    title: str
    subject: str
    grade_level: str
    duration_minutes: int
    learning_objectives: List[str]
    prerequisites: List[str]
    activities: List[Dict[str, Any]]
    assessments: List[Dict[str, Any]]
    materials_needed: List[str]
    homework_assignment: Optional[str] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class StudyGuide:
    """Comprehensive study guide with structured content"""
    title: str
    subject: str
    topics: List[Dict[str, Any]]
    key_concepts: List[str]
    examples: List[Dict[str, Any]]
    practice_questions: List[Dict[str, Any]]
    additional_resources: List[str]
    difficulty_level: str = "medium"
    estimated_study_time: int = 60  # minutes


@dataclass
class Quiz:
    """Interactive quiz with various question types"""
    title: str
    subject: str
    questions: List[Dict[str, Any]]
    total_points: int
    time_limit_minutes: Optional[int]
    difficulty_distribution: Dict[str, float]
    auto_grading: bool = True


class ContentGenerationService:
    """Service for generating educational content and materials"""
    
    def __init__(self, qa_model=None, summarization_model=None):
        self.qa_model = qa_model
        self.summarization_model = summarization_model
        self.content_templates = self._load_content_templates()
        self.subject_taxonomies = self._load_subject_taxonomies()
        
        logger.info("Content Generation Service initialized")
    
    def generate_lesson_plan(self, topic: str, subject: str, grade_level: str, 
                           duration_minutes: int = 50, learning_style: str = "balanced") -> LessonPlan:
        """Generate a comprehensive lesson plan"""
        
        # Generate learning objectives
        objectives = self._generate_learning_objectives(topic, subject, grade_level)
        
        # Generate activities based on learning style and duration
        activities = self._generate_lesson_activities(topic, subject, duration_minutes, learning_style)
        
        # Generate assessments
        assessments = self._generate_lesson_assessments(topic, subject, grade_level)
        
        # Determine prerequisites
        prerequisites = self._determine_prerequisites(topic, subject, grade_level)
        
        # Materials needed
        materials = self._determine_materials_needed(activities, subject)
        
        # Homework assignment
        homework = self._generate_homework_assignment(topic, subject, grade_level)
        
        lesson_plan = LessonPlan(
            title=f"{topic} - {subject} Lesson",
            subject=subject,
            grade_level=grade_level,
            duration_minutes=duration_minutes,
            learning_objectives=objectives,
            prerequisites=prerequisites,
            activities=activities,
            assessments=assessments,
            materials_needed=materials,
            homework_assignment=homework
        )
        
        logger.info(f"Generated lesson plan for {topic} in {subject}")
        return lesson_plan
    
    def generate_study_guide(self, topic: str, subject: str, difficulty_level: str = "medium") -> StudyGuide:
        """Generate a comprehensive study guide"""
        
        # Generate topic breakdown
        topics = self._generate_topic_breakdown(topic, subject)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(topic, subject)
        
        # Generate examples
        examples = self._generate_examples(topic, subject, difficulty_level)
        
        # Generate practice questions
        practice_questions = self._generate_practice_questions(topic, subject, difficulty_level)
        
        # Additional resources
        resources = self._generate_additional_resources(topic, subject)
        
        # Estimate study time
        study_time = self._estimate_study_time(topics, examples, practice_questions)
        
        study_guide = StudyGuide(
            title=f"Study Guide: {topic}",
            subject=subject,
            topics=topics,
            key_concepts=key_concepts,
            examples=examples,
            practice_questions=practice_questions,
            additional_resources=resources,
            difficulty_level=difficulty_level,
            estimated_study_time=study_time
        )
        
        logger.info(f"Generated study guide for {topic} in {subject}")
        return study_guide
    
    def generate_quiz(self, topic: str, subject: str, num_questions: int = 10, 
                     difficulty_distribution: Dict[str, float] = None) -> Quiz:
        """Generate an interactive quiz"""
        
        if difficulty_distribution is None:
            difficulty_distribution = CONTENT_GENERATION_CONFIG["quiz_generation"]["difficulty_distribution"]
        
        questions = []
        total_points = 0
        
        # Calculate number of questions per difficulty
        easy_count = int(num_questions * difficulty_distribution["easy"])
        medium_count = int(num_questions * difficulty_distribution["medium"])
        hard_count = num_questions - easy_count - medium_count
        
        # Generate questions for each difficulty level
        questions.extend(self._generate_questions_by_difficulty(topic, subject, "easy", easy_count))
        questions.extend(self._generate_questions_by_difficulty(topic, subject, "medium", medium_count))
        questions.extend(self._generate_questions_by_difficulty(topic, subject, "hard", hard_count))
        
        # Shuffle questions
        random.shuffle(questions)
        
        # Calculate total points
        total_points = sum(q["points"] for q in questions)
        
        # Estimate time limit
        time_limit = self._estimate_quiz_time(questions)
        
        quiz = Quiz(
            title=f"{topic} Quiz - {subject}",
            subject=subject,
            questions=questions,
            total_points=total_points,
            time_limit_minutes=time_limit,
            difficulty_distribution=difficulty_distribution
        )
        
        logger.info(f"Generated quiz for {topic} with {num_questions} questions")
        return quiz
    
    def generate_explanation(self, concept: str, subject: str, grade_level: str, 
                           learning_style: str = "balanced") -> Dict[str, Any]:
        """Generate detailed explanation of a concept"""
        
        explanation = {
            "concept": concept,
            "subject": subject,
            "grade_level": grade_level,
            "learning_style": learning_style,
            "definition": self._generate_definition(concept, subject, grade_level),
            "detailed_explanation": self._generate_detailed_explanation(concept, subject, learning_style),
            "examples": self._generate_concept_examples(concept, subject),
            "analogies": self._generate_analogies(concept, subject, grade_level),
            "common_misconceptions": self._generate_misconceptions(concept, subject),
            "related_concepts": self._find_related_concepts(concept, subject),
            "practice_exercises": self._generate_concept_exercises(concept, subject)
        }
        
        # Adapt explanation based on learning style
        if learning_style == "visual":
            explanation["visual_aids"] = self._suggest_visual_aids(concept, subject)
        elif learning_style == "auditory":
            explanation["audio_cues"] = self._suggest_audio_explanations(concept, subject)
        elif learning_style == "kinesthetic":
            explanation["hands_on_activities"] = self._suggest_kinesthetic_activities(concept, subject)
        
        return explanation
    
    def generate_practice_problems(self, topic: str, subject: str, difficulty: str = "medium", 
                                 count: int = 5) -> List[Dict[str, Any]]:
        """Generate practice problems with solutions"""
        
        problems = []
        
        for i in range(count):
            problem = {
                "id": f"problem_{i+1}",
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "problem_statement": self._generate_problem_statement(topic, subject, difficulty),
                "solution_steps": self._generate_solution_steps(topic, subject, difficulty),
                "final_answer": self._generate_final_answer(topic, subject, difficulty),
                "hints": self._generate_problem_hints(topic, subject, difficulty),
                "estimated_time_minutes": self._estimate_problem_time(difficulty)
            }
            problems.append(problem)
        
        return problems
    
    def _generate_learning_objectives(self, topic: str, subject: str, grade_level: str) -> List[str]:
        """Generate learning objectives for a lesson"""
        # Base objectives that can be customized based on topic and grade level
        base_objectives = [
            f"Students will understand the fundamental concepts of {topic}",
            f"Students will be able to apply {topic} principles to solve problems",
            f"Students will analyze real-world applications of {topic}",
            f"Students will demonstrate mastery through practical exercises"
        ]
        
        # Customize based on grade level
        if grade_level.lower() in ["elementary", "middle school"]:
            base_objectives = [obj.replace("analyze", "identify") for obj in base_objectives]
            base_objectives = [obj.replace("demonstrate mastery", "show understanding") for obj in base_objectives]
        
        return base_objectives[:3]  # Return top 3 objectives
    
    def _generate_lesson_activities(self, topic: str, subject: str, duration: int, learning_style: str) -> List[Dict[str, Any]]:
        """Generate lesson activities based on duration and learning style"""
        activities = []
        
        # Introduction activity (10% of time)
        intro_time = max(5, int(duration * 0.1))
        activities.append({
            "name": "Introduction and Warm-up",
            "type": "introduction",
            "duration_minutes": intro_time,
            "description": f"Introduce {topic} concepts and activate prior knowledge",
            "materials": ["whiteboard", "presentation slides"]
        })
        
        # Main instruction (40% of time)
        main_time = int(duration * 0.4)
        main_activity = {
            "name": f"Main Instruction: {topic}",
            "type": "instruction",
            "duration_minutes": main_time,
            "description": f"Detailed explanation and demonstration of {topic}",
            "materials": ["textbook", "examples"]
        }
        
        # Adapt main activity based on learning style
        if learning_style == "visual":
            main_activity["materials"].extend(["diagrams", "charts", "visual aids"])
            main_activity["description"] += " with visual demonstrations"
        elif learning_style == "auditory":
            main_activity["materials"].extend(["audio examples", "discussion prompts"])
            main_activity["description"] += " with verbal explanations and discussions"
        elif learning_style == "kinesthetic":
            main_activity["materials"].extend(["manipulatives", "hands-on materials"])
            main_activity["description"] += " with hands-on activities"
        
        activities.append(main_activity)
        
        # Practice activity (35% of time)
        practice_time = int(duration * 0.35)
        activities.append({
            "name": "Guided Practice",
            "type": "practice",
            "duration_minutes": practice_time,
            "description": f"Students practice {topic} with teacher guidance",
            "materials": ["worksheets", "practice problems"]
        })
        
        # Closure (15% of time)
        closure_time = duration - intro_time - main_time - practice_time
        activities.append({
            "name": "Closure and Review",
            "type": "closure",
            "duration_minutes": closure_time,
            "description": f"Summarize key {topic} concepts and preview next lesson",
            "materials": ["summary handout"]
        })
        
        return activities
    
    def _generate_lesson_assessments(self, topic: str, subject: str, grade_level: str) -> List[Dict[str, Any]]:
        """Generate assessments for the lesson"""
        assessments = [
            {
                "type": "formative",
                "name": "Exit Ticket",
                "description": f"Quick check of {topic} understanding",
                "timing": "end_of_lesson",
                "points": 5
            },
            {
                "type": "summative", 
                "name": "Practice Quiz",
                "description": f"Comprehensive assessment of {topic} mastery",
                "timing": "next_class",
                "points": 20
            }
        ]
        
        return assessments
    
    def _determine_prerequisites(self, topic: str, subject: str, grade_level: str) -> List[str]:
        """Determine prerequisite knowledge and skills"""
        # This would ideally use curriculum mapping and learning progressions
        general_prerequisites = [
            f"Basic understanding of fundamental {subject} concepts",
            "Ability to follow multi-step instructions",
            "Previous exposure to related mathematical/scientific concepts"
        ]
        
        return general_prerequisites[:2]  # Return top 2 prerequisites
    
    def _determine_materials_needed(self, activities: List[Dict[str, Any]], subject: str) -> List[str]:
        """Extract unique materials needed from all activities"""
        materials = set()
        
        for activity in activities:
            materials.update(activity.get("materials", []))
        
        # Add subject-specific materials
        if subject.lower() == "science":
            materials.update(["safety goggles", "lab equipment"])
        elif subject.lower() == "mathematics":
            materials.update(["calculator", "graph paper"])
        elif subject.lower() == "english":
            materials.update(["dictionaries", "writing materials"])
        
        return list(materials)
    
    def _generate_homework_assignment(self, topic: str, subject: str, grade_level: str) -> str:
        """Generate appropriate homework assignment"""
        assignments = [
            f"Complete practice problems 1-10 on {topic}",
            f"Read textbook chapter on {topic} and write a one-paragraph summary",
            f"Find one real-world example of {topic} and explain how it applies",
            f"Create a concept map showing relationships between {topic} and previous topics"
        ]
        
        # Select assignment based on grade level
        if grade_level.lower() in ["elementary", "middle school"]:
            return assignments[0]  # Simple practice problems
        else:
            return assignments[2]  # Real-world application
    
    def _generate_topic_breakdown(self, topic: str, subject: str) -> List[Dict[str, Any]]:
        """Break down topic into subtopics"""
        subtopics = [
            {
                "name": f"Introduction to {topic}",
                "description": f"Basic concepts and definitions related to {topic}",
                "difficulty": "easy",
                "estimated_time": 15
            },
            {
                "name": f"Core Principles of {topic}",
                "description": f"Main principles and theories underlying {topic}",
                "difficulty": "medium",
                "estimated_time": 25
            },
            {
                "name": f"Applications of {topic}",
                "description": f"Real-world applications and problem-solving with {topic}",
                "difficulty": "hard",
                "estimated_time": 20
            }
        ]
        
        return subtopics
    
    def _extract_key_concepts(self, topic: str, subject: str) -> List[str]:
        """Extract key concepts for the topic"""
        # This would ideally use NLP to extract from educational content
        concepts = [
            f"Definition and scope of {topic}",
            f"Key terminology related to {topic}",
            f"Fundamental principles of {topic}",
            f"Common applications of {topic}",
            f"Relationship to other {subject} concepts"
        ]
        
        return concepts
    
    def _generate_examples(self, topic: str, subject: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate examples for the topic"""
        examples = []
        
        complexity_levels = {
            "easy": "Basic",
            "medium": "Intermediate", 
            "hard": "Advanced"
        }
        
        for i in range(3):
            example = {
                "title": f"{complexity_levels[difficulty]} Example {i+1}",
                "topic": topic,
                "description": f"Example demonstrating {topic} at {difficulty} level",
                "solution": f"Step-by-step solution for {topic} example",
                "difficulty": difficulty
            }
            examples.append(example)
        
        return examples
    
    def _generate_practice_questions(self, topic: str, subject: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate practice questions for study guide"""
        questions = []
        
        for i in range(5):
            question = {
                "id": f"practice_{i+1}",
                "question": f"Practice question {i+1} about {topic}",
                "type": "short_answer",
                "difficulty": difficulty,
                "answer": f"Sample answer for {topic} question {i+1}",
                "explanation": f"Explanation of why this answer is correct for {topic}"
            }
            questions.append(question)
        
        return questions
    
    def _generate_additional_resources(self, topic: str, subject: str) -> List[str]:
        """Generate list of additional learning resources"""
        resources = [
            f"Textbook chapter on {topic}",
            f"Online videos about {topic}",
            f"Interactive simulations for {topic}",
            f"Practice websites for {subject}",
            f"Educational games related to {topic}"
        ]
        
        return resources
    
    def _estimate_study_time(self, topics: List[Dict], examples: List[Dict], questions: List[Dict]) -> int:
        """Estimate total study time in minutes"""
        topic_time = sum(topic.get("estimated_time", 15) for topic in topics)
        example_time = len(examples) * 5  # 5 minutes per example
        question_time = len(questions) * 3  # 3 minutes per question
        
        return topic_time + example_time + question_time
    
    def _generate_questions_by_difficulty(self, topic: str, subject: str, difficulty: str, count: int) -> List[Dict[str, Any]]:
        """Generate quiz questions for specific difficulty level"""
        questions = []
        question_types = CONTENT_GENERATION_CONFIG["quiz_generation"]["question_types"]
        
        points_map = {"easy": 1, "medium": 2, "hard": 3}
        
        for i in range(count):
            question_type = random.choice(question_types)
            
            question = {
                "id": f"{difficulty}_{i+1}",
                "type": question_type,
                "difficulty": difficulty,
                "topic": topic,
                "subject": subject,
                "points": points_map[difficulty],
                "question_text": f"Sample {difficulty} {question_type} question about {topic}",
                "estimated_time": 2 if difficulty == "easy" else 3 if difficulty == "medium" else 5
            }
            
            # Add type-specific fields
            if question_type == "multiple_choice":
                question["options"] = [
                    f"Option A for {topic}",
                    f"Option B for {topic}",
                    f"Option C for {topic}",
                    f"Option D for {topic}"
                ]
                question["correct_answer"] = "A"
            elif question_type == "true_false":
                question["correct_answer"] = random.choice([True, False])
            elif question_type in ["short_answer", "essay"]:
                question["sample_answer"] = f"Sample answer for {topic} question"
                question["grading_rubric"] = self._generate_grading_rubric(question_type, difficulty)
            
            questions.append(question)
        
        return questions
    
    def _estimate_quiz_time(self, questions: List[Dict[str, Any]]) -> int:
        """Estimate time needed to complete quiz"""
        total_time = sum(q.get("estimated_time", 3) for q in questions)
        return max(total_time, 10)  # Minimum 10 minutes
    
    def _generate_grading_rubric(self, question_type: str, difficulty: str) -> Dict[str, Any]:
        """Generate grading rubric for subjective questions"""
        if question_type == "essay":
            return {
                "excellent": "Demonstrates complete understanding with detailed explanations",
                "good": "Shows good understanding with adequate explanations", 
                "fair": "Shows basic understanding with minimal explanations",
                "poor": "Limited understanding or incorrect explanations"
            }
        else:
            return {
                "correct": "Answer demonstrates understanding of concept",
                "partially_correct": "Answer shows some understanding but lacks detail",
                "incorrect": "Answer shows misunderstanding of concept"
            }
    
    def _generate_definition(self, concept: str, subject: str, grade_level: str) -> str:
        """Generate age-appropriate definition"""
        return f"A {grade_level}-appropriate definition of {concept} in the context of {subject}"
    
    def _generate_detailed_explanation(self, concept: str, subject: str, learning_style: str) -> str:
        """Generate detailed explanation adapted to learning style"""
        base_explanation = f"Detailed explanation of {concept} covering its key aspects and importance in {subject}"
        
        if learning_style == "visual":
            return f"{base_explanation} with visual examples and diagrams"
        elif learning_style == "auditory":
            return f"{base_explanation} with verbal descriptions and sound-based analogies"
        elif learning_style == "kinesthetic":
            return f"{base_explanation} with hands-on examples and physical demonstrations"
        else:
            return base_explanation
    
    def _generate_concept_examples(self, concept: str, subject: str) -> List[str]:
        """Generate examples illustrating the concept"""
        return [
            f"Example 1: Basic application of {concept}",
            f"Example 2: Real-world use of {concept}",
            f"Example 3: Advanced application of {concept}"
        ]
    
    def _generate_analogies(self, concept: str, subject: str, grade_level: str) -> List[str]:
        """Generate age-appropriate analogies"""
        return [
            f"Think of {concept} like...",
            f"{concept} is similar to...",
            f"Imagine {concept} as..."
        ]
    
    def _generate_misconceptions(self, concept: str, subject: str) -> List[str]:
        """Generate common misconceptions about the concept"""
        return [
            f"Common misconception 1 about {concept}",
            f"Students often confuse {concept} with...",
            f"A frequent error when learning {concept} is..."
        ]
    
    def _find_related_concepts(self, concept: str, subject: str) -> List[str]:
        """Find concepts related to the given concept"""
        return [
            f"Related concept 1 in {subject}",
            f"Related concept 2 in {subject}",
            f"Related concept 3 in {subject}"
        ]
    
    def _generate_concept_exercises(self, concept: str, subject: str) -> List[Dict[str, Any]]:
        """Generate practice exercises for the concept"""
        exercises = []
        
        for i in range(3):
            exercise = {
                "name": f"Exercise {i+1}",
                "description": f"Practice exercise for {concept}",
                "difficulty": ["easy", "medium", "hard"][i],
                "estimated_time": [10, 15, 20][i]
            }
            exercises.append(exercise)
        
        return exercises
    
    def _suggest_visual_aids(self, concept: str, subject: str) -> List[str]:
        """Suggest visual aids for the concept"""
        return [
            f"Diagram showing {concept}",
            f"Chart illustrating {concept}",
            f"Infographic about {concept}"
        ]
    
    def _suggest_audio_explanations(self, concept: str, subject: str) -> List[str]:
        """Suggest audio explanations"""
        return [
            f"Podcast episode about {concept}",
            f"Audio lecture on {concept}",
            f"Recorded explanation of {concept}"
        ]
    
    def _suggest_kinesthetic_activities(self, concept: str, subject: str) -> List[str]:
        """Suggest hands-on activities"""
        return [
            f"Hands-on experiment with {concept}",
            f"Physical model of {concept}",
            f"Interactive simulation of {concept}"
        ]
    
    def _generate_problem_statement(self, topic: str, subject: str, difficulty: str) -> str:
        """Generate problem statement"""
        return f"Sample {difficulty} problem about {topic} in {subject}"
    
    def _generate_solution_steps(self, topic: str, subject: str, difficulty: str) -> List[str]:
        """Generate step-by-step solution"""
        num_steps = {"easy": 3, "medium": 5, "hard": 7}[difficulty]
        return [f"Step {i+1}: Solution step for {topic}" for i in range(num_steps)]
    
    def _generate_final_answer(self, topic: str, subject: str, difficulty: str) -> str:
        """Generate final answer"""
        return f"Final answer for {topic} problem"
    
    def _generate_problem_hints(self, topic: str, subject: str, difficulty: str) -> List[str]:
        """Generate helpful hints"""
        return [
            f"Hint 1: Consider the basic principles of {topic}",
            f"Hint 2: Remember the relationship between {topic} and...",
            f"Hint 3: Check your units and calculations"
        ]
    
    def _estimate_problem_time(self, difficulty: str) -> int:
        """Estimate time to solve problem"""
        time_map = {"easy": 5, "medium": 10, "hard": 15}
        return time_map[difficulty]
    
    def _load_content_templates(self) -> Dict[str, Any]:
        """Load content templates (placeholder for actual template system)"""
        return {
            "lesson_plan": {},
            "study_guide": {},
            "quiz": {},
            "explanation": {}
        }
    
    def _load_subject_taxonomies(self) -> Dict[str, Any]:
        """Load subject-specific learning taxonomies"""
        return {
            "mathematics": ["algebra", "geometry", "calculus", "statistics"],
            "science": ["biology", "chemistry", "physics", "earth_science"],
            "english": ["grammar", "literature", "writing", "reading"],
            "history": ["ancient", "medieval", "modern", "contemporary"],
            "geography": ["physical", "human", "economic", "political"]
        }
