# 🎓 AI-Powered Interactive Learning Assistant for Classrooms

**OpenVINO Unnati Hackathon 2025 Submission - Enhanced Edition**

An intelligent, multimodal learning assistant that leverages OpenVINO optimization to provide real-time educational support through question answering, content summarization, speech processing, image analysis, personalized learning, and automated content generation.

## ✨ Enhanced Features

### 🤖 Core AI Capabilities
- **Intelligent Q&A**: Context-aware question answering with educational enhancements
- **Content Summarization**: Automatic generation of key points, summaries, and review questions  
- **Speech Processing**: Speech-to-text transcription and text-to-speech generation
- **Image Analysis**: Educational image captioning and content analysis using BLIP
- **Multimodal Interaction**: Seamlessly combine text, speech, and visual inputs

### 🎯 Personalized Learning
- **Learning Style Detection**: Automatic detection of visual, auditory, kinesthetic, or reading preferences
- **Adaptive Difficulty**: Dynamic content difficulty adjustment based on student performance
- **Student Profiles**: Comprehensive tracking of learning progress and preferences
- **Personalized Content**: Tailored questions, explanations, and study materials
- **Progress Analytics**: Real-time insights into learning patterns and performance

### 📚 Content Generation
- **Lesson Plan Creation**: Automated generation of comprehensive lesson plans
- **Study Guide Generation**: Personalized study guides with examples and practice questions
- **Quiz Creation**: Adaptive quiz generation with multiple question types
- **Study Plan Development**: Multi-day personalized study schedules
- **Educational Explanations**: Concept explanations adapted to learning styles

### ⚡ OpenVINO Optimization
- **Model Acceleration**: All models optimized with OpenVINO for Intel hardware
- **Performance Monitoring**: Real-time inference time tracking and benchmarking
- **Adaptive Deployment**: Automatic fallback from OpenVINO to original models
- **Memory Efficiency**: Optimized memory usage for classroom environments
- **Hardware Acceleration**: Support for CPU, GPU, and NPU

### 🏫 Educational Features
- **Grade-Level Adaptation**: Responses tailored to different education levels
- **Subject Categorization**: Specialized handling for STEM, humanities, and other subjects
- **Learning Progress Tracking**: Session-based context and performance monitoring
- **Classroom Management**: Multi-student session support
- **Accessibility Features**: Screen reader support, high contrast, keyboard navigation

### 🌐 Architecture
- **Modular Backend**: FastAPI-based REST API with comprehensive endpoints
- **Interactive Frontend**: Streamlit web interface with multimodal input support
- **Scalable Design**: Docker containerization and cloud deployment ready
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Intel CPU (recommended for OpenVINO optimization)
- 8GB+ RAM
- 10GB+ free storage

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelAI-Powered-Interactive-Learning-Assistant-for-Classrooms

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up models (this will download and optimize models)
python scripts/setup_models.py
```

### Running the Application

#### Option 1: Full Application (API + UI)
```bash
# Terminal 1: Start the API server
cd src/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the Streamlit UI
cd src/ui
streamlit run app.py --server.port 8501
```

#### Option 2: Enhanced Demo
```bash
# Run comprehensive demo
python enhanced_demo.py

# Run quick demo
python enhanced_demo.py --quick

# Run demo with API
python enhanced_demo.py --api
```

#### Option 3: Direct API Testing
```bash
# Start API only
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/sessions/personalized" \
  -H "Content-Type: application/json" \
  -d '{"student_id": "demo_student", "subject": "Science", "grade_level": "High School"}'
```

## 📡 API Endpoints

### Session Management
- `POST /sessions` - Create standard session
- `POST /sessions/personalized` - Create personalized session
- `GET /sessions/{session_id}/profile` - Get student profile

### Content Generation
- `POST /generate/lesson-plan` - Generate lesson plan
- `POST /generate/study-guide` - Generate study guide
- `POST /generate/quiz` - Generate personalized quiz
- `POST /generate/study-plan` - Generate study plan

### AI Interactions
- `POST /questions/personalized` - Personalized Q&A
- `POST /summarize` - Content summarization
- `POST /speech/recognize` - Speech recognition
- `POST /speech/synthesize` - Text-to-speech
- `POST /image/analyze` - Image analysis

### Analytics & Monitoring
- `GET /analytics/{session_id}` - Learning analytics
- `GET /performance/metrics` - Performance metrics
- `GET /health` - System health check

## 🎯 Personalization Features

### Learning Style Detection
The system automatically detects student learning preferences:
- **Visual**: Prefers diagrams, charts, and visual aids
- **Auditory**: Benefits from verbal explanations and discussions
- **Kinesthetic**: Learns through hands-on activities
- **Reading**: Prefers text-based learning materials

### Adaptive Difficulty
Content difficulty automatically adjusts based on:
- Student performance history
- Subject-specific strengths
- Grade level appropriateness
- Learning pace

### Student Analytics
Comprehensive tracking includes:
- Engagement scores
- Performance trends
- Content preferences
- Time spent on topics
- Interaction patterns

## 📚 Content Generation Examples

### Lesson Plan Generation
```python
{
    "session_id": "session_123",
    "topic": "Photosynthesis",
    "duration_minutes": 50
}
```

### Study Guide Creation
```python
{
    "session_id": "session_123",
    "topic": "Algebra Fundamentals",
    "difficulty_level": "medium"
}
```

### Personalized Quiz
```python
{
    "session_id": "session_123",
    "topic": "World War II",
    "num_questions": 10
}
```

## ⚡ Performance Optimization

### OpenVINO Integration
- **Model Conversion**: Automatic conversion to OpenVINO IR format
- **Precision Optimization**: FP16/INT8 quantization for faster inference
- **Hardware Acceleration**: Leverages Intel CPU, GPU, and NPU
- **Dynamic Batching**: Optimizes batch sizes for better throughput

### Performance Monitoring
- Real-time inference time tracking
- Memory usage optimization
- Automatic performance alerts
- Benchmark comparisons

### Typical Performance Gains
- **Inference Speed**: 2-4x faster than original models
- **Memory Usage**: 30-50% reduction
- **Model Loading**: 50-70% faster startup

## 🏫 Classroom Features

### Multi-Student Support
- Concurrent session management
- Individual learning profiles
- Class-wide analytics
- Group activity coordination

### Accessibility
- Screen reader compatibility
- High contrast mode
- Keyboard navigation
- Text size adjustment
- Multi-language support

### Teacher Dashboard
- Student progress monitoring
- Performance analytics
- Content generation tools
- Classroom management

## 🔧 Configuration

### Model Configuration
Edit `configs/config.py` to customize:
- Model selections
- OpenVINO settings
- Performance thresholds
- UI preferences

### Personalization Settings
Configure learning adaptation:
- Learning style weights
- Difficulty progression
- Progress tracking parameters

### Performance Tuning
Optimize for your hardware:
- CPU thread allocation
- Memory usage limits
- Inference precision
- Cache settings

## 📊 Analytics & Insights

### Learning Analytics
- Session duration and engagement
- Interaction type breakdown
- Performance trend analysis
- Content preference mapping

### Performance Metrics
- Model inference times
- System resource usage
- Operation success rates
- Error tracking

### Personalization Insights
- Learning style detection
- Difficulty adaptation effectiveness
- Progress tracking accuracy
- Engagement correlation

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_personalization.py
pytest tests/test_content_generation.py
pytest tests/test_performance.py
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build container
docker build -t ai-learning-assistant .

# Run container
docker run -p 8000:8000 -p 8501:8501 ai-learning-assistant
```

### Cloud Deployment
The application is ready for deployment on:
- AWS with EC2/ECS
- Google Cloud Platform
- Microsoft Azure
- Intel DevCloud

## 📈 Roadmap

### Upcoming Features
- [ ] Advanced NLP models integration
- [ ] Real-time collaboration tools
- [ ] Enhanced image analysis
- [ ] Voice interaction improvements
- [ ] Mobile app development
- [ ] Integration with LMS platforms

### Performance Improvements
- [ ] Model quantization to INT8
- [ ] Distributed inference
- [ ] Edge deployment optimization
- [ ] Custom OpenVINO operators

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Intel OpenVINO team for optimization frameworks
- Hugging Face for model hosting and tools
- Streamlit team for the amazing UI framework
- OpenAI for Whisper speech recognition
- All contributors and beta testers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the demo scripts
- Consult the API reference

---

**Built with ❤️ for educators and students worldwide**
