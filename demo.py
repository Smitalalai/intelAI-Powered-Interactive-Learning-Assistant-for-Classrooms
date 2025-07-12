#!/usr/bin/env python3
"""
Demo script for the AI-Powered Interactive Learning Assistant
This script demonstrates all the key features and capabilities.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.learning_assistant import LearningAssistantService, InteractionMode
from src.models.base_model import ModelManager
from configs.config import MODEL_CONFIG, API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningAssistantDemo:
    """Demo class showcasing the learning assistant capabilities"""
    
    def __init__(self):
        """Initialize the demo"""
        self.service = None
        self.session_id = None
        self.demo_results = {}
    
    def print_banner(self):
        """Print demo banner"""
        print("\n" + "="*80)
        print("🎓 AI-POWERED INTERACTIVE LEARNING ASSISTANT DEMO")
        print("   OpenVINO Unnati Hackathon 2025")
        print("="*80)
        print()
        print("This demo showcases:")
        print("• 🤖 Intelligent Question Answering")
        print("• 📝 Content Summarization")
        print("• 🎤 Speech Recognition & Text-to-Speech")
        print("• 🖼️ Image Captioning & Analysis")
        print("• 🌐 Multimodal Interaction")
        print("• ⚡ OpenVINO Optimization")
        print("• 📊 Performance Monitoring")
        print()
    
    def initialize_service(self):
        """Initialize the learning assistant service"""
        try:
            print("🔧 Initializing AI Learning Assistant Service...")
            self.service = LearningAssistantService()
            print("✅ Service initialized successfully!")
            
            # Display model status
            self.show_model_status()
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize service: {str(e)}")
            print("Note: Some models may not be available without proper setup.")
            return False
    
    def show_model_status(self):
        """Show status of all models"""
        print("\n📋 Model Status:")
        print("-" * 40)
        
        try:
            status = self.service.model_manager.get_models_status()
            for model_name, info in status.items():
                status_icon = "✅" if info.get("status") == "loaded" else "❌"
                device = info.get("device", "unknown")
                print(f"{status_icon} {model_name.upper():<20} | Device: {device}")
        except Exception as e:
            print(f"❌ Could not retrieve model status: {str(e)}")
        
        print()
    
    def create_demo_session(self):
        """Create a demo learning session"""
        print("🎯 Creating Demo Learning Session...")
        
        session_config = {
            "user_id": "demo_user",
            "subject": "Science",
            "grade_level": "High School",
            "interaction_mode": "multimodal"
        }
        
        try:
            self.session_id = self.service.create_session(session_config)
            print(f"✅ Session created: {self.session_id}")
            print(f"   Subject: {session_config['subject']}")
            print(f"   Grade: {session_config['grade_level']}")
            print(f"   Mode: {session_config['interaction_mode']}")
            return True
        except Exception as e:
            print(f"❌ Failed to create session: {str(e)}")
            return False
    
    def demo_question_answering(self):
        """Demo question answering capabilities"""
        print("\n" + "="*60)
        print("🤖 QUESTION ANSWERING DEMO")
        print("="*60)
        
        demo_questions = [
            {
                "question": "What is photosynthesis?",
                "context": "Biology lesson on plant processes",
                "mode": "educational"
            },
            {
                "question": "How does gravity work?",
                "context": "Physics fundamentals",
                "mode": "simple"
            },
            {
                "question": "Explain the water cycle",
                "context": "Earth science weather unit",
                "mode": "detailed"
            }
        ]
        
        for i, qa_data in enumerate(demo_questions, 1):
            print(f"\n📝 Question {i}: {qa_data['question']}")
            print(f"   Context: {qa_data['context']}")
            print(f"   Mode: {qa_data['mode']}")
            
            try:
                start_time = time.time()
                result = self.service.process_question(
                    self.session_id, 
                    qa_data['question'], 
                    qa_data['context'], 
                    qa_data['mode']
                )
                processing_time = time.time() - start_time
                
                if "error" not in result:
                    print(f"\n💡 Answer: {result.get('answer', 'No answer provided')}")
                    
                    if "follow_up_questions" in result:
                        print("\n🔄 Follow-up Questions:")
                        for fq in result["follow_up_questions"][:2]:
                            print(f"   • {fq}")
                    
                    print(f"\n⚡ Processing time: {processing_time:.2f}s")
                    confidence = result.get('confidence', 0)
                    print(f"🎯 Confidence: {confidence:.1%}")
                else:
                    print(f"❌ Error: {result['error']}")
                
                self.demo_results[f"qa_{i}"] = {
                    "success": "error" not in result,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                print(f"❌ Exception during Q&A: {str(e)}")
            
            time.sleep(1)  # Small delay between questions
    
    def demo_summarization(self):
        """Demo content summarization"""
        print("\n" + "="*60)
        print("📝 CONTENT SUMMARIZATION DEMO")
        print("="*60)
        
        demo_content = """
        Photosynthesis is a complex biological process that occurs in plants, algae, and some bacteria. 
        During this process, these organisms convert light energy, usually from the sun, into chemical energy 
        stored in glucose molecules. The process involves two main stages: the light-dependent reactions 
        (also called the photo stage) and the light-independent reactions (also called the Calvin cycle). 
        
        In the light-dependent reactions, chlorophyll and other pigments in the chloroplasts absorb photons 
        of light. This energy is used to split water molecules, releasing oxygen as a byproduct and creating 
        energy-rich molecules like ATP and NADPH. These molecules then power the Calvin cycle, where carbon 
        dioxide from the atmosphere is fixed into organic compounds, ultimately producing glucose.
        
        Photosynthesis is crucial for life on Earth as it produces the oxygen we breathe and forms the base 
        of most food chains. It also plays a vital role in the global carbon cycle, helping to regulate 
        atmospheric carbon dioxide levels and climate patterns.
        """
        
        summary_types = ["brief", "key_points", "review"]
        
        for i, summary_type in enumerate(summary_types, 1):
            print(f"\n📊 Summarization {i}: {summary_type.title()} Summary")
            
            try:
                start_time = time.time()
                result = self.service.summarize_content(
                    self.session_id,
                    demo_content,
                    summary_type
                )
                processing_time = time.time() - start_time
                
                if "error" not in result:
                    print(f"\n📋 Summary: {result.get('summary', 'No summary provided')}")
                    
                    if "key_points" in result:
                        print("\n🔑 Key Points:")
                        for point in result["key_points"][:3]:
                            print(f"   • {point}")
                    
                    if "review_questions" in result:
                        print("\n❓ Review Questions:")
                        for question in result["review_questions"][:2]:
                            print(f"   • {question}")
                    
                    # Show compression stats
                    if "compression_ratio" in result:
                        compression = result["compression_ratio"] * 100
                        print(f"\n📈 Compression: {compression:.1f}% reduction")
                    
                    print(f"⚡ Processing time: {processing_time:.2f}s")
                else:
                    print(f"❌ Error: {result['error']}")
                
                self.demo_results[f"summarization_{i}"] = {
                    "success": "error" not in result,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                print(f"❌ Exception during summarization: {str(e)}")
            
            time.sleep(1)
    
    def demo_image_processing(self):
        """Demo image captioning and analysis"""
        print("\n" + "="*60)
        print("🖼️ IMAGE PROCESSING DEMO")
        print("="*60)
        
        # Create sample educational images for demo
        print("🎨 Creating sample educational images...")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a simple math diagram
            img_width, img_height = 300, 200
            image = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Draw a simple geometric shape (triangle)
            triangle_points = [(50, 150), (150, 50), (250, 150)]
            draw.polygon(triangle_points, outline='blue', width=3)
            
            # Add labels
            try:
                # Try to use default font
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((100, 160), "Triangle", fill='black', font=font)
            draw.text((20, 160), "A", fill='red', font=font)
            draw.text((140, 30), "B", fill='red', font=font)
            draw.text((240, 160), "C", fill='red', font=font)
            
            # Convert to bytes for processing
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            print("✅ Sample educational diagram created")
            
            # Test image captioning
            print("\n🔍 Analyzing educational image...")
            
            start_time = time.time()
            result = self.service.process_image(
                self.session_id,
                img_bytes.getvalue(),
                context="Geometry lesson",
                analysis_type="educational"
            )
            processing_time = time.time() - start_time
            
            if "error" not in result:
                print(f"\n🏷️ Caption: {result.get('caption', 'No caption generated')}")
                
                if "educational_score" in result:
                    score = result["educational_score"]
                    print(f"🎓 Educational Relevance: {score:.1%}")
                
                if "detected_keywords" in result and result["detected_keywords"]:
                    print(f"🔤 Keywords: {', '.join(result['detected_keywords'])}")
                
                if "classroom_suggestions" in result and result["classroom_suggestions"]:
                    print("\n💡 Classroom Suggestions:")
                    for suggestion in result["classroom_suggestions"]:
                        print(f"   • {suggestion}")
                
                print(f"\n⚡ Processing time: {processing_time:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
            
            self.demo_results["image_processing"] = {
                "success": "error" not in result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"❌ Exception during image processing: {str(e)}")
            print("Note: Image processing requires PIL and may need additional setup")
    
    def demo_multimodal_interaction(self):
        """Demo multimodal interaction combining text and images"""
        print("\n" + "="*60)
        print("🌐 MULTIMODAL INTERACTION DEMO")
        print("="*60)
        
        try:
            # Create a simple educational image
            from PIL import Image, ImageDraw
            import io
            
            # Create a diagram showing the water cycle
            img = Image.new('RGB', (400, 300), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Draw sun
            draw.ellipse([320, 20, 380, 80], fill='yellow', outline='orange')
            
            # Draw cloud
            draw.ellipse([50, 50, 150, 100], fill='white', outline='gray')
            
            # Draw precipitation lines
            for x in range(60, 140, 10):
                draw.line([(x, 100), (x, 150)], fill='blue', width=2)
            
            # Draw ground
            draw.rectangle([0, 250, 400, 300], fill='brown')
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            print("🎨 Created water cycle diagram")
            
            # Multimodal input combining text and image
            inputs = {
                "text": "Explain what's happening in this diagram",
                "image": img_bytes.getvalue(),
                "context": "Earth science lesson on weather patterns"
            }
            
            print("\n🔄 Processing multimodal input...")
            print(f"   Text: {inputs['text']}")
            print(f"   Context: {inputs['context']}")
            print("   Image: Water cycle diagram")
            
            start_time = time.time()
            result = self.service.process_multimodal_input(self.session_id, inputs)
            processing_time = time.time() - start_time
            
            if "error" not in result:
                print(f"\n🎯 Integrated Response:")
                print(f"{result.get('integrated_response', 'No integrated response')}")
                
                if "components" in result:
                    components = result["components"]
                    
                    if "text" in components or "transcribed_text" in components:
                        text_comp = components.get("text") or components.get("transcribed_text", {})
                        if "answer" in text_comp:
                            print(f"\n💬 Text Analysis: {text_comp['answer']}")
                    
                    if "image" in components:
                        img_comp = components["image"]
                        if "caption" in img_comp:
                            print(f"\n🖼️ Image Analysis: {img_comp['caption']}")
                        if "classroom_suggestions" in img_comp:
                            print("\n🏫 Teaching Suggestions:")
                            for suggestion in img_comp["classroom_suggestions"]:
                                print(f"   • {suggestion}")
                
                print(f"\n⚡ Total processing time: {processing_time:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
            
            self.demo_results["multimodal"] = {
                "success": "error" not in result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"❌ Exception during multimodal demo: {str(e)}")
            print("Note: Multimodal processing requires PIL and may need additional setup")
    
    def demo_performance_monitoring(self):
        """Demo performance monitoring and metrics"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE MONITORING DEMO")
        print("="*60)
        
        try:
            # Run a quick benchmark
            print("🏃 Running performance benchmark...")
            
            test_inputs = {
                "question": "What is artificial intelligence?",
                "content": "AI is a broad field of computer science..." * 10,
                "text": "Sample text for speech synthesis"
            }
            
            start_time = time.time()
            benchmark_results = self.service.model_manager.benchmark_all_models(test_inputs)
            benchmark_time = time.time() - start_time
            
            print(f"\n📈 Benchmark Results (completed in {benchmark_time:.2f}s):")
            print("-" * 50)
            
            for model_name, metrics in benchmark_results.items():
                if isinstance(metrics, dict) and "mean_time" in metrics:
                    print(f"🤖 {model_name.upper()}")
                    print(f"   Mean time: {metrics['mean_time']:.3f}s")
                    print(f"   Min time:  {metrics['min_time']:.3f}s")
                    print(f"   Max time:  {metrics['max_time']:.3f}s")
                    print(f"   Device:    {metrics.get('device', 'unknown')}")
                    print()
            
            # Show demo session performance summary
            if self.demo_results:
                print("🎯 Demo Session Performance Summary:")
                print("-" * 40)
                
                total_time = 0
                successful_operations = 0
                
                for operation, data in self.demo_results.items():
                    if data.get("success", False):
                        successful_operations += 1
                        total_time += data.get("processing_time", 0)
                        print(f"✅ {operation}: {data['processing_time']:.2f}s")
                    else:
                        print(f"❌ {operation}: Failed")
                
                print(f"\n📊 Summary:")
                print(f"   Successful operations: {successful_operations}/{len(self.demo_results)}")
                print(f"   Total processing time: {total_time:.2f}s")
                print(f"   Average per operation: {total_time/max(successful_operations, 1):.2f}s")
        
        except Exception as e:
            print(f"❌ Exception during performance monitoring: {str(e)}")
    
    def show_demo_summary(self):
        """Show demo completion summary"""
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETED!")
        print("="*80)
        
        print("\n✨ Features Demonstrated:")
        print("• ✅ Question Answering with context-aware responses")
        print("• ✅ Content Summarization with multiple output formats")
        print("• ✅ Image Analysis for educational content")
        print("• ✅ Multimodal Processing combining text and visuals")
        print("• ✅ Performance Monitoring and Benchmarking")
        print("• ✅ Session Management and Context Tracking")
        
        print("\n🚀 Key Benefits:")
        print("• 🧠 AI-powered educational assistance")
        print("• ⚡ OpenVINO optimization for performance")
        print("• 🎯 Adaptive responses based on grade level")
        print("• 🌐 Multimodal interaction capabilities")
        print("• 📊 Real-time performance monitoring")
        print("• 🏫 Classroom-ready deployment")
        
        print("\n🛠️ Technical Stack:")
        print("• 🤖 Transformers: Advanced NLP models")
        print("• 🖼️ BLIP: Image captioning and analysis")
        print("• ⚡ OpenVINO: Model optimization")
        print("• 🌐 FastAPI: High-performance web API")
        print("• 🎨 Streamlit: Interactive user interface")
        print("• 📊 Real-time monitoring and analytics")
        
        success_rate = sum(1 for r in self.demo_results.values() if r.get("success", False))
        total_demos = len(self.demo_results)
        
        if total_demos > 0:
            print(f"\n📈 Demo Success Rate: {success_rate}/{total_demos} ({success_rate/total_demos:.1%})")
        
        print("\n🎓 Ready for OpenVINO Unnati Hackathon 2025!")
        print("="*80)
    
    async def run_demo(self):
        """Run the complete demo"""
        self.print_banner()
        
        if not self.initialize_service():
            print("❌ Could not initialize service. Demo will continue with limited functionality.")
        
        if not self.create_demo_session():
            print("❌ Could not create session. Some demos may not work.")
        
        print("\n🚀 Starting Feature Demonstrations...")
        
        # Run all demo features
        self.demo_question_answering()
        self.demo_summarization()
        self.demo_image_processing()
        self.demo_multimodal_interaction()
        self.demo_performance_monitoring()
        
        self.show_demo_summary()


def main():
    """Main demo function"""
    try:
        demo = LearningAssistantDemo()
        asyncio.run(demo.run_demo())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This might be due to missing dependencies or model setup issues.")
        print("Please ensure all requirements are installed and models are configured.")


if __name__ == "__main__":
    main()
