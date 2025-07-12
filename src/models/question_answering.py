"""
Question Answering model optimized with OpenVINO
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

from .base_model import BaseOptimizedModel
from configs.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class QuestionAnsweringModel(BaseOptimizedModel):
    """OpenVINO optimized Question Answering model"""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "CPU"):
        model_name = model_name or MODEL_CONFIG["question_answering"]["model_name"]
        super().__init__(model_name, device)
        self.qa_pipeline = None
        self.max_length = MODEL_CONFIG["question_answering"]["max_length"]
        self.temperature = MODEL_CONFIG["question_answering"]["temperature"]
        
    def _get_model_type(self) -> str:
        return "question_answering"
    
    def _load_original_model(self) -> None:
        """Load original HuggingFace model as fallback"""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() and self.device == "GPU" else -1
            )
            logger.info(f"Loaded original QA model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load original QA model: {str(e)}")
            raise
    
    def predict(self, input_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Predict answer for a given question and context
        
        Args:
            input_data: Dictionary with 'question' and 'context' keys
            
        Returns:
            Dictionary with 'answer', 'confidence', and 'start/end' positions
        """
        try:
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            
            if not question or not context:
                return {
                    "answer": "Please provide both a question and context.",
                    "confidence": 0.0,
                    "start": 0,
                    "end": 0
                }
            
            # Use OpenVINO optimized model if available
            if self.compiled_model is not None:
                return self._predict_openvino(question, context)
            
            # Fallback to HuggingFace pipeline
            elif self.qa_pipeline is not None:
                return self._predict_huggingface(question, context)
            
            else:
                raise RuntimeError("No model loaded for question answering")
                
        except Exception as e:
            logger.error(f"Question answering prediction failed: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "start": 0,
                "end": 0
            }
    
    def _predict_openvino(self, question: str, context: str) -> Dict[str, Any]:
        """Predict using OpenVINO optimized model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="np"
            )
            
            # Run inference
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Get input/output layers
            input_layer = self.compiled_model.inputs[0]
            output_layer = self.compiled_model.outputs[0]
            
            # Run inference
            result = self.compiled_model({input_layer.any_name: input_ids})[output_layer]
            
            # Process outputs (simplified - would need proper start/end logits processing)
            start_logits = result[0, :, 0]
            end_logits = result[0, :, 1] if result.shape[-1] > 1 else start_logits
            
            # Find best answer span
            start_idx = np.argmax(start_logits)
            end_idx = np.argmax(end_logits[start_idx:]) + start_idx
            
            # Decode answer
            answer_tokens = input_ids[0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            confidence = float(np.max(start_logits) + np.max(end_logits)) / 2
            
            return {
                "answer": answer,
                "confidence": confidence,
                "start": int(start_idx),
                "end": int(end_idx)
            }
            
        except Exception as e:
            logger.error(f"OpenVINO QA prediction failed: {str(e)}")
            raise
    
    def _predict_huggingface(self, question: str, context: str) -> Dict[str, Any]:
        """Predict using HuggingFace pipeline"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            return {
                "answer": result["answer"],
                "confidence": float(result["score"]),
                "start": int(result["start"]),
                "end": int(result["end"])
            }
            
        except Exception as e:
            logger.error(f"HuggingFace QA prediction failed: {str(e)}")
            raise
    
    def answer_question(self, question: str, context: str) -> str:
        """Simple interface for getting just the answer"""
        result = self.predict({"question": question, "context": context})
        return result.get("answer", "Unable to find answer.")
    
    def batch_predict(self, batch_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple question-answer pairs"""
        results = []
        for item in batch_data:
            results.append(self.predict(item))
        return results
    
    def get_educational_response(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate educational response with explanation
        """
        base_result = self.predict({"question": question, "context": context})
        
        # Add educational enhancements
        educational_response = {
            **base_result,
            "explanation": self._generate_explanation(question, base_result["answer"]),
            "follow_up_questions": self._generate_follow_up_questions(question, context),
            "learning_tips": self._generate_learning_tips(question, context)
        }
        
        return educational_response
    
    def _generate_explanation(self, question: str, answer: str) -> str:
        """Generate explanation for the answer"""
        # Simple explanation generation (could be enhanced with another model)
        if "what" in question.lower():
            return f"The answer '{answer}' addresses what you're asking about by providing the key information."
        elif "why" in question.lower():
            return f"The answer '{answer}' explains the reasoning or cause behind your question."
        elif "how" in question.lower():
            return f"The answer '{answer}' describes the process or method you're asking about."
        else:
            return f"The answer '{answer}' directly responds to your question."
    
    def _generate_follow_up_questions(self, question: str, context: str) -> List[str]:
        """Generate follow-up questions to enhance learning"""
        follow_ups = []
        
        # Basic follow-up question generation
        if "what" in question.lower():
            follow_ups.extend([
                "Why is this important?",
                "How does this relate to other concepts?",
                "Can you give me an example?"
            ])
        elif "why" in question.lower():
            follow_ups.extend([
                "What are the implications of this?",
                "How could this be different?",
                "What evidence supports this?"
            ])
        elif "how" in question.lower():
            follow_ups.extend([
                "What are the steps involved?",
                "Why does this method work?",
                "Are there alternative approaches?"
            ])
        
        return follow_ups[:3]  # Limit to 3 follow-up questions
    
    def _generate_learning_tips(self, question: str, context: str) -> List[str]:
        """Generate learning tips based on the question"""
        tips = [
            "Try to connect this concept to what you already know",
            "Practice explaining this in your own words",
            "Look for real-world examples of this concept"
        ]
        
        # Add specific tips based on question type
        if "formula" in question.lower() or "equation" in question.lower():
            tips.append("Practice working through similar problems step by step")
        elif "definition" in question.lower() or "meaning" in question.lower():
            tips.append("Create your own examples to illustrate this concept")
        
        return tips[:3]  # Limit to 3 tips
