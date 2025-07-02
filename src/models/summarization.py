"""
Text Summarization model optimized with OpenVINO
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from .base_model import BaseOptimizedModel
from configs.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class SummarizationModel(BaseOptimizedModel):
    """OpenVINO optimized Text Summarization model"""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "CPU"):
        model_name = model_name or MODEL_CONFIG["summarization"]["model_name"]
        super().__init__(model_name, device)
        self.summarization_pipeline = None
        self.max_length = MODEL_CONFIG["summarization"]["max_length"]
        self.min_length = MODEL_CONFIG["summarization"]["min_length"]
        
    def _get_model_type(self) -> str:
        return "summarization"
    
    def _load_original_model(self) -> None:
        """Load original HuggingFace model as fallback"""
        try:
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() and self.device == "GPU" else -1
            )
            logger.info(f"Loaded original summarization model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load original summarization model: {str(e)}")
            raise
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary for given text
        
        Args:
            input_data: Dictionary with 'text' key and optional 'max_length', 'min_length'
            
        Returns:
            Dictionary with 'summary', 'original_length', 'summary_length', 'compression_ratio'
        """
        try:
            text = input_data.get("text", "")
            custom_max_length = input_data.get("max_length", self.max_length)
            custom_min_length = input_data.get("min_length", self.min_length)
            
            if not text or len(text.strip()) < 50:
                return {
                    "summary": "Text too short to summarize effectively.",
                    "original_length": len(text),
                    "summary_length": 0,
                    "compression_ratio": 0.0
                }
            
            # Use OpenVINO optimized model if available
            if self.compiled_model is not None:
                summary = self._predict_openvino(text, custom_max_length, custom_min_length)
            # Fallback to HuggingFace pipeline
            elif self.summarization_pipeline is not None:
                summary = self._predict_huggingface(text, custom_max_length, custom_min_length)
            else:
                raise RuntimeError("No model loaded for summarization")
            
            # Calculate metrics
            original_length = len(text)
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0.0
            
            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": compression_ratio
            }
                
        except Exception as e:
            logger.error(f"Summarization prediction failed: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "original_length": len(input_data.get("text", "")),
                "summary_length": 0,
                "compression_ratio": 0.0
            }
    
    def _predict_openvino(self, text: str, max_length: int, min_length: int) -> str:
        """Generate summary using OpenVINO optimized model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=512,  # Input max length
                truncation=True,
                padding=True,
                return_tensors="np"
            )
            
            # Run inference with OpenVINO
            input_layer = self.compiled_model.inputs[0]
            output_layer = self.compiled_model.outputs[0]
            
            result = self.compiled_model({input_layer.any_name: inputs["input_ids"]})[output_layer]
            
            # Decode the generated summary
            # Note: This is a simplified approach - actual implementation would need
            # proper sequence generation with beam search or sampling
            summary_tokens = np.argmax(result, axis=-1)[0]
            summary = self.tokenizer.decode(summary_tokens, skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            logger.error(f"OpenVINO summarization prediction failed: {str(e)}")
            raise
    
    def _predict_huggingface(self, text: str, max_length: int, min_length: int) -> str:
        """Generate summary using HuggingFace pipeline"""
        try:
            # Split long text into chunks if necessary
            max_input_length = 1024  # BART max input length
            
            if len(text) > max_input_length:
                chunks = self._split_text_into_chunks(text, max_input_length)
                summaries = []
                
                for chunk in chunks:
                    result = self.summarization_pipeline(
                        chunk,
                        max_length=max_length // len(chunks),
                        min_length=min_length // len(chunks),
                        do_sample=False
                    )
                    summaries.append(result[0]["summary_text"])
                
                # Combine chunk summaries
                combined_summary = " ".join(summaries)
                
                # Summarize the combined summary if it's still too long
                if len(combined_summary) > max_length * 2:
                    final_result = self.summarization_pipeline(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    return final_result[0]["summary_text"]
                else:
                    return combined_summary
            else:
                result = self.summarization_pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]["summary_text"]
                
        except Exception as e:
            logger.error(f"HuggingFace summarization prediction failed: {str(e)}")
            raise
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def summarize_lesson(self, lesson_content: str, summary_type: str = "general") -> Dict[str, Any]:
        """
        Generate educational summary with different types
        
        Args:
            lesson_content: The lesson text to summarize
            summary_type: Type of summary ('general', 'key_points', 'review')
        """
        base_result = self.predict({"text": lesson_content})
        
        if summary_type == "key_points":
            # Extract key points
            key_points = self._extract_key_points(lesson_content)
            base_result["key_points"] = key_points
            
        elif summary_type == "review":
            # Generate review-style summary
            review_questions = self._generate_review_questions(base_result["summary"])
            base_result["review_questions"] = review_questions
        
        # Add educational metadata
        base_result.update({
            "summary_type": summary_type,
            "reading_time_original": self._estimate_reading_time(lesson_content),
            "reading_time_summary": self._estimate_reading_time(base_result["summary"]),
            "difficulty_level": self._assess_difficulty(lesson_content)
        })
        
        return base_result
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        # Simple key point extraction based on sentence importance
        sentences = text.split('. ')
        
        # Score sentences based on length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence) > 20:  # Filter out very short sentences
                score = len(sentence) / 100  # Length score
                if i < len(sentences) * 0.3:  # Beginning bonus
                    score += 0.5
                if i > len(sentences) * 0.7:  # End bonus
                    score += 0.3
                
                scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        key_points = [sentence for _, sentence in scored_sentences[:5]]
        
        return key_points
    
    def _generate_review_questions(self, summary: str) -> List[str]:
        """Generate review questions based on summary"""
        questions = []
        
        # Simple question generation based on summary content
        sentences = summary.split('. ')
        
        for sentence in sentences[:3]:  # Generate questions for first 3 sentences
            if len(sentence) > 20:
                # Generate different types of questions
                questions.extend([
                    f"What is the main idea of: '{sentence[:50]}...'?",
                    f"Can you explain the concept mentioned in: '{sentence[:50]}...'?",
                    f"How does this relate to the overall topic: '{sentence[:50]}...'?"
                ])
        
        return questions[:5]  # Limit to 5 questions
    
    def _estimate_reading_time(self, text: str, wpm: int = 200) -> int:
        """Estimate reading time in minutes"""
        word_count = len(text.split())
        reading_time = max(1, word_count // wpm)
        return reading_time
    
    def _assess_difficulty(self, text: str) -> str:
        """Assess text difficulty level"""
        # Simple difficulty assessment based on text characteristics
        words = text.split()
        sentences = text.split('.')
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
        
        if avg_word_length < 5 and avg_sentence_length < 15:
            return "Beginner"
        elif avg_word_length < 6 and avg_sentence_length < 20:
            return "Intermediate"
        else:
            return "Advanced"
    
    def batch_summarize(self, texts: List[str], summary_type: str = "general") -> List[Dict[str, Any]]:
        """Summarize multiple texts"""
        results = []
        for text in texts:
            if summary_type == "general":
                results.append(self.predict({"text": text}))
            else:
                results.append(self.summarize_lesson(text, summary_type))
        return results
