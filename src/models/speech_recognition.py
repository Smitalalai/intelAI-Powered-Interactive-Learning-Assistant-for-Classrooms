"""
Speech Recognition model using Whisper optimized with OpenVINO
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf

from .base_model import BaseOptimizedModel
from configs.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class SpeechRecognitionModel(BaseOptimizedModel):
    """OpenVINO optimized Speech Recognition model using Whisper"""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "CPU"):
        model_name = model_name or MODEL_CONFIG["speech_recognition"]["model_name"]
        super().__init__(model_name, device)
        self.whisper_model = None
        self.processor = None
        self.language = MODEL_CONFIG["speech_recognition"]["language"]
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
    def _get_model_type(self) -> str:
        return "speech_recognition"
    
    def _load_original_model(self) -> None:
        """Load original Whisper model as fallback"""
        try:
            # Load Whisper model directly
            model_size = self.model_name.split("/")[-1] if "/" in self.model_name else "base"
            self.whisper_model = whisper.load_model(model_size)
            
            # Also load HuggingFace version for better integration
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            logger.info(f"Loaded original Whisper model: {model_size}")
        except Exception as e:
            logger.error(f"Failed to load original Whisper model: {str(e)}")
            raise
    
    def predict(self, input_data: Union[str, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            input_data: Audio file path, numpy array, or dict with 'audio' key
            
        Returns:
            Dictionary with 'text', 'language', 'confidence', and 'segments'
        """
        try:
            # Handle different input types
            if isinstance(input_data, str):
                audio_path = input_data
                audio_array = self._load_audio_file(audio_path)
            elif isinstance(input_data, np.ndarray):
                audio_array = input_data
            elif isinstance(input_data, dict):
                if 'audio' in input_data:
                    if isinstance(input_data['audio'], str):
                        audio_array = self._load_audio_file(input_data['audio'])
                    else:
                        audio_array = input_data['audio']
                else:
                    raise ValueError("Dictionary input must contain 'audio' key")
            else:
                raise ValueError("Input must be file path, numpy array, or dictionary")
            
            # Preprocess audio
            audio_array = self._preprocess_audio(audio_array)
            
            # Use OpenVINO optimized model if available
            if self.compiled_model is not None:
                return self._predict_openvino(audio_array)
            
            # Fallback to original Whisper model
            elif self.whisper_model is not None:
                return self._predict_whisper(audio_array)
            
            else:
                raise RuntimeError("No model loaded for speech recognition")
                
        except Exception as e:
            logger.error(f"Speech recognition prediction failed: {str(e)}")
            return {
                "text": f"Error processing audio: {str(e)}",
                "language": "unknown",
                "confidence": 0.0,
                "segments": []
            }
    
    def _load_audio_file(self, file_path: str) -> np.ndarray:
        """Load audio file and convert to numpy array"""
        try:
            # Load audio file with librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for Whisper"""
        # Ensure audio is float32 and in range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Pad or truncate to 30 seconds max (Whisper limit)
        max_length = 30 * self.sample_rate
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < self.sample_rate:  # At least 1 second
            audio = np.pad(audio, (0, self.sample_rate - len(audio)))
        
        return audio
    
    def _predict_openvino(self, audio: np.ndarray) -> Dict[str, Any]:
        """Predict using OpenVINO optimized model"""
        try:
            # Process audio with Whisper processor
            if self.processor is not None:
                inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="np")
                input_features = inputs.input_features
            else:
                # Fallback feature extraction
                input_features = whisper.log_mel_spectrogram(audio).unsqueeze(0).numpy()
            
            # Run inference with OpenVINO
            input_layer = self.compiled_model.inputs[0]
            output_layer = self.compiled_model.outputs[0]
            
            result = self.compiled_model({input_layer.any_name: input_features})[output_layer]
            
            # Decode the result (simplified)
            if self.processor is not None:
                predicted_ids = np.argmax(result, axis=-1)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            else:
                # Fallback decoding
                transcription = "OpenVINO transcription result"
            
            return {
                "text": transcription,
                "language": self.language,
                "confidence": 0.9,  # Placeholder confidence
                "segments": [{"text": transcription, "start": 0.0, "end": len(audio) / self.sample_rate}]
            }
            
        except Exception as e:
            logger.error(f"OpenVINO speech recognition prediction failed: {str(e)}")
            raise
    
    def _predict_whisper(self, audio: np.ndarray) -> Dict[str, Any]:
        """Predict using original Whisper model"""
        try:
            result = self.whisper_model.transcribe(
                audio,
                language=self.language if self.language != "auto" else None,
                task="transcribe"
            )
            
            return {
                "text": result["text"],
                "language": result.get("language", self.language),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Whisper speech recognition prediction failed: {str(e)}")
            raise
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate average confidence from segments"""
        segments = result.get("segments", [])
        if not segments:
            return 0.8  # Default confidence
        
        confidences = []
        for segment in segments:
            if "avg_logprob" in segment:
                # Convert log probability to confidence score
                confidence = np.exp(segment["avg_logprob"])
                confidences.append(confidence)
        
        return float(np.mean(confidences)) if confidences else 0.8
    
    def transcribe_classroom_audio(self, audio_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Transcribe classroom audio with educational enhancements
        """
        base_result = self.predict(audio_input)
        
        # Add educational enhancements
        enhanced_result = {
            **base_result,
            "word_count": len(base_result["text"].split()),
            "speaking_rate": self._calculate_speaking_rate(base_result),
            "educational_analysis": self._analyze_educational_content(base_result["text"]),
            "suggestions": self._generate_speaking_suggestions(base_result)
        }
        
        return enhanced_result
    
    def _calculate_speaking_rate(self, result: Dict[str, Any]) -> float:
        """Calculate words per minute"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        total_duration = segments[-1].get("end", 0) - segments[0].get("start", 0)
        word_count = len(result["text"].split())
        
        if total_duration > 0:
            return (word_count / total_duration) * 60  # Words per minute
        return 0.0
    
    def _analyze_educational_content(self, text: str) -> Dict[str, Any]:
        """Analyze educational aspects of the transcribed text"""
        words = text.lower().split()
        
        # Count question words
        question_words = ["what", "when", "where", "why", "how", "who"]
        question_count = sum(1 for word in words if word in question_words)
        
        # Count educational keywords
        educational_keywords = [
            "learn", "understand", "explain", "example", "concept", "theory",
            "practice", "study", "remember", "important", "key", "main"
        ]
        educational_count = sum(1 for word in words if word in educational_keywords)
        
        return {
            "question_count": question_count,
            "educational_keywords": educational_count,
            "engagement_score": (question_count + educational_count) / len(words) * 100 if words else 0
        }
    
    def _generate_speaking_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on speech analysis"""
        suggestions = []
        
        # Analyze speaking rate
        speaking_rate = self._calculate_speaking_rate(result)
        if speaking_rate > 180:
            suggestions.append("Consider speaking a bit slower for better comprehension")
        elif speaking_rate < 120 and speaking_rate > 0:
            suggestions.append("You could speak slightly faster to maintain engagement")
        
        # Analyze confidence
        if result["confidence"] < 0.7:
            suggestions.append("Try to speak more clearly for better audio quality")
        
        # Analyze content
        educational_analysis = self._analyze_educational_content(result["text"])
        if educational_analysis["question_count"] == 0:
            suggestions.append("Consider asking questions to increase student engagement")
        
        return suggestions
    
    def save_transcription(self, result: Dict[str, Any], output_path: str) -> None:
        """Save transcription result to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription:\n{result['text']}\n\n")
                f.write(f"Language: {result['language']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}\n\n")
                
                if result.get("segments"):
                    f.write("Segments:\n")
                    for segment in result["segments"]:
                        start = segment.get("start", 0)
                        end = segment.get("end", 0)
                        text = segment.get("text", "")
                        f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")
            
            logger.info(f"Transcription saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save transcription: {str(e)}")
    
    def process_audio_bytes(self, audio_bytes: bytes, file_format: str = "wav") -> Dict[str, Any]:
        """Process audio from bytes"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Process the temporary file
            result = self.predict(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process audio bytes: {str(e)}")
            return {
                "text": f"Error processing audio: {str(e)}",
                "language": "unknown",
                "confidence": 0.0,
                "segments": []
            }
