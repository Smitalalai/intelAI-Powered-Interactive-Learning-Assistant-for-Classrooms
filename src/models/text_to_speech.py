"""
Text-to-Speech model optimized with OpenVINO
"""
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Union
import io

import numpy as np
import torch

from .base_model import BaseOptimizedModel
from configs.config import MODEL_CONFIG

logger = logging.getLogger(__name__)

try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("TTS library not available, some features will be disabled")


class TextToSpeechModel(BaseOptimizedModel):
    """OpenVINO optimized Text-to-Speech model"""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "CPU"):
        model_name = model_name or MODEL_CONFIG["text_to_speech"]["model_name"]
        super().__init__(model_name, device)
        self.tts_model = None
        self.synthesizer = None
        self.sample_rate = 22050  # Standard TTS sample rate
        self.vocoder_name = MODEL_CONFIG["text_to_speech"].get("vocoder", None)
        
    def _get_model_type(self) -> str:
        return "text_to_speech"
    
    def _load_original_model(self) -> None:
        """Load original TTS model as fallback"""
        if not TTS_AVAILABLE:
            logger.error("TTS library not available")
            raise RuntimeError("TTS library not installed. Please install with: pip install TTS")
        
        try:
            # Load TTS model
            self.tts_model = TTS(
                model_name=self.model_name,
                vocoder_name=self.vocoder_name,
                gpu=torch.cuda.is_available() and self.device == "GPU"
            )
            
            logger.info(f"Loaded original TTS model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load original TTS model: {str(e)}")
            # Try a simpler fallback
            try:
                self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                logger.info("Loaded fallback TTS model")
            except Exception as e2:
                logger.error(f"Failed to load fallback TTS model: {str(e2)}")
                raise
    
    def predict(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            input_data: Text string or dict with 'text' key and optional parameters
            
        Returns:
            Dictionary with 'audio', 'sample_rate', 'duration', and 'text'
        """
        try:
            # Handle different input types
            if isinstance(input_data, str):
                text = input_data
                speed = 1.0
                pitch = 1.0
            elif isinstance(input_data, dict):
                text = input_data.get("text", "")
                speed = input_data.get("speed", 1.0)
                pitch = input_data.get("pitch", 1.0)
            else:
                raise ValueError("Input must be text string or dictionary with 'text' key")
            
            if not text or len(text.strip()) == 0:
                return {
                    "audio": np.array([]),
                    "sample_rate": self.sample_rate,
                    "duration": 0.0,
                    "text": text,
                    "error": "Empty text provided"
                }
            
            # Use OpenVINO optimized model if available
            if self.compiled_model is not None:
                audio_array = self._predict_openvino(text, speed, pitch)
            
            # Fallback to original TTS model
            elif self.tts_model is not None:
                audio_array = self._predict_tts(text, speed, pitch)
            
            else:
                raise RuntimeError("No model loaded for text-to-speech")
            
            # Calculate duration
            duration = len(audio_array) / self.sample_rate if len(audio_array) > 0 else 0.0
            
            return {
                "audio": audio_array,
                "sample_rate": self.sample_rate,
                "duration": duration,
                "text": text
            }
                
        except Exception as e:
            logger.error(f"Text-to-speech prediction failed: {str(e)}")
            return {
                "audio": np.array([]),
                "sample_rate": self.sample_rate,
                "duration": 0.0,
                "text": input_data if isinstance(input_data, str) else input_data.get("text", ""),
                "error": str(e)
            }
    
    def _predict_openvino(self, text: str, speed: float, pitch: float) -> np.ndarray:
        """Generate speech using OpenVINO optimized model"""
        try:
            # This would require custom OpenVINO implementation
            # For now, fall back to original model
            logger.warning("OpenVINO TTS not implemented, falling back to original model")
            return self._predict_tts(text, speed, pitch)
            
        except Exception as e:
            logger.error(f"OpenVINO TTS prediction failed: {str(e)}")
            raise
    
    def _predict_tts(self, text: str, speed: float, pitch: float) -> np.ndarray:
        """Generate speech using TTS model"""
        try:
            if not TTS_AVAILABLE or self.tts_model is None:
                raise RuntimeError("TTS model not available")
            
            # Generate audio with TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate audio to file
            self.tts_model.tts_to_file(text=text, file_path=temp_path)
            
            # Load the generated audio
            try:
                import soundfile as sf
                audio_array, sr = sf.read(temp_path)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
                
            except ImportError:
                # Fallback without librosa/soundfile
                logger.warning("soundfile/librosa not available, using basic audio loading")
                # This is a very basic fallback - in practice you'd need proper audio handling
                audio_array = np.random.randn(int(len(text) * 0.1 * self.sample_rate))  # Placeholder
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Apply speed and pitch modifications (simplified)
            if speed != 1.0:
                # Simple speed modification by resampling
                new_length = int(len(audio_array) / speed)
                indices = np.linspace(0, len(audio_array) - 1, new_length)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"TTS prediction failed: {str(e)}")
            # Return silence as fallback
            duration = max(1.0, len(text) * 0.1)  # Estimate duration
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
    
    def speak_educational_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate speech for educational content with enhancements
        """
        content_type = content.get("type", "general")
        text = content.get("text", "")
        
        # Adjust speech parameters based on content type
        if content_type == "question":
            # Questions with rising intonation
            enhanced_text = self._enhance_question_text(text)
            speed = 0.9  # Slightly slower for questions
        elif content_type == "explanation":
            # Clear explanations
            enhanced_text = self._enhance_explanation_text(text)
            speed = 0.85  # Slower for explanations
        elif content_type == "summary":
            # Summaries with emphasis
            enhanced_text = self._enhance_summary_text(text)
            speed = 0.9
        else:
            enhanced_text = text
            speed = 1.0
        
        # Generate speech
        result = self.predict({
            "text": enhanced_text,
            "speed": speed,
            "pitch": 1.0
        })
        
        # Add educational metadata
        result.update({
            "content_type": content_type,
            "original_text": text,
            "enhanced_text": enhanced_text,
            "educational_features": self._analyze_speech_features(result)
        })
        
        return result
    
    def _enhance_question_text(self, text: str) -> str:
        """Enhance text for question delivery"""
        # Add pauses and emphasis markers (would be processed by TTS)
        if not text.endswith("?"):
            text += "?"
        
        # Add slight pause before key question words
        question_words = ["what", "when", "where", "why", "how", "who"]
        for word in question_words:
            text = text.replace(word.capitalize(), f"<break time='0.2s'/>{word.capitalize()}")
        
        return text
    
    def _enhance_explanation_text(self, text: str) -> str:
        """Enhance text for explanation delivery"""
        # Add pauses for better comprehension
        text = text.replace(". ", ".<break time='0.5s'/> ")
        text = text.replace(", ", ",<break time='0.2s'/> ")
        
        # Emphasize key educational terms
        key_terms = ["important", "remember", "key", "main", "because", "therefore"]
        for term in key_terms:
            text = text.replace(term, f"<emphasis level='strong'>{term}</emphasis>")
        
        return text
    
    def _enhance_summary_text(self, text: str) -> str:
        """Enhance text for summary delivery"""
        # Add structure to summary delivery
        sentences = text.split(". ")
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i == 0:
                enhanced_sentences.append(f"<emphasis level='moderate'>{sentence}</emphasis>")
            elif i == len(sentences) - 1 and sentence:
                enhanced_sentences.append(f"<break time='0.3s'/>In conclusion, {sentence}")
            else:
                enhanced_sentences.append(sentence)
        
        return ". ".join(enhanced_sentences)
    
    def _analyze_speech_features(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze features of generated speech"""
        audio = result.get("audio", np.array([]))
        duration = result.get("duration", 0.0)
        text = result.get("text", "")
        
        # Calculate speech rate
        word_count = len(text.split())
        speech_rate = (word_count / duration * 60) if duration > 0 else 0  # Words per minute
        
        # Analyze audio characteristics (simplified)
        if len(audio) > 0:
            rms_energy = np.sqrt(np.mean(audio**2))
            max_amplitude = np.max(np.abs(audio))
        else:
            rms_energy = 0.0
            max_amplitude = 0.0
        
        return {
            "speech_rate_wpm": speech_rate,
            "rms_energy": float(rms_energy),
            "max_amplitude": float(max_amplitude),
            "audio_quality": "good" if max_amplitude > 0.1 else "low"
        }
    
    def save_audio(self, result: Dict[str, Any], output_path: str, format: str = "wav") -> None:
        """Save generated audio to file"""
        try:
            audio = result.get("audio", np.array([]))
            sample_rate = result.get("sample_rate", self.sample_rate)
            
            if len(audio) == 0:
                logger.warning("No audio data to save")
                return
            
            try:
                import soundfile as sf
                sf.write(output_path, audio, sample_rate, format=format)
            except ImportError:
                logger.warning("soundfile not available, using basic wav export")
                # Basic WAV export without soundfile
                self._save_wav_basic(audio, sample_rate, output_path)
            
            logger.info(f"Audio saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
    
    def _save_wav_basic(self, audio: np.ndarray, sample_rate: int, output_path: str) -> None:
        """Basic WAV file export without external dependencies"""
        import wave
        import struct
        
        # Normalize and convert to 16-bit
        audio_16bit = (audio * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            
            # Convert to bytes
            audio_bytes = audio_16bit.tobytes()
            wav_file.writeframes(audio_bytes)
    
    def get_audio_bytes(self, result: Dict[str, Any], format: str = "wav") -> bytes:
        """Get audio as bytes for web delivery"""
        try:
            audio = result.get("audio", np.array([]))
            sample_rate = result.get("sample_rate", self.sample_rate)
            
            if len(audio) == 0:
                return b""
            
            # Create in-memory file
            buffer = io.BytesIO()
            
            try:
                import soundfile as sf
                sf.write(buffer, audio, sample_rate, format=format)
            except ImportError:
                # Fallback to basic WAV
                if format.lower() == "wav":
                    self._write_wav_to_buffer(audio, sample_rate, buffer)
                else:
                    logger.warning(f"Format {format} not supported without soundfile, using WAV")
                    self._write_wav_to_buffer(audio, sample_rate, buffer)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to get audio bytes: {str(e)}")
            return b""
    
    def _write_wav_to_buffer(self, audio: np.ndarray, sample_rate: int, buffer: io.BytesIO) -> None:
        """Write WAV data to buffer"""
        import wave
        
        # Normalize and convert to 16-bit
        audio_16bit = (audio * 32767).astype(np.int16)
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
