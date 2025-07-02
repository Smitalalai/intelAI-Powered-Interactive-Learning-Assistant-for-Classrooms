"""
Image Captioning Model using BLIP (Bootstrapping Language-Image Pre-training)
Optimized with OpenVINO for enhanced performance in classroom environments.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64

from .base_model import BaseOptimizedModel

logger = logging.getLogger(__name__)

class ImageCaptioningModel(BaseOptimizedModel):
    """
    Image captioning model using BLIP with OpenVINO optimization.
    Generates descriptive captions for educational images and visual content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image captioning model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)
        self.model_name = config.get('image_captioning_model', 'Salesforce/blip-image-captioning-base')
        self.device = config.get('device', 'cpu')
        self.max_length = config.get('max_caption_length', 50)
        self.num_beams = config.get('num_beams', 4)
        
        # Model components
        self.processor = None
        self.model = None
        self.openvino_model = None
        self._is_loaded = False
        
        # Performance metrics
        self.inference_times = []
        
    def load_model(self) -> bool:
        """
        Load the BLIP model and processor.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading image captioning model: {self.model_name}")
            
            # Load processor and model
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to appropriate device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to('cuda')
            else:
                self.model = self.model.to('cpu')
                self.device = 'cpu'
            
            self.model.eval()
            self._is_loaded = True
            
            logger.info("Image captioning model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading image captioning model: {str(e)}")
            return False
    
    def load_openvino_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load OpenVINO optimized version of the model.
        
        Args:
            model_path: Path to the OpenVINO model files
            
        Returns:
            bool: True if OpenVINO model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = f"models/openvino/{self.model_name.replace('/', '_')}"
            
            model_path = Path(model_path)
            xml_path = model_path / "openvino_model.xml"
            
            if not xml_path.exists():
                logger.warning(f"OpenVINO model not found at {xml_path}")
                return False
            
            # Import OpenVINO
            try:
                from openvino.runtime import Core
                
                # Initialize OpenVINO
                core = Core()
                self.openvino_model = core.read_model(str(xml_path))
                self.compiled_model = core.compile_model(self.openvino_model, "CPU")
                
                # Load processor separately (needed for preprocessing)
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                
                self._is_loaded = True
                logger.info("OpenVINO image captioning model loaded successfully")
                return True
                
            except ImportError:
                logger.warning("OpenVINO not available, falling back to original model")
                return self.load_model()
                
        except Exception as e:
            logger.error(f"Error loading OpenVINO model: {str(e)}")
            return self.load_model()  # Fallback to original model
    
    def preprocess_image(self, image_input: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
        """
        Preprocess image input for the model.
        
        Args:
            image_input: Image as file path, bytes, base64 string, or PIL Image
            
        Returns:
            PIL Image object or None if preprocessing failed
        """
        try:
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Handle base64 encoded image
                    header, data = image_input.split(',', 1)
                    image_bytes = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    # Handle file path
                    image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def generate_caption(self, image_input: Union[str, bytes, Image.Image], 
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a caption for the given image.
        
        Args:
            image_input: Image to caption
            context: Optional context to guide caption generation
            
        Returns:
            Dictionary containing caption and metadata
        """
        if not self._is_loaded:
            if not self.load_openvino_model():
                return {
                    'caption': '',
                    'confidence': 0.0,
                    'error': 'Model not loaded'
                }
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_input)
            if image is None:
                return {
                    'caption': '',
                    'confidence': 0.0,
                    'error': 'Image preprocessing failed'
                }
            
            # Prepare inputs
            if context:
                # Conditional caption generation with context
                text_prompt = f"a picture of {context}"
                inputs = self.processor(image, text_prompt, return_tensors="pt")
            else:
                # Unconditional caption generation
                inputs = self.processor(image, return_tensors="pt")
            
            # Move inputs to device
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            start_time = self._get_current_time()
            
            # Generate caption
            if self.openvino_model and hasattr(self, 'compiled_model'):
                # Use OpenVINO model
                output = self._generate_with_openvino(inputs)
            else:
                # Use original model
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        early_stopping=True
                    )
            
            inference_time = self._get_current_time() - start_time
            self.inference_times.append(inference_time)
            
            # Decode caption
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified heuristic)
            confidence = min(len(caption.split()) / 10.0, 1.0)
            
            return {
                'caption': caption,
                'confidence': confidence,
                'inference_time': inference_time,
                'model_type': 'openvino' if self.openvino_model else 'original'
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {
                'caption': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_with_openvino(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Generate caption using OpenVINO model.
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Generated token IDs
        """
        try:
            # Convert PyTorch tensors to numpy
            input_dict = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.numpy()
                else:
                    input_dict[key] = value
            
            # Run inference
            result = self.compiled_model(input_dict)
            
            # Convert back to PyTorch tensor
            output_key = list(result.keys())[0]
            output = torch.from_numpy(result[output_key])
            
            return output
            
        except Exception as e:
            logger.error(f"Error in OpenVINO inference: {str(e)}")
            # Fallback to original model
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
    
    def analyze_educational_content(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Analyze image for educational content and generate detailed description.
        
        Args:
            image_input: Image to analyze
            
        Returns:
            Dictionary containing educational analysis
        """
        # Generate basic caption
        result = self.generate_caption(image_input)
        
        if result.get('error'):
            return result
        
        caption = result['caption']
        
        # Educational content analysis
        educational_keywords = [
            'diagram', 'chart', 'graph', 'equation', 'formula', 'map',
            'illustration', 'drawing', 'experiment', 'laboratory', 'classroom',
            'book', 'board', 'student', 'teacher', 'science', 'math',
            'history', 'geography', 'biology', 'chemistry', 'physics'
        ]
        
        detected_keywords = [word for word in educational_keywords 
                           if word in caption.lower()]
        
        # Determine educational relevance
        educational_score = min(len(detected_keywords) / 3.0, 1.0)
        
        # Generate suggestions for classroom use
        suggestions = []
        if 'diagram' in caption.lower() or 'chart' in caption.lower():
            suggestions.append("This image could be used to explain concepts visually")
        if 'equation' in caption.lower() or 'formula' in caption.lower():
            suggestions.append("Mathematical content suitable for problem-solving sessions")
        if 'experiment' in caption.lower() or 'laboratory' in caption.lower():
            suggestions.append("Scientific content ideal for hands-on learning activities")
        
        return {
            'caption': caption,
            'confidence': result['confidence'],
            'educational_score': educational_score,
            'detected_keywords': detected_keywords,
            'classroom_suggestions': suggestions,
            'inference_time': result.get('inference_time', 0),
            'model_type': result.get('model_type', 'original')
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        if not self.inference_times:
            return {'error': 'No inference data available'}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'model_loaded': self._is_loaded,
            'device': self.device,
            'using_openvino': self.openvino_model is not None
        }
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self._is_loaded
