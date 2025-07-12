"""
Base model class for OpenVINO optimization
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import openvino as ov
# Model optimization tools are no longer needed for this implementation
from transformers import AutoTokenizer, AutoModel

from configs.config import MODEL_CONFIG, OPENVINO_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class BaseOptimizedModel(ABC):
    """Base class for OpenVINO optimized models"""
    
    def __init__(self, model_name: str, device: str = "CPU"):
        self.model_name = model_name
        self.device = device
        self.core = ov.Core()
        self.compiled_model = None
        self.tokenizer = None
        self.model_path = MODELS_DIR / f"{model_name.replace('/', '_')}"
        self.openvino_model_path = self.model_path / "openvino_model.xml"
        
        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
        
    def load_model(self) -> None:
        """Load and optimize the model with OpenVINO"""
        try:
            # Check if OpenVINO model already exists
            if self.openvino_model_path.exists():
                logger.info(f"Loading existing OpenVINO model from {self.openvino_model_path}")
                self._load_openvino_model()
            else:
                logger.info(f"Converting and optimizing model: {self.model_name}")
                self._convert_and_optimize()
                
            # Load tokenizer
            self._load_tokenizer()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def _convert_and_optimize(self) -> None:
        """Convert HuggingFace model to OpenVINO format"""
        try:
            logger.info(f"OpenVINO model conversion not implemented, using fallback for {self.model_name}")
            # For now, we'll skip OpenVINO conversion and use original models
            self._load_original_model()
            
        except Exception as e:
            logger.warning(f"OpenVINO conversion failed, falling back to original model: {str(e)}")
            # Fallback to original model loading
            self._load_original_model()
    
    def _load_openvino_model(self) -> None:
        """Load OpenVINO optimized model"""
        model = self.core.read_model(str(self.openvino_model_path))
        self.compiled_model = self.core.compile_model(
            model=model,
            device_name=self.device,
            config={
                "PERFORMANCE_HINT": OPENVINO_CONFIG["performance_hint"],
                "CACHE_DIR": OPENVINO_CONFIG["cache_dir"]
            }
        )
        logger.info(f"Loaded OpenVINO model on {self.device}")
    
    def _load_original_model(self) -> None:
        """Fallback to load original model without OpenVINO optimization"""
        # This would be implemented by subclasses for specific model types
        logger.warning("Using original model without OpenVINO optimization")
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer for the model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
    
    def benchmark_inference(self, input_data: Any, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark model inference performance"""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "device": self.device
        }
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Abstract method for model prediction"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "openvino_optimized": self.compiled_model is not None,
            "model_path": str(self.model_path),
            "config": MODEL_CONFIG.get(self._get_model_type(), {})
        }
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """Return the model type for configuration lookup"""
        pass


class ModelManager:
    """Manages multiple optimized models"""
    
    def __init__(self):
        self.models: Dict[str, BaseOptimizedModel] = {}
        self.available_devices = self._get_available_devices()
        logger.info(f"Available devices: {self.available_devices}")
    
    def _get_available_devices(self) -> list:
        """Get list of available OpenVINO devices"""
        core = ov.Core()
        return core.available_devices
    
    def register_model(self, model_type: str, model: BaseOptimizedModel) -> None:
        """Register a model with the manager"""
        self.models[model_type] = model
        logger.info(f"Registered model: {model_type}")
    
    def get_model(self, model_type: str) -> Optional[BaseOptimizedModel]:
        """Get a registered model"""
        return self.models.get(model_type)
    
    def load_all_models(self) -> None:
        """Load all registered models"""
        for model_type, model in self.models.items():
            try:
                logger.info(f"Loading model: {model_type}")
                model.load_model()
            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {str(e)}")
    
    def benchmark_all_models(self, test_inputs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Benchmark all loaded models"""
        results = {}
        for model_type, model in self.models.items():
            if model_type in test_inputs:
                try:
                    results[model_type] = model.benchmark_inference(test_inputs[model_type])
                except Exception as e:
                    logger.error(f"Benchmarking failed for {model_type}: {str(e)}")
        return results
    
    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        status = {}
        for model_type, model in self.models.items():
            status[model_type] = {
                "loaded": model.compiled_model is not None,
                "info": model.get_model_info()
            }
        return status
