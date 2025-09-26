"""
Model Manager for Cross-Encoder Reranking

This module handles transformer model lifecycle management including loading,
versioning, fine-tuning, and optimization for cross-encoder reranking models.

Key Features:
- Model loading and caching
- Version control and updates
- Memory optimization
- GPU management
- Performance monitoring

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import torch
import torch.nn as nn
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    # Transformers not available - create mock classes
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoConfig = None
    print(f"Warning: transformers not fully available ({e}), running in mock mode")
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass
import psutil
import gc

@dataclass
class ModelConfig:
    """Configuration for model management"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cache_dir: str = "./models/cross_encoder"
    max_sequence_length: int = 512
    device: str = "auto"  # auto, cpu, cuda
    torch_dtype: str = "float32"  # float32, float16
    enable_quantization: bool = False
    model_revision: str = "main"
    trust_remote_code: bool = False

@dataclass 
class ModelStats:
    """Model performance statistics"""
    model_name: str
    load_time: float
    memory_usage: float
    inference_time: float
    batch_throughput: float
    device: str
    parameters: int

class ModelManager:
    """
    Manages transformer models for cross-encoder reranking
    Handles loading, caching, optimization, and monitoring
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create ModelConfig from config dictionary
        if isinstance(config, dict):
            model_config_dict = config.get("cross_encoder", {})
            self.model_config = ModelConfig(
                model_name=model_config_dict.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                max_sequence_length=model_config_dict.get("max_sequence_length", 512),
                device=model_config_dict.get("device", "auto"),
                cache_dir=config.get("model_manager", {}).get("model_cache_dir", "./models/cross_encoder/")
            )
        else:
            self.model_config = ModelConfig()
            
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_pipeline = None
        
        # Model registry for multiple models
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.active_model = None
        
        # Performance tracking
        self.stats = ModelStats(
            model_name="", load_time=0.0, memory_usage=0.0,
            inference_time=0.0, batch_throughput=0.0, device="", parameters=0
        )
        
        # Model versions and metadata
        self.model_metadata = {
            "cross-encoder/ms-marco-MiniLM-L-12-v2": {
                "description": "General purpose cross-encoder",
                "parameters": 33000000,
                "max_length": 512,
                "recommended_batch_size": 16,
                "fine_tuned_domains": ["general"]
            },
            "cross-encoder/ms-marco-electra-base": {
                "description": "ELECTRA-based cross-encoder",
                "parameters": 110000000,
                "max_length": 512,
                "recommended_batch_size": 8,
                "fine_tuned_domains": ["general"]
            },
            "sentence-transformers/msmarco-bert-base-dot-v5": {
                "description": "BERT-based cross-encoder",
                "parameters": 110000000,
                "max_length": 512,
                "recommended_batch_size": 8,
                "fine_tuned_domains": ["general"]
            }
        }
        
        self.logger.info("Model manager initialized")
    
    async def initialize(self):
        """Initialize the model manager with the default model"""
        
        try:
            # Check if transformers is available at module level
            if not TRANSFORMERS_AVAILABLE:
                self.logger.warning("Transformers not available at module level, using mock mode")
                self.AutoTokenizer = None
                self.AutoModel = None
                self.AutoModelForSequenceClassification = None
                self.pipeline = None
                self.pipeline_available = False
                self.logger.info("Running in mock mode without transformers")
                return  # Skip initialization if transformers not available
            
            # Try importing transformers components with fallback
            try:
                self.AutoTokenizer = AutoTokenizer
                self.AutoModel = AutoModelForSequenceClassification  # We don't actually use AutoModel
                self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
                
                # Try pipeline import separately - it may fail even if core imports work
                try:
                    from transformers import pipeline
                    self.pipeline = pipeline
                    self.pipeline_available = True
                    self.logger.info("Transformers with pipeline support loaded successfully")
                except ImportError as pe:
                    self.logger.warning(f"Pipeline import failed: {pe}")
                    self.pipeline = None
                    self.pipeline_available = False
                    self.logger.info("Transformers core loaded, pipeline unavailable - using manual implementation")
                    
            except ImportError as e:
                self.logger.warning(f"Transformers import failed: {e}")
                # Use mock classes for demo
                self.AutoTokenizer = None
                self.AutoModel = None
                self.AutoModelForSequenceClassification = None
                self.pipeline = None
                self.pipeline_available = False
                self.logger.info("Running in mock mode without transformers")
                return  # Skip initialization if transformers not available
                
            self.logger.info("Initializing model manager...")
            
            # Determine device
            self.device = self._determine_device()
            self.logger.info(f"Using device: {self.device}")
            
            # Load the default model
            await self.load_model(self.model_config.model_name)
            
            # Warm up the model
            await self._warmup_model()
            
            self.logger.info("Model manager ready!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}")
            raise
    
    async def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """
        Load a cross-encoder model
        
        Args:
            model_name: HuggingFace model identifier
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        
        # Check if transformers is available
        if not self.AutoTokenizer:
            self.logger.warning("Transformers not available, using mock model")
            # Create mock model entry
            self.loaded_models[model_name] = {
                "model": None,
                "tokenizer": None,
                "device": self.device,
                "load_time": time.time(),
                "last_used": time.time(),
                "memory_usage": 0,
                "config": {"mock": True}
            }
            self.active_model = model_name
            return True
        
        if not force_reload and self.active_model == model_name:
            self.logger.debug(f"Model {model_name} already loaded")
            return True
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Create cache directory
            cache_dir = Path(self.model_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer
            self.logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                revision=self.model_config.model_revision,
                trust_remote_code=self.model_config.trust_remote_code
            )
            
            # Load model configuration
            self.logger.debug("Loading model configuration...")
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                revision=self.model_config.model_revision,
                trust_remote_code=self.model_config.trust_remote_code
            )
            
            # Load model
            self.logger.debug("Loading model weights...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir,
                revision=self.model_config.model_revision,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=getattr(torch, self.model_config.torch_dtype)
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Apply optimizations
            await self._optimize_model()
            
            # Create pipeline for easier inference (if available)
            if self.pipeline_available and self.pipeline:
                self.model_pipeline = self.pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device.type == 'cuda' else -1,
                    return_all_scores=True
                )
            else:
                self.model_pipeline = None
                self.logger.info("Pipeline not available, using manual inference")
            
            # Update active model
            self.active_model = model_name
            
            # Update statistics
            load_time = time.time() - start_time
            self.stats.model_name = model_name
            self.stats.load_time = load_time
            self.stats.device = str(self.device)
            self.stats.parameters = self._count_parameters()
            self.stats.memory_usage = self._get_memory_usage()
            
            # Register model
            self.model_registry[model_name] = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "pipeline": self.model_pipeline,
                "load_time": load_time,
                "last_used": time.time()
            }
            
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _determine_device(self) -> torch.device:
        """Determine the best device for model inference"""
        
        if self.model_config.device == "cpu":
            return torch.device("cpu")
        elif self.model_config.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        else:  # auto
            if torch.cuda.is_available() and self.config.use_gpu:
                device = torch.device("cuda")
                self.logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
                return device
            else:
                return torch.device("cpu")
    
    async def _optimize_model(self):
        """Apply model optimizations"""
        
        try:
            # Apply quantization if enabled
            if self.model_config.enable_quantization:
                self.logger.debug("Applying quantization...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                self.logger.debug("Compiling model for faster inference...")
                self.model = torch.compile(self.model)
            
            self.logger.debug("Model optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Model optimization failed: {e}")
    
    async def _warmup_model(self):
        """Warm up the model with dummy inputs"""
        
        try:
            self.logger.debug("Warming up model...")
            
            dummy_texts = ["sample query", "sample document content"]
            dummy_pairs = [("query", "document")]
            
            # Warm up with a few inference calls
            for i in range(3):
                await self._inference_batch(dummy_pairs)
            
            # Measure inference time
            start_time = time.time()
            await self._inference_batch(dummy_pairs * 8)  # Small batch
            inference_time = (time.time() - start_time) / 8
            
            self.stats.inference_time = inference_time
            self.stats.batch_throughput = 1.0 / inference_time
            
            self.logger.debug("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    async def inference_batch(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Run inference on a batch of query-document pairs
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        
        # Handle mock case or when transformers not available
        if not self.AutoTokenizer:
            return await self._inference_batch(query_doc_pairs)
        
        # Check if model is loaded (skip for mock mode)
        if not hasattr(self, 'model') or not self.model:
            if hasattr(self, 'loaded_models') and self.active_model and self.active_model in self.loaded_models:
                # Mock model case
                return await self._inference_batch(query_doc_pairs)
            else:
                raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return await self._inference_batch(query_doc_pairs)
    
    async def _inference_batch(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """Internal batch inference implementation"""
        
        # Handle mock case
        if not self.AutoTokenizer:
            self.logger.info(f"Mock inference for {len(query_doc_pairs)} pairs")
            # Return mock scores (random but reproducible)
            import hashlib
            scores = []
            for i, (query, doc) in enumerate(query_doc_pairs):
                # Create deterministic but varied mock scores based on content
                content_hash = hashlib.md5((query + doc).encode()).hexdigest()
                score = (int(content_hash[:8], 16) % 1000) / 1000.0  # 0.0-1.0 range
                scores.append(max(0.1, score))  # Ensure minimum score
            return scores
        
        try:
            # Prepare inputs for cross-encoder
            input_texts = []
            for query, doc in query_doc_pairs:
                # Cross-encoder format: [CLS] query [SEP] document [SEP]
                input_text = f"{query} [SEP] {doc}"
                input_texts.append(input_text[:self.model_config.max_sequence_length])
            
            # If model not actually loaded (mock mode), return mock scores
            if not hasattr(self, 'model') or not self.model:
                self.logger.info(f"Mock inference for {len(query_doc_pairs)} pairs (no model loaded)")
                import hashlib
                scores = []
                for i, text in enumerate(input_texts):
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    score = (int(content_hash[:8], 16) % 1000) / 1000.0
                    scores.append(max(0.1, score))
                return scores
            
            # Tokenize inputs
            inputs = self.tokenizer(
                input_texts,
                truncation=True,
                padding=True,
                max_length=self.model_config.max_sequence_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities (assuming binary classification)
                if logits.shape[-1] == 1:
                    # Single output (regression-style)
                    scores = torch.sigmoid(logits).squeeze(-1)
                else:
                    # Multi-class output, take positive class probability
                    scores = torch.softmax(logits, dim=-1)[:, 1]  # Assume positive class is index 1
                
                # Convert to CPU and return as list
                return scores.cpu().tolist()
        
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from the model (if supported)
        Note: Cross-encoders don't typically provide embeddings,
        but we can extract hidden states
        """
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.model_config.max_sequence_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use [CLS] token embeddings from last hidden state
                hidden_states = outputs.hidden_states[-1]  # Last layer
                cls_embeddings = hidden_states[:, 0, :]  # [CLS] token
                
                return cls_embeddings.cpu().numpy()
                
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            return np.array([])
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        
        if not self.model:
            return 0
        
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the current model"""
        
        if not self.active_model:
            return {}
        
        metadata = self.model_metadata.get(self.active_model, {})
        
        return {
            "model_name": self.active_model,
            "device": str(self.device),
            "parameters": self.stats.parameters,
            "memory_usage_mb": self.stats.memory_usage,
            "load_time": self.stats.load_time,
            "inference_time": self.stats.inference_time,
            "batch_throughput": self.stats.batch_throughput,
            "metadata": metadata,
            "config": {
                "max_sequence_length": self.model_config.max_sequence_length,
                "torch_dtype": self.model_config.torch_dtype,
                "quantization": self.model_config.enable_quantization
            }
        }
    
    async def unload_model(self, model_name: Optional[str] = None):
        """Unload a model to free memory"""
        
        target_model = model_name or self.active_model
        
        if target_model in self.model_registry:
            del self.model_registry[target_model]
        
        if target_model == self.active_model:
            self.model = None
            self.tokenizer = None
            self.model_pipeline = None
            self.active_model = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Model {target_model} unloaded")
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        
        if model_name in self.model_registry:
            # Model already loaded, just switch
            model_data = self.model_registry[model_name]
            self.model = model_data["model"]
            self.tokenizer = model_data["tokenizer"]
            self.model_pipeline = model_data["pipeline"]
            self.active_model = model_name
            model_data["last_used"] = time.time()
            
            self.logger.info(f"Switched to model: {model_name}")
            return True
        else:
            # Need to load the model
            return await self.load_model(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available/supported models"""
        
        return list(self.model_metadata.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        
        return list(self.model_registry.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model manager"""
        
        health = {
            "status": "healthy",
            "active_model": self.active_model,
            "device": str(self.device),
            "memory_usage_mb": self._get_memory_usage(),
            "loaded_models_count": len(self.model_registry),
            "issues": []
        }
        
        # Check if model is loaded
        if not self.model:
            health["status"] = "unhealthy"
            health["issues"].append("No model loaded")
        
        # Check memory usage
        if health["memory_usage_mb"] > 8000:  # 8GB threshold
            health["status"] = "degraded"
            health["issues"].append("High memory usage")
        
        # Check CUDA availability
        if self.config.use_gpu and not torch.cuda.is_available():
            health["status"] = "degraded"
            health["issues"].append("GPU requested but not available")
        
        # Test inference
        try:
            if self.model:
                await self._inference_batch([("test", "test")])
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"Inference test failed: {str(e)}")
        
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model manager statistics"""
        
        total_memory = 0
        for model_info in self.model_registry.values():
            total_memory += model_info.get("memory_usage", 0)
        
        # Get max_cached_models from config
        max_models = 3  # default
        if isinstance(self.config, dict):
            max_models = self.config.get("model_manager", {}).get("max_cached_models", 3)
        
        return {
            "active_model": self.active_model,
            "loaded_models_count": len(self.model_registry),
            "loaded_models": list(self.model_registry.keys()),
            "total_memory_usage_mb": total_memory,
            "device": str(self.device),
            "transformers_available": bool(self.AutoTokenizer),
            "pipeline_available": bool(self.pipeline),
            "model_cache_dir": str(self.model_config.cache_dir),
            "max_cached_models": max_models,
            "memory_threshold": 0.85  # default from config
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        # Unload all models
        for model_name in list(self.model_registry.keys()):
            asyncio.create_task(self.unload_model(model_name))
        
        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        self.logger.info("Model manager cleanup completed")

# Factory function
def create_model_manager(config) -> ModelManager:
    """Create and return a configured model manager"""
    return ModelManager(config)