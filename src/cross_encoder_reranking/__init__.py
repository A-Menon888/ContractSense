"""
Cross-Encoder Reranking Module

This module implements a comprehensive cross-encoder reranking system that enhances
the results from Module 7's hybrid retrieval engine using transformer-based models
for optimal relevance scoring and ranking.

Key Components:
- CrossEncoderEngine: Main orchestration engine
- ModelManager: Transformer model lifecycle management  
- RelevanceScorer: Query-document relevance scoring
- LearningToRank: Adaptive ranking optimization
- ExplanationGenerator: Interpretable ranking explanations
- PerformanceMonitor: System performance monitoring

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional

# Import main components
from .cross_encoder_engine import CrossEncoderEngine, create_cross_encoder_engine
from .model_manager import ModelManager, create_model_manager  
from .relevance_scorer import RelevanceScorer, create_relevance_scorer
from .learning_to_rank import LearningToRank, create_learning_to_rank
from .explanation_generator import ExplanationGenerator, create_explanation_generator
from .performance_monitor import PerformanceMonitor, create_performance_monitor, OperationTracker

# Version information
__version__ = "1.0.0"
__author__ = "ContractSense Team"
__email__ = "team@contractsense.ai"

# Module logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "cross_encoder": {
        "model_name": "microsoft/DialoGPT-medium",  # Default fallback
        "device": "auto",  # Auto-detect GPU/CPU
        "max_sequence_length": 512,
        "batch_size": 16,
        "enable_caching": True,
        "cache_size": 10000
    },
    "model_manager": {
        "model_cache_dir": "./models/cross_encoder/",
        "max_cached_models": 3,
        "enable_quantization": True,
        "use_fast_tokenizer": True,
        "model_timeout": 300,
        "memory_threshold": 0.85
    },
    "relevance_scorer": {
        "batch_size": 16,
        "max_sequence_length": 512,
        "score_calibration": True,
        "enable_feature_attribution": True,
        "use_threading": True,
        "max_workers": 4
    },
    "learning_to_rank": {
        "algorithm": "listnet",
        "learning_rate": 0.001,
        "hidden_size": 128,
        "dropout_rate": 0.2,
        "max_epochs": 100,
        "batch_size": 32,
        "feature_dim": 10,
        "online_learning": True
    },
    "explanation_generator": {
        "enable_attention_viz": True,
        "enable_feature_importance": True,
        "enable_text_highlighting": True,
        "max_tokens_to_highlight": 20,
        "explanation_depth": "detailed",
        "include_visualizations": True,
        "output_format": "json"
    },
    "performance_monitor": {
        "enable_monitoring": True,
        "resource_monitoring_interval": 1.0,
        "alert_thresholds": {
            "high_cpu": 85.0,
            "high_memory": 90.0,
            "high_latency": 5.0,
            "high_error_rate": 0.05
        }
    }
}

class CrossEncoderRerankingModule:
    """
    Main module class that coordinates all cross-encoder reranking components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the cross-encoder reranking module"""
        
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.engine = None
        self.model_manager = None
        self.relevance_scorer = None
        self.learning_to_rank = None
        self.explanation_generator = None
        self.performance_monitor = None
        
        # State
        self.initialized = False
        
        self.logger.info("Cross-encoder reranking module created")
    
    async def initialize(self):
        """Initialize all components"""
        
        if self.initialized:
            self.logger.warning("Module already initialized")
            return
        
        try:
            self.logger.info("Initializing cross-encoder reranking module...")
            
            # Initialize performance monitor first
            if self.config.get("performance_monitor", {}).get("enable_monitoring", True):
                self.performance_monitor = create_performance_monitor(self.config)
                await self.performance_monitor.initialize()
                self.logger.info("✓ Performance monitor initialized")
            
            # Initialize model manager
            self.model_manager = create_model_manager(self.config)
            await self.model_manager.initialize()
            self.logger.info("✓ Model manager initialized")
            
            # Initialize relevance scorer
            self.relevance_scorer = create_relevance_scorer(self.config, self.model_manager)
            await self.relevance_scorer.initialize()
            self.logger.info("✓ Relevance scorer initialized")
            
            # Initialize learning-to-rank
            self.learning_to_rank = create_learning_to_rank(self.config)
            await self.learning_to_rank.initialize()
            self.logger.info("✓ Learning-to-rank initialized")
            
            # Initialize explanation generator
            self.explanation_generator = create_explanation_generator(self.config)
            await self.explanation_generator.initialize()
            self.logger.info("✓ Explanation generator initialized")
            
            # Initialize main engine
            self.engine = create_cross_encoder_engine(
                self.config, 
                self.model_manager,
                self.relevance_scorer,
                self.learning_to_rank,
                self.explanation_generator,
                self.performance_monitor
            )
            await self.engine.initialize()
            self.logger.info("✓ Cross-encoder engine initialized")
            
            self.initialized = True
            self.logger.info("🚀 Cross-encoder reranking module ready!")
            
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            await self.cleanup()
            raise
    
    async def rerank(self, query: str, candidates: list, strategy: str = "cross_encoder_only",
                    top_k: Optional[int] = None, explain: bool = False) -> Dict[str, Any]:
        """
        Main reranking interface
        
        Args:
            query: Search query
            candidates: List of candidate documents
            strategy: Reranking strategy to use
            top_k: Number of top results to return
            explain: Whether to include explanations
            
        Returns:
            Reranked results with scores and optional explanations
        """
        
        if not self.initialized:
            raise RuntimeError("Module not initialized. Call initialize() first.")
        
        return await self.engine.rerank(
            query=query,
            candidates=candidates,
            strategy=strategy,
            top_k=top_k,
            explain=explain
        )
    
    async def add_feedback(self, query: str, document_id: str, 
                          feedback_type: str, feedback_value: float):
        """Add user feedback for learning"""
        
        if not self.initialized:
            raise RuntimeError("Module not initialized")
        
        if self.learning_to_rank:
            await self.learning_to_rank.add_feedback(
                query, document_id, feedback_type, feedback_value
            )
    
    async def train_ranker(self, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the learning-to-rank model"""
        
        if not self.initialized:
            raise RuntimeError("Module not initialized")
        
        if self.learning_to_rank:
            return await self.learning_to_rank.train(validation_split)
        
        return {"error": "learning_to_rank_not_available"}
    
    async def get_performance_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        if not self.initialized:
            raise RuntimeError("Module not initialized")
        
        if self.performance_monitor:
            report = await self.performance_monitor.generate_performance_report(time_window_minutes)
            return {
                "time_period": report.time_period,
                "total_operations": report.total_operations,
                "successful_operations": report.successful_operations,
                "failed_operations": report.failed_operations,
                "average_latency": report.average_latency,
                "p95_latency": report.p95_latency,
                "p99_latency": report.p99_latency,
                "throughput": report.throughput,
                "error_rate": report.error_rate,
                "system_health": {
                    "cpu_usage": report.system_health.cpu_usage,
                    "memory_usage": report.system_health.memory_usage,
                    "gpu_usage": report.system_health.gpu_usage,
                    "healthy": report.system_health.healthy,
                    "warnings": report.system_health.warnings,
                    "recommendations": report.system_health.recommendations
                },
                "bottlenecks": report.bottlenecks,
                "recommendations": report.recommendations
            }
        
        return {"error": "performance_monitor_not_available"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics"""
        
        stats = {
            "module_version": __version__,
            "initialized": self.initialized,
            "config": self.config
        }
        
        if self.initialized:
            if self.engine:
                stats["engine"] = self.engine.get_stats()
            if self.model_manager:
                stats["model_manager"] = self.model_manager.get_stats()
            if self.relevance_scorer:
                stats["relevance_scorer"] = self.relevance_scorer.get_stats()
            if self.learning_to_rank:
                stats["learning_to_rank"] = self.learning_to_rank.get_stats()
            if self.explanation_generator:
                stats["explanation_generator"] = self.explanation_generator.get_stats()
            if self.performance_monitor:
                stats["performance_monitor"] = self.performance_monitor.get_stats()
        
        return stats
    
    async def cleanup(self):
        """Clean up all resources"""
        
        self.logger.info("Cleaning up cross-encoder reranking module...")
        
        cleanup_order = [
            ("engine", self.engine),
            ("explanation_generator", self.explanation_generator),
            ("learning_to_rank", self.learning_to_rank),
            ("relevance_scorer", self.relevance_scorer),
            ("model_manager", self.model_manager),
            ("performance_monitor", self.performance_monitor)
        ]
        
        for name, component in cleanup_order:
            if component and hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                    self.logger.info(f"✓ {name} cleaned up")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name}: {e}")
        
        self.initialized = False
        self.logger.info("Cross-encoder reranking module cleanup completed")

# Factory function for easy module creation
def create_cross_encoder_reranking_module(config: Optional[Dict[str, Any]] = None) -> CrossEncoderRerankingModule:
    """
    Create and return a configured cross-encoder reranking module
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        CrossEncoderRerankingModule instance
    """
    return CrossEncoderRerankingModule(config)

# Convenience functions
async def quick_rerank(query: str, candidates: list, 
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick reranking function for simple use cases
    
    Args:
        query: Search query
        candidates: List of candidate documents
        config: Optional configuration
        
    Returns:
        Reranked results
    """
    module = create_cross_encoder_reranking_module(config)
    
    try:
        await module.initialize()
        result = await module.rerank(query, candidates)
        return result
    finally:
        await module.cleanup()

# Export main components and functions
__all__ = [
    # Main module class
    "CrossEncoderRerankingModule",
    "create_cross_encoder_reranking_module",
    
    # Individual components
    "CrossEncoderEngine",
    "ModelManager", 
    "RelevanceScorer",
    "LearningToRank",
    "ExplanationGenerator",
    "PerformanceMonitor",
    
    # Factory functions
    "create_cross_encoder_engine",
    "create_model_manager",
    "create_relevance_scorer", 
    "create_learning_to_rank",
    "create_explanation_generator",
    "create_performance_monitor",
    
    # Utilities
    "OperationTracker",
    "quick_rerank",
    
    # Configuration
    "DEFAULT_CONFIG",
    
    # Version info
    "__version__",
    "__author__"
]

# Module initialization message
logger.info(f"Cross-encoder reranking module loaded (version {__version__})")
logger.info("Ready to enhance hybrid retrieval results with transformer-based reranking")