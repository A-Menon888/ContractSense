"""
Inference package initialization.
"""

from .predictor import ClausePredictor, ClausePrediction, DocumentPrediction
from .post_processing import PostProcessor, PostProcessingConfig

__all__ = [
    "ClausePredictor",
    "ClausePrediction",
    "DocumentPrediction",
    "PostProcessor",
    "PostProcessingConfig"
]