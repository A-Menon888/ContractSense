"""
ContractSense ML Module - BERT-CRF Clause Extraction System

This module provides advanced machine learning capabilities for automated
clause extraction from legal contracts using BERT-CRF architecture.
"""

from .models.bert_crf import BertCrfModel, ModelConfig
from .training.trainer import ModelTrainer
from .inference.predictor import ClausePredictor
from .utils.tokenization import ContractTokenizer
from .utils.label_encoding import LabelEncoder

__version__ = "1.0.0"
__all__ = [
    "BertCrfModel",
    "ModelConfig", 
    "ModelTrainer",
    "ClausePredictor",
    "ContractTokenizer",
    "LabelEncoder"
]