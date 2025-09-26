"""
Risk Training Package

Contains training utilities for the ML risk classifier:
- Training data preparation from Modules 2-3
- Data augmentation and balancing
- Training configuration and utilities  
"""

from .data_preparation import (
    RiskTrainingDataPreparer,
    TrainingExample
)

__all__ = [
    "RiskTrainingDataPreparer",
    "TrainingExample"
]