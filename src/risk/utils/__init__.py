"""
Risk Utilities Package

Contains utility functions for risk assessment:
- Feature extraction from contract text and metadata
- Text processing and normalization utilities
- Risk calculation helpers
"""

from .feature_extractor import (
    FeatureExtractor,
    ExtractedFeatures,
    MonetaryAmount,
    LegalEntity
)

__all__ = [
    "FeatureExtractor",
    "ExtractedFeatures", 
    "MonetaryAmount",
    "LegalEntity"
]