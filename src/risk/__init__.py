"""
Risk Assessment Module for ContractSense

Module 4: Clause-Level Risk Scoring

This module provides lightweight ML-based risk classification for contract clauses
with explainability, calibration, and comprehensive risk analysis capabilities.

Main Components:
- RiskAssessmentEngine: Main orchestration engine
- RiskClassifier: ML-based risk classifier with RoBERTa/Legal-BERT
- FinancialRiskAnalyzer: Rule-based financial risk analysis
- LegalRiskAnalyzer: Rule-based legal risk analysis
- FeatureExtractor: Contract feature extraction utilities
- RiskTrainingDataPreparer: Training data preparation from Modules 2-3

Usage:
    from risk import RiskAssessmentEngine
    
    engine = RiskAssessmentEngine(model_path="path/to/model.pt")
    assessment = engine.assess_clause_risk(
        clause_id="clause_1",
        clause_type="limitation_of_liability",
        clause_text="Company shall indemnify...",
        contract_metadata={"contract_value": 500000}
    )
"""

from .risk_engine import (
    RiskAssessmentEngine,
    ClauseRiskAssessment,
    DocumentRiskAssessment
)

from .models.ml_risk_classifier import (
    RiskClassifier,
    FeatureVector,
    RiskPrediction
)

from .analyzers.financial_analyzer import (
    FinancialRiskAnalyzer,
    FinancialRiskMetrics
)

from .analyzers.legal_analyzer import (
    LegalRiskAnalyzer,
    LegalRiskMetrics
)

from .utils.feature_extractor import (
    FeatureExtractor,
    ExtractedFeatures,
    MonetaryAmount
)

from .training.data_preparation import (
    RiskTrainingDataPreparer,
    TrainingExample
)

__version__ = "1.0.0"
__author__ = "ContractSense Team"

__all__ = [
    "RiskAssessmentEngine",
    "ClauseRiskAssessment", 
    "DocumentRiskAssessment",
    "RiskClassifier",
    "FeatureVector",
    "RiskPrediction",
    "FinancialRiskAnalyzer",
    "FinancialRiskMetrics",
    "LegalRiskAnalyzer", 
    "LegalRiskMetrics",
    "FeatureExtractor",
    "ExtractedFeatures",
    "MonetaryAmount",
    "RiskTrainingDataPreparer",
    "TrainingExample"
]