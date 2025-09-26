"""
Risk Assessment Orchestrator

Main orchestrator for Module 4 risk assessment combining:
- ML-based risk classifier (RoBERTa/Legal-BERT)
- Rule-based risk analyzers  
- Feature extraction
- Integration with Modules 1-3

This implements the lightweight clause-level risk scoring as specified.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from .models.ml_risk_classifier import (
    RiskClassifier, FeatureVector, RiskPrediction
)
from .analyzers.financial_analyzer import FinancialRiskAnalyzer, FinancialRiskMetrics
from .analyzers.legal_analyzer import LegalRiskAnalyzer, LegalRiskMetrics
from .utils.feature_extractor import FeatureExtractor, ExtractedFeatures
from .models.risk_aggregator import RiskAggregator

logger = logging.getLogger(__name__)

@dataclass
class ClauseRiskAssessment:
    """Complete risk assessment for a single clause"""
    clause_id: str
    clause_type: str
    clause_text: str
    
    # ML-based assessment (primary)
    ml_risk_level: str  # LOW, MEDIUM, HIGH
    ml_confidence: float
    ml_probabilities: Dict[str, float]
    ml_rationale: str
    evidence_tokens: List[Dict[str, Any]]
    
    # Rule-based assessments (supporting)
    financial_risk: FinancialRiskMetrics
    legal_risk: LegalRiskMetrics
    
    # Metadata and features
    extracted_features: ExtractedFeatures
    processing_time: float
    
    # Final assessment
    final_risk_level: str
    final_confidence: float
    review_required: bool

@dataclass
class DocumentRiskAssessment:
    """Risk assessment for entire document"""
    document_id: str
    clause_assessments: List[ClauseRiskAssessment]
    overall_risk_level: str
    risk_distribution: Dict[str, int]  # count by risk level
    high_risk_clauses: List[str]  # clause IDs
    processing_time: float
    recommendations: List[str]

class RiskAssessmentEngine:
    """Main risk assessment engine for Module 4"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        # Initialize ML classifier
        self.ml_classifier = RiskClassifier(device=device)
        if model_path and Path(model_path).exists():
            self.ml_classifier.load_model(model_path)
            self.ml_trained = True
        else:
            self.ml_trained = False
            logger.warning("ML classifier not loaded - will use rule-based assessment only")
        
        # Initialize rule-based analyzers
        self.financial_analyzer = FinancialRiskAnalyzer()
        self.legal_analyzer = LegalRiskAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.risk_aggregator = RiskAggregator()
        
        # Risk level mappings
        self.risk_levels = ["LOW", "MEDIUM", "HIGH"]
    
    def assess_clause_risk(self, 
                          clause_id: str,
                          clause_type: str, 
                          clause_text: str,
                          document_context: str = "",
                          contract_metadata: Optional[Dict[str, Any]] = None) -> ClauseRiskAssessment:
        """
        Assess risk for a single clause using ML + rule-based approaches.
        This is the core function implementing the Module 4 specification.
        """
        start_time = time.time()
        
        # Extract features from clause and context
        extracted_features = self.feature_extractor.extract_features(
            clause_text=clause_text,
            document_context=document_context,
            existing_metadata=contract_metadata or {}
        )
        
        # Create feature vector for ML classifier
        feature_vector = self._create_feature_vector(
            clause_id, clause_type, clause_text, extracted_features, contract_metadata
        )
        
        # ML-based risk assessment (primary method)
        ml_prediction = None
        if self.ml_trained:
            try:
                ml_prediction = self.ml_classifier.predict(feature_vector, explain=True)
            except Exception as e:
                logger.error(f"ML prediction failed for clause {clause_id}: {e}")
                ml_prediction = self._create_fallback_prediction(clause_id)
        else:
            ml_prediction = self._create_fallback_prediction(clause_id)
        
        # Rule-based risk assessments (supporting analysis)
        financial_risk = self.financial_analyzer.analyze_financial_risk(
            clause_type=clause_type,
            clause_text=clause_text,
            contract_value=self._get_contract_value(extracted_features, contract_metadata),
            contract_duration=self._get_contract_duration(contract_metadata)
        )
        
        legal_risk = self.legal_analyzer.analyze_legal_risk(
            clause_type=clause_type,
            clause_text=clause_text,
            governing_law=extracted_features.governing_law,
            industry=contract_metadata.get("industry") if contract_metadata else None
        )
        
        # Combine assessments and make final decision
        final_risk_level, final_confidence, review_required = self._combine_assessments(
            ml_prediction, financial_risk, legal_risk, clause_type
        )
        
        processing_time = time.time() - start_time
        
        return ClauseRiskAssessment(
            clause_id=clause_id,
            clause_type=clause_type,
            clause_text=clause_text,
            ml_risk_level=ml_prediction.risk_level,
            ml_confidence=ml_prediction.confidence_score,
            ml_probabilities=ml_prediction.probabilities,
            ml_rationale=ml_prediction.rationale,
            evidence_tokens=ml_prediction.evidence_tokens,
            financial_risk=financial_risk,
            legal_risk=legal_risk,
            extracted_features=extracted_features,
            processing_time=processing_time,
            final_risk_level=final_risk_level,
            final_confidence=final_confidence,
            review_required=review_required
        )
    
    def assess_document_risk(self, 
                           document_id: str,
                           clauses: List[Dict[str, str]],  # [{"id": "", "type": "", "text": ""}]
                           document_context: str = "",
                           contract_metadata: Optional[Dict[str, Any]] = None) -> DocumentRiskAssessment:
        """Assess risk for entire document by analyzing all clauses"""
        start_time = time.time()
        
        clause_assessments = []
        for clause in clauses:
            assessment = self.assess_clause_risk(
                clause_id=clause["id"],
                clause_type=clause["type"],
                clause_text=clause["text"],
                document_context=document_context,
                contract_metadata=contract_metadata
            )
            clause_assessments.append(assessment)
        
        # Aggregate document-level risk
        risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        high_risk_clauses = []
        
        for assessment in clause_assessments:
            risk_distribution[assessment.final_risk_level] += 1
            if assessment.final_risk_level == "HIGH":
                high_risk_clauses.append(assessment.clause_id)
        
        # Determine overall document risk level
        overall_risk_level = self._determine_overall_risk(risk_distribution, len(clauses))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(clause_assessments, overall_risk_level)
        
        processing_time = time.time() - start_time
        
        return DocumentRiskAssessment(
            document_id=document_id,
            clause_assessments=clause_assessments,
            overall_risk_level=overall_risk_level,
            risk_distribution=risk_distribution,
            high_risk_clauses=high_risk_clauses,
            processing_time=processing_time,
            recommendations=recommendations
        )
    
    def _create_feature_vector(self, clause_id: str, clause_type: str, clause_text: str,
                              extracted_features: ExtractedFeatures,
                              contract_metadata: Optional[Dict[str, Any]]) -> FeatureVector:
        """Create feature vector for ML classifier"""
        
        # Get largest monetary amount
        monetary_amount = None
        if extracted_features.monetary_amounts:
            monetary_amount = extracted_features.monetary_amounts[0].amount
        
        # Get contract metadata
        contract_value = None
        contract_duration = None
        industry = None
        
        if contract_metadata:
            contract_value = contract_metadata.get("contract_value")
            contract_duration = contract_metadata.get("duration_months")
            industry = contract_metadata.get("industry")
        
        return FeatureVector(
            clause_text=clause_text,
            clause_type=clause_type,
            monetary_amount=monetary_amount,
            governing_law=extracted_features.governing_law,
            party_role=extracted_features.party_role,
            contract_value=contract_value,
            contract_duration=contract_duration,
            industry=industry
        )
    
    def _create_fallback_prediction(self, clause_id: str) -> RiskPrediction:
        """Create fallback prediction when ML classifier unavailable"""
        return RiskPrediction(
            clause_id=clause_id,
            risk_level="MEDIUM",  # Conservative fallback
            risk_probability=0.5,
            confidence_score=0.3,  # Low confidence to indicate fallback
            probabilities={"LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.2},
            rationale="ML classifier unavailable - using rule-based assessment",
            evidence_tokens=[],
            metadata={}
        )
    
    def _combine_assessments(self, ml_prediction: RiskPrediction,
                           financial_risk: FinancialRiskMetrics,
                           legal_risk: LegalRiskMetrics,
                           clause_type: str) -> tuple[str, float, bool]:
        """Combine ML and rule-based assessments into final decision"""
        
        # Weight the different assessment methods
        ml_weight = 0.6 if self.ml_trained else 0.0
        financial_weight = 0.25
        legal_weight = 0.25
        rule_weight = 0.4 if not self.ml_trained else 0.4  # Increase rule weight if no ML
        
        # Convert risk scores to normalized values (0-1)
        ml_score = self._risk_level_to_score(ml_prediction.risk_level)
        financial_score = min(financial_risk.overall_financial_risk / 10.0, 1.0)
        legal_score = min(legal_risk.overall_legal_risk / 10.0, 1.0)
        rule_score = (financial_score + legal_score) / 2.0
        
        # Calculate weighted combined score
        if self.ml_trained:
            combined_score = (ml_score * ml_weight + 
                            financial_score * financial_weight + 
                            legal_score * legal_weight)
        else:
            combined_score = rule_score
        
        # Convert back to risk level
        final_risk_level = self._score_to_risk_level(combined_score)
        
        # Calculate combined confidence
        ml_confidence = ml_prediction.confidence_score if self.ml_trained else 0.0
        rule_confidence = (financial_risk.confidence + legal_risk.confidence) / 2.0
        
        if self.ml_trained:
            final_confidence = (ml_confidence * ml_weight + rule_confidence * rule_weight)
        else:
            final_confidence = rule_confidence
        
        # Determine if review is required
        review_required = self._should_require_review(
            ml_prediction, financial_risk, legal_risk, final_confidence, clause_type
        )
        
        return final_risk_level, final_confidence, review_required
    
    def _risk_level_to_score(self, risk_level: str) -> float:
        """Convert risk level to numerical score"""
        mapping = {"LOW": 0.2, "MEDIUM": 0.6, "HIGH": 0.9}
        return mapping.get(risk_level, 0.5)
    
    def _score_to_risk_level(self, score: float) -> str:
        """Convert numerical score to risk level"""
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _should_require_review(self, ml_prediction: RiskPrediction,
                             financial_risk: FinancialRiskMetrics,
                             legal_risk: LegalRiskMetrics,
                             final_confidence: float,
                             clause_type: str) -> bool:
        """Determine if human review is required"""
        
        # Low confidence requires review
        if final_confidence < 0.6:
            return True
        
        # High-risk clauses always require review
        if ml_prediction.risk_level == "HIGH":
            return True
        
        # Disagreement between methods requires review
        if self.ml_trained:
            ml_score = self._risk_level_to_score(ml_prediction.risk_level)
            rule_score = (min(financial_risk.overall_financial_risk / 10.0, 1.0) + 
                         min(legal_risk.overall_legal_risk / 10.0, 1.0)) / 2.0
            
            if abs(ml_score - rule_score) > 0.3:  # Significant disagreement
                return True
        
        # Critical clause types require review
        critical_types = ["limitation_of_liability", "indemnification", "intellectual_property"]
        if clause_type in critical_types:
            return True
        
        return False
    
    def _get_contract_value(self, extracted_features: ExtractedFeatures,
                          contract_metadata: Optional[Dict[str, Any]]) -> Optional[float]:
        """Get contract value from features or metadata"""
        if contract_metadata and "contract_value" in contract_metadata:
            return contract_metadata["contract_value"]
        
        # Use largest extracted monetary amount as proxy
        if extracted_features.monetary_amounts:
            return extracted_features.monetary_amounts[0].amount
        
        return None
    
    def _get_contract_duration(self, contract_metadata: Optional[Dict[str, Any]]) -> Optional[int]:
        """Get contract duration in months from metadata"""
        if contract_metadata:
            return contract_metadata.get("duration_months")
        return None
    
    def _determine_overall_risk(self, risk_distribution: Dict[str, int], total_clauses: int) -> str:
        """Determine overall document risk level"""
        if risk_distribution["HIGH"] > 0:
            # Any high-risk clause makes document high-risk
            return "HIGH"
        elif risk_distribution["MEDIUM"] / total_clauses > 0.3:
            # >30% medium-risk clauses makes document medium-risk
            return "MEDIUM"
        elif risk_distribution["MEDIUM"] > 0:
            # Any medium-risk clause makes document at least medium-risk
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, clause_assessments: List[ClauseRiskAssessment],
                                overall_risk_level: str) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        # Count high-risk clauses
        high_risk_count = sum(1 for assessment in clause_assessments 
                             if assessment.final_risk_level == "HIGH")
        
        if high_risk_count > 0:
            recommendations.append(f"Immediate attention required: {high_risk_count} high-risk clause(s) identified")
        
        # Review requirements
        review_required_count = sum(1 for assessment in clause_assessments 
                                  if assessment.review_required)
        
        if review_required_count > 0:
            recommendations.append(f"Legal review recommended for {review_required_count} clause(s)")
        
        # Specific risk type recommendations
        financial_issues = sum(1 for assessment in clause_assessments 
                              if assessment.financial_risk.overall_financial_risk > 6.0)
        
        if financial_issues > 0:
            recommendations.append(f"Financial risk review needed for {financial_issues} clause(s)")
        
        # Overall recommendations
        if overall_risk_level == "HIGH":
            recommendations.append("Contract poses significant risk - comprehensive legal review recommended")
        elif overall_risk_level == "MEDIUM":
            recommendations.append("Contract has moderate risk - focused review of flagged clauses recommended")
        else:
            recommendations.append("Contract appears to have acceptable risk profile")
        
        return recommendations
    
    def train_ml_classifier(self, training_data: List[Dict[str, Any]], 
                          validation_data: List[Dict[str, Any]],
                          output_dir: str, **training_args):
        """Train the ML risk classifier"""
        
        # Convert training data to features and labels
        train_features = []
        train_labels = []
        
        for item in training_data:
            feature_vector = FeatureVector(
                clause_text=item["clause_text"],
                clause_type=item["clause_type"],
                monetary_amount=item.get("monetary_amount"),
                governing_law=item.get("governing_law"),
                party_role=item.get("party_role"),
                contract_value=item.get("contract_value"),
                contract_duration=item.get("contract_duration"),
                industry=item.get("industry")
            )
            train_features.append(feature_vector)
            train_labels.append(item["risk_label"])
        
        # Convert validation data similarly
        val_features = []
        val_labels = []
        
        for item in validation_data:
            feature_vector = FeatureVector(
                clause_text=item["clause_text"],
                clause_type=item["clause_type"],
                monetary_amount=item.get("monetary_amount"),
                governing_law=item.get("governing_law"),
                party_role=item.get("party_role"),
                contract_value=item.get("contract_value"),
                contract_duration=item.get("contract_duration"),
                industry=item.get("industry")
            )
            val_features.append(feature_vector)
            val_labels.append(item["risk_label"])
        
        # Train the classifier
        self.ml_classifier.train(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            output_dir=output_dir,
            **training_args
        )
        
        self.ml_trained = True
        logger.info("ML risk classifier training completed")
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the trained model on test data"""
        if not self.ml_trained:
            raise ValueError("Model not trained yet")
        
        # Convert test data to features and labels
        test_features = []
        test_labels = []
        
        for item in test_data:
            feature_vector = FeatureVector(
                clause_text=item["clause_text"],
                clause_type=item["clause_type"],
                monetary_amount=item.get("monetary_amount"),
                governing_law=item.get("governing_law"),
                party_role=item.get("party_role"),
                contract_value=item.get("contract_value"),
                contract_duration=item.get("contract_duration"),
                industry=item.get("industry")
            )
            test_features.append(feature_vector)
            test_labels.append(item["risk_label"])
        
        # Evaluate calibration
        calibration_metrics = self.ml_classifier.evaluate_calibration(test_features, test_labels)
        
        return calibration_metrics
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.ml_trained:
            self.ml_classifier.save_model(path)
        else:
            logger.warning("No trained model to save")
    
    def load_model(self, path: str):
        """Load trained model"""
        self.ml_classifier.load_model(path)
        self.ml_trained = True
        logger.info(f"ML risk classifier loaded from {path}")