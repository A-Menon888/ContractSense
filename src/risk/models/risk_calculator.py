"""
Risk Calculator Module

Core risk calculation engine that analyzes contract clauses and computes
multi-dimensional risk scores with confidence assessment.
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .risk_config import (
    RiskCalculationConfig, RiskLevel, RiskThresholds, RiskWeights
)

logger = logging.getLogger(__name__)

@dataclass
class RiskFactor:
    """Individual risk factor identified in clause analysis"""
    factor_type: str
    impact_score: float
    confidence: float
    description: str
    severity: str = "MEDIUM"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_type": self.factor_type,
            "impact_score": self.impact_score,
            "confidence": self.confidence,
            "description": self.description,
            "severity": self.severity
        }

@dataclass
class ClauseRiskAssessment:
    """Complete risk assessment for a single clause"""
    clause_id: str
    clause_type: str
    clause_text: str
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    risk_factors: List[RiskFactor]
    dimensional_scores: Dict[str, float]  # financial, legal, operational, strategic
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "clause_type": self.clause_type,
            "clause_text": self.clause_text,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "risk_factors": [factor.to_dict() for factor in self.risk_factors],
            "dimensional_scores": self.dimensional_scores
        }

@dataclass
class DocumentRiskAssessment:
    """Complete risk assessment for an entire contract document"""
    document_id: str
    overall_risk_score: float
    overall_risk_level: RiskLevel
    confidence: float
    clause_assessments: List[ClauseRiskAssessment]
    dimensional_scores: Dict[str, float]
    risk_distribution: Dict[str, int]  # count by risk level
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "overall_risk_score": self.overall_risk_score,
            "overall_risk_level": self.overall_risk_level.value,
            "confidence": self.confidence,
            "clause_assessments": [assessment.to_dict() for assessment in self.clause_assessments],
            "dimensional_scores": self.dimensional_scores,
            "risk_distribution": self.risk_distribution,
            "processing_time": self.processing_time
        }

class LanguageAnalyzer:
    """Analyzes clause language for risk-indicating patterns"""
    
    # Risk-indicating language patterns
    HIGH_RISK_PATTERNS = {
        "absolute_terms": [
            r"\\b(shall|must|will|absolute|unconditional)\\b",
            r"\\b(mandatory|required|obligated|bound)\\b"
        ],
        "unlimited_scope": [
            r"\\b(unlimited|infinite|any and all|whatsoever)\\b",
            r"\\b(without limitation|including without limitation)\\b"
        ],
        "broad_language": [
            r"\\b(including but not limited to|such as|among others)\\b",
            r"\\b(and/or|whether or not|regardless)\\b"
        ],
        "exclusive_terms": [
            r"\\b(sole|only|exclusive|solely|exclusively)\\b",
            r"\\b(unique|singular|alone)\\b"
        ],
        "immediate_effect": [
            r"\\b(immediately|forthwith|without delay|instant)\\b",
            r"\\b(upon|effective immediately|at once)\\b"
        ],
        "perpetual_terms": [
            r"\\b(perpetual|permanent|forever|indefinite)\\b",
            r"\\b(in perpetuity|for all time)\\b"
        ],
        "irrevocable_terms": [
            r"\\b(irrevocable|cannot be cancelled|non-revocable)\\b",
            r"\\b(binding|enforceable|non-terminable)\\b"
        ]
    }
    
    LOW_RISK_PATTERNS = {
        "conditional_terms": [
            r"\\b(if|unless|provided that|subject to)\\b",
            r"\\b(conditional|contingent upon|depending on)\\b"
        ],
        "reasonable_terms": [
            r"\\b(reasonable|commercially reasonable|fair)\\b",
            r"\\b(appropriate|suitable|proper)\\b"
        ],
        "best_efforts": [
            r"\\b(best efforts|reasonable efforts|good faith)\\b",
            r"\\b(commercially reasonable efforts|diligent)\\b"
        ],
        "standard_terms": [
            r"\\b(standard|customary|usual|typical)\\b",
            r"\\b(ordinary|normal|conventional)\\b"
        ]
    }
    
    def analyze_language_severity(self, clause_text: str) -> Tuple[List[str], float]:
        """Analyze clause text for risk-indicating language patterns"""
        text_lower = clause_text.lower()
        detected_patterns = []
        severity_score = 1.0
        
        # Check for high-risk patterns
        for pattern_type, patterns in self.HIGH_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_patterns.append(pattern_type)
                    # Compound severity for multiple high-risk patterns
                    severity_score *= 1.2
                    break  # Only count each pattern type once
        
        # Check for low-risk patterns (reduce severity)
        for pattern_type, patterns in self.LOW_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_patterns.append(pattern_type)
                    severity_score *= 0.85
                    break
        
        # Cap severity multiplier
        severity_score = min(severity_score, 2.0)
        severity_score = max(severity_score, 0.3)
        
        return detected_patterns, severity_score
    
    def analyze_scope_impact(self, clause_text: str, contract_context: Dict = None) -> Tuple[str, float]:
        """Analyze the scope and potential impact of a clause"""
        text_lower = clause_text.lower()
        
        # Geographic scope indicators
        if re.search(r"\\b(worldwide|global|international|all countries)\\b", text_lower):
            return "global_scope", 1.3
        elif re.search(r"\\b(multiple jurisdictions|cross-border|several states)\\b", text_lower):
            return "multi_jurisdiction", 1.2
        elif re.search(r"\\b(broad|extensive|wide-ranging|comprehensive)\\b", text_lower):
            return "broad_application", 1.1
        elif re.search(r"\\b(limited|specific|particular|certain)\\b", text_lower):
            return "specific_scope", 0.9
        elif re.search(r"\\b(narrow|restricted|confined|minimal)\\b", text_lower):
            return "narrow_scope", 0.8
        
        return "standard_scope", 1.0

class RiskCalculator:
    """Core risk calculation engine"""
    
    def __init__(self, config: RiskCalculationConfig = None):
        self.config = config or RiskCalculationConfig()
        self.language_analyzer = LanguageAnalyzer()
    
    def calculate_clause_risk(self, 
                            clause_id: str,
                            clause_type: str,
                            clause_text: str,
                            contract_context: Dict = None,
                            ml_confidence: float = None) -> ClauseRiskAssessment:
        """Calculate comprehensive risk assessment for a single clause"""
        
        # Get base risk score for clause type
        base_risk = self.config.get_base_risk_score(clause_type)
        
        # Analyze language for risk factors
        language_patterns, severity_multiplier = self.language_analyzer.analyze_language_severity(clause_text)
        scope_type, scope_multiplier = self.language_analyzer.analyze_scope_impact(clause_text, contract_context)
        
        # Apply industry-specific adjustments
        industry_multiplier = self.config.get_industry_multiplier(clause_type)
        
        # Calculate dimensional risk scores
        dimensional_scores = self._calculate_dimensional_scores(
            clause_type, clause_text, base_risk, severity_multiplier, scope_multiplier
        )
        
        # Calculate overall risk score as weighted average of dimensional scores
        overall_risk = (
            dimensional_scores["financial"] * self.config.weights.financial +
            dimensional_scores["legal"] * self.config.weights.legal +
            dimensional_scores["operational"] * self.config.weights.operational +
            dimensional_scores["strategic"] * self.config.weights.strategic
        )
        
        # Apply final adjustments
        overall_risk *= industry_multiplier
        overall_risk = min(overall_risk, 10.0)  # Cap at 10.0
        overall_risk = max(overall_risk, 0.0)   # Floor at 0.0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            clause_text, language_patterns, ml_confidence
        )
        
        # Determine risk level
        risk_level = self.config.thresholds.get_risk_level(overall_risk)
        
        # Generate risk factors
        risk_factors = self._generate_risk_factors(
            clause_type, language_patterns, scope_type, severity_multiplier
        )
        
        return ClauseRiskAssessment(
            clause_id=clause_id,
            clause_type=clause_type,
            clause_text=clause_text,
            risk_score=round(overall_risk, 2),
            risk_level=risk_level,
            confidence=round(confidence, 3),
            risk_factors=risk_factors,
            dimensional_scores={k: round(v, 2) for k, v in dimensional_scores.items()}
        )
    
    def _calculate_dimensional_scores(self, clause_type: str, clause_text: str, 
                                    base_risk: float, severity_multiplier: float,
                                    scope_multiplier: float) -> Dict[str, float]:
        """Calculate risk scores for each dimension (financial, legal, operational, strategic)"""
        
        # Base dimensional weights by clause type
        dimensional_weights = {
            "financial": self._get_financial_weight(clause_type),
            "legal": self._get_legal_weight(clause_type),
            "operational": self._get_operational_weight(clause_type),
            "strategic": self._get_strategic_weight(clause_type)
        }
        
        # Calculate scores for each dimension
        dimensional_scores = {}
        for dimension, weight in dimensional_weights.items():
            # Apply dimension-specific modifiers
            dimension_risk = base_risk * weight * severity_multiplier * scope_multiplier
            
            # Add dimension-specific text analysis
            dimension_modifier = self._analyze_dimension_specific_risks(
                dimension, clause_text
            )
            
            dimension_risk *= dimension_modifier
            dimensional_scores[dimension] = min(dimension_risk, 10.0)
        
        return dimensional_scores
    
    def _get_financial_weight(self, clause_type: str) -> float:
        """Get financial risk weight for clause type"""
        financial_weights = {
            "payment": 1.5,
            "limitation_of_liability": 1.4,
            "indemnification": 1.3,
            "termination": 1.2,
            "warranties": 1.1,
            "force_majeure": 0.9,
            "confidentiality": 0.8,
            "intellectual_property": 0.9,
            "governing_law": 0.7,
            "dispute_resolution": 0.8
        }
        return financial_weights.get(clause_type, 1.0)
    
    def _get_legal_weight(self, clause_type: str) -> float:
        """Get legal risk weight for clause type"""
        legal_weights = {
            "indemnification": 1.5,
            "limitation_of_liability": 1.4,
            "governing_law": 1.3,
            "dispute_resolution": 1.3,
            "intellectual_property": 1.2,
            "confidentiality": 1.2,
            "warranties": 1.1,
            "termination": 1.0,
            "payment": 0.9,
            "force_majeure": 1.0
        }
        return legal_weights.get(clause_type, 1.0)
    
    def _get_operational_weight(self, clause_type: str) -> float:
        """Get operational risk weight for clause type"""
        operational_weights = {
            "termination": 1.5,
            "force_majeure": 1.4,
            "performance_standards": 1.3,
            "service_levels": 1.3,
            "confidentiality": 1.1,
            "payment": 1.1,
            "intellectual_property": 1.0,
            "governing_law": 0.8,
            "dispute_resolution": 0.9,
            "warranties": 1.0
        }
        return operational_weights.get(clause_type, 1.0)
    
    def _get_strategic_weight(self, clause_type: str) -> float:
        """Get strategic risk weight for clause type"""
        strategic_weights = {
            "intellectual_property": 1.5,
            "exclusivity": 1.4,
            "non_compete": 1.3,
            "termination": 1.2,
            "confidentiality": 1.1,
            "assignment": 1.1,
            "payment": 0.9,
            "governing_law": 0.8,
            "dispute_resolution": 0.8,
            "warranties": 0.9
        }
        return strategic_weights.get(clause_type, 1.0)
    
    def _analyze_dimension_specific_risks(self, dimension: str, clause_text: str) -> float:
        """Analyze dimension-specific risk indicators in clause text"""
        text_lower = clause_text.lower()
        
        if dimension == "financial":
            # Look for financial risk indicators
            if re.search(r"\\b(penalty|fine|damages|cost|expense)\\b", text_lower):
                return 1.2
            elif re.search(r"\\b(unlimited|maximum|aggregate)\\b", text_lower):
                return 1.3
        
        elif dimension == "legal":
            # Look for legal risk indicators
            if re.search(r"\\b(liable|responsible|indemnify|defend)\\b", text_lower):
                return 1.2
            elif re.search(r"\\b(breach|violation|non-compliance)\\b", text_lower):
                return 1.3
        
        elif dimension == "operational":
            # Look for operational risk indicators
            if re.search(r"\\b(terminate|suspend|cease|stop)\\b", text_lower):
                return 1.2
            elif re.search(r"\\b(performance|service level|deadline)\\b", text_lower):
                return 1.1
        
        elif dimension == "strategic":
            # Look for strategic risk indicators
            if re.search(r"\\b(exclusive|proprietary|confidential|competitive)\\b", text_lower):
                return 1.2
            elif re.search(r"\\b(intellectual property|trade secret|patent)\\b", text_lower):
                return 1.3
        
        return 1.0
    
    def _calculate_confidence(self, clause_text: str, language_patterns: List[str], 
                            ml_confidence: float = None) -> float:
        """Calculate confidence score for risk assessment"""
        base_confidence = 0.7
        
        # Boost confidence based on detected language patterns
        pattern_boost = min(len(language_patterns) * 0.05, 0.2)
        
        # Boost confidence based on clause length (more text = more analysis)
        length_boost = min(len(clause_text) / 1000, 0.1)
        
        # Use ML confidence if available
        ml_boost = 0.0
        if ml_confidence:
            ml_boost = (ml_confidence - 0.5) * 0.2  # Scale ML confidence to boost
        
        confidence = base_confidence + pattern_boost + length_boost + ml_boost
        return min(confidence, 1.0)
    
    def _generate_risk_factors(self, clause_type: str, language_patterns: List[str],
                             scope_type: str, severity_multiplier: float) -> List[RiskFactor]:
        """Generate specific risk factors identified during analysis"""
        factors = []
        
        # Add language-based risk factors
        for pattern in language_patterns:
            if pattern in ["absolute_terms", "unlimited_scope", "irrevocable_terms"]:
                factors.append(RiskFactor(
                    factor_type=pattern,
                    impact_score=severity_multiplier * 2,
                    confidence=0.9,
                    description=f"High-risk language pattern detected: {pattern.replace('_', ' ')}",
                    severity="HIGH"
                ))
            elif pattern in ["conditional_terms", "reasonable_terms", "best_efforts"]:
                factors.append(RiskFactor(
                    factor_type=pattern,
                    impact_score=abs(severity_multiplier - 1.0),
                    confidence=0.8,
                    description=f"Risk-mitigating language detected: {pattern.replace('_', ' ')}",
                    severity="LOW"
                ))
        
        # Add scope-based risk factors
        if scope_type in ["global_scope", "multi_jurisdiction"]:
            factors.append(RiskFactor(
                factor_type="broad_scope",
                impact_score=2.5,
                confidence=0.85,
                description=f"Broad geographical or jurisdictional scope: {scope_type.replace('_', ' ')}",
                severity="MEDIUM"
            ))
        
        # Add clause-type specific risk factors
        clause_specific_factors = self._get_clause_specific_factors(clause_type)
        factors.extend(clause_specific_factors)
        
        return factors
    
    def _get_clause_specific_factors(self, clause_type: str) -> List[RiskFactor]:
        """Get clause-type specific risk factors"""
        factors = []
        
        if clause_type == "limitation_of_liability":
            factors.append(RiskFactor(
                factor_type="liability_limitation",
                impact_score=3.0,
                confidence=0.95,
                description="Limitation of liability clause may cap financial exposure",
                severity="HIGH"
            ))
        elif clause_type == "indemnification":
            factors.append(RiskFactor(
                factor_type="indemnification_obligation",
                impact_score=2.8,
                confidence=0.9,
                description="Indemnification clause creates potential financial liability",
                severity="HIGH"
            ))
        elif clause_type == "termination":
            factors.append(RiskFactor(
                factor_type="termination_risk",
                impact_score=2.5,
                confidence=0.85,
                description="Termination clause may allow contract cancellation",
                severity="MEDIUM"
            ))
        
        return factors