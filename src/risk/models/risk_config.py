"""
Risk Configuration Module

This module defines the core configuration for risk assessment including:
- Risk category definitions and thresholds
- Clause type base risk scores
- Risk calculation parameters
- Industry-specific adjustments
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

class RiskLevel(Enum):
    """Risk level enumeration with numerical scores"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class RiskThresholds:
    """Risk score thresholds for categorization"""
    critical: float = 8.0
    high: float = 6.0
    medium: float = 4.0
    low: float = 0.0
    
    def get_risk_level(self, score: float) -> RiskLevel:
        """Convert numerical risk score to risk level category"""
        if score >= self.critical:
            return RiskLevel.CRITICAL
        elif score >= self.high:
            return RiskLevel.HIGH
        elif score >= self.medium:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

@dataclass
class RiskWeights:
    """Weights for different risk dimensions"""
    financial: float = 0.30
    legal: float = 0.35
    operational: float = 0.20
    strategic: float = 0.15
    
    def normalize(self) -> 'RiskWeights':
        """Ensure weights sum to 1.0"""
        total = self.financial + self.legal + self.operational + self.strategic
        return RiskWeights(
            financial=self.financial / total,
            legal=self.legal / total,
            operational=self.operational / total,
            strategic=self.strategic / total
        )

class ClauseRiskConfig:
    """Configuration for clause-level risk assessment"""
    
    # Base risk scores by clause type (0-10 scale)
    BASE_RISK_SCORES = {
        # High-risk clause types
        "limitation_of_liability": 8.5,
        "indemnification": 8.0,
        "intellectual_property": 7.5,
        "termination": 7.0,
        "force_majeure": 6.8,
        
        # Medium-risk clause types
        "confidentiality": 6.0,
        "payment": 5.5,
        "governing_law": 5.0,
        "dispute_resolution": 5.0,
        "warranties": 4.8,
        
        # Lower-risk clause types
        "definitions": 3.0,
        "notices": 2.5,
        "assignment": 4.0,
        "amendment": 3.5,
        "entire_agreement": 2.0
    }
    
    # Language severity modifiers
    SEVERITY_MODIFIERS = {
        "absolute_terms": 1.5,      # "shall", "must", "absolute"
        "unlimited_scope": 1.4,     # "unlimited", "any and all"
        "broad_language": 1.3,      # "including but not limited to"
        "exclusive_terms": 1.2,     # "sole", "exclusive", "only"
        "immediate_effect": 1.3,    # "immediately", "forthwith"
        "perpetual_terms": 1.4,     # "perpetual", "permanent"
        "irrevocable_terms": 1.3,   # "irrevocable", "cannot be cancelled"
        
        "conditional_terms": 0.8,   # "if", "unless", "provided that"
        "reasonable_terms": 0.7,    # "reasonable", "commercially reasonable"
        "best_efforts": 0.6,        # "best efforts", "reasonable efforts"
        "standard_terms": 0.9,      # "standard", "customary"
    }
    
    # Scope impact multipliers
    SCOPE_MULTIPLIERS = {
        "global_scope": 1.3,        # Worldwide application
        "multi_jurisdiction": 1.2,   # Multiple legal jurisdictions
        "broad_application": 1.1,    # Wide-ranging application
        "specific_scope": 0.9,       # Limited, specific application
        "narrow_scope": 0.8,         # Very limited scope
    }

class IndustryRiskConfig:
    """Industry-specific risk configuration adjustments"""
    
    INDUSTRY_MULTIPLIERS = {
        "technology": {
            "intellectual_property": 1.3,
            "confidentiality": 1.2,
            "limitation_of_liability": 1.1
        },
        "healthcare": {
            "confidentiality": 1.4,
            "indemnification": 1.2,
            "regulatory_compliance": 1.3
        },
        "financial": {
            "regulatory_compliance": 1.4,
            "indemnification": 1.3,
            "payment": 1.2
        },
        "manufacturing": {
            "warranties": 1.3,
            "limitation_of_liability": 1.2,
            "termination": 1.1
        },
        "services": {
            "performance_standards": 1.2,
            "termination": 1.1,
            "payment": 1.1
        }
    }

class ComplianceRiskConfig:
    """Compliance and regulatory risk configuration"""
    
    REGULATORY_FRAMEWORKS = {
        "GDPR": {
            "required_clauses": ["data_protection", "privacy_rights", "data_processing"],
            "risk_multiplier": 1.5,
            "critical_violations": ["missing_data_protection", "inadequate_consent"]
        },
        "HIPAA": {
            "required_clauses": ["phi_protection", "business_associate", "breach_notification"],
            "risk_multiplier": 1.4,
            "critical_violations": ["missing_phi_protection", "inadequate_safeguards"]
        },
        "SOX": {
            "required_clauses": ["financial_controls", "audit_rights", "disclosure_requirements"],
            "risk_multiplier": 1.3,
            "critical_violations": ["missing_controls", "inadequate_disclosure"]
        },
        "PCI_DSS": {
            "required_clauses": ["payment_security", "cardholder_data", "security_standards"],
            "risk_multiplier": 1.4,
            "critical_violations": ["missing_security", "inadequate_protection"]
        }
    }

class RiskCalculationConfig:
    """Master configuration class for risk calculations"""
    
    def __init__(self, 
                 thresholds: Optional[RiskThresholds] = None,
                 weights: Optional[RiskWeights] = None,
                 industry: Optional[str] = None):
        self.thresholds = thresholds or RiskThresholds()
        self.weights = weights.normalize() if weights else RiskWeights().normalize()
        self.industry = industry
        self.clause_config = ClauseRiskConfig()
        self.industry_config = IndustryRiskConfig()
        self.compliance_config = ComplianceRiskConfig()
    
    def get_base_risk_score(self, clause_type: str) -> float:
        """Get base risk score for a clause type"""
        return self.clause_config.BASE_RISK_SCORES.get(clause_type, 5.0)
    
    def get_industry_multiplier(self, clause_type: str) -> float:
        """Get industry-specific multiplier for clause type"""
        if not self.industry:
            return 1.0
        
        industry_modifiers = self.industry_config.INDUSTRY_MULTIPLIERS.get(
            self.industry, {}
        )
        return industry_modifiers.get(clause_type, 1.0)
    
    def get_severity_modifier(self, severity_type: str) -> float:
        """Get language severity modifier"""
        return self.clause_config.SEVERITY_MODIFIERS.get(severity_type, 1.0)
    
    def get_scope_multiplier(self, scope_type: str) -> float:
        """Get scope impact multiplier"""
        return self.clause_config.SCOPE_MULTIPLIERS.get(scope_type, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "thresholds": {
                "critical": self.thresholds.critical,
                "high": self.thresholds.high,
                "medium": self.thresholds.medium,
                "low": self.thresholds.low
            },
            "weights": {
                "financial": self.weights.financial,
                "legal": self.weights.legal,
                "operational": self.weights.operational,
                "strategic": self.weights.strategic
            },
            "industry": self.industry,
            "base_risk_scores": self.clause_config.BASE_RISK_SCORES,
            "severity_modifiers": self.clause_config.SEVERITY_MODIFIERS,
            "scope_multipliers": self.clause_config.SCOPE_MULTIPLIERS
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RiskCalculationConfig':
        """Create configuration from dictionary"""
        thresholds = RiskThresholds(
            critical=config_dict.get("thresholds", {}).get("critical", 8.0),
            high=config_dict.get("thresholds", {}).get("high", 6.0),
            medium=config_dict.get("thresholds", {}).get("medium", 4.0),
            low=config_dict.get("thresholds", {}).get("low", 0.0)
        )
        
        weights_dict = config_dict.get("weights", {})
        weights = RiskWeights(
            financial=weights_dict.get("financial", 0.30),
            legal=weights_dict.get("legal", 0.35),
            operational=weights_dict.get("operational", 0.20),
            strategic=weights_dict.get("strategic", 0.15)
        )
        
        return cls(
            thresholds=thresholds,
            weights=weights,
            industry=config_dict.get("industry")
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RiskCalculationConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)