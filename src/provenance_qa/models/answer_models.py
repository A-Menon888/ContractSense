"""
Answer Generation Data Models

Data structures for answer generation, validation, and quality assessment
in provenance-aware QA systems.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class AnswerType(Enum):
    """Types of answers generated"""
    DIRECT = "direct"                   # Direct factual answer
    SUMMARY = "summary"                 # Summarized information
    ANALYSIS = "analysis"               # Analytical response
    COMPARISON = "comparison"           # Comparative analysis
    EXPLANATION = "explanation"         # Explanatory answer
    RECOMMENDATION = "recommendation"   # Advisory response
    NOT_FOUND = "not_found"            # Information not available
    AMBIGUOUS = "ambiguous"            # Multiple possible answers
    INSUFFICIENT = "insufficient"       # Insufficient context

class ConfidenceLevel(Enum):
    """Confidence levels for answers"""
    VERY_HIGH = "very_high"    # 90-100% confidence
    HIGH = "high"              # 75-90% confidence
    MODERATE = "moderate"      # 50-75% confidence
    LOW = "low"                # 25-50% confidence
    VERY_LOW = "very_low"      # 0-25% confidence

class ValidationStatus(Enum):
    """Validation status for answers"""
    VALIDATED = "validated"       # Passed all validation checks
    WARNING = "warning"           # Passed with warnings
    FAILED = "failed"            # Failed validation
    PENDING = "pending"          # Validation in progress
    SKIPPED = "skipped"          # Validation skipped

class AnswerSource(Enum):
    """Source of answer generation"""
    GEMINI_25_FLASH = "gemini_25_flash"  # Primary Gemini model
    FALLBACK_MODEL = "fallback_model"    # Fallback LLM
    TEMPLATE_BASED = "template_based"    # Template-based generation
    RETRIEVAL_ONLY = "retrieval_only"    # Direct retrieval result
    HYBRID = "hybrid"                    # Multiple sources combined

@dataclass
class Answer:
    """Generated answer with metadata and validation"""
    answer_id: str
    question_id: str
    
    # Answer content
    text: str = ""
    answer_type: AnswerType = AnswerType.DIRECT
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MODERATE
    
    # Generation metadata
    source: AnswerSource = AnswerSource.GEMINI_25_FLASH
    model_name: str = ""
    generation_time: float = 0.0
    tokens_used: int = 0
    
    # Content analysis
    key_points: List[str] = field(default_factory=list)
    supporting_facts: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Quality metrics
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    clarity_score: float = 0.0
    
    # Processing metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "text": self.text,
            "answer_type": self.answer_type.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "source": self.source.value,
            "model_name": self.model_name,
            "generation_time": self.generation_time,
            "tokens_used": self.tokens_used,
            "key_points": self.key_points,
            "supporting_facts": self.supporting_facts,
            "caveats": self.caveats,
            "limitations": self.limitations,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "relevance_score": self.relevance_score,
            "clarity_score": self.clarity_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_overall_quality_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'relevance': 0.25,
            'clarity': 0.20
        }
        
        return (
            self.completeness_score * weights['completeness'] +
            self.accuracy_score * weights['accuracy'] +
            self.relevance_score * weights['relevance'] +
            self.clarity_score * weights['clarity']
        )
    
    def is_high_quality(self, threshold: float = 0.75) -> bool:
        """Check if answer meets high quality threshold"""
        return (
            self.get_overall_quality_score() >= threshold and
            self.confidence >= threshold and
            self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        )
    
    def has_caveats_or_limitations(self) -> bool:
        """Check if answer has important caveats or limitations"""
        return len(self.caveats) > 0 or len(self.limitations) > 0
    
    def get_confidence_explanation(self) -> str:
        """Get human-readable confidence explanation"""
        confidence_explanations = {
            ConfidenceLevel.VERY_HIGH: "Very high confidence - answer is well-supported by multiple sources",
            ConfidenceLevel.HIGH: "High confidence - answer is supported by reliable sources",
            ConfidenceLevel.MODERATE: "Moderate confidence - answer has some support but may be incomplete",
            ConfidenceLevel.LOW: "Low confidence - limited support for this answer",
            ConfidenceLevel.VERY_LOW: "Very low confidence - answer is speculative or uncertain"
        }
        return confidence_explanations.get(self.confidence_level, "Unknown confidence level")

@dataclass
class ValidationCheck:
    """Individual validation check result"""
    check_name: str
    status: ValidationStatus
    score: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "score": self.score,
            "message": self.message,
            "details": self.details
        }

@dataclass
class ValidationResult:
    """Comprehensive validation result for an answer"""
    answer_id: str
    overall_status: ValidationStatus = ValidationStatus.PENDING
    overall_score: float = 0.0
    
    # Individual validation checks
    checks: List[ValidationCheck] = field(default_factory=list)
    
    # Validation categories
    factual_accuracy: float = 0.0
    source_reliability: float = 0.0
    logical_consistency: float = 0.0
    completeness: float = 0.0
    
    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Processing metadata
    validation_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "answer_id": self.answer_id,
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "checks": [check.to_dict() for check in self.checks],
            "factual_accuracy": self.factual_accuracy,
            "source_reliability": self.source_reliability,
            "logical_consistency": self.logical_consistency,
            "completeness": self.completeness,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "validation_time": self.validation_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def add_check(self, check: ValidationCheck):
        """Add a validation check result"""
        self.checks.append(check)
    
    def get_failed_checks(self) -> List[ValidationCheck]:
        """Get all checks that failed"""
        return [check for check in self.checks if check.status == ValidationStatus.FAILED]
    
    def get_warning_checks(self) -> List[ValidationCheck]:
        """Get all checks with warnings"""
        return [check for check in self.checks if check.status == ValidationStatus.WARNING]
    
    def has_critical_issues(self) -> bool:
        """Check if validation found critical issues"""
        return (
            len(self.critical_issues) > 0 or
            self.overall_status == ValidationStatus.FAILED or
            any(check.status == ValidationStatus.FAILED for check in self.checks)
        )
    
    def is_acceptable(self, min_score: float = 0.6) -> bool:
        """Check if answer is acceptable based on validation"""
        return (
            self.overall_status in [ValidationStatus.VALIDATED, ValidationStatus.WARNING] and
            self.overall_score >= min_score and
            not self.has_critical_issues()
        )
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary"""
        status_descriptions = {
            ValidationStatus.VALIDATED: "✓ Answer passed all validation checks",
            ValidationStatus.WARNING: "⚠ Answer passed with warnings",
            ValidationStatus.FAILED: "✗ Answer failed validation",
            ValidationStatus.PENDING: "◯ Validation in progress",
            ValidationStatus.SKIPPED: "- Validation skipped"
        }
        
        summary_parts = [status_descriptions.get(self.overall_status, "Unknown status")]
        
        if self.overall_score > 0:
            summary_parts.append(f"Score: {self.overall_score:.2f}")
        
        if self.critical_issues:
            summary_parts.append(f"Critical issues: {len(self.critical_issues)}")
        
        if self.warnings:
            summary_parts.append(f"Warnings: {len(self.warnings)}")
        
        return " | ".join(summary_parts)