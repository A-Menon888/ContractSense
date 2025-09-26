"""
Question Analysis Data Models

Data structures for comprehensive question understanding, entity extraction,
and intent classification in legal document QA scenarios.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class QuestionType(Enum):
    """Types of questions the QA system can handle"""
    FACTUAL = "factual"              # "What is the termination clause?"
    COMPARATIVE = "comparative"       # "How do these contracts differ?"
    ANALYTICAL = "analytical"         # "What are the risks in this agreement?"
    PROCEDURAL = "procedural"         # "How to exercise the renewal option?"
    DEFINITIONAL = "definitional"     # "What does 'material breach' mean?"
    QUANTITATIVE = "quantitative"     # "What is the maximum penalty amount?"
    TEMPORAL = "temporal"             # "When does the contract expire?"
    CONDITIONAL = "conditional"       # "What happens if payment is late?"

class QuestionIntent(Enum):
    """Intent behind the question"""
    INFORMATION_SEEKING = "information_seeking"  # General information request
    DECISION_SUPPORT = "decision_support"       # Help with decision making
    COMPLIANCE_CHECK = "compliance_check"       # Regulatory compliance
    RISK_ASSESSMENT = "risk_assessment"         # Risk evaluation
    VERIFICATION = "verification"               # Fact checking
    EXPLANATION = "explanation"                 # Understanding concepts
    COMPARISON = "comparison"                   # Comparing options
    PLANNING = "planning"                       # Strategic planning

class ComplexityLevel(Enum):
    """Complexity assessment for question processing"""
    SIMPLE = "simple"           # Single document, direct lookup
    MODERATE = "moderate"       # Multiple sources, some reasoning
    COMPLEX = "complex"         # Cross-document analysis, inference
    EXPERT = "expert"          # Requires deep legal knowledge

@dataclass
class Entity:
    """Extracted entity from question"""
    text: str
    entity_type: str  # person, organization, date, amount, clause_type, etc.
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

@dataclass
class QuestionAnalysis:
    """Comprehensive analysis of a user question"""
    original_question: str
    normalized_question: str = ""
    
    # Classification results
    question_type: QuestionType = QuestionType.FACTUAL
    intent: QuestionIntent = QuestionIntent.INFORMATION_SEEKING
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    
    # Extracted information
    entities: List[Entity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)
    
    # Processing metadata
    confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "question_type": self.question_type.value,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "entities": [entity.to_dict() for entity in self.entities],
            "keywords": self.keywords,
            "legal_concepts": self.legal_concepts,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_entity_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def has_entity_type(self, entity_type: str) -> bool:
        """Check if question contains entities of specific type"""
        return any(entity.entity_type == entity_type for entity in self.entities)
    
    def is_multi_document(self) -> bool:
        """Determine if question likely requires multiple documents"""
        multi_doc_indicators = [
            "compare", "contrast", "difference", "similar", "both",
            "between", "across", "multiple", "various", "all"
        ]
        question_lower = self.original_question.lower()
        return any(indicator in question_lower for indicator in multi_doc_indicators)
    
    def requires_reasoning(self) -> bool:
        """Determine if question requires complex reasoning"""
        return (
            self.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT] or
            self.question_type in [QuestionType.ANALYTICAL, QuestionType.COMPARATIVE] or
            self.intent in [QuestionIntent.DECISION_SUPPORT, QuestionIntent.RISK_ASSESSMENT]
        )