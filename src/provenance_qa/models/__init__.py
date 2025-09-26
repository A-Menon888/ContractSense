"""
Provenance QA Data Models

Core data structures for the Provenance-Aware Question Answering System.
These models ensure type safety, data consistency, and seamless integration
across all QA components.

Author: ContractSense Team
Date: 2025-09-26
Version: 1.0.0
"""

from .question_models import (
    QuestionType,
    QuestionIntent,
    ComplexityLevel,
    Entity,
    QuestionAnalysis
)

from .context_models import (
    ChunkType,
    RetrievalSource,
    ContextStrategy,
    DocumentChunk,
    ContextWindow
)

from .answer_models import (
    AnswerType,
    ConfidenceLevel,
    ValidationStatus,
    AnswerSource,
    Answer,
    ValidationCheck,
    ValidationResult
)

from .provenance_models import (
    CitationType,
    SourceType,
    CertaintyLevel,
    Citation,
    ProvenanceChain,
    QAResponse
)

__all__ = [
    # Question models
    'QuestionType',
    'QuestionIntent', 
    'ComplexityLevel',
    'Entity',
    'QuestionAnalysis',
    
    # Context models
    'ChunkType',
    'RetrievalSource', 
    'ContextStrategy',
    'DocumentChunk',
    'ContextWindow',
    
    # Answer models
    'AnswerType',
    'ConfidenceLevel',
    'ValidationStatus',
    'AnswerSource',
    'Answer',
    'ValidationCheck',
    'ValidationResult',
    
    # Provenance models
    'CitationType',
    'SourceType',
    'CertaintyLevel',
    'Citation',
    'ProvenanceChain',
    'QAResponse'
]