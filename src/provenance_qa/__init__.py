"""
Provenance-Aware QA System for ContractSense

Complete question answering system with full provenance tracking, 
citation management, and integration with Gemini 2.5 Flash API.

This module provides comprehensive QA capabilities including:
- Intelligent question analysis and classification
- Context assembly and optimization
- Answer generation using Gemini 2.5 Flash
- Complete provenance tracking and citation management
- Answer validation and quality assessment
"""

from .qa_engine import ProvenanceQAEngine
from .question_processor import QuestionProcessor
from .context_assembler import ContextAssembler
from .answer_generator import AnswerGenerator
from .provenance_tracker import ProvenanceTracker
from .answer_validator import AnswerValidator

# Import data models
from .models.question_models import (
    QuestionType, QuestionIntent, ComplexityLevel, 
    Entity, QuestionAnalysis
)
from .models.context_models import (
    DocumentChunk, ContextWindow, ContextStrategy,
    ChunkType, RetrievalSource
)
from .models.answer_models import (
    Answer, AnswerType, ConfidenceLevel, AnswerSource,
    ValidationResult, ValidationCheck, ValidationStatus
)
from .models.provenance_models import (
    Citation, CitationType, SourceType, CertaintyLevel,
    ProvenanceChain, QAResponse
)

# Import utilities
from .utils.common import (
    generate_id, calculate_confidence_score, Timer, RateLimiter
)

__version__ = "1.0.0"

__all__ = [
    # Main engine
    "ProvenanceQAEngine",
    
    # Core components
    "QuestionProcessor",
    "ContextAssembler", 
    "AnswerGenerator",
    "ProvenanceTracker",
    "AnswerValidator",
    
    # Question models
    "QuestionType",
    "QuestionIntent", 
    "ComplexityLevel",
    "Entity",
    "QuestionAnalysis",
    
    # Context models
    "DocumentChunk",
    "ContextWindow",
    "ContextStrategy",
    "ChunkType",
    "RetrievalSource",
    
    # Answer models
    "Answer",
    "AnswerType",
    "ConfidenceLevel", 
    "AnswerSource",
    "ValidationResult",
    "ValidationCheck",
    "ValidationStatus",
    
    # Provenance models
    "Citation",
    "CitationType",
    "SourceType", 
    "CertaintyLevel",
    "ProvenanceChain",
    "QAResponse",
    
    # Utilities
    "generate_id",
    "calculate_confidence_score",
    "Timer",
    "RateLimiter"
]

def create_qa_engine(
    gemini_api_key: str = None,
    workspace_path: str = None,
    enable_validation: bool = True,
    context_strategy: ContextStrategy = ContextStrategy.FOCUSED,
    max_context_tokens: int = 4000
) -> ProvenanceQAEngine:
    """
    Factory function to create a configured QA engine
    
    Args:
        gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
        workspace_path: Path to ContractSense workspace
        enable_validation: Whether to enable answer validation
        context_strategy: Default context assembly strategy
        max_context_tokens: Maximum tokens for context
    
    Returns:
        Configured ProvenanceQAEngine instance
    """
    
    return ProvenanceQAEngine(
        gemini_api_key=gemini_api_key,
        max_context_tokens=max_context_tokens,
        enable_validation=enable_validation,
        context_strategy=context_strategy,
        workspace_path=workspace_path
    )

def quick_question(
    question: str,
    gemini_api_key: str = None,
    **kwargs
) -> QAResponse:
    """
    Quick utility function to ask a question with minimal setup
    
    Args:
        question: The question to ask
        gemini_api_key: Gemini API key (optional)
        **kwargs: Additional arguments for QA engine
    
    Returns:
        QA response with answer and provenance
    """
    
    engine = create_qa_engine(gemini_api_key=gemini_api_key, **kwargs)
    return engine.ask_question(question)