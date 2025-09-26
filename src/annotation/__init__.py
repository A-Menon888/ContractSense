"""
Module 2: Annotation Framework & Dataset Construction

This module handles the creation and management of annotated datasets for training
ML models on contract clauses. It includes:

- Data structures for annotations (spans, tokens, documents)
- Dataset building and train/val splits
- Format conversion (JSON, CoNLL, etc.)
- Annotation validation and quality metrics
- CUAD dataset integration
- Inter-annotator agreement computation
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClauseType(Enum):
    """Supported clause types for annotation"""
    # Core contract clauses
    INDEMNIFICATION = "indemnification"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    TERMINATION = "termination"
    GOVERNING_LAW = "governing_law"
    PAYMENT_TERMS = "payment_terms"
    
    # IP and licensing
    IP_OWNERSHIP = "ip_ownership"
    LICENSE_GRANT = "license_grant"
    NON_COMPETE = "non_compete"
    CONFIDENTIALITY = "confidentiality"
    
    # Business terms
    EXCLUSIVITY = "exclusivity"
    RENEWAL = "renewal"
    FORCE_MAJEURE = "force_majeure"
    DISPUTE_RESOLUTION = "dispute_resolution"
    ASSIGNMENT = "assignment"
    WARRANTY = "warranty"


class LabelingScheme(Enum):
    """Supported labeling schemes for sequence labeling"""
    BIO = "BIO"      # Begin, Inside, Outside
    BIOS = "BIOS"    # Begin, Inside, Outside, Single
    IOBES = "IOBES"  # Inside, Outside, Begin, End, Single


class RiskLevel(Enum):
    """Risk levels for clause annotations"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SpanAnnotation:
    """Represents a text span annotation"""
    start_char: int
    end_char: int
    clause_type: str
    text: str
    confidence: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    notes: str = ""
    annotator_id: Optional[str] = None
    
    def __post_init__(self):
        # Validate span bounds
        if self.start_char >= self.end_char:
            raise ValueError(f"Invalid span: start_char ({self.start_char}) >= end_char ({self.end_char})")
        
        # Validate text length matches span
        expected_length = self.end_char - self.start_char
        if len(self.text) != expected_length:
            logger.warning(f"Text length ({len(self.text)}) doesn't match span length ({expected_length})")
    
    def overlaps_with(self, other: 'SpanAnnotation') -> bool:
        """Check if this span overlaps with another span"""
        return not (self.end_char <= other.start_char or other.end_char <= self.start_char)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "clause_type": self.clause_type,
            "text": self.text,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "notes": self.notes,
            "annotator_id": self.annotator_id
        }


@dataclass 
class TokenAnnotation:
    """Represents a token-level annotation for sequence labeling"""
    token: str
    label: str
    start_char: int
    end_char: int
    token_idx: int
    pos_tag: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "token": self.token,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_idx": self.token_idx,
            "pos_tag": self.pos_tag
        }


@dataclass
class DocumentAnnotation:
    """Complete annotation for a document"""
    document_id: str
    document_path: str
    full_text: str
    span_annotations: List[SpanAnnotation] = field(default_factory=list)
    token_annotations: List[TokenAnnotation] = field(default_factory=list)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: SpanAnnotation) -> None:
        """Add a span annotation with validation"""
        # Check for text consistency
        if span.end_char > len(self.full_text):
            raise ValueError(f"Span end ({span.end_char}) beyond document length ({len(self.full_text)})")
        
        expected_text = self.full_text[span.start_char:span.end_char]
        if span.text != expected_text:
            logger.warning(f"Span text mismatch: '{span.text[:50]}...' vs '{expected_text[:50]}...'")
        
        self.span_annotations.append(span)
    
    def get_spans_by_type(self, clause_type: str) -> List[SpanAnnotation]:
        """Get all spans of a specific clause type"""
        return [span for span in self.span_annotations if span.clause_type == clause_type]
    
    def get_overlapping_spans(self) -> List[Tuple[SpanAnnotation, SpanAnnotation]]:
        """Find all pairs of overlapping spans"""
        overlapping = []
        for i, span1 in enumerate(self.span_annotations):
            for span2 in self.span_annotations[i+1:]:
                if span1.overlaps_with(span2):
                    overlapping.append((span1, span2))
        return overlapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "document_id": self.document_id,
            "document_path": self.document_path,
            "full_text": self.full_text,
            "span_annotations": [span.to_dict() for span in self.span_annotations],
            "token_annotations": [token.to_dict() for token in self.token_annotations],
            "document_metadata": self.document_metadata,
            "annotation_metadata": self.annotation_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentAnnotation':
        """Create DocumentAnnotation from dictionary"""
        # Create span annotations
        span_annotations = []
        for span_data in data.get('span_annotations', []):
            span = SpanAnnotation(
                start_char=span_data['start_char'],
                end_char=span_data['end_char'],
                clause_type=span_data['clause_type'],
                text=span_data['text'],
                confidence=span_data.get('confidence'),
                risk_level=RiskLevel(span_data['risk_level']) if span_data.get('risk_level') else None,
                notes=span_data.get('notes', ''),
                annotator_id=span_data.get('annotator_id')
            )
            span_annotations.append(span)
        
        # Create token annotations
        token_annotations = []
        for token_data in data.get('token_annotations', []):
            token = TokenAnnotation(
                token=token_data['token'],
                label=token_data['label'],
                start_char=token_data['start_char'],
                end_char=token_data['end_char'],
                token_idx=token_data['token_idx'],
                pos_tag=token_data.get('pos_tag')
            )
            token_annotations.append(token)
        
        return cls(
            document_id=data['document_id'],
            document_path=data['document_path'],
            full_text=data['full_text'],
            span_annotations=span_annotations,
            token_annotations=token_annotations,
            document_metadata=data.get('document_metadata', {}),
            annotation_metadata=data.get('annotation_metadata', {})
        )


@dataclass
class AnnotationStats:
    """Statistics about an annotation dataset"""
    total_documents: int
    annotated_documents: int
    total_spans: int
    spans_per_clause_type: Dict[str, int]
    average_spans_per_document: float
    documents_by_risk_level: Dict[str, int] = field(default_factory=dict)
    annotation_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_documents": self.total_documents,
            "annotated_documents": self.annotated_documents,
            "total_spans": self.total_spans,
            "spans_per_clause_type": self.spans_per_clause_type,
            "average_spans_per_document": self.average_spans_per_document,
            "documents_by_risk_level": self.documents_by_risk_level,
            "annotation_coverage": self.annotation_coverage
        }


# Import main classes from submodules
from .schema import AnnotationSchema
from .dataset_builder import DatasetBuilder
from .validator import AnnotationValidator
from .cuad_converter import CUADConverter