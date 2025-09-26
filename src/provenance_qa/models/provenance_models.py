"""
Provenance Tracking Data Models

Data structures for comprehensive provenance tracking, citation management,
and traceability in QA responses.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union

class CitationType(Enum):
    """Types of citations"""
    DIRECT_QUOTE = "direct_quote"       # Exact text from source
    PARAPHRASE = "paraphrase"          # Paraphrased content
    SUMMARY = "summary"                # Summarized information
    INFERENCE = "inference"            # Inferred from source
    REFERENCE = "reference"            # General reference
    CALCULATION = "calculation"        # Derived calculation

class SourceType(Enum):
    """Types of information sources"""
    DOCUMENT = "document"              # Legal document
    SECTION = "section"                # Document section
    CLAUSE = "clause"                  # Specific clause
    TABLE = "table"                    # Tabular data
    METADATA = "metadata"              # Document metadata
    ANNOTATION = "annotation"          # Human annotation
    KNOWLEDGE_GRAPH = "knowledge_graph" # Graph-derived info

class CertaintyLevel(Enum):
    """Levels of certainty for provenance"""
    CERTAIN = "certain"                # 100% certain
    HIGH = "high"                      # Very confident
    MEDIUM = "medium"                  # Reasonably confident
    LOW = "low"                        # Some uncertainty
    SPECULATIVE = "speculative"        # Highly uncertain

@dataclass
class Citation:
    """Individual citation with detailed provenance information"""
    citation_id: str
    
    # Source identification
    source_id: str
    source_type: SourceType = SourceType.DOCUMENT
    source_title: str = ""
    
    # Content information
    cited_text: str = ""
    citation_type: CitationType = CitationType.REFERENCE
    context_before: str = ""
    context_after: str = ""
    
    # Location information
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    section_title: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Relevance and quality
    relevance_score: float = 0.0
    quality_score: float = 0.0
    certainty: CertaintyLevel = CertaintyLevel.MEDIUM
    
    # Usage information
    supports_claim: str = ""
    usage_notes: str = ""
    
    # Processing metadata
    extraction_method: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "source_title": self.source_title,
            "cited_text": self.cited_text,
            "citation_type": self.citation_type.value,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "section_title": self.section_title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "relevance_score": self.relevance_score,
            "quality_score": self.quality_score,
            "certainty": self.certainty.value,
            "supports_claim": self.supports_claim,
            "usage_notes": self.usage_notes,
            "extraction_method": self.extraction_method,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_location_string(self) -> str:
        """Get human-readable location string"""
        parts = []
        
        if self.source_title:
            parts.append(self.source_title)
        
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        
        if self.paragraph_number:
            parts.append(f"Paragraph {self.paragraph_number}")
        
        return ", ".join(parts) if parts else f"Source: {self.source_id}"
    
    def get_formatted_citation(self, style: str = "legal") -> str:
        """Get formatted citation in specified style"""
        if style == "legal":
            return f"{self.get_location_string()}: \"{self.cited_text}\""
        elif style == "academic":
            return f"({self.source_title}, p. {self.page_number or 'unknown'})"
        elif style == "inline":
            return f"[{self.source_title}]"
        else:
            return self.get_location_string()
    
    def is_high_quality(self, min_score: float = 0.7) -> bool:
        """Check if citation meets high quality threshold"""
        return (
            self.quality_score >= min_score and
            self.relevance_score >= min_score and
            self.certainty in [CertaintyLevel.HIGH, CertaintyLevel.CERTAIN]
        )

@dataclass
class ProvenanceChain:
    """Chain of provenance showing derivation path"""
    chain_id: str
    
    # Chain components
    original_sources: List[Citation] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    
    # Chain metadata
    derivation_method: str = ""
    confidence: float = 0.0
    complexity_score: float = 0.0
    
    # Quality assessment
    source_reliability: float = 0.0
    logical_consistency: float = 0.0
    completeness: float = 0.0
    
    # Processing information
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chain_id": self.chain_id,
            "original_sources": [citation.to_dict() for citation in self.original_sources],
            "intermediate_steps": self.intermediate_steps,
            "final_output": self.final_output,
            "derivation_method": self.derivation_method,
            "confidence": self.confidence,
            "complexity_score": self.complexity_score,
            "source_reliability": self.source_reliability,
            "logical_consistency": self.logical_consistency,
            "completeness": self.completeness,
            "creation_time": self.creation_time,
            "metadata": self.metadata
        }
    
    def add_step(self, step_description: str, input_data: Any, output_data: Any, method: str = ""):
        """Add intermediate processing step to chain"""
        step = {
            "step_number": len(self.intermediate_steps) + 1,
            "description": step_description,
            "input": str(input_data)[:1000],  # Truncate for storage
            "output": str(output_data)[:1000],
            "method": method,
            "timestamp": time.time()
        }
        self.intermediate_steps.append(step)
    
    def get_source_count(self) -> int:
        """Get number of original sources"""
        return len(self.original_sources)
    
    def get_unique_documents(self) -> List[str]:
        """Get list of unique source documents"""
        documents = set()
        for citation in self.original_sources:
            documents.add(citation.source_id)
        return list(documents)
    
    def validate_chain(self) -> List[str]:
        """Validate provenance chain and return issues"""
        issues = []
        
        if not self.original_sources:
            issues.append("No original sources in provenance chain")
        
        if not self.final_output:
            issues.append("No final output in provenance chain")
        
        if self.confidence < 0.5:
            issues.append("Low confidence in provenance chain")
        
        # Check for gaps in reasoning
        if len(self.intermediate_steps) == 0 and self.complexity_score > 0.5:
            issues.append("Complex derivation with no intermediate steps")
        
        return issues

@dataclass
class QAResponse:
    """Complete QA response with full provenance tracking"""
    response_id: str
    question_id: str
    
    # Response content
    answer: str = ""
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    
    # Provenance information
    citations: List[Citation] = field(default_factory=list)
    provenance_chains: List[ProvenanceChain] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    
    # Quality and confidence
    overall_confidence: float = 0.0
    answer_quality: float = 0.0
    source_reliability: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    model_used: str = ""
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Response metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "response_id": self.response_id,
            "question_id": self.question_id,
            "answer": self.answer,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "citations": [citation.to_dict() for citation in self.citations],
            "provenance_chains": [chain.to_dict() for chain in self.provenance_chains],
            "source_documents": self.source_documents,
            "overall_confidence": self.overall_confidence,
            "answer_quality": self.answer_quality,
            "source_reliability": self.source_reliability,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "retrieval_stats": self.retrieval_stats,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def add_citation(self, citation: Citation):
        """Add citation to response"""
        self.citations.append(citation)
        if citation.source_id not in self.source_documents:
            self.source_documents.append(citation.source_id)
    
    def add_provenance_chain(self, chain: ProvenanceChain):
        """Add provenance chain to response"""
        self.provenance_chains.append(chain)
    
    def get_citation_by_type(self, citation_type: CitationType) -> List[Citation]:
        """Get citations of specific type"""
        return [c for c in self.citations if c.citation_type == citation_type]
    
    def get_high_confidence_citations(self, threshold: float = 0.8) -> List[Citation]:
        """Get citations with high confidence"""
        return [c for c in self.citations if c.quality_score >= threshold]
    
    def get_formatted_response(self, include_citations: bool = True, citation_style: str = "legal") -> str:
        """Get formatted response with optional citations"""
        parts = [self.answer]
        
        if include_citations and self.citations:
            parts.append("\n\nSources:")
            for i, citation in enumerate(self.citations, 1):
                formatted_citation = citation.get_formatted_citation(citation_style)
                parts.append(f"{i}. {formatted_citation}")
        
        if self.key_findings:
            parts.append("\n\nKey Findings:")
            for finding in self.key_findings:
                parts.append(f"• {finding}")
        
        return "\n".join(parts)
    
    def validate_provenance(self) -> List[str]:
        """Validate all provenance information"""
        issues = []
        
        if not self.citations:
            issues.append("No citations provided for answer")
        
        if not self.source_documents:
            issues.append("No source documents identified")
        
        # Validate each provenance chain
        for chain in self.provenance_chains:
            chain_issues = chain.validate_chain()
            issues.extend([f"Chain {chain.chain_id}: {issue}" for issue in chain_issues])
        
        return issues
    
    def get_provenance_summary(self) -> Dict[str, Any]:
        """Get summary of provenance information"""
        return {
            "citation_count": len(self.citations),
            "source_document_count": len(self.source_documents),
            "provenance_chain_count": len(self.provenance_chains),
            "average_citation_quality": sum(c.quality_score for c in self.citations) / len(self.citations) if self.citations else 0,
            "citation_types": list(set(c.citation_type.value for c in self.citations)),
            "source_types": list(set(c.source_type.value for c in self.citations))
        }