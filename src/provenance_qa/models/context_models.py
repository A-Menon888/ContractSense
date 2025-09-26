"""
Context Management Data Models

Data structures for managing document context windows, context assembly,
and context optimization for provenance-aware QA.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set

class ChunkType(Enum):
    """Types of document chunks"""
    PARAGRAPH = "paragraph"         # Standard paragraph chunk
    SECTION = "section"            # Document section (e.g., clause)
    TABLE = "table"                # Tabular data
    LIST = "list"                  # List or enumeration
    HEADER = "header"              # Section headers
    FOOTER = "footer"              # Document footers
    SIGNATURE = "signature"        # Signature blocks
    METADATA = "metadata"          # Document metadata

class RetrievalSource(Enum):
    """Source of chunk retrieval"""
    VECTOR_SEARCH = "vector_search"     # From vector similarity search
    GRAPH_TRAVERSAL = "graph_traversal" # From knowledge graph
    KEYWORD_MATCH = "keyword_match"     # From keyword matching
    CROSS_ENCODER = "cross_encoder"     # From cross-encoder reranking
    HYBRID = "hybrid"                   # From hybrid retrieval
    MANUAL = "manual"                   # Manually specified

class ContextStrategy(Enum):
    """Strategies for context assembly"""
    FOCUSED = "focused"             # Minimal, highly relevant context
    COMPREHENSIVE = "comprehensive" # Broad context with multiple perspectives
    HIERARCHICAL = "hierarchical"   # Structured by document hierarchy
    TEMPORAL = "temporal"           # Ordered by temporal relevance
    SIMILARITY = "similarity"       # Ordered by similarity scores

@dataclass
class DocumentChunk:
    """Individual document chunk with metadata and provenance"""
    chunk_id: str
    document_id: str
    document_title: str = ""
    
    # Content information
    content: str = ""
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    start_char: int = 0
    end_char: int = 0
    
    # Retrieval information
    retrieval_source: RetrievalSource = RetrievalSource.VECTOR_SEARCH
    relevance_score: float = 0.0
    similarity_score: float = 0.0
    keyword_matches: List[str] = field(default_factory=list)
    
    # Context information
    section_title: str = ""
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    preceding_context: str = ""
    following_context: str = ""
    
    # Processing metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "retrieval_source": self.retrieval_source.value,
            "relevance_score": self.relevance_score,
            "similarity_score": self.similarity_score,
            "keyword_matches": self.keyword_matches,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_full_context(self, include_surrounding: bool = True) -> str:
        """Get full context including surrounding text if available"""
        if not include_surrounding:
            return self.content
        
        parts = []
        if self.preceding_context:
            parts.append(f"[...{self.preceding_context}]")
        
        parts.append(self.content)
        
        if self.following_context:
            parts.append(f"[{self.following_context}...]")
        
        return " ".join(parts)
    
    def get_location_info(self) -> str:
        """Get human-readable location information"""
        location_parts = []
        
        if self.document_title:
            location_parts.append(f"Document: {self.document_title}")
        
        if self.section_title:
            location_parts.append(f"Section: {self.section_title}")
        
        if self.page_number:
            location_parts.append(f"Page: {self.page_number}")
        
        if self.paragraph_number:
            location_parts.append(f"Paragraph: {self.paragraph_number}")
        
        return ", ".join(location_parts) if location_parts else f"Document ID: {self.document_id}"

@dataclass
class ContextWindow:
    """Assembled context window for question answering"""
    window_id: str
    question_id: str
    
    # Context composition
    primary_chunks: List[DocumentChunk] = field(default_factory=list)
    supporting_chunks: List[DocumentChunk] = field(default_factory=list)
    background_chunks: List[DocumentChunk] = field(default_factory=list)
    
    # Assembly strategy
    strategy: ContextStrategy = ContextStrategy.FOCUSED
    max_tokens: int = 4000
    actual_tokens: int = 0
    
    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    diversity_score: float = 0.0
    
    # Processing metadata
    assembly_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "window_id": self.window_id,
            "question_id": self.question_id,
            "primary_chunks": [chunk.to_dict() for chunk in self.primary_chunks],
            "supporting_chunks": [chunk.to_dict() for chunk in self.supporting_chunks],
            "background_chunks": [chunk.to_dict() for chunk in self.background_chunks],
            "strategy": self.strategy.value,
            "max_tokens": self.max_tokens,
            "actual_tokens": self.actual_tokens,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "diversity_score": self.diversity_score,
            "assembly_time": self.assembly_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in priority order"""
        return self.primary_chunks + self.supporting_chunks + self.background_chunks
    
    def get_unique_documents(self) -> Set[str]:
        """Get set of unique document IDs in context"""
        all_chunks = self.get_all_chunks()
        return {chunk.document_id for chunk in all_chunks}
    
    def get_chunk_count(self) -> Dict[str, int]:
        """Get count of chunks by type"""
        return {
            "primary": len(self.primary_chunks),
            "supporting": len(self.supporting_chunks),
            "background": len(self.background_chunks),
            "total": len(self.get_all_chunks())
        }
    
    def get_formatted_context(self, include_metadata: bool = True) -> str:
        """Get formatted context for LLM input"""
        context_parts = []
        
        if include_metadata:
            context_parts.append(f"=== CONTEXT WINDOW ({self.strategy.value.upper()} STRATEGY) ===")
            context_parts.append(f"Documents: {len(self.get_unique_documents())}")
            context_parts.append(f"Chunks: {self.get_chunk_count()['total']}")
            context_parts.append("")
        
        # Primary chunks (most important)
        if self.primary_chunks:
            context_parts.append("=== PRIMARY CONTEXT ===")
            for i, chunk in enumerate(self.primary_chunks, 1):
                location = chunk.get_location_info()
                context_parts.append(f"[{i}] {location}")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        # Supporting chunks
        if self.supporting_chunks:
            context_parts.append("=== SUPPORTING CONTEXT ===")
            for i, chunk in enumerate(self.supporting_chunks, 1):
                location = chunk.get_location_info()
                context_parts.append(f"[S{i}] {location}")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        # Background chunks (if space allows)
        if self.background_chunks:
            context_parts.append("=== BACKGROUND CONTEXT ===")
            for i, chunk in enumerate(self.background_chunks, 1):
                location = chunk.get_location_info()
                context_parts.append(f"[B{i}] {location}")
                context_parts.append(chunk.content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def optimize_for_tokens(self, target_tokens: int) -> 'ContextWindow':
        """Create optimized version within token limit"""
        if self.actual_tokens <= target_tokens:
            return self
        
        # Create new optimized context window
        optimized = ContextWindow(
            window_id=f"{self.window_id}_opt",
            question_id=self.question_id,
            strategy=self.strategy,
            max_tokens=target_tokens
        )
        
        # Prioritize chunks by relevance score
        all_chunks = sorted(
            self.get_all_chunks(),
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        current_tokens = 0
        for chunk in all_chunks:
            # Rough token estimation (4 chars per token)
            chunk_tokens = len(chunk.content) // 4
            if current_tokens + chunk_tokens <= target_tokens:
                optimized.primary_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        optimized.actual_tokens = current_tokens
        return optimized