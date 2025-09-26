"""
Context Assembly Component

Assembles comprehensive context windows from retrieved documents for optimal QA performance.
Manages context optimization, chunk prioritization, and token management.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple

try:
    from ..models.context_models import (
        DocumentChunk, ContextWindow, ContextStrategy, ChunkType, RetrievalSource
    )
    from ..models.question_models import QuestionAnalysis
    from ..utils.common import (
        generate_id, estimate_tokens, calculate_confidence_score,
        calculate_text_similarity, Timer
    )
except ImportError:
    # Fallback for direct execution
    from provenance_qa.models.context_models import (
        DocumentChunk, ContextWindow, ContextStrategy, ChunkType, RetrievalSource
    )
    from provenance_qa.models.question_models import QuestionAnalysis
    from provenance_qa.utils.common import (
        generate_id, estimate_tokens, calculate_confidence_score,
        calculate_text_similarity, Timer
    )

logger = logging.getLogger(__name__)

class ContextAssembler:
    """Advanced context window assembly and optimization"""
    
    def __init__(self, default_max_tokens: int = 4000):
        self.default_max_tokens = default_max_tokens
        self.chunk_type_priorities = self._build_chunk_priorities()
        self.source_weights = self._build_source_weights()
    
    def _build_chunk_priorities(self) -> Dict[ChunkType, float]:
        """Build priority weights for different chunk types"""
        return {
            ChunkType.PARAGRAPH: 1.0,
            ChunkType.SECTION: 0.9,
            ChunkType.LIST: 0.8,
            ChunkType.TABLE: 0.7,
            ChunkType.HEADER: 0.6,
            ChunkType.METADATA: 0.4,
            ChunkType.FOOTER: 0.3,
            ChunkType.SIGNATURE: 0.2
        }
    
    def _build_source_weights(self) -> Dict[RetrievalSource, float]:
        """Build weights for different retrieval sources"""
        return {
            RetrievalSource.CROSS_ENCODER: 1.0,
            RetrievalSource.HYBRID: 0.95,
            RetrievalSource.VECTOR_SEARCH: 0.8,
            RetrievalSource.GRAPH_TRAVERSAL: 0.7,
            RetrievalSource.KEYWORD_MATCH: 0.6,
            RetrievalSource.MANUAL: 0.5
        }
    
    def assemble_context(
        self,
        question_analysis: QuestionAnalysis,
        retrieved_chunks: List[DocumentChunk],
        strategy: ContextStrategy = ContextStrategy.FOCUSED,
        max_tokens: Optional[int] = None
    ) -> ContextWindow:
        """Assemble optimal context window from retrieved chunks"""
        
        with Timer() as timer:
            # Use default max_tokens if not specified
            if max_tokens is None:
                max_tokens = self.default_max_tokens
            
            # Generate context window ID
            window_id = generate_id("ctx")
            question_id = generate_id("q")  # This should come from question analysis
            
            # Create context window
            context_window = ContextWindow(
                window_id=window_id,
                question_id=question_id,
                strategy=strategy,
                max_tokens=max_tokens
            )
            
            # Optimize chunks based on strategy
            optimized_chunks = self._optimize_chunks(
                question_analysis, retrieved_chunks, strategy
            )
            
            # Categorize chunks by importance
            primary_chunks, supporting_chunks, background_chunks = self._categorize_chunks(
                optimized_chunks, question_analysis
            )
            
            # Fit chunks within token limit
            fitted_chunks = self._fit_chunks_to_token_limit(
                primary_chunks, supporting_chunks, background_chunks, max_tokens
            )
            
            # Assign chunks to context window
            context_window.primary_chunks = fitted_chunks["primary"]
            context_window.supporting_chunks = fitted_chunks["supporting"] 
            context_window.background_chunks = fitted_chunks["background"]
            
            # Calculate actual token usage
            context_window.actual_tokens = self._calculate_token_usage(context_window)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(context_window, question_analysis)
            
            # Record assembly time
            context_window.assembly_time = timer.elapsed()
            
            logger.info(f"Assembled context window: {len(context_window.get_all_chunks())} chunks, "
                       f"{context_window.actual_tokens} tokens")
            
            return context_window
    
    def _optimize_chunks(
        self,
        question_analysis: QuestionAnalysis,
        chunks: List[DocumentChunk],
        strategy: ContextStrategy
    ) -> List[DocumentChunk]:
        """Optimize chunk selection and ordering based on strategy"""
        
        if strategy == ContextStrategy.FOCUSED:
            # Focus on highest relevance chunks
            return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        
        elif strategy == ContextStrategy.COMPREHENSIVE:
            # Balance relevance with diversity
            return self._diversify_chunks(chunks, question_analysis)
        
        elif strategy == ContextStrategy.HIERARCHICAL:
            # Order by document structure
            return sorted(chunks, key=lambda c: (c.document_id, c.start_char))
        
        elif strategy == ContextStrategy.TEMPORAL:
            # Order by temporal relevance (if available)
            return sorted(chunks, key=lambda c: c.timestamp, reverse=True)
        
        elif strategy == ContextStrategy.SIMILARITY:
            # Order by similarity to question
            return self._order_by_similarity(chunks, question_analysis)
        
        else:
            return chunks
    
    def _diversify_chunks(
        self,
        chunks: List[DocumentChunk],
        question_analysis: QuestionAnalysis
    ) -> List[DocumentChunk]:
        """Diversify chunk selection to avoid redundancy"""
        if not chunks:
            return chunks
        
        diversified = []
        remaining_chunks = chunks.copy()
        
        # Start with highest relevance chunk
        remaining_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
        diversified.append(remaining_chunks.pop(0))
        
        # Add chunks that are different from already selected ones
        while remaining_chunks and len(diversified) < len(chunks):
            best_chunk = None
            best_diversity_score = -1
            
            for chunk in remaining_chunks:
                # Calculate diversity score (lower similarity = higher diversity)
                similarity_sum = sum(
                    calculate_text_similarity(chunk.content, selected.content)
                    for selected in diversified
                )
                diversity_score = chunk.relevance_score - (similarity_sum / len(diversified))
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_chunk = chunk
            
            if best_chunk:
                diversified.append(best_chunk)
                remaining_chunks.remove(best_chunk)
            else:
                break
        
        return diversified
    
    def _order_by_similarity(
        self,
        chunks: List[DocumentChunk],
        question_analysis: QuestionAnalysis
    ) -> List[DocumentChunk]:
        """Order chunks by similarity to the question"""
        question_text = question_analysis.normalized_question
        
        for chunk in chunks:
            similarity = calculate_text_similarity(chunk.content, question_text)
            # Combine with existing relevance score
            chunk.similarity_score = (similarity + chunk.relevance_score) / 2
        
        return sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
    
    def _categorize_chunks(
        self,
        chunks: List[DocumentChunk],
        question_analysis: QuestionAnalysis
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk], List[DocumentChunk]]:
        """Categorize chunks by importance level"""
        
        if not chunks:
            return [], [], []
        
        # Calculate importance scores
        scored_chunks = []
        for chunk in chunks:
            importance_score = self._calculate_chunk_importance(chunk, question_analysis)
            scored_chunks.append((chunk, importance_score))
        
        # Sort by importance
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Determine thresholds based on score distribution
        scores = [score for _, score in scored_chunks]
        high_threshold = max(scores) * 0.8 if scores else 0.8
        medium_threshold = max(scores) * 0.6 if scores else 0.6
        
        primary_chunks = []
        supporting_chunks = []
        background_chunks = []
        
        for chunk, score in scored_chunks:
            if score >= high_threshold:
                primary_chunks.append(chunk)
            elif score >= medium_threshold:
                supporting_chunks.append(chunk)
            else:
                background_chunks.append(chunk)
        
        return primary_chunks, supporting_chunks, background_chunks
    
    def _calculate_chunk_importance(
        self,
        chunk: DocumentChunk,
        question_analysis: QuestionAnalysis
    ) -> float:
        """Calculate importance score for a chunk"""
        importance_factors = {}
        
        # Base relevance score
        importance_factors["relevance"] = chunk.relevance_score
        
        # Chunk type priority
        type_priority = self.chunk_type_priorities.get(chunk.chunk_type, 0.5)
        importance_factors["type_priority"] = type_priority
        
        # Source reliability
        source_weight = self.source_weights.get(chunk.retrieval_source, 0.5)
        importance_factors["source_weight"] = source_weight
        
        # Keyword matches
        keyword_score = min(1.0, len(chunk.keyword_matches) / 5)
        importance_factors["keywords"] = keyword_score
        
        # Content length (prefer substantial content)
        content_length = len(chunk.content)
        if 100 <= content_length <= 1000:
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100
        else:
            length_score = max(0.3, 1000 / content_length)
        
        importance_factors["content_length"] = length_score
        
        # Entity matches (if question has entities)
        entity_score = 0.0
        if question_analysis.entities:
            entity_texts = [e.text.lower() for e in question_analysis.entities]
            chunk_lower = chunk.content.lower()
            matches = sum(1 for entity_text in entity_texts if entity_text in chunk_lower)
            entity_score = min(1.0, matches / len(question_analysis.entities))
        
        importance_factors["entities"] = entity_score
        
        # Calculate weighted importance score
        weights = {
            "relevance": 0.3,
            "type_priority": 0.15,
            "source_weight": 0.15,
            "keywords": 0.15,
            "content_length": 0.1,
            "entities": 0.15
        }
        
        return calculate_confidence_score(importance_factors, weights)
    
    def _fit_chunks_to_token_limit(
        self,
        primary_chunks: List[DocumentChunk],
        supporting_chunks: List[DocumentChunk],
        background_chunks: List[DocumentChunk],
        max_tokens: int
    ) -> Dict[str, List[DocumentChunk]]:
        """Fit chunks within token limit while prioritizing important chunks"""
        
        result = {"primary": [], "supporting": [], "background": []}
        current_tokens = 0
        
        # Reserve tokens for formatting (approximately 200 tokens)
        available_tokens = max_tokens - 200
        
        # First, add primary chunks (highest priority)
        for chunk in primary_chunks:
            chunk_tokens = estimate_tokens(chunk.content)
            if current_tokens + chunk_tokens <= available_tokens:
                result["primary"].append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        # Then add supporting chunks
        for chunk in supporting_chunks:
            chunk_tokens = estimate_tokens(chunk.content)
            if current_tokens + chunk_tokens <= available_tokens:
                result["supporting"].append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        # Finally, add background chunks if space remains
        for chunk in background_chunks:
            chunk_tokens = estimate_tokens(chunk.content)
            if current_tokens + chunk_tokens <= available_tokens:
                result["background"].append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        return result
    
    def _calculate_token_usage(self, context_window: ContextWindow) -> int:
        """Calculate actual token usage of context window"""
        total_tokens = 0
        
        # Add tokens for content
        for chunk in context_window.get_all_chunks():
            total_tokens += estimate_tokens(chunk.content)
        
        # Add tokens for formatting (section headers, etc.)
        total_tokens += 200  # Approximate formatting overhead
        
        return total_tokens
    
    def _calculate_quality_metrics(
        self,
        context_window: ContextWindow,
        question_analysis: QuestionAnalysis
    ):
        """Calculate quality metrics for the context window"""
        
        all_chunks = context_window.get_all_chunks()
        
        if not all_chunks:
            context_window.coherence_score = 0.0
            context_window.relevance_score = 0.0
            context_window.completeness_score = 0.0
            context_window.diversity_score = 0.0
            return
        
        # Relevance score (average of chunk relevance scores)
        context_window.relevance_score = sum(c.relevance_score for c in all_chunks) / len(all_chunks)
        
        # Coherence score (based on content similarity between chunks)
        coherence_scores = []
        for i, chunk1 in enumerate(all_chunks):
            for chunk2 in all_chunks[i+1:]:
                similarity = calculate_text_similarity(chunk1.content, chunk2.content)
                coherence_scores.append(similarity)
        
        context_window.coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        # Diversity score (inverse of coherence - we want some diversity)
        context_window.diversity_score = 1.0 - context_window.coherence_score
        
        # Completeness score (based on entity coverage)
        if question_analysis.entities:
            entity_texts = [e.text.lower() for e in question_analysis.entities]
            all_content = " ".join(chunk.content.lower() for chunk in all_chunks)
            
            covered_entities = sum(1 for entity_text in entity_texts if entity_text in all_content)
            context_window.completeness_score = covered_entities / len(question_analysis.entities)
        else:
            # If no entities, base completeness on keyword coverage
            if question_analysis.keywords:
                all_content = " ".join(chunk.content.lower() for chunk in all_chunks)
                covered_keywords = sum(1 for keyword in question_analysis.keywords if keyword.lower() in all_content)
                context_window.completeness_score = covered_keywords / len(question_analysis.keywords)
            else:
                context_window.completeness_score = 0.8  # Default moderate score
    
    def optimize_context_for_question_type(
        self,
        context_window: ContextWindow,
        question_analysis: QuestionAnalysis
    ) -> ContextWindow:
        """Optimize context window for specific question types"""
        
        try:
            from ..models.question_models import QuestionType
        except ImportError:
            from provenance_qa.models.question_models import QuestionType
        
        # Create optimized copy
        optimized = ContextWindow(
            window_id=f"{context_window.window_id}_opt",
            question_id=context_window.question_id,
            strategy=context_window.strategy,
            max_tokens=context_window.max_tokens
        )
        
        all_chunks = context_window.get_all_chunks()
        
        if question_analysis.question_type == QuestionType.COMPARATIVE:
            # For comparative questions, ensure we have chunks from multiple sources
            document_groups = {}
            for chunk in all_chunks:
                doc_id = chunk.document_id
                if doc_id not in document_groups:
                    document_groups[doc_id] = []
                document_groups[doc_id].append(chunk)
            
            # Select best chunks from each document
            for doc_chunks in document_groups.values():
                best_chunk = max(doc_chunks, key=lambda c: c.relevance_score)
                optimized.primary_chunks.append(best_chunk)
        
        elif question_analysis.question_type == QuestionType.QUANTITATIVE:
            # For quantitative questions, prioritize chunks with numbers/amounts
            numeric_chunks = []
            other_chunks = []
            
            for chunk in all_chunks:
                if any(entity.entity_type in ["money", "percentage"] for entity in question_analysis.entities):
                    numeric_chunks.append(chunk)
                else:
                    other_chunks.append(chunk)
            
            optimized.primary_chunks = numeric_chunks[:5]  # Top 5 numeric chunks
            optimized.supporting_chunks = other_chunks[:5]
        
        else:
            # Default optimization - use existing categorization
            optimized.primary_chunks = context_window.primary_chunks
            optimized.supporting_chunks = context_window.supporting_chunks
            optimized.background_chunks = context_window.background_chunks
        
        # Recalculate metrics
        optimized.actual_tokens = self._calculate_token_usage(optimized)
        self._calculate_quality_metrics(optimized, question_analysis)
        
        return optimized