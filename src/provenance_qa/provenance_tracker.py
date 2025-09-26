"""
Provenance Tracking Component

Tracks the complete provenance chain from source documents through processing
to final answers, ensuring full traceability and citation generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    from ..models.provenance_models import (
        Citation, CitationType, SourceType, CertaintyLevel,
        ProvenanceChain, QAResponse
    )
    from ..models.context_models import ContextWindow, DocumentChunk
    from ..models.answer_models import Answer
    from ..models.question_models import QuestionAnalysis
    from ..utils.common import generate_id, calculate_confidence_score, Timer
except ImportError:
    # Fallback for direct execution
    from provenance_qa.models.provenance_models import (
        Citation, CitationType, SourceType, CertaintyLevel,
        ProvenanceChain, QAResponse
    )
    from provenance_qa.models.context_models import ContextWindow, DocumentChunk
    from provenance_qa.models.answer_models import Answer
    from provenance_qa.models.question_models import QuestionAnalysis
    from provenance_qa.utils.common import generate_id, calculate_confidence_score, Timer

logger = logging.getLogger(__name__)

class ProvenanceTracker:
    """Comprehensive provenance tracking for QA responses"""
    
    def __init__(self):
        self.citation_extractors = self._build_citation_extractors()
        self.source_reliability_weights = self._build_source_weights()
    
    def _build_citation_extractors(self) -> Dict[CitationType, callable]:
        """Build citation extraction methods for different citation types"""
        return {
            CitationType.DIRECT_QUOTE: self._extract_direct_quotes,
            CitationType.PARAPHRASE: self._extract_paraphrases,
            CitationType.SUMMARY: self._extract_summaries,
            CitationType.INFERENCE: self._extract_inferences,
            CitationType.REFERENCE: self._extract_references
        }
    
    def _build_source_weights(self) -> Dict[SourceType, float]:
        """Build reliability weights for different source types"""
        return {
            SourceType.DOCUMENT: 1.0,
            SourceType.SECTION: 0.95,
            SourceType.CLAUSE: 0.9,
            SourceType.TABLE: 0.85,
            SourceType.METADATA: 0.6,
            SourceType.ANNOTATION: 0.7,
            SourceType.KNOWLEDGE_GRAPH: 0.8
        }
    
    def create_qa_response(
        self,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        answer: Answer
    ) -> QAResponse:
        """Create comprehensive QA response with full provenance tracking"""
        
        with Timer() as timer:
            # Generate response ID
            response_id = generate_id("resp")
            
            # Create QA response
            qa_response = QAResponse(
                response_id=response_id,
                question_id=question_analysis.original_question[:50],  # Use question as ID
                answer=answer.text,
                model_used=answer.model_name
            )
            
            # Generate citations from context and answer
            citations = self._generate_citations(context_window, answer)
            for citation in citations:
                qa_response.add_citation(citation)
            
            # Create provenance chain
            provenance_chain = self._create_provenance_chain(
                question_analysis, context_window, answer, citations
            )
            qa_response.add_provenance_chain(provenance_chain)
            
            # Extract key findings from answer
            qa_response.key_findings = answer.key_points.copy()
            
            # Create summary
            qa_response.summary = self._create_response_summary(answer, citations)
            
            # Calculate confidence and quality metrics
            self._calculate_response_metrics(qa_response, answer, context_window)
            
            # Add retrieval statistics
            qa_response.retrieval_stats = self._gather_retrieval_stats(context_window)
            
            # Record processing time
            qa_response.processing_time = timer.elapsed()
            
            logger.info(f"Created QA response with {len(citations)} citations and full provenance")
            
            return qa_response
    
    def _generate_citations(
        self,
        context_window: ContextWindow,
        answer: Answer
    ) -> List[Citation]:
        """Generate citations from context chunks and answer"""
        citations = []
        
        all_chunks = context_window.get_all_chunks()
        
        for chunk in all_chunks:
            # Create citation for each chunk
            citation = self._create_citation_from_chunk(chunk, answer)
            if citation:
                citations.append(citation)
        
        # Sort citations by relevance
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        
        return citations
    
    def _create_citation_from_chunk(
        self,
        chunk: DocumentChunk,
        answer: Answer
    ) -> Optional[Citation]:
        """Create citation from document chunk"""
        
        # Generate citation ID
        citation_id = generate_id("cite")
        
        # Determine source type
        source_type = self._determine_source_type(chunk)
        
        # Determine citation type based on how chunk relates to answer
        citation_type = self._determine_citation_type(chunk, answer)
        
        # Calculate relevance and quality scores
        relevance_score = self._calculate_citation_relevance(chunk, answer)
        quality_score = self._calculate_citation_quality(chunk)
        
        # Determine certainty level
        certainty = self._determine_citation_certainty(chunk, citation_type)
        
        # Create citation
        citation = Citation(
            citation_id=citation_id,
            source_id=chunk.document_id,
            source_type=source_type,
            source_title=chunk.document_title,
            cited_text=chunk.content[:500],  # Limit text length
            citation_type=citation_type,
            context_before=chunk.preceding_context[:100] if chunk.preceding_context else "",
            context_after=chunk.following_context[:100] if chunk.following_context else "",
            page_number=chunk.page_number,
            paragraph_number=chunk.paragraph_number,
            section_title=chunk.section_title,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            relevance_score=relevance_score,
            quality_score=quality_score,
            certainty=certainty,
            supports_claim="",  # Will be filled by analysis
            extraction_method=f"{chunk.retrieval_source.value}_retrieval"
        )
        
        return citation
    
    def _determine_source_type(self, chunk: DocumentChunk) -> SourceType:
        """Determine the source type from chunk information"""
        
        if chunk.chunk_type.value == "table":
            return SourceType.TABLE
        elif chunk.chunk_type.value == "section":
            return SourceType.SECTION
        elif chunk.chunk_type.value == "metadata":
            return SourceType.METADATA
        elif "clause" in chunk.section_title.lower():
            return SourceType.CLAUSE
        else:
            return SourceType.DOCUMENT
    
    def _determine_citation_type(self, chunk: DocumentChunk, answer: Answer) -> CitationType:
        """Determine citation type based on relationship to answer"""
        
        # Simple heuristic based on text similarity and content
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        
        similarity = calculate_text_similarity(chunk.content, answer.text)
        
        if similarity > 0.8:
            return CitationType.DIRECT_QUOTE
        elif similarity > 0.5:
            return CitationType.PARAPHRASE
        elif similarity > 0.3:
            return CitationType.SUMMARY
        elif any(kw in answer.text.lower() for kw in chunk.keyword_matches):
            return CitationType.INFERENCE
        else:
            return CitationType.REFERENCE
    
    def _calculate_citation_relevance(self, chunk: DocumentChunk, answer: Answer) -> float:
        """Calculate how relevant the citation is to the answer"""
        
        relevance_factors = {}
        
        # Base relevance from chunk scoring
        relevance_factors["chunk_relevance"] = chunk.relevance_score
        
        # Text similarity to answer
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        text_similarity = calculate_text_similarity(chunk.content, answer.text)
        relevance_factors["text_similarity"] = text_similarity
        
        # Keyword matches
        keyword_score = min(1.0, len(chunk.keyword_matches) / 5)
        relevance_factors["keywords"] = keyword_score
        
        # Source priority based on retrieval method
        source_priority = {
            "cross_encoder": 1.0,
            "hybrid": 0.9,
            "vector_search": 0.8,
            "graph_traversal": 0.7,
            "keyword_match": 0.6
        }
        
        source_score = source_priority.get(chunk.retrieval_source.value, 0.5)
        relevance_factors["source_priority"] = source_score
        
        # Calculate weighted relevance
        weights = {
            "chunk_relevance": 0.4,
            "text_similarity": 0.3,
            "keywords": 0.2,
            "source_priority": 0.1
        }
        
        return calculate_confidence_score(relevance_factors, weights)
    
    def _calculate_citation_quality(self, chunk: DocumentChunk) -> float:
        """Calculate overall quality of the citation"""
        
        quality_factors = {}
        
        # Content length (prefer substantial content)
        content_length = len(chunk.content)
        if 100 <= content_length <= 800:
            quality_factors["content_length"] = 1.0
        elif content_length < 100:
            quality_factors["content_length"] = content_length / 100
        else:
            quality_factors["content_length"] = 800 / content_length
        
        # Location information completeness
        location_score = 0.0
        if chunk.section_title:
            location_score += 0.3
        if chunk.page_number:
            location_score += 0.2
        if chunk.paragraph_number:
            location_score += 0.2
        if chunk.start_char and chunk.end_char:
            location_score += 0.3
        
        quality_factors["location_info"] = location_score
        
        # Context availability
        context_score = 0.0
        if chunk.preceding_context:
            context_score += 0.5
        if chunk.following_context:
            context_score += 0.5
        
        quality_factors["context"] = context_score
        
        # Document metadata
        metadata_score = 0.5  # Base score
        if chunk.document_title:
            metadata_score += 0.3
        if chunk.metadata:
            metadata_score += 0.2
        
        quality_factors["metadata"] = min(1.0, metadata_score)
        
        # Calculate weighted quality
        weights = {
            "content_length": 0.3,
            "location_info": 0.3,
            "context": 0.2,
            "metadata": 0.2
        }
        
        return calculate_confidence_score(quality_factors, weights)
    
    def _determine_citation_certainty(self, chunk: DocumentChunk, citation_type: CitationType) -> CertaintyLevel:
        """Determine certainty level for citation"""
        
        # Base certainty on citation type
        type_certainty = {
            CitationType.DIRECT_QUOTE: CertaintyLevel.CERTAIN,
            CitationType.PARAPHRASE: CertaintyLevel.HIGH,
            CitationType.SUMMARY: CertaintyLevel.MEDIUM,
            CitationType.INFERENCE: CertaintyLevel.LOW,
            CitationType.REFERENCE: CertaintyLevel.MEDIUM
        }
        
        base_certainty = type_certainty.get(citation_type, CertaintyLevel.MEDIUM)
        
        # Adjust based on source reliability
        if chunk.relevance_score > 0.8:
            return CertaintyLevel.CERTAIN
        elif chunk.relevance_score > 0.6:
            return CertaintyLevel.HIGH
        elif chunk.relevance_score > 0.4:
            return CertaintyLevel.MEDIUM
        elif chunk.relevance_score > 0.2:
            return CertaintyLevel.LOW
        else:
            return CertaintyLevel.SPECULATIVE
    
    def _create_provenance_chain(
        self,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        answer: Answer,
        citations: List[Citation]
    ) -> ProvenanceChain:
        """Create comprehensive provenance chain"""
        
        # Generate chain ID
        chain_id = generate_id("chain")
        
        # Create provenance chain
        chain = ProvenanceChain(
            chain_id=chain_id,
            original_sources=citations.copy(),
            final_output=answer.text,
            derivation_method=f"{answer.source.value}_generation"
        )
        
        # Add processing steps
        chain.add_step(
            "Question Analysis",
            question_analysis.original_question,
            f"Type: {question_analysis.question_type.value}, Intent: {question_analysis.intent.value}",
            "nlp_analysis"
        )
        
        chain.add_step(
            "Context Assembly",
            f"{len(context_window.get_all_chunks())} chunks retrieved",
            f"Strategy: {context_window.strategy.value}, Tokens: {context_window.actual_tokens}",
            "context_assembly"
        )
        
        chain.add_step(
            "Answer Generation",
            f"Model: {answer.model_name}",
            f"Type: {answer.answer_type.value}, Confidence: {answer.confidence:.2f}",
            "llm_generation"
        )
        
        # Calculate chain metrics
        chain.confidence = answer.confidence
        chain.complexity_score = self._calculate_chain_complexity(chain)
        chain.source_reliability = sum(c.quality_score for c in citations) / len(citations) if citations else 0.0
        chain.logical_consistency = self._assess_logical_consistency(chain, answer)
        chain.completeness = self._assess_chain_completeness(chain, question_analysis)
        
        return chain
    
    def _calculate_chain_complexity(self, chain: ProvenanceChain) -> float:
        """Calculate complexity score for provenance chain"""
        
        complexity_factors = {}
        
        # Number of sources
        source_complexity = min(1.0, len(chain.original_sources) / 10)
        complexity_factors["sources"] = source_complexity
        
        # Number of processing steps
        step_complexity = min(1.0, len(chain.intermediate_steps) / 5)
        complexity_factors["steps"] = step_complexity
        
        # Diversity of source types
        source_types = set(c.source_type for c in chain.original_sources)
        type_diversity = min(1.0, len(source_types) / 4)
        complexity_factors["diversity"] = type_diversity
        
        return calculate_confidence_score(complexity_factors)
    
    def _assess_logical_consistency(self, chain: ProvenanceChain, answer: Answer) -> float:
        """Assess logical consistency of the provenance chain"""
        
        consistency_factors = {}
        
        # Answer quality as proxy for consistency
        consistency_factors["answer_quality"] = answer.get_overall_quality_score()
        
        # Citation relevance consistency
        if chain.original_sources:
            avg_relevance = sum(c.relevance_score for c in chain.original_sources) / len(chain.original_sources)
            consistency_factors["citation_relevance"] = avg_relevance
        else:
            consistency_factors["citation_relevance"] = 0.0
        
        # Processing step coherence (simplified)
        step_coherence = 1.0 if len(chain.intermediate_steps) >= 3 else 0.7
        consistency_factors["step_coherence"] = step_coherence
        
        return calculate_confidence_score(consistency_factors)
    
    def _assess_chain_completeness(
        self,
        chain: ProvenanceChain,
        question_analysis: QuestionAnalysis
    ) -> float:
        """Assess completeness of the provenance chain"""
        
        completeness_factors = {}
        
        # Source coverage
        has_sources = len(chain.original_sources) > 0
        completeness_factors["has_sources"] = 1.0 if has_sources else 0.0
        
        # Processing step coverage
        expected_steps = ["Question Analysis", "Context Assembly", "Answer Generation"]
        actual_step_descriptions = [step["description"] for step in chain.intermediate_steps]
        
        covered_steps = sum(1 for expected in expected_steps 
                           if any(expected in actual for actual in actual_step_descriptions))
        
        completeness_factors["processing_steps"] = covered_steps / len(expected_steps)
        
        # Final output presence
        completeness_factors["final_output"] = 1.0 if chain.final_output else 0.0
        
        return calculate_confidence_score(completeness_factors)
    
    def _create_response_summary(self, answer: Answer, citations: List[Citation]) -> str:
        """Create concise summary of the response"""
        
        summary_parts = []
        
        # Answer summary
        if len(answer.text) > 200:
            summary_parts.append(f"Answer: {answer.text[:200]}...")
        else:
            summary_parts.append(f"Answer: {answer.text}")
        
        # Citation summary
        if citations:
            doc_count = len(set(c.source_id for c in citations))
            summary_parts.append(f"Based on {len(citations)} citations from {doc_count} document(s)")
        
        # Quality indicator
        if answer.confidence >= 0.8:
            summary_parts.append("High confidence response")
        elif answer.confidence >= 0.6:
            summary_parts.append("Moderate confidence response")
        else:
            summary_parts.append("Low confidence response")
        
        return " | ".join(summary_parts)
    
    def _calculate_response_metrics(
        self,
        qa_response: QAResponse,
        answer: Answer,
        context_window: ContextWindow
    ):
        """Calculate overall response quality metrics"""
        
        # Overall confidence from answer
        qa_response.overall_confidence = answer.confidence
        
        # Answer quality from answer metrics
        qa_response.answer_quality = answer.get_overall_quality_score()
        
        # Source reliability from citations
        if qa_response.citations:
            qa_response.source_reliability = sum(c.quality_score for c in qa_response.citations) / len(qa_response.citations)
        else:
            qa_response.source_reliability = 0.0
    
    def _gather_retrieval_stats(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Gather statistics about the retrieval process"""
        
        all_chunks = context_window.get_all_chunks()
        
        if not all_chunks:
            return {}
        
        # Count chunks by source
        source_counts = {}
        for chunk in all_chunks:
            source = chunk.retrieval_source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count chunks by type
        type_counts = {}
        for chunk in all_chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # Calculate average scores
        avg_relevance = sum(c.relevance_score for c in all_chunks) / len(all_chunks)
        avg_similarity = sum(c.similarity_score for c in all_chunks) / len(all_chunks)
        
        return {
            "total_chunks": len(all_chunks),
            "unique_documents": len(context_window.get_unique_documents()),
            "source_distribution": source_counts,
            "type_distribution": type_counts,
            "average_relevance": avg_relevance,
            "average_similarity": avg_similarity,
            "token_usage": context_window.actual_tokens,
            "context_strategy": context_window.strategy.value
        }
    
    def extract_direct_quotes(self, answer_text: str, source_text: str) -> List[str]:
        """Extract potential direct quotes from answer that match source"""
        quotes = []
        
        # Simple approach: find common phrases of 5+ words
        import re
        
        # Split into sentences and phrases
        answer_sentences = re.split(r'[.!?]+', answer_text)
        source_sentences = re.split(r'[.!?]+', source_text)
        
        for answer_sent in answer_sentences:
            answer_sent = answer_sent.strip()
            if len(answer_sent.split()) < 5:  # Skip short phrases
                continue
                
            for source_sent in source_sentences:
                source_sent = source_sent.strip()
                
                # Check for substantial overlap
                from ..utils.common import calculate_text_similarity
                similarity = calculate_text_similarity(answer_sent, source_sent)
                
                if similarity > 0.7:  # High similarity threshold for quotes
                    quotes.append(answer_sent)
                    break
        
        return quotes
    
    def _extract_direct_quotes(self, answer_text: str, citations: List[Citation]) -> List[str]:
        """Extract direct quotes from answer"""
        return []  # Placeholder implementation
    
    def _extract_paraphrases(self, answer_text: str, citations: List[Citation]) -> List[str]:
        """Extract paraphrased content from answer"""
        return []  # Placeholder implementation
    
    def _extract_summaries(self, answer_text: str, citations: List[Citation]) -> List[str]:
        """Extract summarized content from answer"""
        return []  # Placeholder implementation
    
    def _extract_inferences(self, answer_text: str, citations: List[Citation]) -> List[str]:
        """Extract inferred content from answer"""
        return []  # Placeholder implementation
    
    def _extract_references(self, answer_text: str, citations: List[Citation]) -> List[str]:
        """Extract general references from answer"""
        return []  # Placeholder implementation