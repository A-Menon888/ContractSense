"""
Hybrid Retrieval System

This module implements sophisticated hybrid retrieval combining knowledge graph
traversal with vector similarity search for enhanced contract analysis.

Key Features:
- Graph-guided vector search using knowledge graph relationships
- Multi-modal retrieval strategies (keyword, semantic, structural)
- Intelligent result fusion and ranking
- Query expansion using graph ontology
- Context-aware relevance scoring
- Integration with Module 5 knowledge graph
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# Import vector search components
from .vector_store import VectorStore, SearchQuery, SearchResult
from .embedding_generator import EmbeddingGenerator

# Import knowledge graph components
try:
    from ..knowledge_graph.integration import KnowledgeGraphIntegrator
    from ..knowledge_graph.schema import NodeType, RelationshipType
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KnowledgeGraphIntegrator = None
    NodeType = RelationshipType = None
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logging.warning("Knowledge graph not available - limited hybrid capabilities")

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Different retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    GRAPH_GUIDED_VECTOR = "graph_guided_vector"
    VECTOR_EXPANDED_GRAPH = "vector_expanded_graph"
    ADAPTIVE = "adaptive"

@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval"""
    # Strategy settings
    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_PARALLEL
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    
    # Search parameters
    vector_top_k: int = 50
    graph_max_depth: int = 2
    final_top_k: int = 10
    
    # Quality thresholds
    min_vector_score: float = 0.1
    min_graph_relevance: float = 0.2
    diversity_threshold: float = 0.8
    
    # Query expansion
    enable_query_expansion: bool = True
    expansion_terms: int = 5
    expansion_weight: float = 0.3
    
    # Result fusion
    fusion_method: str = "rrf"  # rrf (reciprocal rank fusion), weighted, learned
    rrf_k: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    parallel_execution: bool = True

@dataclass
class RetrievalResult:
    """Enhanced search result with hybrid information"""
    chunk_id: str
    content: str
    score: float
    
    # Source information
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    fusion_score: Optional[float] = None
    
    # Metadata and context
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    
    # Related entities and relationships
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    relationship_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "fusion_score": self.fusion_score,
            "metadata": self.metadata,
            "graph_context": self.graph_context,
            "explanation": self.explanation,
            "related_entities": self.related_entities,
            "relationship_path": self.relationship_path
        }

@dataclass
class HybridQuery:
    """Enhanced query with hybrid parameters"""
    text: str
    
    # Query types and hints
    query_type: str = "general"  # general, entity, clause, risk, compliance
    entity_hints: List[str] = field(default_factory=list)
    relationship_hints: List[str] = field(default_factory=list)
    document_filters: List[str] = field(default_factory=list)
    
    # Search parameters
    strategy: Optional[RetrievalStrategy] = None
    top_k: int = 10
    include_context: bool = True
    include_explanations: bool = True
    
    # Filtering
    metadata_filter: Dict[str, Any] = field(default_factory=dict)
    date_range: Optional[Tuple[datetime, datetime]] = None
    risk_levels: List[str] = field(default_factory=list)

class HybridRetriever:
    """Main hybrid retrieval engine"""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_generator: EmbeddingGenerator,
                 config: HybridConfig = None):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or HybridConfig()
        
        # Initialize knowledge graph integrator
        self.graph_integrator = None
        if KNOWLEDGE_GRAPH_AVAILABLE:
            try:
                self.graph_integrator = KnowledgeGraphIntegrator()
                logger.info("Knowledge graph integrator initialized")
            except Exception as e:
                logger.warning(f"Could not initialize graph integrator: {e}")
        
        # Query cache
        self._query_cache = {}
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "vector_queries": 0,
            "graph_queries": 0,
            "hybrid_queries": 0,
            "average_response_time": 0.0
        }
        
        logger.info(f"Hybrid retriever initialized with strategy: {self.config.default_strategy}")
    
    def retrieve(self, query: HybridQuery) -> List[RetrievalResult]:
        """Main retrieval method"""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(query)
            if self.config.enable_caching and cache_key in self._query_cache:
                cached_result, cache_time = self._query_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Determine strategy
            strategy = query.strategy or self.config.default_strategy
            
            # Execute retrieval based on strategy
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                results = self._vector_retrieval(query)
            elif strategy == RetrievalStrategy.GRAPH_ONLY:
                results = self._graph_retrieval(query)
            elif strategy == RetrievalStrategy.HYBRID_PARALLEL:
                results = self._hybrid_parallel_retrieval(query)
            elif strategy == RetrievalStrategy.GRAPH_GUIDED_VECTOR:
                results = self._graph_guided_vector_retrieval(query)
            elif strategy == RetrievalStrategy.VECTOR_EXPANDED_GRAPH:
                results = self._vector_expanded_graph_retrieval(query)
            elif strategy == RetrievalStrategy.ADAPTIVE:
                results = self._adaptive_retrieval(query)
            else:
                results = self._hybrid_parallel_retrieval(query)  # Default fallback
            
            # Post-process results
            results = self._post_process_results(results, query)
            
            # Cache results
            if self.config.enable_caching:
                self._query_cache[cache_key] = (results, datetime.now())
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_stats(response_time)
            
            logger.info(f"Retrieved {len(results)} results in {response_time:.3f}s using {strategy}")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _vector_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Pure vector similarity search"""
        self.stats["vector_queries"] += 1
        
        try:
            # Generate query embedding
            query_chunks = self.embedding_generator.chunk_document("query", query.text)
            if not query_chunks:
                return []
            
            query_embeddings = self.embedding_generator.generate_embeddings(query_chunks[:1])
            if not query_embeddings:
                return []
            
            # Create search query
            search_query = SearchQuery(
                query_text=query.text,
                query_embedding=query_embeddings[0].embedding,
                top_k=query.top_k,
                score_threshold=self.config.min_vector_score,
                metadata_filter=query.metadata_filter
            )
            
            # Execute vector search
            vector_results = self.vector_store.search(search_query)
            
            # Convert to RetrievalResult
            results = []
            for result in vector_results:
                hybrid_result = RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=result.score,
                    vector_score=result.score,
                    metadata=result.metadata,
                    explanation="Vector similarity match"
                )
                results.append(hybrid_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []
    
    def _graph_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Pure knowledge graph traversal"""
        self.stats["graph_queries"] += 1
        
        if not self.graph_integrator:
            logger.warning("Graph integrator not available")
            return []
        
        try:
            # Use graph integrator to find relevant content
            graph_query = {
                "query_text": query.text,
                "entity_hints": query.entity_hints,
                "relationship_hints": query.relationship_hints,
                "max_depth": self.config.graph_max_depth,
                "limit": query.top_k
            }
            
            graph_response = self.graph_integrator.query_graph_for_insights(
                "hybrid_search", **graph_query
            )
            
            if "error" in graph_response:
                logger.warning(f"Graph query failed: {graph_response['error']}")
                return []
            
            # Convert graph results to RetrievalResult
            results = []
            for item in graph_response.get("results", []):
                hybrid_result = RetrievalResult(
                    chunk_id=item.get("chunk_id", f"graph_{len(results)}"),
                    content=item.get("content", ""),
                    score=item.get("relevance_score", 0.5),
                    graph_score=item.get("relevance_score", 0.5),
                    metadata=item.get("metadata", {}),
                    graph_context=item.get("graph_context", {}),
                    related_entities=item.get("entities", []),
                    relationship_path=item.get("path", []),
                    explanation=item.get("explanation", "Graph traversal match")
                )
                results.append(hybrid_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []
    
    def _hybrid_parallel_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Parallel execution of vector and graph retrieval with result fusion"""
        self.stats["hybrid_queries"] += 1
        
        try:
            # Execute both retrievals
            vector_results = self._vector_retrieval(query)
            graph_results = self._graph_retrieval(query)
            
            # Combine and fuse results
            fused_results = self._fuse_results(vector_results, graph_results, query)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Hybrid parallel retrieval failed: {e}")
            return []
    
    def _graph_guided_vector_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Use graph to guide vector search with query expansion"""
        self.stats["hybrid_queries"] += 1
        
        try:
            # First, use graph to find relevant entities and expand query
            expanded_query = self._expand_query_with_graph(query)
            
            # Then perform vector search with expanded query
            vector_results = self._vector_retrieval(expanded_query)
            
            # Enrich results with graph context
            enriched_results = self._enrich_with_graph_context(vector_results)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Graph-guided vector retrieval failed: {e}")
            return self._vector_retrieval(query)  # Fallback to pure vector
    
    def _vector_expanded_graph_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Use vector search to find relevant content, then expand with graph"""
        self.stats["hybrid_queries"] += 1
        
        try:
            # First, perform vector search to find relevant chunks
            vector_results = self._vector_retrieval(query)
            
            if not vector_results:
                return []
            
            # Extract entities from top results
            top_chunks = vector_results[:5]  # Use top 5 for expansion
            graph_entities = []
            
            for result in top_chunks:
                # Find entities in the content using graph
                entities = self._extract_entities_from_content(result.content)
                graph_entities.extend(entities)
            
            # Use entities to expand search in graph
            if graph_entities:
                expanded_query = HybridQuery(
                    text=query.text,
                    entity_hints=list(set(graph_entities)),
                    **{k: v for k, v in query.__dict__.items() if k not in ['text', 'entity_hints']}
                )
                graph_results = self._graph_retrieval(expanded_query)
                
                # Combine results
                combined_results = self._fuse_results(vector_results, graph_results, query)
                return combined_results
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector-expanded graph retrieval failed: {e}")
            return self._vector_retrieval(query)  # Fallback
    
    def _adaptive_retrieval(self, query: HybridQuery) -> List[RetrievalResult]:
        """Adaptive strategy selection based on query characteristics"""
        try:
            # Analyze query to determine best strategy
            strategy = self._analyze_query_for_strategy(query)
            
            # Update query strategy and recurse
            query.strategy = strategy
            return self.retrieve(query)
            
        except Exception as e:
            logger.error(f"Adaptive retrieval failed: {e}")
            return self._hybrid_parallel_retrieval(query)  # Safe fallback
    
    def _analyze_query_for_strategy(self, query: HybridQuery) -> RetrievalStrategy:
        """Analyze query to determine optimal retrieval strategy"""
        query_text = query.text.lower()
        
        # Entity-heavy queries benefit from graph
        entity_indicators = ["who", "what", "which company", "party", "entity", "organization"]
        if any(indicator in query_text for indicator in entity_indicators):
            return RetrievalStrategy.GRAPH_GUIDED_VECTOR
        
        # Relationship queries benefit from graph
        relationship_indicators = ["relationship", "connected", "related", "between", "linked"]
        if any(indicator in query_text for indicator in relationship_indicators):
            return RetrievalStrategy.GRAPH_ONLY
        
        # Semantic similarity queries benefit from vector
        semantic_indicators = ["similar", "like", "comparable", "equivalent", "meaning"]
        if any(indicator in query_text for indicator in semantic_indicators):
            return RetrievalStrategy.VECTOR_ONLY
        
        # Complex queries benefit from hybrid
        if len(query.text.split()) > 10 or query.entity_hints or query.relationship_hints:
            return RetrievalStrategy.HYBRID_PARALLEL
        
        # Default to hybrid
        return RetrievalStrategy.HYBRID_PARALLEL
    
    def _expand_query_with_graph(self, query: HybridQuery) -> HybridQuery:
        """Expand query using knowledge graph ontology"""
        if not self.graph_integrator or not self.config.enable_query_expansion:
            return query
        
        try:
            # Find related terms using graph
            expansion_response = self.graph_integrator.query_graph_for_insights(
                "query_expansion", 
                query_text=query.text,
                max_terms=self.config.expansion_terms
            )
            
            if "error" not in expansion_response:
                expanded_terms = expansion_response.get("expanded_terms", [])
                expanded_text = query.text
                
                if expanded_terms:
                    expansion_text = " ".join(expanded_terms)
                    expanded_text = f"{query.text} {expansion_text}"
                
                return HybridQuery(
                    text=expanded_text,
                    **{k: v for k, v in query.__dict__.items() if k != 'text'}
                )
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return query
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """Extract entities from content using graph knowledge"""
        if not self.graph_integrator:
            return []
        
        try:
            extraction_response = self.graph_integrator.query_graph_for_insights(
                "entity_extraction",
                text=content
            )
            
            if "error" not in extraction_response:
                return extraction_response.get("entities", [])
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return []
    
    def _enrich_with_graph_context(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Enrich vector results with graph context"""
        if not self.graph_integrator:
            return results
        
        enriched_results = []
        
        for result in results:
            try:
                # Get graph context for this result
                context_response = self.graph_integrator.query_graph_for_insights(
                    "content_context",
                    chunk_id=result.chunk_id,
                    content=result.content
                )
                
                if "error" not in context_response:
                    result.graph_context = context_response.get("context", {})
                    result.related_entities = context_response.get("entities", [])
                    result.relationship_path = context_response.get("relationships", [])
                    
                    # Update explanation
                    if result.explanation:
                        result.explanation += " (enriched with graph context)"
                
                enriched_results.append(result)
                
            except Exception as e:
                logger.warning(f"Graph enrichment failed for {result.chunk_id}: {e}")
                enriched_results.append(result)  # Keep original result
        
        return enriched_results
    
    def _fuse_results(self, vector_results: List[RetrievalResult], 
                     graph_results: List[RetrievalResult], 
                     query: HybridQuery) -> List[RetrievalResult]:
        """Fuse results from vector and graph retrieval"""
        
        if self.config.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(vector_results, graph_results)
        elif self.config.fusion_method == "weighted":
            return self._weighted_fusion(vector_results, graph_results)
        else:
            # Simple concatenation with deduplication
            return self._simple_fusion(vector_results, graph_results)
    
    def _reciprocal_rank_fusion(self, vector_results: List[RetrievalResult], 
                               graph_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF) algorithm"""
        
        # Create mapping from chunk_id to results
        all_results = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
            
            # Calculate RRF score
            rrf_score = 1.0 / (self.config.rrf_k + i + 1)
            all_results[result.chunk_id].vector_score = result.score
            all_results[result.chunk_id].fusion_score = rrf_score * self.config.vector_weight
        
        # Process graph results
        for i, result in enumerate(graph_results):
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
                all_results[result.chunk_id].fusion_score = 0.0
            
            # Add graph RRF score
            rrf_score = 1.0 / (self.config.rrf_k + i + 1)
            all_results[result.chunk_id].graph_score = result.score
            all_results[result.chunk_id].fusion_score += rrf_score * self.config.graph_weight
            
            # Merge graph-specific information
            if hasattr(result, 'graph_context'):
                all_results[result.chunk_id].graph_context.update(result.graph_context)
            if hasattr(result, 'related_entities'):
                all_results[result.chunk_id].related_entities.extend(result.related_entities)
        
        # Sort by fusion score and update final scores
        fused_results = list(all_results.values())
        fused_results.sort(key=lambda x: x.fusion_score or 0, reverse=True)
        
        # Update final scores
        for result in fused_results:
            result.score = result.fusion_score or 0
            result.explanation = f"RRF fusion (v:{result.vector_score:.3f}, g:{result.graph_score:.3f})"
        
        return fused_results
    
    def _weighted_fusion(self, vector_results: List[RetrievalResult], 
                        graph_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Weighted score fusion"""
        
        all_results = {}
        
        # Process vector results
        for result in vector_results:
            all_results[result.chunk_id] = result
            result.fusion_score = result.score * self.config.vector_weight
        
        # Process graph results  
        for result in graph_results:
            if result.chunk_id in all_results:
                all_results[result.chunk_id].graph_score = result.score
                all_results[result.chunk_id].fusion_score += result.score * self.config.graph_weight
            else:
                all_results[result.chunk_id] = result
                result.fusion_score = result.score * self.config.graph_weight
        
        # Sort and update scores
        fused_results = list(all_results.values())
        fused_results.sort(key=lambda x: x.fusion_score, reverse=True)
        
        for result in fused_results:
            result.score = result.fusion_score
            result.explanation = f"Weighted fusion (α={self.config.vector_weight})"
        
        return fused_results
    
    def _simple_fusion(self, vector_results: List[RetrievalResult], 
                      graph_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Simple concatenation with deduplication"""
        
        seen_ids = set()
        fused_results = []
        
        # Add vector results first
        for result in vector_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                result.explanation = "Vector similarity"
                fused_results.append(result)
        
        # Add graph results that aren't duplicates
        for result in graph_results:
            if result.chunk_id not in seen_ids:
                seen_ids.add(result.chunk_id)
                result.explanation = "Graph traversal"
                fused_results.append(result)
        
        return fused_results
    
    def _post_process_results(self, results: List[RetrievalResult], 
                             query: HybridQuery) -> List[RetrievalResult]:
        """Post-process results for quality and diversity"""
        
        if not results:
            return results
        
        # Apply diversity filtering
        if self.config.diversity_threshold < 1.0:
            results = self._apply_diversity_filtering(results)
        
        # Limit to requested number
        results = results[:query.top_k]
        
        # Add explanations if requested
        if query.include_explanations:
            self._add_explanations(results, query)
        
        return results
    
    def _apply_diversity_filtering(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply diversity filtering to reduce redundant results"""
        
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include top result
        
        for candidate in results[1:]:
            is_diverse = True
            
            for existing in diverse_results:
                # Simple content similarity check
                similarity = self._compute_content_similarity(candidate.content, existing.content)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(candidate)
        
        return diverse_results
    
    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        """Compute simple content similarity"""
        # Simple Jaccard similarity on words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _add_explanations(self, results: List[RetrievalResult], query: HybridQuery):
        """Add detailed explanations to results"""
        
        for result in results:
            explanation_parts = []
            
            if result.vector_score:
                explanation_parts.append(f"Vector similarity: {result.vector_score:.3f}")
            
            if result.graph_score:
                explanation_parts.append(f"Graph relevance: {result.graph_score:.3f}")
            
            if result.related_entities:
                entity_names = [e.get("name", "unknown") for e in result.related_entities[:3]]
                explanation_parts.append(f"Related entities: {', '.join(entity_names)}")
            
            if explanation_parts:
                result.explanation = "; ".join(explanation_parts)
            elif not result.explanation:
                result.explanation = "Hybrid retrieval match"
    
    def _get_cache_key(self, query: HybridQuery) -> str:
        """Generate cache key for query"""
        import hashlib
        query_str = f"{query.text}:{query.query_type}:{query.top_k}:{str(query.metadata_filter)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _update_stats(self, response_time: float):
        """Update retrieval statistics"""
        self.stats["total_queries"] += 1
        
        total_time = self.stats["average_response_time"] * (self.stats["total_queries"] - 1)
        total_time += response_time
        self.stats["average_response_time"] = total_time / self.stats["total_queries"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "retrieval_stats": self.stats,
            "config": {
                "default_strategy": self.config.default_strategy.value,
                "vector_weight": self.config.vector_weight,
                "graph_weight": self.config.graph_weight,
                "fusion_method": self.config.fusion_method
            },
            "cache_size": len(self._query_cache) if self.config.enable_caching else 0
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")

# Factory function
def create_hybrid_retriever(vector_store: VectorStore,
                           embedding_generator: EmbeddingGenerator,
                           config: HybridConfig = None) -> HybridRetriever:
    """Factory function to create hybrid retriever"""
    
    return HybridRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        config=config or HybridConfig()
    )