"""
Vector Searcher for Hybrid Retrieval Engine

This module implements semantic vector search coordination, interfacing with
the vector search system built in Module 6 to provide semantic flexibility
in legal document retrieval.

Key Features:
- Semantic similarity search using embeddings
- Query expansion and refinement
- Multi-vector search strategies
- Semantic clustering and filtering

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import logging
import time
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_engine import SearchResult
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum

@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_results: int = 50
    enable_query_expansion: bool = True
    expansion_terms: int = 3
    semantic_clustering: bool = True
    timeout_seconds: int = 15
    enable_caching: bool = True
    cache_ttl_seconds: int = 600

class SearchMode(Enum):
    """Vector search modes"""
    SIMILARITY = "similarity"
    SEMANTIC_EXPANSION = "semantic_expansion" 
    HYBRID_BOOST = "hybrid_boost"
    CLUSTERING = "clustering"

class VectorSearcher:
    """
    Vector searcher that performs semantic search using embeddings
    and coordinates with the vector search system from Module 6
    """
    
    def __init__(self, config: VectorSearchConfig, vector_store_manager):
        self.config = config
        self.vector_manager = vector_store_manager
        self.logger = logging.getLogger(__name__)
        
        # Search strategies
        self._load_search_strategies()
        
        # Legal domain vocabulary for query expansion
        self._load_legal_vocabulary()
        
        # Query cache
        self.query_cache: Dict[str, Tuple[List, float]] = {}
        
        # Statistics
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "embedding_time": 0.0,
            "similarity_computation_time": 0.0,
            "query_expansion_time": 0.0
        }
        
        self.logger.info("Vector searcher initialized")
    
    def _load_search_strategies(self):
        """Load different vector search strategies"""
        
        self.strategies = {
            'direct_similarity': {
                'description': 'Direct cosine similarity search',
                'weight': 1.0,
                'min_similarity': self.config.similarity_threshold
            },
            
            'semantic_expansion': {
                'description': 'Expand query with semantic neighbors',
                'weight': 0.8,
                'min_similarity': self.config.similarity_threshold * 0.8
            },
            
            'contextual_search': {
                'description': 'Search with contextual embeddings',
                'weight': 0.9,
                'min_similarity': self.config.similarity_threshold * 0.9
            },
            
            'entity_boosted': {
                'description': 'Boost results containing query entities',
                'weight': 1.1,
                'min_similarity': self.config.similarity_threshold * 0.7
            },
            
            'clause_type_filtered': {
                'description': 'Filter by clause type embeddings',
                'weight': 1.0,
                'min_similarity': self.config.similarity_threshold
            }
        }
    
    def _load_legal_vocabulary(self):
        """Load legal domain vocabulary for query expansion"""
        
        self.legal_vocabulary = {
            # Contract types
            'contract': ['agreement', 'covenant', 'pact', 'treaty', 'accord'],
            'agreement': ['contract', 'deal', 'arrangement', 'understanding'],
            
            # Legal terms
            'liability': ['responsibility', 'obligation', 'accountability', 'exposure'],
            'indemnify': ['protect', 'compensate', 'reimburse', 'hold harmless'],
            'breach': ['violation', 'default', 'non-compliance', 'infringement'],
            'termination': ['cancellation', 'dissolution', 'expiry', 'conclusion'],
            
            # Monetary terms
            'payment': ['compensation', 'remuneration', 'settlement', 'disbursement'],
            'penalty': ['fine', 'sanction', 'damages', 'forfeit'],
            'fee': ['charge', 'cost', 'expense', 'rate'],
            
            # Temporal terms
            'duration': ['term', 'period', 'timeframe', 'span'],
            'deadline': ['due date', 'expiry', 'cutoff', 'time limit'],
            'notice': ['notification', 'warning', 'announcement', 'advisement'],
            
            # Parties
            'party': ['entity', 'organization', 'individual', 'participant'],
            'contractor': ['vendor', 'supplier', 'provider', 'service provider'],
            'client': ['customer', 'buyer', 'purchaser', 'consumer']
        }
        
        # Flatten to create expansion dictionary
        self.expansion_terms = {}
        for base_term, expansions in self.legal_vocabulary.items():
            self.expansion_terms[base_term] = expansions
            # Add reverse mappings
            for expansion in expansions:
                if expansion not in self.expansion_terms:
                    self.expansion_terms[expansion] = [base_term]
                else:
                    self.expansion_terms[expansion].append(base_term)
    
    async def search(self, query) -> List['SearchResult']:
        """
        Execute vector search based on processed query
        
        Args:
            query: SearchQuery object with extracted entities and metadata
            
        Returns:
            List of SearchResult objects from vector search
        """
        from .hybrid_engine import SearchResult  # Avoid circular import
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._build_cache_key(query)
            if self.config.enable_caching and cache_key in self.query_cache:
                cached_results, cache_time = self.query_cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    self.stats["cache_hits"] += 1
                    self.logger.debug(f"Vector search cache hit for query: {query.text}")
                    return cached_results
            
            self.stats["cache_misses"] += 1
            
            # Generate search variants
            search_variants = await self._generate_search_variants(query)
            
            # Execute multiple search strategies in parallel
            search_tasks = []
            for variant, strategy in search_variants:
                task = self._execute_search_strategy(variant, strategy, query)
                search_tasks.append(task)
            
            # Wait for all searches to complete
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and process results
            all_results = []
            for result in search_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Search strategy failed: {result}")
                    continue
                if isinstance(result, list):
                    all_results.extend(result)
            
            # Apply semantic clustering if enabled
            if self.config.semantic_clustering:
                all_results = await self._apply_semantic_clustering(all_results)
            
            # Remove duplicates and sort by vector score
            unique_results = self._deduplicate_results(all_results)
            
            # Limit results
            final_results = unique_results[:self.config.max_results]
            
            # Cache results
            if self.config.enable_caching:
                self.query_cache[cache_key] = (final_results, time.time())
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats["queries_executed"] += 1
            self.stats["total_execution_time"] += total_time
            
            self.logger.debug(
                f"Vector search completed: {len(final_results)} results "
                f"in {total_time:.3f}s"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def _generate_search_variants(self, query) -> List[Tuple[str, str]]:
        """Generate different search variants and strategies"""
        
        variants = []
        
        # Original query with direct similarity
        variants.append((query.text, 'direct_similarity'))
        
        # Expanded query if enabled
        if self.config.enable_query_expansion:
            expansion_start = time.time()
            expanded_query = await self._expand_query(query.text)
            self.stats["query_expansion_time"] += time.time() - expansion_start
            
            if expanded_query != query.text:
                variants.append((expanded_query, 'semantic_expansion'))
        
        # Entity-focused queries
        if query.entities:
            for entity in query.entities[:2]:  # Limit to avoid too many queries
                entity_query = f"{query.text} {entity}"
                variants.append((entity_query, 'entity_boosted'))
        
        # Clause type focused queries
        if query.clause_types:
            for clause_type in query.clause_types:
                clause_query = f"{query.text} {clause_type} clause"
                variants.append((clause_query, 'clause_type_filtered'))
        
        # Contextual query with legal terms
        contextual_query = self._add_legal_context(query.text)
        if contextual_query != query.text:
            variants.append((contextual_query, 'contextual_search'))
        
        self.logger.debug(f"Generated {len(variants)} search variants")
        return variants
    
    async def _expand_query(self, query_text: str) -> str:
        """Expand query with semantically related terms"""
        
        words = query_text.lower().split()
        expanded_terms = set(words)
        
        # Add legal vocabulary expansions
        for word in words:
            if word in self.expansion_terms:
                expansions = self.expansion_terms[word][:self.config.expansion_terms]
                expanded_terms.update(expansions)
        
        # Create expanded query
        expanded_query = ' '.join(expanded_terms)
        
        self.logger.debug(f"Expanded query: '{query_text}' -> '{expanded_query}'")
        return expanded_query
    
    def _add_legal_context(self, query_text: str) -> str:
        """Add legal context terms to query"""
        
        context_terms = []
        
        # Add general legal context
        if 'contract' not in query_text.lower():
            context_terms.append('contract')
        
        # Add clause context if not present
        if 'clause' not in query_text.lower() and any(term in query_text.lower() 
                                                      for term in ['liability', 'termination', 'payment']):
            context_terms.append('clause')
        
        if context_terms:
            return f"{query_text} {' '.join(context_terms)}"
        
        return query_text
    
    async def _execute_search_strategy(self, search_text: str, strategy: str, original_query) -> List['SearchResult']:
        """Execute a specific vector search strategy"""
        from .hybrid_engine import SearchResult  # Avoid circular import
        
        try:
            strategy_config = self.strategies[strategy]
            
            # Generate embeddings
            embedding_start = time.time()
            query_embedding = await self._generate_embedding(search_text)
            self.stats["embedding_time"] += time.time() - embedding_start
            
            # Search vector store
            similarity_start = time.time()
            vector_results = await self._search_vector_store(
                query_embedding, 
                strategy_config['min_similarity'],
                strategy_config['weight']
            )
            self.stats["similarity_computation_time"] += time.time() - similarity_start
            
            # Convert to SearchResult objects
            search_results = []
            for result in vector_results:
                try:
                    # Calculate vector score with strategy weight
                    vector_score = result['similarity'] * strategy_config['weight']
                    
                    search_result = SearchResult(
                        content=result.get('content', ''),
                        document_id=result.get('document_id', ''),
                        clause_id=result.get('clause_id', ''),
                        final_score=vector_score,
                        graph_score=0.0,  # Will be set during fusion
                        vector_score=vector_score,
                        confidence=vector_score,
                        source_path=[f"VectorSearch:{strategy}"],
                        document_title=result.get('document_title', ''),
                        clause_type=result.get('clause_type', ''),
                        metadata={
                            'similarity': result['similarity'],
                            'strategy': strategy,
                            'embedding_model': self.config.embedding_model,
                            **result.get('metadata', {})
                        },
                        explanation=f"Found via {strategy_config['description']} (similarity: {result['similarity']:.3f})"
                    )
                    
                    search_results.append(search_result)
                    
                except Exception as e:
                    self.logger.error(f"Error converting vector result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search strategy '{strategy}' failed: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using vector store manager"""
        
        try:
            if hasattr(self.vector_manager, 'generate_embedding'):
                return await self.vector_manager.generate_embedding(text)
            elif hasattr(self.vector_manager, 'embed_query'):
                return await self.vector_manager.embed_query(text)
            else:
                # Fallback embedding generation
                self.logger.warning("Vector manager missing embedding method")
                return np.random.rand(384)  # Fallback dimension
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _search_vector_store(self, query_embedding: np.ndarray, 
                                 min_similarity: float, weight: float) -> List[Dict[str, Any]]:
        """Search the vector store with the given embedding"""
        
        try:
            if hasattr(self.vector_manager, 'similarity_search'):
                results = await self.vector_manager.similarity_search(
                    query_embedding=query_embedding,
                    threshold=min_similarity,
                    limit=self.config.max_results
                )
                
                # Apply strategy weight
                for result in results:
                    if 'similarity' in result:
                        result['similarity'] *= weight
                
                return results
                
            elif hasattr(self.vector_manager, 'search'):
                return await self.vector_manager.search(
                    query_embedding, 
                    threshold=min_similarity,
                    top_k=self.config.max_results
                )
            else:
                self.logger.warning("Vector manager missing search method")
                return []
                
        except Exception as e:
            self.logger.error(f"Vector store search failed: {e}")
            return []
    
    async def _apply_semantic_clustering(self, results: List['SearchResult']) -> List['SearchResult']:
        """Apply semantic clustering to group similar results"""
        
        if len(results) < 2:
            return results
        
        try:
            # Extract embeddings for clustering
            embeddings = []
            for result in results:
                if 'embedding' in result.metadata:
                    embeddings.append(result.metadata['embedding'])
                else:
                    # Generate embedding for clustering
                    embedding = await self._generate_embedding(result.content[:500])
                    embeddings.append(embedding)
                    result.metadata['embedding'] = embedding
            
            # Simple clustering based on similarity threshold
            clusters = []
            clustered_results = []
            used_indices = set()
            
            for i, result in enumerate(results):
                if i in used_indices:
                    continue
                
                cluster = [result]
                used_indices.add(i)
                
                # Find similar results
                for j, other_result in enumerate(results):
                    if j in used_indices or i == j:
                        continue
                    
                    similarity = self._compute_embedding_similarity(
                        embeddings[i], embeddings[j]
                    )
                    
                    if similarity > 0.85:  # High similarity threshold for clustering
                        cluster.append(other_result)
                        used_indices.add(j)
                
                # Take the best result from each cluster
                if cluster:
                    best_result = max(cluster, key=lambda x: x.vector_score)
                    # Add cluster information to metadata
                    best_result.metadata['cluster_size'] = len(cluster)
                    clustered_results.append(best_result)
            
            self.logger.debug(f"Clustered {len(results)} results into {len(clustered_results)} clusters")
            return clustered_results
            
        except Exception as e:
            self.logger.error(f"Semantic clustering failed: {e}")
            return results
    
    def _compute_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _deduplicate_results(self, results: List['SearchResult']) -> List['SearchResult']:
        """Remove duplicate results based on content similarity"""
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use combination of document_id and clause_id as deduplication key
            dedup_key = f"{result.document_id}:{result.clause_id}"
            
            if dedup_key not in seen_content:
                seen_content.add(dedup_key)
                unique_results.append(result)
        
        # Sort by vector score
        return sorted(unique_results, key=lambda x: x.vector_score, reverse=True)
    
    def _build_cache_key(self, query) -> str:
        """Build cache key for query"""
        import hashlib
        
        key_components = [
            query.text,
            str(sorted(query.entities)),
            str(sorted(query.clause_types)),
            str(self.config.similarity_threshold),
            str(self.config.enable_query_expansion)
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding and vector store statistics"""
        
        stats = {
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "cache_size": len(self.query_cache)
        }
        
        # Get vector store statistics if available
        if hasattr(self.vector_manager, 'get_stats'):
            try:
                vector_stats = await self.vector_manager.get_stats()
                stats.update(vector_stats)
            except Exception as e:
                self.logger.error(f"Failed to get vector store stats: {e}")
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector searcher statistics"""
        
        avg_execution_time = 0.0
        if self.stats["queries_executed"] > 0:
            avg_execution_time = self.stats["total_execution_time"] / self.stats["queries_executed"]
        
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
        
        return {
            **self.stats,
            "average_execution_time": avg_execution_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache)
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Vector searcher cache cleared")

# Factory function
def create_vector_searcher(config: VectorSearchConfig = None, vector_manager=None) -> VectorSearcher:
    """Create and return a configured vector searcher"""
    if config is None:
        config = VectorSearchConfig()
    return VectorSearcher(config, vector_manager)