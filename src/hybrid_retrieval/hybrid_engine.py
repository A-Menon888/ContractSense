"""
Hybrid Retrieval Engine (Module 7)

This module implements the core hybrid retrieval engine that combines knowledge graph
traversal with vector search to provide both precise and flexible search capabilities
for legal document analysis.

The engine processes natural language queries, executes parallel graph and vector searches,
and intelligently fuses results for optimal relevance and coverage.

Key Components:
- HybridEngine: Main orchestration engine  
- QueryProcessor: Natural language understanding and parsing
- GraphSearcher: Knowledge graph traversal and Cypher generation
- VectorSearcher: Semantic vector search coordination
- ResultFusion: Score combination and intelligent ranking

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import json
from pathlib import Path

@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval engine"""
    enable_parallel_search: bool = True
    enable_result_fusion: bool = True
    max_results: int = 20
    timeout_seconds: int = 30
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_performance_monitoring: bool = True
    fallback_on_error: bool = True
    min_confidence_threshold: float = 0.1
    
# Core data models
@dataclass
class SearchQuery:
    """Represents a hybrid search query with metadata"""
    text: str
    query_id: str = ""
    user_id: str = ""
    
    # Query parsing results
    entities: List[str] = field(default_factory=list)
    clause_types: List[str] = field(default_factory=list)
    amounts: List[Dict[str, Any]] = field(default_factory=list)
    date_ranges: List[Dict[str, str]] = field(default_factory=list)
    
    # Search parameters
    max_results: int = 10
    min_confidence: float = 0.1
    strategy: str = "auto"  # auto, graph_heavy, vector_heavy, balanced
    
    # Advanced filters
    filters: Dict[str, Any] = field(default_factory=dict)
    include_confidence: bool = True
    include_explanation: bool = False

class SearchStrategy(Enum):
    """Available search strategies"""
    AUTO = "auto"
    GRAPH_HEAVY = "graph_heavy" 
    VECTOR_HEAVY = "vector_heavy"
    BALANCED = "balanced"
    GRAPH_ONLY = "graph_only"
    VECTOR_ONLY = "vector_only"

@dataclass
class SearchResult:
    """Unified search result from hybrid engine"""
    content: str
    document_id: str
    clause_id: str = ""
    
    # Scoring information
    final_score: float = 0.0
    graph_score: float = 0.0
    vector_score: float = 0.0
    confidence: float = 0.0
    
    # Source information  
    source_path: List[str] = field(default_factory=list)  # Graph path
    document_title: str = ""
    clause_type: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class HybridSearchResponse:
    """Complete response from hybrid search"""
    query: SearchQuery
    results: List[SearchResult]
    
    # Performance metrics
    total_time: float = 0.0
    graph_time: float = 0.0
    vector_time: float = 0.0
    fusion_time: float = 0.0
    
    # Search statistics
    total_candidates: int = 0
    graph_candidates: int = 0  
    vector_candidates: int = 0
    duplicates_removed: int = 0
    
    # Strategy information
    strategy_used: str = ""
    strategy_explanation: str = ""
    
    # Metadata
    success: bool = True
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

class HybridEngine:
    """
    Main hybrid retrieval engine that orchestrates graph and vector search
    
    The engine processes queries through the following pipeline:
    1. Query parsing and understanding
    2. Strategy selection and planning
    3. Parallel graph and vector search execution
    4. Result fusion and ranking  
    5. Response formatting and metadata
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (will be set during setup)
        self.query_processor = None
        self.graph_searcher = None  
        self.vector_searcher = None
        self.result_fusion = None
        self.performance_monitor = None
        
        # Performance settings
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Strategy weights
        self.strategy_weights = {
            SearchStrategy.AUTO: {"graph": 0.6, "vector": 0.4},
            SearchStrategy.GRAPH_HEAVY: {"graph": 0.8, "vector": 0.2},
            SearchStrategy.VECTOR_HEAVY: {"graph": 0.2, "vector": 0.8},
            SearchStrategy.BALANCED: {"graph": 0.5, "vector": 0.5},
            SearchStrategy.GRAPH_ONLY: {"graph": 1.0, "vector": 0.0},
            SearchStrategy.VECTOR_ONLY: {"graph": 0.0, "vector": 1.0}
        }
        
        # Cache for frequent queries
        self.query_cache: Dict[str, HybridSearchResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.stats = {
            "queries_processed": 0,
            "total_search_time": 0.0,
            "average_response_time": 0.0,
            "graph_queries": 0,
            "vector_queries": 0,
            "cache_hit_rate": 0.0
        }
        
        self.logger.info("Hybrid retrieval engine initialized")
    
    def setup(self, knowledge_graph_integrator, vector_search_integrator):
        """
        Initialize the hybrid engine with required components
        
        Args:
            knowledge_graph_integrator: Module 5 knowledge graph integration
            vector_search_integrator: Module 6 vector search integration
        """
        try:
            # Import and initialize components
            from .query_processor import QueryProcessor, QueryProcessorConfig
            from .graph_searcher import GraphSearcher, GraphSearchConfig  
            from .vector_searcher import VectorSearcher, VectorSearchConfig
            from .result_fusion import ResultFusion, FusionConfig
            from .performance_monitor import PerformanceMonitor
            
            # Initialize query processor
            query_config = QueryProcessorConfig(
                enable_entity_extraction=True,
                enable_intent_classification=True,
                enable_query_expansion=self.config.get('query_expansion', True)
            )
            self.query_processor = QueryProcessor(query_config)
            
            # Initialize graph searcher
            graph_config = GraphSearchConfig(
                max_traversal_depth=self.config.get('max_graph_depth', 3),
                enable_cypher_optimization=True,
                timeout_seconds=self.config.get('graph_timeout', 10)
            )
            self.graph_searcher = GraphSearcher(
                graph_config, 
                knowledge_graph_integrator
            )
            
            # Initialize vector searcher  
            vector_config = VectorSearchConfig(
                max_results=self.config.get('max_vector_results', 20),
                similarity_threshold=self.config.get('vector_threshold', 0.1),
                enable_reranking=False  # Module 8 will handle this
            )
            self.vector_searcher = VectorSearcher(
                vector_config,
                vector_search_integrator  
            )
            
            # Initialize result fusion
            fusion_config = FusionConfig(
                deduplication_threshold=self.config.get('dedup_threshold', 0.8),
                enable_diversity_ranking=True,
                max_final_results=self.config.get('max_results', 10)
            )
            self.result_fusion = ResultFusion(fusion_config)
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            self.logger.info("✓ Hybrid engine components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup hybrid engine: {e}")
            return False
    
    async def search(self, query: Union[str, SearchQuery]) -> HybridSearchResponse:
        """
        Execute hybrid search combining graph traversal and vector search
        
        Args:
            query: Search query string or SearchQuery object
            
        Returns:
            HybridSearchResponse with results and metadata
        """
        start_time = time.time()
        
        # Convert string to SearchQuery if needed
        if isinstance(query, str):
            query = SearchQuery(text=query)
        
        self.logger.info(f"Processing hybrid search query: '{query.text}'")
        
        try:
            # Step 1: Query processing and understanding
            processed_query = await self._process_query(query)
            
            # Step 2: Strategy selection
            strategy = self._select_strategy(processed_query)
            
            # Step 3: Check cache
            if self.cache_enabled:
                cached_response = self._check_cache(processed_query, strategy)
                if cached_response:
                    self.cache_hits += 1
                    self.logger.info("Cache hit - returning cached results")
                    return cached_response
            
            self.cache_misses += 1
            
            # Step 4: Parallel search execution
            graph_results, vector_results, search_times = await self._execute_parallel_search(
                processed_query, strategy
            )
            
            # Step 5: Result fusion and ranking
            fusion_start = time.time()
            fused_results = await self._fuse_results(
                graph_results, vector_results, processed_query, strategy
            )
            fusion_time = time.time() - fusion_start
            
            # Step 6: Build response
            total_time = time.time() - start_time
            response = self._build_response(
                processed_query, fused_results, search_times, 
                fusion_time, total_time, strategy,
                len(graph_results), len(vector_results)
            )
            
            # Step 7: Cache response if enabled
            if self.cache_enabled and response.success:
                self._cache_response(processed_query, strategy, response)
            
            # Update statistics
            self._update_stats(total_time)
            
            self.logger.info(
                f"Hybrid search completed in {total_time:.3f}s - "
                f"{len(response.results)} results"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return HybridSearchResponse(
                query=query,
                results=[],
                success=False,
                error_message=str(e),
                total_time=time.time() - start_time
            )
    
    async def _process_query(self, query: SearchQuery) -> SearchQuery:
        """Process and enhance the input query"""
        if not self.query_processor:
            return query
        
        try:
            # Extract entities, clause types, amounts, etc.
            processed = await self.query_processor.process_query(query)
            return processed
        except Exception as e:
            self.logger.warning(f"Query processing failed: {e}")
            return query
    
    def _select_strategy(self, query: SearchQuery) -> SearchStrategy:
        """Select optimal search strategy based on query characteristics"""
        
        # If user specified strategy, use it
        if query.strategy != "auto":
            try:
                return SearchStrategy(query.strategy)
            except ValueError:
                self.logger.warning(f"Invalid strategy '{query.strategy}', using AUTO")
        
        # Auto-select based on query characteristics
        if query.entities and query.clause_types:
            # Lots of structured elements -> favor graph
            return SearchStrategy.GRAPH_HEAVY
        elif len(query.text.split()) > 10:
            # Long natural language query -> favor vector
            return SearchStrategy.VECTOR_HEAVY  
        elif query.amounts or query.date_ranges:
            # Specific values -> favor graph
            return SearchStrategy.GRAPH_HEAVY
        else:
            # Default balanced approach
            return SearchStrategy.BALANCED
    
    async def _execute_parallel_search(
        self, query: SearchQuery, strategy: SearchStrategy
    ) -> Tuple[List[SearchResult], List[SearchResult], Dict[str, float]]:
        """Execute graph and vector searches in parallel"""
        
        search_times = {"graph": 0.0, "vector": 0.0}
        weights = self.strategy_weights[strategy]
        
        tasks = []
        
        # Create search tasks based on strategy weights
        if weights["graph"] > 0:
            tasks.append(("graph", self.graph_searcher.search(query)))
            
        if weights["vector"] > 0:
            tasks.append(("vector", self.vector_searcher.search(query)))
        
        # Execute searches in parallel
        graph_results = []
        vector_results = []
        
        if not tasks:
            return graph_results, vector_results, search_times
        
        # Run searches concurrently with timeout
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_type = {}
            
            for search_type, coro in tasks:
                future = executor.submit(asyncio.run, coro)
                future_to_type[future] = search_type
            
            # Collect results as they complete
            for future in as_completed(future_to_type.keys(), timeout=self.timeout_seconds):
                search_type = future_to_type[future]
                
                try:
                    start_time = time.time()
                    results = future.result()
                    search_times[search_type] = time.time() - start_time
                    
                    if search_type == "graph":
                        graph_results = results
                        self.stats["graph_queries"] += 1
                    else:
                        vector_results = results  
                        self.stats["vector_queries"] += 1
                        
                    self.logger.debug(
                        f"{search_type} search completed: {len(results)} results "
                        f"in {search_times[search_type]:.3f}s"
                    )
                    
                except Exception as e:
                    self.logger.error(f"{search_type} search failed: {e}")
                    search_times[search_type] = 0.0
        
        return graph_results, vector_results, search_times
    
    async def _fuse_results(
        self, graph_results: List[SearchResult], vector_results: List[SearchResult],
        query: SearchQuery, strategy: SearchStrategy
    ) -> List[SearchResult]:
        """Fuse and rank results from graph and vector searches"""
        
        if not self.result_fusion:
            # Simple concatenation fallback
            all_results = graph_results + vector_results
            return sorted(all_results, key=lambda x: x.final_score, reverse=True)[:query.max_results]
        
        try:
            weights = self.strategy_weights[strategy]
            fused = await self.result_fusion.fuse_results(
                graph_results, vector_results, weights, query
            )
            return fused[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Result fusion failed: {e}")
            # Fallback to simple combination
            all_results = graph_results + vector_results
            return sorted(all_results, key=lambda x: x.final_score, reverse=True)[:query.max_results]
    
    def _build_response(
        self, query: SearchQuery, results: List[SearchResult], 
        search_times: Dict[str, float], fusion_time: float, total_time: float,
        strategy: SearchStrategy, graph_count: int, vector_count: int
    ) -> HybridSearchResponse:
        """Build complete search response with metadata"""
        
        return HybridSearchResponse(
            query=query,
            results=results,
            total_time=total_time,
            graph_time=search_times.get("graph", 0.0),
            vector_time=search_times.get("vector", 0.0),
            fusion_time=fusion_time,
            total_candidates=graph_count + vector_count,
            graph_candidates=graph_count,
            vector_candidates=vector_count,
            duplicates_removed=max(0, (graph_count + vector_count) - len(results)),
            strategy_used=strategy.value,
            strategy_explanation=self._get_strategy_explanation(strategy),
            success=True
        )
    
    def _get_strategy_explanation(self, strategy: SearchStrategy) -> str:
        """Get human-readable explanation of strategy selection"""
        explanations = {
            SearchStrategy.AUTO: "Automatically selected optimal strategy",
            SearchStrategy.GRAPH_HEAVY: "Emphasized graph traversal for precise entity relationships",
            SearchStrategy.VECTOR_HEAVY: "Emphasized semantic search for natural language understanding",
            SearchStrategy.BALANCED: "Balanced combination of graph and vector search",
            SearchStrategy.GRAPH_ONLY: "Pure graph traversal search",
            SearchStrategy.VECTOR_ONLY: "Pure semantic vector search"
        }
        return explanations.get(strategy, "Unknown strategy")
    
    def _check_cache(self, query: SearchQuery, strategy: SearchStrategy) -> Optional[HybridSearchResponse]:
        """Check if query result is cached"""
        cache_key = self._build_cache_key(query, strategy)
        return self.query_cache.get(cache_key)
    
    def _cache_response(self, query: SearchQuery, strategy: SearchStrategy, response: HybridSearchResponse):
        """Cache successful query response"""
        cache_key = self._build_cache_key(query, strategy)
        
        # Simple LRU cache with size limit
        max_cache_size = self.config.get('max_cache_size', 1000)
        if len(self.query_cache) >= max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = response
    
    def _build_cache_key(self, query: SearchQuery, strategy: SearchStrategy) -> str:
        """Build cache key from query and strategy"""
        import hashlib
        
        key_data = {
            "text": query.text,
            "strategy": strategy.value,
            "max_results": query.max_results,
            "min_confidence": query.min_confidence,
            "filters": query.filters
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, query_time: float):
        """Update engine statistics"""
        self.stats["queries_processed"] += 1
        self.stats["total_search_time"] += query_time
        self.stats["average_response_time"] = (
            self.stats["total_search_time"] / self.stats["queries_processed"]
        )
        
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.stats["cache_hit_rate"] = self.cache_hits / total_cache_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current engine statistics"""
        return {
            **self.stats,
            "cache_size": len(self.query_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Query cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "status": "healthy",
            "components": {},
            "performance": self.get_stats()
        }
        
        # Check component health
        if self.query_processor:
            health["components"]["query_processor"] = "healthy"
        else:
            health["components"]["query_processor"] = "not_initialized"
            health["status"] = "degraded"
        
        if self.graph_searcher:
            health["components"]["graph_searcher"] = "healthy"  
        else:
            health["components"]["graph_searcher"] = "not_initialized"
            health["status"] = "degraded"
            
        if self.vector_searcher:
            health["components"]["vector_searcher"] = "healthy"
        else:
            health["components"]["vector_searcher"] = "not_initialized"
            health["status"] = "degraded"
        
        if self.result_fusion:
            health["components"]["result_fusion"] = "healthy"
        else:
            health["components"]["result_fusion"] = "not_initialized"
            health["status"] = "degraded"
        
        return health

# Factory function for easy instantiation
def create_hybrid_engine(config: HybridConfig = None,
                        query_processor=None,
                        graph_searcher=None, 
                        vector_searcher=None,
                        result_fusion=None,
                        performance_monitor=None) -> HybridEngine:
    """Create and return a configured hybrid engine instance"""
    if config is None:
        config = HybridConfig()
    return HybridEngine(
        config=config,
        query_processor=query_processor,
        graph_searcher=graph_searcher,
        vector_searcher=vector_searcher,
        result_fusion=result_fusion,
        performance_monitor=performance_monitor
    )

# Module exports
__all__ = [
    'HybridEngine', 
    'HybridConfig',
    'SearchQuery', 
    'SearchResult',
    'HybridSearchResponse',
    'SearchStrategy',
    'create_hybrid_engine'
]