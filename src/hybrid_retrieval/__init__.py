"""
Hybrid Retrieval Engine - Module 7

This module implements a sophisticated hybrid retrieval engine that combines
the precision of graph traversal with the flexibility of semantic vector search
for optimal legal document retrieval.

Key Components:
- HybridEngine: Main orchestration and coordination
- QueryProcessor: Advanced natural language understanding  
- GraphSearcher: Graph traversal and Cypher query generation
- VectorSearcher: Semantic search using embeddings
- ResultFusion: Intelligent score combination and ranking
- PerformanceMonitor: Comprehensive metrics and optimization

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

# Import order is important to avoid circular dependencies
from .hybrid_engine import (
    SearchQuery,
    SearchResult, 
    SearchStrategy,
    HybridConfig,
    HybridEngine,
    create_hybrid_engine
)

from .query_processor import (
    QueryProcessorConfig,
    QueryProcessor,
    create_query_processor
)

from .graph_searcher import (
    GraphSearchConfig,
    QueryType,
    GraphSearcher,
    create_graph_searcher
)

from .vector_searcher import (
    VectorSearchConfig,
    SearchMode,
    VectorSearcher,
    create_vector_searcher
)

from .result_fusion import (
    FusionConfig,
    FusionStrategy,
    ResultFusion,
    create_result_fusion
)

from .performance_monitor import (
    PerformanceConfig,
    QueryMetrics,
    SystemMetrics,
    AlertLevel,
    PerformanceAlert,
    PerformanceMonitor,
    create_performance_monitor
)

# Version information
__version__ = "1.0.0"
__author__ = "ContractSense Team"

# Public API
__all__ = [
    # Core classes
    'HybridEngine',
    'QueryProcessor', 
    'GraphSearcher',
    'VectorSearcher',
    'ResultFusion',
    'PerformanceMonitor',
    
    # Data models
    'SearchQuery',
    'SearchResult',
    'QueryMetrics',
    'SystemMetrics',
    'PerformanceAlert',
    
    # Enums
    'SearchStrategy',
    'QueryType', 
    'SearchMode',
    'FusionStrategy',
    'AlertLevel',
    
    # Configuration classes
    'HybridConfig',
    'QueryProcessorConfig',
    'GraphSearchConfig',
    'VectorSearchConfig', 
    'FusionConfig',
    'PerformanceConfig',
    
    # Factory functions
    'create_hybrid_engine',
    'create_query_processor',
    'create_graph_searcher',
    'create_vector_searcher', 
    'create_result_fusion',
    'create_performance_monitor'
]

# Module-level logging configuration
import logging

logger = logging.getLogger(__name__)
logger.info(f"Hybrid Retrieval Engine v{__version__} initialized")

def get_version() -> str:
    """Get the module version"""
    return __version__

def get_components() -> dict:
    """Get information about all components"""
    return {
        'HybridEngine': 'Main orchestration engine combining graph and vector search',
        'QueryProcessor': 'Natural language understanding and query enhancement', 
        'GraphSearcher': 'Graph traversal with Cypher query generation',
        'VectorSearcher': 'Semantic search using embeddings',
        'ResultFusion': 'Intelligent score combination and result ranking',
        'PerformanceMonitor': 'Comprehensive performance metrics and optimization'
    }