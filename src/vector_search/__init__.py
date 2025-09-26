"""
Vector Search Module (Module 6)

This module provides comprehensive vector search capabilities for ContractSense,
integrating semantic similarity search with knowledge graph traversal for
enhanced contract analysis and retrieval.

Key Components:
- EmbeddingGenerator: Multi-model embedding generation pipeline
- VectorStore: Multiple backend support (Chroma, FAISS, in-memory)
- HybridRetriever: Graph-guided vector search and result fusion
- QueryProcessor: Intelligent query understanding and optimization
- VectorSearchIntegrator: Main integration layer

Features:
- Multiple embedding models (OpenAI, Sentence-BERT, etc.)
- Hybrid retrieval strategies combining vector and graph search
- Intelligent query processing with intent detection
- Performance optimization and caching
- Integration with Module 5 knowledge graph
- Comprehensive monitoring and analytics

Usage:
    from src.vector_search import VectorSearchIntegrator, VectorSearchConfig
    
    # Create configuration
    config = VectorSearchConfig(
        embedding_model="all-MiniLM-L6-v2",
        vector_backend="chroma",
        default_strategy="hybrid_parallel"
    )
    
    # Initialize integrator
    integrator = VectorSearchIntegrator(config)
    
    # Index documents
    integrator.index_contracts_from_module5()
    
    # Search
    results = integrator.search("Find all termination clauses with liability provisions")
"""

__version__ = "1.0.0"
__author__ = "ContractSense Team"

import logging

# Configure module logging
logger = logging.getLogger(__name__)

# Import main components
from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingConfig, 
    GeneratedEmbedding,
    TextChunk,
    create_embedding_generator
)

from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorSearchEngine,
    SearchResult,
    SearchQuery,
    create_vector_store
)

from .hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    HybridQuery,
    RetrievalResult,
    RetrievalStrategy,
    create_hybrid_retriever
)

from .query_processor import (
    QueryProcessor,
    QueryAnalysis,
    QueryIntent,
    ExtractedEntity,
    create_query_processor
)

from .integration import (
    VectorSearchIntegrator,
    VectorSearchConfig,
    create_vector_search_integrator
)

# Export all public components
__all__ = [
    # Main integration
    "VectorSearchIntegrator",
    "VectorSearchConfig", 
    "create_vector_search_integrator",
    
    # Embedding generation
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "GeneratedEmbedding", 
    "TextChunk",
    "create_embedding_generator",
    
    # Vector storage
    "VectorStore",
    "VectorStoreConfig",
    "VectorSearchEngine",
    "SearchResult",
    "SearchQuery", 
    "create_vector_store",
    
    # Hybrid retrieval
    "HybridRetriever",
    "HybridConfig",
    "HybridQuery", 
    "RetrievalResult",
    "RetrievalStrategy",
    "create_hybrid_retriever",
    
    # Query processing
    "QueryProcessor",
    "QueryAnalysis",
    "QueryIntent",
    "ExtractedEntity",
    "create_query_processor"
]

# Module information
MODULE_INFO = {
    "name": "Vector Search Module",
    "version": __version__,
    "description": "Hybrid vector search system with knowledge graph integration",
    "components": {
        "embedding_generator": "Multi-model embedding generation pipeline",
        "vector_store": "Vector database with multiple backend support", 
        "hybrid_retriever": "Graph-guided hybrid retrieval engine",
        "query_processor": "Intelligent query understanding and optimization",
        "integration": "Main system integration layer"
    },
    "features": [
        "Multiple embedding model support",
        "Hybrid retrieval strategies",
        "Knowledge graph integration", 
        "Intelligent query processing",
        "Performance optimization",
        "Comprehensive monitoring"
    ],
    "dependencies": {
        "required": ["numpy", "pathlib", "dataclasses", "typing", "logging"],
        "optional": [
            "sentence-transformers",
            "openai", 
            "chromadb",
            "faiss-cpu",
            "transformers",
            "torch"
        ]
    }
}

def get_module_info() -> dict:
    """Get comprehensive module information"""
    return MODULE_INFO.copy()

def check_dependencies() -> dict:
    """Check availability of optional dependencies"""
    
    dependencies = {
        "sentence_transformers": False,
        "openai": False,
        "chromadb": False, 
        "faiss": False,
        "transformers": False,
        "torch": False
    }
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
    except ImportError:
        pass
    
    # Check OpenAI
    try:
        import openai
        dependencies["openai"] = True
    except ImportError:
        pass
    
    # Check ChromaDB
    try:
        import chromadb
        dependencies["chromadb"] = True
    except ImportError:
        pass
    
    # Check FAISS
    try:
        import faiss
        dependencies["faiss"] = True
    except ImportError:
        pass
    
    # Check Transformers
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    # Check PyTorch
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    return dependencies

def get_recommended_config() -> VectorSearchConfig:
    """Get recommended configuration based on available dependencies"""
    
    deps = check_dependencies()
    
    # Determine best embedding model
    if deps["sentence_transformers"]:
        embedding_model = "all-MiniLM-L6-v2"
        embedding_type = "sentence_transformers"
    elif deps["openai"]:
        embedding_model = "text-embedding-ada-002" 
        embedding_type = "openai"
    else:
        embedding_model = "fallback"
        embedding_type = "fallback"
        logger.warning("No embedding libraries available, using fallback")
    
    # Determine best vector backend
    if deps["chromadb"]:
        vector_backend = "chroma"
    elif deps["faiss"]:
        vector_backend = "faiss"
    else:
        vector_backend = "memory"
        logger.warning("No vector database libraries available, using memory backend")
    
    config = VectorSearchConfig(
        embedding_model=embedding_model,
        embedding_type=embedding_type,
        vector_backend=vector_backend,
        enable_caching=True,
        parallel_processing=True
    )
    
    logger.info(f"Recommended configuration: {embedding_type} + {vector_backend}")
    return config

def create_default_integrator(workspace_root: str = None) -> VectorSearchIntegrator:
    """Create integrator with recommended configuration"""
    
    config = get_recommended_config()
    return create_vector_search_integrator(config, workspace_root)

# Module initialization message
logger.info(f"Vector Search Module {__version__} loaded")
logger.info("Components: embedding_generator, vector_store, hybrid_retriever, query_processor, integration")

# Check dependencies on import
_deps = check_dependencies()
_available_deps = [k for k, v in _deps.items() if v]
_missing_deps = [k for k, v in _deps.items() if not v]

if _available_deps:
    logger.info(f"Available optional dependencies: {', '.join(_available_deps)}")

if _missing_deps:
    logger.info(f"Missing optional dependencies: {', '.join(_missing_deps)}")
    logger.info("Some features may be limited. Install missing dependencies for full functionality.")

# Performance and capability summary
CAPABILITY_SUMMARY = {
    "embedding_models": [
        "sentence-transformers" if _deps["sentence_transformers"] else None,
        "openai" if _deps["openai"] else None,
        "transformers" if _deps["transformers"] else None,
        "fallback (always available)"
    ],
    "vector_backends": [
        "chroma" if _deps["chromadb"] else None,
        "faiss" if _deps["faiss"] else None, 
        "memory (always available)"
    ],
    "features": {
        "basic_vector_search": True,
        "hybrid_retrieval": True,
        "knowledge_graph_integration": True,  # Depends on Module 5
        "query_processing": True,
        "multiple_embedding_models": any([_deps["sentence_transformers"], _deps["openai"], _deps["transformers"]]),
        "persistent_storage": _deps["chromadb"],
        "high_performance_search": _deps["faiss"],
        "gpu_acceleration": _deps["torch"]
    }
}

def get_capabilities() -> dict:
    """Get current system capabilities"""
    return CAPABILITY_SUMMARY.copy()