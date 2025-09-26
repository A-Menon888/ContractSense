"""
Vector Search Integration Module

This module provides the main integration layer for Module 6, combining all
vector search components into a unified interface for the ContractSense system.

Key Features:
- Unified API for vector search functionality
- Integration with existing Modules 1-5
- Comprehensive configuration management
- Performance monitoring and optimization
- Error handling and fallback mechanisms
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Import vector search components
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig, create_embedding_generator
from .vector_store import VectorStore, VectorStoreConfig, VectorSearchEngine, create_vector_store
from .hybrid_retriever import HybridRetriever, HybridConfig, HybridQuery, RetrievalResult, create_hybrid_retriever
from .query_processor import QueryProcessor, QueryAnalysis, create_query_processor

# Import knowledge graph integration
try:
    from ..knowledge_graph.integration import KnowledgeGraphIntegrator
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KnowledgeGraphIntegrator = None
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logging.warning("Knowledge graph not available for vector search integration")

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchConfig:
    """Comprehensive configuration for vector search system"""
    
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_type: str = "sentence_transformers"  # sentence_transformers, openai, transformers
    embedding_device: str = "cpu"
    
    # Vector store configuration  
    vector_backend: str = "memory"  # memory, chroma, faiss
    vector_persist_dir: str = "vector_data"
    collection_name: str = "contract_embeddings"
    
    # Hybrid retrieval configuration
    default_strategy: str = "hybrid_parallel"
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    
    # Performance settings
    batch_size: int = 32
    max_chunk_size: int = 512
    enable_caching: bool = True
    parallel_processing: bool = True
    
    # Quality settings
    min_score_threshold: float = 0.1
    diversity_threshold: float = 0.8
    max_results: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorSearchConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VectorSearchConfig':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class VectorSearchIntegrator:
    """Main integration class for vector search functionality"""
    
    def __init__(self, config: VectorSearchConfig = None, workspace_root: str = None):
        """Initialize the vector search integrator"""
        
        self.config = config or VectorSearchConfig()
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        
        # Initialize components
        self.embedding_generator = None
        self.vector_store = None
        self.hybrid_retriever = None
        self.query_processor = None
        self.graph_integrator = None
        
        # Statistics and monitoring
        self.stats = {
            "initialization_time": 0.0,
            "total_queries": 0,
            "total_documents_indexed": 0,
            "total_embeddings_generated": 0,
            "average_query_time": 0.0,
            "system_status": "initializing"
        }
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all vector search components"""
        
        start_time = time.time()
        
        try:
            logger.info("Initializing vector search system...")
            
            # Initialize embedding generator
            embedding_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                model_type=self.config.embedding_type,
                device=self.config.embedding_device,
                batch_size=self.config.batch_size,
                chunk_size=self.config.max_chunk_size,
                cache_embeddings=self.config.enable_caching
            )
            
            # Remove conflicting parameters from the config dict
            embedding_kwargs = embedding_config.__dict__.copy()
            embedding_kwargs.pop('model_name', None)
            embedding_kwargs.pop('model_type', None)
            
            self.embedding_generator = create_embedding_generator(
                model_name=self.config.embedding_model,
                model_type=self.config.embedding_type,
                **embedding_kwargs
            )
            
            logger.info("✓ Embedding generator initialized")
            
            # Initialize vector store
            vector_config = VectorStoreConfig(
                backend=self.config.vector_backend,
                persist_directory=str(self.workspace_root / self.config.vector_persist_dir),
                collection_name=self.config.collection_name,
                batch_size=self.config.batch_size,
                enable_persistence=True
            )
            
            self.vector_store = create_vector_store(vector_config)
            logger.info(f"✓ Vector store initialized ({self.config.vector_backend})")
            
            # Initialize query processor
            self.query_processor = create_query_processor()
            logger.info("✓ Query processor initialized")
            
            # Initialize hybrid retriever
            hybrid_config = HybridConfig(
                vector_weight=self.config.vector_weight,
                graph_weight=self.config.graph_weight,
                final_top_k=self.config.max_results,
                min_vector_score=self.config.min_score_threshold,
                diversity_threshold=self.config.diversity_threshold,
                enable_caching=self.config.enable_caching
            )
            
            self.hybrid_retriever = create_hybrid_retriever(
                self.vector_store,
                self.embedding_generator,
                hybrid_config
            )
            
            logger.info("✓ Hybrid retriever initialized")
            
            # Initialize knowledge graph integrator if available
            if KNOWLEDGE_GRAPH_AVAILABLE:
                try:
                    self.graph_integrator = KnowledgeGraphIntegrator()
                    logger.info("✓ Knowledge graph integration enabled")
                except Exception as e:
                    logger.warning(f"Knowledge graph integration failed: {e}")
            
            initialization_time = time.time() - start_time
            self.stats["initialization_time"] = initialization_time
            self.stats["system_status"] = "ready"
            
            logger.info(f"Vector search system initialized successfully in {initialization_time:.2f}s")
            
        except Exception as e:
            self.stats["system_status"] = "error"
            logger.error(f"Failed to initialize vector search system: {e}")
            raise
    
    def index_contract(self, contract_data: Dict[str, Any]) -> bool:
        """Index a single contract for vector search"""
        
        try:
            if self.stats["system_status"] != "ready":
                logger.error("System not ready for indexing")
                return False
            
            document_id = contract_data.get("id", "unknown")
            text_content = contract_data.get("text", "")
            metadata = contract_data.get("metadata", {})
            
            logger.info(f"Indexing contract: {document_id}")
            
            # Create chunks
            chunks = self.embedding_generator.chunk_document(
                document_text=text_content,
                document_id=document_id,
                metadata=metadata
            )
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Store in vector database
            success = self.vector_store.add_embeddings(embeddings)
            
            if success:
                self.stats["total_documents_indexed"] += 1
                self.stats["total_embeddings_generated"] += len(embeddings)
                logger.info(f"Successfully indexed contract {document_id} ({len(embeddings)} chunks)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to index contract {contract_data.get('id', 'unknown')}: {e}")
            return False
    
    def index_contracts_from_module5(self, contract_ids: List[str] = None) -> Dict[str, Any]:
        """Index contracts using Module 5 knowledge graph data"""
        
        if not self.graph_integrator:
            return {"error": "Knowledge graph not available"}
        
        try:
            logger.info("Indexing contracts from Module 5 knowledge graph")
            
            # Get contract list from graph if not provided
            if contract_ids is None:
                graph_response = self.graph_integrator.query_graph_for_insights("list_contracts")
                
                if "error" in graph_response:
                    return {"error": f"Could not get contract list: {graph_response['error']}"}
                
                contract_ids = graph_response.get("contract_ids", [])
            
            results = {
                "total_contracts": len(contract_ids),
                "successful_indexes": 0,
                "failed_indexes": 0,
                "errors": []
            }
            
            # Process each contract
            for contract_id in contract_ids:
                try:
                    # Use embedding generator's Module 5 integration
                    embeddings = self.embedding_generator.process_contract_from_module5(contract_id)
                    
                    if embeddings:
                        success = self.vector_store.add_embeddings(embeddings)
                        if success:
                            results["successful_indexes"] += 1
                            self.stats["total_embeddings_generated"] += len(embeddings)
                        else:
                            results["failed_indexes"] += 1
                            results["errors"].append(f"Failed to store embeddings for {contract_id}")
                    else:
                        results["failed_indexes"] += 1
                        results["errors"].append(f"No embeddings generated for {contract_id}")
                        
                except Exception as e:
                    results["failed_indexes"] += 1
                    results["errors"].append(f"Error processing {contract_id}: {str(e)}")
            
            self.stats["total_documents_indexed"] += results["successful_indexes"]
            
            logger.info(f"Module 5 indexing complete: {results['successful_indexes']}/{results['total_contracts']} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to index contracts from Module 5: {e}")
            return {"error": str(e)}
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Main search interface"""
        
        start_time = time.time()
        
        try:
            if self.stats["system_status"] != "ready":
                logger.error("System not ready for search")
                return []
            
            logger.debug(f"Processing search query: {query}")
            
            # Process query
            query_analysis = self.query_processor.process_query(query)
            
            # Convert to hybrid query
            hybrid_query = self.query_processor.to_hybrid_query(
                query_analysis,
                additional_params=kwargs
            )
            
            # Execute retrieval
            results = self.hybrid_retriever.retrieve(hybrid_query)
            
            # Convert to dictionaries for API response
            result_dicts = [result.to_dict() for result in results]
            
            # Update statistics
            query_time = time.time() - start_time
            self._update_query_stats(query_time)
            
            logger.debug(f"Search completed in {query_time:.3f}s, {len(results)} results")
            
            return result_dicts
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def advanced_search(self, query_config: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced search with detailed configuration"""
        
        try:
            # Extract query text
            query_text = query_config.get("query", "")
            if not query_text:
                return {"error": "Query text required"}
            
            # Process query
            query_analysis = self.query_processor.process_query(query_text)
            
            # Create hybrid query with advanced parameters
            hybrid_query = HybridQuery(
                text=query_text,
                query_type=query_config.get("query_type", "general"),
                entity_hints=query_config.get("entity_hints", []),
                relationship_hints=query_config.get("relationship_hints", []),
                strategy=getattr(self.hybrid_retriever, query_config.get("strategy", "adaptive"), None),
                top_k=query_config.get("top_k", 10),
                metadata_filter=query_config.get("metadata_filter", {}),
                include_context=query_config.get("include_context", True),
                include_explanations=query_config.get("include_explanations", True)
            )
            
            # Execute search
            results = self.hybrid_retriever.retrieve(hybrid_query)
            
            # Compile comprehensive response
            response = {
                "query_analysis": query_analysis.to_dict(),
                "results": [result.to_dict() for result in results],
                "result_count": len(results),
                "search_metadata": {
                    "strategy_used": hybrid_query.strategy.value if hybrid_query.strategy else "adaptive",
                    "filters_applied": hybrid_query.metadata_filter,
                    "processing_time": time.time()
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            return {"error": str(e)}
    
    def get_document_embeddings(self, document_id: str) -> Dict[str, Any]:
        """Get embeddings for a specific document"""
        
        try:
            # Search for all chunks of this document
            results = self.vector_store.search({
                "metadata_filter": {"document_id": document_id},
                "top_k": 1000,  # Get all chunks
                "include_embeddings": True
            })
            
            if not results:
                return {"error": f"No embeddings found for document {document_id}"}
            
            return {
                "document_id": document_id,
                "chunk_count": len(results),
                "embeddings": [
                    {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "embedding": result.embedding.tolist() if result.embedding is not None else None,
                        "metadata": result.metadata
                    }
                    for result in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get embeddings for document {document_id}: {e}")
            return {"error": str(e)}
    
    def analyze_corpus(self) -> Dict[str, Any]:
        """Analyze the current corpus of indexed documents"""
        
        try:
            # Get basic statistics
            total_embeddings = self.vector_store.count()
            
            # Get component statistics
            embedding_stats = self.embedding_generator.get_model_info()
            vector_stats = self.vector_store.get_stats()
            retriever_stats = self.hybrid_retriever.get_stats()
            processor_stats = self.query_processor.get_stats()
            
            analysis = {
                "corpus_statistics": {
                    "total_embeddings": total_embeddings,
                    "total_documents_indexed": self.stats["total_documents_indexed"],
                    "embedding_dimension": embedding_stats.get("dimension", "unknown")
                },
                "system_performance": {
                    "initialization_time": self.stats["initialization_time"],
                    "total_queries_processed": self.stats["total_queries"],
                    "average_query_time": self.stats["average_query_time"],
                    "system_status": self.stats["system_status"]
                },
                "component_stats": {
                    "embedding_generator": embedding_stats,
                    "vector_store": vector_stats,
                    "hybrid_retriever": retriever_stats,
                    "query_processor": processor_stats
                },
                "configuration": self.config.to_dict()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Corpus analysis failed: {e}")
            return {"error": str(e)}
    
    def _update_query_stats(self, query_time: float):
        """Update query statistics"""
        
        self.stats["total_queries"] += 1
        
        total_time = self.stats["average_query_time"] * (self.stats["total_queries"] - 1)
        total_time += query_time
        self.stats["average_query_time"] = total_time / self.stats["total_queries"]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        
        health = {
            "system_status": self.stats["system_status"],
            "components": {},
            "overall_health": "healthy"
        }
        
        try:
            # Check embedding generator
            if self.embedding_generator:
                try:
                    # Test embedding generation
                    test_chunks = self.embedding_generator.chunk_document("test", "test document")
                    if test_chunks:
                        health["components"]["embedding_generator"] = "healthy"
                    else:
                        health["components"]["embedding_generator"] = "warning"
                except Exception as e:
                    health["components"]["embedding_generator"] = f"error: {e}"
                    health["overall_health"] = "degraded"
            
            # Check vector store
            if self.vector_store:
                try:
                    count = self.vector_store.count()
                    health["components"]["vector_store"] = f"healthy ({count} embeddings)"
                except Exception as e:
                    health["components"]["vector_store"] = f"error: {e}"
                    health["overall_health"] = "degraded"
            
            # Check hybrid retriever
            if self.hybrid_retriever:
                health["components"]["hybrid_retriever"] = "healthy"
            
            # Check query processor
            if self.query_processor:
                health["components"]["query_processor"] = "healthy"
            
            # Check knowledge graph integration
            if self.graph_integrator:
                health["components"]["knowledge_graph"] = "available"
            else:
                health["components"]["knowledge_graph"] = "not_available"
            
        except Exception as e:
            health["overall_health"] = "error"
            health["error"] = str(e)
        
        return health
    
    def save_system_state(self, filepath: str = None):
        """Save system configuration and state"""
        
        if filepath is None:
            filepath = self.workspace_root / "vector_search_state.json"
        
        try:
            state = {
                "config": self.config.to_dict(),
                "stats": self.stats,
                "component_info": {
                    "embedding_generator": self.embedding_generator.get_model_info() if self.embedding_generator else None,
                    "vector_store": self.vector_store.get_stats() if self.vector_store else None,
                    "hybrid_retriever": self.hybrid_retriever.get_stats() if self.hybrid_retriever else None,
                    "query_processor": self.query_processor.get_stats() if self.query_processor else None
                },
                "timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")

# Factory function for easy instantiation
def create_vector_search_integrator(
    config: VectorSearchConfig = None,
    workspace_root: str = None
) -> VectorSearchIntegrator:
    """Factory function to create vector search integrator"""
    
    return VectorSearchIntegrator(config, workspace_root)