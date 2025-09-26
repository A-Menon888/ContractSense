"""
Vector Store Implementation

This module provides a comprehensive vector database interface supporting multiple
backends (Chroma, FAISS, in-memory) for semantic similarity search.

Key Features:
- Multiple backend support with consistent interface
- Metadata filtering and hybrid search capabilities
- Batch operations for efficiency
- Persistence and backup functionality
- Integration with embedding generator
- Performance monitoring and optimization
"""

import logging
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime
import sqlite3
from abc import ABC, abstractmethod

# Optional dependencies with fallback handling
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMA_AVAILABLE = False
    logging.warning("chromadb not available - will use alternative backend")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logging.warning("faiss not available - will use alternative backend")

# Import our embedding components
from .embedding_generator import GeneratedEmbedding, TextChunk

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    backend: str = "memory"  # chroma, faiss, memory
    persist_directory: str = "vector_store_data"
    collection_name: str = "contract_embeddings"
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # Performance settings
    batch_size: int = 100
    index_type: str = "flat"  # flat, ivf, hnsw (FAISS)
    ef_construction: int = 200  # HNSW parameter
    m: int = 16  # HNSW parameter
    
    # Chroma settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist: bool = True
    
    # Memory settings
    enable_persistence: bool = True
    backup_interval: int = 1000  # embeddings
    
@dataclass 
class SearchResult:
    """Represents a search result with metadata"""
    chunk_id: str
    content: str
    score: float  # Similarity score
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        return result

@dataclass
class SearchQuery:
    """Represents a search query with parameters"""
    query_text: str
    query_embedding: Optional[np.ndarray] = None
    top_k: int = 10
    score_threshold: float = 0.0
    metadata_filter: Dict[str, Any] = None
    include_embeddings: bool = False
    
    def __post_init__(self):
        if self.metadata_filter is None:
            self.metadata_filter = {}

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.stats = {
            "total_embeddings": 0,
            "total_searches": 0,
            "total_search_time": 0.0,
            "average_search_time": 0.0
        }
    
    @abstractmethod
    def add_embeddings(self, embeddings: List[GeneratedEmbedding]) -> bool:
        """Add embeddings to the store"""
        pass
    
    @abstractmethod
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    def delete_by_id(self, ids: List[str]) -> bool:
        """Delete embeddings by ID"""
        pass
    
    @abstractmethod
    def get_by_id(self, ids: List[str]) -> List[Optional[SearchResult]]:
        """Get embeddings by ID"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of embeddings"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all embeddings"""
        pass
    
    def update_stats(self, search_time: float):
        """Update search statistics"""
        self.stats["total_searches"] += 1
        self.stats["total_search_time"] += search_time
        self.stats["average_search_time"] = (
            self.stats["total_search_time"] / self.stats["total_searches"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            **self.stats,
            "backend": self.config.backend,
            "collection": self.config.collection_name
        }

class MemoryVectorStore(VectorStore):
    """In-memory vector store using numpy for similarity search"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.embeddings = {}  # id -> embedding
        self.metadata = {}    # id -> metadata
        self.embeddings_matrix = None
        self.embedding_ids = []
        self._needs_rebuild = True
        
        # Set up persistence
        if config.enable_persistence:
            self.persist_path = Path(config.persist_directory)
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _rebuild_matrix(self):
        """Rebuild the embeddings matrix for efficient search"""
        if not self.embeddings or not self._needs_rebuild:
            return
        
        self.embedding_ids = list(self.embeddings.keys())
        embeddings_list = [self.embeddings[id_] for id_ in self.embedding_ids]
        
        # Check for consistent dimensions
        if embeddings_list:
            dimensions = [emb.shape[0] if len(emb.shape) == 1 else emb.shape[1] for emb in embeddings_list]
            if len(set(dimensions)) > 1:
                logger.warning(f"Inconsistent embedding dimensions: {set(dimensions)}")
                # Filter to the most common dimension
                most_common_dim = max(set(dimensions), key=dimensions.count)
                embeddings_list = [emb for emb in embeddings_list if 
                                 (emb.shape[0] if len(emb.shape) == 1 else emb.shape[1]) == most_common_dim]
                # Update embedding_ids accordingly
                self.embedding_ids = [id_ for id_, emb in zip(self.embedding_ids, embeddings_list) 
                                    if (emb.shape[0] if len(emb.shape) == 1 else emb.shape[1]) == most_common_dim]
        
        if embeddings_list:
            self.embeddings_matrix = np.vstack(embeddings_list)
        else:
            self.embeddings_matrix = np.array([])
            
        self._needs_rebuild = False
        
        logger.debug(f"Rebuilt embeddings matrix: {self.embeddings_matrix.shape}")
    
    def add_embeddings(self, embeddings: List[GeneratedEmbedding]) -> bool:
        """Add embeddings to memory store"""
        try:
            for emb in embeddings:
                self.embeddings[emb.chunk.chunk_id] = emb.embedding
                self.metadata[emb.chunk.chunk_id] = emb.chunk.to_embedding_dict()
            
            self._needs_rebuild = True
            self.stats["total_embeddings"] += len(embeddings)
            
            # Persist if configured
            if self.config.enable_persistence and len(embeddings) % self.config.backup_interval == 0:
                self._save_to_disk()
            
            logger.info(f"Added {len(embeddings)} embeddings to memory store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using cosine similarity"""
        start_time = time.time()
        
        try:
            if query.query_embedding is None:
                raise ValueError("Query embedding required for memory store search")

            self._rebuild_matrix()
            
            if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
                return []
            
            # Ensure query embedding has correct dimensions
            query_embedding = np.array(query.query_embedding)
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            if self.embeddings_matrix.shape[1] != query_embedding.shape[0]:
                logger.error(f"Dimension mismatch: matrix has {self.embeddings_matrix.shape[1]} dimensions, query has {query_embedding.shape[0]}")
                return []

            # Calculate similarities
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []
            
            if self.config.distance_metric == "cosine":
                # Cosine similarity
                embeddings_norm = np.linalg.norm(self.embeddings_matrix, axis=1)
                valid_indices = embeddings_norm > 0
                
                if not np.any(valid_indices):
                    return []
                
                similarities = np.zeros(len(self.embeddings_matrix))
                similarities[valid_indices] = np.dot(
                    self.embeddings_matrix[valid_indices], 
                    query_embedding
                ) / (embeddings_norm[valid_indices] * query_norm)
                
            elif self.config.distance_metric == "dot_product":
                similarities = np.dot(self.embeddings_matrix, query_embedding)
                
            else:  # euclidean (convert to similarity)
                distances = np.linalg.norm(
                    self.embeddings_matrix - query_embedding, axis=1
                )
                similarities = 1 / (1 + distances)  # Convert distance to similarity
            
            # Apply score threshold
            valid_indices = similarities >= query.score_threshold
            
            if not np.any(valid_indices):
                return []
            
            # Get top k results
            valid_similarities = similarities[valid_indices]
            valid_embedding_indices = np.where(valid_indices)[0]
            
            top_indices = np.argsort(valid_similarities)[-query.top_k:][::-1]
            
            results = []
            for idx in top_indices:
                embedding_idx = valid_embedding_indices[idx]
                chunk_id = self.embedding_ids[embedding_idx]
                metadata = self.metadata.get(chunk_id, {})
                
                # Apply metadata filter
                if query.metadata_filter:
                    if not self._matches_filter(metadata, query.metadata_filter):
                        continue
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=metadata.get("content", ""),
                    score=float(valid_similarities[idx]),
                    metadata=metadata,
                    embedding=self.embeddings[chunk_id] if query.include_embeddings else None
                )
                results.append(result)
            
            search_time = time.time() - start_time
            self.update_stats(search_time)
            
            logger.debug(f"Memory search returned {len(results)} results in {search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def delete_by_id(self, ids: List[str]) -> bool:
        """Delete embeddings by ID"""
        try:
            deleted_count = 0
            for chunk_id in ids:
                if chunk_id in self.embeddings:
                    del self.embeddings[chunk_id]
                    del self.metadata[chunk_id]
                    deleted_count += 1
            
            if deleted_count > 0:
                self._needs_rebuild = True
                self.stats["total_embeddings"] -= deleted_count
            
            logger.info(f"Deleted {deleted_count} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return False
    
    def get_by_id(self, ids: List[str]) -> List[Optional[SearchResult]]:
        """Get embeddings by ID"""
        results = []
        for chunk_id in ids:
            if chunk_id in self.embeddings:
                metadata = self.metadata.get(chunk_id, {})
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=metadata.get("content", ""),
                    score=1.0,  # Perfect match for exact lookup
                    metadata=metadata,
                    embedding=self.embeddings[chunk_id]
                )
                results.append(result)
            else:
                results.append(None)
        
        return results
    
    def count(self) -> int:
        """Get total number of embeddings"""
        return len(self.embeddings)
    
    def clear(self) -> bool:
        """Clear all embeddings"""
        try:
            self.embeddings.clear()
            self.metadata.clear()
            self.embeddings_matrix = None
            self.embedding_ids = []
            self._needs_rebuild = True
            self.stats["total_embeddings"] = 0
            
            logger.info("Cleared all embeddings from memory store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            return False
    
    def _save_to_disk(self):
        """Save store to disk"""
        try:
            data = {
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
                "metadata": self.metadata,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
            save_path = self.persist_path / "memory_store.json"
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved memory store to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
    
    def _load_from_disk(self):
        """Load store from disk"""
        try:
            load_path = self.persist_path / "memory_store.json"
            if not load_path.exists():
                return
            
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            self.embeddings = {k: np.array(v) for k, v in data["embeddings"].items()}
            self.metadata = data["metadata"]
            self.stats = data.get("stats", self.stats)
            self._needs_rebuild = True
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings from disk")
            
        except Exception as e:
            logger.warning(f"Could not load from disk: {e}")

class ChromaVectorStore(VectorStore):
    """Chroma vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not available")
        
        # Initialize Chroma client
        if config.chroma_persist:
            self.client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(config.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"description": "Contract embeddings for semantic search"}
            )
        
        logger.info(f"Initialized Chroma collection: {config.collection_name}")
    
    def add_embeddings(self, embeddings: List[GeneratedEmbedding]) -> bool:
        """Add embeddings to Chroma"""
        try:
            # Prepare data for Chroma
            ids = []
            chroma_embeddings = []
            documents = []
            metadatas = []
            
            for emb in embeddings:
                ids.append(emb.chunk.chunk_id)
                chroma_embeddings.append(emb.embedding.tolist())
                documents.append(emb.chunk.content)
                
                # Chroma metadata must be flat and JSON serializable
                metadata = self._flatten_metadata(emb.chunk.to_embedding_dict()["metadata"])
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = self.config.batch_size
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                
                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=chroma_embeddings[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
            
            self.stats["total_embeddings"] += len(embeddings)
            logger.info(f"Added {len(embeddings)} embeddings to Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to Chroma: {e}")
            return False
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float]]:
        """Flatten nested metadata for Chroma compatibility"""
        flattened = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = ""):
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                
                if isinstance(v, dict):
                    flatten_dict(v, key)
                elif isinstance(v, (list, tuple)):
                    flattened[key] = str(v)  # Convert to string
                elif isinstance(v, (str, int, float, bool)):
                    flattened[key] = v
                else:
                    flattened[key] = str(v)
        
        flatten_dict(metadata)
        return flattened
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Chroma"""
        start_time = time.time()
        
        try:
            # Prepare query for Chroma
            query_embeddings = None
            if query.query_embedding is not None:
                query_embeddings = [query.query_embedding.tolist()]
            
            # Convert metadata filter to Chroma format
            where = None
            if query.metadata_filter:
                where = self._convert_metadata_filter(query.metadata_filter)
            
            # Execute search
            results = self.collection.query(
                query_embeddings=query_embeddings,
                query_texts=[query.query_text] if query.query_embedding is None else None,
                n_results=query.top_k,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"] if query.include_embeddings else ["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    # Chroma returns distances, convert to similarity scores
                    distance = results["distances"][0][i]
                    if self.config.distance_metric == "cosine":
                        score = 1 - distance  # Cosine distance to similarity
                    else:
                        score = 1 / (1 + distance)  # Distance to similarity
                    
                    if score < query.score_threshold:
                        continue
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=results["documents"][0][i],
                        score=score,
                        metadata=results["metadatas"][0][i],
                        embedding=np.array(results["embeddings"][0][i]) if query.include_embeddings and "embeddings" in results else None
                    )
                    search_results.append(result)
            
            search_time = time.time() - start_time
            self.update_stats(search_time)
            
            logger.debug(f"Chroma search returned {len(search_results)} results in {search_time:.3f}s")
            return search_results
            
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            return []
    
    def _convert_metadata_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata filter to Chroma format"""
        # Simple conversion - Chroma has specific filter syntax
        return filter_dict  # Simplified for now
    
    def delete_by_id(self, ids: List[str]) -> bool:
        """Delete embeddings by ID from Chroma"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings from Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete from Chroma: {e}")
            return False
    
    def get_by_id(self, ids: List[str]) -> List[Optional[SearchResult]]:
        """Get embeddings by ID from Chroma"""
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"]
            )
            
            search_results = []
            for i, chunk_id in enumerate(results["ids"]):
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][i],
                    score=1.0,
                    metadata=results["metadatas"][i],
                    embedding=np.array(results["embeddings"][i]) if "embeddings" in results else None
                )
                search_results.append(result)
            
            # Fill missing IDs with None
            result_dict = {r.chunk_id: r for r in search_results}
            return [result_dict.get(id_) for id_ in ids]
            
        except Exception as e:
            logger.error(f"Failed to get by ID from Chroma: {e}")
            return [None] * len(ids)
    
    def count(self) -> int:
        """Get total number of embeddings in Chroma"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count Chroma embeddings: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all embeddings from Chroma"""
        try:
            # Delete collection and recreate
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "Contract embeddings for semantic search"}
            )
            
            self.stats["total_embeddings"] = 0
            logger.info("Cleared all embeddings from Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Chroma store: {e}")
            return False

# Factory function
def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Factory function to create appropriate vector store"""
    
    if config.backend == "chroma" and CHROMA_AVAILABLE:
        return ChromaVectorStore(config)
    
    elif config.backend == "faiss" and FAISS_AVAILABLE:
        # TODO: Implement FAISS backend
        logger.warning("FAISS backend not yet implemented, falling back to memory")
        return MemoryVectorStore(config)
    
    else:
        # Default to memory store
        if config.backend != "memory":
            logger.warning(f"Backend '{config.backend}' not available, using memory store")
        return MemoryVectorStore(config)

# High-level interface
class VectorSearchEngine:
    """High-level vector search interface combining embedding generation and storage"""
    
    def __init__(self, embedding_generator, vector_store: VectorStore):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        
        logger.info(f"Initialized vector search engine with {vector_store.__class__.__name__}")
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for search"""
        try:
            all_embeddings = []
            
            for doc in documents:
                document_id = doc.get("id", f"doc_{len(all_embeddings)}")
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                # Create chunks
                chunks = self.embedding_generator.chunk_document(document_id, text, metadata)
                
                # Generate embeddings
                embeddings = self.embedding_generator.generate_embeddings(chunks)
                all_embeddings.extend(embeddings)
            
            # Add to vector store
            success = self.vector_store.add_embeddings(all_embeddings)
            
            if success:
                logger.info(f"Successfully indexed {len(documents)} documents ({len(all_embeddings)} chunks)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def search(self, query_text: str, **kwargs) -> List[SearchResult]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_chunks = self.embedding_generator.chunk_document("query", query_text)
            if not query_chunks:
                return []
            
            query_embeddings = self.embedding_generator.generate_embeddings(query_chunks[:1])
            if not query_embeddings:
                return []
            
            # Create search query
            search_query = SearchQuery(
                query_text=query_text,
                query_embedding=query_embeddings[0].embedding,
                **kwargs
            )
            
            # Execute search
            return self.vector_store.search(search_query)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_generator": self.embedding_generator.get_model_info()
        }