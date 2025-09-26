"""
Embedding Generation Pipeline

This module provides comprehensive embedding generation capabilities for contracts,
integrating with the knowledge graph to create semantically rich vector representations.

Key Features:
- Multiple embedding model support (OpenAI, Sentence-BERT, etc.)
- Intelligent chunking strategies (document, clause, entity-level)
- Batch processing for efficiency and scalability
- Integration with Module 5 knowledge graph
- Quality validation and monitoring
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime
import numpy as np

# Optional dependencies with fallback handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - using fallback embeddings")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False
    logging.warning("openai not available - will skip OpenAI embeddings")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = AutoModel = torch = None
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available - limited model support")

# Import knowledge graph components
try:
    from ..knowledge_graph.integration import KnowledgeGraphIntegrator
    from ..knowledge_graph.schema import NodeType, GraphNode
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KnowledgeGraphIntegrator = None
    NodeType = GraphNode = None
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logging.warning("Knowledge graph not available - limited integration")

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"  # Default to lightweight model
    model_type: str = "sentence_transformers"  # sentence_transformers, openai, transformers
    chunk_size: int = 512  # Maximum tokens per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    batch_size: int = 32  # Batch size for processing
    normalize_embeddings: bool = True  # L2 normalize embeddings
    include_metadata: bool = True  # Include rich metadata
    cache_embeddings: bool = True  # Cache for reuse
    
    # Model-specific settings
    openai_api_key: Optional[str] = None
    device: str = "cpu"  # cpu, cuda, mps
    max_retries: int = 3  # For API calls
    
@dataclass 
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str
    chunk_id: str
    chunk_type: str  # document, clause, entity, metadata
    chunk_index: int
    start_pos: int = 0
    end_pos: int = 0
    
    # Source information
    document_id: str = ""
    source_file: str = ""
    
    # Knowledge graph links
    graph_node_id: Optional[str] = None
    entity_type: Optional[str] = None
    
    # Content metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_embedding_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for embedding storage"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "graph_node_id": self.graph_node_id,
            "metadata": {
                **self.metadata,
                "start_pos": self.start_pos,
                "end_pos": self.end_pos,
                "source_file": self.source_file,
                "entity_type": self.entity_type
            }
        }

@dataclass
class GeneratedEmbedding:
    """Represents a generated embedding with metadata"""
    chunk: TextChunk
    embedding: np.ndarray
    model_name: str
    generation_time: datetime
    embedding_hash: str = ""
    
    def __post_init__(self):
        """Calculate embedding hash for deduplication"""
        if not self.embedding_hash:
            embedding_bytes = self.embedding.tobytes()
            self.embedding_hash = hashlib.md5(embedding_bytes).hexdigest()
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "embedding": self.embedding.tolist(),
            "model_name": self.model_name,
            "generation_time": self.generation_time.isoformat(),
            "embedding_hash": self.embedding_hash,
            "metadata": self.chunk.to_embedding_dict()["metadata"]
        }

class EmbeddingModel:
    """Base class for embedding models"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.model_info = {}
        
    def load_model(self):
        """Load the embedding model"""
        raise NotImplementedError
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError

class SentenceTransformerModel(EmbeddingModel):
    """Sentence-BERT embedding model"""
    
    def load_model(self):
        """Load Sentence-BERT model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        
        try:
            logger.info(f"Loading Sentence-BERT model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            
            self.model_info = {
                "model_name": self.config.model_name,
                "max_seq_length": getattr(self.model, 'max_seq_length', 512),
                "dimension": self.model.get_sentence_embedding_dimension(),
                "device": str(self.model.device)
            }
            
            logger.info(f"Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Sentence-BERT"""
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 50
            )
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model"""
    
    def load_model(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.config.openai_api_key
        
        self.model_info = {
            "model_name": self.config.model_name,
            "dimension": self._get_openai_dimension(),
            "provider": "openai"
        }
        
        logger.info(f"OpenAI model initialized: {self.model_info}")
    
    def _get_openai_dimension(self) -> int:
        """Get dimension for OpenAI models"""
        dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimension_map.get(self.config.model_name, 1536)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI"""
        if self.model is None:
            self.load_model()
        
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            for attempt in range(self.config.max_retries):
                try:
                    response = openai.Embedding.create(
                        model=self.config.model_name,
                        input=batch
                    )
                    
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"API call failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to generate embeddings after {self.config.max_retries} attempts: {e}")
                        raise
        
        embeddings_array = np.array(embeddings)
        
        if self.config.normalize_embeddings:
            embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        return embeddings_array
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._get_openai_dimension()

class FallbackEmbeddingModel(EmbeddingModel):
    """Fallback embedding model using simple TF-IDF or random vectors"""
    
    def load_model(self):
        """Initialize fallback model"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(max_features=384, stop_words='english')
            self.model_type = "tfidf"
            self.dimension = 384
            
        except ImportError:
            # Ultimate fallback - random embeddings (for testing only)
            logger.warning("Using random embeddings - for testing only!")
            self.model = None
            self.model_type = "random"
            self.dimension = 384
        
        self.model_info = {
            "model_name": "fallback",
            "type": self.model_type,
            "dimension": self.dimension
        }
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate fallback embeddings"""
        if self.model is None:
            self.load_model()
        
        if self.model_type == "tfidf":
            try:
                # For TF-IDF, we need to fit on all texts first
                embeddings = self.model.fit_transform(texts).toarray()
                
                # Ensure consistent dimensions by padding or truncating
                current_dim = embeddings.shape[1]
                if current_dim < self.dimension:
                    # Pad with zeros
                    padding = np.zeros((embeddings.shape[0], self.dimension - current_dim))
                    embeddings = np.hstack([embeddings, padding])
                elif current_dim > self.dimension:
                    # Truncate
                    embeddings = embeddings[:, :self.dimension]
                
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning(f"TF-IDF failed: {e}, falling back to random")
                # Fall back to random
                pass
        
        # Random embeddings (last resort)
        logger.warning("Generating random embeddings - for testing only!")
        np.random.seed(42)  # Deterministic for testing
        embeddings = np.random.normal(0, 1, (len(texts), self.dimension)).astype(np.float32)
        
        # Ensure consistent dimensions
        if embeddings.shape[1] != self.dimension:
            embeddings = np.random.normal(0, 1, (len(texts), self.dimension)).astype(np.float32)
        
        if self.config.normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return getattr(self, 'dimension', 384)

class EmbeddingGenerator:
    """Main embedding generation pipeline"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = self._initialize_model()
        self.stats = {
            "embeddings_generated": 0,
            "chunks_processed": 0,
            "processing_time": 0.0,
            "cache_hits": 0
        }
        
        # Cache for embeddings
        self._embedding_cache = {}
        self._graph_integrator = None
        
        if KNOWLEDGE_GRAPH_AVAILABLE:
            try:
                self._graph_integrator = KnowledgeGraphIntegrator()
            except Exception as e:
                logger.warning(f"Could not initialize graph integrator: {e}")
    
    def _initialize_model(self) -> EmbeddingModel:
        """Initialize the appropriate embedding model"""
        try:
            if self.config.model_type == "openai" and OPENAI_AVAILABLE:
                return OpenAIEmbeddingModel(self.config)
            
            elif self.config.model_type == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                return SentenceTransformerModel(self.config)
            
            else:
                logger.warning("Requested model not available, using fallback")
                return FallbackEmbeddingModel(self.config)
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.model_type} model: {e}")
            logger.info("Falling back to basic model")
            return FallbackEmbeddingModel(self.config)
    
    def chunk_document(self, document_text: str, document_id: str, 
                      metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Chunk document into embeddable pieces"""
        chunks = []
        
        # Simple sentence-based chunking for now
        sentences = self._split_into_sentences(document_text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > self.config.chunk_size:
                # Finalize current chunk
                if current_chunk:
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        chunk_type="document",
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        document_id=document_id,
                        metadata=metadata or {}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence
                current_start += len(current_chunk)
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                chunk_type="document",
                chunk_index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                document_id=document_id,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        # Basic sentence splitting - could be enhanced with nltk or spacy
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[GeneratedEmbedding]:
        """Generate embeddings for text chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        # Extract texts and check cache
        texts_to_process = []
        chunk_indices = []
        cached_embeddings = []
        
        for i, chunk in enumerate(chunks):
            cache_key = self._get_cache_key(chunk.content)
            
            if self.config.cache_embeddings and cache_key in self._embedding_cache:
                # Use cached embedding
                cached_embedding = self._embedding_cache[cache_key]
                embedding = GeneratedEmbedding(
                    chunk=chunk,
                    embedding=cached_embedding,
                    model_name=self.config.model_name,
                    generation_time=datetime.now()
                )
                cached_embeddings.append((i, embedding))
                self.stats["cache_hits"] += 1
            else:
                texts_to_process.append(chunk.content)
                chunk_indices.append(i)
        
        generated_embeddings = [None] * len(chunks)
        
        # Add cached embeddings
        for i, embedding in cached_embeddings:
            generated_embeddings[i] = embedding
        
        # Generate new embeddings
        if texts_to_process:
            try:
                embeddings_array = self.model.encode(texts_to_process)
                
                for i, (chunk_idx, embedding_vec) in enumerate(zip(chunk_indices, embeddings_array)):
                    chunk = chunks[chunk_idx]
                    
                    embedding = GeneratedEmbedding(
                        chunk=chunk,
                        embedding=embedding_vec,
                        model_name=self.config.model_name,
                        generation_time=datetime.now()
                    )
                    
                    generated_embeddings[chunk_idx] = embedding
                    
                    # Cache the embedding
                    if self.config.cache_embeddings:
                        cache_key = self._get_cache_key(chunk.content)
                        self._embedding_cache[cache_key] = embedding_vec
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["embeddings_generated"] += len(chunks)
        self.stats["chunks_processed"] += len(chunks)
        self.stats["processing_time"] += processing_time
        
        logger.info(f"Generated {len(chunks)} embeddings in {processing_time:.2f}s")
        
        return [emb for emb in generated_embeddings if emb is not None]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.config.model_name}:{text}".encode()).hexdigest()
    
    def process_contract_from_module5(self, document_id: str) -> List[GeneratedEmbedding]:
        """Process contract using Module 5 knowledge graph data"""
        if not self._graph_integrator:
            raise ValueError("Knowledge graph integrator not available")
        
        logger.info(f"Processing contract {document_id} with graph integration")
        
        try:
            # Get enhanced contract data from graph
            graph_data = self._graph_integrator.query_graph_for_insights(
                "contract_details", document_id=document_id
            )
            
            if "error" in graph_data:
                logger.warning(f"Could not get graph data: {graph_data['error']}")
                return []
            
            # Create chunks from different content types
            chunks = []
            
            # Document-level chunk
            if "full_text" in graph_data:
                doc_chunks = self.chunk_document(
                    graph_data["full_text"],
                    document_id,
                    {"content_type": "document", "source": "module5"}
                )
                chunks.extend(doc_chunks)
            
            # Clause-level chunks
            if "clauses" in graph_data:
                for clause in graph_data["clauses"]:
                    chunk = TextChunk(
                        content=clause.get("text", ""),
                        chunk_id=f"{document_id}_clause_{clause.get('clause_id', 'unknown')}",
                        chunk_type="clause",
                        chunk_index=len(chunks),
                        document_id=document_id,
                        graph_node_id=clause.get("clause_id"),
                        metadata={
                            "content_type": "clause",
                            "clause_type": clause.get("type", "unknown"),
                            "risk_level": clause.get("risk", {}).get("risk_level", "unknown"),
                            "confidence": clause.get("confidence", 0.0)
                        }
                    )
                    chunks.append(chunk)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            logger.info(f"Generated {len(embeddings)} embeddings for contract {document_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to process contract {document_id}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "config": {
                "model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "chunk_size": self.config.chunk_size,
                "batch_size": self.config.batch_size
            },
            "model_info": getattr(self.model, 'model_info', {}),
            "dimension": self.model.get_dimension(),
            "stats": self.stats
        }
        return info
    
    def save_cache(self, cache_path: str):
        """Save embedding cache to disk"""
        try:
            cache_data = {
                "model_name": self.config.model_name,
                "cache": {k: v.tolist() for k, v in self._embedding_cache.items()}
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"Saved {len(self._embedding_cache)} cached embeddings to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, cache_path: str):
        """Load embedding cache from disk"""
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Only load cache if model matches
            if cache_data.get("model_name") == self.config.model_name:
                self._embedding_cache = {
                    k: np.array(v) for k, v in cache_data["cache"].items()
                }
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            else:
                logger.warning("Cache model mismatch, starting fresh")
                
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

# Factory function for easy instantiation
def create_embedding_generator(
    model_name: str = "all-MiniLM-L6-v2",
    model_type: str = "sentence_transformers",
    **kwargs
) -> EmbeddingGenerator:
    """Factory function to create embedding generator with sensible defaults"""
    
    config = EmbeddingConfig(
        model_name=model_name,
        model_type=model_type,
        **kwargs
    )
    
    return EmbeddingGenerator(config)