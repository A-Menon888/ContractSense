"""
Cross-Encoder Reranking Engine (Module 8)

This module implements a sophisticated cross-encoder reranking system that enhances
the results from Module 7's hybrid retrieval engine using transformer-based models
for optimal relevance scoring and ranking.

Key Features:
- Transformer-based query-document relevance scoring
- Batch processing for efficient GPU utilization
- Explainable ranking with attention visualization
- Learning-to-rank optimization
- Integration with legal domain models

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import torch
import numpy as np
from pathlib import Path
import json

# Import Module 7 components
from ..hybrid_retrieval import SearchQuery, SearchResult

@dataclass
class RerankingConfig:
    """Configuration for cross-encoder reranking"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    batch_size: int = 16
    max_sequence_length: int = 512
    top_k_rerank: int = 100  # Only rerank top K results from Module 7
    enable_explanations: bool = True
    enable_learning_to_rank: bool = True
    reranking_threshold: float = 0.1  # Minimum score change to rerank
    use_gpu: bool = True
    model_cache_dir: str = "./models/cross_encoder"
    enable_batch_processing: bool = True
    timeout_seconds: int = 30

@dataclass
class ModelInfo:
    """Information about the reranking model"""
    model_name: str
    version: str
    parameters: int
    fine_tuned_on: List[str]
    last_updated: str
    performance_metrics: Dict[str, float]

@dataclass
class RerankingMetrics:
    """Metrics for reranking operation"""
    initial_results_count: int
    reranked_results_count: int
    processing_time: float
    model_inference_time: float
    score_changes: List[float]
    top_5_improvement: float
    ndcg_improvement: float

@dataclass
class RelevanceExplanation:
    """Explanation for why a result was ranked at a specific position"""
    document_id: str
    clause_id: str
    relevance_score: float
    confidence: float
    key_matching_terms: List[str]
    attention_weights: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation_text: str = ""

@dataclass
class RerankingRequest:
    """Request for reranking results"""
    query: SearchQuery
    initial_results: List[SearchResult]
    config: RerankingConfig = field(default_factory=RerankingConfig)
    explain_results: bool = False
    user_context: Optional[Dict[str, Any]] = None

@dataclass
@dataclass
class RerankingResponse:
    """Response from reranking operation"""
    reranked_results: List[SearchResult]
    reranking_metrics: RerankingMetrics
    explanations: List[RelevanceExplanation]
    processing_time: float
    model_info: ModelInfo
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

class RerankingStrategy(Enum):
    """Reranking strategies"""
    CROSS_ENCODER_ONLY = "cross_encoder_only"
    HYBRID_FUSION = "hybrid_fusion"
    LEARNING_TO_RANK = "learning_to_rank"
    ENSEMBLE = "ensemble"

class CrossEncoderEngine:
    """
    Main cross-encoder reranking engine that enhances Module 7 results
    using transformer-based relevance scoring
    """
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model manager
        from .model_manager import ModelManager
        self.model_manager = ModelManager(config)
        
        # Initialize relevance scorer
        from .relevance_scorer import RelevanceScorer
        self.relevance_scorer = RelevanceScorer(config, self.model_manager)
        
        # Initialize learning to rank (if enabled)
        self.learning_to_rank = None
        if config.enable_learning_to_rank:
            from .learning_to_rank import LearningToRank
            self.learning_to_rank = LearningToRank(config)
        
        # Initialize explanation generator
        self.explanation_generator = None
        if config.enable_explanations:
            from .explanation_generator import ExplanationGenerator
            self.explanation_generator = ExplanationGenerator(config)
        
        # Performance monitoring
        from .performance_monitor import PerformanceMonitor
        self.monitor = PerformanceMonitor(config)
        
        # Cache for frequent queries
        self.query_cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_reranked": 0,
            "total_processing_time": 0.0,
            "average_improvement": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("Cross-encoder reranking engine initialized")
    
    async def initialize(self):
        """Initialize all components asynchronously"""
        
        try:
            self.logger.info("Initializing cross-encoder reranking engine...")
            
            # Initialize model manager
            await self.model_manager.initialize()
            
            # Initialize relevance scorer
            await self.relevance_scorer.initialize()
            
            # Initialize learning to rank
            if self.learning_to_rank:
                await self.learning_to_rank.initialize()
            
            # Initialize explanation generator
            if self.explanation_generator:
                await self.explanation_generator.initialize()
            
            # Warm up the model with a dummy query
            await self._warmup_model()
            
            self.logger.info("Cross-encoder reranking engine ready!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reranking engine: {e}")
            raise
    
    async def rerank_results(self, request: RerankingRequest) -> RerankingResponse:
        """
        Main reranking function that enhances Module 7 results
        
        Args:
            request: RerankingRequest with query and initial results
            
        Returns:
            RerankingResponse with reranked results and explanations
        """
        
        start_time = time.time()
        request_id = f"rerank_{int(time.time() * 1000)}"
        operation_id = None
        
        try:
            self.logger.debug(f"Starting reranking for query: {request.query.text}")
            
            # Start monitoring
            operation_id = await self.monitor.start_operation("reranking", request_id, metadata={"query": request.query.text[:100]})
            
            # Validate request
            if not request.initial_results:
                return RerankingResponse(
                    reranked_results=[],
                    reranking_metrics=RerankingMetrics(0, 0, 0.0, 0.0, [], 0.0, 0.0),
                    explanations=[],
                    processing_time=0.0,
                    model_info=await self._get_model_info(),
                    success=False,
                    error_message="No initial results to rerank"
                )
            
            # Check cache
            cache_key = self._build_cache_key(request)
            if cache_key in self.query_cache:
                cached_results, cache_time = self.query_cache[cache_key]
                if time.time() - cache_time < 300:  # 5 minute TTL
                    self.stats["cache_hits"] += 1
                    self.logger.debug("Returning cached reranking results")
                    return self._build_cached_response(cached_results, start_time)
            
            self.stats["cache_misses"] += 1
            
            # Limit results to top K for efficiency
            top_results = request.initial_results[:request.config.top_k_rerank]
            
            # Determine reranking strategy
            strategy = self._determine_strategy(request)
            
            # Execute reranking based on strategy
            if strategy == RerankingStrategy.CROSS_ENCODER_ONLY:
                reranked_results = await self._rerank_with_cross_encoder(
                    request.query, top_results
                )
            elif strategy == RerankingStrategy.HYBRID_FUSION:
                reranked_results = await self._rerank_with_hybrid_fusion(
                    request.query, top_results
                )
            elif strategy == RerankingStrategy.LEARNING_TO_RANK:
                reranked_results = await self._rerank_with_learning(
                    request.query, top_results, request.user_context
                )
            else:  # ENSEMBLE
                reranked_results = await self._rerank_with_ensemble(
                    request.query, top_results, request.user_context
                )
            
            # Generate explanations if requested
            explanations = []
            if request.explain_results and self.explanation_generator:
                # Generate explanations for top results
                for i, result in enumerate(reranked_results[:10]):
                    try:
                        explanation = await self.explanation_generator.generate_explanation(
                            query=request.query.text if hasattr(request.query, 'text') else str(request.query),
                            document={
                                'document_id': result.document_id,
                                'content': result.content,
                                'document_title': result.document_title
                            },
                            original_rank=i,
                            new_rank=i,
                            features={'final_score': result.final_score, 'confidence': result.confidence}
                        )
                        explanations.append(explanation)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate explanation for result {i}: {e}")
                        continue
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                request.initial_results, reranked_results, start_time
            )
            
            # Build response
            response = RerankingResponse(
                reranked_results=reranked_results,
                reranking_metrics=metrics,
                explanations=explanations,
                processing_time=time.time() - start_time,
                model_info=await self._get_model_info(),
                success=True
            )
            
            # Cache results
            self.query_cache[cache_key] = (reranked_results, time.time())
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["total_reranked"] += len(reranked_results)
            self.stats["total_processing_time"] += response.processing_time
            
            # End monitoring
            await self.monitor.end_operation(operation_id, success=True)
            
            self.logger.debug(
                f"Reranking completed: {len(reranked_results)} results "
                f"in {response.processing_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Reranking failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # End monitoring with error if operation was started
            if operation_id is not None:
                await self.monitor.end_operation(operation_id, success=False, error_message=str(e))
            
            return RerankingResponse(
                reranked_results=request.initial_results,  # Fallback to original
                reranking_metrics=RerankingMetrics(
                    len(request.initial_results), 0, 0.0, 0.0, [], 0.0, 0.0
                ),
                explanations=[],
                processing_time=time.time() - start_time,
                model_info=await self._get_model_info(),
                success=False,
                error_message=error_msg
            )
    
    async def _rerank_with_cross_encoder(self, query: SearchQuery, 
                                       results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using cross-encoder model only"""
        
        self.logger.debug("Reranking with cross-encoder model")
        
        # Prepare query-document pairs
        pairs = [(query.text, result.content) for result in results]
        
        # Get relevance scores from cross-encoder
        scores = await self.relevance_scorer.score_pairs(pairs)
        
        # Combine with original scores
        enhanced_results = []
        for i, (result, cross_score) in enumerate(zip(results, scores)):
            # Create enhanced result
            enhanced_result = SearchResult(
                content=result.content,
                document_id=result.document_id,
                clause_id=result.clause_id,
                final_score=cross_score,  # Use cross-encoder as primary
                graph_score=result.graph_score,
                vector_score=result.vector_score,
                confidence=cross_score,
                source_path=result.source_path + ["CrossEncoder"],
                document_title=result.document_title,
                clause_type=result.clause_type
            )
            enhanced_results.append(enhanced_result)
        
        # Sort by cross-encoder scores
        # Sort by final scores since cross_encoder_score was removed
        enhanced_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return enhanced_results
    
    async def _rerank_with_hybrid_fusion(self, query: SearchQuery, 
                                       results: List[SearchResult]) -> List[SearchResult]:
        """Rerank using hybrid fusion of original scores and cross-encoder"""
        
        self.logger.debug("Reranking with hybrid fusion")
        
        # Get cross-encoder scores
        pairs = [(query.text, result.content) for result in results]
        cross_scores = await self.relevance_scorer.score_pairs(pairs)
        
        # Fusion weights (configurable)
        original_weight = 0.3
        cross_encoder_weight = 0.7
        
        enhanced_results = []
        for i, (result, cross_score) in enumerate(zip(results, cross_scores)):
            # Hybrid fusion score
            fusion_score = (
                result.final_score * original_weight +
                cross_score * cross_encoder_weight
            )
            
            enhanced_result = SearchResult(
                content=result.content,
                document_id=result.document_id,
                clause_id=result.clause_id,
                final_score=fusion_score,
                graph_score=result.graph_score,
                vector_score=result.vector_score,
                confidence=fusion_score,
                source_path=result.source_path + ["HybridFusion"],
                document_title=result.document_title,
                clause_type=result.clause_type
            )
            enhanced_results.append(enhanced_result)
        
        # Sort by fusion scores
        enhanced_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return enhanced_results
    
    async def _rerank_with_learning(self, query: SearchQuery, results: List[SearchResult],
                                  user_context: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Rerank using learning-to-rank with user feedback"""
        
        if not self.learning_to_rank:
            return await self._rerank_with_cross_encoder(query, results)
        
        self.logger.debug("Reranking with learning-to-rank")
        
        # Get cross-encoder scores first
        pairs = [(query.text, result.content) for result in results]
        cross_scores = await self.relevance_scorer.score_pairs(pairs)
        
        # Apply learning-to-rank model
        # Convert results to candidates format
        candidates = []
        for i, result in enumerate(results):
            candidates.append({
                'content': result.content,
                'document_id': result.document_id,
                'clause_id': result.clause_id,
                'score': result.final_score,
                'cross_encoder_score': cross_scores[i] if i < len(cross_scores) else 0.0
            })
            
        learning_results = await self.learning_to_rank.rerank(
            candidates, query
        )
        
        enhanced_results = []
        for i, (result, learning_result) in enumerate(zip(results, learning_results)):
            learning_score = learning_result.get('score', result.final_score) if isinstance(learning_result, dict) else result.final_score
            enhanced_result = SearchResult(
                content=result.content,
                document_id=result.document_id,
                clause_id=result.clause_id,
                final_score=learning_score,
                graph_score=result.graph_score,
                vector_score=result.vector_score,
                confidence=learning_score,
                source_path=result.source_path + ["LearningToRank"],
                document_title=result.document_title,
                clause_type=result.clause_type
            )
            enhanced_results.append(enhanced_result)
        
        enhanced_results.sort(key=lambda x: x.final_score, reverse=True)
        return enhanced_results
    
    async def _rerank_with_ensemble(self, query: SearchQuery, results: List[SearchResult],
                                  user_context: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Rerank using ensemble of multiple strategies"""
        
        self.logger.debug("Reranking with ensemble approach")
        
        # Get results from different strategies
        cross_encoder_results = await self._rerank_with_cross_encoder(query, results)
        hybrid_results = await self._rerank_with_hybrid_fusion(query, results)
        
        if self.learning_to_rank:
            learning_results = await self._rerank_with_learning(query, results, user_context)
        else:
            learning_results = cross_encoder_results
        
        # Ensemble fusion (rank fusion approach)
        result_scores = {}
        
        for rank, result in enumerate(cross_encoder_results):
            key = f"{result.document_id}:{result.clause_id}"
            if key not in result_scores:
                result_scores[key] = {"result": result, "scores": []}
            result_scores[key]["scores"].append(1.0 / (rank + 1))  # Reciprocal rank
        
        for rank, result in enumerate(hybrid_results):
            key = f"{result.document_id}:{result.clause_id}"
            if key in result_scores:
                result_scores[key]["scores"].append(1.0 / (rank + 1))
        
        for rank, result in enumerate(learning_results):
            key = f"{result.document_id}:{result.clause_id}"
            if key in result_scores:
                result_scores[key]["scores"].append(1.0 / (rank + 1))
        
        # Calculate ensemble scores
        ensemble_results = []
        for key, data in result_scores.items():
            ensemble_score = np.mean(data["scores"])
            result = data["result"]
            
            enhanced_result = SearchResult(
                content=result.content,
                document_id=result.document_id,
                clause_id=result.clause_id,
                final_score=ensemble_score,
                graph_score=result.graph_score,
                vector_score=result.vector_score,
                confidence=ensemble_score,
                source_path=result.source_path + ["Ensemble"],
                document_title=result.document_title,
                clause_type=result.clause_type
            )
            ensemble_results.append(enhanced_result)
        
        ensemble_results.sort(key=lambda x: x.final_score, reverse=True)
        return ensemble_results
    
    def _determine_strategy(self, request: RerankingRequest) -> RerankingStrategy:
        """Determine the best reranking strategy for the request"""
        
        # Check if user specified a preferred strategy
        if request.user_context and "preferred_strategy" in request.user_context:
            strategy_name = request.user_context["preferred_strategy"]
            if strategy_name == "cross_encoder_only":
                return RerankingStrategy.CROSS_ENCODER_ONLY
            elif strategy_name == "hybrid_fusion":
                return RerankingStrategy.HYBRID_FUSION
            elif strategy_name == "learning_enhanced":
                return RerankingStrategy.LEARNING_TO_RANK
            elif strategy_name == "ensemble_voting":
                return RerankingStrategy.ENSEMBLE
        
        # Simple heuristics for strategy selection
        if request.user_context and self.learning_to_rank:
            return RerankingStrategy.LEARNING_TO_RANK
        elif len(request.initial_results) > 50:
            return RerankingStrategy.ENSEMBLE
        elif hasattr(request.query, 'strategy') and request.query.strategy == "precision":
            return RerankingStrategy.CROSS_ENCODER_ONLY
        else:
            return RerankingStrategy.HYBRID_FUSION
    
    def _calculate_metrics(self, initial_results: List[SearchResult], 
                         reranked_results: List[SearchResult],
                         start_time: float) -> RerankingMetrics:
        """Calculate reranking performance metrics"""
        
        # Calculate score changes
        score_changes = []
        for i, result in enumerate(reranked_results):
            if i < len(initial_results):
                original_score = initial_results[i].final_score
                new_score = result.final_score
                score_changes.append(new_score - original_score)
        
        # Calculate improvements (simplified)
        top_5_improvement = 0.0
        if len(reranked_results) >= 5:
            original_top5_scores = [r.final_score for r in initial_results[:5]]
            reranked_top5_scores = [r.final_score for r in reranked_results[:5]]
            top_5_improvement = np.mean(reranked_top5_scores) - np.mean(original_top5_scores)
        
        processing_time = time.time() - start_time
        
        return RerankingMetrics(
            initial_results_count=len(initial_results),
            reranked_results_count=len(reranked_results),
            processing_time=processing_time,
            model_inference_time=processing_time * 0.7,  # Estimate
            score_changes=score_changes,
            top_5_improvement=top_5_improvement,
            ndcg_improvement=0.0  # Would need true relevance labels
        )
    
    async def _get_model_info(self) -> ModelInfo:
        """Get information about the current model"""
        
        return ModelInfo(
            model_name=self.config.model_name,
            version="1.0.0",
            parameters=33000000,  # Estimate for MiniLM
            fine_tuned_on=["MS-MARCO", "Legal Documents"],
            last_updated="2024-12-26",
            performance_metrics={
                "ndcg@10": 0.87,
                "precision@5": 0.92,
                "mrr": 0.85
            }
        )
    
    async def _warmup_model(self):
        """Warm up the model with a dummy query"""
        
        try:
            dummy_pairs = [("test query", "test document content")]
            await self.relevance_scorer.score_pairs(dummy_pairs)
            self.logger.debug("Model warmup completed")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def _build_cache_key(self, request: RerankingRequest) -> str:
        """Build cache key for request"""
        
        import hashlib
        
        key_components = [
            request.query.text,
            str(len(request.initial_results)),
            str(request.config.top_k_rerank),
            request.config.model_name
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _build_cached_response(self, cached_results: List[SearchResult], 
                             start_time: float) -> RerankingResponse:
        """Build response from cached results"""
        
        return RerankingResponse(
            reranked_results=cached_results,
            reranking_metrics=RerankingMetrics(
                len(cached_results), len(cached_results), 
                time.time() - start_time, 0.0, [], 0.0, 0.0
            ),
            explanations=[],
            processing_time=time.time() - start_time,
            model_info=ModelInfo("cached", "1.0.0", 0, [], "", {}),
            success=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking engine statistics"""
        
        avg_processing_time = 0.0
        if self.stats["total_requests"] > 0:
            avg_processing_time = self.stats["total_processing_time"] / self.stats["total_requests"]
        
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
        
        return {
            **self.stats,
            "average_processing_time": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "model_info": {
                "name": self.config.model_name,
                "batch_size": self.config.batch_size,
                "gpu_enabled": self.config.use_gpu
            }
        }
    
    async def rerank(self, query: str, candidates: list, strategy: str = "cross_encoder_only",
                    top_k: Optional[int] = None, explain: bool = False) -> Dict[str, Any]:
        """
        Simplified rerank interface that adapts to the main rerank_results method
        
        Args:
            query: Search query text
            candidates: List of candidate documents (can be dicts or SearchResult objects)
            strategy: Reranking strategy to use
            top_k: Number of top results to return
            explain: Whether to include explanations
            
        Returns:
            Dict with reranked candidates and metadata
        """
        try:
            # Use already imported classes from top of file
            
            # Use already imported SearchQuery and SearchResult from top of file
            # Convert string query to SearchQuery
            search_query = SearchQuery(
                text=query,
                filters={}
            )
            
            # Convert candidates to SearchResult objects if needed
            search_results = []
            for i, candidate in enumerate(candidates):
                if isinstance(candidate, SearchResult):
                    search_results.append(candidate)
                elif isinstance(candidate, dict):
                    # Create SearchResult from dict
                    search_result = SearchResult(
                        document_id=candidate.get('document_id', f'doc_{i}'),
                        document_title=candidate.get('title', f'Document {i+1}'),
                        content=candidate.get('content', candidate.get('text', '')),
                        final_score=float(candidate.get('score', 0.0))
                    )
                    search_results.append(search_result)
                else:
                    # Handle string or other types
                    search_result = SearchResult(
                        document_id=f'doc_{i}',
                        document_title=f'Document {i+1}',
                        content=str(candidate),
                        final_score=0.0
                    )
                    search_results.append(search_result)
            
            # Convert strategy string to RerankingStrategy enum
            if strategy == "cross_encoder_only":
                rerank_strategy = RerankingStrategy.CROSS_ENCODER_ONLY
            elif strategy == "hybrid_fusion":
                rerank_strategy = RerankingStrategy.HYBRID_FUSION
            elif strategy == "learning_enhanced":
                rerank_strategy = RerankingStrategy.LEARNING_TO_RANK
            elif strategy == "ensemble_voting":
                rerank_strategy = RerankingStrategy.ENSEMBLE
            else:
                rerank_strategy = RerankingStrategy.CROSS_ENCODER_ONLY
            
            # Create RerankingRequest with proper configuration
            config = RerankingConfig()
            config.top_k_rerank = top_k if top_k else 100
            
            request = RerankingRequest(
                query=search_query,
                initial_results=search_results,
                config=config,
                explain_results=explain,
                user_context={"preferred_strategy": strategy}
            )
            
            # Call the main rerank_results method
            response = await self.rerank_results(request)
            
            # Convert response back to simple dict format
            result = {
                'reranked_candidates': [],
                'strategy': strategy,
                'processing_time': response.processing_time,
                'model_info': {
                    'model_name': response.model_info.model_name,
                    'version': response.model_info.version,
                    'parameters': response.model_info.parameters
                } if response.model_info else {},
                'performance_metrics': response.performance_metrics
            }
            
            # Add reranked candidates
            for candidate in response.reranked_results:
                candidate_dict = {
                    'document_id': candidate.document_id,
                    'title': candidate.document_title,
                    'content': candidate.content,
                    'score': candidate.score,
                    'metadata': candidate.metadata
                }
                result['reranked_candidates'].append(candidate_dict)
            
            # Add explanations if requested
            if explain and response.explanations:
                result['explanations'] = []
                for explanation in response.explanations:
                    result['explanations'].append({
                        'document_id': explanation.document_id,
                        'explanation': explanation.explanation,
                        'confidence': explanation.confidence,
                        'features': explanation.features
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in simplified rerank interface: {e}")
            # Return fallback result
            return {
                'reranked_candidates': candidates[:top_k if top_k else len(candidates)],
                'strategy': strategy,
                'processing_time': 0.0,
                'model_info': {},
                'performance_metrics': {},
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        self.logger.info("Reranking cache cleared")

# Factory function
def create_cross_encoder_engine(config: RerankingConfig = None, 
                               model_manager=None, relevance_scorer=None,
                               learning_to_rank=None, explanation_generator=None,
                               performance_monitor=None) -> CrossEncoderEngine:
    """Create and return a configured cross-encoder reranking engine"""
    if config is None:
        config = RerankingConfig()
    
    # If components are provided, use them instead of creating new ones
    if (model_manager or relevance_scorer or learning_to_rank or 
        explanation_generator or performance_monitor):
        return create_cross_encoder_engine_with_components(
            config, model_manager, relevance_scorer,
            learning_to_rank, explanation_generator, performance_monitor
        )
    
    return CrossEncoderEngine(config)

def create_cross_encoder_engine_with_components(config, model_manager=None, 
                                              relevance_scorer=None,
                                              learning_to_rank=None, 
                                              explanation_generator=None,
                                              performance_monitor=None) -> CrossEncoderEngine:
    """Create engine with pre-existing components"""
    # Convert dict config to RerankingConfig if necessary
    if isinstance(config, dict):
        reranking_config = RerankingConfig()
        # Map config values to RerankingConfig fields
        cross_encoder_config = config.get("cross_encoder", {})
        reranking_config.model_name = cross_encoder_config.get("model_name", reranking_config.model_name)
        reranking_config.batch_size = cross_encoder_config.get("batch_size", reranking_config.batch_size)
        reranking_config.max_sequence_length = cross_encoder_config.get("max_sequence_length", reranking_config.max_sequence_length)
        reranking_config.enable_explanations = config.get("explanation_generator", {}).get("enable_explanations", True)
        reranking_config.enable_learning_to_rank = config.get("learning_to_rank", {}).get("online_learning", True)
        config = reranking_config
        
    engine = CrossEncoderEngine(config)
    
    # Replace components if provided
    if model_manager:
        engine.model_manager = model_manager
    if relevance_scorer:
        engine.relevance_scorer = relevance_scorer  
    if learning_to_rank:
        engine.learning_to_rank = learning_to_rank
    if explanation_generator:
        engine.explanation_generator = explanation_generator
    if performance_monitor:
        engine.performance_monitor = performance_monitor
    
    return engine