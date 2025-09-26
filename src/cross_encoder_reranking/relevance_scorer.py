"""
Relevance Scorer for Cross-Encoder Reranking

This module implements the relevance scoring logic using cross-encoder models
to compute query-document relevance scores for reranking.

Key Features:
- Efficient batch processing
- Score calibration and normalization
- Feature attribution for explainability
- Performance optimization

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class ScoringConfig:
    """Configuration for relevance scoring"""
    batch_size: int = 16
    max_sequence_length: int = 512
    score_calibration: bool = True
    enable_feature_attribution: bool = True
    use_threading: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30

@dataclass
class ScoringResult:
    """Result from relevance scoring"""
    scores: List[float]
    processing_time: float
    batch_count: int
    average_score: float
    calibrated_scores: Optional[List[float]] = None
    feature_attributions: Optional[List[Dict[str, float]]] = None

class RelevanceScorer:
    """
    Handles relevance scoring using cross-encoder models
    Provides efficient batch processing and score calibration
    """
    
    def __init__(self, config, model_manager):
        self.config = config
        self.scoring_config = ScoringConfig()
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Score calibration parameters (learned from training data)
        self.calibration_params = {
            "alpha": 1.0,  # Temperature scaling parameter
            "beta": 0.0,   # Bias term
            "min_score": 0.0,
            "max_score": 1.0
        }
        
        # Threading for parallel processing
        self.thread_pool = None
        if self.scoring_config.use_threading:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.scoring_config.max_workers
            )
        
        # Performance tracking
        self.stats = {
            "total_batches": 0,
            "total_pairs": 0,
            "total_time": 0.0,
            "average_batch_time": 0.0,
            "scores_computed": 0,
            "calibration_applied": 0
        }
        
        # Cache for frequent query-document pairs
        self.score_cache: Dict[str, Tuple[float, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info("Relevance scorer initialized")
    
    async def initialize(self):
        """Initialize the relevance scorer"""
        
        try:
            self.logger.info("Initializing relevance scorer...")
            
            # Load calibration parameters if available
            await self._load_calibration_params()
            
            # Warm up scorer
            await self._warmup_scorer()
            
            self.logger.info("Relevance scorer ready!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize relevance scorer: {e}")
            raise
    
    async def score_pairs(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score a list of query-document pairs
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        
        start_time = time.time()
        
        if not query_doc_pairs:
            return []
        
        try:
            self.logger.debug(f"Scoring {len(query_doc_pairs)} query-document pairs")
            
            # Check cache for known pairs
            cached_scores, uncached_pairs, cache_indices = self._check_cache(query_doc_pairs)
            
            # Score uncached pairs
            if uncached_pairs:
                new_scores = await self._score_pairs_batch(uncached_pairs)
                
                # Update cache
                for pair, score in zip(uncached_pairs, new_scores):
                    cache_key = self._create_cache_key(pair[0], pair[1])
                    self.score_cache[cache_key] = (score, time.time())
            else:
                new_scores = []
            
            # Combine cached and new scores
            final_scores = self._combine_scores(
                cached_scores, new_scores, cache_indices, len(query_doc_pairs)
            )
            
            # Apply score calibration if enabled
            if self.scoring_config.score_calibration:
                final_scores = self._calibrate_scores(final_scores)
                self.stats["calibration_applied"] += len(final_scores)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_pairs"] += len(query_doc_pairs)
            self.stats["total_time"] += processing_time
            self.stats["scores_computed"] += len(final_scores)
            
            self.logger.debug(
                f"Scored {len(query_doc_pairs)} pairs in {processing_time:.3f}s "
                f"(cache hits: {self.cache_hits}, misses: {self.cache_misses})"
            )
            
            return final_scores
            
        except Exception as e:
            self.logger.error(f"Scoring failed: {e}")
            # Return default scores as fallback
            return [0.5] * len(query_doc_pairs)
    
    async def _score_pairs_batch(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """Score pairs using batch processing"""
        
        batch_size = self.scoring_config.batch_size
        all_scores = []
        
        # Process in batches
        for i in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[i:i + batch_size]
            batch_start = time.time()
            
            try:
                # Get scores from model manager
                batch_scores = await self.model_manager.inference_batch(batch)
                all_scores.extend(batch_scores)
                
                # Update statistics
                batch_time = time.time() - batch_start
                self.stats["total_batches"] += 1
                self.stats["average_batch_time"] = (
                    (self.stats["average_batch_time"] * (self.stats["total_batches"] - 1) + batch_time) /
                    self.stats["total_batches"]
                )
                
                self.logger.debug(f"Batch {self.stats['total_batches']} processed in {batch_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Batch scoring failed: {e}")
                # Add default scores for failed batch
                all_scores.extend([0.5] * len(batch))
        
        return all_scores
    
    def _check_cache(self, query_doc_pairs: List[Tuple[str, str]]) -> Tuple[
        List[Optional[float]], List[Tuple[str, str]], List[int]
    ]:
        """Check cache for existing scores"""
        
        cached_scores = []
        uncached_pairs = []
        cache_indices = []
        
        for i, (query, doc) in enumerate(query_doc_pairs):
            cache_key = self._create_cache_key(query, doc)
            
            if cache_key in self.score_cache:
                score, timestamp = self.score_cache[cache_key]
                # Check if cache entry is still valid (5 minute TTL)
                if time.time() - timestamp < 300:
                    cached_scores.append(score)
                    self.cache_hits += 1
                else:
                    # Expired cache entry
                    del self.score_cache[cache_key]
                    cached_scores.append(None)
                    uncached_pairs.append((query, doc))
                    cache_indices.append(i)
                    self.cache_misses += 1
            else:
                cached_scores.append(None)
                uncached_pairs.append((query, doc))
                cache_indices.append(i)
                self.cache_misses += 1
        
        return cached_scores, uncached_pairs, cache_indices
    
    def _combine_scores(self, cached_scores: List[Optional[float]], 
                       new_scores: List[float], cache_indices: List[int],
                       total_pairs: int) -> List[float]:
        """Combine cached and newly computed scores"""
        
        final_scores = [0.0] * total_pairs
        new_score_idx = 0
        
        for i in range(total_pairs):
            if cached_scores[i] is not None:
                final_scores[i] = cached_scores[i]
            else:
                final_scores[i] = new_scores[new_score_idx]
                new_score_idx += 1
        
        return final_scores
    
    def _calibrate_scores(self, scores: List[float]) -> List[float]:
        """Apply score calibration to improve reliability"""
        
        calibrated = []
        
        for score in scores:
            # Temperature scaling with bias
            calibrated_score = (score / self.calibration_params["alpha"]) + self.calibration_params["beta"]
            
            # Apply sigmoid for proper probability range
            calibrated_score = 1.0 / (1.0 + math.exp(-calibrated_score))
            
            # Clip to valid range
            calibrated_score = max(
                self.calibration_params["min_score"],
                min(self.calibration_params["max_score"], calibrated_score)
            )
            
            calibrated.append(calibrated_score)
        
        return calibrated
    
    def _create_cache_key(self, query: str, doc: str) -> str:
        """Create cache key for query-document pair"""
        
        import hashlib
        
        # Truncate long texts for cache key
        query_truncated = query[:100] if len(query) > 100 else query
        doc_truncated = doc[:200] if len(doc) > 200 else doc
        
        combined = f"{query_truncated}|{doc_truncated}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _load_calibration_params(self):
        """Load score calibration parameters"""
        
        try:
            # In practice, these would be loaded from a trained calibration model
            # For now, using default values
            self.calibration_params = {
                "alpha": 1.2,  # Slightly sharpen the scores
                "beta": 0.1,   # Small positive bias
                "min_score": 0.0,
                "max_score": 1.0
            }
            
            self.logger.debug("Calibration parameters loaded")
            
        except Exception as e:
            self.logger.warning(f"Failed to load calibration params: {e}")
    
    async def _warmup_scorer(self):
        """Warm up the scorer with sample pairs"""
        
        try:
            sample_pairs = [
                ("sample query", "sample document content"),
                ("test question", "test answer content")
            ]
            
            await self.score_pairs(sample_pairs)
            self.logger.debug("Scorer warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Scorer warmup failed: {e}")
    
    async def score_with_attribution(self, query_doc_pairs: List[Tuple[str, str]]) -> Tuple[
        List[float], List[Dict[str, float]]
    ]:
        """
        Score pairs and provide feature attribution
        
        Returns:
            Tuple of (scores, attributions)
        """
        
        if not self.scoring_config.enable_feature_attribution:
            scores = await self.score_pairs(query_doc_pairs)
            return scores, [{}] * len(scores)
        
        try:
            # Get base scores
            scores = await self.score_pairs(query_doc_pairs)
            
            # Compute feature attributions (simplified approach)
            attributions = []
            for query, doc in query_doc_pairs:
                attribution = self._compute_simple_attribution(query, doc, scores[len(attributions)])
                attributions.append(attribution)
            
            return scores, attributions
            
        except Exception as e:
            self.logger.error(f"Attribution scoring failed: {e}")
            scores = await self.score_pairs(query_doc_pairs)
            return scores, [{}] * len(scores)
    
    def _compute_simple_attribution(self, query: str, doc: str, score: float) -> Dict[str, float]:
        """Compute simple feature attribution based on term overlap"""
        
        try:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            
            # Term overlap features
            overlap = query_terms.intersection(doc_terms)
            overlap_ratio = len(overlap) / max(len(query_terms), 1)
            
            # Length features
            query_length = len(query.split())
            doc_length = len(doc.split())
            length_ratio = min(query_length, doc_length) / max(query_length, doc_length, 1)
            
            # Simple attribution weights
            attribution = {
                "term_overlap": overlap_ratio * 0.4,
                "length_similarity": length_ratio * 0.2,
                "base_score": score * 0.4
            }
            
            # Normalize to sum to 1
            total = sum(attribution.values())
            if total > 0:
                attribution = {k: v / total for k, v in attribution.items()}
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Attribution computation failed: {e}")
            return {"unknown": 1.0}
    
    async def batch_score_with_metadata(self, query_doc_pairs: List[Tuple[str, str]],
                                      include_timing: bool = True) -> ScoringResult:
        """Score pairs and return detailed metadata"""
        
        start_time = time.time()
        
        try:
            scores = await self.score_pairs(query_doc_pairs)
            
            # Compute calibrated scores if different
            calibrated_scores = None
            if self.scoring_config.score_calibration:
                calibrated_scores = self._calibrate_scores(scores)
            
            # Compute feature attributions if enabled
            feature_attributions = None
            if self.scoring_config.enable_feature_attribution:
                _, feature_attributions = await self.score_with_attribution(query_doc_pairs)
            
            processing_time = time.time() - start_time
            
            return ScoringResult(
                scores=scores,
                processing_time=processing_time,
                batch_count=math.ceil(len(query_doc_pairs) / self.scoring_config.batch_size),
                average_score=np.mean(scores) if scores else 0.0,
                calibrated_scores=calibrated_scores,
                feature_attributions=feature_attributions
            )
            
        except Exception as e:
            self.logger.error(f"Batch scoring with metadata failed: {e}")
            return ScoringResult(
                scores=[0.5] * len(query_doc_pairs),
                processing_time=time.time() - start_time,
                batch_count=0,
                average_score=0.5
            )
    
    def update_calibration(self, true_scores: List[float], predicted_scores: List[float]):
        """Update calibration parameters based on feedback"""
        
        try:
            if len(true_scores) != len(predicted_scores) or len(true_scores) < 10:
                self.logger.warning("Insufficient data for calibration update")
                return
            
            # Simple temperature scaling update
            true_array = np.array(true_scores)
            pred_array = np.array(predicted_scores)
            
            # Compute optimal temperature using simple regression
            mean_true = np.mean(true_array)
            mean_pred = np.mean(pred_array)
            
            if mean_pred != 0:
                new_alpha = mean_true / mean_pred
                self.calibration_params["alpha"] = 0.9 * self.calibration_params["alpha"] + 0.1 * new_alpha
            
            # Update bias term
            bias = mean_true - mean_pred
            self.calibration_params["beta"] = 0.9 * self.calibration_params["beta"] + 0.1 * bias
            
            self.logger.info("Calibration parameters updated")
            
        except Exception as e:
            self.logger.error(f"Calibration update failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics"""
        
        cache_hit_rate = 0.0
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = self.cache_hits / total_cache_requests
        
        avg_pair_time = 0.0
        if self.stats["total_pairs"] > 0:
            avg_pair_time = self.stats["total_time"] / self.stats["total_pairs"]
        
        return {
            **self.stats,
            "cache_size": len(self.score_cache),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "average_pair_time": avg_pair_time,
            "calibration_params": self.calibration_params.copy()
        }
    
    def clear_cache(self):
        """Clear the score cache"""
        self.score_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Score cache cleared")
    
    def cleanup(self):
        """Clean up resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        self.clear_cache()
        self.logger.info("Relevance scorer cleanup completed")

# Factory function
def create_relevance_scorer(config, model_manager) -> RelevanceScorer:
    """Create and return a configured relevance scorer"""
    return RelevanceScorer(config, model_manager)