"""
Result Fusion Engine for Hybrid Retrieval

This module implements sophisticated algorithms for combining and ranking 
results from graph traversal and vector search to produce optimal legal 
document retrieval results.

Key Features:
- Multi-criteria score fusion
- Semantic similarity clustering
- Legal domain expertise integration
- Confidence calibration and ranking

Author: ContractSense Team
Date: 2025-09-25  
Version: 1.0.0
"""

import logging
import time
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_engine import SearchResult
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from enum import Enum

@dataclass
class FusionConfig:
    """Configuration for result fusion"""
    graph_weight: float = 0.6  # Higher weight for precise graph results
    vector_weight: float = 0.4  # Lower weight but adds semantic flexibility
    confidence_threshold: float = 0.5
    max_results: int = 20
    enable_diversity_filtering: bool = True
    diversity_threshold: float = 0.8
    boost_exact_matches: bool = True
    exact_match_boost: float = 0.3
    legal_domain_boost: bool = True
    domain_boost_factor: float = 0.2

class FusionStrategy(Enum):
    """Result fusion strategies"""
    WEIGHTED_AVERAGE = "weighted_average"
    RANK_FUSION = "rank_fusion"
    BORDA_COUNT = "borda_count"
    RECIPROCAL_RANK = "reciprocal_rank"
    HYBRID_CONFIDENCE = "hybrid_confidence"

class ResultFusion:
    """
    Result fusion engine that combines graph and vector search results
    using sophisticated ranking and scoring algorithms
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Legal domain knowledge for result boosting
        self._load_legal_domain_knowledge()
        
        # Fusion strategies
        self._load_fusion_strategies()
        
        # Statistics
        self.stats = {
            "fusions_performed": 0,
            "total_fusion_time": 0.0,
            "average_graph_results": 0.0,
            "average_vector_results": 0.0,
            "average_final_results": 0.0,
            "diversity_filtered": 0,
            "domain_boosted": 0
        }
        
        self.logger.info("Result fusion engine initialized")
    
    def _load_legal_domain_knowledge(self):
        """Load legal domain knowledge for result boosting"""
        
        # High-importance clause types (boost results containing these)
        self.high_importance_clauses = {
            'liability', 'indemnification', 'termination', 'confidentiality',
            'intellectual_property', 'payment', 'breach', 'force_majeure',
            'governing_law', 'dispute_resolution'
        }
        
        # Legal entity patterns (boost results containing these)
        self.legal_entity_patterns = {
            'corporation', 'corp', 'inc', 'llc', 'ltd', 'company',
            'partnership', 'firm', 'organization', 'entity'
        }
        
        # Monetary significance indicators
        self.monetary_indicators = {
            'million', 'billion', 'damages', 'penalty', 'fee', 'cost',
            'payment', 'compensation', 'settlement', 'fine'
        }
        
        # Legal precision indicators (favor exact legal language)
        self.precision_indicators = {
            'shall', 'must', 'required', 'obligated', 'bound',
            'covenant', 'warrant', 'represent', 'agree'
        }
    
    def _load_fusion_strategies(self):
        """Load different fusion strategies"""
        
        self.strategies = {
            FusionStrategy.WEIGHTED_AVERAGE: self._weighted_average_fusion,
            FusionStrategy.RANK_FUSION: self._rank_fusion,
            FusionStrategy.BORDA_COUNT: self._borda_count_fusion,
            FusionStrategy.RECIPROCAL_RANK: self._reciprocal_rank_fusion,
            FusionStrategy.HYBRID_CONFIDENCE: self._hybrid_confidence_fusion
        }
    
    async def fuse_results(self, graph_results: List['SearchResult'], 
                          vector_results: List['SearchResult'],
                          query, strategy: FusionStrategy = FusionStrategy.HYBRID_CONFIDENCE) -> List['SearchResult']:
        """
        Fuse graph and vector search results using the specified strategy
        
        Args:
            graph_results: Results from graph search
            vector_results: Results from vector search  
            query: Original search query for context
            strategy: Fusion strategy to use
            
        Returns:
            Fused and ranked list of SearchResult objects
        """
        from .hybrid_engine import SearchResult  # Avoid circular import
        
        start_time = time.time()
        
        try:
            self.logger.debug(
                f"Fusing results: {len(graph_results)} graph + {len(vector_results)} vector"
            )
            
            # Update statistics
            self.stats["average_graph_results"] = (
                (self.stats["average_graph_results"] * self.stats["fusions_performed"] + len(graph_results)) /
                (self.stats["fusions_performed"] + 1)
            )
            self.stats["average_vector_results"] = (
                (self.stats["average_vector_results"] * self.stats["fusions_performed"] + len(vector_results)) /
                (self.stats["fusions_performed"] + 1)
            )
            
            # Apply fusion strategy
            fusion_function = self.strategies[strategy]
            fused_results = await fusion_function(graph_results, vector_results, query)
            
            # Apply post-fusion processing
            fused_results = self._apply_legal_domain_boosting(fused_results, query)
            fused_results = self._apply_exact_match_boosting(fused_results, query)
            fused_results = self._calibrate_confidence_scores(fused_results)
            
            # Apply diversity filtering if enabled
            if self.config.enable_diversity_filtering:
                fused_results = self._apply_diversity_filtering(fused_results)
            
            # Final ranking and limiting
            fused_results = self._final_ranking(fused_results)
            final_results = fused_results[:self.config.max_results]
            
            # Update final statistics
            fusion_time = time.time() - start_time
            self.stats["fusions_performed"] += 1
            self.stats["total_fusion_time"] += fusion_time
            self.stats["average_final_results"] = (
                (self.stats["average_final_results"] * (self.stats["fusions_performed"] - 1) + len(final_results)) /
                self.stats["fusions_performed"]
            )
            
            self.logger.debug(
                f"Fusion completed: {len(final_results)} results in {fusion_time:.3f}s "
                f"using {strategy.value} strategy"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Result fusion failed: {e}")
            # Fallback: return combined results with basic scoring
            return self._fallback_fusion(graph_results, vector_results)
    
    async def _weighted_average_fusion(self, graph_results: List['SearchResult'], 
                                     vector_results: List['SearchResult'], query) -> List['SearchResult']:
        """Fusion using weighted average of graph and vector scores"""
        
        # Create combined result map
        result_map = {}
        
        # Process graph results
        for result in graph_results:
            key = self._create_result_key(result)
            result_map[key] = result
            result.final_score = result.graph_score * self.config.graph_weight
            result.confidence = result.graph_score
        
        # Process vector results
        for result in vector_results:
            key = self._create_result_key(result)
            if key in result_map:
                # Combine scores for overlapping results
                existing = result_map[key]
                combined_score = (
                    existing.graph_score * self.config.graph_weight +
                    result.vector_score * self.config.vector_weight
                )
                existing.final_score = combined_score
                existing.vector_score = result.vector_score
                existing.confidence = min(existing.confidence + result.confidence, 1.0)
                existing.explanation += f" | {result.explanation}"
            else:
                # Add vector-only result
                result.final_score = result.vector_score * self.config.vector_weight
                result.confidence = result.vector_score
                result_map[key] = result
        
        return list(result_map.values())
    
    async def _rank_fusion(self, graph_results: List['SearchResult'], 
                         vector_results: List['SearchResult'], query) -> List['SearchResult']:
        """Fusion using rank-based scoring"""
        
        # Sort results within each list
        graph_sorted = sorted(graph_results, key=lambda x: x.graph_score, reverse=True)
        vector_sorted = sorted(vector_results, key=lambda x: x.vector_score, reverse=True)
        
        # Create rank maps
        graph_ranks = {self._create_result_key(r): i for i, r in enumerate(graph_sorted)}
        vector_ranks = {self._create_result_key(r): i for i, r in enumerate(vector_sorted)}
        
        # Combine results
        result_map = {}
        
        for result in graph_results:
            key = self._create_result_key(result)
            graph_rank = graph_ranks[key]
            graph_score = 1.0 - (graph_rank / max(len(graph_results), 1))
            
            result.final_score = graph_score * self.config.graph_weight
            result.confidence = graph_score
            result_map[key] = result
        
        for result in vector_results:
            key = self._create_result_key(result)
            vector_rank = vector_ranks[key]
            vector_score = 1.0 - (vector_rank / max(len(vector_results), 1))
            
            if key in result_map:
                existing = result_map[key]
                existing.final_score += vector_score * self.config.vector_weight
                existing.vector_score = result.vector_score
                existing.confidence = min(existing.confidence + vector_score, 1.0)
                existing.explanation += f" | {result.explanation}"
            else:
                result.final_score = vector_score * self.config.vector_weight
                result.confidence = vector_score
                result_map[key] = result
        
        return list(result_map.values())
    
    async def _borda_count_fusion(self, graph_results: List['SearchResult'], 
                                vector_results: List['SearchResult'], query) -> List['SearchResult']:
        """Fusion using Borda count method"""
        
        # Create combined result set
        all_results = graph_results + vector_results
        unique_keys = set(self._create_result_key(r) for r in all_results)
        
        result_map = {}
        for result in all_results:
            key = self._create_result_key(result)
            if key not in result_map:
                result_map[key] = result
        
        # Calculate Borda scores
        graph_sorted = sorted(graph_results, key=lambda x: x.graph_score, reverse=True)
        vector_sorted = sorted(vector_results, key=lambda x: x.vector_score, reverse=True)
        
        for key in unique_keys:
            result = result_map[key]
            borda_score = 0.0
            
            # Graph Borda points
            for i, graph_result in enumerate(graph_sorted):
                if self._create_result_key(graph_result) == key:
                    graph_points = (len(graph_sorted) - i) / len(graph_sorted)
                    borda_score += graph_points * self.config.graph_weight
                    break
            
            # Vector Borda points  
            for i, vector_result in enumerate(vector_sorted):
                if self._create_result_key(vector_result) == key:
                    vector_points = (len(vector_sorted) - i) / len(vector_sorted)
                    borda_score += vector_points * self.config.vector_weight
                    break
            
            result.final_score = borda_score
            result.confidence = borda_score
        
        return list(result_map.values())
    
    async def _reciprocal_rank_fusion(self, graph_results: List['SearchResult'], 
                                    vector_results: List['SearchResult'], query) -> List['SearchResult']:
        """Fusion using reciprocal rank fusion (RRF)"""
        
        k = 60  # RRF parameter
        
        # Sort results
        graph_sorted = sorted(graph_results, key=lambda x: x.graph_score, reverse=True)
        vector_sorted = sorted(vector_results, key=lambda x: x.vector_score, reverse=True)
        
        # Calculate RRF scores
        result_map = {}
        
        # Process graph results
        for rank, result in enumerate(graph_sorted):
            key = self._create_result_key(result)
            rrf_score = 1.0 / (k + rank + 1) * self.config.graph_weight
            result.final_score = rrf_score
            result.confidence = rrf_score * len(graph_sorted)  # Normalize
            result_map[key] = result
        
        # Process vector results
        for rank, result in enumerate(vector_sorted):
            key = self._create_result_key(result)
            rrf_score = 1.0 / (k + rank + 1) * self.config.vector_weight
            
            if key in result_map:
                existing = result_map[key]
                existing.final_score += rrf_score
                existing.vector_score = result.vector_score
                existing.confidence += rrf_score * len(vector_sorted)
                existing.explanation += f" | {result.explanation}"
            else:
                result.final_score = rrf_score
                result.confidence = rrf_score * len(vector_sorted)
                result_map[key] = result
        
        return list(result_map.values())
    
    async def _hybrid_confidence_fusion(self, graph_results: List['SearchResult'], 
                                      vector_results: List['SearchResult'], query) -> List['SearchResult']:
        """Advanced fusion using hybrid confidence modeling"""
        
        result_map = {}
        
        # Process graph results with confidence modeling
        for result in graph_results:
            key = self._create_result_key(result)
            
            # Model graph confidence based on traversal depth and match quality
            graph_confidence = self._calculate_graph_confidence(result, query)
            result.confidence = graph_confidence
            result.final_score = graph_confidence * self.config.graph_weight
            
            result_map[key] = result
        
        # Process vector results with semantic confidence
        for result in vector_results:
            key = self._create_result_key(result)
            
            # Model vector confidence based on similarity and semantic coherence
            vector_confidence = self._calculate_vector_confidence(result, query)
            
            if key in result_map:
                existing = result_map[key]
                
                # Combine confidences using geometric mean for conservative fusion
                combined_confidence = math.sqrt(existing.confidence * vector_confidence)
                existing.confidence = combined_confidence
                existing.vector_score = result.vector_score
                
                # Final score uses combined confidence
                existing.final_score = (
                    existing.graph_score * self.config.graph_weight +
                    result.vector_score * self.config.vector_weight
                ) * combined_confidence
                
                existing.explanation += f" | {result.explanation}"
            else:
                result.confidence = vector_confidence
                result.final_score = result.vector_score * self.config.vector_weight * vector_confidence
                result_map[key] = result
        
        return list(result_map.values())
    
    def _calculate_graph_confidence(self, result: 'SearchResult', query) -> float:
        """Calculate confidence score for graph result"""
        
        confidence = result.graph_score
        
        # Boost for exact entity matches
        if hasattr(query, 'entities') and query.entities:
            for entity in query.entities:
                if entity.lower() in result.content.lower():
                    confidence = min(confidence + 0.1, 1.0)
        
        # Boost for high-importance clauses
        if result.clause_type in self.high_importance_clauses:
            confidence = min(confidence + 0.15, 1.0)
        
        # Boost for legal precision indicators
        content_lower = result.content.lower()
        precision_matches = sum(1 for indicator in self.precision_indicators 
                              if indicator in content_lower)
        if precision_matches > 0:
            confidence = min(confidence + precision_matches * 0.05, 1.0)
        
        return confidence
    
    def _calculate_vector_confidence(self, result: 'SearchResult', query) -> float:
        """Calculate confidence score for vector result"""
        
        # Base confidence from similarity score
        confidence = result.vector_score
        
        # Boost for semantic coherence (longer matching content)
        content_length = len(result.content)
        if content_length > 500:  # Longer content often more informative
            confidence = min(confidence + 0.1, 1.0)
        
        # Boost for legal domain terms
        content_lower = result.content.lower()
        domain_matches = sum(1 for indicator in self.legal_entity_patterns 
                           if indicator in content_lower)
        if domain_matches > 0:
            confidence = min(confidence + domain_matches * 0.05, 1.0)
        
        return confidence
    
    def _apply_legal_domain_boosting(self, results: List['SearchResult'], query) -> List['SearchResult']:
        """Apply legal domain knowledge boosting"""
        
        if not self.config.legal_domain_boost:
            return results
        
        boosted_count = 0
        
        for result in results:
            original_score = result.final_score
            boost_factor = 0.0
            
            content_lower = result.content.lower()
            
            # Boost for high-importance clause types
            if result.clause_type in self.high_importance_clauses:
                boost_factor += 0.2
            
            # Boost for monetary significance
            monetary_matches = sum(1 for indicator in self.monetary_indicators 
                                 if indicator in content_lower)
            if monetary_matches > 0:
                boost_factor += monetary_matches * 0.1
            
            # Boost for legal precision
            precision_matches = sum(1 for indicator in self.precision_indicators 
                                  if indicator in content_lower)
            if precision_matches > 0:
                boost_factor += precision_matches * 0.05
            
            if boost_factor > 0:
                result.final_score = min(
                    result.final_score * (1 + boost_factor * self.config.domain_boost_factor),
                    1.0
                )
                boosted_count += 1
                result.explanation += f" [Domain boost: +{boost_factor:.2f}]"
        
        self.stats["domain_boosted"] += boosted_count
        return results
    
    def _apply_exact_match_boosting(self, results: List['SearchResult'], query) -> List['SearchResult']:
        """Boost results with exact matches to query terms"""
        
        if not self.config.boost_exact_matches:
            return results
        
        query_terms = set(query.text.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            exact_matches = len(query_terms.intersection(content_terms))
            
            if exact_matches > 0:
                boost = min(exact_matches * self.config.exact_match_boost / len(query_terms), 0.3)
                result.final_score = min(result.final_score + boost, 1.0)
                result.explanation += f" [Exact matches: {exact_matches}]"
        
        return results
    
    def _calibrate_confidence_scores(self, results: List['SearchResult']) -> List['SearchResult']:
        """Calibrate confidence scores for better ranking"""
        
        if not results:
            return results
        
        scores = [r.final_score for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.1
        
        for result in results:
            # Z-score normalization with sigmoid transformation
            z_score = (result.final_score - mean_score) / max(std_score, 0.01)
            calibrated = 1.0 / (1.0 + np.exp(-z_score))
            
            result.confidence = calibrated
            result.final_score = calibrated
        
        return results
    
    def _apply_diversity_filtering(self, results: List['SearchResult']) -> List['SearchResult']:
        """Filter results to maintain diversity"""
        
        if len(results) <= 5:  # Don't filter if we have few results
            return results
        
        diverse_results = []
        used_documents = set()
        
        # Sort by score first
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        
        for result in sorted_results:
            # Check if we already have results from this document
            if result.document_id in used_documents:
                # Only include if significantly higher score or different clause type
                existing_from_doc = [r for r in diverse_results if r.document_id == result.document_id]
                if existing_from_doc:
                    best_existing = max(existing_from_doc, key=lambda x: x.final_score)
                    if (result.final_score > best_existing.final_score * 1.2 or 
                        result.clause_type != best_existing.clause_type):
                        diverse_results.append(result)
            else:
                diverse_results.append(result)
                used_documents.add(result.document_id)
        
        filtered_count = len(results) - len(diverse_results)
        self.stats["diversity_filtered"] += filtered_count
        
        return diverse_results
    
    def _final_ranking(self, results: List['SearchResult']) -> List['SearchResult']:
        """Apply final ranking with tie-breaking"""
        
        return sorted(results, key=lambda x: (
            x.final_score,
            x.confidence,
            len(x.content),  # Prefer longer content as tie-breaker
            -hash(x.document_id)  # Deterministic tie-breaking
        ), reverse=True)
    
    def _create_result_key(self, result: 'SearchResult') -> str:
        """Create unique key for result deduplication"""
        return f"{result.document_id}:{result.clause_id}:{result.clause_type}"
    
    def _fallback_fusion(self, graph_results: List['SearchResult'], 
                        vector_results: List['SearchResult']) -> List['SearchResult']:
        """Fallback fusion when main strategies fail"""
        
        all_results = []
        
        # Add graph results with boosted scores
        for result in graph_results:
            result.final_score = result.graph_score * 1.2  # Boost graph results
            all_results.append(result)
        
        # Add vector results
        for result in vector_results:
            result.final_score = result.vector_score
            all_results.append(result)
        
        # Simple deduplication and ranking
        unique_results = {}
        for result in all_results:
            key = self._create_result_key(result)
            if key not in unique_results or result.final_score > unique_results[key].final_score:
                unique_results[key] = result
        
        return sorted(unique_results.values(), key=lambda x: x.final_score, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        
        avg_fusion_time = 0.0
        if self.stats["fusions_performed"] > 0:
            avg_fusion_time = self.stats["total_fusion_time"] / self.stats["fusions_performed"]
        
        return {
            **self.stats,
            "average_fusion_time": avg_fusion_time,
            "config": {
                "graph_weight": self.config.graph_weight,
                "vector_weight": self.config.vector_weight,
                "confidence_threshold": self.config.confidence_threshold
            }
        }

# Factory function
def create_result_fusion(config: FusionConfig = None) -> ResultFusion:
    """Create and return a configured result fusion engine"""
    if config is None:
        config = FusionConfig()
    return ResultFusion(config)