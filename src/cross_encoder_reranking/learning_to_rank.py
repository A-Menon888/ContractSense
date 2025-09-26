"""
Learning-to-Rank Implementation for Cross-Encoder Reranking

This module implements learning-to-rank algorithms to optimize ranking
performance using feedback data and user interactions.

Key Features:
- ListNet and RankNet algorithms
- Online learning with user feedback
- Feature importance learning
- Performance evaluation metrics

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import pickle
from collections import defaultdict, deque
import math
from sklearn.metrics import ndcg_score

@dataclass
class RankingFeatures:
    """Features for learning-to-rank"""
    cross_encoder_score: float
    bm25_score: float
    vector_similarity: float
    term_overlap: float
    document_length: int
    query_length: int
    position_bias: float
    historical_relevance: float = 0.0
    click_through_rate: float = 0.0
    additional_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class RankingExample:
    """Training example for learning-to-rank"""
    query_id: str
    document_id: str
    features: RankingFeatures
    relevance_label: float  # 0-4 scale or 0-1 binary
    user_feedback: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class LtrConfig:
    """Configuration for learning-to-rank"""
    algorithm: str = "listnet"  # "listnet", "ranknet", "lambdamart"
    learning_rate: float = 0.001
    hidden_size: int = 128
    dropout_rate: float = 0.2
    max_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 32
    feature_dim: int = 10  # Number of ranking features
    online_learning: bool = True
    feedback_weight: float = 0.3
    evaluation_metrics: List[str] = field(default_factory=lambda: ["ndcg@10", "map", "mrr"])

class ListNet(nn.Module):
    """
    ListNet neural network for learning-to-rank
    Uses listwise ranking loss
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Neural network layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        return self.layers(x).squeeze(-1)

class RankNet(nn.Module):
    """
    RankNet neural network for pairwise ranking
    Uses pairwise ranking loss
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Shared neural network for scoring
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass - returns scores"""
        return self.scorer(x).squeeze(-1)
    
    def pairwise_forward(self, x1, x2):
        """Forward pass for pairwise comparison"""
        score1 = self.scorer(x1).squeeze(-1)
        score2 = self.scorer(x2).squeeze(-1)
        return score1, score2

class LearningToRank:
    """
    Learning-to-rank implementation with multiple algorithms
    Supports online learning and user feedback integration
    """
    
    def __init__(self, config):
        self.config = config
        self.ltr_config = LtrConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on algorithm
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
        
        # Training data storage
        self.training_examples: List[RankingExample] = []
        self.validation_examples: List[RankingExample] = []
        
        # Online learning buffer
        self.feedback_buffer = deque(maxlen=10000)
        
        # Feature statistics for normalization
        self.feature_stats = {
            "means": np.zeros(self.ltr_config.feature_dim),
            "stds": np.ones(self.ltr_config.feature_dim),
            "mins": np.zeros(self.ltr_config.feature_dim),
            "maxs": np.ones(self.ltr_config.feature_dim)
        }
        
        # Performance tracking
        self.metrics_history = defaultdict(list)
        self.training_stats = {
            "epochs_trained": 0,
            "examples_processed": 0,
            "last_validation_score": 0.0,
            "best_validation_score": 0.0,
            "feedback_examples": 0
        }
        
        # Feature importance tracking
        self.feature_importance = np.zeros(self.ltr_config.feature_dim)
        
        self.logger.info(f"Learning-to-rank initialized with {self.ltr_config.algorithm}")
    
    def _initialize_model(self):
        """Initialize the ranking model"""
        
        try:
            if self.ltr_config.algorithm == "listnet":
                self.model = ListNet(
                    input_dim=self.ltr_config.feature_dim,
                    hidden_size=self.ltr_config.hidden_size,
                    dropout_rate=self.ltr_config.dropout_rate
                ).to(self.device)
                
            elif self.ltr_config.algorithm == "ranknet":
                self.model = RankNet(
                    input_dim=self.ltr_config.feature_dim,
                    hidden_size=self.ltr_config.hidden_size,
                    dropout_rate=self.ltr_config.dropout_rate
                ).to(self.device)
                
            else:
                raise ValueError(f"Unsupported algorithm: {self.ltr_config.algorithm}")
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.ltr_config.learning_rate
            )
            
            self.logger.info(f"Model initialized: {self.ltr_config.algorithm}")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    async def initialize(self):
        """Initialize the learning-to-rank system"""
        
        try:
            self.logger.info("Initializing learning-to-rank system...")
            
            # Load pre-trained model if available
            await self._load_model()
            
            # Load feature statistics
            await self._load_feature_stats()
            
            # Warm up the model
            await self._warmup_model()
            
            self.logger.info("Learning-to-rank system ready!")
            
        except Exception as e:
            self.logger.error(f"LTR initialization failed: {e}")
            raise
    
    async def rerank(self, candidates: List[Dict[str, Any]], 
                    query: str) -> List[Dict[str, Any]]:
        """
        Rerank candidates using the learned model
        
        Args:
            candidates: List of candidate documents with features
            query: Query string
            
        Returns:
            Reranked list of candidates
        """
        
        if not candidates:
            return candidates
        
        try:
            # Extract features for each candidate
            features_list = []
            for candidate in candidates:
                features = self._extract_features(candidate, query)
                features_list.append(features)
            
            # Convert to tensor and normalize
            features_tensor = self._features_to_tensor(features_list)
            features_tensor = self._normalize_features(features_tensor)
            
            # Get scores from model
            with torch.no_grad():
                scores = self.model(features_tensor)
                scores = scores.cpu().numpy()
            
            # Sort candidates by score (descending)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Return reranked candidates with scores
            reranked = []
            for candidate, score in scored_candidates:
                candidate_copy = candidate.copy()
                candidate_copy["ltr_score"] = float(score)
                reranked.append(candidate_copy)
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return candidates
    
    def _extract_features(self, candidate: Dict[str, Any], query: str) -> RankingFeatures:
        """Extract ranking features from candidate and query"""
        
        try:
            # Extract basic features
            cross_encoder_score = candidate.get("cross_encoder_score", 0.0)
            bm25_score = candidate.get("bm25_score", 0.0)
            vector_similarity = candidate.get("vector_similarity", 0.0)
            
            # Text analysis features
            document_text = candidate.get("text", "")
            query_terms = set(query.lower().split())
            doc_terms = set(document_text.lower().split())
            
            term_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            document_length = len(document_text.split())
            query_length = len(query.split())
            
            # Position bias (if available)
            position_bias = 1.0 / (candidate.get("original_rank", 1) + 1)
            
            # Historical features
            doc_id = candidate.get("document_id", "")
            historical_relevance = self._get_historical_relevance(doc_id, query)
            click_through_rate = self._get_click_through_rate(doc_id)
            
            # Additional features from candidate
            additional_features = {}
            for key, value in candidate.items():
                if key.startswith("feature_") and isinstance(value, (int, float)):
                    additional_features[key] = float(value)
            
            return RankingFeatures(
                cross_encoder_score=cross_encoder_score,
                bm25_score=bm25_score,
                vector_similarity=vector_similarity,
                term_overlap=term_overlap,
                document_length=document_length,
                query_length=query_length,
                position_bias=position_bias,
                historical_relevance=historical_relevance,
                click_through_rate=click_through_rate,
                additional_features=additional_features
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return RankingFeatures(
                cross_encoder_score=0.0,
                bm25_score=0.0,
                vector_similarity=0.0,
                term_overlap=0.0,
                document_length=0,
                query_length=0,
                position_bias=0.0
            )
    
    def _features_to_tensor(self, features_list: List[RankingFeatures]) -> torch.Tensor:
        """Convert features to tensor"""
        
        feature_arrays = []
        
        for features in features_list:
            feature_array = np.array([
                features.cross_encoder_score,
                features.bm25_score,
                features.vector_similarity,
                features.term_overlap,
                features.document_length / 1000.0,  # Normalize length
                features.query_length / 100.0,
                features.position_bias,
                features.historical_relevance,
                features.click_through_rate,
                sum(features.additional_features.values()) / max(len(features.additional_features), 1)
            ])
            
            feature_arrays.append(feature_array)
        
        return torch.tensor(np.array(feature_arrays), dtype=torch.float32, device=self.device)
    
    def _normalize_features(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize features using stored statistics"""
        
        try:
            means = torch.tensor(self.feature_stats["means"], device=self.device)
            stds = torch.tensor(self.feature_stats["stds"], device=self.device)
            
            # Z-score normalization
            normalized = (features_tensor - means) / (stds + 1e-8)
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Feature normalization failed: {e}")
            return features_tensor
    
    async def add_training_example(self, query: str, documents: List[Dict[str, Any]],
                                 relevance_labels: List[float], user_feedback: Optional[Dict] = None):
        """Add a training example"""
        
        try:
            query_id = f"query_{len(self.training_examples)}"
            
            for doc, label in zip(documents, relevance_labels):
                features = self._extract_features(doc, query)
                
                example = RankingExample(
                    query_id=query_id,
                    document_id=doc.get("document_id", f"doc_{len(self.training_examples)}"),
                    features=features,
                    relevance_label=label,
                    user_feedback=user_feedback
                )
                
                self.training_examples.append(example)
            
            # Update feature statistics
            self._update_feature_stats()
            
            # Online learning if enabled
            if self.ltr_config.online_learning and len(self.training_examples) % 50 == 0:
                await self._online_learning_step()
            
            self.logger.debug(f"Added training example with {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Adding training example failed: {e}")
    
    async def add_feedback(self, query: str, document_id: str, 
                          feedback_type: str, feedback_value: float):
        """Add user feedback for online learning"""
        
        try:
            feedback_data = {
                "query": query,
                "document_id": document_id,
                "feedback_type": feedback_type,  # "click", "like", "relevance"
                "feedback_value": feedback_value,  # 0-1 scale
                "timestamp": time.time()
            }
            
            self.feedback_buffer.append(feedback_data)
            self.training_stats["feedback_examples"] += 1
            
            # Process feedback if buffer is large enough
            if len(self.feedback_buffer) >= self.ltr_config.batch_size:
                await self._process_feedback_batch()
            
            self.logger.debug(f"Added feedback: {feedback_type} = {feedback_value}")
            
        except Exception as e:
            self.logger.error(f"Adding feedback failed: {e}")
    
    async def train(self, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the learning-to-rank model"""
        
        if len(self.training_examples) < 10:
            self.logger.warning("Insufficient training data")
            return {"error": "insufficient_data"}
        
        try:
            self.logger.info(f"Training LTR model with {len(self.training_examples)} examples")
            
            # Split data
            split_idx = int(len(self.training_examples) * (1 - validation_split))
            train_examples = self.training_examples[:split_idx]
            val_examples = self.training_examples[split_idx:]
            
            best_val_score = float('-inf')
            patience_counter = 0
            
            for epoch in range(self.ltr_config.max_epochs):
                # Training
                train_loss = await self._train_epoch(train_examples)
                
                # Validation
                val_metrics = await self._evaluate(val_examples)
                val_score = val_metrics.get("ndcg@10", 0.0)
                
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_ndcg={val_score:.4f}")
                
                # Early stopping
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    await self._save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.ltr_config.early_stopping_patience:
                        self.logger.info("Early stopping triggered")
                        break
                
                # Update metrics history
                self.metrics_history["train_loss"].append(train_loss)
                self.metrics_history["val_ndcg"].append(val_score)
            
            self.training_stats["epochs_trained"] = epoch + 1
            self.training_stats["best_validation_score"] = best_val_score
            
            return {
                "epochs_trained": epoch + 1,
                "best_val_score": best_val_score,
                "final_train_loss": train_loss
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"error": str(e)}
    
    async def _train_epoch(self, examples: List[RankingExample]) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Group examples by query
        query_groups = defaultdict(list)
        for example in examples:
            query_groups[example.query_id].append(example)
        
        # Process each query group
        for query_id, query_examples in query_groups.items():
            if len(query_examples) < 2:
                continue
                
            # Extract features and labels
            features_list = [ex.features for ex in query_examples]
            labels = [ex.relevance_label for ex in query_examples]
            
            # Convert to tensors
            features_tensor = self._features_to_tensor(features_list)
            features_tensor = self._normalize_features(features_tensor)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
            
            # Compute loss based on algorithm
            if self.ltr_config.algorithm == "listnet":
                loss = self._listnet_loss(features_tensor, labels_tensor)
            elif self.ltr_config.algorithm == "ranknet":
                loss = self._ranknet_loss(features_tensor, labels_tensor)
            else:
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _listnet_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ListNet loss"""
        
        # Get scores from model
        scores = self.model(features)
        
        # Compute probability distributions
        score_probs = torch.softmax(scores, dim=0)
        label_probs = torch.softmax(labels, dim=0)
        
        # KL divergence loss
        loss = -torch.sum(label_probs * torch.log(score_probs + 1e-8))
        
        return loss
    
    def _ranknet_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute RankNet pairwise loss"""
        
        batch_size = features.size(0)
        total_loss = 0.0
        pair_count = 0
        
        # Generate all pairs
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if labels[i] == labels[j]:
                    continue  # Skip equal relevance pairs
                
                # Determine which document should rank higher
                if labels[i] > labels[j]:
                    score_i, score_j = self.model.pairwise_forward(
                        features[i:i+1], features[j:j+1]
                    )
                    target = 1.0  # i should rank higher than j
                else:
                    score_j, score_i = self.model.pairwise_forward(
                        features[j:j+1], features[i:i+1]
                    )
                    target = 1.0  # j should rank higher than i
                
                # Sigmoid cross-entropy loss
                score_diff = score_i - score_j
                loss = torch.log(1 + torch.exp(-target * score_diff))
                
                total_loss += loss
                pair_count += 1
        
        return total_loss / max(pair_count, 1)
    
    async def _evaluate(self, examples: List[RankingExample]) -> Dict[str, float]:
        """Evaluate model performance"""
        
        self.model.eval()
        
        query_groups = defaultdict(list)
        for example in examples:
            query_groups[example.query_id].append(example)
        
        all_ndcg_scores = []
        all_map_scores = []
        all_mrr_scores = []
        
        with torch.no_grad():
            for query_id, query_examples in query_groups.items():
                if len(query_examples) < 2:
                    continue
                
                # Get features and true labels
                features_list = [ex.features for ex in query_examples]
                true_labels = np.array([ex.relevance_label for ex in query_examples])
                
                # Get predicted scores
                features_tensor = self._features_to_tensor(features_list)
                features_tensor = self._normalize_features(features_tensor)
                pred_scores = self.model(features_tensor).cpu().numpy()
                
                # Compute NDCG@10
                if len(true_labels) >= 2:
                    ndcg = ndcg_score([true_labels], [pred_scores], k=10)
                    all_ndcg_scores.append(ndcg)
                
                # Compute MAP and MRR
                map_score = self._compute_map(true_labels, pred_scores)
                mrr_score = self._compute_mrr(true_labels, pred_scores)
                
                all_map_scores.append(map_score)
                all_mrr_scores.append(mrr_score)
        
        return {
            "ndcg@10": np.mean(all_ndcg_scores) if all_ndcg_scores else 0.0,
            "map": np.mean(all_map_scores) if all_map_scores else 0.0,
            "mrr": np.mean(all_mrr_scores) if all_mrr_scores else 0.0
        }
    
    def _compute_map(self, true_labels: np.ndarray, pred_scores: np.ndarray) -> float:
        """Compute Mean Average Precision"""
        
        # Sort by predicted scores
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_labels = true_labels[sorted_indices]
        
        # Compute average precision
        num_relevant = np.sum(sorted_labels > 0)
        if num_relevant == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, label in enumerate(sorted_labels):
            if label > 0:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precision_sum += precision
        
        return precision_sum / num_relevant
    
    def _compute_mrr(self, true_labels: np.ndarray, pred_scores: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank"""
        
        # Sort by predicted scores
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_labels = true_labels[sorted_indices]
        
        # Find first relevant document
        for i, label in enumerate(sorted_labels):
            if label > 0:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _get_historical_relevance(self, document_id: str, query: str) -> float:
        """Get historical relevance score for document-query pair"""
        
        # Simple implementation - would use actual historical data in practice
        return 0.0
    
    def _get_click_through_rate(self, document_id: str) -> float:
        """Get click-through rate for document"""
        
        # Simple implementation - would use actual CTR data in practice
        return 0.0
    
    def _update_feature_stats(self):
        """Update feature statistics for normalization"""
        
        if len(self.training_examples) < 10:
            return
        
        # Extract features from recent examples
        features_list = []
        for example in self.training_examples[-1000:]:  # Use recent examples
            feature_array = np.array([
                example.features.cross_encoder_score,
                example.features.bm25_score,
                example.features.vector_similarity,
                example.features.term_overlap,
                example.features.document_length / 1000.0,
                example.features.query_length / 100.0,
                example.features.position_bias,
                example.features.historical_relevance,
                example.features.click_through_rate,
                sum(example.features.additional_features.values()) / max(len(example.features.additional_features), 1)
            ])
            features_list.append(feature_array)
        
        features_array = np.array(features_list)
        
        # Update statistics
        self.feature_stats["means"] = np.mean(features_array, axis=0)
        self.feature_stats["stds"] = np.std(features_array, axis=0) + 1e-8
        self.feature_stats["mins"] = np.min(features_array, axis=0)
        self.feature_stats["maxs"] = np.max(features_array, axis=0)
    
    async def _online_learning_step(self):
        """Perform one step of online learning"""
        
        if len(self.training_examples) < self.ltr_config.batch_size:
            return
        
        try:
            # Use recent examples for online learning
            recent_examples = self.training_examples[-self.ltr_config.batch_size:]
            
            # Mini-batch training
            train_loss = await self._train_epoch(recent_examples)
            
            self.logger.debug(f"Online learning step completed, loss: {train_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Online learning step failed: {e}")
    
    async def _process_feedback_batch(self):
        """Process accumulated feedback for learning"""
        
        try:
            # Convert feedback to training examples (simplified)
            for feedback in self.feedback_buffer:
                # Create pseudo training example from feedback
                # In practice, this would be more sophisticated
                pass
            
            # Clear processed feedback
            self.feedback_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Processing feedback batch failed: {e}")
    
    async def _load_model(self):
        """Load pre-trained model"""
        
        try:
            # In practice, load from checkpoint
            self.logger.debug("No pre-trained model found, using random initialization")
            
        except Exception as e:
            self.logger.warning(f"Model loading failed: {e}")
    
    async def _save_model(self):
        """Save current model"""
        
        try:
            # In practice, save to checkpoint
            self.logger.debug("Model saved (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
    
    async def _load_feature_stats(self):
        """Load feature statistics"""
        
        try:
            # In practice, load from saved stats
            self.logger.debug("Using default feature statistics")
            
        except Exception as e:
            self.logger.warning(f"Loading feature stats failed: {e}")
    
    async def _warmup_model(self):
        """Warm up the model with dummy data"""
        
        try:
            dummy_features = torch.randn(2, self.ltr_config.feature_dim, device=self.device)
            
            with torch.no_grad():
                _ = self.model(dummy_features)
            
            self.logger.debug("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning-to-rank statistics"""
        
        return {
            **self.training_stats,
            "training_examples": len(self.training_examples),
            "validation_examples": len(self.validation_examples),
            "feedback_buffer_size": len(self.feedback_buffer),
            "feature_stats": self.feature_stats.copy(),
            "metrics_history": dict(self.metrics_history)
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        if self.model:
            del self.model
        if self.optimizer:
            del self.optimizer
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.logger.info("Learning-to-rank cleanup completed")

# Factory function
def create_learning_to_rank(config) -> LearningToRank:
    """Create and return a configured learning-to-rank system"""
    return LearningToRank(config)