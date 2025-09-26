"""
Explanation Generator for Cross-Encoder Reranking

This module generates human-readable explanations for reranking decisions
using attention visualization, feature importance, and similarity analysis.

Key Features:
- Attention weight visualization
- Feature importance analysis
- Text similarity explanations
- Interactive explanation formats

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

@dataclass
class ExplanationConfig:
    """Configuration for explanation generation"""
    enable_attention_viz: bool = True
    enable_feature_importance: bool = True
    enable_text_highlighting: bool = True
    max_tokens_to_highlight: int = 20
    similarity_threshold: float = 0.7
    explanation_depth: str = "detailed"  # "basic", "detailed", "comprehensive"
    include_visualizations: bool = True
    output_format: str = "json"  # "json", "html", "markdown"

@dataclass
class TokenImportance:
    """Importance score for individual tokens"""
    token: str
    importance: float
    position: int
    context: str

@dataclass
class FeatureExplanation:
    """Explanation for individual features"""
    feature_name: str
    feature_value: float
    importance: float
    contribution: float
    description: str
    category: str = "unknown"

@dataclass
class SimilarityExplanation:
    """Explanation for similarity computation"""
    query_terms: List[str]
    document_terms: List[str]
    matching_terms: List[str]
    semantic_similarity: float
    lexical_similarity: float
    explanation_text: str

@dataclass
class ComprehensiveExplanation:
    """Complete explanation for a reranking decision"""
    query: str
    document_id: str
    document_title: str
    final_score: float
    original_rank: int
    new_rank: int
    
    # Component explanations
    feature_explanations: List[FeatureExplanation]
    similarity_explanation: SimilarityExplanation
    token_importance: List[TokenImportance]
    
    # Summary and reasoning
    primary_reason: str
    confidence: float
    alternative_explanations: List[str]
    
    # Visualizations (base64 encoded)
    attention_heatmap: Optional[str] = None
    feature_importance_chart: Optional[str] = None
    
    # Metadata
    explanation_time: float = field(default_factory=time.time)
    model_version: str = "1.0.0"

class AttentionVisualizer:
    """Visualizes attention weights from transformer models"""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_attention_heatmap(self, tokens: List[str], attention_weights: np.ndarray,
                               query_tokens: List[str]) -> str:
        """Create attention heatmap visualization"""
        
        try:
            if not self.config.include_visualizations:
                return ""
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Truncate tokens if too many
            max_tokens = 50
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                attention_weights = attention_weights[:max_tokens, :max_tokens]
            
            # Create heatmap
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True,
                square=True,
                ax=ax
            )
            
            ax.set_title('Attention Weights Heatmap')
            ax.set_xlabel('Tokens')
            ax.set_ylabel('Tokens')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"Attention heatmap creation failed: {e}")
            return ""
    
    def analyze_attention_patterns(self, tokens: List[str], attention_weights: np.ndarray,
                                 query_tokens: List[str]) -> List[TokenImportance]:
        """Analyze attention patterns to identify important tokens"""
        
        try:
            token_importance = []
            
            # Sum attention weights across all heads/layers
            if attention_weights.ndim > 2:
                attention_weights = np.mean(attention_weights, axis=0)
            
            # Calculate importance for each token
            for i, token in enumerate(tokens):
                # Average attention received by this token
                importance = np.mean(attention_weights[:, i])
                
                # Context around the token
                start_idx = max(0, i - 2)
                end_idx = min(len(tokens), i + 3)
                context = " ".join(tokens[start_idx:end_idx])
                
                token_importance.append(TokenImportance(
                    token=token,
                    importance=float(importance),
                    position=i,
                    context=context
                ))
            
            # Sort by importance
            token_importance.sort(key=lambda x: x.importance, reverse=True)
            
            return token_importance[:self.config.max_tokens_to_highlight]
            
        except Exception as e:
            self.logger.error(f"Attention analysis failed: {e}")
            return []

class FeatureAnalyzer:
    """Analyzes feature contributions and importance"""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature descriptions
        self.feature_descriptions = {
            "cross_encoder_score": "Semantic similarity score from transformer model",
            "bm25_score": "Traditional keyword-based relevance score",
            "vector_similarity": "Dense vector cosine similarity",
            "term_overlap": "Proportion of query terms found in document",
            "document_length": "Length of the document in tokens",
            "query_length": "Length of the query in tokens",
            "position_bias": "Original position in search results",
            "historical_relevance": "Past relevance for similar queries",
            "click_through_rate": "Historical click-through rate for this document"
        }
        
        # Feature categories
        self.feature_categories = {
            "cross_encoder_score": "semantic",
            "bm25_score": "lexical",
            "vector_similarity": "semantic",
            "term_overlap": "lexical",
            "document_length": "structural",
            "query_length": "structural",
            "position_bias": "positional",
            "historical_relevance": "behavioral",
            "click_through_rate": "behavioral"
        }
    
    def analyze_features(self, features: Dict[str, float], 
                        feature_importance: np.ndarray) -> List[FeatureExplanation]:
        """Analyze feature contributions"""
        
        explanations = []
        
        try:
            feature_names = list(features.keys())
            
            for i, (name, value) in enumerate(features.items()):
                importance = feature_importance[i] if i < len(feature_importance) else 0.0
                
                # Calculate contribution (importance * value)
                contribution = importance * value
                
                explanation = FeatureExplanation(
                    feature_name=name,
                    feature_value=value,
                    importance=importance,
                    contribution=contribution,
                    description=self.feature_descriptions.get(name, f"Feature: {name}"),
                    category=self.feature_categories.get(name, "unknown")
                )
                
                explanations.append(explanation)
            
            # Sort by contribution magnitude
            explanations.sort(key=lambda x: abs(x.contribution), reverse=True)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Feature analysis failed: {e}")
            return []
    
    def create_feature_importance_chart(self, explanations: List[FeatureExplanation]) -> str:
        """Create feature importance visualization"""
        
        try:
            if not self.config.include_visualizations or not explanations:
                return ""
            
            # Extract data for plotting
            names = [exp.feature_name for exp in explanations[:10]]  # Top 10
            contributions = [exp.contribution for exp in explanations[:10]]
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color bars based on positive/negative contribution
            colors = ['green' if c >= 0 else 'red' for c in contributions]
            
            bars = ax.barh(names, contributions, color=colors, alpha=0.7)
            
            ax.set_xlabel('Feature Contribution')
            ax.set_title('Feature Importance for Reranking Decision')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, contributions):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left' if width >= 0 else 'right',
                       va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"Feature importance chart creation failed: {e}")
            return ""

class TextAnalyzer:
    """Analyzes text similarity and provides explanations"""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_similarity(self, query: str, document: str) -> SimilarityExplanation:
        """Analyze query-document similarity"""
        
        try:
            # Tokenize and normalize
            query_tokens = self._tokenize_and_normalize(query)
            doc_tokens = self._tokenize_and_normalize(document)
            
            # Find exact matches
            query_set = set(query_tokens)
            doc_set = set(doc_tokens)
            matching_terms = list(query_set.intersection(doc_set))
            
            # Calculate lexical similarity
            lexical_sim = len(matching_terms) / max(len(query_set), 1)
            
            # Simple semantic similarity (would use embeddings in practice)
            semantic_sim = self._compute_semantic_similarity(query_tokens, doc_tokens)
            
            # Generate explanation text
            explanation_text = self._generate_similarity_explanation(
                query_tokens, doc_tokens, matching_terms, lexical_sim, semantic_sim
            )
            
            return SimilarityExplanation(
                query_terms=query_tokens,
                document_terms=doc_tokens[:50],  # Limit for display
                matching_terms=matching_terms,
                semantic_similarity=semantic_sim,
                lexical_similarity=lexical_sim,
                explanation_text=explanation_text
            )
            
        except Exception as e:
            self.logger.error(f"Similarity analysis failed: {e}")
            return SimilarityExplanation(
                query_terms=[],
                document_terms=[],
                matching_terms=[],
                semantic_similarity=0.0,
                lexical_similarity=0.0,
                explanation_text="Analysis failed"
            )
    
    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        
        # Simple tokenization (would use proper tokenizer in practice)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Remove short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
        
        return tokens
    
    def _compute_semantic_similarity(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute semantic similarity (simplified)"""
        
        # Simple approach: use word overlap with partial matching
        # In practice, would use embeddings
        
        total_similarity = 0.0
        comparison_count = 0
        
        for q_token in query_tokens:
            best_match = 0.0
            for d_token in doc_tokens:
                # Simple string similarity
                sim = self._string_similarity(q_token, d_token)
                best_match = max(best_match, sim)
            
            total_similarity += best_match
            comparison_count += 1
        
        return total_similarity / max(comparison_count, 1)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity using edit distance"""
        
        if s1 == s2:
            return 1.0
        
        # Simple Jaccard similarity on character n-grams
        n = 3
        s1_grams = set(s1[i:i+n] for i in range(len(s1)-n+1))
        s2_grams = set(s2[i:i+n] for i in range(len(s2)-n+1))
        
        if not s1_grams and not s2_grams:
            return 1.0
        
        intersection = s1_grams.intersection(s2_grams)
        union = s1_grams.union(s2_grams)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_similarity_explanation(self, query_tokens: List[str], doc_tokens: List[str],
                                       matching_terms: List[str], lexical_sim: float,
                                       semantic_sim: float) -> str:
        """Generate human-readable similarity explanation"""
        
        explanations = []
        
        # Exact matches
        if matching_terms:
            if len(matching_terms) == 1:
                explanations.append(f"The document contains the query term '{matching_terms[0]}'")
            else:
                explanations.append(f"The document contains {len(matching_terms)} query terms: {', '.join(matching_terms[:5])}")
        else:
            explanations.append("No exact term matches found")
        
        # Lexical similarity
        if lexical_sim > 0.5:
            explanations.append(f"High lexical overlap ({lexical_sim:.1%})")
        elif lexical_sim > 0.2:
            explanations.append(f"Moderate lexical overlap ({lexical_sim:.1%})")
        else:
            explanations.append(f"Low lexical overlap ({lexical_sim:.1%})")
        
        # Semantic similarity
        if semantic_sim > 0.7:
            explanations.append("Strong semantic similarity detected")
        elif semantic_sim > 0.4:
            explanations.append("Moderate semantic similarity")
        else:
            explanations.append("Limited semantic similarity")
        
        return ". ".join(explanations) + "."

class ExplanationGenerator:
    """
    Main class for generating comprehensive explanations
    """
    
    def __init__(self, config):
        self.config = config
        self.explanation_config = ExplanationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.attention_visualizer = AttentionVisualizer(self.explanation_config)
        self.feature_analyzer = FeatureAnalyzer(self.explanation_config)
        self.text_analyzer = TextAnalyzer(self.explanation_config)
        
        # Statistics
        self.stats = {
            "explanations_generated": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "visualizations_created": 0
        }
        
        self.logger.info("Explanation generator initialized")
    
    async def initialize(self):
        """Initialize the explanation generator"""
        
        try:
            self.logger.info("Initializing explanation generator...")
            
            # Set up matplotlib for headless operation
            plt.switch_backend('Agg')
            
            self.logger.info("Explanation generator ready!")
            
        except Exception as e:
            self.logger.error(f"Explanation generator initialization failed: {e}")
            raise
    
    async def generate_explanation(self, query: str, document: Dict[str, Any],
                                 original_rank: int, new_rank: int,
                                 features: Dict[str, float],
                                 feature_importance: Optional[np.ndarray] = None,
                                 attention_weights: Optional[np.ndarray] = None) -> ComprehensiveExplanation:
        """Generate comprehensive explanation for reranking decision"""
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"Generating explanation for document {document.get('document_id', 'unknown')}")
            
            # Extract basic information
            document_id = document.get("document_id", "unknown")
            document_title = document.get("title", "Untitled")
            document_text = document.get("text", "")
            final_score = document.get("ltr_score", document.get("cross_encoder_score", 0.0))
            
            # Analyze features
            feature_explanations = []
            if self.explanation_config.enable_feature_importance and feature_importance is not None:
                feature_explanations = self.feature_analyzer.analyze_features(features, feature_importance)
            
            # Analyze text similarity
            similarity_explanation = self.text_analyzer.analyze_similarity(query, document_text)
            
            # Analyze attention patterns
            token_importance = []
            if self.explanation_config.enable_attention_viz and attention_weights is not None:
                # Tokenize document (simplified)
                tokens = document_text.split()[:100]  # Limit tokens
                query_tokens = query.split()
                
                token_importance = self.attention_visualizer.analyze_attention_patterns(
                    tokens, attention_weights, query_tokens
                )
            
            # Generate primary reason
            primary_reason = self._determine_primary_reason(
                feature_explanations, similarity_explanation, original_rank, new_rank
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(feature_explanations, similarity_explanation)
            
            # Generate alternative explanations
            alternative_explanations = self._generate_alternatives(feature_explanations, similarity_explanation)
            
            # Create visualizations
            attention_heatmap = ""
            feature_importance_chart = ""
            
            if self.explanation_config.include_visualizations:
                if attention_weights is not None:
                    tokens = document_text.split()[:50]
                    query_tokens = query.split()
                    attention_heatmap = self.attention_visualizer.create_attention_heatmap(
                        tokens, attention_weights, query_tokens
                    )
                    if attention_heatmap:
                        self.stats["visualizations_created"] += 1
                
                if feature_explanations:
                    feature_importance_chart = self.feature_analyzer.create_feature_importance_chart(
                        feature_explanations
                    )
                    if feature_importance_chart:
                        self.stats["visualizations_created"] += 1
            
            # Create comprehensive explanation
            explanation = ComprehensiveExplanation(
                query=query,
                document_id=document_id,
                document_title=document_title,
                final_score=final_score,
                original_rank=original_rank,
                new_rank=new_rank,
                feature_explanations=feature_explanations,
                similarity_explanation=similarity_explanation,
                token_importance=token_importance,
                primary_reason=primary_reason,
                confidence=confidence,
                alternative_explanations=alternative_explanations,
                attention_heatmap=attention_heatmap,
                feature_importance_chart=feature_importance_chart
            )
            
            # Update statistics
            generation_time = time.time() - start_time
            self.stats["explanations_generated"] += 1
            self.stats["total_generation_time"] += generation_time
            self.stats["average_generation_time"] = (
                self.stats["total_generation_time"] / self.stats["explanations_generated"]
            )
            
            self.logger.debug(f"Explanation generated in {generation_time:.3f}s")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            
            # Return minimal explanation on error
            return ComprehensiveExplanation(
                query=query,
                document_id=document.get("document_id", "unknown"),
                document_title=document.get("title", "Untitled"),
                final_score=0.0,
                original_rank=original_rank,
                new_rank=new_rank,
                feature_explanations=[],
                similarity_explanation=SimilarityExplanation(
                    query_terms=[], document_terms=[], matching_terms=[],
                    semantic_similarity=0.0, lexical_similarity=0.0,
                    explanation_text="Explanation generation failed"
                ),
                token_importance=[],
                primary_reason="Analysis unavailable",
                confidence=0.0,
                alternative_explanations=[]
            )
    
    def _determine_primary_reason(self, feature_explanations: List[FeatureExplanation],
                                similarity_explanation: SimilarityExplanation,
                                original_rank: int, new_rank: int) -> str:
        """Determine the primary reason for reranking decision"""
        
        try:
            rank_change = new_rank - original_rank
            
            if rank_change == 0:
                return "Document maintained its original position"
            
            # Find most important contributing factor
            if feature_explanations:
                top_feature = feature_explanations[0]
                
                if rank_change > 0:  # Promoted
                    if top_feature.feature_name == "cross_encoder_score":
                        return f"Promoted due to strong semantic relevance (score: {top_feature.feature_value:.3f})"
                    elif top_feature.feature_name == "bm25_score":
                        return f"Promoted due to keyword relevance (BM25: {top_feature.feature_value:.3f})"
                    elif top_feature.feature_name == "term_overlap":
                        return f"Promoted due to high query term overlap ({top_feature.feature_value:.1%})"
                    else:
                        return f"Promoted primarily by {top_feature.feature_name}"
                else:  # Demoted
                    return f"Demoted due to lower {top_feature.feature_name} compared to other results"
            
            # Fallback to similarity analysis
            if similarity_explanation.matching_terms:
                return f"Reranked based on {len(similarity_explanation.matching_terms)} matching terms"
            
            return "Reranked based on overall relevance assessment"
            
        except Exception as e:
            self.logger.error(f"Primary reason determination failed: {e}")
            return "Reranking reason unavailable"
    
    def _calculate_confidence(self, feature_explanations: List[FeatureExplanation],
                            similarity_explanation: SimilarityExplanation) -> float:
        """Calculate confidence in the reranking decision"""
        
        try:
            confidence_factors = []
            
            # Feature-based confidence
            if feature_explanations:
                # Confidence based on top feature's contribution magnitude
                top_contribution = abs(feature_explanations[0].contribution)
                confidence_factors.append(min(top_contribution * 2, 1.0))
                
                # Consistency across features
                positive_features = sum(1 for f in feature_explanations[:5] if f.contribution > 0)
                total_features = min(len(feature_explanations), 5)
                consistency = abs(positive_features / total_features - 0.5) * 2
                confidence_factors.append(consistency)
            
            # Similarity-based confidence
            if similarity_explanation.matching_terms:
                term_confidence = min(len(similarity_explanation.matching_terms) / 3, 1.0)
                confidence_factors.append(term_confidence)
            
            similarity_confidence = (similarity_explanation.lexical_similarity + 
                                   similarity_explanation.semantic_similarity) / 2
            confidence_factors.append(similarity_confidence)
            
            # Overall confidence
            if confidence_factors:
                confidence = np.mean(confidence_factors)
                return min(max(confidence, 0.0), 1.0)
            
            return 0.5  # Default neutral confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _generate_alternatives(self, feature_explanations: List[FeatureExplanation],
                             similarity_explanation: SimilarityExplanation) -> List[str]:
        """Generate alternative explanations"""
        
        alternatives = []
        
        try:
            # Alternative feature-based explanations
            if len(feature_explanations) > 1:
                for feat in feature_explanations[1:4]:  # Top 3 alternatives
                    if abs(feat.contribution) > 0.1:
                        alternatives.append(
                            f"Could also be influenced by {feat.feature_name} "
                            f"(contribution: {feat.contribution:.3f})"
                        )
            
            # Alternative similarity explanations
            if similarity_explanation.semantic_similarity > 0.3:
                alternatives.append(
                    f"Semantic similarity ({similarity_explanation.semantic_similarity:.1%}) "
                    "suggests conceptual relevance"
                )
            
            if similarity_explanation.lexical_similarity > 0.2:
                alternatives.append(
                    f"Lexical overlap ({similarity_explanation.lexical_similarity:.1%}) "
                    "indicates keyword relevance"
                )
            
            return alternatives[:3]  # Limit to top 3
            
        except Exception as e:
            self.logger.error(f"Alternative generation failed: {e}")
            return []
    
    def format_explanation(self, explanation: ComprehensiveExplanation, 
                         format_type: str = "json") -> str:
        """Format explanation in requested format"""
        
        try:
            if format_type == "json":
                return self._format_as_json(explanation)
            elif format_type == "html":
                return self._format_as_html(explanation)
            elif format_type == "markdown":
                return self._format_as_markdown(explanation)
            else:
                return self._format_as_text(explanation)
                
        except Exception as e:
            self.logger.error(f"Explanation formatting failed: {e}")
            return f"Formatting error: {e}"
    
    def _format_as_json(self, explanation: ComprehensiveExplanation) -> str:
        """Format explanation as JSON"""
        
        # Convert to dictionary for JSON serialization
        explanation_dict = {
            "query": explanation.query,
            "document_id": explanation.document_id,
            "document_title": explanation.document_title,
            "scores": {
                "final_score": explanation.final_score,
                "confidence": explanation.confidence
            },
            "ranking": {
                "original_rank": explanation.original_rank,
                "new_rank": explanation.new_rank,
                "rank_change": explanation.new_rank - explanation.original_rank
            },
            "primary_reason": explanation.primary_reason,
            "feature_analysis": [
                {
                    "name": f.feature_name,
                    "value": f.feature_value,
                    "importance": f.importance,
                    "contribution": f.contribution,
                    "description": f.description,
                    "category": f.category
                } for f in explanation.feature_explanations
            ],
            "similarity_analysis": {
                "matching_terms": explanation.similarity_explanation.matching_terms,
                "lexical_similarity": explanation.similarity_explanation.lexical_similarity,
                "semantic_similarity": explanation.similarity_explanation.semantic_similarity,
                "explanation": explanation.similarity_explanation.explanation_text
            },
            "important_tokens": [
                {
                    "token": t.token,
                    "importance": t.importance,
                    "position": t.position,
                    "context": t.context
                } for t in explanation.token_importance
            ],
            "alternatives": explanation.alternative_explanations,
            "metadata": {
                "explanation_time": explanation.explanation_time,
                "model_version": explanation.model_version,
                "has_attention_viz": bool(explanation.attention_heatmap),
                "has_feature_chart": bool(explanation.feature_importance_chart)
            }
        }
        
        return json.dumps(explanation_dict, indent=2)
    
    def _format_as_markdown(self, explanation: ComprehensiveExplanation) -> str:
        """Format explanation as Markdown"""
        
        md_lines = [
            f"# Reranking Explanation",
            f"",
            f"**Query:** {explanation.query}",
            f"**Document:** {explanation.document_title}",
            f"**Final Score:** {explanation.final_score:.3f}",
            f"**Confidence:** {explanation.confidence:.1%}",
            f"",
            f"## Primary Reason",
            f"{explanation.primary_reason}",
            f"",
            f"## Feature Analysis",
        ]
        
        for feat in explanation.feature_explanations[:5]:
            md_lines.append(f"- **{feat.feature_name}**: {feat.feature_value:.3f} "
                          f"(contribution: {feat.contribution:.3f})")
        
        md_lines.extend([
            f"",
            f"## Text Similarity",
            f"{explanation.similarity_explanation.explanation_text}",
            f"",
            f"**Matching Terms:** {', '.join(explanation.similarity_explanation.matching_terms)}",
        ])
        
        if explanation.alternative_explanations:
            md_lines.extend([
                f"",
                f"## Alternative Explanations",
            ])
            for alt in explanation.alternative_explanations:
                md_lines.append(f"- {alt}")
        
        return "\n".join(md_lines)
    
    def _format_as_text(self, explanation: ComprehensiveExplanation) -> str:
        """Format explanation as plain text"""
        
        text_lines = [
            f"Reranking Explanation for: {explanation.document_title}",
            f"Query: {explanation.query}",
            f"Score: {explanation.final_score:.3f} (Confidence: {explanation.confidence:.1%})",
            f"",
            f"Primary Reason: {explanation.primary_reason}",
            f"",
            f"Key Features:",
        ]
        
        for feat in explanation.feature_explanations[:3]:
            text_lines.append(f"  - {feat.feature_name}: {feat.feature_value:.3f}")
        
        text_lines.extend([
            f"",
            f"Text Analysis: {explanation.similarity_explanation.explanation_text}",
        ])
        
        return "\n".join(text_lines)
    
    def _format_as_html(self, explanation: ComprehensiveExplanation) -> str:
        """Format explanation as HTML (simplified)"""
        
        html = f"""
        <div class="reranking-explanation">
            <h2>Reranking Explanation</h2>
            <p><strong>Query:</strong> {explanation.query}</p>
            <p><strong>Document:</strong> {explanation.document_title}</p>
            <p><strong>Score:</strong> {explanation.final_score:.3f} 
               (Confidence: {explanation.confidence:.1%})</p>
            
            <h3>Primary Reason</h3>
            <p>{explanation.primary_reason}</p>
            
            <h3>Feature Analysis</h3>
            <ul>
        """
        
        for feat in explanation.feature_explanations[:5]:
            html += f"<li><strong>{feat.feature_name}:</strong> {feat.feature_value:.3f} " \
                   f"(contribution: {feat.contribution:.3f})</li>"
        
        html += f"""
            </ul>
            
            <h3>Text Similarity</h3>
            <p>{explanation.similarity_explanation.explanation_text}</p>
        </div>
        """
        
        return html
    
    def get_stats(self) -> Dict[str, Any]:
        """Get explanation generator statistics"""
        
        return self.stats.copy()
    
    def cleanup(self):
        """Clean up resources"""
        
        plt.close('all')
        self.logger.info("Explanation generator cleanup completed")

# Factory function
def create_explanation_generator(config) -> ExplanationGenerator:
    """Create and return a configured explanation generator"""
    return ExplanationGenerator(config)