"""
Post-processing utilities for clause extraction predictions.

Handles cleaning, merging, and refining of raw model predictions
to improve final extraction quality.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
import re
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from .predictor import ClausePrediction, DocumentPrediction

@dataclass
class PostProcessingConfig:
    """Configuration for post-processing operations."""
    merge_adjacent_clauses: bool = True
    min_clause_length: int = 3  # Minimum tokens
    max_clause_length: int = 500  # Maximum tokens
    overlap_strategy: str = "highest_confidence"  # "merge", "highest_confidence", "longest"
    confidence_boost_keywords: Dict[str, List[str]] = None
    filter_duplicate_clauses: bool = True
    normalize_clause_boundaries: bool = True

class PostProcessor:
    """
    Post-processing utilities for clause extraction results.
    
    Applies various cleaning and refinement operations to improve
    the quality of extracted clauses from BERT-CRF predictions.
    """
    
    def __init__(self, config: PostProcessingConfig = None):
        """
        Initialize post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessingConfig()
        
        # Default confidence boost keywords
        if self.config.confidence_boost_keywords is None:
            self.config.confidence_boost_keywords = {
                "termination": ["terminate", "end", "expire", "cancel", "dissolution"],
                "payment": ["pay", "payment", "fee", "cost", "price", "invoice"],
                "confidentiality": ["confidential", "non-disclosure", "proprietary", "secret"],
                "intellectual_property": ["patent", "trademark", "copyright", "ip", "intellectual"],
                "limitation_of_liability": ["liability", "damages", "limit", "limitation"],
                "indemnification": ["indemnify", "indemnification", "hold harmless"],
                "force_majeure": ["force majeure", "act of god", "unforeseeable"],
                "governing_law": ["governed by", "jurisdiction", "applicable law"],
                "dispute_resolution": ["dispute", "arbitration", "mediation", "court"],
                "warranty": ["warrant", "warranty", "guarantee", "represent"],
                "compliance": ["comply", "compliance", "regulation", "law"]
            }
    
    def process_document(self, prediction: DocumentPrediction) -> DocumentPrediction:
        """
        Apply full post-processing pipeline to document prediction.
        
        Args:
            prediction: Original document prediction
            
        Returns:
            Post-processed document prediction
        """
        processed_predictions = prediction.predictions.copy()
        
        # Step 1: Filter by length constraints
        processed_predictions = self._filter_by_length(processed_predictions)
        
        # Step 2: Resolve overlapping clauses
        processed_predictions = self._resolve_overlaps(processed_predictions)
        
        # Step 3: Merge adjacent similar clauses
        if self.config.merge_adjacent_clauses:
            processed_predictions = self._merge_adjacent_clauses(processed_predictions)
        
        # Step 4: Normalize clause boundaries
        if self.config.normalize_clause_boundaries:
            processed_predictions = self._normalize_boundaries(
                processed_predictions, 
                prediction.raw_tokens
            )
        
        # Step 5: Apply confidence boosting based on keywords
        processed_predictions = self._boost_confidence_with_keywords(processed_predictions)
        
        # Step 6: Filter duplicates
        if self.config.filter_duplicate_clauses:
            processed_predictions = self._filter_duplicates(processed_predictions)
        
        # Step 7: Sort by position
        processed_predictions = sorted(processed_predictions, key=lambda x: x.start)
        
        # Create new prediction object
        return DocumentPrediction(
            document_id=prediction.document_id,
            predictions=processed_predictions,
            processing_time=prediction.processing_time,
            confidence_scores=prediction.confidence_scores,
            raw_tokens=prediction.raw_tokens,
            predicted_labels=prediction.predicted_labels
        )
    
    def _filter_by_length(self, predictions: List[ClausePrediction]) -> List[ClausePrediction]:
        """Filter predictions by length constraints."""
        filtered = []
        
        for pred in predictions:
            length = pred.end - pred.start
            if self.config.min_clause_length <= length <= self.config.max_clause_length:
                filtered.append(pred)
        
        return filtered
    
    def _resolve_overlaps(self, predictions: List[ClausePrediction]) -> List[ClausePrediction]:
        """Resolve overlapping clause predictions."""
        if len(predictions) <= 1:
            return predictions
        
        # Sort by start position
        sorted_preds = sorted(predictions, key=lambda x: x.start)
        resolved = []
        
        i = 0
        while i < len(sorted_preds):
            current = sorted_preds[i]
            overlapping = [current]
            
            # Find all overlapping predictions
            j = i + 1
            while j < len(sorted_preds):
                if sorted_preds[j].start < current.end:
                    overlapping.append(sorted_preds[j])
                    j += 1
                else:
                    break
            
            # Resolve overlaps
            if len(overlapping) == 1:
                resolved.append(current)
            else:
                resolved_pred = self._resolve_overlap_group(overlapping)
                if resolved_pred:
                    resolved.append(resolved_pred)
            
            i = j
        
        return resolved
    
    def _resolve_overlap_group(self, overlapping: List[ClausePrediction]) -> Optional[ClausePrediction]:
        """Resolve a group of overlapping predictions."""
        if self.config.overlap_strategy == "highest_confidence":
            return max(overlapping, key=lambda x: x.confidence)
        
        elif self.config.overlap_strategy == "longest":
            return max(overlapping, key=lambda x: x.end - x.start)
        
        elif self.config.overlap_strategy == "merge":
            # Merge overlapping clauses if they're the same type
            clause_types = set(pred.clause_type for pred in overlapping)
            
            if len(clause_types) == 1:
                # Same clause type - merge
                min_start = min(pred.start for pred in overlapping)
                max_end = max(pred.end for pred in overlapping)
                avg_confidence = np.mean([pred.confidence for pred in overlapping])
                
                # Reconstruct text (assuming tokens are available)
                merged_text = " ".join([pred.text for pred in overlapping])
                
                return ClausePrediction(
                    start=min_start,
                    end=max_end,
                    clause_type=list(clause_types)[0],
                    confidence=avg_confidence,
                    text=merged_text,
                    token_indices=(min_start, max_end)
                )
            else:
                # Different types - use highest confidence
                return max(overlapping, key=lambda x: x.confidence)
        
        return None
    
    def _merge_adjacent_clauses(self, predictions: List[ClausePrediction]) -> List[ClausePrediction]:
        """Merge adjacent clauses of the same type."""
        if len(predictions) <= 1:
            return predictions
        
        sorted_preds = sorted(predictions, key=lambda x: x.start)
        merged = []
        
        current = sorted_preds[0]
        
        for next_pred in sorted_preds[1:]:
            # Check if adjacent and same type
            if (next_pred.start <= current.end + 2 and  # Allow 1-2 token gap
                next_pred.clause_type == current.clause_type):
                
                # Merge clauses
                merged_text = f"{current.text} {next_pred.text}".strip()
                avg_confidence = (current.confidence + next_pred.confidence) / 2
                
                current = ClausePrediction(
                    start=current.start,
                    end=next_pred.end,
                    clause_type=current.clause_type,
                    confidence=avg_confidence,
                    text=merged_text,
                    token_indices=(current.start, next_pred.end)
                )
            else:
                # Not adjacent or different type
                merged.append(current)
                current = next_pred
        
        merged.append(current)
        return merged
    
    def _normalize_boundaries(
        self, 
        predictions: List[ClausePrediction], 
        tokens: List[str]
    ) -> List[ClausePrediction]:
        """Normalize clause boundaries to sentence/punctuation boundaries."""
        normalized = []
        
        for pred in predictions:
            # Find better start boundary
            new_start = self._find_sentence_start(pred.start, tokens)
            
            # Find better end boundary  
            new_end = self._find_sentence_end(pred.end, tokens)
            
            # Update text if boundaries changed
            if new_start != pred.start or new_end != pred.end:
                new_text = " ".join(tokens[new_start:new_end])
                
                normalized.append(ClausePrediction(
                    start=new_start,
                    end=new_end,
                    clause_type=pred.clause_type,
                    confidence=pred.confidence,
                    text=new_text,
                    token_indices=(new_start, new_end)
                ))
            else:
                normalized.append(pred)
        
        return normalized
    
    def _find_sentence_start(self, start: int, tokens: List[str]) -> int:
        """Find appropriate sentence start boundary."""
        # Look backwards for sentence boundaries
        for i in range(start, max(0, start - 10), -1):
            if i == 0:
                return i
            
            prev_token = tokens[i - 1] if i > 0 else ""
            
            # Sentence boundary markers
            if prev_token.endswith('.') or prev_token.endswith('!') or prev_token.endswith('?'):
                return i
            
            # Paragraph markers
            if prev_token in ['\n', '\n\n', '\\n']:
                return i
        
        return start  # Return original if no boundary found
    
    def _find_sentence_end(self, end: int, tokens: List[str]) -> int:
        """Find appropriate sentence end boundary."""
        # Look forwards for sentence boundaries
        for i in range(end, min(len(tokens), end + 10)):
            if i >= len(tokens):
                return len(tokens)
            
            token = tokens[i]
            
            # Sentence boundary markers
            if token.endswith('.') or token.endswith('!') or token.endswith('?'):
                return i + 1
            
            # Paragraph markers
            if token in ['\n', '\n\n', '\\n']:
                return i
        
        return end  # Return original if no boundary found
    
    def _boost_confidence_with_keywords(self, predictions: List[ClausePrediction]) -> List[ClausePrediction]:
        """Boost confidence for predictions containing relevant keywords."""
        boosted = []
        
        for pred in predictions:
            # Check for relevant keywords
            text_lower = pred.text.lower()
            keywords = self.config.confidence_boost_keywords.get(pred.clause_type, [])
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            
            # Apply confidence boost
            if keyword_matches > 0:
                boost_factor = min(1.2, 1.0 + (keyword_matches * 0.05))  # Max 20% boost
                new_confidence = min(1.0, pred.confidence * boost_factor)
                
                boosted.append(ClausePrediction(
                    start=pred.start,
                    end=pred.end,
                    clause_type=pred.clause_type,
                    confidence=new_confidence,
                    text=pred.text,
                    token_indices=pred.token_indices
                ))
            else:
                boosted.append(pred)
        
        return boosted
    
    def _filter_duplicates(self, predictions: List[ClausePrediction]) -> List[ClausePrediction]:
        """Filter out duplicate clause predictions."""
        seen_spans = set()
        filtered = []
        
        # Sort by confidence (descending) to keep best duplicates
        sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        
        for pred in sorted_preds:
            span_key = (pred.start, pred.end, pred.clause_type)
            
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                filtered.append(pred)
        
        return filtered
    
    def get_processing_statistics(self, original: DocumentPrediction, processed: DocumentPrediction) -> Dict[str, Any]:
        """
        Get statistics about post-processing changes.
        
        Args:
            original: Original document prediction
            processed: Post-processed document prediction
            
        Returns:
            Dictionary with processing statistics
        """
        orig_clauses = len(original.predictions)
        proc_clauses = len(processed.predictions)
        
        # Calculate confidence changes
        orig_confidences = [p.confidence for p in original.predictions]
        proc_confidences = [p.confidence for p in processed.predictions]
        
        orig_avg_conf = np.mean(orig_confidences) if orig_confidences else 0.0
        proc_avg_conf = np.mean(proc_confidences) if proc_confidences else 0.0
        
        return {
            "original_clause_count": orig_clauses,
            "processed_clause_count": proc_clauses,
            "clauses_removed": orig_clauses - proc_clauses,
            "removal_rate": (orig_clauses - proc_clauses) / orig_clauses if orig_clauses > 0 else 0.0,
            "original_avg_confidence": orig_avg_conf,
            "processed_avg_confidence": proc_avg_conf,
            "confidence_change": proc_avg_conf - orig_avg_conf,
            "processing_applied": {
                "length_filtering": True,
                "overlap_resolution": True,
                "adjacent_merging": self.config.merge_adjacent_clauses,
                "boundary_normalization": self.config.normalize_clause_boundaries,
                "confidence_boosting": True,
                "duplicate_filtering": self.config.filter_duplicate_clauses
            }
        }