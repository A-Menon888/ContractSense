"""
Training metrics and evaluation utilities for BERT-CRF model.

Implements comprehensive metrics for sequence labeling evaluation
including entity-level and token-level performance measures.
"""

from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityResult:
    """Results for a single entity/clause type."""
    precision: float
    recall: float
    f1: float
    support: int
    
@dataclass 
class EvaluationResults:
    """Comprehensive evaluation results."""
    entity_results: Dict[str, EntityResult]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    accuracy: float
    total_entities: int
    correct_entities: int
    
class SequenceLabelingMetrics:
    """
    Metrics calculator for sequence labeling tasks.
    
    Computes token-level and entity-level metrics for clause extraction,
    handling different tagging schemes (BIO, BIOS, IOBES).
    """
    
    def __init__(self, label_encoder, ignore_labels: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            label_encoder: LabelEncoder instance for converting labels
            ignore_labels: Labels to ignore in evaluation (e.g., ['O'])
        """
        self.label_encoder = label_encoder
        self.ignore_labels = set(ignore_labels or ['O'])
        self.clause_types = label_encoder.get_clause_types()
        
    def compute_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> EvaluationResults:
        """
        Compute comprehensive metrics for predictions.
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            EvaluationResults with detailed metrics
        """
        # Flatten sequences for token-level metrics
        flat_true = [label for seq in y_true for label in seq]
        flat_pred = [label for seq in y_pred for label in seq]
        
        # Convert to spans for entity-level metrics
        true_spans = []
        pred_spans = []
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            true_spans.extend(self._convert_to_spans(true_seq))
            pred_spans.extend(self._convert_to_spans(pred_seq))
        
        # Compute entity-level metrics
        entity_results = self._compute_entity_metrics(true_spans, pred_spans)
        
        # Compute overall metrics
        overall_precision, overall_recall, overall_f1 = self._compute_overall_metrics(
            true_spans, pred_spans
        )
        
        # Token-level accuracy
        correct_tokens = sum(1 for t, p in zip(flat_true, flat_pred) if t == p)
        accuracy = correct_tokens / len(flat_true) if flat_true else 0.0
        
        # Entity counts
        total_entities = len(true_spans)
        correct_entities = len(set(true_spans) & set(pred_spans))
        
        return EvaluationResults(
            entity_results=entity_results,
            overall_precision=overall_precision,
            overall_recall=overall_recall, 
            overall_f1=overall_f1,
            accuracy=accuracy,
            total_entities=total_entities,
            correct_entities=correct_entities
        )
    
    def _convert_to_spans(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """Convert sequence labels to spans."""
        return self.label_encoder.convert_labels_to_spans(labels)
    
    def _compute_entity_metrics(
        self, 
        true_spans: List[Tuple[int, int, str]], 
        pred_spans: List[Tuple[int, int, str]]
    ) -> Dict[str, EntityResult]:
        """
        Compute per-entity metrics.
        
        Args:
            true_spans: List of true (start, end, type) spans
            pred_spans: List of predicted (start, end, type) spans
            
        Returns:
            Dictionary mapping entity types to EntityResult
        """
        entity_results = {}
        
        # Group spans by entity type
        true_by_type = defaultdict(set)
        pred_by_type = defaultdict(set)
        
        for span in true_spans:
            start, end, entity_type = span
            true_by_type[entity_type].add((start, end))
            
        for span in pred_spans:
            start, end, entity_type = span
            pred_by_type[entity_type].add((start, end))
        
        # Compute metrics for each entity type
        all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
        
        for entity_type in all_types:
            true_set = true_by_type[entity_type]
            pred_set = pred_by_type[entity_type]
            
            # Compute precision, recall, F1
            tp = len(true_set & pred_set)  # True positives
            fp = len(pred_set - true_set)  # False positives
            fn = len(true_set - pred_set)  # False negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            entity_results[entity_type] = EntityResult(
                precision=precision,
                recall=recall,
                f1=f1,
                support=len(true_set)
            )
        
        return entity_results
    
    def _compute_overall_metrics(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]]
    ) -> Tuple[float, float, float]:
        """
        Compute overall precision, recall, and F1 across all entities.
        
        Args:
            true_spans: List of true spans
            pred_spans: List of predicted spans
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        true_set = set(true_spans)
        pred_set = set(pred_spans)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def classification_report(self, y_true: List[List[str]], y_pred: List[List[str]]) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report string
        """
        results = self.compute_metrics(y_true, y_pred)
        
        report_lines = []
        report_lines.append("Classification Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Header
        report_lines.append(f"{'Entity':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-" * 70)
        
        # Per-entity results
        for entity_type in sorted(results.entity_results.keys()):
            result = results.entity_results[entity_type]
            report_lines.append(
                f"{entity_type:<20} "
                f"{result.precision:<10.3f} "
                f"{result.recall:<10.3f} "
                f"{result.f1:<10.3f} "
                f"{result.support:<10d}"
            )
        
        report_lines.append("-" * 70)
        
        # Overall results
        report_lines.append(
            f"{'Overall':<20} "
            f"{results.overall_precision:<10.3f} "
            f"{results.overall_recall:<10.3f} "
            f"{results.overall_f1:<10.3f} "
            f"{results.total_entities:<10d}"
        )
        
        report_lines.append("")
        report_lines.append(f"Token-level accuracy: {results.accuracy:.3f}")
        report_lines.append(f"Entity-level accuracy: {results.correct_entities / results.total_entities:.3f}")
        
        return "\n".join(report_lines)
    
    def confusion_matrix(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Compute confusion matrix for entity types.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Nested dictionary representing confusion matrix
        """
        # Convert to spans
        true_spans = []
        pred_spans = []
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            true_seq_spans = self._convert_to_spans(true_seq)
            pred_seq_spans = self._convert_to_spans(pred_seq)
            
            true_spans.extend(true_seq_spans)
            pred_spans.extend(pred_seq_spans)
        
        # Build confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        
        # Map spans to entity types for position lookup
        pred_span_dict = {}
        for start, end, entity_type in pred_spans:
            pred_span_dict[(start, end)] = entity_type
        
        # For each true span, find corresponding prediction
        for start, end, true_type in true_spans:
            pred_type = pred_span_dict.get((start, end), "MISSED")
            confusion[true_type][pred_type] += 1
        
        # Add false positives
        true_span_dict = {}
        for start, end, entity_type in true_spans:
            true_span_dict[(start, end)] = entity_type
        
        for start, end, pred_type in pred_spans:
            if (start, end) not in true_span_dict:
                confusion["BACKGROUND"][pred_type] += 1
        
        return dict(confusion)
    
    def compute_macro_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
        """
        Compute macro-averaged metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with macro-averaged metrics
        """
        results = self.compute_metrics(y_true, y_pred)
        
        # Compute macro averages
        precisions = [result.precision for result in results.entity_results.values()]
        recalls = [result.recall for result in results.entity_results.values()]
        f1s = [result.f1 for result in results.entity_results.values()]
        
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1s) if f1s else 0.0
        
        return {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": results.overall_precision,
            "micro_recall": results.overall_recall,
            "micro_f1": results.overall_f1,
            "accuracy": results.accuracy
        }
    
    def error_analysis(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]],
        texts: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            texts: Optional list of token sequences for context
            
        Returns:
            Dictionary with error analysis results
        """
        errors = {
            "boundary_errors": [],    # Wrong span boundaries
            "type_errors": [],        # Wrong entity type
            "missed_entities": [],    # Entities not detected
            "false_positives": [],    # Predicted entities that don't exist
        }
        
        for i, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
            true_spans = self._convert_to_spans(true_seq)
            pred_spans = self._convert_to_spans(pred_seq)
            
            true_span_dict = {(start, end): entity_type for start, end, entity_type in true_spans}
            pred_span_dict = {(start, end): entity_type for start, end, entity_type in pred_spans}
            
            # Find different types of errors
            for (start, end), true_type in true_span_dict.items():
                if (start, end) in pred_span_dict:
                    pred_type = pred_span_dict[(start, end)]
                    if pred_type != true_type:
                        # Type error
                        error_info = {
                            "sequence_id": i,
                            "span": (start, end),
                            "true_type": true_type,
                            "pred_type": pred_type,
                        }
                        if texts and i < len(texts):
                            error_info["context"] = texts[i][max(0, start-2):min(len(texts[i]), end+2)]
                        errors["type_errors"].append(error_info)
                else:
                    # Check for boundary errors (overlapping spans with different boundaries)
                    found_overlap = False
                    for (p_start, p_end), pred_type in pred_span_dict.items():
                        if (start < p_end and end > p_start):  # Overlap condition
                            error_info = {
                                "sequence_id": i,
                                "true_span": (start, end),
                                "pred_span": (p_start, p_end),
                                "true_type": true_type,
                                "pred_type": pred_type,
                            }
                            if texts and i < len(texts):
                                error_info["context"] = texts[i][max(0, min(start, p_start)-2):
                                                                max(end, p_end)+2]
                            errors["boundary_errors"].append(error_info)
                            found_overlap = True
                            break
                    
                    if not found_overlap:
                        # Missed entity
                        error_info = {
                            "sequence_id": i,
                            "span": (start, end),
                            "entity_type": true_type,
                        }
                        if texts and i < len(texts):
                            error_info["context"] = texts[i][max(0, start-2):min(len(texts[i]), end+2)]
                        errors["missed_entities"].append(error_info)
            
            # Find false positives
            for (start, end), pred_type in pred_span_dict.items():
                if (start, end) not in true_span_dict:
                    # Check if it's not a boundary error (already counted)
                    is_boundary_error = False
                    for (t_start, t_end) in true_span_dict.keys():
                        if (start < t_end and end > t_start):
                            is_boundary_error = True
                            break
                    
                    if not is_boundary_error:
                        error_info = {
                            "sequence_id": i,
                            "span": (start, end),
                            "entity_type": pred_type,
                        }
                        if texts and i < len(texts):
                            error_info["context"] = texts[i][max(0, start-2):min(len(texts[i]), end+2)]
                        errors["false_positives"].append(error_info)
        
        # Add summary statistics
        errors["summary"] = {
            "total_boundary_errors": len(errors["boundary_errors"]),
            "total_type_errors": len(errors["type_errors"]),
            "total_missed_entities": len(errors["missed_entities"]),
            "total_false_positives": len(errors["false_positives"]),
        }
        
        return errors