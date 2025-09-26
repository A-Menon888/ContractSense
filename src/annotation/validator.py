"""
Annotation Validator for ContractSense

Validates annotation quality, consistency, and inter-annotator agreement.
Provides tools for annotation review and quality assurance.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
import json
from collections import Counter, defaultdict
import numpy as np
from dataclasses import asdict
import logging

from . import DocumentAnnotation, SpanAnnotation, AnnotationStats, logger
from .schema import AnnotationSchema


class AnnotationValidator:
    """Validates annotation quality and consistency"""
    
    def __init__(self, schema: Optional[AnnotationSchema] = None):
        self.schema = schema or AnnotationSchema()
    
    def validate_single_annotation(self, doc_annotation: DocumentAnnotation) -> Dict[str, Any]:
        """
        Validate a single document annotation
        
        Returns:
            Validation report with errors, warnings, and suggestions
        """
        
        report = {
            "document_id": doc_annotation.document_id,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "stats": {}
        }
        
        # Schema validation
        schema_errors = self.schema.validate_annotation(doc_annotation)
        if schema_errors:
            report["errors"].extend(schema_errors)
            report["is_valid"] = False
        
        # Text consistency checks
        text_errors = self._validate_text_consistency(doc_annotation)
        report["errors"].extend(text_errors)
        if text_errors:
            report["is_valid"] = False
        
        # Span overlap checks
        overlap_warnings = self._check_span_overlaps(doc_annotation)
        report["warnings"].extend(overlap_warnings)
        
        # Quality checks
        quality_warnings = self._check_annotation_quality(doc_annotation)
        report["warnings"].extend(quality_warnings)
        
        # Improvement suggestions
        suggestions = self._generate_suggestions(doc_annotation)
        report["suggestions"].extend(suggestions)
        
        # Basic stats
        report["stats"] = {
            "total_spans": len(doc_annotation.span_annotations),
            "clause_types": len(set(s.clause_type for s in doc_annotation.span_annotations)),
            "avg_span_length": np.mean([len(s.text) for s in doc_annotation.span_annotations]) if doc_annotation.span_annotations else 0,
            "avg_confidence": np.mean([s.confidence for s in doc_annotation.span_annotations if s.confidence is not None]) if doc_annotation.span_annotations else 0
        }
        
        return report
    
    def _validate_text_consistency(self, doc: DocumentAnnotation) -> List[str]:
        """Check that span texts match the full document text"""
        
        errors = []
        
        for i, span in enumerate(doc.span_annotations):
            # Check if span indices are within document bounds
            if span.start_char < 0 or span.end_char > len(doc.full_text):
                errors.append(f"Span {i}: indices out of bounds ({span.start_char}-{span.end_char})")
                continue
            
            # Check if span text matches document text at those positions
            expected_text = doc.full_text[span.start_char:span.end_char]
            if span.text != expected_text:
                errors.append(f"Span {i}: text mismatch. Expected: '{expected_text[:50]}...', Got: '{span.text[:50]}...'")
        
        return errors
    
    def _check_span_overlaps(self, doc: DocumentAnnotation) -> List[str]:
        """Check for overlapping spans"""
        
        warnings = []
        
        spans = doc.span_annotations
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                span1, span2 = spans[i], spans[j]
                
                # Check for overlap
                if not (span1.end_char <= span2.start_char or span2.end_char <= span1.start_char):
                    overlap_start = max(span1.start_char, span2.start_char)
                    overlap_end = min(span1.end_char, span2.end_char)
                    overlap_text = doc.full_text[overlap_start:overlap_end]
                    
                    warnings.append(
                        f"Overlapping spans: {span1.clause_type} and {span2.clause_type} "
                        f"overlap at chars {overlap_start}-{overlap_end}: '{overlap_text[:30]}...'"
                    )
        
        return warnings
    
    def _check_annotation_quality(self, doc: DocumentAnnotation) -> List[str]:
        """Check for annotation quality issues"""
        
        warnings = []
        
        for span in doc.span_annotations:
            # Check for very short spans
            if len(span.text.strip()) < 10:
                warnings.append(f"Very short {span.clause_type} span: '{span.text[:30]}'")
            
            # Check for suspiciously long spans
            if len(span.text) > 5000:
                warnings.append(f"Very long {span.clause_type} span: {len(span.text)} characters")
            
            # Check for low confidence
            if span.confidence is not None and span.confidence < 0.3:
                warnings.append(f"Low confidence {span.clause_type} span: {span.confidence:.3f}")
            
            # Check for empty or whitespace-only text
            if not span.text.strip():
                warnings.append(f"Empty or whitespace-only {span.clause_type} span")
            
            # Check for spans that are mostly punctuation
            punct_ratio = sum(1 for c in span.text if not c.isalnum() and not c.isspace()) / len(span.text)
            if punct_ratio > 0.5:
                warnings.append(f"{span.clause_type} span is mostly punctuation: {punct_ratio:.2%}")
        
        return warnings
    
    def _generate_suggestions(self, doc: DocumentAnnotation) -> List[str]:
        """Generate suggestions for improving annotation quality"""
        
        suggestions = []
        
        # Check for missing common clause types
        present_types = set(span.clause_type for span in doc.span_annotations)
        common_types = {"termination", "limitation_of_liability", "indemnification"}
        missing_common = common_types - present_types
        
        if missing_common:
            suggestions.append(f"Consider checking for common clause types: {', '.join(missing_common)}")
        
        # Suggest reviewing very long documents with few annotations
        if len(doc.full_text) > 50000 and len(doc.span_annotations) < 5:
            suggestions.append("Long document with few annotations - consider reviewing for missed clauses")
        
        # Suggest confidence review
        low_conf_count = sum(1 for s in doc.span_annotations if s.confidence is not None and s.confidence < 0.7)
        if low_conf_count > len(doc.span_annotations) * 0.3:
            suggestions.append("Many low-confidence annotations - consider manual review")
        
        return suggestions
    
    def compute_inter_annotator_agreement(self, 
                                        annotations_list: List[List[DocumentAnnotation]], 
                                        agreement_method: str = "exact_match") -> Dict[str, float]:
        """
        Compute inter-annotator agreement between multiple annotators
        
        Args:
            annotations_list: List of annotation sets, one per annotator
            agreement_method: Method for computing agreement ('exact_match', 'token_level', 'span_overlap')
            
        Returns:
            Agreement scores and statistics
        """
        
        if len(annotations_list) < 2:
            raise ValueError("Need at least 2 annotators for agreement computation")
        
        # Group annotations by document
        doc_groups = defaultdict(list)
        for annotator_annotations in annotations_list:
            for doc in annotator_annotations:
                doc_groups[doc.document_id].append(doc)
        
        # Only consider documents annotated by all annotators
        complete_docs = {doc_id: docs for doc_id, docs in doc_groups.items() 
                        if len(docs) == len(annotations_list)}
        
        if not complete_docs:
            return {"error": "No documents annotated by all annotators"}
        
        if agreement_method == "exact_match":
            return self._compute_exact_match_agreement(complete_docs)
        elif agreement_method == "token_level":
            return self._compute_token_level_agreement(complete_docs)
        elif agreement_method == "span_overlap":
            return self._compute_span_overlap_agreement(complete_docs)
        else:
            raise ValueError(f"Unknown agreement method: {agreement_method}")
    
    def _compute_exact_match_agreement(self, doc_groups: Dict[str, List[DocumentAnnotation]]) -> Dict[str, float]:
        """Compute exact match agreement between annotations"""
        
        total_comparisons = 0
        exact_matches = 0
        
        for doc_id, doc_annotations in doc_groups.items():
            # Compare all pairs of annotators
            for i in range(len(doc_annotations)):
                for j in range(i + 1, len(doc_annotations)):
                    doc1, doc2 = doc_annotations[i], doc_annotations[j]
                    
                    # Convert spans to sets for comparison
                    spans1 = {(s.start_char, s.end_char, s.clause_type) for s in doc1.span_annotations}
                    spans2 = {(s.start_char, s.end_char, s.clause_type) for s in doc2.span_annotations}
                    
                    if spans1 == spans2:
                        exact_matches += 1
                    
                    total_comparisons += 1
        
        agreement_rate = exact_matches / total_comparisons if total_comparisons > 0 else 0
        
        return {
            "method": "exact_match",
            "agreement_rate": agreement_rate,
            "exact_matches": exact_matches,
            "total_comparisons": total_comparisons,
            "documents_compared": len(doc_groups)
        }
    
    def _compute_token_level_agreement(self, doc_groups: Dict[str, List[DocumentAnnotation]]) -> Dict[str, float]:
        """Compute token-level agreement (requires tokenization)"""
        
        # This is a simplified version - in practice you'd use proper tokenization
        total_tokens = 0
        agreement_tokens = 0
        
        for doc_id, doc_annotations in doc_groups.items():
            # Compare all pairs
            for i in range(len(doc_annotations)):
                for j in range(i + 1, len(doc_annotations)):
                    doc1, doc2 = doc_annotations[i], doc_annotations[j]
                    
                    # Create character-level labels (simplified)
                    text_len = len(doc1.full_text)
                    labels1 = ["O"] * text_len
                    labels2 = ["O"] * text_len
                    
                    # Apply annotations
                    for span in doc1.span_annotations:
                        for k in range(span.start_char, span.end_char):
                            labels1[k] = span.clause_type
                    
                    for span in doc2.span_annotations:
                        for k in range(span.start_char, span.end_char):
                            labels2[k] = span.clause_type
                    
                    # Compute agreement
                    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
                    agreement_tokens += agreements
                    total_tokens += text_len
        
        agreement_rate = agreement_tokens / total_tokens if total_tokens > 0 else 0
        
        return {
            "method": "token_level",
            "agreement_rate": agreement_rate,
            "agreement_tokens": agreement_tokens,
            "total_tokens": total_tokens,
            "documents_compared": len(doc_groups)
        }
    
    def _compute_span_overlap_agreement(self, doc_groups: Dict[str, List[DocumentAnnotation]]) -> Dict[str, float]:
        """Compute agreement based on span overlap (more lenient)"""
        
        total_spans = 0
        overlapping_spans = 0
        
        for doc_id, doc_annotations in doc_groups.items():
            # Compare all pairs
            for i in range(len(doc_annotations)):
                for j in range(i + 1, len(doc_annotations)):
                    doc1, doc2 = doc_annotations[i], doc_annotations[j]
                    
                    # Find overlapping spans
                    for span1 in doc1.span_annotations:
                        for span2 in doc2.span_annotations:
                            if (span1.clause_type == span2.clause_type and 
                                self._spans_overlap(span1, span2)):
                                overlapping_spans += 1
                                break
                        total_spans += 1
                    
                    total_spans += len(doc2.span_annotations)
        
        agreement_rate = overlapping_spans / total_spans if total_spans > 0 else 0
        
        return {
            "method": "span_overlap",
            "agreement_rate": agreement_rate,
            "overlapping_spans": overlapping_spans,
            "total_spans": total_spans,
            "documents_compared": len(doc_groups)
        }
    
    def _spans_overlap(self, span1: SpanAnnotation, span2: SpanAnnotation, 
                      min_overlap: float = 0.5) -> bool:
        """Check if two spans have sufficient overlap"""
        
        overlap_start = max(span1.start_char, span2.start_char)
        overlap_end = min(span1.end_char, span2.end_char)
        
        if overlap_start >= overlap_end:
            return False
        
        overlap_len = overlap_end - overlap_start
        span1_len = span1.end_char - span1.start_char
        span2_len = span2.end_char - span2.start_char
        
        # Check if overlap is at least min_overlap of either span
        overlap_ratio1 = overlap_len / span1_len
        overlap_ratio2 = overlap_len / span2_len
        
        return overlap_ratio1 >= min_overlap or overlap_ratio2 >= min_overlap
    
    def generate_validation_report(self, 
                                 documents: List[DocumentAnnotation],
                                 output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive validation report for a set of documents"""
        
        report = {
            "validation_summary": {
                "total_documents": len(documents),
                "valid_documents": 0,
                "documents_with_errors": 0,
                "documents_with_warnings": 0
            },
            "error_summary": Counter(),
            "warning_summary": Counter(),
            "document_reports": []
        }
        
        for doc in documents:
            doc_report = self.validate_single_annotation(doc)
            report["document_reports"].append(doc_report)
            
            if doc_report["is_valid"]:
                report["validation_summary"]["valid_documents"] += 1
            else:
                report["validation_summary"]["documents_with_errors"] += 1
            
            if doc_report["warnings"]:
                report["validation_summary"]["documents_with_warnings"] += 1
            
            # Aggregate errors and warnings
            for error in doc_report["errors"]:
                error_type = error.split(":")[0] if ":" in error else "Unknown"
                report["error_summary"][error_type] += 1
            
            for warning in doc_report["warnings"]:
                warning_type = warning.split(":")[0] if ":" in warning else "Unknown"
                report["warning_summary"][warning_type] += 1
        
        # Convert Counters to dicts for JSON serialization
        report["error_summary"] = dict(report["error_summary"])
        report["warning_summary"] = dict(report["warning_summary"])
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Validation report saved to: {output_path}")
        
        return report
    
    def suggest_annotation_improvements(self, 
                                      documents: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Analyze documents and suggest improvements to annotation strategy"""
        
        suggestions = {
            "dataset_level": [],
            "annotation_guidelines": [],
            "quality_improvements": []
        }
        
        # Analyze clause type distribution
        clause_counts = Counter()
        for doc in documents:
            for span in doc.span_annotations:
                clause_counts[span.clause_type] += 1
        
        # Identify rare clause types
        total_spans = sum(clause_counts.values())
        rare_types = [ct for ct, count in clause_counts.items() if count / total_spans < 0.01]
        
        if rare_types:
            suggestions["dataset_level"].append(
                f"Consider collecting more examples of rare clause types: {', '.join(rare_types)}"
            )
        
        # Check for class imbalance
        most_common = clause_counts.most_common(1)[0]
        if most_common[1] > total_spans * 0.5:
            suggestions["dataset_level"].append(
                f"Dataset is heavily skewed towards {most_common[0]} clauses ({most_common[1]/total_spans:.1%})"
            )
        
        # Analyze confidence scores
        all_confidences = [s.confidence for doc in documents for s in doc.span_annotations 
                          if s.confidence is not None]
        
        if all_confidences:
            avg_confidence = np.mean(all_confidences)
            if avg_confidence < 0.7:
                suggestions["quality_improvements"].append(
                    f"Average confidence is low ({avg_confidence:.3f}) - consider annotation review"
                )
        
        # Check annotation density
        docs_with_few_annotations = sum(1 for doc in documents if len(doc.span_annotations) < 3)
        if docs_with_few_annotations > len(documents) * 0.3:
            suggestions["annotation_guidelines"].append(
                "Many documents have very few annotations - review annotation guidelines"
            )
        
        return suggestions