"""
Dataset Builder for ContractSense

Converts processed documents from Module 1 into annotated datasets ready for ML training.
Handles format conversion, train/val splits, and quality validation.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import random
import numpy as np
from collections import defaultdict, Counter
import logging

# Add ingestion module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion import ProcessedDocument, Clause

from . import (
    DocumentAnnotation, SpanAnnotation, TokenAnnotation, AnnotationStats,
    LabelingScheme, logger
)
from .schema import AnnotationSchema


class DatasetBuilder:
    """Builds training datasets from processed documents and annotations"""
    
    def __init__(self, schema: Optional[AnnotationSchema] = None):
        self.schema = schema or AnnotationSchema()
        self.labeling_scheme = self.schema.labeling_scheme
    
    def convert_processed_document_to_annotation(self, 
                                               processed_doc: ProcessedDocument,
                                               document_id: Optional[str] = None) -> DocumentAnnotation:
        """
        Convert a ProcessedDocument (from Module 1) to DocumentAnnotation format
        
        Args:
            processed_doc: ProcessedDocument from ingestion pipeline
            document_id: Optional document ID, will generate if not provided
            
        Returns:
            DocumentAnnotation with converted clause spans
        """
        
        if document_id is None:
            # Generate document ID from file path
            doc_path = Path(processed_doc.document_path)
            document_id = doc_path.stem
        
        # Convert detected clauses to span annotations
        span_annotations = []
        for clause in processed_doc.clauses:
            if clause.clause_type:  # Only include clauses with types
                span_annotation = SpanAnnotation(
                    start_char=clause.start_offset,
                    end_char=clause.end_offset,
                    clause_type=clause.clause_type,
                    text=clause.text,
                    confidence=clause.confidence,
                    notes=f"Auto-detected by {clause.detection_method}"
                )
                span_annotations.append(span_annotation)
        
        # Create document annotation
        doc_annotation = DocumentAnnotation(
            document_id=document_id,
            document_path=processed_doc.document_path,
            full_text=processed_doc.full_text,
            span_annotations=span_annotations,
            document_metadata={
                "total_pages": len(processed_doc.pages),
                "document_type": processed_doc.document_type.value,
                **processed_doc.metadata
            },
            annotation_metadata={
                "conversion_method": "auto_from_module1",
                "original_clauses_count": len(processed_doc.clauses),
                "converted_spans_count": len(span_annotations)
            }
        )
        
        return doc_annotation
    
    def build_token_annotations(self, 
                               doc_annotation: DocumentAnnotation,
                               tokenization_method: str = "simple") -> List[TokenAnnotation]:
        """
        Convert span annotations to token-level annotations for sequence labeling
        
        Args:
            doc_annotation: Document with span annotations
            tokenization_method: Method for tokenization ('simple', 'whitespace')
            
        Returns:
            List of token annotations in BIO/BIOS/IOBES format
        """
        
        full_text = doc_annotation.full_text
        
        # Simple tokenization (can be enhanced with spaCy later)
        if tokenization_method == "simple":
            tokens = self._simple_tokenize(full_text)
        elif tokenization_method == "whitespace":
            tokens = self._whitespace_tokenize(full_text)
        else:
            raise ValueError(f"Unknown tokenization method: {tokenization_method}")
        
        # Initialize all tokens as "O" (Outside)
        token_labels = ["O"] * len(tokens)
        
        # Apply span annotations to tokens
        for span in doc_annotation.span_annotations:
            self._apply_span_to_tokens(
                tokens, token_labels, span, 
                scheme=self.labeling_scheme
            )
        
        # Create token annotations
        token_annotations = []
        for i, (token, label) in enumerate(zip(tokens, token_labels)):
            token_annotation = TokenAnnotation(
                token=token['text'],
                label=label,
                start_char=token['start'],
                end_char=token['end'],
                token_idx=i
            )
            token_annotations.append(token_annotation)
        
        return token_annotations
    
    def _simple_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Simple tokenization that preserves character offsets"""
        import re
        
        tokens = []
        # Split on whitespace and punctuation but keep offsets
        pattern = r'\S+'
        
        for match in re.finditer(pattern, text):
            tokens.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return tokens
    
    def _whitespace_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Whitespace-only tokenization"""
        tokens = []
        current_pos = 0
        
        for token_text in text.split():
            # Find the token in the original text
            start_pos = text.find(token_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(token_text)
            
            tokens.append({
                'text': token_text,
                'start': start_pos,
                'end': end_pos
            })
            
            current_pos = end_pos
        
        return tokens
    
    def _apply_span_to_tokens(self, 
                             tokens: List[Dict[str, Any]], 
                             token_labels: List[str],
                             span: SpanAnnotation,
                             scheme: LabelingScheme) -> None:
        """Apply span annotation to token labels using specified scheme"""
        
        # Find tokens that overlap with the span
        overlapping_tokens = []
        for i, token in enumerate(tokens):
            # Check if token overlaps with span
            if not (token['end'] <= span.start_char or token['start'] >= span.end_char):
                overlapping_tokens.append(i)
        
        if not overlapping_tokens:
            return
        
        # Apply labeling scheme
        if scheme == LabelingScheme.BIO:
            # First token gets B- (Beginning), rest get I- (Inside)
            token_labels[overlapping_tokens[0]] = f"B-{span.clause_type}"
            for i in overlapping_tokens[1:]:
                token_labels[i] = f"I-{span.clause_type}"
        
        elif scheme == LabelingScheme.BIOS:
            if len(overlapping_tokens) == 1:
                # Single token gets S- (Single)
                token_labels[overlapping_tokens[0]] = f"S-{span.clause_type}"
            else:
                # First token gets B-, rest get I-
                token_labels[overlapping_tokens[0]] = f"B-{span.clause_type}"
                for i in overlapping_tokens[1:]:
                    token_labels[i] = f"I-{span.clause_type}"
        
        elif scheme == LabelingScheme.IOBES:
            if len(overlapping_tokens) == 1:
                # Single token gets S- (Single)
                token_labels[overlapping_tokens[0]] = f"S-{span.clause_type}"
            else:
                # First token gets B-, last gets E-, middle get I-
                token_labels[overlapping_tokens[0]] = f"B-{span.clause_type}"
                token_labels[overlapping_tokens[-1]] = f"E-{span.clause_type}"
                for i in overlapping_tokens[1:-1]:
                    token_labels[i] = f"I-{span.clause_type}"
    
    def create_train_val_split(self, 
                              documents: List[DocumentAnnotation],
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.2,
                              stratify_by: Optional[str] = None,
                              random_seed: int = 42) -> Tuple[List[DocumentAnnotation], List[DocumentAnnotation]]:
        """
        Split documents into training and validation sets
        
        Args:
            documents: List of annotated documents
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            stratify_by: Optional field to stratify by (e.g., 'contract_type')
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_documents, val_documents)
        """
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")
        
        if stratify_by:
            # Stratified split
            return self._stratified_split(documents, train_ratio, stratify_by)
        else:
            # Random split
            shuffled_docs = documents.copy()
            random.shuffle(shuffled_docs)
            
            split_idx = int(len(shuffled_docs) * train_ratio)
            train_docs = shuffled_docs[:split_idx]
            val_docs = shuffled_docs[split_idx:]
            
            return train_docs, val_docs
    
    def _stratified_split(self, 
                         documents: List[DocumentAnnotation], 
                         train_ratio: float, 
                         stratify_field: str) -> Tuple[List[DocumentAnnotation], List[DocumentAnnotation]]:
        """Perform stratified split based on a metadata field"""
        
        # Group documents by stratification field
        groups = defaultdict(list)
        for doc in documents:
            value = doc.document_metadata.get(stratify_field, "unknown")
            groups[value].append(doc)
        
        train_docs = []
        val_docs = []
        
        # Split each group
        for group_docs in groups.values():
            group_docs = list(group_docs)
            random.shuffle(group_docs)
            
            split_idx = int(len(group_docs) * train_ratio)
            train_docs.extend(group_docs[:split_idx])
            val_docs.extend(group_docs[split_idx:])
        
        return train_docs, val_docs
    
    def export_to_conll(self, 
                       documents: List[DocumentAnnotation],
                       output_path: Path,
                       include_metadata: bool = False) -> None:
        """
        Export token annotations to CoNLL format
        
        Format: TOKEN LABEL
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                if include_metadata:
                    f.write(f"# Document: {doc.document_id}\n")
                    f.write(f"# Path: {doc.document_path}\n")
                
                # Generate token annotations if not present
                if not doc.token_annotations:
                    doc.token_annotations = self.build_token_annotations(doc)
                
                for token_ann in doc.token_annotations:
                    f.write(f"{token_ann.token}\t{token_ann.label}\n")
                
                f.write("\n")  # Empty line between documents
        
        logger.info(f"Exported {len(documents)} documents to CoNLL format: {output_path}")
    
    def export_to_json(self, 
                      documents: List[DocumentAnnotation],
                      output_path: Path) -> None:
        """Export annotations to JSON format"""
        
        data = {
            "dataset_info": {
                "total_documents": len(documents),
                "created": "2025-09-24",
                "schema_version": "1.0"
            },
            "documents": [doc.to_dict() for doc in documents]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(documents)} documents to JSON format: {output_path}")
    
    def compute_dataset_stats(self, documents: List[DocumentAnnotation]) -> AnnotationStats:
        """Compute statistics about the dataset"""
        
        total_documents = len(documents)
        annotated_documents = sum(1 for doc in documents if doc.span_annotations)
        total_spans = sum(len(doc.span_annotations) for doc in documents)
        
        # Count spans per clause type
        spans_per_type = Counter()
        for doc in documents:
            for span in doc.span_annotations:
                spans_per_type[span.clause_type] += 1
        
        avg_spans_per_doc = total_spans / max(annotated_documents, 1)
        
        stats = AnnotationStats(
            total_documents=total_documents,
            annotated_documents=annotated_documents,
            total_spans=total_spans,
            spans_per_clause_type=dict(spans_per_type),
            average_spans_per_document=avg_spans_per_doc
        )
        
        return stats
    
    def validate_dataset(self, documents: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Validate dataset quality and consistency"""
        
        validation_report = {
            "total_documents": len(documents),
            "valid_documents": 0,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        for doc in documents:
            # Validate against schema
            errors = self.schema.validate_annotation(doc)
            if errors:
                validation_report["errors"].extend([
                    f"Document {doc.document_id}: {error}" 
                    for error in errors
                ])
            else:
                validation_report["valid_documents"] += 1
            
            # Check for potential issues
            warnings = self._check_annotation_quality(doc)
            validation_report["warnings"].extend([
                f"Document {doc.document_id}: {warning}"
                for warning in warnings
            ])
        
        # Compute overall stats
        validation_report["stats"] = self.compute_dataset_stats(documents).to_dict()
        
        return validation_report
    
    def _check_annotation_quality(self, doc: DocumentAnnotation) -> List[str]:
        """Check for annotation quality issues"""
        
        warnings = []
        
        # Check for very short spans
        for span in doc.span_annotations:
            if len(span.text.strip()) < 10:
                warnings.append(f"Very short span for {span.clause_type}: '{span.text[:30]}...'")
        
        # Check for suspiciously long spans
        for span in doc.span_annotations:
            if len(span.text) > 2000:
                warnings.append(f"Very long span for {span.clause_type}: {len(span.text)} characters")
        
        # Check for low confidence annotations
        low_confidence = [s for s in doc.span_annotations if s.confidence < 0.5]
        if low_confidence:
            warnings.append(f"Low confidence annotations: {len(low_confidence)} spans")
        
        return warnings