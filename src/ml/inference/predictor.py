"""
Real-time clause prediction for contract documents.

Provides fast inference capabilities for the trained BERT-CRF model
with confidence scoring and result formatting.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

from ..models.bert_crf import BertCrfModel
from ..models.model_config import ModelConfig
from ..utils.tokenization import ContractTokenizer
from ..utils.label_encoding import LabelEncoder
from ...annotation import DocumentAnnotation, SpanAnnotation

logger = logging.getLogger(__name__)

@dataclass
class ClausePrediction:
    """Single clause prediction with confidence score."""
    start: int
    end: int
    clause_type: str
    confidence: float
    text: str
    token_indices: Tuple[int, int]

@dataclass
class DocumentPrediction:
    """Complete document prediction results."""
    document_id: str
    predictions: List[ClausePrediction]
    processing_time: float
    confidence_scores: List[float]
    raw_tokens: List[str]
    predicted_labels: List[str]

class ClausePredictor:
    """
    Real-time clause extraction predictor.
    
    Provides fast inference on contract documents with confidence scoring
    and result formatting compatible with Module 2 annotations.
    """
    
    def __init__(
        self,
        model: BertCrfModel,
        tokenizer: ContractTokenizer,
        label_encoder: LabelEncoder,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize clause predictor.
        
        Args:
            model: Trained BERT-CRF model
            tokenizer: Contract tokenizer
            label_encoder: Label encoder
            device: Inference device
            confidence_threshold: Minimum confidence for predictions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized ClausePredictor on device: {self.device}")
    
    def predict_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        return_confidence: bool = True
    ) -> DocumentPrediction:
        """
        Predict clauses in a document.
        
        Args:
            text: Document text
            document_id: Optional document identifier
            return_confidence: Whether to compute confidence scores
            
        Returns:
            DocumentPrediction with extracted clauses
        """
        start_time = datetime.now()
        
        if document_id is None:
            document_id = f"doc_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Tokenize text
        tokens = text.split()  # Simple whitespace tokenization
        
        # Process with BERT tokenizer
        tokenization_result = self.tokenizer.tokenize_with_alignment(
            tokens=tokens,
            add_special_tokens=True
        )
        
        # Convert to tensor
        input_ids = torch.tensor([tokenization_result.input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([tokenization_result.attention_mask], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([tokenization_result.token_type_ids], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_confidence=return_confidence
            )
        
        predictions = outputs["predictions"][0]  # First (and only) batch item
        confidence_scores = outputs.get("confidence_scores", [None] * len(predictions))[0]
        
        # Convert predictions back to labels
        predicted_labels = self.label_encoder.decode_labels(predictions)
        
        # Align with original tokens
        aligned_labels = self.tokenizer.merge_subword_predictions(
            tokenization_result=tokenization_result,
            predictions=predicted_labels,
            merge_strategy="first"
        )
        
        # Convert to spans
        spans = self.label_encoder.convert_labels_to_spans(aligned_labels)
        
        # Create clause predictions
        clause_predictions = []
        for start_idx, end_idx, clause_type in spans:
            # Calculate confidence score for this span
            if confidence_scores:
                # Average confidence over the span
                span_confidence = np.mean([
                    confidence_scores[i] for i in range(start_idx, min(end_idx, len(confidence_scores)))
                    if confidence_scores[i] is not None
                ])
            else:
                span_confidence = 1.0  # Default if no confidence available
            
            # Skip low-confidence predictions
            if span_confidence < self.confidence_threshold:
                continue
            
            # Extract clause text
            clause_text = " ".join(tokens[start_idx:end_idx])
            
            clause_predictions.append(ClausePrediction(
                start=start_idx,
                end=end_idx,
                clause_type=clause_type,
                confidence=span_confidence,
                text=clause_text,
                token_indices=(start_idx, end_idx)
            ))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentPrediction(
            document_id=document_id,
            predictions=clause_predictions,
            processing_time=processing_time,
            confidence_scores=confidence_scores or [],
            raw_tokens=tokens,
            predicted_labels=aligned_labels
        )
    
    def predict_batch(
        self,
        texts: List[str],
        document_ids: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> List[DocumentPrediction]:
        """
        Predict clauses for multiple documents.
        
        Args:
            texts: List of document texts
            document_ids: Optional list of document identifiers
            batch_size: Batch size for processing
            
        Returns:
            List of DocumentPrediction results
        """
        if document_ids is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            document_ids = [f"doc_{i}_{timestamp}" for i in range(len(texts))]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = document_ids[i:i + batch_size]
            
            # Process each document in the batch
            for text, doc_id in zip(batch_texts, batch_ids):
                result = self.predict_document(text, doc_id)
                results.append(result)
        
        return results
    
    def to_module2_format(self, prediction: DocumentPrediction) -> DocumentAnnotation:
        """
        Convert prediction to Module 2 DocumentAnnotation format.
        
        Args:
            prediction: Document prediction result
            
        Returns:
            DocumentAnnotation compatible with Module 2
        """
        # Create span annotations
        span_annotations = []
        for pred in prediction.predictions:
            span_ann = SpanAnnotation(
                start=pred.start,
                end=pred.end,
                text=pred.text,
                clause_type=pred.clause_type,
                confidence=pred.confidence,
                metadata={
                    "predicted_by": "bert_crf",
                    "token_indices": pred.token_indices
                }
            )
            span_annotations.append(span_ann)
        
        # Create document annotation
        doc_annotation = DocumentAnnotation(
            document_id=prediction.document_id,
            spans=span_annotations,
            metadata={
                "processing_time": prediction.processing_time,
                "model": "bert_crf",
                "num_predictions": len(prediction.predictions),
                "confidence_threshold": self.confidence_threshold
            }
        )
        
        return doc_annotation
    
    def filter_predictions(
        self,
        prediction: DocumentPrediction,
        min_confidence: Optional[float] = None,
        clause_types: Optional[List[str]] = None,
        min_length: int = 1
    ) -> DocumentPrediction:
        """
        Filter predictions based on criteria.
        
        Args:
            prediction: Original document prediction
            min_confidence: Minimum confidence threshold
            clause_types: Only include these clause types
            min_length: Minimum clause length in tokens
            
        Returns:
            Filtered DocumentPrediction
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        filtered_predictions = []
        
        for pred in prediction.predictions:
            # Check confidence
            if pred.confidence < min_confidence:
                continue
            
            # Check clause type
            if clause_types and pred.clause_type not in clause_types:
                continue
            
            # Check minimum length
            if (pred.end - pred.start) < min_length:
                continue
            
            filtered_predictions.append(pred)
        
        # Create new prediction object
        return DocumentPrediction(
            document_id=prediction.document_id,
            predictions=filtered_predictions,
            processing_time=prediction.processing_time,
            confidence_scores=prediction.confidence_scores,
            raw_tokens=prediction.raw_tokens,
            predicted_labels=prediction.predicted_labels
        )
    
    def get_prediction_summary(self, prediction: DocumentPrediction) -> Dict[str, Any]:
        """
        Get summary statistics for a prediction.
        
        Args:
            prediction: Document prediction
            
        Returns:
            Summary statistics
        """
        if not prediction.predictions:
            return {
                "document_id": prediction.document_id,
                "num_clauses": 0,
                "clause_types": [],
                "avg_confidence": 0.0,
                "processing_time": prediction.processing_time
            }
        
        clause_types = [pred.clause_type for pred in prediction.predictions]
        confidences = [pred.confidence for pred in prediction.predictions]
        
        from collections import Counter
        clause_counts = Counter(clause_types)
        
        return {
            "document_id": prediction.document_id,
            "num_clauses": len(prediction.predictions),
            "clause_types": list(clause_counts.keys()),
            "clause_distribution": dict(clause_counts),
            "avg_confidence": np.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "processing_time": prediction.processing_time,
            "tokens_processed": len(prediction.raw_tokens)
        }
    
    def export_predictions(
        self,
        predictions: List[DocumentPrediction],
        output_format: str = "json",
        output_file: Optional[str] = None
    ) -> Union[str, Dict]:
        """
        Export predictions to various formats.
        
        Args:
            predictions: List of document predictions
            output_format: Export format ("json", "csv", "conll")
            output_file: Optional output file path
            
        Returns:
            Exported data as string or dictionary
        """
        if output_format.lower() == "json":
            return self._export_to_json(predictions, output_file)
        elif output_format.lower() == "csv":
            return self._export_to_csv(predictions, output_file)
        elif output_format.lower() == "conll":
            return self._export_to_conll(predictions, output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _export_to_json(self, predictions: List[DocumentPrediction], output_file: Optional[str]) -> Dict:
        """Export predictions to JSON format."""
        import json
        
        data = []
        for pred in predictions:
            doc_data = {
                "document_id": pred.document_id,
                "processing_time": pred.processing_time,
                "clauses": [
                    {
                        "start": clause.start,
                        "end": clause.end,
                        "clause_type": clause.clause_type,
                        "text": clause.text,
                        "confidence": clause.confidence
                    }
                    for clause in pred.predictions
                ]
            }
            data.append(doc_data)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Predictions exported to {output_file}")
        
        return data
    
    def _export_to_csv(self, predictions: List[DocumentPrediction], output_file: Optional[str]) -> str:
        """Export predictions to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['document_id', 'start', 'end', 'clause_type', 'text', 'confidence'])
        
        # Write data
        for pred in predictions:
            for clause in pred.predictions:
                writer.writerow([
                    pred.document_id,
                    clause.start,
                    clause.end,
                    clause.clause_type,
                    clause.text.replace('\n', ' ').replace('\r', ' '),
                    clause.confidence
                ])
        
        result = output.getvalue()
        
        if output_file:
            with open(output_file, 'w', newline='') as f:
                f.write(result)
            logger.info(f"Predictions exported to {output_file}")
        
        return result
    
    def _export_to_conll(self, predictions: List[DocumentPrediction], output_file: Optional[str]) -> str:
        """Export predictions to CoNLL format."""
        lines = []
        
        for pred in predictions:
            lines.append(f"# Document: {pred.document_id}")
            
            # Write tokens with labels
            for i, (token, label) in enumerate(zip(pred.raw_tokens, pred.predicted_labels)):
                lines.append(f"{token}\t{label}")
            
            lines.append("")  # Empty line between documents
        
        result = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            logger.info(f"Predictions exported to {output_file}")
        
        return result
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer: ContractTokenizer,
        label_encoder: LabelEncoder,
        device: Optional[torch.device] = None
    ) -> 'ClausePredictor':
        """
        Load predictor from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer: Contract tokenizer
            label_encoder: Label encoder
            device: Inference device
            
        Returns:
            Loaded ClausePredictor instance
        """
        # Load model from checkpoint
        model = BertCrfModel.load_model(checkpoint_path)
        
        # Create predictor
        predictor = cls(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            device=device
        )
        
        logger.info(f"ClausePredictor loaded from {checkpoint_path}")
        return predictor