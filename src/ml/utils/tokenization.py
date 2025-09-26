"""
Tokenization utilities for BERT-CRF model.

Handles BERT tokenization with alignment to original tokens
for sequence labeling tasks.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import AutoTokenizer
import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    offset_mapping: List[Tuple[int, int]]
    word_ids: List[Optional[int]]
    original_tokens: List[str]
    bert_tokens: List[str]

class ContractTokenizer:
    """
    BERT tokenizer with alignment tracking for legal contracts.
    
    Handles tokenization while maintaining alignment between
    original tokens and BERT subword tokens for sequence labeling.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize contract tokenizer.
        
        Args:
            model_name: BERT model name for tokenizer
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Special tokens
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token = self.tokenizer.sep_token  
        self.sep_token_id = self.tokenizer.sep_token_id
        
        logger.info(f"Initialized ContractTokenizer with {model_name}, max_length={max_length}")
    
    def tokenize_with_alignment(
        self, 
        tokens: List[str],
        labels: Optional[List[str]] = None,
        add_special_tokens: bool = True
    ) -> TokenizationResult:
        """
        Tokenize token list with label alignment.
        
        Args:
            tokens: List of original tokens
            labels: Optional list of labels aligned with tokens
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            
        Returns:
            TokenizationResult with alignment information
        """
        # Join tokens to form text (preserving spaces)
        text = " ".join(tokens)
        
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        
        # Get word IDs to track token alignment
        word_ids = encoding.word_ids()
        
        # Get BERT tokens
        bert_tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        result = TokenizationResult(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'], 
            token_type_ids=encoding['token_type_ids'],
            offset_mapping=encoding['offset_mapping'],
            word_ids=word_ids,
            original_tokens=tokens,
            bert_tokens=bert_tokens
        )
        
        return result
    
    def align_labels_with_tokens(
        self,
        tokenization_result: TokenizationResult,
        labels: List[str],
        label_all_tokens: bool = False,
        ignore_label: str = "O"
    ) -> List[str]:
        """
        Align labels with BERT tokens.
        
        Args:
            tokenization_result: Result from tokenize_with_alignment
            labels: Labels aligned with original tokens
            label_all_tokens: If True, label all subword tokens; if False, only first subword
            ignore_label: Label to use for ignored tokens
            
        Returns:
            List of labels aligned with BERT tokens
        """
        aligned_labels = []
        word_ids = tokenization_result.word_ids
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(ignore_label)
            elif word_idx != previous_word_idx:
                # First subword token of a word
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(ignore_label)
            else:
                # Subsequent subword tokens of the same word
                if label_all_tokens and word_idx < len(labels):
                    # Use same label as first subword
                    original_label = labels[word_idx]
                    # Convert B- to I- for subsequent subwords
                    if original_label.startswith('B-'):
                        aligned_labels.append('I-' + original_label[2:])
                    else:
                        aligned_labels.append(original_label)
                else:
                    aligned_labels.append(ignore_label)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def tokenize_batch(
        self,
        token_lists: List[List[str]],
        label_lists: Optional[List[List[str]]] = None,
        pad_to_max_length: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of token sequences.
        
        Args:
            token_lists: List of token sequences
            label_lists: Optional list of label sequences
            pad_to_max_length: Whether to pad to max length
            
        Returns:
            Dictionary with batched tensors
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        
        max_len = 0
        
        for i, tokens in enumerate(token_lists):
            # Tokenize with alignment
            result = self.tokenize_with_alignment(tokens)
            
            batch_input_ids.append(result.input_ids)
            batch_attention_mask.append(result.attention_mask)
            batch_token_type_ids.append(result.token_type_ids)
            
            max_len = max(max_len, len(result.input_ids))
            
            # Handle labels if provided
            if label_lists is not None and i < len(label_lists):
                labels = label_lists[i]
                aligned_labels = self.align_labels_with_tokens(result, labels)
                batch_labels.append(aligned_labels)
            else:
                batch_labels.append(['O'] * len(result.input_ids))
        
        # Pad sequences if requested
        if pad_to_max_length:
            target_length = min(max_len, self.max_length)
            
            for i in range(len(batch_input_ids)):
                # Pad input_ids
                current_len = len(batch_input_ids[i])
                if current_len < target_length:
                    pad_length = target_length - current_len
                    batch_input_ids[i].extend([self.pad_token_id] * pad_length)
                    batch_attention_mask[i].extend([0] * pad_length)
                    batch_token_type_ids[i].extend([0] * pad_length)
                    batch_labels[i].extend(['O'] * pad_length)
                elif current_len > target_length:
                    # Truncate
                    batch_input_ids[i] = batch_input_ids[i][:target_length]
                    batch_attention_mask[i] = batch_attention_mask[i][:target_length]
                    batch_token_type_ids[i] = batch_token_type_ids[i][:target_length]
                    batch_labels[i] = batch_labels[i][:target_length]
        
        # Convert to tensors
        result = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(batch_token_type_ids, dtype=torch.long)
        }
        
        if label_lists is not None:
            result['labels'] = batch_labels  # Keep as list of strings for now
        
        return result
    
    def decode_predictions(
        self,
        input_ids: torch.Tensor,
        predictions: List[List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        skip_special_tokens: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Decode model predictions back to original token space.
        
        Args:
            input_ids: Original input IDs
            predictions: Model predictions (list of label IDs per sequence)
            attention_mask: Attention mask to identify valid tokens
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of decoded predictions per sequence
        """
        results = []
        
        for i, (sequence_ids, pred_labels) in enumerate(zip(input_ids, predictions)):
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(sequence_ids.tolist())
            
            # Get attention mask for this sequence
            if attention_mask is not None:
                mask = attention_mask[i].tolist()
            else:
                mask = [1] * len(tokens)
            
            # Decode predictions for valid tokens
            decoded_predictions = []
            for j, (token, pred_label) in enumerate(zip(tokens, pred_labels)):
                if mask[j] == 1:  # Valid token
                    if skip_special_tokens and token in [self.cls_token, self.sep_token, self.pad_token]:
                        continue
                    
                    decoded_predictions.append({
                        'token': token,
                        'prediction': pred_label,
                        'position': j
                    })
            
            results.append({
                'tokens': [p['token'] for p in decoded_predictions],
                'predictions': [p['prediction'] for p in decoded_predictions],
                'positions': [p['position'] for p in decoded_predictions]
            })
        
        return results
    
    def merge_subword_predictions(
        self,
        tokenization_result: TokenizationResult,
        predictions: List[str],
        merge_strategy: str = "first"
    ) -> List[str]:
        """
        Merge subword predictions back to word level.
        
        Args:
            tokenization_result: Original tokenization result
            predictions: Predictions for each BERT token
            merge_strategy: Strategy for merging ("first", "majority", "longest")
            
        Returns:
            Word-level predictions
        """
        word_predictions = []
        word_ids = tokenization_result.word_ids
        
        current_word_id = None
        current_word_predictions = []
        
        for i, (word_id, pred) in enumerate(zip(word_ids, predictions)):
            if word_id is None:
                # Special token, skip
                continue
                
            if word_id != current_word_id:
                # New word, finalize previous word
                if current_word_predictions:
                    merged_pred = self._merge_predictions(current_word_predictions, merge_strategy)
                    word_predictions.append(merged_pred)
                
                # Start new word
                current_word_id = word_id
                current_word_predictions = [pred]
            else:
                # Same word, accumulate prediction
                current_word_predictions.append(pred)
        
        # Finalize last word
        if current_word_predictions:
            merged_pred = self._merge_predictions(current_word_predictions, merge_strategy)
            word_predictions.append(merged_pred)
        
        return word_predictions
    
    def _merge_predictions(self, predictions: List[str], strategy: str) -> str:
        """
        Merge multiple predictions for a single word.
        
        Args:
            predictions: List of predictions for subwords
            strategy: Merging strategy
            
        Returns:
            Merged prediction
        """
        if not predictions:
            return "O"
        
        if strategy == "first":
            return predictions[0]
        elif strategy == "majority":
            # Return most common prediction
            from collections import Counter
            counts = Counter(predictions)
            return counts.most_common(1)[0][0]
        elif strategy == "longest":
            # Return prediction from longest span
            max_len = 0
            best_pred = predictions[0]
            
            for pred in predictions:
                if pred != "O" and len(pred) > max_len:
                    max_len = len(pred)
                    best_pred = pred
            
            return best_pred if max_len > 0 else predictions[0]
        else:
            # Default to first
            return predictions[0]
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get information about the tokenizer.
        
        Returns:
            Dictionary with tokenizer information
        """
        return {
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.max_length,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "special_tokens": list(self.tokenizer.special_tokens_map.keys())
        }