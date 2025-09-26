"""
Data loading and preprocessing for BERT-CRF training.

Handles conversion from Module 2 annotations to training data
and efficient batch processing for model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Iterator
import json
import logging
from dataclasses import dataclass
import random
import numpy as np

from ..utils.tokenization import ContractTokenizer
from ..utils.label_encoding import LabelEncoder
from ...annotation import DocumentAnnotation, TokenAnnotation

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Single training example for BERT-CRF model."""
    tokens: List[str]
    labels: List[str]
    document_id: str
    clause_types: List[str]

class ContractDataset(Dataset):
    """
    PyTorch dataset for contract clause extraction.
    
    Loads and preprocesses annotations from Module 2 format
    for BERT-CRF training.
    """
    
    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer: ContractTokenizer,
        label_encoder: LabelEncoder,
        max_length: int = 512,
        label_all_tokens: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of training examples
            tokenizer: Contract tokenizer for BERT preprocessing
            label_encoder: Label encoder for converting labels to IDs
            max_length: Maximum sequence length
            label_all_tokens: Whether to label all subword tokens
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        
        logger.info(f"Initialized ContractDataset with {len(examples)} examples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        example = self.examples[idx]
        
        # Tokenize with alignment
        tokenization_result = self.tokenizer.tokenize_with_alignment(
            tokens=example.tokens,
            add_special_tokens=True
        )
        
        # Align labels with BERT tokens
        aligned_labels = self.tokenizer.align_labels_with_tokens(
            tokenization_result=tokenization_result,
            labels=example.labels,
            label_all_tokens=self.label_all_tokens,
            ignore_label="O"
        )
        
        # Encode labels to IDs
        label_ids = self.label_encoder.encode_labels(aligned_labels)
        
        # Pad or truncate to max_length
        input_ids = tokenization_result.input_ids[:self.max_length]
        attention_mask = tokenization_result.attention_mask[:self.max_length]
        token_type_ids = tokenization_result.token_type_ids[:self.max_length]
        label_ids = label_ids[:self.max_length]
        
        # Pad if necessary
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            token_type_ids.extend([0] * padding_length)
            label_ids.extend([self.label_encoder.label2id["O"]] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "document_id": example.document_id
        }

class DataCollator:
    """Custom data collator for dynamic padding."""
    
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        """
        Initialize data collator.
        
        Args:
            pad_token_id: Padding token ID for input sequences
            label_pad_token_id: Padding token ID for labels
        """
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with dynamic padding.
        
        Args:
            batch: List of examples from dataset
            
        Returns:
            Batched and padded tensors
        """
        # Find maximum length in batch
        max_len = max(len(example["input_ids"]) for example in batch)
        
        # Initialize batch tensors
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        batch_document_ids = []
        
        for example in batch:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            token_type_ids = example["token_type_ids"]
            labels = example["labels"]
            
            # Pad sequences
            padding_length = max_len - len(input_ids)
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                token_type_ids = torch.cat([
                    token_type_ids,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), self.label_pad_token_id, dtype=torch.long)
                ])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)
            batch_document_ids.append(example["document_id"])
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "token_type_ids": torch.stack(batch_token_type_ids),
            "labels": torch.stack(batch_labels),
            "document_ids": batch_document_ids
        }

class DataLoader:
    """
    Data loader for BERT-CRF training.
    
    Handles loading annotations from Module 2 and converting
    to training format for BERT-CRF model.
    """
    
    def __init__(
        self,
        tokenizer: ContractTokenizer,
        label_encoder: LabelEncoder,
        max_length: int = 512,
        validation_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize data loader.
        
        Args:
            tokenizer: Contract tokenizer
            label_encoder: Label encoder
            max_length: Maximum sequence length
            validation_split: Fraction of data for validation
            random_seed: Random seed for reproducible splits
        """
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.validation_split = validation_split
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def load_from_annotations(self, annotation_file: str) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Load training examples from Module 2 annotation file.
        
        Args:
            annotation_file: Path to Module 2 annotation JSON file
            
        Returns:
            Tuple of (training_examples, validation_examples)
        """
        logger.info(f"Loading annotations from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        examples = []
        
        # Process each document
        for doc_data in data:
            document_id = doc_data.get("document_id", "unknown")
            
            # Handle different annotation formats
            if "token_annotations" in doc_data:
                # Token-level annotations format
                token_annotations = doc_data["token_annotations"]
                
                tokens = [ann["token"] for ann in token_annotations]
                labels = [ann.get("label", "O") for ann in token_annotations]
                
                # Extract clause types present in this document
                clause_types = list(set(
                    label.split("-")[1] for label in labels 
                    if label != "O" and "-" in label
                ))
                
                examples.append(TrainingExample(
                    tokens=tokens,
                    labels=labels,
                    document_id=document_id,
                    clause_types=clause_types
                ))
                
            elif "spans" in doc_data and "text" in doc_data:
                # Span-based annotations format
                text = doc_data["text"]
                spans = doc_data["spans"]
                
                # Simple tokenization (whitespace-based)
                # In production, use proper tokenizer
                tokens = text.split()
                
                # Convert spans to sequence labels
                labels = self.label_encoder.convert_spans_to_labels(
                    spans=[(s["start"], s["end"], s["clause_type"]) for s in spans],
                    sequence_length=len(tokens)
                )
                
                clause_types = list(set(s["clause_type"] for s in spans))
                
                examples.append(TrainingExample(
                    tokens=tokens,
                    labels=labels,
                    document_id=document_id,
                    clause_types=clause_types
                ))
        
        logger.info(f"Loaded {len(examples)} training examples")
        
        # Split into train/validation
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - self.validation_split))
        
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} validation")
        
        return train_examples, val_examples
    
    def load_from_module2_output(self, output_dir: str) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Load training data from Module 2 output directory.
        
        Args:
            output_dir: Path to Module 2 output directory
            
        Returns:
            Tuple of (training_examples, validation_examples)
        """
        import os
        
        # Look for annotation files
        annotation_files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.json') and 'annotation' in filename:
                annotation_files.append(os.path.join(output_dir, filename))
        
        if not annotation_files:
            raise FileNotFoundError(f"No annotation files found in {output_dir}")
        
        all_examples = []
        
        # Load from all annotation files
        for file_path in annotation_files:
            try:
                train_ex, val_ex = self.load_from_annotations(file_path)
                all_examples.extend(train_ex)
                all_examples.extend(val_ex)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_examples:
            raise ValueError(f"No valid examples loaded from {output_dir}")
        
        # Re-split all examples
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - self.validation_split))
        
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        return train_examples, val_examples
    
    def create_data_loaders(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            batch_size: Batch size
            num_workers: Number of data loader workers
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = ContractDataset(
            examples=train_examples,
            tokenizer=self.tokenizer,
            label_encoder=self.label_encoder,
            max_length=self.max_length,
            label_all_tokens=False
        )
        
        val_dataset = ContractDataset(
            examples=val_examples,
            tokenizer=self.tokenizer,
            label_encoder=self.label_encoder,
            max_length=self.max_length,
            label_all_tokens=False
        )
        
        # Create data collator
        data_collator = DataCollator(
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
            label_pad_token_id=-100  # Ignore in loss computation
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_dataset_statistics(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            examples: List of training examples
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_examples": len(examples),
            "avg_sequence_length": 0,
            "max_sequence_length": 0,
            "min_sequence_length": float('inf'),
            "clause_type_distribution": {},
            "label_distribution": {},
            "documents_by_clause_count": {}
        }
        
        if not examples:
            return stats
        
        sequence_lengths = []
        all_clause_types = []
        all_labels = []
        
        for example in examples:
            seq_len = len(example.tokens)
            sequence_lengths.append(seq_len)
            all_clause_types.extend(example.clause_types)
            all_labels.extend(example.labels)
        
        # Sequence length statistics
        stats["avg_sequence_length"] = np.mean(sequence_lengths)
        stats["max_sequence_length"] = max(sequence_lengths)
        stats["min_sequence_length"] = min(sequence_lengths)
        
        # Clause type distribution
        from collections import Counter
        clause_counter = Counter(all_clause_types)
        stats["clause_type_distribution"] = dict(clause_counter)
        
        # Label distribution
        label_counter = Counter(all_labels)
        stats["label_distribution"] = dict(label_counter)
        
        # Documents by clause count
        clause_counts = [len(ex.clause_types) for ex in examples]
        clause_count_dist = Counter(clause_counts)
        stats["documents_by_clause_count"] = dict(clause_count_dist)
        
        return stats