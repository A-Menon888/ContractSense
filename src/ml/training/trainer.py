"""
Complete training pipeline for BERT-CRF clause extraction model.

Implements training loop with validation, checkpointing, and comprehensive
logging for contract clause extraction tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..models.bert_crf import BertCrfModel
from ..models.model_config import ModelConfig
from .data_loader import DataLoader, TrainingExample
from .metrics import SequenceLabelingMetrics, EvaluationResults
from ..utils.tokenization import ContractTokenizer
from ..utils.label_encoding import LabelEncoder

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (higher is better)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

class ModelTrainer:
    """
    Complete training pipeline for BERT-CRF clause extraction.
    
    Handles model training, validation, checkpointing, and evaluation
    with comprehensive logging and monitoring.
    """
    
    def __init__(
        self,
        model: BertCrfModel,
        config: ModelConfig,
        tokenizer: ContractTokenizer,
        label_encoder: LabelEncoder,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: BERT-CRF model instance
            config: Model configuration
            tokenizer: Contract tokenizer
            label_encoder: Label encoder
            device: Training device (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Initialize metrics calculator
        self.metrics_calculator = SequenceLabelingMetrics(label_encoder)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1_score = 0.0
        self.training_history = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=0.001
        )
        
        logger.info(f"Initialized ModelTrainer on device: {self.device}")
    
    def setup_optimizer_and_scheduler(
        self, 
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup optimizer and learning rate scheduler.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Separate parameters for different learning rates
        bert_params = []
        crf_params = []
        
        for name, param in self.model.named_parameters():
            if 'crf' in name:
                crf_params.append(param)
            else:
                bert_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = AdamW([
            {'params': bert_params, 'lr': self.config.learning_rate},
            {'params': crf_params, 'lr': self.config.crf_learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.num_epochs
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log periodically
            if self.global_step % self.config.log_steps == 0:
                logger.info(
                    f"Step {self.global_step}: Loss={loss.item():.4f}, "
                    f"LR={scheduler.get_last_lr()[0]:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }
    
    def evaluate(
        self, 
        val_loader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            return_predictions: Whether to return detailed predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_true_labels = []
        all_input_ids = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                predictions = outputs['predictions']
                
                total_loss += loss.item()
                
                # Convert predictions and labels back to strings
                for i, pred_sequence in enumerate(predictions):
                    # Get true labels (excluding padding)
                    attention_mask = batch['attention_mask'][i]
                    valid_length = attention_mask.sum().item()
                    
                    true_label_ids = batch['labels'][i][:valid_length].cpu().numpy()
                    pred_label_ids = pred_sequence[:valid_length]
                    
                    # Convert to label strings
                    true_labels = self.label_encoder.decode_labels(true_label_ids.tolist())
                    pred_labels = self.label_encoder.decode_labels(pred_label_ids)
                    
                    # Remove special tokens
                    input_tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(
                        batch['input_ids'][i][:valid_length]
                    )
                    
                    # Filter out special tokens
                    filtered_true = []
                    filtered_pred = []
                    
                    for j, (token, true_lbl, pred_lbl) in enumerate(zip(input_tokens, true_labels, pred_labels)):
                        if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                            filtered_true.append(true_lbl)
                            filtered_pred.append(pred_lbl)
                    
                    all_true_labels.append(filtered_true)
                    all_predictions.append(filtered_pred)
                    all_input_ids.append(batch['input_ids'][i][:valid_length])
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_calculator.compute_metrics(all_true_labels, all_predictions)
        
        results = {
            'val_loss': avg_loss,
            'precision': metrics.overall_precision,
            'recall': metrics.overall_recall,
            'f1': metrics.overall_f1,
            'accuracy': metrics.accuracy,
            'entity_results': metrics.entity_results
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['true_labels'] = all_true_labels
            results['input_ids'] = all_input_ids
        
        return results
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        save_dir: str = "models"
    ) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Save best model
            current_f1 = val_metrics['f1']
            if current_f1 > self.best_f1_score:
                self.best_f1_score = current_f1
                best_model_path = os.path.join(save_dir, "best_model.pt")
                self.save_checkpoint(best_model_path, epoch, optimizer, scheduler)
                logger.info(f"New best model saved with F1: {current_f1:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)
            
            # Early stopping check
            if self.early_stopping(current_f1):
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        # Final evaluation with detailed results
        logger.info("Training completed. Running final evaluation...")
        final_results = self.evaluate(val_loader, return_predictions=True)
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Generate classification report
        report = self.metrics_calculator.classification_report(
            final_results['true_labels'], 
            final_results['predictions']
        )
        
        logger.info("Classification Report:\n" + report)
        
        # Save classification report
        report_path = os.path.join(save_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_results,
            'best_f1_score': self.best_f1_score,
            'classification_report': report
        }
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1_score': self.best_f1_score,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_f1_score = checkpoint['best_f1_score']
        self.training_history = checkpoint['training_history']
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with model information
        """
        model_info = self.model.get_model_size()
        
        summary = {
            'model_name': 'BERT-CRF',
            'bert_model': self.config.bert_model_name,
            'num_labels': self.config.num_labels,
            'labeling_scheme': self.label_encoder.scheme.value,
            'max_sequence_length': self.config.max_sequence_length,
            'parameters': model_info,
            'device': str(self.device),
            'training_state': {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_f1_score': self.best_f1_score
            }
        }
        
        return summary