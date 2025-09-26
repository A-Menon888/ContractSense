"""
BERT-CRF model architecture for clause extraction from legal contracts.

This module implements a BERT-CRF model that combines:
1. BERT encoder for contextual understanding of legal text
2. CRF layer for sequence labeling with constraint enforcement
3. Custom loss functions for joint optimization
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF
from typing import Optional, Tuple, Dict, Any, List
import warnings
import logging

from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class BertCrfModel(BertPreTrainedModel):
    """
    BERT-CRF model for Named Entity Recognition in legal contracts.
    
    Combines BERT's contextual embeddings with CRF's sequence labeling
    capabilities to extract clause boundaries and types with high precision.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize BERT-CRF model.
        
        Args:
            config: Model configuration containing hyperparameters
        """
        # Initialize BERT config from pretrained model
        from transformers import BertConfig
        bert_config = BertConfig.from_pretrained(config.bert_model_name)
        bert_config.num_labels = config.num_labels
        
        super().__init__(bert_config)
        
        self.model_config = config
        self.num_labels = config.num_labels
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(
            config.bert_model_name,
            add_pooling_layer=False,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, config.num_labels)
        
        # CRF layer for sequence labeling
        self.crf = CRF(config.num_labels, batch_first=True)
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized BERT-CRF model with {config.num_labels} labels")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BERT-CRF model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type IDs for BERT
            labels: Ground truth labels of shape (batch_size, sequence_length)
            
        Returns:
            Dictionary containing loss, predictions, and other model outputs
        """
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Get sequence output (last hidden state)
        sequence_output = bert_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Classification logits
        logits = self.classifier(sequence_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # Training mode - compute loss
            loss = self._compute_loss(logits, labels, attention_mask)
            outputs["loss"] = loss
            
            # Get CRF predictions for evaluation
            predictions = self.crf.decode(logits, attention_mask.bool())
            outputs["predictions"] = predictions
        else:
            # Inference mode - get CRF predictions
            predictions = self.crf.decode(logits, attention_mask.bool())
            outputs["predictions"] = predictions
            
        return outputs
    
    def _compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CRF loss.
        
        Args:
            logits: Model logits of shape (batch_size, sequence_length, num_labels)
            labels: Ground truth labels of shape (batch_size, sequence_length)  
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            CRF loss tensor
        """
        # Convert attention mask to boolean
        mask = attention_mask.bool()
        
        # Compute negative log likelihood using CRF
        log_likelihood = self.crf(logits, labels, mask, reduction='mean')
        
        # CRF returns log likelihood, we want negative log likelihood as loss
        loss = -log_likelihood
        
        return loss
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions on input text.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            token_type_ids: Token type IDs for BERT
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing predictions and optional confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            predictions = outputs["predictions"]
            results = {"predictions": predictions}
            
            if return_confidence:
                # Compute confidence scores from logits
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                
                # Get confidence as max probability for each token
                confidence_scores = []
                for batch_idx, pred_seq in enumerate(predictions):
                    batch_confidences = []
                    for token_idx, pred_label in enumerate(pred_seq):
                        if token_idx < probs.shape[1]:
                            conf = probs[batch_idx, token_idx, pred_label].item()
                            batch_confidences.append(conf)
                    confidence_scores.append(batch_confidences)
                
                results["confidence_scores"] = confidence_scores
                
            return results
    
    def save_model(self, save_path: str) -> None:
        """
        Save model state dict and configuration.
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config.to_dict(),
            'num_labels': self.num_labels,
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, config: Optional[ModelConfig] = None) -> 'BertCrfModel':
        """
        Load model from saved state dict.
        
        Args:
            load_path: Path to load the model from
            config: Optional model configuration (will load from save if not provided)
            
        Returns:
            Loaded BertCrfModel instance
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        
        if config is None:
            config = ModelConfig.from_dict(checkpoint['model_config'])
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {load_path}")
        return model
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size information.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    def freeze_bert_layers(self, num_layers_to_freeze: int = 6) -> None:
        """
        Freeze the first N layers of BERT for transfer learning.
        
        Args:
            num_layers_to_freeze: Number of BERT layers to freeze
        """
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze specified number of encoder layers
        for i in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
                
        logger.info(f"Froze {num_layers_to_freeze} BERT layers")
    
    def unfreeze_all_layers(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        
        logger.info("Unfroze all model layers")


class BertCrfModelWithFeatures(BertCrfModel):
    """
    Extended BERT-CRF model with additional features for legal text.
    
    Adds positional, syntactic, and semantic features to improve
    clause extraction performance.
    """
    
    def __init__(self, config: ModelConfig, feature_dim: int = 50):
        """
        Initialize extended BERT-CRF model with features.
        
        Args:
            config: Model configuration
            feature_dim: Dimension of additional feature embeddings
        """
        super().__init__(config)
        
        self.feature_dim = feature_dim
        
        # Additional feature embeddings
        self.position_embeddings = nn.Embedding(512, feature_dim)
        self.sentence_position_embeddings = nn.Embedding(100, feature_dim)
        
        # Update classifier to handle additional features
        total_hidden_size = self.bert.config.hidden_size + 2 * feature_dim
        self.classifier = nn.Linear(total_hidden_size, config.num_labels)
        
        # Re-initialize weights for new layers
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for new layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sentence_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with additional features.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs for position embeddings
            sentence_positions: Sentence-level position IDs
            labels: Ground truth labels
            
        Returns:
            Model outputs with additional features
        """
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = bert_outputs.last_hidden_state
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # Add positional features if provided
        features_to_concat = [sequence_output]
        
        if position_ids is not None:
            pos_embeddings = self.position_embeddings(position_ids)
            features_to_concat.append(pos_embeddings)
        else:
            # Default position embeddings
            default_pos_ids = torch.arange(seq_len, device=sequence_output.device)
            default_pos_ids = default_pos_ids.unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = self.position_embeddings(default_pos_ids)
            features_to_concat.append(pos_embeddings)
            
        if sentence_positions is not None:
            sent_pos_embeddings = self.sentence_position_embeddings(sentence_positions)
            features_to_concat.append(sent_pos_embeddings)
        else:
            # Default sentence positions (all zeros)
            default_sent_pos = torch.zeros(batch_size, seq_len, dtype=torch.long, 
                                         device=sequence_output.device)
            sent_pos_embeddings = self.sentence_position_embeddings(default_sent_pos)
            features_to_concat.append(sent_pos_embeddings)
        
        # Concatenate all features
        enhanced_output = torch.cat(features_to_concat, dim=-1)
        enhanced_output = self.dropout(enhanced_output)
        
        # Classification
        logits = self.classifier(enhanced_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss = self._compute_loss(logits, labels, attention_mask)
            outputs["loss"] = loss
            predictions = self.crf.decode(logits, attention_mask.bool())
            outputs["predictions"] = predictions
        else:
            predictions = self.crf.decode(logits, attention_mask.bool())
            outputs["predictions"] = predictions
            
        return outputs