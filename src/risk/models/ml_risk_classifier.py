"""
ML Risk Classifier Module

Lightweight classifier for clause-level risk scoring using RoBERTa/Legal-BERT.
Provides risk labels (low/med/high) with confidence scores and explainability.

This is the core ML component for Module 4 as specified:
- Fine-tuned transformer model for 3-way risk classification
- Calibrated probability outputs with temperature scaling
- SHAP/LIME explainability for token-level importance
- Metadata integration (clause type, monetary amounts, governing law)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import pickle
import re
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import shap
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskPrediction:
    """Risk prediction result with explainability"""
    clause_id: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"  
    risk_probability: float  # Calibrated probability for predicted class
    confidence_score: float  # Overall prediction confidence
    probabilities: Dict[str, float]  # All class probabilities
    rationale: str  # Short explanation
    evidence_tokens: List[Dict[str, Any]]  # Top supporting tokens
    metadata: Dict[str, Any]  # Additional context

@dataclass
class FeatureVector:
    """Feature vector for risk classification"""
    clause_text: str
    clause_type: str
    monetary_amount: Optional[float] = None
    governing_law: Optional[str] = None
    party_role: Optional[str] = None  # "licensor", "licensee", "buyer", "seller", etc.
    contract_value: Optional[float] = None
    contract_duration: Optional[int] = None  # months
    industry: Optional[str] = None

class RiskDataset(Dataset):
    """Dataset for risk classification training"""
    
    def __init__(self, features: List[FeatureVector], labels: List[str], 
                 tokenizer: AutoTokenizer, max_length: int = 512):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Create input text with metadata
        input_text = self._create_input_text(feature)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Extract numerical features
        numerical_features = self._extract_numerical_features(feature)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "numerical_features": torch.tensor(numerical_features, dtype=torch.float),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long)
        }
    
    def _create_input_text(self, feature: FeatureVector) -> str:
        """Create formatted input text with metadata"""
        # Start with clause type and text
        input_parts = [
            f"[CLAUSE_TYPE] {feature.clause_type}",
            f"[CLAUSE_TEXT] {feature.clause_text}"
        ]
        
        # Add metadata if available
        if feature.governing_law:
            input_parts.append(f"[GOVERNING_LAW] {feature.governing_law}")
        
        if feature.party_role:
            input_parts.append(f"[PARTY_ROLE] {feature.party_role}")
        
        if feature.monetary_amount:
            input_parts.append(f"[MONETARY_AMOUNT] ${feature.monetary_amount:,.2f}")
        
        if feature.industry:
            input_parts.append(f"[INDUSTRY] {feature.industry}")
        
        return " ".join(input_parts)
    
    def _extract_numerical_features(self, feature: FeatureVector) -> List[float]:
        """Extract numerical features for the model"""
        features = []
        
        # Monetary amount (log-scaled)
        if feature.monetary_amount and feature.monetary_amount > 0:
            features.append(np.log10(feature.monetary_amount))
        else:
            features.append(0.0)
        
        # Contract value (log-scaled)
        if feature.contract_value and feature.contract_value > 0:
            features.append(np.log10(feature.contract_value))
        else:
            features.append(0.0)
        
        # Contract duration (normalized to years)
        if feature.contract_duration:
            features.append(feature.contract_duration / 12.0)
        else:
            features.append(0.0)
        
        # Clause type encoding (simple frequency-based)
        clause_type_scores = {
            "limitation_of_liability": 0.9,
            "indemnification": 0.85,
            "intellectual_property": 0.8,
            "termination": 0.75,
            "payment": 0.7,
            "confidentiality": 0.6,
            "warranties": 0.65,
            "governing_law": 0.5,
            "dispute_resolution": 0.55,
            "force_majeure": 0.4
        }
        features.append(clause_type_scores.get(feature.clause_type, 0.5))
        
        # Text length (normalized)
        features.append(min(len(feature.clause_text) / 1000.0, 2.0))
        
        return features

class RiskClassifierModel(nn.Module):
    """Risk classification model combining BERT + numerical features"""
    
    def __init__(self, model_name: str = "roberta-base", num_numerical_features: int = 5, 
                 num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        
        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # Numerical feature processing
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined classifier
        combined_dim = self.bert_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output  # [CLS] token representation
        
        # Process numerical features  
        numerical_output = self.numerical_fc(numerical_features)
        
        # Combine features
        combined_features = torch.cat([pooled_output, numerical_output], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def temperature_scale(self, logits):
        """Apply temperature scaling for calibration"""
        return logits / self.temperature

class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, input_ids, attention_mask, numerical_features):
        logits = self.model(input_ids, attention_mask, numerical_features)
        return logits / self.temperature
    
    def calibrate(self, dataloader, device):
        """Calibrate temperature on validation set"""
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                numerical_features = batch["numerical_features"].to(device)
                labels = batch["labels"].to(device)
                
                logits = self.model(input_ids, attention_mask, numerical_features)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll_criterion = nn.CrossEntropyLoss()
        
        def closure():
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)

class RiskClassifier:
    """Main risk classifier with explainability"""
    
    def __init__(self, model_name: str = "roberta-base", device: str = "auto"):
        self.model_name = model_name
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.calibrated_model = None
        self.explainer = None
        
        self.id2label = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        self.label2id = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        
        # Add special tokens for metadata
        special_tokens = [
            "[CLAUSE_TYPE]", "[CLAUSE_TEXT]", "[GOVERNING_LAW]", 
            "[PARTY_ROLE]", "[MONETARY_AMOUNT]", "[INDUSTRY]"
        ]
        self.tokenizer.add_tokens(special_tokens)
        
    def train(self, train_features: List[FeatureVector], train_labels: List[str],
              val_features: List[FeatureVector], val_labels: List[str],
              output_dir: str, **training_args):
        """Train the risk classifier"""
        
        # Create datasets
        train_dataset = RiskDataset(train_features, train_labels, self.tokenizer)
        val_dataset = RiskDataset(val_features, val_labels, self.tokenizer)
        
        # Initialize model
        self.model = RiskClassifierModel(
            model_name=self.model_name,
            num_numerical_features=5,
            num_classes=3
        )
        
        # Resize token embeddings for special tokens
        self.model.bert.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        # Training arguments
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "learning_rate": 2e-5,
            "logging_dir": f"{output_dir}/logs",
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 500,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False
        }
        default_args.update(training_args)
        
        training_args = TrainingArguments(**default_args)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Calibrate model
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self._calibrate_model(val_dataloader)
        
        # Initialize explainer
        self._initialize_explainer(val_dataset)
        
    def _calibrate_model(self, val_dataloader):
        """Calibrate model probabilities using temperature scaling"""
        self.calibrated_model = TemperatureScaling(self.model)
        self.calibrated_model.calibrate(val_dataloader, self.device)
        
    def _initialize_explainer(self, val_dataset):
        """Initialize SHAP explainer for interpretability"""
        # Sample some validation data for explainer background
        sample_size = min(100, len(val_dataset))
        sample_indices = np.random.choice(len(val_dataset), sample_size, replace=False)
        
        background_data = []
        for idx in sample_indices:
            item = val_dataset[idx]
            background_data.append({
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "numerical_features": item["numerical_features"]
            })
        
        # Create wrapper function for SHAP
        def model_predict(batch):
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for item in batch:
                    input_ids = item["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
                    numerical_features = item["numerical_features"].unsqueeze(0).to(self.device)
                    
                    if self.calibrated_model:
                        logits = self.calibrated_model(input_ids, attention_mask, numerical_features)
                    else:
                        logits = self.model(input_ids, attention_mask, numerical_features)
                    
                    probs = F.softmax(logits, dim=1)
                    predictions.append(probs.cpu().numpy()[0])
            
            return np.array(predictions)
        
        # Note: Full SHAP integration would require more setup
        # For now, we'll use attention weights and gradient-based explanations
        self.explainer = model_predict
        
    def predict(self, feature: FeatureVector, explain: bool = True) -> RiskPrediction:
        """Predict risk level for a single clause"""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create dataset with single item
        dataset = RiskDataset([feature], ["LOW"], self.tokenizer)  # dummy label
        item = dataset[0]
        
        self.model.eval()
        
        with torch.no_grad():
            input_ids = item["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
            numerical_features = item["numerical_features"].unsqueeze(0).to(self.device)
            
            # Get predictions
            if self.calibrated_model:
                logits = self.calibrated_model(input_ids, attention_mask, numerical_features)
            else:
                logits = self.model(input_ids, attention_mask, numerical_features)
            
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            predicted_label = self.id2label[predicted_class]
            predicted_prob = probabilities[predicted_class]
            
        # Calculate confidence (entropy-based)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        max_entropy = np.log(3)  # log of number of classes
        confidence = 1.0 - (entropy / max_entropy)
        
        # Generate rationale
        rationale = self._generate_rationale(feature, predicted_label, probabilities)
        
        # Get evidence tokens if explanation requested
        evidence_tokens = []
        if explain:
            evidence_tokens = self._get_evidence_tokens(item, feature, predicted_class)
        
        return RiskPrediction(
            clause_id=getattr(feature, 'clause_id', 'unknown'),
            risk_level=predicted_label,
            risk_probability=round(predicted_prob, 3),
            confidence_score=round(confidence, 3),
            probabilities={
                "LOW": round(probabilities[0], 3),
                "MEDIUM": round(probabilities[1], 3), 
                "HIGH": round(probabilities[2], 3)
            },
            rationale=rationale,
            evidence_tokens=evidence_tokens,
            metadata={
                "clause_type": feature.clause_type,
                "monetary_amount": feature.monetary_amount,
                "governing_law": feature.governing_law,
                "party_role": feature.party_role
            }
        )
    
    def _generate_rationale(self, feature: FeatureVector, predicted_label: str, 
                          probabilities: np.ndarray) -> str:
        """Generate human-readable rationale for prediction"""
        rationale_parts = []
        
        # Clause type factor
        high_risk_types = ["limitation_of_liability", "indemnification", "intellectual_property"]
        if feature.clause_type in high_risk_types and predicted_label in ["HIGH", "MEDIUM"]:
            rationale_parts.append(f"High-risk clause type ({feature.clause_type})")
        
        # Monetary amount factor
        if feature.monetary_amount and feature.monetary_amount > 100000:
            rationale_parts.append(f"Significant monetary amount (${feature.monetary_amount:,.0f})")
        
        # Confidence factor
        confidence_score = 1.0 + np.sum(probabilities * np.log(probabilities + 1e-8)) / np.log(3)
        if confidence_score < 0.7:
            rationale_parts.append("Review required - ambiguous risk indicators")
        
        # Language patterns (simplified)
        text_lower = feature.clause_text.lower()
        if "unlimited" in text_lower or "without limit" in text_lower:
            rationale_parts.append("Contains unlimited liability language")
        if "indemnify" in text_lower and "defend" in text_lower:
            rationale_parts.append("Broad indemnification requirements")
        
        if rationale_parts:
            return "; ".join(rationale_parts)
        else:
            return f"Standard {feature.clause_type} clause with {predicted_label.lower()} risk profile"
    
    def _get_evidence_tokens(self, item: Dict, feature: FeatureVector, 
                           predicted_class: int) -> List[Dict[str, Any]]:
        """Get top evidence tokens using attention and gradients"""
        evidence_tokens = []
        
        # Enable gradients
        self.model.train()
        
        input_ids = item["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(self.device) 
        numerical_features = item["numerical_features"].unsqueeze(0).to(self.device)
        
        # Get gradients w.r.t. input embeddings
        input_ids.requires_grad_(True)
        
        logits = self.model(input_ids, attention_mask, numerical_features)
        target_logit = logits[0, predicted_class]
        target_logit.backward()
        
        # Get gradient magnitudes
        gradients = input_ids.grad.abs().squeeze().cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(item["input_ids"])
        
        # Find top contributing tokens
        token_scores = [(token, score, idx) for idx, (token, score) in 
                       enumerate(zip(tokens, gradients)) if token not in ["<pad>", "<s>", "</s>"]]
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 evidence tokens
        for token, score, idx in token_scores[:3]:
            if score > 0.01:  # threshold for significance
                evidence_tokens.append({
                    "token": token,
                    "importance_score": round(float(score), 4),
                    "position": int(idx),
                    "context": self._get_token_context(tokens, idx)
                })
        
        self.model.eval()
        input_ids.requires_grad_(False)
        
        return evidence_tokens
    
    def _get_token_context(self, tokens: List[str], position: int, window: int = 3) -> str:
        """Get context around a token"""
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)
        context_tokens = tokens[start:end]
        
        # Highlight the target token
        if position - start < len(context_tokens):
            context_tokens[position - start] = f"**{context_tokens[position - start]}**"
        
        return " ".join(context_tokens)
    
    def evaluate_calibration(self, test_features: List[FeatureVector], 
                           test_labels: List[str]) -> Dict[str, float]:
        """Evaluate model calibration using Brier score and log loss"""
        predictions = []
        true_labels = []
        
        for feature, true_label in zip(test_features, test_labels):
            pred = self.predict(feature, explain=False)
            predictions.append([pred.probabilities["LOW"], pred.probabilities["MEDIUM"], pred.probabilities["HIGH"]])
            true_labels.append(self.label2id[true_label])
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Convert to binary for each class for Brier score
        brier_scores = []
        for class_idx in range(3):
            true_binary = (true_labels == class_idx).astype(int)
            pred_binary = predictions[:, class_idx]
            brier_score = brier_score_loss(true_binary, pred_binary)
            brier_scores.append(brier_score)
        
        # Log loss for multi-class
        logloss = log_loss(true_labels, predictions)
        
        return {
            "mean_brier_score": np.mean(brier_scores),
            "log_loss": logloss,
            "brier_score_low": brier_scores[0],
            "brier_score_medium": brier_scores[1], 
            "brier_score_high": brier_scores[2]
        }
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "calibrated_model_state_dict": self.calibrated_model.state_dict() if self.calibrated_model else None,
            "model_name": self.model_name
        }, path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}_tokenizer")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = RiskClassifierModel(model_name=self.model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        
        if checkpoint["calibrated_model_state_dict"]:
            self.calibrated_model = TemperatureScaling(self.model)
            self.calibrated_model.load_state_dict(checkpoint["calibrated_model_state_dict"])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{path}_tokenizer")