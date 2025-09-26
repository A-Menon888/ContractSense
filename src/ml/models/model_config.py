"""
Model configuration and hyperparameters for BERT-CRF clause extraction.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ModelConfig:
    """Configuration class for BERT-CRF model hyperparameters and settings."""
    
    # Model Architecture
    bert_model_name: str = "bert-base-uncased"
    num_labels: int = 31  # 15 clause types * 2 (B-, I-) + O = 31 labels for BIO scheme
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_sequence_length: int = 512
    
    # Training Configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    crf_learning_rate: float = 1e-3
    num_epochs: int = 20
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    
    # Validation and Early Stopping
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    save_best_model: bool = True
    metric_for_best_model: str = "f1_macro"
    
    # Data Processing
    labeling_scheme: str = "BIO"  # BIO, BIOS, or IOBES
    handle_misaligned_tokens: str = "average"  # average, first, last
    stride: int = 128  # for handling sequences longer than max_sequence_length
    
    # Model Persistence
    model_save_path: str = "models/bert_crf_latest.pt"
    checkpoint_dir: str = "models/checkpoints"
    config_save_path: str = "models/model_config.json"
    
    # Hardware and Performance
    device: str = "auto"  # auto, cpu, cuda
    fp16: bool = False  # mixed precision training
    dataloader_num_workers: int = 4
    
    # Logging and Monitoring
    log_level: str = "INFO"
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate labeling scheme
        if self.labeling_scheme not in ["BIO", "BIOS", "IOBES"]:
            raise ValueError(f"Unsupported labeling scheme: {self.labeling_scheme}")
            
        # Calculate num_labels based on scheme
        if self.labeling_scheme == "BIO":
            # 15 clause types * 2 (B-, I-) + O = 31
            self.num_labels = 31
        elif self.labeling_scheme == "BIOS": 
            # 15 clause types * 3 (B-, I-, S-) + O = 46
            self.num_labels = 46
        elif self.labeling_scheme == "IOBES":
            # 15 clause types * 4 (B-, I-, E-, S-) + O = 61  
            self.num_labels = 61
            
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config_save_path), exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """Save configuration to JSON file."""
        import json
        filepath = filepath or self.config_save_path
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configuration instance
default_config = ModelConfig()

# Predefined configurations for different scenarios
configs = {
    "development": ModelConfig(
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        validation_split=0.3,
        early_stopping_patience=2
    ),
    
    "production": ModelConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-5,
        validation_split=0.15,
        early_stopping_patience=5,
        fp16=True
    ),
    
    "fast_training": ModelConfig(
        batch_size=64,
        num_epochs=10,
        learning_rate=3e-5,
        gradient_accumulation_steps=2,
        fp16=True
    ),
    
    "high_quality": ModelConfig(
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-5,
        early_stopping_patience=10,
        labeling_scheme="IOBES",
        max_sequence_length=768
    )
}