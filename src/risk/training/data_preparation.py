"""
Training Data Preparation for Risk Classifier

Integrates with Module 2 and Module 3 outputs to prepare training data
for the ML-based risk classifier.

This module:
- Loads clause annotations from Module 2 (clause types)
- Loads risk annotations from Module 3 (if available)
- Creates labeled training data for risk classification
- Handles data augmentation and balancing
"""

import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import random
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """Single training example for risk classification"""
    clause_id: str
    clause_text: str
    clause_type: str
    risk_label: str  # LOW, MEDIUM, HIGH
    
    # Optional features
    monetary_amount: Optional[float] = None
    governing_law: Optional[str] = None
    party_role: Optional[str] = None
    contract_value: Optional[float] = None
    contract_duration: Optional[int] = None
    industry: Optional[str] = None
    
    # Metadata
    document_id: str = ""
    annotation_source: str = ""  # "manual", "rule_based", "synthetic"
    confidence: float = 1.0

class RiskTrainingDataPreparer:
    """Prepare training data for risk classification"""
    
    def __init__(self, cuad_data_path: str, module2_output_path: str, 
                 module3_output_path: Optional[str] = None):
        self.cuad_data_path = Path(cuad_data_path)
        self.module2_output_path = Path(module2_output_path)
        self.module3_output_path = Path(module3_output_path) if module3_output_path else None
        
        # Risk mapping for clause types (rule-based initial labeling)
        self.clause_type_risk_mapping = {
            # High-risk clause types
            "limitation_of_liability": "HIGH",
            "indemnification": "HIGH",
            "intellectual_property": "HIGH",
            "termination": "HIGH",
            "liquidated_damages": "HIGH",
            "most_favored_nation": "HIGH",
            "non_compete": "HIGH",
            "exclusivity": "HIGH",
            "change_of_control": "HIGH",
            
            # Medium-risk clause types
            "governing_law": "MEDIUM",
            "dispute_resolution": "MEDIUM",
            "confidentiality": "MEDIUM",
            "assignment": "MEDIUM",
            "force_majeure": "MEDIUM",
            "amendment": "MEDIUM",
            "notice": "MEDIUM",
            "warranty": "MEDIUM",
            "compliance": "MEDIUM",
            "insurance": "MEDIUM",
            
            # Low-risk clause types (default)
            "effective_date": "LOW",
            "expiration_date": "LOW",
            "renewal": "LOW",
            "counterparts": "LOW",
            "headings": "LOW",
            "severability": "LOW"
        }
        
        # Risk keywords that can override default mappings
        self.high_risk_keywords = {
            "unlimited liability", "personal liability", "criminal liability",
            "indemnify", "hold harmless", "defend",
            "liquidated damages", "penalty", "punitive damages",
            "non-compete", "non-solicitation", "restraint of trade",
            "exclusive", "sole", "only", "exclusively",
            "irrevocable", "perpetual", "permanent",
            "breach", "default", "violation", "failure to perform",
            "injunction", "specific performance", "equitable relief"
        }
        
        self.medium_risk_keywords = {
            "material breach", "cure period", "notice",
            "reasonable efforts", "best efforts", "commercially reasonable",
            "confidential", "proprietary", "trade secret",
            "assignment", "transfer", "delegate",
            "force majeure", "act of god", "unforeseeable",
            "governing law", "jurisdiction", "dispute",
            "warranty", "represent", "guarantee"
        }
    
    def prepare_training_data(self, train_split: float = 0.7, val_split: float = 0.15,
                            test_split: float = 0.15, balance_classes: bool = True,
                            min_text_length: int = 50) -> Tuple[List[TrainingExample], 
                                                              List[TrainingExample], 
                                                              List[TrainingExample]]:
        """
        Prepare complete training, validation, and test datasets
        """
        
        logger.info("Loading and processing data sources...")
        
        # Load all available data sources
        examples = []
        
        # Load CUAD annotations from Module 2
        module2_examples = self._load_module2_data()
        examples.extend(module2_examples)
        logger.info(f"Loaded {len(module2_examples)} examples from Module 2")
        
        # Load risk annotations from Module 3 (if available)
        if self.module3_output_path and self.module3_output_path.exists():
            module3_examples = self._load_module3_data()
            examples.extend(module3_examples)
            logger.info(f"Loaded {len(module3_examples)} examples from Module 3")
        
        # Generate synthetic examples for data augmentation
        synthetic_examples = self._generate_synthetic_examples()
        examples.extend(synthetic_examples)
        logger.info(f"Generated {len(synthetic_examples)} synthetic examples")
        
        # Filter examples
        examples = [ex for ex in examples if len(ex.clause_text) >= min_text_length]
        logger.info(f"Filtered to {len(examples)} examples after length filtering")
        
        # Balance classes if requested
        if balance_classes:
            examples = self._balance_classes(examples)
            logger.info(f"Balanced dataset to {len(examples)} examples")
        
        # Split data
        train_examples, val_examples, test_examples = self._split_data(
            examples, train_split, val_split, test_split
        )
        
        logger.info(f"Data splits: Train={len(train_examples)}, "
                   f"Val={len(val_examples)}, Test={len(test_examples)}")
        
        return train_examples, val_examples, test_examples
    
    def _load_module2_data(self) -> List[TrainingExample]:
        """Load clause annotations from Module 2 output"""
        examples = []
        
        try:
            # Look for processed JSON files in Module 2 output
            for json_file in self.module2_output_path.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                document_id = json_file.stem
                
                # Extract clauses and their types
                for clause_id, clause_info in doc_data.get("clauses", {}).items():
                    clause_text = clause_info.get("text", "")
                    clause_type = clause_info.get("type", "unknown")
                    
                    if clause_text and clause_type != "unknown":
                        # Map clause type to risk label
                        risk_label = self._map_clause_to_risk(clause_type, clause_text)
                        
                        # Extract additional features
                        features = self._extract_features_from_clause(clause_info)
                        
                        example = TrainingExample(
                            clause_id=clause_id,
                            clause_text=clause_text,
                            clause_type=clause_type,
                            risk_label=risk_label,
                            document_id=document_id,
                            annotation_source="module2",
                            **features
                        )
                        examples.append(example)
        
        except Exception as e:
            logger.error(f"Error loading Module 2 data: {e}")
        
        return examples
    
    def _load_module3_data(self) -> List[TrainingExample]:
        """Load risk annotations from Module 3 output"""
        examples = []
        
        try:
            # Look for risk annotation files in Module 3 output
            risk_files = list(self.module3_output_path.glob("*risk*.json"))
            
            for json_file in risk_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    risk_data = json.load(f)
                
                document_id = json_file.stem
                
                # Extract risk-annotated clauses
                for clause_id, risk_info in risk_data.get("clause_risks", {}).items():
                    clause_text = risk_info.get("text", "")
                    clause_type = risk_info.get("type", "unknown")
                    risk_label = risk_info.get("risk_level", "MEDIUM")
                    confidence = risk_info.get("confidence", 1.0)
                    
                    if clause_text:
                        example = TrainingExample(
                            clause_id=clause_id,
                            clause_text=clause_text,
                            clause_type=clause_type,
                            risk_label=risk_label,
                            document_id=document_id,
                            annotation_source="module3",
                            confidence=confidence
                        )
                        examples.append(example)
        
        except Exception as e:
            logger.error(f"Error loading Module 3 data: {e}")
        
        return examples
    
    def _map_clause_to_risk(self, clause_type: str, clause_text: str) -> str:
        """Map clause type and text to risk level"""
        
        # Start with default mapping
        base_risk = self.clause_type_risk_mapping.get(clause_type, "MEDIUM")
        
        # Check for risk keywords that might override the base mapping
        text_lower = clause_text.lower()
        
        # High-risk keywords can escalate to HIGH
        for keyword in self.high_risk_keywords:
            if keyword in text_lower:
                return "HIGH"
        
        # Medium-risk keywords can escalate LOW to MEDIUM
        if base_risk == "LOW":
            for keyword in self.medium_risk_keywords:
                if keyword in text_lower:
                    return "MEDIUM"
        
        return base_risk
    
    def _extract_features_from_clause(self, clause_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional features from clause information"""
        features = {}
        
        # Extract monetary amounts
        if "monetary_amount" in clause_info:
            features["monetary_amount"] = clause_info["monetary_amount"]
        
        # Extract governing law
        if "governing_law" in clause_info:
            features["governing_law"] = clause_info["governing_law"]
        
        # Extract party role
        if "party_role" in clause_info:
            features["party_role"] = clause_info["party_role"]
        
        return features
    
    def _generate_synthetic_examples(self) -> List[TrainingExample]:
        """Generate synthetic training examples for data augmentation"""
        synthetic_examples = []
        
        # High-risk synthetic examples
        high_risk_templates = [
            "The Company shall indemnify and hold harmless the Client from any and all claims, damages, losses, and expenses arising out of the Company's performance under this Agreement.",
            "Notwithstanding anything herein to the contrary, neither Party shall be liable for any indirect, incidental, special, or consequential damages.",
            "The Company grants Client an exclusive license to use the intellectual property described herein.",
            "This Agreement may be terminated immediately by either party upon material breach without cure period.",
            "The Company agrees to pay liquidated damages of $100,000 for each day of delay in performance."
        ]
        
        for i, template in enumerate(high_risk_templates):
            example = TrainingExample(
                clause_id=f"synthetic_high_{i}",
                clause_text=template,
                clause_type="synthetic",
                risk_label="HIGH",
                annotation_source="synthetic",
                confidence=0.8
            )
            synthetic_examples.append(example)
        
        # Medium-risk synthetic examples  
        medium_risk_templates = [
            "All disputes arising under this Agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.",
            "The parties acknowledge that confidential information may be disclosed during the performance of this Agreement.",
            "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
            "The Company warrants that its services will be performed in a workmanlike manner in accordance with industry standards.",
            "Either party may assign this Agreement with the prior written consent of the other party, such consent not to be unreasonably withheld."
        ]
        
        for i, template in enumerate(medium_risk_templates):
            example = TrainingExample(
                clause_id=f"synthetic_medium_{i}",
                clause_text=template,
                clause_type="synthetic",
                risk_label="MEDIUM",
                annotation_source="synthetic",
                confidence=0.8
            )
            synthetic_examples.append(example)
        
        # Low-risk synthetic examples
        low_risk_templates = [
            "This Agreement shall become effective on the date last signed by the parties.",
            "This Agreement may be executed in multiple counterparts, each of which shall constitute an original.",
            "The headings in this Agreement are for convenience only and shall not affect the interpretation hereof.",
            "If any provision of this Agreement is found to be unenforceable, the remainder shall remain in full force and effect.",
            "All notices required hereunder shall be in writing and delivered by certified mail to the addresses set forth herein."
        ]
        
        for i, template in enumerate(low_risk_templates):
            example = TrainingExample(
                clause_id=f"synthetic_low_{i}",
                clause_text=template,
                clause_type="synthetic",
                risk_label="LOW",
                annotation_source="synthetic",
                confidence=0.8
            )
            synthetic_examples.append(example)
        
        return synthetic_examples
    
    def _balance_classes(self, examples: List[TrainingExample], 
                        target_ratio: Dict[str, float] = None) -> List[TrainingExample]:
        """Balance classes by upsampling minority classes"""
        
        if target_ratio is None:
            target_ratio = {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
        
        # Count current distribution
        label_counts = Counter(ex.risk_label for ex in examples)
        total_examples = len(examples)
        
        logger.info(f"Original distribution: {dict(label_counts)}")
        
        # Group examples by label
        examples_by_label = defaultdict(list)
        for ex in examples:
            examples_by_label[ex.risk_label].append(ex)
        
        # Calculate target counts
        target_total = int(total_examples * 1.2)  # Slight increase in total
        target_counts = {
            label: int(target_total * ratio) 
            for label, ratio in target_ratio.items()
        }
        
        balanced_examples = []
        
        for label, target_count in target_counts.items():
            current_examples = examples_by_label[label]
            current_count = len(current_examples)
            
            if current_count == 0:
                continue
            
            if current_count >= target_count:
                # Downsample
                selected = random.sample(current_examples, target_count)
            else:
                # Upsample by repeating examples
                selected = current_examples.copy()
                while len(selected) < target_count:
                    needed = min(target_count - len(selected), current_count)
                    selected.extend(random.sample(current_examples, needed))
            
            balanced_examples.extend(selected)
        
        random.shuffle(balanced_examples)
        
        final_counts = Counter(ex.risk_label for ex in balanced_examples)
        logger.info(f"Balanced distribution: {dict(final_counts)}")
        
        return balanced_examples
    
    def _split_data(self, examples: List[TrainingExample], 
                   train_split: float, val_split: float, test_split: float) -> Tuple[List[TrainingExample], 
                                                                                   List[TrainingExample], 
                                                                                   List[TrainingExample]]:
        """Split data into train/validation/test sets"""
        
        # Ensure splits sum to 1.0
        total_split = train_split + val_split + test_split
        train_split /= total_split
        val_split /= total_split
        test_split /= total_split
        
        # Shuffle examples
        random.shuffle(examples)
        
        total_count = len(examples)
        train_count = int(total_count * train_split)
        val_count = int(total_count * val_split)
        
        train_examples = examples[:train_count]
        val_examples = examples[train_count:train_count + val_count]
        test_examples = examples[train_count + val_count:]
        
        return train_examples, val_examples, test_examples
    
    def export_training_data(self, train_examples: List[TrainingExample],
                           val_examples: List[TrainingExample],
                           test_examples: List[TrainingExample],
                           output_dir: str):
        """Export training data to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionaries for JSON serialization
        def examples_to_dict(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
            return [
                {
                    "clause_id": ex.clause_id,
                    "clause_text": ex.clause_text,
                    "clause_type": ex.clause_type,
                    "risk_label": ex.risk_label,
                    "monetary_amount": ex.monetary_amount,
                    "governing_law": ex.governing_law,
                    "party_role": ex.party_role,
                    "contract_value": ex.contract_value,
                    "contract_duration": ex.contract_duration,
                    "industry": ex.industry,
                    "document_id": ex.document_id,
                    "annotation_source": ex.annotation_source,
                    "confidence": ex.confidence
                }
                for ex in examples
            ]
        
        # Save train data
        with open(output_path / "train_data.json", 'w', encoding='utf-8') as f:
            json.dump(examples_to_dict(train_examples), f, indent=2, ensure_ascii=False)
        
        # Save validation data
        with open(output_path / "val_data.json", 'w', encoding='utf-8') as f:
            json.dump(examples_to_dict(val_examples), f, indent=2, ensure_ascii=False)
        
        # Save test data
        with open(output_path / "test_data.json", 'w', encoding='utf-8') as f:
            json.dump(examples_to_dict(test_examples), f, indent=2, ensure_ascii=False)
        
        # Save dataset statistics
        stats = {
            "total_examples": len(train_examples) + len(val_examples) + len(test_examples),
            "train_count": len(train_examples),
            "val_count": len(val_examples),
            "test_count": len(test_examples),
            "train_distribution": dict(Counter(ex.risk_label for ex in train_examples)),
            "val_distribution": dict(Counter(ex.risk_label for ex in val_examples)),
            "test_distribution": dict(Counter(ex.risk_label for ex in test_examples)),
            "annotation_sources": dict(Counter(ex.annotation_source for ex in train_examples + val_examples + test_examples))
        }
        
        with open(output_path / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training data exported to {output_dir}")
        logger.info(f"Dataset statistics: {stats}")
    
    def create_training_config(self, output_dir: str) -> Dict[str, Any]:
        """Create configuration for training the risk classifier"""
        
        config = {
            "model_config": {
                "model_name": "nlpaueb/legal-bert-base-uncased",
                "num_labels": 3,  # LOW, MEDIUM, HIGH
                "max_length": 512,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1
            },
            "training_config": {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "num_epochs": 10,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "eval_steps": 100,
                "save_steps": 500,
                "logging_steps": 50
            },
            "calibration_config": {
                "temperature_scaling": True,
                "isotonic_regression": True,
                "platt_scaling": False
            },
            "explainability_config": {
                "enable_shap": True,
                "enable_lime": False,  # SHAP preferred for transformers
                "max_explanation_length": 100
            },
            "evaluation_config": {
                "compute_metrics": ["accuracy", "f1_macro", "f1_per_class", "precision", "recall"],
                "calibration_metrics": ["brier_score", "ece", "mce", "reliability_diagram"]
            }
        }
        
        # Save config
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved to {config_path}")
        return config