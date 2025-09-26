"""
Label encoding and decoding utilities for BERT-CRF model.

Handles conversion between clause type annotations and model labels
for different tagging schemes (BIO, BIOS, IOBES).
"""

from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class TaggingScheme(Enum):
    """Supported sequence tagging schemes."""
    BIO = "BIO"
    BIOS = "BIOS" 
    IOBES = "IOBES"

@dataclass
class LabelInfo:
    """Information about a label in the tagging scheme."""
    id: int
    name: str
    description: str
    clause_type: Optional[str] = None
    tag_type: Optional[str] = None  # B, I, O, S, E

class LabelEncoder:
    """
    Encodes and decodes labels for sequence labeling tasks.
    
    Converts between human-readable clause type annotations and
    numeric labels required by the BERT-CRF model.
    """
    
    def __init__(self, scheme: TaggingScheme = TaggingScheme.BIO):
        """
        Initialize label encoder.
        
        Args:
            scheme: Tagging scheme to use (BIO, BIOS, IOBES)
        """
        self.scheme = scheme
        
        # Clause types from Module 2 schema
        self.clause_types = [
            "termination",
            "payment", 
            "intellectual_property",
            "confidentiality",
            "limitation_of_liability",
            "indemnification",
            "force_majeure", 
            "governing_law",
            "dispute_resolution",
            "non_compete",
            "data_protection",
            "warranty",
            "service_level",
            "compliance",
            "amendment"
        ]
        
        # Build label mappings
        self._build_label_mappings()
        
        logger.info(f"Initialized LabelEncoder with {scheme.value} scheme, "
                   f"{len(self.label2id)} labels total")
    
    def _build_label_mappings(self) -> None:
        """Build bidirectional mappings between labels and IDs."""
        self.label_info: Dict[str, LabelInfo] = {}
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        
        label_id = 0
        
        # Outside label (always present)
        outside_label = "O"
        self.label_info[outside_label] = LabelInfo(
            id=label_id,
            name=outside_label,
            description="Outside any clause"
        )
        self.label2id[outside_label] = label_id
        self.id2label[label_id] = outside_label
        label_id += 1
        
        # Generate labels based on tagging scheme
        for clause_type in self.clause_types:
            if self.scheme == TaggingScheme.BIO:
                # B-<type>, I-<type>
                for tag_type in ['B', 'I']:
                    label = f"{tag_type}-{clause_type}"
                    self.label_info[label] = LabelInfo(
                        id=label_id,
                        name=label,
                        description=f"{tag_type} tag for {clause_type}",
                        clause_type=clause_type,
                        tag_type=tag_type
                    )
                    self.label2id[label] = label_id
                    self.id2label[label_id] = label
                    label_id += 1
                    
            elif self.scheme == TaggingScheme.BIOS:
                # B-<type>, I-<type>, S-<type> 
                for tag_type in ['B', 'I', 'S']:
                    label = f"{tag_type}-{clause_type}"
                    self.label_info[label] = LabelInfo(
                        id=label_id,
                        name=label,
                        description=f"{tag_type} tag for {clause_type}",
                        clause_type=clause_type,
                        tag_type=tag_type
                    )
                    self.label2id[label] = label_id
                    self.id2label[label_id] = label
                    label_id += 1
                    
            elif self.scheme == TaggingScheme.IOBES:
                # B-<type>, I-<type>, E-<type>, S-<type>
                for tag_type in ['B', 'I', 'E', 'S']:
                    label = f"{tag_type}-{clause_type}"
                    self.label_info[label] = LabelInfo(
                        id=label_id,
                        name=label,
                        description=f"{tag_type} tag for {clause_type}",
                        clause_type=clause_type,
                        tag_type=tag_type
                    )
                    self.label2id[label] = label_id
                    self.id2label[label_id] = label
                    label_id += 1
    
    @property
    def num_labels(self) -> int:
        """Get total number of labels."""
        return len(self.label2id)
    
    def encode_labels(self, annotation_labels: List[str]) -> List[int]:
        """
        Convert annotation labels to model label IDs.
        
        Args:
            annotation_labels: List of annotation labels (e.g., ['O', 'B-termination', 'I-termination'])
            
        Returns:
            List of label IDs
        """
        label_ids = []
        for label in annotation_labels:
            if label in self.label2id:
                label_ids.append(self.label2id[label])
            else:
                logger.warning(f"Unknown label '{label}', using O instead")
                label_ids.append(self.label2id['O'])
        return label_ids
    
    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """
        Convert model label IDs back to annotation labels.
        
        Args:
            label_ids: List of label IDs
            
        Returns:
            List of annotation labels
        """
        labels = []
        for label_id in label_ids:
            if label_id in self.id2label:
                labels.append(self.id2label[label_id])
            else:
                logger.warning(f"Unknown label ID {label_id}, using O instead") 
                labels.append('O')
        return labels
    
    def convert_spans_to_labels(
        self, 
        spans: List[Tuple[int, int, str]], 
        sequence_length: int
    ) -> List[str]:
        """
        Convert span annotations to sequence labels.
        
        Args:
            spans: List of (start, end, clause_type) tuples
            sequence_length: Length of the token sequence
            
        Returns:
            List of sequence labels
        """
        labels = ['O'] * sequence_length
        
        # Sort spans by start position to handle overlaps
        spans = sorted(spans, key=lambda x: x[0])
        
        for start, end, clause_type in spans:
            # Validate span boundaries
            if start < 0 or end > sequence_length or start >= end:
                logger.warning(f"Invalid span ({start}, {end}) for sequence length {sequence_length}")
                continue
                
            if clause_type not in self.clause_types:
                logger.warning(f"Unknown clause type '{clause_type}', skipping span")
                continue
            
            # Apply tagging scheme
            if self.scheme == TaggingScheme.BIO:
                if start == end - 1:
                    # Single token span
                    labels[start] = f"B-{clause_type}"
                else:
                    # Multi-token span
                    labels[start] = f"B-{clause_type}"
                    for i in range(start + 1, end):
                        labels[i] = f"I-{clause_type}"
                        
            elif self.scheme == TaggingScheme.BIOS:
                if start == end - 1:
                    # Single token span
                    labels[start] = f"S-{clause_type}"
                else:
                    # Multi-token span
                    labels[start] = f"B-{clause_type}"
                    for i in range(start + 1, end):
                        labels[i] = f"I-{clause_type}"
                        
            elif self.scheme == TaggingScheme.IOBES:
                if start == end - 1:
                    # Single token span
                    labels[start] = f"S-{clause_type}"
                else:
                    # Multi-token span
                    labels[start] = f"B-{clause_type}"
                    for i in range(start + 1, end - 1):
                        labels[i] = f"I-{clause_type}"
                    labels[end - 1] = f"E-{clause_type}"
        
        return labels
    
    def convert_labels_to_spans(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """
        Convert sequence labels back to span annotations.
        
        Args:
            labels: List of sequence labels
            
        Returns:
            List of (start, end, clause_type) tuples
        """
        spans = []
        current_span = None
        
        for i, label in enumerate(labels):
            if label == 'O':
                # Close current span if any
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                    
            elif label.startswith('B-'):
                # Close current span if any
                if current_span is not None:
                    spans.append(current_span)
                
                # Start new span
                clause_type = label[2:]  # Remove 'B-' prefix
                current_span = (i, i + 1, clause_type)
                
            elif label.startswith('I-'):
                # Continue current span
                if current_span is not None:
                    clause_type = label[2:]  # Remove 'I-' prefix
                    if current_span[2] == clause_type:
                        # Extend current span
                        current_span = (current_span[0], i + 1, clause_type)
                    else:
                        # Type mismatch, close current and start new
                        spans.append(current_span)
                        current_span = (i, i + 1, clause_type)
                else:
                    # I- without B-, start new span
                    clause_type = label[2:]
                    current_span = (i, i + 1, clause_type)
                    
            elif label.startswith('E-'):  # IOBES scheme
                # End current span
                clause_type = label[2:]
                if current_span is not None and current_span[2] == clause_type:
                    # Extend and close current span
                    current_span = (current_span[0], i + 1, clause_type)
                    spans.append(current_span)
                    current_span = None
                else:
                    # E- without matching B-, create single token span
                    spans.append((i, i + 1, clause_type))
                    
            elif label.startswith('S-'):  # BIOS and IOBES schemes
                # Single token span
                clause_type = label[2:]
                spans.append((i, i + 1, clause_type))
                
                # Close current span if any
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
        
        # Close final span if any
        if current_span is not None:
            spans.append(current_span)
        
        return spans
    
    def get_clause_types(self) -> List[str]:
        """Get list of supported clause types."""
        return self.clause_types.copy()
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the label encoding.
        
        Returns:
            Dictionary with label statistics
        """
        stats = {
            "scheme": self.scheme.value,
            "num_labels": self.num_labels,
            "num_clause_types": len(self.clause_types),
            "clause_types": self.clause_types.copy(),
            "labels_by_type": {}
        }
        
        # Count labels by tag type
        tag_type_counts = {}
        for label_info in self.label_info.values():
            tag_type = label_info.tag_type or "O"
            tag_type_counts[tag_type] = tag_type_counts.get(tag_type, 0) + 1
        
        stats["labels_by_tag_type"] = tag_type_counts
        
        return stats
    
    def validate_labels(self, labels: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate sequence labels for consistency.
        
        Args:
            labels: List of sequence labels
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        is_valid = True
        
        # Check for unknown labels
        for i, label in enumerate(labels):
            if label not in self.label2id:
                errors.append(f"Unknown label '{label}' at position {i}")
                is_valid = False
        
        # Check for sequence consistency
        if self.scheme in [TaggingScheme.BIO, TaggingScheme.BIOS, TaggingScheme.IOBES]:
            for i, label in enumerate(labels):
                if label.startswith('I-') and i == 0:
                    errors.append(f"I-tag '{label}' at start of sequence (position {i})")
                    is_valid = False
                elif label.startswith('I-') and i > 0:
                    prev_label = labels[i-1]
                    clause_type = label[2:]
                    
                    # I- must follow B- or I- of same type
                    valid_predecessors = [f'B-{clause_type}', f'I-{clause_type}']
                    if self.scheme == TaggingScheme.IOBES:
                        valid_predecessors.extend([f'E-{clause_type}'])
                        
                    if prev_label not in valid_predecessors:
                        errors.append(f"Invalid sequence: '{prev_label}' -> '{label}' at position {i}")
                        is_valid = False
                        
                if self.scheme == TaggingScheme.IOBES and label.startswith('E-'):
                    if i == 0:
                        errors.append(f"E-tag '{label}' at start of sequence (position {i})")
                        is_valid = False
                    else:
                        prev_label = labels[i-1]
                        clause_type = label[2:]
                        
                        # E- must follow B- or I- of same type
                        valid_predecessors = [f'B-{clause_type}', f'I-{clause_type}']
                        if prev_label not in valid_predecessors:
                            errors.append(f"Invalid sequence: '{prev_label}' -> '{label}' at position {i}")
                            is_valid = False
        
        return is_valid, errors
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save label encoder configuration to file.
        
        Args:
            filepath: Path to save the configuration
        """
        import json
        
        config = {
            "scheme": self.scheme.value,
            "clause_types": self.clause_types,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "num_labels": self.num_labels
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Label encoder saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'LabelEncoder':
        """
        Load label encoder from file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            Loaded LabelEncoder instance
        """
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        scheme = TaggingScheme(config["scheme"])
        encoder = cls(scheme)
        
        # Override with loaded configuration
        encoder.clause_types = config["clause_types"]
        encoder.label2id = config["label2id"]
        encoder.id2label = {int(k): v for k, v in config["id2label"].items()}
        
        # Rebuild label info
        encoder._build_label_info_from_mappings()
        
        logger.info(f"Label encoder loaded from {filepath}")
        return encoder
    
    def _build_label_info_from_mappings(self) -> None:
        """Rebuild label info from loaded mappings."""
        self.label_info = {}
        
        for label, label_id in self.label2id.items():
            if label == 'O':
                self.label_info[label] = LabelInfo(
                    id=label_id,
                    name=label,
                    description="Outside any clause"
                )
            else:
                # Parse label to extract tag type and clause type
                parts = label.split('-', 1)
                if len(parts) == 2:
                    tag_type, clause_type = parts
                    self.label_info[label] = LabelInfo(
                        id=label_id,
                        name=label,
                        description=f"{tag_type} tag for {clause_type}",
                        clause_type=clause_type,
                        tag_type=tag_type
                    )
                else:
                    # Fallback for malformed labels
                    self.label_info[label] = LabelInfo(
                        id=label_id,
                        name=label,
                        description=f"Label {label}"
                    )