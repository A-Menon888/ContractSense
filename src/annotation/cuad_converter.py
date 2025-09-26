"""
CUAD Dataset Converter for ContractSense

Converts CUAD (Contract Understanding Atticus Dataset) annotations to ContractSense format.
CUAD contains 510 legal contracts with expert annotations for 41 clause types.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import json
import pandas as pd
from collections import defaultdict
import logging

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion import ProcessedDocument, DocumentType

from . import DocumentAnnotation, SpanAnnotation, ClauseType, logger
from .schema import AnnotationSchema


class CUADConverter:
    """Converts CUAD dataset to ContractSense annotation format"""
    
    # CUAD clause type mapping to our ClauseType enum
    CUAD_CLAUSE_MAPPING = {
        "Parties": "parties",
        "Agreement Date": "effective_date",
        "Expiration Date": "expiration_date",
        "Renewal Term": "renewal",
        "Notice Period To Terminate Renewal": "termination",
        "Governing Law": "governing_law",
        "Most Favored Nation": "most_favored_nation",
        "Non-Compete": "non_compete",
        "Exclusivity": "exclusivity",
        "No-Solicit Of Customers": "no_solicitation",
        "Competitive Restriction Exception": "competitive_restriction",
        "No-Solicit Of Employees": "no_solicitation",
        "Non-Disparagement": "non_disparagement",
        "Termination For Convenience": "termination",
        "Rofr-Rofo-Rofn": "right_of_first_refusal",
        "Change Of Control": "change_of_control",
        "Anti-Assignment": "anti_assignment",
        "Revenue/Profit Sharing": "revenue_sharing",
        "Price Restrictions": "pricing",
        "Minimum Commitment": "minimum_commitment",
        "Volume Restriction": "volume_restriction",
        "Ip Ownership Assignment": "ip_ownership",
        "Joint Ip Ownership": "ip_ownership",
        "License Grant": "license_grant",
        "Non-Transferable License": "license_grant",
        "Affiliate License-Licensee": "license_grant",
        "Affiliate License-Licensor": "license_grant",
        "Unlimited/All-You-Can-Eat-License": "license_grant",
        "Irrevocable Or Perpetual License": "license_grant",
        "Source Code Escrow": "source_code_escrow",
        "Post-Termination Services": "post_termination_services",
        "Audit Rights": "audit_rights",
        "Uncapped Liability": "limitation_of_liability",
        "Cap On Liability": "limitation_of_liability",
        "Liquidated Damages": "liquidated_damages",
        "Warranty Duration": "warranty",
        "Insurance": "insurance_requirements",
        "Covenant Not To Sue": "covenant_not_to_sue",
        "Third Party Beneficiary": "third_party_beneficiary"
    }
    
    def __init__(self, 
                 cuad_json_path: Path,
                 cuad_txt_dir: Path,
                 schema: Optional[AnnotationSchema] = None):
        """
        Initialize CUAD converter
        
        Args:
            cuad_json_path: Path to CUAD_v1.json file
            cuad_txt_dir: Path to directory containing .txt files
            schema: Annotation schema to use
        """
        
        self.cuad_json_path = cuad_json_path
        self.cuad_txt_dir = cuad_txt_dir
        self.schema = schema or AnnotationSchema()
        
        # Load CUAD data
        self.cuad_data = self._load_cuad_json()
        logger.info(f"Loaded CUAD dataset with {len(self.cuad_data)} entries")
    
    def _load_cuad_json(self) -> Dict[str, Any]:
        """Load the CUAD JSON file"""
        
        with open(self.cuad_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def convert_to_contractsense_format(self, 
                                      limit: Optional[int] = None,
                                      include_negative_examples: bool = True) -> List[DocumentAnnotation]:
        """
        Convert CUAD annotations to ContractSense format
        
        Args:
            limit: Maximum number of documents to convert
            include_negative_examples: Include documents with no annotations
            
        Returns:
            List of DocumentAnnotation objects
        """
        
        annotations = []
        processed_count = 0
        
        # CUAD data is organized as list of documents with paragraphs containing QAs
        # We need to extract all question-answer pairs and organize by document
        doc_annotations = defaultdict(list)
        
        for doc_item in self.cuad_data['data']:
            title = doc_item.get('title', 'unknown')
            
            for paragraph in doc_item.get('paragraphs', []):
                context = paragraph.get('context', '')
                
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    
                    # Map question to clause type (simplified for demo)
                    clause_type = self._map_question_to_clause_type(question)
                    if not clause_type:
                        continue
                    
                    # Extract answers
                    for answer_data in qa.get('answers', []):
                        span_info = {
                            'clause_type': clause_type,
                            'text': answer_data.get('text', ''),
                            'answer_start': answer_data.get('answer_start', 0)
                        }
                        doc_annotations[title].append(span_info)
        
        # Convert to DocumentAnnotation objects
        for filename, span_list in doc_annotations.items():
            if limit and processed_count >= limit:
                break
            
            try:
                doc_annotation = self._create_document_annotation(filename, span_list)
                if doc_annotation:
                    annotations.append(doc_annotation)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Failed to convert document {filename}: {e}")
                continue
        
        logger.info(f"Converted {len(annotations)} CUAD documents to ContractSense format")
        return annotations
    
    def _create_document_annotation(self, 
                                   filename: str, 
                                   span_list: List[Dict[str, Any]]) -> Optional[DocumentAnnotation]:
        """Create DocumentAnnotation from CUAD data"""
        
        # Find the corresponding text file
        txt_path = self.cuad_txt_dir / filename
        if not txt_path.exists():
            logger.warning(f"Text file not found: {txt_path}")
            return None
        
        # Read full text
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        
        # Create span annotations
        span_annotations = []
        for span_info in span_list:
            try:
                start_char = span_info['answer_start']
                text = span_info['text']
                end_char = start_char + len(text)
                
                # Validate that the span text matches
                if end_char <= len(full_text):
                    actual_text = full_text[start_char:end_char]
                    if actual_text == text:
                        span_annotation = SpanAnnotation(
                            start_char=start_char,
                            end_char=end_char,
                            clause_type=span_info['clause_type'],
                            text=text,
                            confidence=1.0,  # CUAD annotations are expert-labeled
                            notes="Converted from CUAD dataset"
                        )
                        span_annotations.append(span_annotation)
                    else:
                        logger.warning(f"Text mismatch in {filename} at {start_char}: expected '{text[:50]}', got '{actual_text[:50]}'")
                else:
                    logger.warning(f"Span out of bounds in {filename}: {start_char}-{end_char} > {len(full_text)}")
            
            except Exception as e:
                logger.error(f"Failed to create span annotation for {filename}: {e}")
                continue
        
        # Create document annotation
        document_id = Path(filename).stem
        doc_annotation = DocumentAnnotation(
            document_id=document_id,
            document_path=str(txt_path),
            full_text=full_text,
            span_annotations=span_annotations,
            document_metadata={
                "source": "CUAD",
                "filename": filename,
                "document_length": len(full_text),
                "total_spans": len(span_annotations)
            },
            annotation_metadata={
                "converted_from": "CUAD_v1",
                "annotation_quality": "expert_labeled"
            }
        )
        
        return doc_annotation
    
    def _map_question_to_clause_type(self, question: str) -> Optional[str]:
        """Map CUAD question to our clause type (simplified mapping for demo)"""
        question_lower = question.lower()
        
        if 'indemnif' in question_lower:
            return 'indemnification'
        elif 'liability' in question_lower:
            return 'limitation_of_liability'  
        elif 'terminat' in question_lower:
            return 'termination'
        elif 'governing' in question_lower or 'law' in question_lower:
            return 'governing_law'
        elif 'payment' in question_lower or 'fee' in question_lower:
            return 'payment_terms'
        elif 'confidential' in question_lower:
            return 'confidentiality'
        elif 'intellectual' in question_lower or 'ip' in question_lower:
            return 'ip_ownership'
        elif 'warranty' in question_lower:
            return 'warranty'
        else:
            return None  # Skip unmapped question types for demo
    
    def analyze_cuad_dataset(self) -> Dict[str, Any]:
        """Analyze the CUAD dataset structure and statistics"""
        
        analysis = {
            "total_documents": 0,
            "total_paragraphs": 0,
            "total_questions": 0,
            "sample_questions": [],
            "document_titles": []
        }
        
        # Analyze document structure
        for doc_item in self.cuad_data['data']:
            analysis["total_documents"] += 1
            title = doc_item.get('title', 'unknown')
            analysis["document_titles"].append(title)
            
            for paragraph in doc_item.get('paragraphs', []):
                analysis["total_paragraphs"] += 1
                
                for qa in paragraph.get('qas', []):
                    analysis["total_questions"] += 1
                    question = qa.get('question', '')
                    
                    if len(analysis["sample_questions"]) < 10:
                        analysis["sample_questions"].append(question)
        
        return analysis
    
    def create_train_val_test_split(self, 
                                   annotations: List[DocumentAnnotation],
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Dict[str, List[DocumentAnnotation]]:
        """
        Create train/validation/test splits for CUAD data
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        
        import random
        random.seed(random_seed)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        # Shuffle annotations
        shuffled_annotations = annotations.copy()
        random.shuffle(shuffled_annotations)
        
        total = len(shuffled_annotations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': shuffled_annotations[:train_end],
            'val': shuffled_annotations[train_end:val_end],
            'test': shuffled_annotations[val_end:]
        }
        
        logger.info(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def export_cuad_splits(self, 
                          output_dir: Path,
                          limit: Optional[int] = None) -> Dict[str, Path]:
        """
        Convert CUAD and export train/val/test splits
        
        Returns:
            Dictionary mapping split names to output file paths
        """
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert CUAD data
        annotations = self.convert_to_contractsense_format(limit=limit)
        
        if not annotations:
            raise ValueError("No annotations were successfully converted")
        
        # Create splits
        splits = self.create_train_val_test_split(annotations)
        
        # Export each split
        output_paths = {}
        for split_name, split_annotations in splits.items():
            output_path = output_dir / f"cuad_{split_name}.json"
            
            # Export to JSON
            export_data = {
                "split_info": {
                    "split_name": split_name,
                    "total_documents": len(split_annotations),
                    "source": "CUAD_v1",
                    "converted_by": "ContractSense"
                },
                "documents": [doc.to_dict() for doc in split_annotations]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            output_paths[split_name] = output_path
            logger.info(f"Exported {split_name} split to {output_path}")
        
        return output_paths
    
    def validate_conversion(self, annotations: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Validate the converted annotations"""
        
        from .validator import AnnotationValidator
        
        validator = AnnotationValidator(self.schema)
        validation_report = validator.generate_validation_report(annotations)
        
        # Add CUAD-specific validation
        cuad_stats = self._compute_cuad_specific_stats(annotations)
        validation_report["cuad_specific"] = cuad_stats
        
        return validation_report
    
    def _compute_cuad_specific_stats(self, annotations: List[DocumentAnnotation]) -> Dict[str, Any]:
        """Compute CUAD-specific statistics"""
        
        stats = {
            "total_converted_documents": len(annotations),
            "clause_types_found": set(),
            "average_spans_per_document": 0,
            "documents_by_span_count": defaultdict(int)
        }
        
        total_spans = 0
        for doc in annotations:
            span_count = len(doc.span_annotations)
            total_spans += span_count
            stats["documents_by_span_count"][span_count] += 1
            
            for span in doc.span_annotations:
                stats["clause_types_found"].add(span.clause_type)
        
        stats["clause_types_found"] = list(stats["clause_types_found"])
        stats["average_spans_per_document"] = total_spans / len(annotations) if annotations else 0
        stats["documents_by_span_count"] = dict(stats["documents_by_span_count"])
        
        return stats