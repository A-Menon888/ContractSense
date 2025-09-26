"""
Annotation Schema for ContractSense

Defines the complete annotation schema including clause types, risk taxonomy,
and metadata fields for contract analysis.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json
from pathlib import Path
from dataclasses import dataclass

from . import ClauseType, RiskLevel, LabelingScheme, logger

if TYPE_CHECKING:
    from . import DocumentAnnotation


@dataclass
class ClauseTypeDefinition:
    """Definition of a clause type with metadata"""
    name: str
    description: str
    examples: List[str]
    keywords: List[str]
    risk_category: RiskLevel
    priority: int = 1


class AnnotationSchema:
    """Defines the annotation schema for contract clause labeling"""
    
    def __init__(self):
        self.clause_type_definitions = self._initialize_clause_types()
        self.risk_taxonomy = self._initialize_risk_taxonomy()
        self.labeling_scheme = LabelingScheme.BIO  # Default scheme
        self.metadata_fields = self._initialize_metadata_fields()
    
    def _initialize_clause_types(self) -> List[ClauseTypeDefinition]:
        """Initialize the standard clause types for contract annotation"""
        
        return [
            # High Priority Clauses (Priority 1)
            ClauseTypeDefinition(
                name="indemnification",
                description="Clauses where one party agrees to compensate the other for certain damages or losses",
                examples=[
                    "Each party shall indemnify and hold harmless the other party",
                    "Company agrees to defend, indemnify and hold harmless Client",
                    "Provider shall indemnify Customer against any third-party claims"
                ],
                keywords=["indemnify", "indemnification", "hold harmless", "defend", "compensate"],
                risk_category=RiskLevel.HIGH,
                priority=1
            ),
            
            ClauseTypeDefinition(
                name="limitation_of_liability",
                description="Clauses that limit or cap the liability of one or both parties",
                examples=[
                    "In no event shall Company's liability exceed the amount paid",
                    "Provider's aggregate liability shall not exceed $100,000",
                    "Neither party shall be liable for indirect, incidental, or consequential damages"
                ],
                keywords=["liability", "limit", "cap", "aggregate", "consequential", "indirect", "damages"],
                risk_category=RiskLevel.HIGH,
                priority=1
            ),
            
            ClauseTypeDefinition(
                name="termination",
                description="Clauses specifying conditions under which the contract can be terminated",
                examples=[
                    "Either party may terminate this agreement with 30 days written notice",
                    "This agreement shall terminate automatically upon breach",
                    "Customer may terminate for convenience with 60 days notice"
                ],
                keywords=["terminate", "termination", "expire", "end", "breach", "notice"],
                risk_category=RiskLevel.MEDIUM,
                priority=1
            ),
            
            # Medium Priority Clauses (Priority 2)
            ClauseTypeDefinition(
                name="governing_law",
                description="Clauses specifying which jurisdiction's laws govern the contract",
                examples=[
                    "This Agreement shall be governed by the laws of California",
                    "Any disputes shall be resolved under New York law",
                    "The parties agree that Delaware law applies to this contract"
                ],
                keywords=["governing law", "governed by", "jurisdiction", "applicable law", "laws of"],
                risk_category=RiskLevel.MEDIUM,
                priority=2
            ),
            
            ClauseTypeDefinition(
                name="payment_terms",
                description="Clauses specifying payment amounts, schedules, and methods",
                examples=[
                    "Customer shall pay Provider $1,000 monthly",
                    "Payment is due within 30 days of invoice",
                    "Late payments incur a 1.5% monthly fee"
                ],
                keywords=["payment", "fee", "invoice", "due", "monthly", "quarterly", "cost"],
                risk_category=RiskLevel.MEDIUM,
                priority=2
            ),
            
            ClauseTypeDefinition(
                name="confidentiality",
                description="Clauses protecting confidential information and trade secrets",
                examples=[
                    "Each party agrees to maintain the confidentiality of proprietary information",
                    "Confidential information shall not be disclosed to third parties",
                    "The receiving party shall use confidential information solely for permitted purposes"
                ],
                keywords=["confidential", "proprietary", "non-disclosure", "trade secret", "confidentiality"],
                risk_category=RiskLevel.HIGH,
                priority=2
            ),
            
            # Lower Priority but Important Clauses (Priority 3)
            ClauseTypeDefinition(
                name="ip_ownership",
                description="Clauses defining ownership of intellectual property",
                examples=[
                    "All intellectual property created under this agreement belongs to Company",
                    "Customer retains ownership of its pre-existing intellectual property",
                    "Work product shall be jointly owned by both parties"
                ],
                keywords=["intellectual property", "copyright", "patent", "trademark", "ownership", "IP"],
                risk_category=RiskLevel.HIGH,
                priority=3
            ),
            
            ClauseTypeDefinition(
                name="warranty",
                description="Clauses providing warranties or disclaiming warranties",
                examples=[
                    "Provider warrants that services will be performed professionally",
                    "Software is provided 'as is' without warranties",
                    "Company disclaims all implied warranties"
                ],
                keywords=["warrant", "warranty", "guarantee", "as is", "disclaim", "fitness for purpose"],
                risk_category=RiskLevel.MEDIUM,
                priority=3
            ),
            
            ClauseTypeDefinition(
                name="force_majeure",
                description="Clauses excusing performance due to unforeseeable circumstances",
                examples=[
                    "Neither party shall be liable for delays due to acts of God",
                    "Force majeure events include natural disasters and government actions",
                    "Performance is excused during unforeseeable circumstances beyond control"
                ],
                keywords=["force majeure", "acts of god", "unforeseeable", "beyond control", "natural disaster"],
                risk_category=RiskLevel.LOW,
                priority=3
            )
        ]
    
    def _initialize_risk_taxonomy(self) -> Dict[str, Any]:
        """Initialize risk categories and their definitions"""
        
        return {
            "risk_levels": {
                RiskLevel.CRITICAL.value: {
                    "description": "Clauses that could result in significant financial exposure or legal liability",
                    "examples": ["Unlimited liability", "Broad indemnification", "Exclusive remedies"],
                    "review_priority": 1
                },
                RiskLevel.HIGH.value: {
                    "description": "Clauses with substantial business or legal implications",
                    "examples": ["IP ownership", "Confidentiality breaches", "Termination for convenience"],
                    "review_priority": 2
                },
                RiskLevel.MEDIUM.value: {
                    "description": "Important business terms that require attention",
                    "examples": ["Payment terms", "Governing law", "Standard warranties"],
                    "review_priority": 3
                },
                RiskLevel.LOW.value: {
                    "description": "Standard contractual provisions with minimal risk",
                    "examples": ["Force majeure", "Notice provisions", "Severability"],
                    "review_priority": 4
                }
            },
            "risk_factors": [
                {
                    "factor": "financial_exposure",
                    "description": "Potential financial losses or obligations",
                    "indicators": ["unlimited liability", "uncapped damages", "minimum commitments"]
                },
                {
                    "factor": "ip_risk", 
                    "description": "Intellectual property ownership or licensing risks",
                    "indicators": ["broad IP grants", "work for hire", "joint ownership"]
                },
                {
                    "factor": "operational_risk",
                    "description": "Risks affecting business operations",
                    "indicators": ["exclusivity", "non-compete", "termination rights"]
                },
                {
                    "factor": "compliance_risk",
                    "description": "Regulatory or legal compliance obligations",
                    "indicators": ["data privacy", "industry regulations", "audit rights"]
                }
            ]
        }
    
    def _initialize_metadata_fields(self) -> Dict[str, Any]:
        """Initialize metadata fields for annotations"""
        
        return {
            "document_metadata": {
                "document_type": ["contract", "amendment", "addendum", "exhibit"],
                "industry": ["technology", "healthcare", "financial", "manufacturing", "other"],
                "contract_value": "numeric",
                "effective_date": "date",
                "expiration_date": "date",
                "parties": ["buyer", "seller", "licensor", "licensee", "service_provider", "customer"]
            },
            "annotation_metadata": {
                "annotator_id": "string",
                "annotation_date": "date",
                "annotation_version": "string",
                "quality_score": "numeric",
                "review_status": ["pending", "approved", "rejected", "needs_revision"],
                "annotation_method": ["manual", "semi_automated", "fully_automated"]
            },
            "span_metadata": {
                "confidence": "numeric",
                "risk_level": [level.value for level in RiskLevel],
                "requires_legal_review": "boolean",
                "business_criticality": ["low", "medium", "high", "critical"],
                "negotiability": ["fixed", "negotiable", "highly_negotiable"]
            }
        }
    
    def validate_annotation(self, annotation: 'DocumentAnnotation') -> List[str]:
        """
        Validate an annotation against the schema
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate clause types
        valid_clause_types = {defn.name for defn in self.clause_type_definitions}
        for span in annotation.span_annotations:
            if span.clause_type not in valid_clause_types:
                errors.append(f"Invalid clause type: {span.clause_type}")
        
        # Validate risk levels
        valid_risk_levels = {level.value for level in RiskLevel}
        for span in annotation.span_annotations:
            if span.risk_level and span.risk_level.value not in valid_risk_levels:
                errors.append(f"Invalid risk level: {span.risk_level}")
        
        # Validate span integrity
        for i, span in enumerate(annotation.span_annotations):
            if span.start_char >= span.end_char:
                errors.append(f"Span {i}: Invalid character range {span.start_char}-{span.end_char}")
            
            if span.end_char > len(annotation.full_text):
                errors.append(f"Span {i}: End position beyond document length")
        
        return errors
    
    def get_clause_type_definition(self, clause_type: str) -> Optional[ClauseTypeDefinition]:
        """Get the definition for a specific clause type"""
        
        for defn in self.clause_type_definitions:
            if defn.name == clause_type:
                return defn
        return None
    
    def get_high_priority_clause_types(self) -> List[str]:
        """Get clause types with highest priority (priority 1)"""
        
        return [
            defn.name 
            for defn in self.clause_type_definitions 
            if defn.priority == 1
        ]
    
    def get_clause_types_by_risk(self, risk_level: RiskLevel) -> List[str]:
        """Get clause types with specific risk level"""
        
        return [
            defn.name 
            for defn in self.clause_type_definitions 
            if defn.risk_category == risk_level
        ]
    
    def export_schema(self, output_path: Path) -> None:
        """Export schema definition to JSON file"""
        
        schema_data = {
            "version": "1.0",
            "clause_types": [
                {
                    "name": defn.name,
                    "description": defn.description,
                    "examples": defn.examples,
                    "keywords": defn.keywords,
                    "risk_category": defn.risk_category.value,
                    "priority": defn.priority
                }
                for defn in self.clause_type_definitions
            ],
            "risk_taxonomy": self.risk_taxonomy,
            "metadata_fields": self.metadata_fields,
            "labeling_scheme": self.labeling_scheme.value
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Schema exported to {output_path}")
    
    @classmethod
    def load_schema(cls, schema_path: Path) -> 'AnnotationSchema':
        """Load schema from JSON file"""
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Create instance and populate from loaded data
        instance = cls()
        
        # Load clause type definitions
        instance.clause_type_definitions = []
        for ct_data in schema_data.get('clause_types', []):
            defn = ClauseTypeDefinition(
                name=ct_data['name'],
                description=ct_data['description'],
                examples=ct_data['examples'],
                keywords=ct_data['keywords'],
                risk_category=RiskLevel(ct_data['risk_category']),
                priority=ct_data.get('priority', 1)
            )
            instance.clause_type_definitions.append(defn)
        
        # Load other schema components
        instance.risk_taxonomy = schema_data.get('risk_taxonomy', {})
        instance.metadata_fields = schema_data.get('metadata_fields', {})
        
        labeling_scheme_value = schema_data.get('labeling_scheme', 'BIO')
        instance.labeling_scheme = LabelingScheme(labeling_scheme_value)
        
        logger.info(f"Schema loaded from {schema_path}")
        return instance