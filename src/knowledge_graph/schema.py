"""
Knowledge Graph Schema Definition for ContractSense

This module defines the ontology and schema for the Neo4j knowledge graph
that will store contract entities, relationships, and provenance information.

Graph Ontology:
- Party: Legal entities (companies, individuals)
- Agreement: Contract documents 
- Clause: Individual contract clauses
- ClauseType: Categories of clauses (liability, payment, etc.)
- MonetaryTerm: Financial amounts and terms
- Date: Important dates (effective, expiry, etc.)
- Obligation: Legal obligations and duties
- LiabilityCap: Liability limitations and caps
- GoverningLaw: Jurisdiction and governing law
- RiskAssessment: Risk scores and classifications

Key Relationships:
- [Agreement]-[HAS_CLAUSE]->[Clause]
- [Party]-[PARTY_TO]->[Agreement]  
- [Clause]-[HAS_TYPE]->[ClauseType]
- [Clause]-[CONTAINS]->[MonetaryTerm]
- [Agreement]-[GOVERNED_BY]->[GoverningLaw]
- [Clause]-[HAS_RISK]->[RiskAssessment]
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Enumeration of all node types in the knowledge graph"""
    PARTY = "Party"
    AGREEMENT = "Agreement"
    CLAUSE = "Clause"
    CLAUSE_TYPE = "ClauseType"
    MONETARY_TERM = "MonetaryTerm"
    DATE = "Date"
    OBLIGATION = "Obligation"
    LIABILITY_CAP = "LiabilityCap"
    GOVERNING_LAW = "GoverningLaw"
    RISK_ASSESSMENT = "RiskAssessment"
    DOCUMENT_SPAN = "DocumentSpan"

class RelationshipType(Enum):
    """Enumeration of all relationship types in the knowledge graph"""
    # Core document relationships
    HAS_CLAUSE = "HAS_CLAUSE"
    PARTY_TO = "PARTY_TO"
    
    # Clause relationships
    HAS_TYPE = "HAS_TYPE"
    CONTAINS = "CONTAINS"
    REFERENCES = "REFERENCES"
    
    # Legal relationships
    GOVERNED_BY = "GOVERNED_BY"
    HAS_RISK = "HAS_RISK"
    LIMITS_LIABILITY = "LIMITS_LIABILITY"
    CREATES_OBLIGATION = "CREATES_OBLIGATION"
    
    # Provenance relationships
    EXTRACTED_FROM = "EXTRACTED_FROM"
    SPANS_TEXT = "SPANS_TEXT"
    
    # Temporal relationships
    EFFECTIVE_FROM = "EFFECTIVE_FROM"
    EXPIRES_ON = "EXPIRES_ON"
    
    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    DEPENDS_ON = "DEPENDS_ON"

@dataclass
class GraphNode:
    """Base class for all graph nodes"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    
    def validate(self) -> Dict[str, Any]:
        """Validate the node structure and properties"""
        errors = []
        
        if not self.node_id or not isinstance(self.node_id, str):
            errors.append("Node ID must be a non-empty string")
        
        if not isinstance(self.node_type, NodeType):
            errors.append("Node type must be a valid NodeType enum")
        
        if not isinstance(self.properties, dict):
            errors.append("Properties must be a dictionary")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement for this node"""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        return f"CREATE (n:{self.node_type.value} {{id: $node_id, {props_str}}})"
    
    def to_cypher_merge(self) -> str:
        """Generate Cypher MERGE statement for this node"""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        return f"MERGE (n:{self.node_type.value} {{id: $node_id}}) SET n += {{{props_str}}}"

@dataclass
class GraphRelationship:
    """Base class for all graph relationships"""
    from_node_id: str
    to_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    
    def validate(self) -> Dict[str, Any]:
        """Validate the relationship structure and properties"""
        errors = []
        
        if not self.from_node_id or not isinstance(self.from_node_id, str):
            errors.append("From node ID must be a non-empty string")
        
        if not self.to_node_id or not isinstance(self.to_node_id, str):
            errors.append("To node ID must be a non-empty string")
        
        if not isinstance(self.relationship_type, RelationshipType):
            errors.append("Relationship type must be a valid RelationshipType enum")
        
        if not isinstance(self.properties, dict):
            errors.append("Properties must be a dictionary")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def to_cypher(self) -> str:
        """Generate Cypher statement for this relationship"""
        props_str = ""
        if self.properties:
            props_dict = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
            props_str = f" {{{props_dict}}}"
        
        return f"""
        MATCH (a {{id: $from_node_id}}), (b {{id: $to_node_id}})
        MERGE (a)-[:{self.relationship_type.value}{props_str}]->(b)
        """

class ContractGraphSchema:
    """Schema definition and validation for the contract knowledge graph"""
    
    def __init__(self):
        self.node_schemas = self._define_node_schemas()
        self.relationship_schemas = self._define_relationship_schemas()
        self.indexes = self._define_indexes()
        self.constraints = self._define_constraints()
    
    def _define_node_schemas(self) -> Dict[NodeType, Dict[str, str]]:
        """Define expected properties for each node type"""
        return {
            NodeType.PARTY: {
                "name": "string",
                "type": "string",  # company, individual, government
                "jurisdiction": "string",
                "aliases": "list",
                "extracted_confidence": "float"
            },
            
            NodeType.AGREEMENT: {
                "title": "string",
                "document_id": "string",
                "file_path": "string",
                "document_type": "string",  # MSA, SOW, NDA, etc.
                "effective_date": "date",
                "expiry_date": "date",
                "parties_count": "integer",
                "clauses_count": "integer",
                "risk_level": "string",
                "processing_timestamp": "datetime"
            },
            
            NodeType.CLAUSE: {
                "text": "string",
                "clause_id": "string",
                "clause_number": "string",
                "start_pos": "integer",
                "end_pos": "integer",
                "confidence_score": "float",
                "word_count": "integer",
                "extraction_method": "string"  # BERT-CRF, rule-based, etc.
            },
            
            NodeType.CLAUSE_TYPE: {
                "name": "string",
                "category": "string",
                "description": "string",
                "risk_weight": "float",
                "common_keywords": "list"
            },
            
            NodeType.MONETARY_TERM: {
                "amount": "float",
                "currency": "string",
                "term_type": "string",  # payment, penalty, cap, etc.
                "frequency": "string",  # one-time, monthly, annual
                "description": "string",
                "extraction_confidence": "float"
            },
            
            NodeType.DATE: {
                "date_value": "date",
                "date_type": "string",  # effective, expiry, milestone
                "description": "string",
                "is_relative": "boolean",
                "extraction_confidence": "float"
            },
            
            NodeType.OBLIGATION: {
                "description": "string",
                "obligor": "string",  # who has the obligation
                "obligee": "string",  # who benefits
                "obligation_type": "string",  # payment, delivery, performance
                "enforceability": "string",
                "risk_level": "string"
            },
            
            NodeType.LIABILITY_CAP: {
                "cap_amount": "float",
                "cap_currency": "string",
                "cap_type": "string",  # total, annual, per-incident
                "applies_to": "string",
                "exceptions": "list",
                "risk_impact": "string"
            },
            
            NodeType.GOVERNING_LAW: {
                "jurisdiction": "string",
                "state_province": "string",
                "country": "string",
                "dispute_resolution": "string",
                "court_system": "string"
            },
            
            NodeType.RISK_ASSESSMENT: {
                "risk_level": "string",  # LOW, MEDIUM, HIGH
                "risk_score": "float",
                "confidence": "float",
                "risk_factors": "list",
                "mitigation_suggestions": "list",
                "assessment_method": "string",
                "assessment_timestamp": "datetime"
            },
            
            NodeType.DOCUMENT_SPAN: {
                "document_id": "string",
                "start_char": "integer",
                "end_char": "integer",
                "text_snippet": "string",
                "context_before": "string",
                "context_after": "string",
                "extraction_method": "string",
                "confidence": "float"
            }
        }
    
    def _define_relationship_schemas(self) -> Dict[RelationshipType, Dict[str, str]]:
        """Define expected properties for each relationship type"""
        return {
            RelationshipType.HAS_CLAUSE: {
                "clause_order": "integer",
                "section": "string",
                "confidence": "float"
            },
            
            RelationshipType.PARTY_TO: {
                "role": "string",  # client, vendor, licensor, etc.
                "primary_contact": "boolean",
                "signing_capacity": "string"
            },
            
            RelationshipType.HAS_TYPE: {
                "classification_confidence": "float",
                "classification_method": "string"
            },
            
            RelationshipType.CONTAINS: {
                "relevance_score": "float",
                "extraction_method": "string"
            },
            
            RelationshipType.GOVERNED_BY: {
                "explicitly_stated": "boolean",
                "confidence": "float"
            },
            
            RelationshipType.HAS_RISK: {
                "risk_contribution": "float",
                "assessment_timestamp": "datetime"
            },
            
            RelationshipType.EXTRACTED_FROM: {
                "extraction_timestamp": "datetime",
                "extraction_method": "string",
                "confidence": "float"
            },
            
            RelationshipType.SPANS_TEXT: {
                "char_start": "integer",
                "char_end": "integer",
                "word_start": "integer",
                "word_end": "integer"
            }
        }
    
    def _define_indexes(self) -> List[str]:
        """Define indexes to create for optimal query performance"""
        return [
            "CREATE INDEX party_name_idx IF NOT EXISTS FOR (p:Party) ON (p.name)",
            "CREATE INDEX agreement_document_id_idx IF NOT EXISTS FOR (a:Agreement) ON (a.document_id)",
            "CREATE INDEX clause_id_idx IF NOT EXISTS FOR (c:Clause) ON (c.clause_id)",
            "CREATE INDEX clause_type_name_idx IF NOT EXISTS FOR (ct:ClauseType) ON (ct.name)",
            "CREATE INDEX monetary_amount_idx IF NOT EXISTS FOR (mt:MonetaryTerm) ON (mt.amount)",
            "CREATE INDEX date_value_idx IF NOT EXISTS FOR (d:Date) ON (d.date_value)",
            "CREATE INDEX governing_law_jurisdiction_idx IF NOT EXISTS FOR (gl:GoverningLaw) ON (gl.jurisdiction)",
            "CREATE INDEX risk_level_idx IF NOT EXISTS FOR (ra:RiskAssessment) ON (ra.risk_level)",
            "CREATE INDEX document_span_doc_id_idx IF NOT EXISTS FOR (ds:DocumentSpan) ON (ds.document_id)"
        ]
    
    def _define_constraints(self) -> List[str]:
        """Define uniqueness constraints for data integrity"""
        return [
            "CREATE CONSTRAINT party_id_unique IF NOT EXISTS FOR (p:Party) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT agreement_id_unique IF NOT EXISTS FOR (a:Agreement) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT clause_type_name_unique IF NOT EXISTS FOR (ct:ClauseType) REQUIRE ct.name IS UNIQUE",
            "CREATE CONSTRAINT document_span_id_unique IF NOT EXISTS FOR (ds:DocumentSpan) REQUIRE ds.id IS UNIQUE"
        ]
    
    def get_cypher_schema_setup(self) -> List[str]:
        """Get all Cypher statements needed to set up the schema"""
        statements = []
        statements.extend(self.constraints)
        statements.extend(self.indexes)
        return statements
    
    def validate_node(self, node: GraphNode) -> List[str]:
        """Validate a node against the schema"""
        errors = []
        
        if node.node_type not in self.node_schemas:
            errors.append(f"Unknown node type: {node.node_type}")
            return errors
        
        expected_props = self.node_schemas[node.node_type]
        
        # Check for missing required properties
        required_props = {"id"}  # All nodes must have an ID
        missing_props = required_props - set(node.properties.keys()) - {"node_id"}
        if missing_props:
            errors.append(f"Missing required properties: {missing_props}")
        
        # Validate property types (basic validation)
        for prop_name, prop_value in node.properties.items():
            if prop_name in expected_props:
                expected_type = expected_props[prop_name]
                if not self._validate_property_type(prop_value, expected_type):
                    errors.append(f"Invalid type for {prop_name}: expected {expected_type}, got {type(prop_value)}")
        
        return errors
    
    def _validate_property_type(self, value: Any, expected_type: str) -> bool:
        """Basic property type validation"""
        if value is None:
            return True  # Allow None values
        
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "list": list,
            "date": str,  # Dates stored as ISO strings
            "datetime": str  # Datetimes stored as ISO strings
        }
        
        expected_python_type = type_mapping.get(expected_type, str)
        return isinstance(value, expected_python_type)
    
    def get_sample_queries(self) -> Dict[str, str]:
        """Get sample Cypher queries for testing the schema"""
        return {
            "find_high_risk_clauses": """
                MATCH (a:Agreement)-[:HAS_CLAUSE]->(c:Clause)-[:HAS_RISK]->(r:RiskAssessment)
                WHERE r.risk_level = 'HIGH'
                RETURN a.title, c.text, r.risk_score
                ORDER BY r.risk_score DESC
            """,
            
            "find_liability_caps": """
                MATCH (a:Agreement)-[:HAS_CLAUSE]->(c:Clause)-[:CONTAINS]->(lc:LiabilityCap)
                RETURN a.title, c.text, lc.cap_amount, lc.cap_currency
                ORDER BY lc.cap_amount DESC
            """,
            
            "find_agreements_by_party": """
                MATCH (p:Party)-[:PARTY_TO]->(a:Agreement)
                WHERE p.name CONTAINS $party_name
                RETURN p.name, a.title, a.effective_date
                ORDER BY a.effective_date DESC
            """,
            
            "find_clauses_by_type": """
                MATCH (ct:ClauseType)<-[:HAS_TYPE]-(c:Clause)<-[:HAS_CLAUSE]-(a:Agreement)
                WHERE ct.name = $clause_type
                RETURN a.title, c.text, c.confidence_score
                ORDER BY c.confidence_score DESC
            """,
            
            "find_monetary_terms_above_threshold": """
                MATCH (a:Agreement)-[:HAS_CLAUSE]->(c:Clause)-[:CONTAINS]->(mt:MonetaryTerm)
                WHERE mt.amount > $threshold
                RETURN a.title, c.text, mt.amount, mt.currency, mt.term_type
                ORDER BY mt.amount DESC
            """,
            
            "trace_clause_provenance": """
                MATCH (c:Clause)-[:EXTRACTED_FROM]->(ds:DocumentSpan)
                WHERE c.clause_id = $clause_id
                RETURN c.text, ds.document_id, ds.start_char, ds.end_char, ds.text_snippet
            """,
            
            "aggregate_risk_by_agreement": """
                MATCH (a:Agreement)-[:HAS_CLAUSE]->(c:Clause)-[:HAS_RISK]->(r:RiskAssessment)
                RETURN a.title, 
                       COUNT(c) as total_clauses,
                       AVG(r.risk_score) as avg_risk_score,
                       COUNT(CASE WHEN r.risk_level = 'HIGH' THEN 1 END) as high_risk_count
                ORDER BY avg_risk_score DESC
            """
        }

# Predefined clause types for consistent categorization
STANDARD_CLAUSE_TYPES = [
    {"name": "limitation_of_liability", "category": "liability", "risk_weight": 0.9},
    {"name": "indemnification", "category": "liability", "risk_weight": 0.85},
    {"name": "intellectual_property", "category": "ip", "risk_weight": 0.8},
    {"name": "confidentiality", "category": "information", "risk_weight": 0.6},
    {"name": "termination", "category": "lifecycle", "risk_weight": 0.7},
    {"name": "payment", "category": "financial", "risk_weight": 0.5},
    {"name": "governing_law", "category": "legal", "risk_weight": 0.4},
    {"name": "dispute_resolution", "category": "legal", "risk_weight": 0.6},
    {"name": "force_majeure", "category": "performance", "risk_weight": 0.3},
    {"name": "assignment", "category": "transfer", "risk_weight": 0.5},
    {"name": "warranty", "category": "assurance", "risk_weight": 0.6},
    {"name": "compliance", "category": "regulatory", "risk_weight": 0.7},
    {"name": "insurance", "category": "risk", "risk_weight": 0.4},
    {"name": "effective_date", "category": "temporal", "risk_weight": 0.1},
    {"name": "renewal", "category": "lifecycle", "risk_weight": 0.3}
]

def create_standard_clause_types() -> List[GraphNode]:
    """Create standard clause type nodes"""
    nodes = []
    for clause_type_data in STANDARD_CLAUSE_TYPES:
        node = GraphNode(
            node_id=f"clause_type_{clause_type_data['name']}",
            node_type=NodeType.CLAUSE_TYPE,
            properties={
                "name": clause_type_data["name"],
                "category": clause_type_data["category"], 
                "risk_weight": clause_type_data["risk_weight"],
                "description": f"Standard {clause_type_data['name'].replace('_', ' ')} clause"
            }
        )
        nodes.append(node)
    
    return nodes