"""
Knowledge Graph Ingestion Pipeline

This module provides the pipeline to convert extracted contract data
(from Modules 1-4) into structured Neo4j graph nodes and relationships.

Key Features:
- Convert Module 3 clause extraction output to graph nodes
- Create bidirectional links between graph and document spans  
- Batch processing for efficient ingestion
- Data validation and error handling
- Progress tracking and logging
"""

import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .schema import (
    GraphNode, GraphRelationship, NodeType, RelationshipType
)
from .neo4j_manager import GraphDataManager
from .entity_extractor import EnhancedEntityExtractor, ContractMetadata

logger = logging.getLogger(__name__)

class GraphIngestionPipeline:
    """Pipeline for ingesting contract data into knowledge graph"""
    
    def __init__(self, graph_manager: GraphDataManager):
        """
        Initialize ingestion pipeline
        
        Args:
            graph_manager: GraphDataManager instance for database operations
        """
        self.graph_manager = graph_manager
        self.entity_extractor = EnhancedEntityExtractor()
        
        # Statistics tracking
        self.ingestion_stats = {
            "documents_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": []
        }
    
    def ingest_from_module3_output(self, module3_output_dir: str) -> Dict[str, Any]:
        """
        Ingest contracts from Module 3 output directory
        
        Args:
            module3_output_dir: Path to Module 3 output directory
            
        Returns:
            Dictionary with ingestion statistics and results
        """
        logger.info(f"Starting ingestion from Module 3 output: {module3_output_dir}")
        
        output_path = Path(module3_output_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Module 3 output directory not found: {module3_output_dir}")
        
        # Find all processed JSON files
        json_files = list(output_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        processed_contracts = []
        
        for json_file in json_files:
            try:
                logger.info(f"Processing file: {json_file.name}")
                
                # Load Module 3 output
                with open(json_file, 'r', encoding='utf-8') as f:
                    module3_data = json.load(f)
                
                # Process the contract
                result = self.ingest_single_contract(module3_data, str(json_file))
                processed_contracts.append(result)
                
                self.ingestion_stats["documents_processed"] += 1
                
            except Exception as e:
                error_msg = f"Failed to process {json_file.name}: {str(e)}"
                logger.error(error_msg)
                self.ingestion_stats["errors"].append(error_msg)
        
        logger.info(f"Ingestion completed. Processed {len(processed_contracts)} contracts")
        
        return {
            "processed_contracts": processed_contracts,
            "statistics": self.ingestion_stats
        }
    
    def ingest_single_contract(self, module3_data: Dict[str, Any], 
                             source_file: str = "") -> Dict[str, Any]:
        """
        Ingest a single contract from Module 3 output format
        
        Args:
            module3_data: Module 3 extracted data
            source_file: Path to source file for metadata
            
        Returns:
            Dictionary with ingestion results for this contract
        """
        document_id = module3_data.get("document_id", f"doc_{uuid.uuid4().hex[:8]}")
        logger.debug(f"Ingesting contract: {document_id}")
        
        try:
            # Extract comprehensive metadata
            document_text = module3_data.get("full_text", "")
            clauses_data = module3_data.get("clauses", [])
            
            metadata = self.entity_extractor.extract_comprehensive_metadata(
                document_text=document_text,
                document_id=document_id,
                file_path=source_file,
                clauses=clauses_data
            )
            
            # Convert to graph nodes and relationships
            nodes, relationships = self.convert_to_graph_elements(metadata, module3_data)
            
            # Ingest into graph database
            nodes_created = self.graph_manager.bulk_create_nodes(nodes)
            relationships_created = self.graph_manager.bulk_create_relationships(relationships)
            
            # Update statistics
            self.ingestion_stats["nodes_created"] += nodes_created
            self.ingestion_stats["relationships_created"] += relationships_created
            
            result = {
                "document_id": document_id,
                "nodes_created": nodes_created,
                "relationships_created": relationships_created,
                "parties_found": len(metadata.parties),
                "clauses_processed": len(clauses_data),
                "status": "success"
            }
            
            logger.debug(f"Successfully ingested {document_id}: {nodes_created} nodes, {relationships_created} relationships")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to ingest contract {document_id}: {str(e)}"
            logger.error(error_msg)
            self.ingestion_stats["errors"].append(error_msg)
            
            return {
                "document_id": document_id,
                "status": "error",
                "error": error_msg
            }
    
    def convert_to_graph_elements(self, metadata: ContractMetadata, 
                                module3_data: Dict[str, Any]) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Convert contract metadata to graph nodes and relationships
        
        Args:
            metadata: Extracted contract metadata
            module3_data: Raw Module 3 data for provenance
            
        Returns:
            Tuple of (nodes, relationships) lists
        """
        nodes = []
        relationships = []
        
        # Create Agreement node
        agreement_node = self._create_agreement_node(metadata)
        nodes.append(agreement_node)
        
        # Create Party nodes and relationships
        party_nodes, party_relationships = self._create_party_elements(metadata, agreement_node.node_id)
        nodes.extend(party_nodes)
        relationships.extend(party_relationships)
        
        # Create Clause nodes and relationships
        clause_nodes, clause_relationships = self._create_clause_elements(
            metadata, module3_data, agreement_node.node_id
        )
        nodes.extend(clause_nodes)
        relationships.extend(clause_relationships)
        
        # Create MonetaryTerm nodes and relationships
        monetary_nodes, monetary_relationships = self._create_monetary_elements(
            metadata, clause_nodes
        )
        nodes.extend(monetary_nodes)
        relationships.extend(monetary_relationships)
        
        # Create Date nodes and relationships
        date_nodes, date_relationships = self._create_date_elements(metadata, agreement_node.node_id)
        nodes.extend(date_nodes)
        relationships.extend(date_relationships)
        
        # Create GoverningLaw node if present
        if metadata.governing_law:
            law_node, law_relationship = self._create_governing_law_elements(
                metadata, agreement_node.node_id
            )
            nodes.append(law_node)
            relationships.append(law_relationship)
        
        # Create DocumentSpan nodes for provenance
        span_nodes, span_relationships = self._create_document_span_elements(
            metadata, module3_data, clause_nodes
        )
        nodes.extend(span_nodes)
        relationships.extend(span_relationships)
        
        logger.debug(f"Generated {len(nodes)} nodes and {len(relationships)} relationships for {metadata.document_id}")
        
        return nodes, relationships
    
    def _create_agreement_node(self, metadata: ContractMetadata) -> GraphNode:
        """Create Agreement node"""
        return GraphNode(
            node_id=f"agreement_{metadata.document_id}",
            node_type=NodeType.AGREEMENT,
            properties={
                "title": metadata.title,
                "document_id": metadata.document_id,
                "file_path": metadata.file_path,
                "document_type": metadata.document_type,
                "effective_date": metadata.dates[0].date_value if metadata.dates else None,
                "parties_count": len(metadata.parties),
                "risk_level": metadata.risk_level,
                "processing_timestamp": metadata.processing_timestamp
            }
        )
    
    def _create_party_elements(self, metadata: ContractMetadata, 
                             agreement_id: str) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Create Party nodes and PARTY_TO relationships"""
        nodes = []
        relationships = []
        
        for i, party in enumerate(metadata.parties):
            # Create Party node
            party_node = GraphNode(
                node_id=f"party_{metadata.document_id}_{i}",
                node_type=NodeType.PARTY,
                properties={
                    "name": party.name,
                    "type": party.party_type,
                    "aliases": party.aliases,
                    "jurisdiction": party.jurisdiction,
                    "extracted_confidence": party.confidence
                }
            )
            nodes.append(party_node)
            
            # Create PARTY_TO relationship
            party_relationship = GraphRelationship(
                from_node_id=party_node.node_id,
                to_node_id=agreement_id,
                relationship_type=RelationshipType.PARTY_TO,
                properties={
                    "role": "primary" if i == 0 else "secondary",
                    "primary_contact": i == 0
                }
            )
            relationships.append(party_relationship)
        
        return nodes, relationships
    
    def _create_clause_elements(self, metadata: ContractMetadata, module3_data: Dict[str, Any],
                              agreement_id: str) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Create Clause nodes and relationships"""
        nodes = []
        relationships = []
        
        clauses_data = module3_data.get("clauses", [])
        
        for i, clause_data in enumerate(clauses_data):
            clause_id = clause_data.get("clause_id", f"clause_{metadata.document_id}_{i}")
            
            # Create Clause node
            clause_node = GraphNode(
                node_id=clause_id,
                node_type=NodeType.CLAUSE,
                properties={
                    "text": clause_data.get("text", ""),
                    "clause_id": clause_id,
                    "clause_number": str(i + 1),
                    "start_pos": clause_data.get("start_pos", 0),
                    "end_pos": clause_data.get("end_pos", 0),
                    "confidence_score": clause_data.get("confidence", 0.0),
                    "word_count": len(clause_data.get("text", "").split()),
                    "extraction_method": clause_data.get("extraction_method", "BERT-CRF")
                }
            )
            nodes.append(clause_node)
            
            # Create HAS_CLAUSE relationship
            has_clause_rel = GraphRelationship(
                from_node_id=agreement_id,
                to_node_id=clause_id,
                relationship_type=RelationshipType.HAS_CLAUSE,
                properties={
                    "clause_order": i + 1,
                    "confidence": clause_data.get("confidence", 0.0)
                }
            )
            relationships.append(has_clause_rel)
            
            # Create ClauseType node and HAS_TYPE relationship
            clause_type = clause_data.get("type", "general")
            clause_type_id = f"clause_type_{clause_type}"
            
            # ClauseType nodes are created during schema setup, so just create relationship
            has_type_rel = GraphRelationship(
                from_node_id=clause_id,
                to_node_id=clause_type_id,
                relationship_type=RelationshipType.HAS_TYPE,
                properties={
                    "classification_confidence": clause_data.get("type_confidence", 0.0),
                    "classification_method": "ML"
                }
            )
            relationships.append(has_type_rel)
            
            # Create RiskAssessment node if risk data available
            if "risk" in clause_data:
                risk_node, risk_rel = self._create_risk_assessment_elements(
                    clause_data["risk"], clause_id, metadata.document_id
                )
                nodes.append(risk_node)
                relationships.append(risk_rel)
        
        return nodes, relationships
    
    def _create_monetary_elements(self, metadata: ContractMetadata,
                                clause_nodes: List[GraphNode]) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Create MonetaryTerm nodes and CONTAINS relationships"""
        nodes = []
        relationships = []
        
        for i, monetary_term in enumerate(metadata.monetary_terms):
            # Create MonetaryTerm node
            monetary_node = GraphNode(
                node_id=f"monetary_{metadata.document_id}_{i}",
                node_type=NodeType.MONETARY_TERM,
                properties={
                    "amount": monetary_term.get("amount", 0.0),
                    "currency": monetary_term.get("currency", "USD"),
                    "term_type": monetary_term.get("term_type", "general"),
                    "description": f"{monetary_term.get('currency', 'USD')} {monetary_term.get('amount', 0.0)}",
                    "extraction_confidence": monetary_term.get("confidence", 0.0)
                }
            )
            nodes.append(monetary_node)
            
            # Link to most relevant clause (first clause for now - could be enhanced)
            if clause_nodes:
                contains_rel = GraphRelationship(
                    from_node_id=clause_nodes[0].node_id,  # Link to first clause
                    to_node_id=monetary_node.node_id,
                    relationship_type=RelationshipType.CONTAINS,
                    properties={
                        "relevance_score": 0.7,
                        "extraction_method": "pattern_matching"
                    }
                )
                relationships.append(contains_rel)
        
        return nodes, relationships
    
    def _create_date_elements(self, metadata: ContractMetadata,
                            agreement_id: str) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Create Date nodes and temporal relationships"""
        nodes = []
        relationships = []
        
        for i, date_info in enumerate(metadata.dates):
            # Create Date node
            date_node = GraphNode(
                node_id=f"date_{metadata.document_id}_{i}",
                node_type=NodeType.DATE,
                properties={
                    "date_value": date_info.date_value,
                    "date_type": date_info.date_type,
                    "description": date_info.description,
                    "is_relative": date_info.is_relative,
                    "extraction_confidence": date_info.confidence
                }
            )
            nodes.append(date_node)
            
            # Create temporal relationship based on date type
            if date_info.date_type == "effective":
                rel_type = RelationshipType.EFFECTIVE_FROM
            elif date_info.date_type in ["expiry", "termination"]:
                rel_type = RelationshipType.EXPIRES_ON
            else:
                rel_type = RelationshipType.REFERENCES
            
            date_rel = GraphRelationship(
                from_node_id=agreement_id,
                to_node_id=date_node.node_id,
                relationship_type=rel_type,
                properties={
                    "confidence": date_info.confidence
                }
            )
            relationships.append(date_rel)
        
        return nodes, relationships
    
    def _create_governing_law_elements(self, metadata: ContractMetadata,
                                     agreement_id: str) -> Tuple[GraphNode, GraphRelationship]:
        """Create GoverningLaw node and relationship"""
        law_node = GraphNode(
            node_id=f"governing_law_{metadata.document_id}",
            node_type=NodeType.GOVERNING_LAW,
            properties={
                "jurisdiction": metadata.governing_law,
                "country": "Unknown",  # Could be enhanced with jurisdiction parsing
                "dispute_resolution": "courts"  # Default
            }
        )
        
        law_relationship = GraphRelationship(
            from_node_id=agreement_id,
            to_node_id=law_node.node_id,
            relationship_type=RelationshipType.GOVERNED_BY,
            properties={
                "explicitly_stated": True,
                "confidence": 0.8
            }
        )
        
        return law_node, law_relationship
    
    def _create_risk_assessment_elements(self, risk_data: Dict[str, Any], clause_id: str,
                                       document_id: str) -> Tuple[GraphNode, GraphRelationship]:
        """Create RiskAssessment node and HAS_RISK relationship"""
        risk_node = GraphNode(
            node_id=f"risk_{clause_id}",
            node_type=NodeType.RISK_ASSESSMENT,
            properties={
                "risk_level": risk_data.get("risk_level", "MEDIUM"),
                "risk_score": risk_data.get("risk_score", 0.5),
                "confidence": risk_data.get("confidence", 0.0),
                "risk_factors": risk_data.get("risk_factors", []),
                "assessment_method": risk_data.get("assessment_method", "ML"),
                "assessment_timestamp": datetime.now().isoformat()
            }
        )
        
        risk_relationship = GraphRelationship(
            from_node_id=clause_id,
            to_node_id=risk_node.node_id,
            relationship_type=RelationshipType.HAS_RISK,
            properties={
                "risk_contribution": risk_data.get("risk_score", 0.5),
                "assessment_timestamp": datetime.now().isoformat()
            }
        )
        
        return risk_node, risk_relationship
    
    def _create_document_span_elements(self, metadata: ContractMetadata, module3_data: Dict[str, Any],
                                     clause_nodes: List[GraphNode]) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Create DocumentSpan nodes for provenance tracking"""
        nodes = []
        relationships = []
        
        clauses_data = module3_data.get("clauses", [])
        
        for i, clause_data in enumerate(clauses_data):
            if i < len(clause_nodes):
                clause_node = clause_nodes[i]
                
                # Create DocumentSpan node
                span_node = GraphNode(
                    node_id=f"span_{metadata.document_id}_{i}",
                    node_type=NodeType.DOCUMENT_SPAN,
                    properties={
                        "document_id": metadata.document_id,
                        "start_char": clause_data.get("start_pos", 0),
                        "end_char": clause_data.get("end_pos", 0),
                        "text_snippet": clause_data.get("text", "")[:200],  # First 200 chars
                        "extraction_method": clause_data.get("extraction_method", "BERT-CRF"),
                        "confidence": clause_data.get("confidence", 0.0)
                    }
                )
                nodes.append(span_node)
                
                # Create EXTRACTED_FROM relationship
                extraction_rel = GraphRelationship(
                    from_node_id=clause_node.node_id,
                    to_node_id=span_node.node_id,
                    relationship_type=RelationshipType.EXTRACTED_FROM,
                    properties={
                        "extraction_timestamp": datetime.now().isoformat(),
                        "extraction_method": clause_data.get("extraction_method", "BERT-CRF"),
                        "confidence": clause_data.get("confidence", 0.0)
                    }
                )
                relationships.append(extraction_rel)
        
        return nodes, relationships
    
    def get_ingestion_summary(self) -> Dict[str, Any]:
        """Get summary of ingestion statistics"""
        return {
            "summary": {
                "documents_processed": self.ingestion_stats["documents_processed"],
                "total_nodes_created": self.ingestion_stats["nodes_created"],
                "total_relationships_created": self.ingestion_stats["relationships_created"],
                "error_count": len(self.ingestion_stats["errors"])
            },
            "errors": self.ingestion_stats["errors"]
        }
    
    def validate_ingestion(self) -> Dict[str, Any]:
        """Validate the ingested data by running sample queries"""
        logger.info("Running ingestion validation...")
        
        validation_results = {}
        
        try:
            # Check node counts
            node_stats = {}
            for node_type in NodeType:
                nodes = self.graph_manager.find_nodes_by_type(node_type, limit=1000)
                node_stats[node_type.value] = len(nodes)
            
            validation_results["node_counts"] = node_stats
            
            # Run sample queries
            sample_results = self.graph_manager.execute_sample_queries()
            validation_results["sample_query_results"] = {
                name: len(results) for name, results in sample_results.items()
            }
            
            # Check for basic data integrity
            integrity_checks = self._run_integrity_checks()
            validation_results["integrity_checks"] = integrity_checks
            
            validation_results["status"] = "success"
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results = {
                "status": "error",
                "error": str(e)
            }
        
        return validation_results
    
    def _run_integrity_checks(self) -> Dict[str, Any]:
        """Run basic integrity checks on the ingested data"""
        checks = {}
        
        try:
            # Check that all agreements have at least one clause
            query = """
            MATCH (a:Agreement)
            WHERE NOT (a)-[:HAS_CLAUSE]->()
            RETURN count(a) as agreements_without_clauses
            """
            result = self.graph_manager.conn.execute_query(query)
            checks["agreements_without_clauses"] = result[0]["agreements_without_clauses"] if result else 0
            
            # Check that all clauses have provenance
            query = """
            MATCH (c:Clause)
            WHERE NOT (c)-[:EXTRACTED_FROM]->()
            RETURN count(c) as clauses_without_provenance
            """
            result = self.graph_manager.conn.execute_query(query)
            checks["clauses_without_provenance"] = result[0]["clauses_without_provenance"] if result else 0
            
            # Check orphaned nodes (nodes with no relationships)
            query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n) as node_type, count(n) as count
            """
            result = self.graph_manager.conn.execute_query(query)
            checks["orphaned_nodes"] = result
            
        except Exception as e:
            checks["error"] = str(e)
        
        return checks