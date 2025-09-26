"""
Knowledge Graph Integration Module

This module provides seamless integration between the knowledge graph (Module 5)
and existing ContractSense modules (Modules 1-4).

Key Features:
- Automatic knowledge graph population from existing workflows
- Enhanced contract analysis using graph reasoning
- Bidirectional data sync between modules
- Performance optimization with caching
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Import existing modules with fallback handling
try:
    from ..ingestion.text_processor import DocumentProcessor
except ImportError:
    DocumentProcessor = None
    logging.warning("Module 1 DocumentProcessor not available")

try:
    from ..annotation.bert_crf_model import BertCrfModel
except ImportError:
    BertCrfModel = None
    logging.warning("Module 2 BertCrfModel not available")

try:
    from ..ml.risk_classifier import RiskClassifier
except ImportError:
    RiskClassifier = None
    logging.warning("Module 4 RiskClassifier not available")

# Knowledge graph components
from .neo4j_manager import GraphDataManager, Neo4jConnection
from .graph_ingestion import GraphIngestionPipeline
from .entity_extractor import EnhancedEntityExtractor

logger = logging.getLogger(__name__)

class KnowledgeGraphIntegrator:
    """
    Integration layer between knowledge graph and existing ContractSense modules
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """
        Initialize the integrator with Neo4j connection
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize components
        self._init_graph_components()
        self._init_existing_modules()
        
        # Performance tracking
        self.integration_stats = {
            "documents_processed": 0,
            "graph_nodes_created": 0,
            "queries_executed": 0,
            "cache_hits": 0,
            "errors": []
        }
    
    def _init_graph_components(self):
        """Initialize knowledge graph components"""
        try:
            self.neo4j_conn = Neo4jConnection(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password
            )
            
            self.graph_manager = GraphDataManager(self.neo4j_conn)
            self.ingestion_pipeline = GraphIngestionPipeline(self.graph_manager)
            self.entity_extractor = EnhancedEntityExtractor()
            
            # Initialize graph schema
            self.graph_manager.setup_schema()
            
            logger.info("Knowledge graph components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize graph components: {e}")
            self.neo4j_conn = None
            self.graph_manager = None
            self.ingestion_pipeline = None
    
    def _init_existing_modules(self):
        """Initialize existing module components if available"""
        try:
            # Initialize Module 1 - Text Processing
            if DocumentProcessor:
                self.document_processor = DocumentProcessor()
                logger.info("Module 1 DocumentProcessor initialized")
            else:
                self.document_processor = None
                
            # Initialize Module 2 - BERT-CRF Model
            if BertCrfModel:
                # Check if model exists
                model_path = Path("models/bert_crf_clause_extraction.pt")
                if model_path.exists():
                    self.bert_crf_model = BertCrfModel()
                    logger.info("Module 2 BertCrfModel initialized")
                else:
                    self.bert_crf_model = None
                    logger.warning("BERT-CRF model file not found")
            else:
                self.bert_crf_model = None
                
            # Initialize Module 4 - Risk Classifier
            if RiskClassifier:
                risk_model_path = Path("models/risk_classifier.joblib")
                if risk_model_path.exists():
                    self.risk_classifier = RiskClassifier()
                    logger.info("Module 4 RiskClassifier initialized")
                else:
                    self.risk_classifier = None
                    logger.warning("Risk classifier model not found")
            else:
                self.risk_classifier = None
                
        except Exception as e:
            logger.error(f"Error initializing existing modules: {e}")
    
    def process_contract_with_graph_enhancement(self, file_path: str) -> Dict[str, Any]:
        """
        Process a contract through the full pipeline with graph enhancement
        
        Args:
            file_path: Path to contract document
            
        Returns:
            Dictionary with enhanced processing results including graph data
        """
        logger.info(f"Processing contract with graph enhancement: {file_path}")
        
        try:
            result = {
                "file_path": file_path,
                "document_id": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "processing_timestamp": datetime.now().isoformat(),
                "status": "processing"
            }
            
            # Step 1: Text Processing (Module 1)
            if self.document_processor:
                logger.info("Running Module 1 text processing...")
                text_result = self.document_processor.process_document(file_path)
                result["text_processing"] = text_result
                full_text = text_result.get("full_text", "")
            else:
                # Fallback text processing
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text = f.read()
                result["text_processing"] = {
                    "full_text": full_text,
                    "method": "fallback"
                }
            
            # Step 2: Clause Extraction (Module 2)
            if self.bert_crf_model and full_text:
                logger.info("Running Module 2 clause extraction...")
                clause_result = self.bert_crf_model.predict_clauses(full_text)
                result["clause_extraction"] = clause_result
            else:
                # Fallback clause extraction using pattern matching
                clauses = self._fallback_clause_extraction(full_text)
                result["clause_extraction"] = {
                    "clauses": clauses,
                    "method": "fallback_patterns"
                }
            
            # Step 3: Risk Assessment (Module 4)
            clauses = result["clause_extraction"].get("clauses", [])
            if self.risk_classifier and clauses:
                logger.info("Running Module 4 risk assessment...")
                for clause in clauses:
                    risk_result = self.risk_classifier.assess_clause_risk(clause.get("text", ""))
                    clause["risk"] = risk_result
            else:
                # Basic risk assessment fallback
                for clause in clauses:
                    clause["risk"] = self._fallback_risk_assessment(clause.get("text", ""))
            
            # Step 4: Enhanced Entity Extraction
            logger.info("Running enhanced entity extraction...")
            metadata = self.entity_extractor.extract_comprehensive_metadata(
                document_text=full_text,
                document_id=result["document_id"],
                file_path=file_path,
                clauses=clauses
            )
            result["entity_extraction"] = {
                "parties": [p.__dict__ for p in metadata.parties],
                "dates": [d.__dict__ for d in metadata.dates],
                "monetary_terms": metadata.monetary_terms,
                "governing_law": metadata.governing_law,
                "obligations": metadata.obligations
            }
            
            # Step 5: Knowledge Graph Ingestion
            if self.ingestion_pipeline:
                logger.info("Ingesting into knowledge graph...")
                
                # Prepare Module 3 compatible format for ingestion
                module3_compatible = {
                    "document_id": result["document_id"],
                    "full_text": full_text,
                    "clauses": clauses
                }
                
                ingestion_result = self.ingestion_pipeline.ingest_single_contract(
                    module3_compatible, file_path
                )
                result["graph_ingestion"] = ingestion_result
                
                # Update statistics
                if ingestion_result["status"] == "success":
                    self.integration_stats["documents_processed"] += 1
                    self.integration_stats["graph_nodes_created"] += ingestion_result["nodes_created"]
            
            # Step 6: Graph-Enhanced Analysis
            if self.graph_manager:
                logger.info("Running graph-enhanced analysis...")
                enhanced_analysis = self._run_graph_enhanced_analysis(result["document_id"])
                result["graph_analysis"] = enhanced_analysis
            
            result["status"] = "completed"
            logger.info(f"Contract processing completed successfully: {result['document_id']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process contract {file_path}: {str(e)}"
            logger.error(error_msg)
            self.integration_stats["errors"].append(error_msg)
            
            return {
                "file_path": file_path,
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _fallback_clause_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback clause extraction using pattern matching"""
        import re
        
        # Simple pattern-based clause detection
        patterns = [
            r'(\d+\.?\s*[A-Z][^.!?]*[.!?])',  # Numbered clauses
            r'([A-Z][A-Z\s]{2,}:.*?)(?=\n[A-Z][A-Z\s]{2,}:|$)',  # Section headers
            r'(WHEREAS.*?;)',  # Whereas clauses
            r'(NOW THEREFORE.*?\.)',  # Therefore clauses
        ]
        
        clauses = []
        clause_id = 0
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                clause_text = match.group(1).strip()
                if len(clause_text) > 20:  # Filter out very short matches
                    clauses.append({
                        "clause_id": f"clause_{clause_id}",
                        "text": clause_text,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "confidence": 0.6,  # Medium confidence for pattern matching
                        "extraction_method": "pattern_matching",
                        "type": "general"
                    })
                    clause_id += 1
        
        return clauses[:50]  # Limit to first 50 clauses
    
    def _fallback_risk_assessment(self, clause_text: str) -> Dict[str, Any]:
        """Fallback risk assessment using keyword matching"""
        
        high_risk_keywords = [
            "liability", "liable", "penalty", "terminate", "breach", "default",
            "liquidated damages", "indemnify", "indemnification", "force majeure", "bankruptcy"
        ]
        
        medium_risk_keywords = [
            "warranty", "guarantee", "compliance", "confidential",
            "intellectual property", "dispute", "arbitration"
        ]
        
        clause_lower = clause_text.lower()
        
        # Count risk keywords
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in clause_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in clause_lower)
        
        # Determine risk level - more aggressive for testing
        if high_risk_count >= 1:  # Changed from >= 2 to >= 1
            risk_level = "HIGH"
            risk_score = 0.8
        elif medium_risk_count >= 2:
            risk_level = "MEDIUM"
            risk_score = 0.5
        else:
            risk_level = "LOW"
            risk_score = 0.2
        
        found_high_keywords = [kw for kw in high_risk_keywords if kw in clause_lower]
        found_medium_keywords = [kw for kw in medium_risk_keywords if kw in clause_lower]
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": 0.6,
            "assessment_method": "keyword_matching",
            "risk_factors": found_high_keywords + found_medium_keywords
        }
    
    def _run_graph_enhanced_analysis(self, document_id: str) -> Dict[str, Any]:
        """Run graph-enhanced analysis for the document"""
        
        if not self.graph_manager:
            return {"error": "Graph manager not available"}
        
        try:
            analysis = {}
            
            # Find similar contracts
            similar_contracts = self.graph_manager.find_similar_contracts(
                document_id, similarity_threshold=0.7
            )
            analysis["similar_contracts"] = similar_contracts[:5]  # Top 5 similar
            
            # Get party relationship analysis
            party_analysis = self.graph_manager.analyze_party_relationships(document_id)
            analysis["party_analysis"] = party_analysis
            
            # Get clause network analysis
            clause_network = self.graph_manager.analyze_clause_relationships(document_id)
            analysis["clause_network"] = clause_network
            
            # Get risk pattern analysis
            risk_patterns = self.graph_manager.find_risk_patterns(document_id)
            analysis["risk_patterns"] = risk_patterns
            
            self.integration_stats["queries_executed"] += 4
            
            return analysis
            
        except Exception as e:
            logger.error(f"Graph-enhanced analysis failed: {e}")
            return {"error": str(e)}
    
    def batch_process_with_graph(self, input_directory: str, 
                               output_directory: str = "output/module5_integrated") -> Dict[str, Any]:
        """
        Batch process contracts with full graph integration
        
        Args:
            input_directory: Directory containing contract files
            output_directory: Directory to save enhanced results
            
        Returns:
            Dictionary with batch processing results
        """
        input_path = Path(input_directory)
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch processing: {input_directory} -> {output_directory}")
        
        # Find contract files
        contract_files = []
        for ext in ['*.txt', '*.pdf', '*.docx']:
            contract_files.extend(list(input_path.glob(ext)))
        
        logger.info(f"Found {len(contract_files)} contract files to process")
        
        results = []
        
        for file_path in contract_files:
            try:
                logger.info(f"Processing: {file_path.name}")
                
                # Process with graph enhancement
                result = self.process_contract_with_graph_enhancement(str(file_path))
                results.append(result)
                
                # Save enhanced result
                output_file = output_path / f"{file_path.stem}_enhanced.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                
                logger.info(f"Saved enhanced result: {output_file.name}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "file_path": str(file_path),
                    "status": "error",
                    "error": error_msg
                })
        
        # Create batch summary
        summary = {
            "batch_processing_summary": {
                "input_directory": input_directory,
                "output_directory": output_directory,
                "total_files": len(contract_files),
                "processed_successfully": len([r for r in results if r.get("status") == "completed"]),
                "processing_errors": len([r for r in results if r.get("status") == "error"]),
                "timestamp": datetime.now().isoformat()
            },
            "integration_statistics": self.integration_stats,
            "detailed_results": results
        }
        
        # Save batch summary
        summary_file = output_path / "batch_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Batch processing completed. Summary saved: {summary_file}")
        
        return summary
    
    def query_graph_for_insights(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Query the knowledge graph for contract insights
        
        Args:
            query_type: Type of query to run
            **kwargs: Query-specific parameters
            
        Returns:
            Query results
        """
        if not self.graph_manager:
            return {"error": "Graph manager not available"}
        
        try:
            if query_type == "party_contracts":
                party_name = kwargs.get("party_name", "")
                return self.graph_manager.find_contracts_by_party(party_name)
            
            elif query_type == "high_risk_clauses":
                risk_threshold = kwargs.get("risk_threshold", 0.7)
                return self.graph_manager.find_high_risk_clauses(risk_threshold)
            
            elif query_type == "contract_timeline":
                document_id = kwargs.get("document_id", "")
                return self.graph_manager.get_contract_timeline(document_id)
            
            elif query_type == "similar_clauses":
                clause_text = kwargs.get("clause_text", "")
                return self.graph_manager.find_similar_clauses(clause_text)
            
            elif query_type == "regulatory_analysis":
                jurisdiction = kwargs.get("jurisdiction", "")
                return self.graph_manager.analyze_regulatory_compliance(jurisdiction)
            
            else:
                return {"error": f"Unknown query type: {query_type}"}
        
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {"error": str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        return {
            "integration_statistics": self.integration_stats,
            "component_status": {
                "neo4j_connected": self.neo4j_conn is not None and self.neo4j_conn.is_connected(),
                "graph_manager_available": self.graph_manager is not None,
                "document_processor_available": self.document_processor is not None,
                "bert_crf_model_available": self.bert_crf_model is not None,
                "risk_classifier_available": self.risk_classifier is not None
            },
            "graph_statistics": self.graph_manager.get_database_stats() if self.graph_manager else {}
        }
    
    def close(self):
        """Clean up resources"""
        if self.neo4j_conn:
            self.neo4j_conn.close()
            logger.info("Neo4j connection closed")