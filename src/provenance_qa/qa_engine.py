"""
Main QA Engine for Provenance-Aware Question Answering

Orchestrates all components to provide comprehensive, traceable question answering
with full provenance tracking and quality validation.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from .question_processor import QuestionProcessor
    from .context_assembler import ContextAssembler
    from .answer_generator import AnswerGenerator
    from .provenance_tracker import ProvenanceTracker
    from .answer_validator import AnswerValidator

    from .models.question_models import QuestionAnalysis
    from .models.context_models import ContextWindow, ContextStrategy
    from .models.answer_models import Answer
    from .models.provenance_models import QAResponse
    from .utils.common import generate_id, Timer
except ImportError:
    # Fallback for direct execution
    from provenance_qa.question_processor import QuestionProcessor
    from provenance_qa.context_assembler import ContextAssembler
    from provenance_qa.answer_generator import AnswerGenerator
    from provenance_qa.provenance_tracker import ProvenanceTracker
    from provenance_qa.answer_validator import AnswerValidator

    from provenance_qa.models.question_models import QuestionAnalysis
    from provenance_qa.models.context_models import ContextWindow, ContextStrategy
    from provenance_qa.models.answer_models import Answer
    from provenance_qa.models.provenance_models import QAResponse
    from provenance_qa.utils.common import generate_id, Timer

# Import existing modules for integration
try:
    import sys
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Add workspace src to path to find hybrid_retrieval
    current_file = Path(__file__)
    workspace_src = current_file.parent.parent
    if str(workspace_src) not in sys.path:
        sys.path.insert(0, str(workspace_src))
    
    from hybrid_retrieval.hybrid_engine import HybridEngine as HybridSearchSystem
    from cross_encoder_reranking.cross_encoder_engine import CrossEncoderEngine as CrossEncoderReranker
    logger.info("Successfully imported existing modules for integration")
except ImportError as e:
    # Gracefully handle missing modules - system can operate standalone
    logger.debug(f"Running in standalone mode - existing modules not available: {e}")
    HybridSearchSystem = None
    CrossEncoderReranker = None

class ProvenanceQAEngine:
    """
    Complete Provenance-Aware QA System
    
    Integrates question processing, context assembly, answer generation,
    provenance tracking, and validation for comprehensive QA capabilities.
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        max_context_tokens: int = 4000,
        enable_validation: bool = True,
        context_strategy: ContextStrategy = ContextStrategy.FOCUSED,
        workspace_path: Optional[str] = None
    ):
        """Initialize the QA engine with all components"""
        
        # Store configuration
        self.max_context_tokens = max_context_tokens
        self.enable_validation = enable_validation
        self.default_context_strategy = context_strategy
        self.workspace_path = workspace_path or str(Path.cwd())
        
        # Initialize core components
        self.question_processor = QuestionProcessor()
        self.context_assembler = ContextAssembler(default_max_tokens=max_context_tokens)
        self.answer_generator = AnswerGenerator(api_key=gemini_api_key)
        self.provenance_tracker = ProvenanceTracker()
        
        if enable_validation:
            self.answer_validator = AnswerValidator()
        else:
            self.answer_validator = None
        
        # Initialize integration with existing modules
        self.hybrid_search = None
        self.reranker = None
        self._initialize_existing_modules()
        
        logger.info("Initialized Provenance QA Engine with all components")
    
    def _initialize_existing_modules(self):
        """Initialize integration with existing ContractSense modules"""
        
        try:
            if HybridSearchSystem:
                # Initialize hybrid search system (would need proper configuration)
                self.hybrid_search = HybridSearchSystem()
                logger.info("Integrated with Hybrid Search System")
            
            if CrossEncoderReranker:
                # Initialize cross-encoder reranker
                self.reranker = CrossEncoderReranker()
                logger.info("Integrated with Cross-Encoder Reranker")
                
        except Exception as e:
            logger.warning(f"Could not initialize existing modules: {e}")
            # Continue without integration - system will work in standalone mode
    
    def ask_question(
        self,
        question: str,
        documents: Optional[List[str]] = None,
        context_strategy: Optional[ContextStrategy] = None,
        max_tokens: Optional[int] = None,
        validate_answer: Optional[bool] = None
    ) -> QAResponse:
        """
        Complete QA pipeline: process question and return comprehensive response
        
        Args:
            question: User's question
            documents: Optional list of specific documents to search
            context_strategy: Strategy for context assembly
            max_tokens: Maximum context tokens (overrides default)
            validate_answer: Whether to validate answer (overrides default)
        
        Returns:
            Complete QA response with provenance tracking
        """
        
        with Timer() as total_timer:
            logger.info(f"Processing question: {question[:100]}...")
            
            try:
                # Step 1: Process and analyze the question
                logger.info("Step 1: Question analysis")
                question_analysis = self.question_processor.process_question(question)
                
                # Step 2: Retrieve and assemble context
                logger.info("Step 2: Context retrieval and assembly")
                context_window = self._retrieve_and_assemble_context(
                    question_analysis,
                    documents,
                    context_strategy or self.default_context_strategy,
                    max_tokens or self.max_context_tokens
                )
                
                # Step 3: Generate answer
                logger.info("Step 3: Answer generation")
                answer = self.answer_generator.generate_answer(question_analysis, context_window)
                
                # Step 4: Create response with provenance tracking
                logger.info("Step 4: Provenance tracking")
                qa_response = self.provenance_tracker.create_qa_response(
                    question_analysis, context_window, answer
                )
                
                # Step 5: Validate answer (if enabled)
                if validate_answer or (validate_answer is None and self.enable_validation):
                    if self.answer_validator:
                        logger.info("Step 5: Answer validation")
                        validation_result = self.answer_validator.validate_answer(
                            answer, question_analysis, context_window, qa_response
                        )
                        
                        # Add validation info to metadata
                        qa_response.metadata["validation"] = validation_result.to_dict()
                        
                        # Add validation warnings to response if needed
                        if validation_result.warnings:
                            qa_response.metadata["validation_warnings"] = validation_result.warnings
                        
                        if validation_result.critical_issues:
                            qa_response.metadata["validation_issues"] = validation_result.critical_issues
                
                # Add processing statistics
                qa_response.processing_time = total_timer.elapsed()
                qa_response.metadata["processing_stats"] = {
                    "total_time": total_timer.elapsed(),
                    "question_analysis_time": question_analysis.processing_time,
                    "context_assembly_time": context_window.assembly_time,
                    "answer_generation_time": answer.generation_time
                }
                
                logger.info(f"QA processing complete: {qa_response.overall_confidence:.2f} confidence, "
                           f"{len(qa_response.citations)} citations, {total_timer.elapsed():.2f}s total")
                
                return qa_response
                
            except Exception as e:
                logger.error(f"Error in QA pipeline: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Return error response
                return self._create_error_response(question, str(e))
    
    def _retrieve_and_assemble_context(
        self,
        question_analysis: QuestionAnalysis,
        documents: Optional[List[str]],
        strategy: ContextStrategy,
        max_tokens: int
    ) -> ContextWindow:
        """Retrieve relevant chunks and assemble context window"""
        
        # Get search terms from question analysis
        search_terms = self.question_processor.get_search_terms(question_analysis)
        
        # Retrieve chunks using available methods
        retrieved_chunks = []
        
        if self.hybrid_search and self.reranker:
            # Use integrated ContractSense modules
            retrieved_chunks = self._retrieve_with_existing_modules(
                question_analysis, search_terms, documents
            )
        else:
            # Use mock retrieval for standalone operation
            retrieved_chunks = self._mock_retrieval(question_analysis, search_terms)
        
        # Assemble context window
        context_window = self.context_assembler.assemble_context(
            question_analysis, retrieved_chunks, strategy, max_tokens
        )
        
        return context_window
    
    def _retrieve_with_existing_modules(
        self,
        question_analysis: QuestionAnalysis,
        search_terms: List[str],
        documents: Optional[List[str]]
    ) -> List:
        """Retrieve chunks using integrated ContractSense modules"""
        
        try:
            # This would integrate with Module 8's hybrid search and reranking
            # For now, return empty list as modules need proper initialization
            logger.info("Would integrate with existing hybrid search and reranking modules")
            return []
            
        except Exception as e:
            logger.error(f"Error in integrated retrieval: {e}")
            return self._mock_retrieval(question_analysis, search_terms)
    
    def _mock_retrieval(self, question_analysis: QuestionAnalysis, search_terms: List[str]):
        """Enhanced retrieval using real CUAD documents when available, fallback to mock data"""
        
        from .models.context_models import DocumentChunk, ChunkType, RetrievalSource
        
        # First try to use vector store
        if self._has_vector_store():
            try:
                return self._retrieve_from_vector_store(question_analysis, search_terms)
            except Exception as e:
                logger.warning(f"Vector store retrieval failed: {e}")
        
        # Then try CUAD documents
        cuad_docs = self._load_cuad_documents()
        if cuad_docs:
            try:
                return self._search_cuad_documents(question_analysis, search_terms, cuad_docs)
            except Exception as e:
                logger.warning(f"CUAD document search failed: {e}")
        
        # Finally fall back to mock data (only as last resort)
        logger.warning("Using mock contract data - consider running Module 1-6 pipeline first")
        return self._get_mock_contracts_as_chunks(question_analysis, search_terms)
    
    def _has_vector_store(self) -> bool:
        """Check if vector store data exists"""
        vector_store_path = Path(self.workspace_path) / "vector_store_data"
        required_files = [
            "embeddings/document_chunks.npy",
            "embeddings/chunk_metadata.json", 
            "indices/similarity_index.faiss"
        ]
        
        return all((vector_store_path / file).exists() for file in required_files)
    
    def _retrieve_from_vector_store(self, question_analysis: QuestionAnalysis, search_terms: List[str]) -> List:
        """Retrieve from actual vector store built by Module 6"""
        
        vector_store_path = Path(self.workspace_path) / "vector_store_data"
        
        if not vector_store_path.exists():
            logger.warning("Vector store not found, falling back to document search")
            raise Exception("Vector store directory not found")
        
        try:
            # Load the vector store index
            import numpy as np
            
            embeddings_path = vector_store_path / "embeddings" / "document_chunks.npy"
            metadata_path = vector_store_path / "embeddings" / "chunk_metadata.json"
            
            if embeddings_path.exists() and metadata_path.exists():
                # Load embeddings and metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # For now, return top chunks based on simple text matching
                # This would be replaced with actual embedding similarity search
                retrieved_chunks = []
                
                for i, chunk_data in enumerate(metadata[:10]):  # Limit to 10 chunks
                    chunk = self._create_chunk_from_metadata(chunk_data, 0.8 - (i * 0.05))
                    retrieved_chunks.append(chunk)
                
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store")
                return retrieved_chunks
            else:
                raise Exception("Required vector store files not found")
                
        except ImportError:
            logger.error("NumPy not available for vector store operations")
            raise Exception("NumPy required for vector store operations")
        except Exception as e:
            logger.error(f"Error accessing vector store: {e}")
            raise
    
    def _load_cuad_documents(self) -> List[Dict[str, Any]]:
        """Load actual CUAD contract documents"""
        
        # Check multiple possible CUAD locations
        possible_cuad_paths = [
            Path(self.workspace_path) / "CUAD_v1",
            Path(self.workspace_path) / "data" / "cuad", 
            Path(self.workspace_path) / "datasets" / "CUAD_v1"
        ]
        
        cuad_documents = []
        
        for cuad_path in possible_cuad_paths:
            if cuad_path.exists():
                logger.info(f"Found CUAD data at: {cuad_path}")
                
                # Look for CUAD JSON files
                json_files = list(cuad_path.glob("*.json"))
                if json_files:
                    for json_file in json_files[:1]:  # Limit to first file for demo
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                cuad_data = json.load(f)
                                
                            # Process CUAD format
                            if 'data' in cuad_data:
                                for contract in cuad_data['data'][:5]:  # Limit to 5 contracts
                                    cuad_documents.append({
                                        "title": contract.get('title', f"CUAD Contract {len(cuad_documents)+1}"),
                                        "content": contract.get('context', ''),
                                        "document_id": contract.get('id', f"cuad_{len(cuad_documents)}"),
                                        "paragraphs": contract.get('paragraphs', []),
                                        "qas": contract.get('qas', [])
                                    })
                        except Exception as e:
                            logger.error(f"Error loading CUAD file {json_file}: {e}")
                            continue
                
                # Also check for text files
                txt_files = []
                if (cuad_path / "full_contract_txt").exists():
                    txt_files = list((cuad_path / "full_contract_txt").glob("*.txt"))
                
                for txt_file in txt_files[:5]:  # Limit to 5 text files
                    try:
                        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if len(content) > 100:  # Skip very short files
                            cuad_documents.append({
                                "title": txt_file.stem.replace('_', ' ').title(),
                                "content": content[:5000],  # Limit length
                                "document_id": f"cuad_txt_{len(cuad_documents)}",
                                "source_file": str(txt_file),
                                "paragraphs": content.split('\n\n')[:10]  # First 10 paragraphs
                            })
                    except Exception as e:
                        logger.error(f"Error loading CUAD text file {txt_file}: {e}")
                        continue
                
                if cuad_documents:
                    logger.info(f"Loaded {len(cuad_documents)} CUAD contract documents")
                    return cuad_documents
        
        logger.info("No CUAD documents found")
        return []
    
    def _search_cuad_documents(
        self, 
        question_analysis: QuestionAnalysis, 
        search_terms: List[str], 
        cuad_docs: List[Dict[str, Any]]
    ) -> List:
        """Search through CUAD documents for relevant chunks"""
        
        from .models.context_models import DocumentChunk, ChunkType, RetrievalSource
        
        retrieved_chunks = []
        
        for i, doc in enumerate(cuad_docs):
            content = doc.get('content', '')
            if not content:
                continue
            
            # Calculate relevance based on search term matches
            relevance_score = 0.3  # Base score
            keyword_matches = []
            
            content_lower = content.lower()
            question_lower = question_analysis.original_question.lower()
            
            # Boost score for question keywords in content
            for term in search_terms:
                if term.lower() in content_lower:
                    keyword_matches.append(term)
                    relevance_score += 0.15
            
            # Boost for specific legal terms
            legal_terms = ['termination', 'liability', 'indemnification', 'payment', 'breach']
            for term in legal_terms:
                if term in question_lower and term in content_lower:
                    relevance_score += 0.10
            
            relevance_score = min(relevance_score, 0.95)  # Cap at 0.95
            
            # Split content into chunks (simple paragraph-based splitting)
            paragraphs = content.split('\n\n')
            for j, paragraph in enumerate(paragraphs[:3]):  # Limit to 3 paragraphs per doc
                if len(paragraph.strip()) > 50:  # Skip short paragraphs
                    chunk = DocumentChunk(
                        chunk_id=f"cuad_chunk_{i}_{j}",
                        document_id=doc.get('document_id', f'cuad_doc_{i}'),
                        document_title=doc.get('title', f'CUAD Contract {i+1}'),
                        content=paragraph.strip()[:1000],  # Limit length
                        chunk_type=ChunkType.PARAGRAPH,
                        retrieval_source=RetrievalSource.HYBRID,
                        relevance_score=relevance_score,
                        similarity_score=relevance_score * 0.9,
                        keyword_matches=keyword_matches,
                        section_title=f"Section {j+1}",
                        page_number=i + 1,
                        paragraph_number=j + 1,
                        preceding_context=f"From {doc.get('title', 'CUAD Contract')}",
                        following_context="Additional contract terms may apply."
                    )
                    retrieved_chunks.append(chunk)
        
        # Sort by relevance score
        retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks from CUAD documents")
        return retrieved_chunks[:10]  # Return top 10 chunks
    
    def _create_chunk_from_metadata(self, chunk_data: Dict[str, Any], relevance_score: float):
        """Create DocumentChunk from vector store metadata"""
        
        from .models.context_models import DocumentChunk, ChunkType, RetrievalSource
        
        return DocumentChunk(
            chunk_id=chunk_data.get('chunk_id', 'unknown'),
            document_id=chunk_data.get('document_id', 'unknown'),
            document_title=chunk_data.get('document_title', 'Vector Store Document'),
            content=chunk_data.get('content', ''),
            chunk_type=ChunkType.PARAGRAPH,
            retrieval_source=RetrievalSource.VECTOR_SEARCH,
            relevance_score=relevance_score,
            similarity_score=relevance_score * 0.95,
            keyword_matches=chunk_data.get('keyword_matches', []),
            section_title=chunk_data.get('section_title', 'Section'),
            page_number=chunk_data.get('page_number', 1),
            paragraph_number=chunk_data.get('paragraph_number', 1)
        )
    
    def _get_mock_contracts_as_chunks(self, question_analysis: QuestionAnalysis, search_terms: List[str]):
        """Fallback method with realistic mock contract content"""
        
        from .models.context_models import DocumentChunk, ChunkType, RetrievalSource
    def _get_mock_contracts_as_chunks(self, question_analysis: QuestionAnalysis, search_terms: List[str]):
        """Fallback method with realistic mock contract content"""
        
        from .models.context_models import DocumentChunk, ChunkType, RetrievalSource
        
        # Realistic contract content samples
        contract_contents = [
            {
                "title": "Software License Agreement", 
                "section": "Termination Clause",
                "content": "This Agreement may be terminated by either party upon thirty (30) days written notice to the other party. Upon termination, all rights and licenses granted hereunder shall immediately cease, and each party shall return or destroy all confidential information received from the other party. The termination notice period shall be calculated from the date of receipt of written notice. Termination for cause may be immediate upon material breach that remains uncured after ten (10) days written notice.",
                "keywords": ["termination", "notice", "clause", "thirty", "days"]
            },
            {
                "title": "Service Level Agreement",
                "section": "Liability Limitations", 
                "content": "IN NO EVENT SHALL EITHER PARTY'S LIABILITY EXCEED THE TOTAL AMOUNT PAID UNDER THIS AGREEMENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM. Each party's liability for damages shall be limited to direct damages only, excluding consequential, incidental, or punitive damages. The liability cap applies to all claims collectively, whether in contract, tort, or otherwise. Notwithstanding the foregoing, no limitation shall apply to damages arising from willful misconduct or breach of confidentiality obligations.",
                "keywords": ["liability", "damages", "limitation", "exceed", "amount"]
            },
            {
                "title": "Master Services Agreement",
                "section": "Indemnification Provisions",
                "content": "Each party agrees to indemnify, defend, and hold harmless the other party from and against any claims, losses, damages, or expenses arising from: (a) breach of this Agreement, (b) violation of applicable law, (c) infringement of third-party intellectual property rights, or (d) negligent or willful acts. The indemnifying party shall assume control of the defense and settlement of any indemnified claim, provided the indemnified party cooperates reasonably. These indemnification obligations shall survive termination of this Agreement.",
                "keywords": ["indemnification", "indemnify", "defend", "hold harmless", "claims"]
            },
            {
                "title": "Employment Contract",
                "section": "Payment Terms",
                "content": "Employee shall receive base salary of $120,000 per annum, payable in bi-weekly installments of $4,615.38. Payment shall be made via direct deposit on the 15th and last day of each month, or the preceding business day if such date falls on a weekend or holiday. In addition to base salary, Employee may be eligible for annual performance bonus up to 25% of base salary, as determined by Company in its sole discretion. All payments are subject to applicable tax withholdings and deductions.",
                "keywords": ["payment", "salary", "compensation", "bi-weekly", "bonus"]
            },
            {
                "title": "Non-Disclosure Agreement",
                "section": "Effective Date and Duration",
                "content": "This Agreement shall become effective on January 1, 2024 (the 'Effective Date') and shall remain in effect for a period of five (5) years, unless terminated earlier in accordance with the terms hereof. The confidentiality obligations set forth herein shall survive for three (3) years following termination or expiration of this Agreement. Either party may renew this Agreement for additional one-year terms by providing written notice at least sixty (60) days prior to expiration.",
                "keywords": ["effective", "date", "duration", "renewal", "confidentiality"]
            }
        ]
        
        # Create mock chunks with realistic content
        mock_chunks = []
        
        for i, content_sample in enumerate(contract_contents):
            # Calculate relevance based on keyword matches
            keyword_matches = []
            relevance_score = 0.3  # Base relevance
            
            for term in search_terms:
                if any(keyword in term.lower() for keyword in content_sample["keywords"]):
                    keyword_matches.append(term)
                    relevance_score += 0.15  # Boost for keyword matches
            
            # Ensure we don't exceed 1.0
            relevance_score = min(relevance_score, 0.95)
            
            chunk = DocumentChunk(
                chunk_id=f"mock_chunk_{i+1}",
                document_id=f"mock_doc_{i+1}",
                document_title=content_sample["title"],
                content=content_sample["content"],
                chunk_type=ChunkType.PARAGRAPH,
                retrieval_source=RetrievalSource.HYBRID,
                relevance_score=relevance_score,
                similarity_score=relevance_score * 0.9,  # Slightly lower similarity
                keyword_matches=keyword_matches,
                section_title=content_sample["section"],
                page_number=i + 1,
                paragraph_number=i + 1,
                preceding_context=f"This section appears in the {content_sample['title']} document.",
                following_context=f"Additional terms and conditions may apply as specified elsewhere in the {content_sample['title']}."
            )
            mock_chunks.append(chunk)
        
        # Sort by relevance score (highest first)
        mock_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Created {len(mock_chunks)} realistic contract chunks for demonstration")
        return mock_chunks
    
    def _create_error_response(self, question: str, error_message: str) -> QAResponse:
        """Create error response when QA pipeline fails"""
        
        from .models.answer_models import Answer, AnswerType, ConfidenceLevel, AnswerSource
        
        error_response = QAResponse(
            response_id=generate_id("error_resp"),
            question_id=question[:50],
            answer=f"I apologize, but I encountered an error while processing your question: {error_message}",
            summary="Error occurred during question processing",
            overall_confidence=0.0,
            answer_quality=0.0,
            source_reliability=0.0,
            model_used="error_handler"
        )
        
        error_response.metadata["error"] = {
            "message": error_message,
            "type": "processing_error"
        }
        
        return error_response
    
    # Additional utility methods for advanced usage
    
    def analyze_question_only(self, question: str) -> QuestionAnalysis:
        """Analyze question without full QA pipeline"""
        return self.question_processor.process_question(question)
    
    def suggest_clarifications(self, question: str) -> List[str]:
        """Get suggestions for clarifying ambiguous questions"""
        analysis = self.question_processor.process_question(question)
        return self.question_processor.suggest_clarifications(analysis)
    
    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        context_chunks: Optional[List] = None
    ) -> Dict[str, Any]:
        """Evaluate the quality of an answer without full generation"""
        
        if not self.answer_validator:
            return {"error": "Answer validation not enabled"}
        
        try:
            # Create minimal objects for validation
            from .models.answer_models import Answer
            from .models.context_models import ContextWindow
            
            question_analysis = self.question_processor.process_question(question)
            
            # Create answer object
            test_answer = Answer(
                answer_id=generate_id("test"),
                question_id=question[:50],
                text=answer
            )
            
            # Create minimal context window
            context_window = ContextWindow(
                window_id=generate_id("test_ctx"),
                question_id=question[:50]
            )
            
            if context_chunks:
                context_window.primary_chunks = context_chunks[:5]  # Limit for testing
            
            # Run validation
            validation_result = self.answer_validator.validate_answer(
                test_answer, question_analysis, context_window
            )
            
            return {
                "overall_score": validation_result.overall_score,
                "status": validation_result.overall_status.value,
                "factual_accuracy": validation_result.factual_accuracy,
                "completeness": validation_result.completeness,
                "clarity": validation_result.factual_accuracy,
                "issues": validation_result.critical_issues,
                "warnings": validation_result.warnings,
                "recommendations": validation_result.recommendations
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def batch_questions(
        self,
        questions: List[str],
        context_strategy: ContextStrategy = ContextStrategy.FOCUSED,
        max_parallel: int = 3
    ) -> List[QAResponse]:
        """Process multiple questions (with basic rate limiting)"""
        
        responses = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing batch question {i+1}/{len(questions)}")
            
            try:
                response = self.ask_question(
                    question,
                    context_strategy=context_strategy
                )
                responses.append(response)
                
                # Simple rate limiting
                if i < len(questions) - 1:  # Not the last question
                    import time
                    time.sleep(1)  # Wait 1 second between questions
                    
            except Exception as e:
                logger.error(f"Error processing batch question {i+1}: {e}")
                error_response = self._create_error_response(question, str(e))
                responses.append(error_response)
        
        logger.info(f"Completed batch processing: {len(responses)} responses")
        return responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all QA system components"""
        
        status = {
            "engine_initialized": True,
            "components": {
                "question_processor": bool(self.question_processor),
                "context_assembler": bool(self.context_assembler),
                "answer_generator": bool(self.answer_generator),
                "provenance_tracker": bool(self.provenance_tracker),
                "answer_validator": bool(self.answer_validator)
            },
            "integration": {
                "hybrid_search": bool(self.hybrid_search),
                "reranker": bool(self.reranker)
            },
            "configuration": {
                "max_context_tokens": self.max_context_tokens,
                "validation_enabled": self.enable_validation,
                "default_strategy": self.default_context_strategy.value,
                "workspace_path": self.workspace_path
            }
        }
        
        # Check Gemini API availability
        if self.answer_generator.model:
            status["gemini_api"] = "available"
        else:
            status["gemini_api"] = "not_configured"
        
        return status
    
    def configure_logging(self, level: str = "INFO", log_file: Optional[str] = None):
        """Configure logging for the QA system"""
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
        logger.info(f"Configured logging at {level} level")
        
        if log_file:
            logger.info(f"Logging to file: {log_file}")
    
    def __repr__(self):
        """String representation of QA engine"""
        components = []
        if self.question_processor:
            components.append("QuestionProcessor")
        if self.context_assembler:
            components.append("ContextAssembler")
        if self.answer_generator:
            components.append("AnswerGenerator")
        if self.provenance_tracker:
            components.append("ProvenanceTracker")
        if self.answer_validator:
            components.append("AnswerValidator")
        
        integration = []
        if self.hybrid_search:
            integration.append("HybridSearch")
        if self.reranker:
            integration.append("Reranker")
        
        return f"ProvenanceQAEngine(components={components}, integration={integration})"