"""
Document Parser - Main orchestrator for ContractSense ingestion pipeline

This is the main entry point for document processing. It handles file type detection,
routes to appropriate parsers, and applies normalization and clause detection.
"""

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    
from pathlib import Path
from typing import Union, Optional, List
import logging
from enum import Enum

from . import ProcessedDocument, DocumentType, logger
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser  
from .text_normalizer import TextNormalizer
from .clause_detector import ClauseDetector


class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    FULL = "full"  # Complete processing with all steps
    FAST = "fast"  # Skip OCR and advanced normalization
    STRUCTURE_ONLY = "structure_only"  # Only extract structure, no clause detection


class DocumentParser:
    """Main orchestrator for document ingestion and processing"""
    
    def __init__(self, 
                 enable_ocr: bool = True,
                 enable_normalization: bool = True,
                 enable_clause_detection: bool = True,
                 processing_mode: ProcessingMode = ProcessingMode.FULL):
        """
        Initialize the document parser
        
        Args:
            enable_ocr: Whether to use OCR for scanned PDFs
            enable_normalization: Whether to apply text normalization
            enable_clause_detection: Whether to detect clause boundaries
            processing_mode: Processing mode (full, fast, structure_only)
        """
        self.enable_ocr = enable_ocr
        self.enable_normalization = enable_normalization
        self.enable_clause_detection = enable_clause_detection
        self.processing_mode = processing_mode
        
        # Initialize parsers
        self.pdf_parser = PDFParser(ocr_enabled=enable_ocr)
        self.docx_parser = DOCXParser()
        self.text_normalizer = TextNormalizer()
        self.clause_detector = ClauseDetector()
        
        # Adjust settings based on processing mode
        if processing_mode == ProcessingMode.FAST:
            self.enable_ocr = False
            self.enable_normalization = False
        elif processing_mode == ProcessingMode.STRUCTURE_ONLY:
            self.enable_clause_detection = False
    
    def parse(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Parse a document file and return structured data
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessedDocument with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Starting document processing: {file_path}")
        
        # Step 1: Detect file type
        doc_type = self._detect_document_type(file_path)
        logger.info(f"Detected document type: {doc_type}")
        
        # Step 2: Parse with appropriate parser
        doc = self._parse_with_appropriate_parser(file_path, doc_type)
        
        # Step 3: Apply text normalization
        if self.enable_normalization and doc.pages:
            logger.info("Applying text normalization")
            doc = self.text_normalizer.normalize(doc)
        
        # Step 4: Detect clauses
        if self.enable_clause_detection and doc.pages:
            logger.info("Detecting clauses")
            doc = self.clause_detector.detect_clauses(doc)
        
        # Step 5: Add final metadata
        doc.metadata.update({
            "processing_mode": self.processing_mode.value,
            "ocr_enabled": self.enable_ocr,
            "normalization_enabled": self.enable_normalization,
            "clause_detection_enabled": self.enable_clause_detection,
            "file_size_bytes": file_path.stat().st_size,
            "file_name": file_path.name
        })
        
        logger.info(f"Document processing completed. "
                   f"Pages: {len(doc.pages)}, "
                   f"Clauses: {len(doc.clauses)}, "
                   f"Characters: {len(doc.full_text)}")
        
        return doc
    
    def parse_batch(self, file_paths: List[Union[str, Path]]) -> List[ProcessedDocument]:
        """
        Parse multiple documents
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessedDocument objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                doc = self.parse(file_path)
                results.append(doc)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Create error document
                error_doc = ProcessedDocument(
                    document_path=str(file_path),
                    document_type=DocumentType.UNKNOWN
                )
                error_doc.processing_errors.append(f"Parsing failed: {str(e)}")
                results.append(error_doc)
        
        return results
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect the document type based on file extension and content"""
        
        # First try file extension
        extension = file_path.suffix.lower()
        if extension == '.pdf':
            return DocumentType.PDF
        elif extension in ['.docx', '.doc']:
            return DocumentType.DOCX
        elif extension == '.txt':
            return DocumentType.TXT
        
        # Fallback to magic number detection if available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                if mime_type == 'application/pdf':
                    return DocumentType.PDF
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                  'application/msword']:
                    return DocumentType.DOCX
                elif mime_type.startswith('text/'):
                    return DocumentType.TXT
            except Exception as e:
                logger.warning(f"Could not detect MIME type for {file_path}: {e}")
        else:
            logger.warning("python-magic not available, using file extension only for type detection")
        
        return DocumentType.UNKNOWN
    
    def _parse_with_appropriate_parser(self, file_path: Path, doc_type: DocumentType) -> ProcessedDocument:
        """Route to the appropriate parser based on document type"""
        
        if doc_type == DocumentType.PDF:
            return self.pdf_parser.parse(file_path)
        elif doc_type == DocumentType.DOCX:
            return self.docx_parser.parse(file_path)
        elif doc_type == DocumentType.TXT:
            return self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    def _parse_txt(self, file_path: Path) -> ProcessedDocument:
        """Simple parser for plain text files"""
        
        doc = ProcessedDocument(
            document_path=str(file_path),
            document_type=DocumentType.TXT
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple paragraph splitting
            paragraphs_text = content.split('\n\n')
            
            from . import Page, Paragraph, TextSpan
            
            page = Page(page_num=1)
            char_offset = 0
            
            for i, para_text in enumerate(paragraphs_text):
                para_text = para_text.strip()
                if not para_text:
                    continue
                
                span = TextSpan(
                    text=para_text,
                    start_offset=char_offset,
                    end_offset=char_offset + len(para_text),
                    page_num=1,
                    confidence=1.0
                )
                
                paragraph = Paragraph(
                    spans=[span],
                    paragraph_id=f"para_{i}"
                )
                
                page.paragraphs.append(paragraph)
                char_offset += len(para_text) + 2  # +2 for paragraph breaks
            
            doc.pages = [page]
            doc.metadata["extraction_method"] = "text_file"
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            doc.processing_errors.append(f"Text file parsing error: {str(e)}")
        
        return doc


# Convenience functions for common use cases
def parse_document(file_path: Union[str, Path], 
                  enable_ocr: bool = True,
                  enable_normalization: bool = True,
                  enable_clause_detection: bool = True) -> ProcessedDocument:
    """
    Parse a single document with default settings
    
    Args:
        file_path: Path to the document
        enable_ocr: Whether to use OCR for scanned PDFs
        enable_normalization: Whether to normalize text
        enable_clause_detection: Whether to detect clauses
    
    Returns:
        ProcessedDocument
    """
    parser = DocumentParser(
        enable_ocr=enable_ocr,
        enable_normalization=enable_normalization,
        enable_clause_detection=enable_clause_detection
    )
    return parser.parse(file_path)


def parse_document_fast(file_path: Union[str, Path]) -> ProcessedDocument:
    """
    Parse a document with fast processing (no OCR, minimal normalization)
    
    Args:
        file_path: Path to the document
    
    Returns:
        ProcessedDocument
    """
    parser = DocumentParser(processing_mode=ProcessingMode.FAST)
    return parser.parse(file_path)