"""
Module 1: Document Ingestion & Preprocessing

This module handles the conversion of various document formats (PDF, DOCX, scanned PDFs) 
into structured text units with preserved layout information and clause boundaries.

Components:
- DocumentParser: Main orchestrator
- PDFParser: Handle selectable PDFs with pdfminer/PyMuPDF
- OCRParser: Handle scanned PDFs with Tesseract
- DOCXParser: Handle Word documents
- LayoutExtractor: Extract layout-aware structure
- TextNormalizer: Clean and normalize extracted text
- ClauseDetector: Detect clause boundaries using heuristics
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a bounding box for text elements"""
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class TextSpan:
    """Represents a span of text with metadata"""
    text: str
    start_offset: int
    end_offset: int
    bbox: Optional[BoundingBox] = None
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    page_num: int = 0
    confidence: float = 1.0  # OCR confidence if applicable


@dataclass
class Paragraph:
    """Represents a paragraph composed of text spans"""
    spans: List[TextSpan] = field(default_factory=list)
    paragraph_id: str = ""
    is_heading: bool = False
    heading_level: int = 0
    is_numbered: bool = False
    number_text: str = ""
    
    @property
    def text(self) -> str:
        return " ".join(span.text for span in self.spans)
    
    @property
    def start_offset(self) -> int:
        return min(span.start_offset for span in self.spans) if self.spans else 0
    
    @property
    def end_offset(self) -> int:
        return max(span.end_offset for span in self.spans) if self.spans else 0


@dataclass
class Page:
    """Represents a document page"""
    page_num: int
    paragraphs: List[Paragraph] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    
    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class Clause:
    """Represents a detected clause"""
    clause_id: str
    text: str
    clause_type: Optional[str] = None
    start_offset: int = 0
    end_offset: int = 0
    page_nums: List[int] = field(default_factory=list)
    paragraph_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    detection_method: str = "heuristic"  # heuristic, regex, ml


@dataclass
class ProcessedDocument:
    """Main container for processed document data"""
    document_path: str
    document_type: DocumentType
    pages: List[Page] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_errors: List[str] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document_path": self.document_path,
            "document_type": self.document_type.value,
            "pages": [
                {
                    "page_num": page.page_num,
                    "width": page.width,
                    "height": page.height,
                    "paragraphs": [
                        {
                            "paragraph_id": para.paragraph_id,
                            "text": para.text,
                            "start_offset": para.start_offset,
                            "end_offset": para.end_offset,
                            "is_heading": para.is_heading,
                            "heading_level": para.heading_level,
                            "is_numbered": para.is_numbered,
                            "number_text": para.number_text,
                            "spans": [
                                {
                                    "text": span.text,
                                    "start_offset": span.start_offset,
                                    "end_offset": span.end_offset,
                                    "bbox": {
                                        "x0": span.bbox.x0,
                                        "y0": span.bbox.y0,
                                        "x1": span.bbox.x1,
                                        "y1": span.bbox.y1,
                                        "page": span.bbox.page
                                    } if span.bbox else None,
                                    "font_size": span.font_size,
                                    "font_name": span.font_name,
                                    "is_bold": span.is_bold,
                                    "is_italic": span.is_italic,
                                    "page_num": span.page_num,
                                    "confidence": span.confidence
                                } for span in para.spans
                            ]
                        } for para in page.paragraphs
                    ]
                } for page in self.pages
            ],
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "text": clause.text,
                    "clause_type": clause.clause_type,
                    "start_offset": clause.start_offset,
                    "end_offset": clause.end_offset,
                    "page_nums": clause.page_nums,
                    "paragraph_ids": clause.paragraph_ids,
                    "confidence": clause.confidence,
                    "detection_method": clause.detection_method
                } for clause in self.clauses
            ],
            "metadata": self.metadata,
            "processing_errors": self.processing_errors
        }
    
    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save processed document to JSON file"""
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create ProcessedDocument from dictionary"""
        # Create basic document
        doc = cls(
            document_path=data['document_path'],
            document_type=DocumentType(data['document_type']),
            metadata=data.get('metadata', {}),
            processing_errors=data.get('processing_errors', [])
        )
        
        # Add pages
        for page_data in data.get('pages', []):
            page = Page(
                page_num=page_data['page_num'],
                width=page_data.get('width', 0.0),
                height=page_data.get('height', 0.0)
            )
            
            # Add paragraphs
            for para_data in page_data.get('paragraphs', []):
                paragraph = Paragraph(
                    paragraph_id=para_data.get('paragraph_id', ''),
                    is_heading=para_data.get('is_heading', False),
                    heading_level=para_data.get('heading_level', 0),
                    is_numbered=para_data.get('is_numbered', False),
                    number_text=para_data.get('number_text', '')
                )
                
                # Add spans
                for span_data in para_data.get('spans', []):
                    bbox = None
                    if span_data.get('bbox'):
                        bbox_data = span_data['bbox']
                        bbox = BoundingBox(
                            x0=bbox_data['x0'],
                            y0=bbox_data['y0'],
                            x1=bbox_data['x1'],
                            y1=bbox_data['y1'],
                            page=bbox_data.get('page', 0)
                        )
                    
                    span = TextSpan(
                        text=span_data['text'],
                        start_offset=span_data['start_offset'],
                        end_offset=span_data['end_offset'],
                        bbox=bbox,
                        font_size=span_data.get('font_size'),
                        font_name=span_data.get('font_name'),
                        is_bold=span_data.get('is_bold', False),
                        is_italic=span_data.get('is_italic', False),
                        page_num=span_data.get('page_num', 0),
                        confidence=span_data.get('confidence', 1.0)
                    )
                    paragraph.spans.append(span)
                
                page.paragraphs.append(paragraph)
            doc.pages.append(page)
        
        # Add clauses
        for clause_data in data.get('clauses', []):
            clause = Clause(
                clause_id=clause_data['clause_id'],
                text=clause_data['text'],
                clause_type=clause_data.get('clause_type'),
                start_offset=clause_data.get('start_offset', 0),
                end_offset=clause_data.get('end_offset', 0),
                page_nums=clause_data.get('page_nums', []),
                paragraph_ids=clause_data.get('paragraph_ids', []),
                confidence=clause_data.get('confidence', 0.0),
                detection_method=clause_data.get('detection_method', 'heuristic')
            )
            doc.clauses.append(clause)
        
        return doc