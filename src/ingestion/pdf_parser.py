"""
PDF Parser for ContractSense

Handles both selectable and scanned PDFs using PyMuPDF and pdfminer as fallback.
For scanned PDFs, uses OCR (Tesseract) to extract text.
"""

import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LTPage
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
import tempfile
import io
from typing import List, Optional, Tuple
from pathlib import Path
import logging
import re

from . import (
    TextSpan, BoundingBox, Paragraph, Page, ProcessedDocument, 
    DocumentType, logger
)

# Configure Tesseract path for Windows (adjust as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class PDFParser:
    """Main PDF parser supporting selectable and scanned PDFs"""
    
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled and TESSERACT_AVAILABLE
        self.text_confidence_threshold = 0.1  # Min confidence for text detection
        
        if ocr_enabled and not TESSERACT_AVAILABLE:
            logger.warning("OCR requested but Tesseract not available. OCR disabled.")
    
    def parse(self, pdf_path: Path) -> ProcessedDocument:
        """
        Parse a PDF document and return structured data
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted text and layout information
        """
        logger.info(f"Parsing PDF: {pdf_path}")
        
        doc = ProcessedDocument(
            document_path=str(pdf_path),
            document_type=DocumentType.PDF
        )
        
        try:
            # First try PyMuPDF for fast extraction
            pages = self._parse_with_pymupdf(pdf_path)
            
            # Check if we got meaningful text
            total_text = "".join(page.text for page in pages).strip()
            
            if not total_text or self._is_likely_scanned(pages):
                logger.info("PDF appears to be scanned or has poor text extraction. Trying OCR...")
                if self.ocr_enabled:
                    pages = self._parse_with_ocr(pdf_path)
                else:
                    logger.warning("OCR disabled, but PDF appears scanned")
            
            doc.pages = pages
            doc.metadata["total_pages"] = len(pages)
            doc.metadata["total_characters"] = len(doc.full_text)
            doc.metadata["extraction_method"] = "pymupdf" if total_text else "ocr"
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            doc.processing_errors.append(f"PDF parsing error: {str(e)}")
        
        return doc
    
    def _parse_with_pymupdf(self, pdf_path: Path) -> List[Page]:
        """Parse PDF using PyMuPDF for selectable text"""
        pages = []
        
        with fitz.open(str(pdf_path)) as pdf_doc:
            for page_num in range(len(pdf_doc)):
                fitz_page = pdf_doc[page_num]
                
                # Get page dimensions
                rect = fitz_page.rect
                page = Page(
                    page_num=page_num + 1,  # 1-indexed
                    width=float(rect.width),
                    height=float(rect.height)
                )
                
                # Extract text with detailed information
                text_dict = fitz_page.get_text("dict")
                
                paragraph_id = 0
                char_offset = 0
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:  # Skip image blocks
                        continue
                    
                    paragraph_spans = []
                    paragraph_text = ""
                    
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            # Extract formatting information
                            font_size = span.get("size", 0)
                            font_name = span.get("font", "")
                            flags = span.get("flags", 0)
                            
                            # Determine bold/italic from flags
                            is_bold = bool(flags & 2**4)  # Bold flag
                            is_italic = bool(flags & 2**1)  # Italic flag
                            
                            # Create bounding box
                            bbox_coords = span.get("bbox", [0, 0, 0, 0])
                            bbox = BoundingBox(
                                x0=bbox_coords[0],
                                y0=bbox_coords[1],
                                x1=bbox_coords[2],
                                y1=bbox_coords[3],
                                page=page_num + 1
                            )
                            
                            # Create text span
                            text_span = TextSpan(
                                text=text,
                                start_offset=char_offset,
                                end_offset=char_offset + len(text),
                                bbox=bbox,
                                font_size=font_size,
                                font_name=font_name,
                                is_bold=is_bold,
                                is_italic=is_italic,
                                page_num=page_num + 1,
                                confidence=1.0
                            )
                            
                            paragraph_spans.append(text_span)
                            paragraph_text += text + " "
                            char_offset += len(text) + 1  # +1 for space
                    
                    if paragraph_spans:
                        # Analyze paragraph structure
                        paragraph = Paragraph(
                            spans=paragraph_spans,
                            paragraph_id=f"page_{page_num + 1}_para_{paragraph_id}"
                        )
                        
                        # Detect if this looks like a heading or numbered section
                        self._analyze_paragraph_structure(paragraph)
                        
                        page.paragraphs.append(paragraph)
                        paragraph_id += 1
                
                pages.append(page)
        
        return pages
    
    def _parse_with_ocr(self, pdf_path: Path) -> List[Page]:
        """Parse PDF using OCR for scanned documents"""
        if not TESSERACT_AVAILABLE:
            logger.error("OCR requested but Tesseract not available")
            return []
            
        pages = []
        
        with fitz.open(str(pdf_path)) as pdf_doc:
            for page_num in range(len(pdf_doc)):
                fitz_page = pdf_doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = fitz_page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # OCR the image
                image = Image.open(io.BytesIO(img_data))
                
                # Get OCR data with bounding boxes
                ocr_data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config='--oem 3 --psm 6'  # Uniform block of text
                )
                
                # Create page
                page = Page(
                    page_num=page_num + 1,
                    width=float(fitz_page.rect.width),
                    height=float(fitz_page.rect.height)
                )
                
                # Process OCR results
                char_offset = 0
                current_paragraph_spans = []
                paragraph_id = 0
                
                for i, text in enumerate(ocr_data['text']):
                    text = text.strip()
                    if not text:
                        continue
                    
                    confidence = float(ocr_data['conf'][i]) / 100.0
                    if confidence < self.text_confidence_threshold:
                        continue
                    
                    # Scale coordinates back from 2x zoom
                    left = float(ocr_data['left'][i]) / 2.0
                    top = float(ocr_data['top'][i]) / 2.0
                    width = float(ocr_data['width'][i]) / 2.0
                    height = float(ocr_data['height'][i]) / 2.0
                    
                    bbox = BoundingBox(
                        x0=left,
                        y0=top,
                        x1=left + width,
                        y1=top + height,
                        page=page_num + 1
                    )
                    
                    text_span = TextSpan(
                        text=text,
                        start_offset=char_offset,
                        end_offset=char_offset + len(text),
                        bbox=bbox,
                        page_num=page_num + 1,
                        confidence=confidence
                    )
                    
                    # Simple paragraph detection based on vertical gaps
                    if (current_paragraph_spans and 
                        abs(bbox.y0 - current_paragraph_spans[-1].bbox.y1) > 10):
                        # New paragraph
                        if current_paragraph_spans:
                            paragraph = Paragraph(
                                spans=current_paragraph_spans,
                                paragraph_id=f"page_{page_num + 1}_para_{paragraph_id}"
                            )
                            self._analyze_paragraph_structure(paragraph)
                            page.paragraphs.append(paragraph)
                            paragraph_id += 1
                        
                        current_paragraph_spans = [text_span]
                    else:
                        current_paragraph_spans.append(text_span)
                    
                    char_offset += len(text) + 1
                
                # Add final paragraph
                if current_paragraph_spans:
                    paragraph = Paragraph(
                        spans=current_paragraph_spans,
                        paragraph_id=f"page_{page_num + 1}_para_{paragraph_id}"
                    )
                    self._analyze_paragraph_structure(paragraph)
                    page.paragraphs.append(paragraph)
                
                pages.append(page)
        
        return pages
    
    def _is_likely_scanned(self, pages: List[Page]) -> bool:
        """
        Determine if the PDF is likely scanned based on text extraction quality
        """
        if not pages:
            return True
        
        total_chars = sum(len(page.text) for page in pages)
        if total_chars < 100:  # Very little text extracted
            return True
        
        # Check for signs of poor extraction (lots of single characters, weird spacing)
        full_text = " ".join(page.text for page in pages)
        words = full_text.split()
        
        if len(words) == 0:
            return True
        
        # If more than 30% of words are single characters, likely scanned
        single_chars = sum(1 for word in words if len(word) == 1)
        single_char_ratio = single_chars / len(words)
        
        return single_char_ratio > 0.3
    
    def _analyze_paragraph_structure(self, paragraph: Paragraph) -> None:
        """
        Analyze paragraph to detect headings, numbering, etc.
        """
        if not paragraph.spans:
            return
        
        text = paragraph.text.strip()
        
        # Check for numbering patterns
        numbering_patterns = [
            r'^\d+\.\s*',  # 1. 2. 3.
            r'^\d+\.\d+\s*',  # 1.1 1.2
            r'^\([a-z]\)\s*',  # (a) (b) (c)
            r'^\([0-9]+\)\s*',  # (1) (2) (3)
            r'^[A-Z]\.\s*',  # A. B. C.
            r'^[IVX]+\.\s*',  # I. II. III. (Roman numerals)
        ]
        
        for pattern in numbering_patterns:
            match = re.match(pattern, text)
            if match:
                paragraph.is_numbered = True
                paragraph.number_text = match.group(0).strip()
                break
        
        # Check for heading characteristics
        if paragraph.spans:
            first_span = paragraph.spans[0]
            
            # Heuristics for heading detection
            is_heading = False
            heading_level = 0
            
            # Font size based heading detection
            if first_span.font_size and first_span.font_size > 12:
                is_heading = True
                if first_span.font_size > 16:
                    heading_level = 1
                elif first_span.font_size > 14:
                    heading_level = 2
                else:
                    heading_level = 3
            
            # Bold text might be a heading
            elif first_span.is_bold and len(text) < 100:
                is_heading = True
                heading_level = 3
            
            # All caps short text might be heading
            elif text.isupper() and len(text) < 80:
                is_heading = True
                heading_level = 2
            
            # Numbered sections are often headings
            elif paragraph.is_numbered and len(text) < 150:
                is_heading = True
                heading_level = 3 if paragraph.number_text.count('.') > 0 else 2
            
            paragraph.is_heading = is_heading
            paragraph.heading_level = heading_level