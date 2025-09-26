"""
Text Normalizer for ContractSense

Handles cleaning and normalization of extracted text from various sources.
Removes headers/footers, fixes encoding issues, normalizes whitespace, etc.
"""

import re
from typing import List, Set, Dict, Tuple
from pathlib import Path
import unicodedata
import logging

from . import ProcessedDocument, Page, Paragraph, TextSpan, logger


class TextNormalizer:
    """Handles text cleaning and normalization"""
    
    def __init__(self):
        # Common header/footer patterns to remove
        self.header_footer_patterns = [
            r'^\s*Page\s+\d+\s*$',
            r'^\s*\d+\s*$',  # Page numbers
            r'^\s*-\s*\d+\s*-\s*$',  # Centered page numbers
            r'^\s*Source:\s+.*?$',  # CUAD source lines
            r'^\s*THIS EXHIBIT HAS BEEN REDACTED.*?$',  # Confidentiality notices
            r'^\s*CONFIDENTIAL.*?$',
            r'^\s*\[REDACTED\].*?$',
            r'^\s*\*\*\*.*?\*\*\*\s*$',  # Redaction markers
        ]
        
        # Ligature replacements
        self.ligature_map = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '–': '-',  # en dash to hyphen
            '—': '-',  # em dash to hyphen
            '"': '"',  # smart quotes
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        # Common OCR errors to fix
        self.ocr_corrections = {
            r'\b1\b': 'I',  # Common OCR error: 1 instead of I
            r'\brn\b': 'm',  # rn -> m
            r'\bvv\b': 'w',  # vv -> w
            r'\|': 'l',      # | -> l
        }
    
    def normalize(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Normalize all text in the document
        
        Args:
            doc: ProcessedDocument to normalize
            
        Returns:
            ProcessedDocument with normalized text
        """
        logger.info(f"Normalizing text for document: {doc.document_path}")
        
        try:
            # First pass: identify and remove headers/footers
            potential_headers, potential_footers = self._identify_headers_footers(doc)
            
            # Second pass: normalize text in each paragraph
            for page in doc.pages:
                for paragraph in page.paragraphs:
                    self._normalize_paragraph(paragraph, potential_headers, potential_footers)
            
            # Third pass: merge split sentences and fix paragraph boundaries
            self._merge_split_sentences(doc)
            
            # Update metadata
            doc.metadata["normalization_applied"] = True
            doc.metadata["headers_removed"] = len(potential_headers)
            doc.metadata["footers_removed"] = len(potential_footers)
            
        except Exception as e:
            logger.error(f"Error normalizing document: {e}")
            doc.processing_errors.append(f"Text normalization error: {str(e)}")
        
        return doc
    
    def _identify_headers_footers(self, doc: ProcessedDocument) -> Tuple[Set[str], Set[str]]:
        """
        Identify potential headers and footers by finding repeated text patterns
        at the top and bottom of pages
        """
        if len(doc.pages) < 2:
            return set(), set()
        
        # Collect first and last paragraphs from each page
        first_paragraphs = []
        last_paragraphs = []
        
        for page in doc.pages:
            if page.paragraphs:
                first_paragraphs.append(page.paragraphs[0].text.strip())
                last_paragraphs.append(page.paragraphs[-1].text.strip())
        
        # Find repeated patterns in first paragraphs (headers)
        header_candidates = self._find_repeated_patterns(first_paragraphs)
        
        # Find repeated patterns in last paragraphs (footers)
        footer_candidates = self._find_repeated_patterns(last_paragraphs)
        
        # Filter out patterns that are too common or contain important content
        headers = self._filter_header_footer_candidates(header_candidates, is_header=True)
        footers = self._filter_header_footer_candidates(footer_candidates, is_header=False)
        
        return headers, footers
    
    def _find_repeated_patterns(self, text_list: List[str], min_occurrences: int = 2) -> Dict[str, int]:
        """Find text patterns that occur multiple times"""
        pattern_counts = {}
        
        for text in text_list:
            if len(text.strip()) < 5:  # Too short
                continue
            
            # Normalize for pattern matching (remove page numbers, etc.)
            normalized = re.sub(r'\b\d+\b', '[NUMBER]', text)
            
            if normalized in pattern_counts:
                pattern_counts[normalized] += 1
            else:
                pattern_counts[normalized] = 1
        
        # Return patterns that occur at least min_occurrences times
        return {pattern: count for pattern, count in pattern_counts.items() 
                if count >= min_occurrences}
    
    def _filter_header_footer_candidates(self, candidates: Dict[str, int], is_header: bool) -> Set[str]:
        """Filter header/footer candidates to avoid removing important content"""
        filtered = set()
        
        for pattern, count in candidates.items():
            # Skip if pattern contains important legal keywords
            important_keywords = [
                'agreement', 'contract', 'party', 'section', 'clause',
                'whereas', 'therefore', 'hereby', 'shall', 'will'
            ]
            
            pattern_lower = pattern.lower()
            has_important_content = any(keyword in pattern_lower for keyword in important_keywords)
            
            if has_important_content:
                continue
            
            # Check against known header/footer patterns
            is_header_footer = any(re.match(regex, pattern, re.IGNORECASE) 
                                 for regex in self.header_footer_patterns)
            
            if is_header_footer:
                filtered.add(pattern)
        
        return filtered
    
    def _normalize_paragraph(self, paragraph: Paragraph, headers: Set[str], footers: Set[str]) -> None:
        """Normalize text in a single paragraph"""
        
        # Check if this paragraph is a header/footer to remove
        normalized_text = re.sub(r'\b\d+\b', '[NUMBER]', paragraph.text)
        if normalized_text in headers or normalized_text in footers:
            # Mark for removal by clearing spans
            paragraph.spans = []
            return
        
        # Normalize each span
        for span in paragraph.spans:
            span.text = self._normalize_text(span.text)
        
        # Update offsets after normalization
        self._update_span_offsets(paragraph)
    
    def _normalize_text(self, text: str) -> str:
        """Apply various text normalizations"""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Fix ligatures
        for ligature, replacement in self.ligature_map.items():
            text = text.replace(ligature, replacement)
        
        # Fix common OCR errors
        for pattern, replacement in self.ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.;:!?])(\w)', r'\1 \2', text)  # Add space after punctuation
        
        # Handle redaction markers
        text = re.sub(r'\*{3,}', '[REDACTED]', text)
        text = re.sub(r'_{3,}', '[REDACTED]', text)
        
        return text.strip()
    
    def _update_span_offsets(self, paragraph: Paragraph) -> None:
        """Update span offsets after text normalization"""
        current_offset = paragraph.start_offset if paragraph.spans else 0
        
        for span in paragraph.spans:
            span.start_offset = current_offset
            span.end_offset = current_offset + len(span.text)
            current_offset = span.end_offset + 1  # +1 for space between spans
    
    def _merge_split_sentences(self, doc: ProcessedDocument) -> None:
        """
        Merge sentences that were split across paragraph boundaries
        This is common in PDF extraction where line breaks create artificial paragraph splits
        """
        
        for page in doc.pages:
            merged_paragraphs = []
            i = 0
            
            while i < len(page.paragraphs):
                current_para = page.paragraphs[i]
                
                # Skip empty paragraphs
                if not current_para.spans or not current_para.text.strip():
                    i += 1
                    continue
                
                # Check if current paragraph ends mid-sentence and next starts with lowercase
                if i + 1 < len(page.paragraphs):
                    next_para = page.paragraphs[i + 1]
                    
                    if (next_para.spans and 
                        self._should_merge_paragraphs(current_para, next_para)):
                        
                        # Merge paragraphs
                        merged_para = self._merge_paragraphs(current_para, next_para)
                        merged_paragraphs.append(merged_para)
                        i += 2  # Skip the next paragraph since we merged it
                        continue
                
                merged_paragraphs.append(current_para)
                i += 1
            
            page.paragraphs = merged_paragraphs
    
    def _should_merge_paragraphs(self, para1: Paragraph, para2: Paragraph) -> bool:
        """Determine if two paragraphs should be merged"""
        
        text1 = para1.text.strip()
        text2 = para2.text.strip()
        
        if not text1 or not text2:
            return False
        
        # Don't merge if either is a heading
        if para1.is_heading or para2.is_heading:
            return False
        
        # Don't merge if either is numbered differently
        if para1.is_numbered and para2.is_numbered:
            return False
        
        # Merge if first paragraph doesn't end with sentence-ending punctuation
        # and second paragraph starts with lowercase
        ends_mid_sentence = not re.search(r'[.!?]$', text1)
        starts_lowercase = text2[0].islower()
        
        # Also check if the first paragraph is very short (likely a line break)
        is_short_line = len(text1) < 80
        
        return (ends_mid_sentence and starts_lowercase) or is_short_line
    
    def _merge_paragraphs(self, para1: Paragraph, para2: Paragraph) -> Paragraph:
        """Merge two paragraphs into one"""
        
        # Combine spans
        merged_spans = para1.spans + para2.spans
        
        # Update offsets
        for i, span in enumerate(merged_spans):
            if i == len(para1.spans):  # First span from para2
                # Add space between paragraphs
                span.start_offset = merged_spans[i-1].end_offset + 1
                span.end_offset = span.start_offset + len(span.text)
            elif i > len(para1.spans):  # Subsequent spans from para2
                span.start_offset = merged_spans[i-1].end_offset + 1
                span.end_offset = span.start_offset + len(span.text)
        
        # Create merged paragraph
        merged_para = Paragraph(
            spans=merged_spans,
            paragraph_id=para1.paragraph_id,  # Keep first paragraph's ID
            is_heading=para1.is_heading,  # Inherit from first paragraph
            heading_level=para1.heading_level,
            is_numbered=para1.is_numbered,
            number_text=para1.number_text
        )
        
        return merged_para