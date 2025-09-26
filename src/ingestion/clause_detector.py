"""
Clause Detector for ContractSense

Uses heuristics, regex patterns, and simple rules to detect clause boundaries
and classify clause types in legal documents.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging

from . import ProcessedDocument, Clause, Paragraph, logger


@dataclass
class ClausePattern:
    """Represents a pattern for detecting specific clause types"""
    clause_type: str
    patterns: List[str]  # Regex patterns
    keywords: List[str]  # Important keywords that boost confidence
    negative_keywords: List[str]  # Keywords that decrease confidence
    confidence_boost: float = 0.1  # How much to boost confidence when pattern matches


class ClauseDetector:
    """Detects clause boundaries and types using rule-based heuristics"""
    
    def __init__(self):
        self.clause_patterns = self._initialize_clause_patterns()
        
        # Section/clause boundary patterns
        self.boundary_patterns = [
            r'^\d+\.\s+[A-Z][^.]*\.',  # 1. SECTION TITLE.
            r'^\d+\.\d+\s+[A-Z]',      # 1.1 Subsection
            r'^[A-Z]{2,}[:\.]',        # ALL CAPS SECTION:
            r'^ARTICLE\s+[IVX]+',      # ARTICLE I, II, etc.
            r'^SECTION\s+\d+',         # SECTION 1, 2, etc.
            r'^\([a-z]\)\s*[A-Z]',     # (a) Subsection
            r'^\([0-9]+\)\s*[A-Z]',    # (1) Subsection
        ]
    
    def detect_clauses(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Detect clauses in the document and add them to the document
        
        Args:
            doc: ProcessedDocument to analyze
            
        Returns:
            ProcessedDocument with detected clauses
        """
        logger.info(f"Detecting clauses in document: {doc.document_path}")
        
        try:
            # Step 1: Detect section boundaries
            sections = self._detect_sections(doc)
            
            # Step 2: Classify each section by clause type
            clauses = []
            for section in sections:
                clause_type, confidence = self._classify_section(section)
                
                clause = Clause(
                    clause_id=f"clause_{len(clauses)}",
                    text=section["text"],
                    clause_type=clause_type,
                    start_offset=section["start_offset"],
                    end_offset=section["end_offset"],
                    page_nums=section["page_nums"],
                    paragraph_ids=section["paragraph_ids"],
                    confidence=confidence,
                    detection_method="heuristic"
                )
                clauses.append(clause)
            
            # Step 3: Post-process and validate clauses
            clauses = self._post_process_clauses(clauses, doc)
            
            doc.clauses = clauses
            doc.metadata["total_clauses_detected"] = len(clauses)
            doc.metadata["clause_detection_applied"] = True
            
        except Exception as e:
            logger.error(f"Error detecting clauses: {e}")
            doc.processing_errors.append(f"Clause detection error: {str(e)}")
        
        return doc
    
    def _initialize_clause_patterns(self) -> List[ClausePattern]:
        """Initialize patterns for detecting specific clause types"""
        
        patterns = [
            # Indemnification/Indemnity
            ClausePattern(
                clause_type="indemnification",
                patterns=[
                    r'indemnif\w*',
                    r'hold\s+harmless',
                    r'defend.*against',
                ],
                keywords=["indemnify", "indemnification", "hold harmless", "defend"],
                negative_keywords=["limitation", "except"]
            ),
            
            # Limitation of Liability
            ClausePattern(
                clause_type="limitation_of_liability",
                patterns=[
                    r'limitation\s+of\s+liability',
                    r'limit.*liability',
                    r'maximum\s+liability',
                    r'aggregate\s+liability',
                ],
                keywords=["limitation", "liability", "damages", "limit", "maximum"],
                negative_keywords=["indemnification", "unlimited"]
            ),
            
            # Termination
            ClausePattern(
                clause_type="termination",
                patterns=[
                    r'terminat\w*',
                    r'end\s+this\s+agreement',
                    r'expire\w*',
                ],
                keywords=["terminate", "termination", "expire", "end", "dissolution"],
                negative_keywords=["employment"]
            ),
            
            # Confidentiality/Non-Disclosure
            ClausePattern(
                clause_type="confidentiality",
                patterns=[
                    r'confidential\w*',
                    r'non.disclosure',
                    r'proprietary\s+information',
                    r'trade\s+secret',
                ],
                keywords=["confidential", "non-disclosure", "proprietary", "trade secret"],
                negative_keywords=["public", "disclosed"]
            ),
            
            # Governing Law
            ClausePattern(
                clause_type="governing_law",
                patterns=[
                    r'governing\s+law',
                    r'governed\s+by',
                    r'laws?\s+of\s+\w+',
                    r'jurisdiction',
                ],
                keywords=["governing", "governed", "laws", "jurisdiction", "court"],
                negative_keywords=[]
            ),
            
            # Payment/Financial
            ClausePattern(
                clause_type="payment",
                patterns=[
                    r'payment\w*',
                    r'fee\w*',
                    r'compensation',
                    r'\$[\d,]+',
                    r'invoice\w*',
                ],
                keywords=["payment", "fee", "compensation", "invoice", "salary"],
                negative_keywords=[]
            ),
            
            # Assignment
            ClausePattern(
                clause_type="assignment",
                patterns=[
                    r'assign\w*',
                    r'transfer\w*',
                    r'delegate\w*',
                ],
                keywords=["assign", "assignment", "transfer", "delegate"],
                negative_keywords=["employment", "work"]
            ),
            
            # Force Majeure
            ClausePattern(
                clause_type="force_majeure",
                patterns=[
                    r'force\s+majeure',
                    r'acts?\s+of\s+god',
                    r'unforeseeable\s+circumstances',
                ],
                keywords=["force majeure", "act of god", "unforeseeable", "beyond control"],
                negative_keywords=[]
            ),
            
            # Intellectual Property
            ClausePattern(
                clause_type="intellectual_property",
                patterns=[
                    r'intellectual\s+property',
                    r'copyright\w*',
                    r'trademark\w*',
                    r'patent\w*',
                    r'trade\s+mark',
                ],
                keywords=["intellectual property", "copyright", "trademark", "patent", "IP"],
                negative_keywords=[]
            ),
            
            # Non-Compete
            ClausePattern(
                clause_type="non_compete",
                patterns=[
                    r'non.compet\w*',
                    r'compete\w*.*restrict\w*',
                    r'restraint.*trade',
                ],
                keywords=["non-compete", "compete", "restriction", "restrain"],
                negative_keywords=[]
            ),
            
            # Warranty
            ClausePattern(
                clause_type="warranty",
                patterns=[
                    r'warrant\w*',
                    r'represent\w*.*warrant\w*',
                    r'guarantee\w*',
                ],
                keywords=["warranty", "warrant", "represent", "guarantee"],
                negative_keywords=["disclaim", "without warranty"]
            ),
            
            # Dispute Resolution
            ClausePattern(
                clause_type="dispute_resolution",
                patterns=[
                    r'dispute\w*.*resolution',
                    r'arbitrat\w*',
                    r'mediat\w*',
                    r'litigation',
                ],
                keywords=["dispute", "arbitration", "mediation", "resolution"],
                negative_keywords=[]
            )
        ]
        
        return patterns
    
    def _detect_sections(self, doc: ProcessedDocument) -> List[Dict]:
        """
        Detect section boundaries in the document
        
        Returns:
            List of sections, each containing text, offsets, and metadata
        """
        sections = []
        current_section = {
            "text": "",
            "start_offset": 0,
            "end_offset": 0,
            "page_nums": [],
            "paragraph_ids": []
        }
        
        for page in doc.pages:
            for paragraph in page.paragraphs:
                if not paragraph.spans or not paragraph.text.strip():
                    continue
                
                text = paragraph.text.strip()
                
                # Check if this paragraph starts a new section
                is_section_boundary = (
                    paragraph.is_heading or
                    any(re.match(pattern, text, re.IGNORECASE) 
                        for pattern in self.boundary_patterns)
                )
                
                if is_section_boundary and current_section["text"]:
                    # Close current section and start new one
                    sections.append(current_section.copy())
                    current_section = {
                        "text": text,
                        "start_offset": paragraph.start_offset,
                        "end_offset": paragraph.end_offset,
                        "page_nums": [page.page_num],
                        "paragraph_ids": [paragraph.paragraph_id]
                    }
                else:
                    # Add to current section
                    if current_section["text"]:
                        current_section["text"] += " " + text
                    else:
                        current_section["text"] = text
                        current_section["start_offset"] = paragraph.start_offset
                    
                    current_section["end_offset"] = paragraph.end_offset
                    if page.page_num not in current_section["page_nums"]:
                        current_section["page_nums"].append(page.page_num)
                    current_section["paragraph_ids"].append(paragraph.paragraph_id)
        
        # Add the final section
        if current_section["text"]:
            sections.append(current_section)
        
        # Filter out very short sections (likely not real clauses)
        sections = [s for s in sections if len(s["text"].split()) > 10]
        
        return sections
    
    def _classify_section(self, section: Dict) -> Tuple[Optional[str], float]:
        """
        Classify a section by clause type
        
        Returns:
            Tuple of (clause_type, confidence_score)
        """
        text = section["text"].lower()
        best_match = None
        best_confidence = 0.0
        
        for pattern in self.clause_patterns:
            confidence = 0.0
            
            # Check regex patterns
            pattern_matches = sum(1 for p in pattern.patterns 
                                if re.search(p, text, re.IGNORECASE))
            if pattern_matches > 0:
                confidence += pattern_matches * pattern.confidence_boost
            
            # Check positive keywords
            keyword_matches = sum(1 for keyword in pattern.keywords 
                                if keyword.lower() in text)
            confidence += keyword_matches * 0.05
            
            # Penalize for negative keywords
            negative_matches = sum(1 for keyword in pattern.negative_keywords 
                                 if keyword.lower() in text)
            confidence -= negative_matches * 0.03
            
            # Boost confidence if section title matches
            if any(keyword.lower() in section["text"][:100].lower() 
                   for keyword in pattern.keywords):
                confidence += 0.1
            
            # Update best match
            if confidence > best_confidence and confidence > 0.1:
                best_confidence = confidence
                best_match = pattern.clause_type
        
        # Cap confidence at 0.9 for heuristic methods
        best_confidence = min(best_confidence, 0.9)
        
        return best_match, best_confidence
    
    def _post_process_clauses(self, clauses: List[Clause], doc: ProcessedDocument) -> List[Clause]:
        """
        Post-process detected clauses to improve quality
        """
        # Remove very short clauses
        clauses = [c for c in clauses if len(c.text.split()) > 5]
        
        # Merge adjacent clauses of the same type
        merged_clauses = []
        i = 0
        while i < len(clauses):
            current = clauses[i]
            
            # Look ahead for clauses of the same type
            if (i + 1 < len(clauses) and 
                clauses[i + 1].clause_type == current.clause_type and
                clauses[i + 1].start_offset - current.end_offset < 100):  # Close proximity
                
                # Merge clauses
                next_clause = clauses[i + 1]
                merged_clause = Clause(
                    clause_id=current.clause_id,
                    text=current.text + " " + next_clause.text,
                    clause_type=current.clause_type,
                    start_offset=current.start_offset,
                    end_offset=next_clause.end_offset,
                    page_nums=list(set(current.page_nums + next_clause.page_nums)),
                    paragraph_ids=current.paragraph_ids + next_clause.paragraph_ids,
                    confidence=max(current.confidence, next_clause.confidence),
                    detection_method=current.detection_method
                )
                merged_clauses.append(merged_clause)
                i += 2  # Skip next clause
            else:
                merged_clauses.append(current)
                i += 1
        
        # Sort by start offset
        merged_clauses.sort(key=lambda x: x.start_offset)
        
        # Update clause IDs
        for i, clause in enumerate(merged_clauses):
            clause.clause_id = f"clause_{i:03d}"
        
        return merged_clauses