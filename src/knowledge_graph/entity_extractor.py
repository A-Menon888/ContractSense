"""
Enhanced Entity Extraction for Knowledge Graph

This module extends the existing risk feature extraction capabilities
to provide comprehensive entity and relationship extraction suitable
for populating the Neo4j knowledge graph.

Key Features:
- Party identification and classification
- Contract metadata extraction  
- Enhanced monetary term extraction
- Date and temporal information extraction
- Legal obligation identification
- Relationship extraction between entities
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging

# Import existing modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from risk.utils.feature_extractor import FeatureExtractor, MonetaryAmount, ExtractedFeatures
    from annotation.clause_extractor import ClauseExtractor
except ImportError:
    # Fallback for development
    FeatureExtractor = None
    MonetaryAmount = None
    ExtractedFeatures = None
    ClauseExtractor = None

from .schema import GraphNode, GraphRelationship, NodeType, RelationshipType

logger = logging.getLogger(__name__)

@dataclass
class ExtractedParty:
    """Represents an extracted party/entity"""
    name: str
    party_type: str  # company, individual, government
    aliases: List[str]
    jurisdiction: Optional[str] = None
    confidence: float = 0.0
    
@dataclass 
class ExtractedDate:
    """Represents an extracted date with context"""
    date_value: str  # ISO format
    date_type: str  # effective, expiry, milestone, etc.
    description: str
    is_relative: bool = False
    confidence: float = 0.0

@dataclass
class ExtractedObligation:
    """Represents a legal obligation"""
    description: str
    obligor: str  # who has the obligation
    obligee: str  # who benefits  
    obligation_type: str  # payment, delivery, performance
    enforceability: str = "binding"
    confidence: float = 0.0

@dataclass
class ContractMetadata:
    """Complete contract metadata for graph ingestion"""
    document_id: str
    title: str
    document_type: str
    file_path: str
    parties: List[ExtractedParty]
    dates: List[ExtractedDate]
    monetary_terms: List[Dict[str, Any]]  # MonetaryAmount or dict
    obligations: List[ExtractedObligation]
    governing_law: Optional[str] = None
    risk_level: str = "MEDIUM"
    processing_timestamp: str = None

class EnhancedEntityExtractor:
    """Enhanced entity extraction for knowledge graph population"""
    
    def __init__(self):
        """Initialize the enhanced entity extractor"""
        
        # Initialize base feature extractor if available
        self.feature_extractor = FeatureExtractor() if FeatureExtractor else None
        
        # Party identification patterns
        self.company_patterns = [
            r'\b([A-Z][A-Za-z\s&,.-]+?)\s+(Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company|LP|LLP|PC)\b',
            r'\b([A-Z][A-Za-z\s&,.-]+?)\s+(GmbH|AG|SA|BV|AB|AS|Oy|SpA|SRL|SARL)\b',
            r'\b([A-Z]{2,})\s+(?:Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?)\b'
        ]
        
        self.person_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # First Last [Middle]
            r'\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        # Date extraction patterns
        self.date_patterns = [
            (r'effective\s+(?:date\s+)?(?:as\s+of\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'effective'),
            (r'effective\s+(?:date\s+)?(?:as\s+of\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'effective'),
            (r'expir(?:es?|ation)\s+(?:date\s+)?(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'expiry'),
            (r'expir(?:es?|ation)\s+(?:date\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'expiry'),
            (r'term(?:ination)?\s+(?:date\s+)?(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'termination'),
            (r'term(?:ination)?\s+(?:date\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'termination'),
            (r'commence(?:s|ment)?\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'commencement'),
            (r'commence(?:s|ment)?\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'commencement'),
            (r'(?:entered\s+into\s+)?(?:on\s+|dated\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'general'),
            (r'(?:on\s+|dated\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'general')
        ]
        
        # Obligation extraction patterns
        self.obligation_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+(?:shall|will|must|agrees?\s+to|undertakes?\s+to)\s+([\w\s,.-]+?)(?:\.|;|$)', 'binding'),
            (r'(\w+(?:\s+\w+)*)\s+(?:may|can|is permitted to)\s+([\w\s,.-]+?)(?:\.|;|$)', 'optional'),
            (r'in\s+the\s+event\s+that\s+(\w+(?:\s+\w+)*)\s+([\w\s,.-]+?)(?:\.|;|$)', 'conditional')
        ]
        
        # Contract type classification
        self.contract_type_keywords = {
            'MSA': ['master service agreement', 'msa', 'master agreement'],
            'SOW': ['statement of work', 'sow', 'work order'],
            'NDA': ['non-disclosure', 'confidentiality agreement', 'nda'],
            'LICENSE': ['license agreement', 'licensing', 'software license'],
            'EMPLOYMENT': ['employment agreement', 'employment contract'],
            'PURCHASE': ['purchase agreement', 'purchase order', 'supply agreement'],
            'LEASE': ['lease agreement', 'rental agreement', 'tenancy'],
            'PARTNERSHIP': ['partnership agreement', 'joint venture'],
            'CONSULTING': ['consulting agreement', 'consulting services']
        }
        
        # Jurisdiction patterns
        self.jurisdiction_patterns = [
            r'governed\s+by\s+(?:the\s+)?laws?\s+of\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+and|\.|,|$)',
            r'jurisdiction\s+of\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+courts?|\.|,|$)',
            r'laws?\s+of\s+(?:the\s+)?State\s+of\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)',
            r'([A-Z][a-zA-Z\s]+?)\s+law\s+(?:shall\s+)?govern'
        ]
    
    def extract_comprehensive_metadata(self, document_text: str, document_id: str, 
                                     file_path: str = "", clauses: List[Dict] = None) -> ContractMetadata:
        """
        Extract comprehensive contract metadata for graph ingestion
        
        Args:
            document_text: Full contract text
            document_id: Unique document identifier
            file_path: Path to original document file
            clauses: List of extracted clauses (optional)
            
        Returns:
            ContractMetadata with all extracted information
        """
        logger.info(f"Extracting comprehensive metadata for document {document_id}")
        
        # Extract parties
        parties = self._extract_parties(document_text)
        logger.debug(f"Extracted {len(parties)} parties")
        
        # Extract dates
        dates = self._extract_dates(document_text)
        logger.debug(f"Extracted {len(dates)} dates")
        
        # Extract monetary terms
        monetary_terms = self._extract_monetary_terms(document_text)
        logger.debug(f"Extracted {len(monetary_terms)} monetary terms")
        
        # Extract obligations
        obligations = self._extract_obligations(document_text)
        logger.debug(f"Extracted {len(obligations)} obligations")
        
        # Classify contract type
        contract_type = self._classify_contract_type(document_text, file_path)
        
        # Extract governing law
        governing_law = self._extract_governing_law(document_text)
        
        # Generate title
        title = self._generate_title(file_path, parties, contract_type)
        
        # Determine risk level (if risk module available)
        risk_level = self._assess_document_risk_level(clauses) if clauses else "MEDIUM"
        
        return ContractMetadata(
            document_id=document_id,
            title=title,
            document_type=contract_type,
            file_path=file_path,
            parties=parties,
            dates=dates,
            monetary_terms=monetary_terms,
            obligations=obligations,
            governing_law=governing_law,
            risk_level=risk_level,
            processing_timestamp=datetime.now().isoformat()
        )
    
    def _extract_parties(self, text: str) -> List[ExtractedParty]:
        """Extract party information from contract text"""
        parties = []
        found_names = set()
        
        # Extract companies
        for pattern in self.company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                full_name = match.group(0).strip()
                company_name = match.group(1).strip()
                
                if company_name not in found_names and len(company_name) > 2:
                    # Calculate confidence based on context
                    confidence = self._calculate_party_confidence(full_name, text)
                    
                    party = ExtractedParty(
                        name=company_name,
                        party_type="company",
                        aliases=[full_name] if full_name != company_name else [],
                        confidence=confidence
                    )
                    parties.append(party)
                    found_names.add(company_name)
        
        # Extract individuals (with lower confidence)
        for pattern in self.person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1).strip()
                
                if person_name not in found_names and len(person_name) > 5:
                    confidence = self._calculate_party_confidence(person_name, text) * 0.7  # Lower confidence for persons
                    
                    party = ExtractedParty(
                        name=person_name,
                        party_type="individual", 
                        aliases=[],
                        confidence=confidence
                    )
                    parties.append(party)
                    found_names.add(person_name)
        
        # Sort by confidence and take top candidates
        parties.sort(key=lambda p: p.confidence, reverse=True)
        return parties[:10]  # Limit to top 10 parties
    
    def _calculate_party_confidence(self, party_name: str, text: str) -> float:
        """Calculate confidence score for extracted party"""
        confidence = 0.5  # Base confidence
        
        # Count occurrences
        occurrences = len(re.findall(re.escape(party_name), text, re.IGNORECASE))
        confidence += min(occurrences * 0.1, 0.3)  # Up to +0.3 for multiple occurrences
        
        # Check for party-related keywords nearby
        party_keywords = ['party', 'client', 'vendor', 'contractor', 'licensee', 'licensor', 'buyer', 'seller']
        for keyword in party_keywords:
            pattern = rf'\b{re.escape(party_name)}\b.{{0,50}}\b{keyword}\b|\b{keyword}\b.{{0,50}}\b{re.escape(party_name)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                confidence += 0.2
                break
        
        # Check position in document (parties often mentioned early)
        party_position = text.lower().find(party_name.lower())
        if party_position != -1 and party_position < len(text) * 0.2:  # First 20% of document
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_dates(self, text: str) -> List[ExtractedDate]:
        """Extract date information from contract text"""
        dates = []
        found_dates = set()
        
        for pattern, date_type in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                
                if date_str not in found_dates:
                    # Normalize date format
                    normalized_date = self._normalize_date(date_str)
                    if normalized_date:
                        description = f"{date_type.title()} date: {date_str}"
                        
                        date_obj = ExtractedDate(
                            date_value=normalized_date,
                            date_type=date_type,
                            description=description,
                            confidence=0.8 if date_type != 'general' else 0.6
                        )
                        dates.append(date_obj)
                        found_dates.add(date_str)
        
        return dates
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date string to ISO format"""
        # Simple date normalization - extend as needed
        try:
            # Handle common formats
            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%y', '%m-%d-%y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.date().isoformat()
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _extract_monetary_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary terms using existing feature extractor or patterns"""
        if self.feature_extractor:
            # Use existing feature extractor
            features = self.feature_extractor.extract_features(text)
            return features.monetary_amounts
        else:
            # Fallback pattern-based extraction
            return self._extract_monetary_terms_fallback(text)
    
    def _extract_monetary_terms_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback monetary extraction using patterns"""
        amounts = []
        
        # Common monetary patterns
        patterns = [
            r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+dollars?',
            r'USD\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'EUR\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'GBP\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1)
                try:
                    amount_value = float(amount_str.replace(',', ''))
                    # Create MonetaryAmount-like object
                    amounts.append({
                        'amount': amount_value,
                        'currency': 'USD',  # Default currency
                        'term_type': 'general',  # Add missing field
                        'confidence': 0.7
                    })
                except ValueError:
                    continue
        
        return amounts[:20]  # Limit to top 20 amounts
    
    def _extract_obligations(self, text: str) -> List[ExtractedObligation]:
        """Extract legal obligations from text"""
        obligations = []
        
        for pattern, enforceability in self.obligation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                obligor = match.group(1).strip()
                obligation_text = match.group(2).strip()
                
                if len(obligation_text) > 10 and len(obligor) > 2:
                    # Classify obligation type
                    obligation_type = self._classify_obligation_type(obligation_text)
                    
                    obligation = ExtractedObligation(
                        description=obligation_text,
                        obligor=obligor,
                        obligee="counterparty",  # Default - could be enhanced
                        obligation_type=obligation_type,
                        enforceability=enforceability,
                        confidence=0.7
                    )
                    obligations.append(obligation)
        
        return obligations[:15]  # Limit to top 15 obligations
    
    def _classify_obligation_type(self, obligation_text: str) -> str:
        """Classify the type of obligation"""
        text_lower = obligation_text.lower()
        
        if any(word in text_lower for word in ['pay', 'payment', 'remit', 'compensate']):
            return 'payment'
        elif any(word in text_lower for word in ['deliver', 'provide', 'supply', 'furnish']):
            return 'delivery'
        elif any(word in text_lower for word in ['perform', 'execute', 'complete', 'fulfill']):
            return 'performance'
        elif any(word in text_lower for word in ['maintain', 'preserve', 'keep', 'retain']):
            return 'maintenance'
        elif any(word in text_lower for word in ['comply', 'adhere', 'conform', 'follow']):
            return 'compliance'
        else:
            return 'general'
    
    def _classify_contract_type(self, text: str, file_path: str) -> str:
        """Classify the type of contract"""
        text_lower = text.lower()
        file_lower = file_path.lower()
        
        # Check filename first
        for contract_type, keywords in self.contract_type_keywords.items():
            if any(keyword.lower() in file_lower for keyword in keywords):
                return contract_type
        
        # Check document content
        for contract_type, keywords in self.contract_type_keywords.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                return contract_type
        
        return 'GENERAL'
    
    def _extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law/jurisdiction information"""
        for pattern in self.jurisdiction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                jurisdiction = match.group(1).strip()
                if len(jurisdiction) > 2:
                    return jurisdiction
        
        return None
    
    def _generate_title(self, file_path: str, parties: List[ExtractedParty], contract_type: str) -> str:
        """Generate a descriptive title for the contract"""
        filename = Path(file_path).stem if file_path else "Contract"
        
        # Use filename if descriptive
        if len(filename) > 10 and not filename.startswith('contract_'):
            return filename.replace('_', ' ').replace('-', ' ').title()
        
        # Generate from parties and type
        if len(parties) >= 2:
            party1 = parties[0].name
            party2 = parties[1].name
            return f"{contract_type} Agreement between {party1} and {party2}"
        elif len(parties) == 1:
            return f"{contract_type} Agreement - {parties[0].name}"
        else:
            return f"{contract_type} Agreement"
    
    def _assess_document_risk_level(self, clauses: List[Dict]) -> str:
        """Assess overall document risk level from clauses"""
        if not clauses:
            return "MEDIUM"
        
        # Count high-risk clause types
        high_risk_types = {'limitation_of_liability', 'indemnification', 'liquidated_damages', 
                          'termination', 'intellectual_property'}
        high_risk_count = 0
        
        for clause in clauses:
            clause_type = clause.get('type', '').lower()
            if clause_type in high_risk_types:
                high_risk_count += 1
        
        # Assess based on high-risk clause density
        if high_risk_count >= 3 or high_risk_count / len(clauses) > 0.3:
            return "HIGH"
        elif high_risk_count >= 1 or len(clauses) > 15:
            return "MEDIUM"
        else:
            return "LOW"