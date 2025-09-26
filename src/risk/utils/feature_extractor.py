"""
Risk Feature Extraction Utilities

Extracts features from contract clauses for risk classification:
- Monetary amounts and financial figures
- Legal entity and party role detection
- Jurisdiction and governing law extraction
- Contract metadata integration
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MonetaryAmount:
    """Extracted monetary amount with context"""
    amount: float
    currency: str
    context: str  # surrounding text
    confidence: float

@dataclass  
class ExtractedFeatures:
    """Complete feature extraction result"""
    monetary_amounts: List[MonetaryAmount]
    party_role: Optional[str]
    governing_law: Optional[str] 
    jurisdiction: Optional[str]
    time_periods: List[str]
    legal_entities: List[str]
    risk_keywords: List[str]

class MonetaryExtractor:
    """Extracts monetary amounts from contract text"""
    
    # Monetary amount patterns
    AMOUNT_PATTERNS = [
        # Standard formats: $1,000, $1,000.00, $1000
        r'\$\s*(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?',
        # Written amounts: one million dollars, 1 million USD
        r'(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:hundred|thousand|million|billion)\s+(?:dollars?|USD|usd)',
        # Currency codes: USD 1,000, EUR 500
        r'(?:USD|EUR|GBP|CAD|AUD|JPY|CHF)\s*(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?',
        # Alternative formats: 1,000 dollars, 500.00 USD
        r'(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?\s+(?:dollars?|USD|EUR|GBP|CAD)'
    ]
    
    # Written number mappings
    WRITTEN_NUMBERS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000
    }
    
    def extract_monetary_amounts(self, text: str) -> List[MonetaryAmount]:
        """Extract all monetary amounts from text"""
        amounts = []
        text_lower = text.lower()
        
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                amount_text = match.group()
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()
                
                # Parse amount and currency
                parsed_amount, currency, confidence = self._parse_amount_text(amount_text)
                
                if parsed_amount > 0:
                    amounts.append(MonetaryAmount(
                        amount=parsed_amount,
                        currency=currency,
                        context=context,
                        confidence=confidence
                    ))
        
        # Remove duplicates and sort by amount
        amounts = self._deduplicate_amounts(amounts)
        return sorted(amounts, key=lambda x: x.amount, reverse=True)
    
    def _parse_amount_text(self, amount_text: str) -> Tuple[float, str, float]:
        """Parse amount text into numerical value and currency"""
        amount_text = amount_text.strip()
        currency = "USD"  # default
        confidence = 0.8
        
        # Extract currency
        if re.search(r'USD|usd', amount_text):
            currency = "USD"
        elif re.search(r'EUR|eur', amount_text):
            currency = "EUR"
        elif re.search(r'GBP|gbp', amount_text):
            currency = "GBP"
        elif amount_text.startswith('$'):
            currency = "USD"
        
        # Remove currency symbols and clean text
        clean_text = re.sub(r'[^\d.,\w\s]', '', amount_text)
        clean_text = re.sub(r'USD|EUR|GBP|CAD|dollars?', '', clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()
        
        try:
            # Handle written numbers
            if any(word in clean_text.lower() for word in self.WRITTEN_NUMBERS.keys()):
                amount = self._parse_written_amount(clean_text)
            else:
                # Handle numerical amounts
                # Remove commas and parse
                clean_text = clean_text.replace(',', '')
                amount = float(clean_text)
            
            return amount, currency, confidence
            
        except (ValueError, TypeError):
            logger.warning(f"Could not parse monetary amount: {amount_text}")
            return 0.0, currency, 0.0
    
    def _parse_written_amount(self, text: str) -> float:
        """Parse written monetary amounts like 'one million dollars'"""
        text_lower = text.lower()
        total = 0.0
        current = 0.0
        
        words = text_lower.split()
        for word in words:
            if word in self.WRITTEN_NUMBERS:
                value = self.WRITTEN_NUMBERS[word]
                if value == 100:
                    current *= 100
                elif value >= 1000:
                    total += current * value
                    current = 0
                else:
                    current += value
        
        return total + current if total > 0 else current
    
    def _deduplicate_amounts(self, amounts: List[MonetaryAmount]) -> List[MonetaryAmount]:
        """Remove duplicate amounts that are very close to each other"""
        if len(amounts) <= 1:
            return amounts
        
        deduplicated = []
        for amount in amounts:
            is_duplicate = False
            for existing in deduplicated:
                # Consider amounts within 5% of each other as duplicates
                if abs(amount.amount - existing.amount) / max(amount.amount, existing.amount) < 0.05:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(amount)
        
        return deduplicated

class LegalEntityExtractor:
    """Extracts legal entities and party roles from contract text"""
    
    # Party role indicators
    PARTY_ROLE_PATTERNS = {
        "licensor": [r'licensor', r'licensing\s+party', r'grantor'],
        "licensee": [r'licensee', r'licensed\s+party', r'grantee'],
        "buyer": [r'buyer', r'purchaser', r'acquiring\s+party'],
        "seller": [r'seller', r'vendor', r'disposing\s+party'],
        "contractor": [r'contractor', r'service\s+provider', r'consultant'],
        "client": [r'client', r'customer', r'service\s+recipient'],
        "landlord": [r'landlord', r'lessor', r'property\s+owner'],
        "tenant": [r'tenant', r'lessee', r'renter'],
        "lender": [r'lender', r'creditor', r'financing\s+party'],
        "borrower": [r'borrower', r'debtor', r'borrowing\s+party']
    }
    
    # Legal entity types
    ENTITY_PATTERNS = [
        r'\b\w+\s+(?:Inc\.?|Corporation|Corp\.?)\b',
        r'\b\w+\s+(?:LLC|L\.L\.C\.)\b', 
        r'\b\w+\s+(?:LP|L\.P\.|LLP|L\.L\.P\.)\b',
        r'\b\w+\s+(?:Ltd\.?|Limited)\b',
        r'\b\w+\s+(?:Co\.?|Company)\b',
        r'\b\w+\s+(?:Partnership|Partners)\b'
    ]
    
    def extract_party_role(self, text: str, clause_text: str) -> Optional[str]:
        """Extract the party role from contract context"""
        combined_text = f"{text} {clause_text}".lower()
        
        # Look for explicit role indicators
        for role, patterns in self.PARTY_ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return role
        
        # Infer from context
        if re.search(r'payment.*due.*to|pay.*to|remit.*to', combined_text):
            return "payer"
        elif re.search(r'receive.*payment|entitled.*to.*payment', combined_text):
            return "payee"
        elif re.search(r'provide.*services|perform.*work', combined_text):
            return "service_provider"
        elif re.search(r'receive.*services|customer', combined_text):
            return "service_recipient"
        
        return None
    
    def extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entity names from text"""
        entities = []
        
        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Clean and deduplicate
        entities = [entity.strip() for entity in entities]
        entities = list(set(entities))  # remove duplicates
        
        return entities

class JurisdictionExtractor:
    """Extracts governing law and jurisdiction information"""
    
    # US states and common jurisdictions
    US_STATES = [
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming"
    ]
    
    COUNTRIES = [
        "united states", "canada", "united kingdom", "germany", "france",
        "japan", "china", "australia", "brazil", "india", "mexico"
    ]
    
    def extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law from contract text"""
        text_lower = text.lower()
        
        # Look for explicit governing law clauses
        gov_law_patterns = [
            r'governed\s+by\s+the\s+laws?\s+of\s+([^,.;]+)',
            r'laws?\s+of\s+([^,.;]+)\s+shall\s+govern',
            r'construed\s+in\s+accordance\s+with\s+the\s+laws?\s+of\s+([^,.;]+)',
            r'subject\s+to\s+the\s+laws?\s+of\s+([^,.;]+)'
        ]
        
        for pattern in gov_law_patterns:
            match = re.search(pattern, text_lower)
            if match:
                jurisdiction = match.group(1).strip()
                return self._normalize_jurisdiction(jurisdiction)
        
        return None
    
    def extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract court jurisdiction from contract text"""
        text_lower = text.lower()
        
        jurisdiction_patterns = [
            r'courts?\s+of\s+([^,.;]+)\s+shall\s+have\s+jurisdiction',
            r'exclusive\s+jurisdiction\s+of\s+the\s+courts?\s+of\s+([^,.;]+)',
            r'submit\s+to\s+the\s+jurisdiction\s+of\s+([^,.;]+)',
            r'courts?\s+located\s+in\s+([^,.;]+)'
        ]
        
        for pattern in jurisdiction_patterns:
            match = re.search(pattern, text_lower)
            if match:
                jurisdiction = match.group(1).strip()
                return self._normalize_jurisdiction(jurisdiction)
        
        return None
    
    def _normalize_jurisdiction(self, jurisdiction: str) -> str:
        """Normalize jurisdiction text"""
        jurisdiction = jurisdiction.lower().strip()
        
        # Remove common prefixes/suffixes
        jurisdiction = re.sub(r'^the\s+', '', jurisdiction)
        jurisdiction = re.sub(r'\s+state$', '', jurisdiction)
        
        # Check if it's a US state
        for state in self.US_STATES:
            if state in jurisdiction:
                return state.title()
        
        # Check if it's a country
        for country in self.COUNTRIES:
            if country in jurisdiction:
                return country.title()
        
        return jurisdiction.title()

class RiskKeywordExtractor:
    """Extracts risk-indicating keywords and phrases"""
    
    # High-risk keywords by category
    RISK_KEYWORDS = {
        "liability": [
            "unlimited liability", "joint and several", "indemnify", "hold harmless",
            "defend", "liability", "damages", "losses", "claims"
        ],
        "termination": [
            "terminate immediately", "without notice", "for cause", "at will",
            "breach", "default", "suspension", "cancellation"
        ],
        "financial": [
            "penalty", "interest", "late fee", "acceleration", "default interest",
            "liquidated damages", "consequential damages", "lost profits"
        ],
        "intellectual_property": [
            "work for hire", "assignment", "exclusive license", "patent infringement",
            "trade secret", "confidential information", "proprietary"
        ],
        "regulatory": [
            "compliance", "audit", "inspection", "regulatory approval", "license",
            "permit", "certification", "standards"
        ]
    }
    
    def extract_risk_keywords(self, text: str) -> List[str]:
        """Extract risk keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for category, keywords in self.RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        return found_keywords

class FeatureExtractor:
    """Main feature extraction orchestrator"""
    
    def __init__(self):
        self.monetary_extractor = MonetaryExtractor()
        self.entity_extractor = LegalEntityExtractor()
        self.jurisdiction_extractor = JurisdictionExtractor()
        self.keyword_extractor = RiskKeywordExtractor()
    
    def extract_features(self, clause_text: str, 
                        document_context: str = "",
                        existing_metadata: Dict[str, Any] = None) -> ExtractedFeatures:
        """Extract all features from clause and document context"""
        
        combined_text = f"{document_context} {clause_text}"
        
        # Extract monetary amounts
        monetary_amounts = self.monetary_extractor.extract_monetary_amounts(clause_text)
        
        # Extract party role
        party_role = self.entity_extractor.extract_party_role(document_context, clause_text)
        
        # Extract governing law and jurisdiction
        governing_law = self.jurisdiction_extractor.extract_governing_law(combined_text)
        jurisdiction = self.jurisdiction_extractor.extract_jurisdiction(combined_text)
        
        # Extract legal entities
        legal_entities = self.entity_extractor.extract_legal_entities(combined_text)
        
        # Extract risk keywords
        risk_keywords = self.keyword_extractor.extract_risk_keywords(clause_text)
        
        # Extract time periods (simplified)
        time_periods = self._extract_time_periods(clause_text)
        
        # Merge with existing metadata if provided
        if existing_metadata:
            governing_law = governing_law or existing_metadata.get("governing_law")
            party_role = party_role or existing_metadata.get("party_role")
        
        return ExtractedFeatures(
            monetary_amounts=monetary_amounts,
            party_role=party_role,
            governing_law=governing_law,
            jurisdiction=jurisdiction,
            time_periods=time_periods,
            legal_entities=legal_entities,
            risk_keywords=risk_keywords
        )
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """Extract time periods and deadlines"""
        time_patterns = [
            r'\d+\s+(?:days?|weeks?|months?|years?)',
            r'(?:within|by|before|after)\s+\d+\s+(?:days?|weeks?|months?|years?)',
            r'(?:thirty|sixty|ninety)\s+(?:30|60|90)?\s*days?',
            r'immediately|forthwith|without delay|at once'
        ]
        
        periods = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            periods.extend(matches)
        
        return list(set(periods))  # Remove duplicates
    
    def get_largest_monetary_amount(self, features: ExtractedFeatures) -> Optional[float]:
        """Get the largest monetary amount from extracted features"""
        if not features.monetary_amounts:
            return None
        return features.monetary_amounts[0].amount  # Already sorted by amount descending