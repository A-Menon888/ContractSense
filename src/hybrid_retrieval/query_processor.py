"""
Query Processor for Hybrid Retrieval Engine

This module implements sophisticated natural language query processing for legal
document search, including entity extraction, intent classification, and query
enhancement for optimal hybrid search execution.

Key Features:
- Legal entity extraction (parties, amounts, dates, clause types)
- Intent classification for search strategy selection
- Query expansion with legal domain knowledge
- Complex query parsing and normalization

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import re
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_engine import SearchQuery
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import json
from datetime import datetime, date

@dataclass
class QueryProcessorConfig:
    """Configuration for query processing"""
    enable_entity_extraction: bool = True
    enable_intent_classification: bool = True
    enable_query_expansion: bool = True
    enable_spell_correction: bool = False
    max_expansion_terms: int = 5
    confidence_threshold: float = 0.3

class QueryIntent(Enum):
    """Types of query intents for legal document search"""
    CLAUSE_EXTRACTION = "clause_extraction"
    ENTITY_LOOKUP = "entity_lookup"  
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    TEMPORAL = "temporal"
    MONETARY = "monetary"
    UNKNOWN = "unknown"

@dataclass
class ExtractedEntity:
    """Represents an entity extracted from a query"""
    text: str
    type: str  # party, amount, date, clause_type, etc.
    confidence: float
    start_pos: int
    end_pos: int
    normalized_value: Any = None
    metadata: Dict[str, Any] = None

@dataclass
class QueryAnalysis:
    """Complete analysis of a processed query"""
    original_text: str
    normalized_text: str
    intent: QueryIntent
    confidence: float
    
    # Extracted components
    entities: List[ExtractedEntity]
    clause_types: List[str]
    amounts: List[Dict[str, Any]]
    dates: List[Dict[str, Any]]
    
    # Query characteristics
    complexity_score: float
    specificity_score: float
    
    # Enhancements
    expansion_terms: List[str]
    suggested_filters: Dict[str, Any]
    
    # Strategy recommendations
    recommended_strategy: str
    strategy_confidence: float

class QueryProcessor:
    """
    Advanced query processor for legal document search
    
    Processes natural language queries to extract structured information
    and provide guidance for optimal hybrid search execution.
    """
    
    def __init__(self, config: QueryProcessorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Legal domain patterns and vocabularies
        self._load_legal_patterns()
        self._load_legal_vocabulary()
        
        self.logger.info("Query processor initialized")
    
    def _load_legal_patterns(self):
        """Load regular expression patterns for legal entity extraction"""
        
        # Party name patterns
        self.party_patterns = [
            r'\b([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|LP|LLP)\.?)\b',
            r'\b([A-Z][a-zA-Z\s]+ (?:Corporation|Company|Limited|Partnership))\b',
            r'\b([A-Z][a-zA-Z\s]+ (?:Holdings|Ventures|Enterprises|Solutions))\b',
            r'\bParty\s+([A-Z])\b',
            r'\b(The\s+[A-Z][a-zA-Z\s]+ (?:Inc|Corp|LLC|Ltd)\.?)\b'
        ]
        
        # Monetary amount patterns
        self.amount_patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:million|M|thousand|K|billion|B)?',
            r'(?:USD|dollars?)\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars?)',
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?\b'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b', 
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b'
        ]
        
        # Notice period patterns
        self.notice_patterns = [
            r'(\d+)\s*(?:days?|months?|years?)\s*(?:notice|advance notice)',
            r'(?:notice|advance notice)\s*(?:of\s*)?(\d+)\s*(?:days?|months?|years?)',
            r'(\d+)[-\s](?:day|month|year)\s*(?:notice|period)'
        ]
        
    def _load_legal_vocabulary(self):
        """Load legal domain vocabulary and synonyms"""
        
        # Clause type vocabularies  
        self.clause_types = {
            'termination': ['termination', 'terminate', 'end', 'expiry', 'expiration', 'dissolution'],
            'liability': ['liability', 'liable', 'damages', 'harm', 'loss', 'injury'],
            'confidentiality': ['confidentiality', 'confidential', 'nda', 'non-disclosure', 'proprietary', 'secret'],
            'payment': ['payment', 'pay', 'compensation', 'remuneration', 'fees', 'salary', 'wages'],
            'intellectual_property': ['ip', 'intellectual property', 'patent', 'trademark', 'copyright', 'trade secret'],
            'indemnification': ['indemnify', 'indemnification', 'hold harmless', 'defend'],
            'force_majeure': ['force majeure', 'act of god', 'unforeseeable circumstances'],
            'arbitration': ['arbitration', 'arbitrate', 'dispute resolution', 'mediation'],
            'governing_law': ['governing law', 'jurisdiction', 'applicable law'],
            'assignment': ['assignment', 'assign', 'transfer', 'delegate']
        }
        
        # Legal action verbs
        self.action_verbs = {
            'find': ['find', 'locate', 'search', 'identify', 'discover'],
            'extract': ['extract', 'get', 'retrieve', 'pull', 'obtain'],
            'compare': ['compare', 'contrast', 'analyze', 'evaluate', 'assess'],
            'check': ['check', 'verify', 'validate', 'confirm', 'ensure'],
            'list': ['list', 'show', 'display', 'enumerate', 'itemize']
        }
        
        # Query expansion terms by domain
        self.expansion_terms = {
            'liability': ['limitation', 'cap', 'ceiling', 'maximum', 'exclude', 'disclaim'],
            'termination': ['breach', 'default', 'cure period', 'notice', 'effect'],
            'payment': ['schedule', 'due date', 'late fees', 'interest', 'installment'],
            'confidentiality': ['breach', 'disclosure', 'recipient', 'purpose', 'duration'],
            'intellectual_property': ['ownership', 'license', 'derivative works', 'infringement']
        }
        
        # Intent classification keywords
        self.intent_keywords = {
            QueryIntent.CLAUSE_EXTRACTION: ['find', 'extract', 'get', 'clauses', 'provisions', 'terms'],
            QueryIntent.ENTITY_LOOKUP: ['who', 'what', 'which', 'parties', 'entities'],
            QueryIntent.RELATIONSHIP: ['relationship', 'between', 'connect', 'related', 'associate'],
            QueryIntent.COMPARISON: ['compare', 'difference', 'versus', 'vs', 'contrast'],
            QueryIntent.COMPLIANCE: ['comply', 'compliance', 'requirement', 'standard', 'regulation'],
            QueryIntent.RISK_ASSESSMENT: ['risk', 'assess', 'analyze', 'evaluate', 'danger'],
            QueryIntent.TEMPORAL: ['when', 'date', 'time', 'period', 'duration', 'after', 'before'],
            QueryIntent.MONETARY: ['amount', 'cost', 'price', 'value', 'money', '$']
        }
    
    async def process_query(self, query) -> 'SearchQuery':
        """
        Process a search query and extract structured information
        
        Args:
            query: SearchQuery object with text to process
            
        Returns:
            Enhanced SearchQuery with extracted entities and metadata
        """
        from .hybrid_engine import SearchQuery  # Avoid circular import
        
        self.logger.debug(f"Processing query: '{query.text}'")
        
        try:
            # Normalize query text
            normalized_text = self._normalize_text(query.text)
            
            # Extract entities
            entities = []
            if self.config.enable_entity_extraction:
                entities = self._extract_entities(normalized_text)
            
            # Classify intent
            intent = QueryIntent.UNKNOWN
            intent_confidence = 0.0
            if self.config.enable_intent_classification:
                intent, intent_confidence = self._classify_intent(normalized_text)
            
            # Extract specific components
            clause_types = self._extract_clause_types(normalized_text)
            amounts = self._extract_amounts(normalized_text)
            dates = self._extract_dates(normalized_text)
            
            # Calculate query characteristics
            complexity = self._calculate_complexity(normalized_text, entities)
            specificity = self._calculate_specificity(entities, clause_types, amounts, dates)
            
            # Generate expansion terms
            expansion_terms = []
            if self.config.enable_query_expansion:
                expansion_terms = self._generate_expansion_terms(clause_types, intent)
            
            # Update query object with extracted information
            enhanced_query = SearchQuery(
                text=query.text,
                query_id=query.query_id,
                user_id=query.user_id,
                entities=[e.text for e in entities if e.type == 'party'],
                clause_types=clause_types,
                amounts=amounts,
                date_ranges=dates,
                max_results=query.max_results,
                min_confidence=query.min_confidence,
                strategy=query.strategy,
                filters=query.filters,
                include_confidence=query.include_confidence,
                include_explanation=query.include_explanation
            )
            
            # Add processing metadata
            if not hasattr(enhanced_query, 'processing_metadata'):
                enhanced_query.processing_metadata = {}
            
            enhanced_query.processing_metadata.update({
                'normalized_text': normalized_text,
                'intent': intent.value,
                'intent_confidence': intent_confidence,
                'extracted_entities': [
                    {
                        'text': e.text,
                        'type': e.type,
                        'confidence': e.confidence,
                        'normalized': e.normalized_value
                    } for e in entities
                ],
                'complexity_score': complexity,
                'specificity_score': specificity,
                'expansion_terms': expansion_terms,
                'processing_time': datetime.now().isoformat()
            })
            
            self.logger.debug(
                f"Query processed - Intent: {intent.value} "
                f"({intent_confidence:.3f}), Entities: {len(entities)}, "
                f"Clause types: {len(clause_types)}"
            )
            
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return query  # Return original query on failure
    
    def _normalize_text(self, text: str) -> str:
        """Normalize query text for better processing"""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Expand common abbreviations
        abbreviations = {
            'ip': 'intellectual property',
            'nda': 'non-disclosure agreement',
            'sla': 'service level agreement',
            'msa': 'master service agreement',
            'sow': 'statement of work'
        }
        
        for abbr, expansion in abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', expansion, normalized)
        
        return normalized
    
    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract structured entities from query text"""
        entities = []
        
        # Extract party names
        for pattern in self.party_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(1),
                    type='party',
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=match.group(1).strip()
                ))
        
        # Extract monetary amounts
        for pattern in self.amount_patterns:
            for match in re.finditer(pattern, text):
                amount_text = match.group(1)
                # Parse and normalize amount
                normalized_amount = self._parse_amount(amount_text, match.group(0))
                
                entities.append(ExtractedEntity(
                    text=match.group(0),
                    type='amount',
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=normalized_amount
                ))
        
        # Extract dates
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text):
                date_text = match.group(0)
                normalized_date = self._parse_date(date_text)
                
                entities.append(ExtractedEntity(
                    text=date_text,
                    type='date',
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=normalized_date
                ))
        
        # Extract notice periods
        for pattern in self.notice_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                period_text = match.group(0)
                normalized_period = self._parse_notice_period(match.group(1), period_text)
                
                entities.append(ExtractedEntity(
                    text=period_text,
                    type='notice_period',
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=normalized_period
                ))
        
        return entities
    
    def _classify_intent(self, text: str) -> Tuple[QueryIntent, float]:
        """Classify the intent of the query"""
        intent_scores = {}
        
        # Calculate scores for each intent based on keyword matching
        for intent, keywords in self.intent_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
            
            # Normalize by number of keywords
            intent_scores[intent] = score / len(keywords) if keywords else 0.0
        
        # Additional heuristics
        # Questions starting with "who", "what" -> ENTITY_LOOKUP
        if text.startswith(('who ', 'what ', 'which ')):
            intent_scores[QueryIntent.ENTITY_LOOKUP] += 0.5
        
        # "Find" or "get" clauses -> CLAUSE_EXTRACTION
        if any(verb in text for verb in ['find', 'get', 'extract', 'show']):
            if any(clause in text for clause in ['clause', 'provision', 'term']):
                intent_scores[QueryIntent.CLAUSE_EXTRACTION] += 0.5
        
        # Comparative language -> COMPARISON
        if any(word in text for word in ['compare', 'versus', 'vs', 'difference', 'between']):
            intent_scores[QueryIntent.COMPARISON] += 0.3
        
        # Find best intent
        if not intent_scores or max(intent_scores.values()) < self.config.confidence_threshold:
            return QueryIntent.UNKNOWN, 0.0
        
        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
        confidence = intent_scores[best_intent]
        
        return best_intent, confidence
    
    def _extract_clause_types(self, text: str) -> List[str]:
        """Extract clause types mentioned in the query"""
        found_types = []
        
        for clause_type, synonyms in self.clause_types.items():
            for synonym in synonyms:
                if synonym in text:
                    if clause_type not in found_types:
                        found_types.append(clause_type)
                    break
        
        return found_types
    
    def _extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts with context"""
        amounts = []
        
        for pattern in self.amount_patterns:
            for match in re.finditer(pattern, text):
                amount_data = {
                    'text': match.group(0),
                    'value': self._parse_amount(match.group(1), match.group(0)),
                    'position': match.start()
                }
                amounts.append(amount_data)
        
        return amounts
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract date ranges and temporal expressions"""
        dates = []
        
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text):
                date_data = {
                    'text': match.group(0),
                    'parsed': self._parse_date(match.group(0)),
                    'position': match.start()
                }
                dates.append(date_data)
        
        # Look for date ranges
        range_patterns = [
            r'from\s+(\d{4})\s+to\s+(\d{4})',
            r'between\s+(\d{4})\s+and\s+(\d{4})',
            r'after\s+(\d{4})',
            r'before\s+(\d{4})',
            r'since\s+(\d{4})'
        ]
        
        for pattern in range_patterns:
            for match in re.finditer(pattern, text):
                range_data = {
                    'text': match.group(0),
                    'type': 'range',
                    'start': match.group(1) if match.groups() else None,
                    'end': match.group(2) if len(match.groups()) > 1 else None,
                    'position': match.start()
                }
                dates.append(range_data)
        
        return dates
    
    def _parse_amount(self, amount_str: str, full_match: str) -> float:
        """Parse monetary amount string to float"""
        try:
            # Remove commas and convert to float
            amount = float(amount_str.replace(',', ''))
            
            # Handle multipliers
            if 'million' in full_match.lower() or 'm' in full_match.lower():
                amount *= 1_000_000
            elif 'billion' in full_match.lower() or 'b' in full_match.lower():
                amount *= 1_000_000_000
            elif 'thousand' in full_match.lower() or 'k' in full_match.lower():
                amount *= 1_000
            
            return amount
        except ValueError:
            return 0.0
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format"""
        try:
            # Try different date formats
            formats = ['%m/%d/%Y', '%Y-%m-%d', '%B %d, %Y', '%d %B %Y']
            
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.date().isoformat()
                except ValueError:
                    continue
            
            return None
        except Exception:
            return None
    
    def _parse_notice_period(self, number: str, full_text: str) -> Dict[str, Any]:
        """Parse notice period into structured format"""
        try:
            num_value = int(number)
            
            # Determine unit
            unit = 'days'  # default
            if 'month' in full_text.lower():
                unit = 'months'
            elif 'year' in full_text.lower():
                unit = 'years'
            
            return {
                'value': num_value,
                'unit': unit,
                'days': num_value if unit == 'days' else (num_value * 30 if unit == 'months' else num_value * 365)
            }
        except ValueError:
            return {'value': 0, 'unit': 'days', 'days': 0}
    
    def _calculate_complexity(self, text: str, entities: List[ExtractedEntity]) -> float:
        """Calculate query complexity score"""
        # Base complexity from text length
        base_score = min(len(text.split()) / 20.0, 1.0)
        
        # Add complexity for entities
        entity_score = min(len(entities) / 10.0, 0.5)
        
        # Add complexity for operators and conjunctions
        operators = ['and', 'or', 'not', 'but', 'except', 'where', 'if']
        operator_score = sum(1 for op in operators if op in text) / len(operators)
        
        return min(base_score + entity_score + operator_score, 1.0)
    
    def _calculate_specificity(self, entities: List[ExtractedEntity], 
                             clause_types: List[str], amounts: List[Dict], 
                             dates: List[Dict]) -> float:
        """Calculate query specificity score"""
        specificity = 0.0
        
        # Specific entities add to specificity
        specificity += len(entities) * 0.1
        
        # Specific clause types add to specificity  
        specificity += len(clause_types) * 0.15
        
        # Amounts and dates add to specificity
        specificity += len(amounts) * 0.2
        specificity += len(dates) * 0.15
        
        return min(specificity, 1.0)
    
    def _generate_expansion_terms(self, clause_types: List[str], intent: QueryIntent) -> List[str]:
        """Generate query expansion terms based on extracted information"""
        expansion_terms = []
        
        # Add expansion terms for each clause type
        for clause_type in clause_types:
            if clause_type in self.expansion_terms:
                expansion_terms.extend(self.expansion_terms[clause_type])
        
        # Add intent-specific expansion terms
        if intent == QueryIntent.RISK_ASSESSMENT:
            expansion_terms.extend(['liability', 'exposure', 'threat', 'vulnerability', 'danger'])
        elif intent == QueryIntent.COMPLIANCE:
            expansion_terms.extend(['regulation', 'requirement', 'standard', 'rule', 'law'])
        elif intent == QueryIntent.CLAUSE_EXTRACTION:
            expansion_terms.extend(['provision', 'term', 'condition', 'section', 'article'])
        
        # Remove duplicates and limit
        expansion_terms = list(set(expansion_terms))
        return expansion_terms[:self.config.max_expansion_terms]
    
    def analyze_query(self, query_text: str) -> QueryAnalysis:
        """
        Perform complete analysis of a query (for debugging/analysis)
        
        Args:
            query_text: Raw query text
            
        Returns:
            Complete QueryAnalysis object
        """
        normalized_text = self._normalize_text(query_text)
        entities = self._extract_entities(normalized_text)
        intent, intent_confidence = self._classify_intent(normalized_text)
        clause_types = self._extract_clause_types(normalized_text)
        amounts = self._extract_amounts(normalized_text)
        dates = self._extract_dates(normalized_text)
        
        complexity = self._calculate_complexity(normalized_text, entities)
        specificity = self._calculate_specificity(entities, clause_types, amounts, dates)
        
        expansion_terms = self._generate_expansion_terms(clause_types, intent)
        
        # Generate strategy recommendation
        recommended_strategy = self._recommend_strategy(
            intent, entities, clause_types, amounts, dates, complexity
        )
        
        return QueryAnalysis(
            original_text=query_text,
            normalized_text=normalized_text,
            intent=intent,
            confidence=intent_confidence,
            entities=entities,
            clause_types=clause_types,
            amounts=amounts,
            dates=dates,
            complexity_score=complexity,
            specificity_score=specificity,
            expansion_terms=expansion_terms,
            suggested_filters={},
            recommended_strategy=recommended_strategy,
            strategy_confidence=0.8
        )
    
    def _recommend_strategy(self, intent: QueryIntent, entities: List[ExtractedEntity],
                          clause_types: List[str], amounts: List[Dict],
                          dates: List[Dict], complexity: float) -> str:
        """Recommend search strategy based on query analysis"""
        
        # Structured queries with specific entities -> graph heavy
        if entities and clause_types and (amounts or dates):
            return "graph_heavy"
        
        # Relationship queries -> graph focused
        if intent in [QueryIntent.RELATIONSHIP, QueryIntent.ENTITY_LOOKUP]:
            return "graph_only" if entities else "graph_heavy"
        
        # Complex natural language -> vector heavy
        if complexity > 0.7 and not entities:
            return "vector_heavy"
        
        # Comparison queries -> hybrid
        if intent == QueryIntent.COMPARISON:
            return "hybrid_parallel"
        
        # Default balanced approach
        return "balanced"

# Factory function
def create_query_processor(config: QueryProcessorConfig = None) -> QueryProcessor:
    """Create and return a configured query processor"""
    if config is None:
        config = QueryProcessorConfig()
    return QueryProcessor(config)