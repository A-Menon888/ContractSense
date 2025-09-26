"""
Query Processor

This module provides intelligent query understanding and processing capabilities
for contract analysis, including intent detection, entity extraction, and query optimization.

Key Features:
- Query intent classification (search, analysis, comparison, etc.)
- Named entity recognition for contracts
- Query expansion and optimization
- Natural language to structured query translation
- Context-aware query refinement
- Integration with knowledge graph ontology
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

# Import hybrid retrieval components
from .hybrid_retriever import HybridQuery, RetrievalStrategy

# Import knowledge graph components  
try:
    from ..knowledge_graph.schema import NodeType, RelationshipType
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    NodeType = RelationshipType = None
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logging.warning("Knowledge graph schema not available")

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Different types of query intents"""
    SEARCH = "search"                    # Find similar content
    ENTITY_LOOKUP = "entity_lookup"      # Find specific entities
    RELATIONSHIP = "relationship"        # Explore relationships
    ANALYSIS = "analysis"               # Analyze patterns or risks
    COMPARISON = "comparison"           # Compare contracts/clauses
    COMPLIANCE = "compliance"           # Check compliance requirements
    RISK_ASSESSMENT = "risk_assessment" # Assess risk levels
    CLAUSE_EXTRACTION = "clause_extraction" # Extract specific clause types
    TIMELINE = "timeline"               # Temporal queries
    SUMMARY = "summary"                 # Summarization requests
    UNKNOWN = "unknown"                 # Cannot determine intent

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    canonical_form: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryAnalysis:
    """Comprehensive query analysis results"""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    confidence: float
    
    # Extracted components
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    temporal_expressions: List[str] = field(default_factory=list)
    clause_types: List[str] = field(default_factory=list)
    
    # Query characteristics
    complexity_score: float = 0.0
    specificity_score: float = 0.0
    ambiguity_indicators: List[str] = field(default_factory=list)
    
    # Suggested parameters
    suggested_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    suggested_filters: Dict[str, Any] = field(default_factory=dict)
    expansion_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_query": self.original_query,
            "cleaned_query": self.cleaned_query,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "confidence": e.confidence,
                    "canonical_form": e.canonical_form
                }
                for e in self.entities
            ],
            "relationships": self.relationships,
            "temporal_expressions": self.temporal_expressions,
            "clause_types": self.clause_types,
            "complexity_score": self.complexity_score,
            "specificity_score": self.specificity_score,
            "suggested_strategy": self.suggested_strategy.value,
            "suggested_filters": self.suggested_filters,
            "expansion_terms": self.expansion_terms
        }

class QueryProcessor:
    """Main query processing engine"""
    
    def __init__(self):
        self.stats = {
            "queries_processed": 0,
            "intent_accuracy": {},
            "entity_extraction_count": 0,
            "average_complexity": 0.0
        }
        
        # Load patterns and rules
        self._load_patterns()
        
        # Initialize entity patterns
        self._init_entity_patterns()
        
        logger.info("Query processor initialized")
    
    def _load_patterns(self):
        """Load intent classification patterns and rules"""
        
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r"find|search|look for|show me|get",
                r"similar to|like|comparable",
                r"contains|includes|mentions"
            ],
            QueryIntent.ENTITY_LOOKUP: [
                r"who is|what is|which company",
                r"tell me about|information about",
                r"details of|profile of"
            ],
            QueryIntent.RELATIONSHIP: [
                r"relationship between|connected to|related to",
                r"how.*connected|association",
                r"links|connections|ties"
            ],
            QueryIntent.ANALYSIS: [
                r"analyze|analysis|examine",
                r"pattern|trend|insight",
                r"statistics|metrics|data"
            ],
            QueryIntent.COMPARISON: [
                r"compare|comparison|versus|vs",
                r"difference|similar|contrast",
                r"better|worse|superior"
            ],
            QueryIntent.COMPLIANCE: [
                r"compliant|compliance|regulation",
                r"legal requirement|mandate|rule",
                r"violates|violation|breach"
            ],
            QueryIntent.RISK_ASSESSMENT: [
                r"risk|risky|dangerous",
                r"threat|vulnerability|exposure",
                r"assess.*risk|risk.*level"
            ],
            QueryIntent.CLAUSE_EXTRACTION: [
                r"clause|section|provision",
                r"term|condition|requirement",
                r"extract.*clause|find.*clause"
            ],
            QueryIntent.TIMELINE: [
                r"when|timeline|chronology",
                r"before|after|during|between.*date",
                r"history|sequence|order"
            ],
            QueryIntent.SUMMARY: [
                r"summarize|summary|overview",
                r"key points|main.*point",
                r"brief|concise|outline"
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _init_entity_patterns(self):
        """Initialize entity extraction patterns"""
        
        # Company/Organization patterns
        self.company_patterns = [
            re.compile(r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(Inc\.?|LLC|Corp\.?|Corporation|Company|Ltd\.?|Limited|LP|LLP)\b'),
            re.compile(r'\b(The\s+)?([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(Group|Holdings?|Enterprises?|Solutions?|Systems?|Technologies?)\b'),
        ]
        
        # Person name patterns
        self.person_patterns = [
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\b'),
        ]
        
        # Date patterns
        self.date_patterns = [
            re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'),
            re.compile(r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b'),
            re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'),
        ]
        
        # Monetary amount patterns
        self.money_patterns = [
            re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?', re.IGNORECASE),
            re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?)\b', re.IGNORECASE),
        ]
        
        # Contract-specific terms
        if KNOWLEDGE_GRAPH_AVAILABLE:
            self.clause_type_patterns = {
                "payment_terms": [r"payment.*term", r"invoice", r"billing", r"payment.*schedule"],
                "termination": [r"terminat", r"end.*agreement", r"expir"],
                "confidentiality": [r"confidential", r"non.*disclos", r"proprietary", r"trade.*secret"],
                "liability": [r"liability", r"damages", r"indemnif", r"limitation.*liability"],
                "intellectual_property": [r"intellectual.*property", r"copyright", r"patent", r"trademark", r"IP"],
                "governing_law": [r"governing.*law", r"jurisdiction", r"applicable.*law"],
                "force_majeure": [r"force.*majeure", r"act.*god", r"unforeseeable"],
                "warranty": [r"warrant", r"guarantee", r"represent"],
            }
        else:
            self.clause_type_patterns = {}
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """Main query processing method"""
        
        try:
            # Clean and normalize query
            cleaned_query = self._clean_query(query)
            
            # Classify intent
            intent, intent_confidence = self._classify_intent(cleaned_query)
            
            # Extract entities
            entities = self._extract_entities(cleaned_query)
            
            # Extract relationships
            relationships = self._extract_relationships(cleaned_query)
            
            # Extract temporal expressions
            temporal_expressions = self._extract_temporal_expressions(cleaned_query)
            
            # Extract clause types
            clause_types = self._extract_clause_types(cleaned_query)
            
            # Calculate query characteristics
            complexity_score = self._calculate_complexity(cleaned_query, entities)
            specificity_score = self._calculate_specificity(cleaned_query, entities)
            ambiguity_indicators = self._detect_ambiguity(cleaned_query)
            
            # Suggest strategy and parameters
            suggested_strategy = self._suggest_strategy(intent, entities, relationships)
            suggested_filters = self._suggest_filters(entities, temporal_expressions, clause_types)
            expansion_terms = self._generate_expansion_terms(cleaned_query, intent, entities)
            
            # Create analysis result
            analysis = QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned_query,
                intent=intent,
                confidence=intent_confidence,
                entities=entities,
                relationships=relationships,
                temporal_expressions=temporal_expressions,
                clause_types=clause_types,
                complexity_score=complexity_score,
                specificity_score=specificity_score,
                ambiguity_indicators=ambiguity_indicators,
                suggested_strategy=suggested_strategy,
                suggested_filters=suggested_filters,
                expansion_terms=expansion_terms
            )
            
            # Update statistics
            self._update_stats(analysis)
            
            logger.debug(f"Processed query: intent={intent.value}, confidence={intent_confidence:.3f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return minimal analysis
            return QueryAnalysis(
                original_query=query,
                cleaned_query=query,
                intent=QueryIntent.UNKNOWN,
                confidence=0.0
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        
        # Basic cleaning
        cleaned = query.strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Normalize punctuation
        cleaned = re.sub(r'[^\w\s\-\.\,\?\!\$\%]', ' ', cleaned)
        
        # Remove very short tokens (less than 2 chars) that aren't meaningful
        tokens = cleaned.split()
        meaningful_tokens = []
        for token in tokens:
            if len(token) >= 2 or token.lower() in ['a', 'i', 'is', 'in', 'of', 'to', 'or']:
                meaningful_tokens.append(token)
        
        cleaned = ' '.join(meaningful_tokens)
        
        return cleaned
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent using pattern matching"""
        
        intent_scores = {intent: 0.0 for intent in QueryIntent}
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = len(pattern.findall(query))
                if matches > 0:
                    # Weight by pattern specificity and match count
                    intent_scores[intent] += matches * 1.0
        
        # If no patterns match, try keyword-based classification
        if max(intent_scores.values()) == 0:
            intent_scores = self._keyword_based_classification(query)
        
        # Find best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent] / max(sum(intent_scores.values()), 1.0)
        
        # If confidence is too low, mark as unknown
        if confidence < 0.3:
            return QueryIntent.UNKNOWN, confidence
        
        return best_intent, confidence
    
    def _keyword_based_classification(self, query: str) -> Dict[QueryIntent, float]:
        """Fallback keyword-based intent classification"""
        
        query_lower = query.lower()
        scores = {intent: 0.0 for intent in QueryIntent}
        
        # Simple keyword matching
        keyword_mappings = {
            QueryIntent.SEARCH: ["find", "search", "show", "get", "retrieve"],
            QueryIntent.ENTITY_LOOKUP: ["who", "what", "which", "company", "person"],
            QueryIntent.RELATIONSHIP: ["relationship", "connect", "link", "between"],
            QueryIntent.ANALYSIS: ["analyze", "analysis", "pattern", "trend"],
            QueryIntent.COMPARISON: ["compare", "versus", "vs", "difference"],
            QueryIntent.COMPLIANCE: ["compliance", "legal", "regulation", "rule"],
            QueryIntent.RISK_ASSESSMENT: ["risk", "dangerous", "threat"],
            QueryIntent.CLAUSE_EXTRACTION: ["clause", "section", "term", "provision"],
            QueryIntent.TIMELINE: ["when", "date", "time", "before", "after"],
            QueryIntent.SUMMARY: ["summary", "summarize", "overview", "brief"]
        }
        
        for intent, keywords in keyword_mappings.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[intent] += 1.0
        
        return scores
    
    def _extract_entities(self, query: str) -> List[ExtractedEntity]:
        """Extract named entities from query"""
        
        entities = []
        
        # Extract companies
        for pattern in self.company_patterns:
            for match in pattern.finditer(query):
                entity = ExtractedEntity(
                    text=match.group().strip(),
                    entity_type="COMPANY",
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    canonical_form=match.group().strip()
                )
                entities.append(entity)
        
        # Extract persons
        for pattern in self.person_patterns:
            for match in pattern.finditer(query):
                # Simple heuristic to avoid false positives
                text = match.group().strip()
                if len(text.split()) >= 2 and text not in ["Inc", "Corp", "LLC"]:
                    entity = ExtractedEntity(
                        text=text,
                        entity_type="PERSON",
                        confidence=0.6,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        canonical_form=text
                    )
                    entities.append(entity)
        
        # Extract dates
        for pattern in self.date_patterns:
            for match in pattern.finditer(query):
                entity = ExtractedEntity(
                    text=match.group().strip(),
                    entity_type="DATE",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    canonical_form=match.group().strip()
                )
                entities.append(entity)
        
        # Extract monetary amounts
        for pattern in self.money_patterns:
            for match in pattern.finditer(query):
                entity = ExtractedEntity(
                    text=match.group().strip(),
                    entity_type="MONEY",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    canonical_form=match.group().strip()
                )
                entities.append(entity)
        
        # Sort by position and remove overlapping entities
        entities.sort(key=lambda x: x.start_pos)
        filtered_entities = []
        
        for entity in entities:
            # Check for overlap with existing entities
            overlap = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Keep entity with higher confidence
                    if entity.confidence <= existing.confidence:
                        overlap = True
                        break
                    else:
                        # Remove existing lower confidence entity
                        filtered_entities.remove(existing)
                        break
            
            if not overlap:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _extract_relationships(self, query: str) -> List[str]:
        """Extract relationship indicators from query"""
        
        relationship_patterns = [
            r"relationship between",
            r"connected to",
            r"related to", 
            r"associated with",
            r"linked to",
            r"ties to",
            r"partnership",
            r"agreement with",
            r"contract with"
        ]
        
        relationships = []
        query_lower = query.lower()
        
        for pattern in relationship_patterns:
            if re.search(pattern, query_lower):
                relationships.append(pattern.replace(r"\b", "").replace(r"\s+", " "))
        
        return relationships
    
    def _extract_temporal_expressions(self, query: str) -> List[str]:
        """Extract temporal expressions from query"""
        
        temporal_expressions = []
        
        # Date patterns already extracted in entities
        # Look for relative temporal expressions
        temporal_patterns = [
            r"before \d{4}",
            r"after \d{4}", 
            r"in \d{4}",
            r"during \d{4}",
            r"between.*and",
            r"last year",
            r"this year",
            r"next year",
            r"recently",
            r"historically"
        ]
        
        query_lower = query.lower()
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query_lower)
            temporal_expressions.extend(matches)
        
        return temporal_expressions
    
    def _extract_clause_types(self, query: str) -> List[str]:
        """Extract clause type indicators from query"""
        
        clause_types = []
        query_lower = query.lower()
        
        for clause_type, patterns in self.clause_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    clause_types.append(clause_type)
                    break  # Only add each clause type once
        
        return clause_types
    
    def _calculate_complexity(self, query: str, entities: List[ExtractedEntity]) -> float:
        """Calculate query complexity score"""
        
        # Factors that increase complexity
        factors = []
        
        # Length factor
        word_count = len(query.split())
        factors.append(min(word_count / 20.0, 1.0))  # Normalize to max 1.0
        
        # Entity count factor
        factors.append(min(len(entities) / 5.0, 1.0))
        
        # Boolean operators
        boolean_count = len(re.findall(r'\b(and|or|not|but)\b', query.lower()))
        factors.append(min(boolean_count / 3.0, 1.0))
        
        # Nested expressions (parentheses, quotes)
        nesting_count = query.count('(') + query.count('"')
        factors.append(min(nesting_count / 2.0, 1.0))
        
        # Question words (who, what, when, where, why, how)
        question_count = len(re.findall(r'\b(who|what|when|where|why|how)\b', query.lower()))
        factors.append(min(question_count / 2.0, 1.0))
        
        # Calculate weighted average
        complexity = sum(factors) / len(factors) if factors else 0.0
        
        return complexity
    
    def _calculate_specificity(self, query: str, entities: List[ExtractedEntity]) -> float:
        """Calculate query specificity score"""
        
        specificity_factors = []
        
        # Entity specificity (more entities = more specific)
        specificity_factors.append(min(len(entities) / 3.0, 1.0))
        
        # Proper noun count
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+', query))
        specificity_factors.append(min(proper_nouns / 5.0, 1.0))
        
        # Numbers and dates increase specificity
        number_count = len(re.findall(r'\b\d+', query))
        specificity_factors.append(min(number_count / 3.0, 1.0))
        
        # Quotes indicate specific terms
        quote_count = query.count('"')
        specificity_factors.append(min(quote_count / 2.0, 1.0))
        
        # Vague terms decrease specificity
        vague_terms = ["something", "anything", "things", "stuff", "some", "any", "general"]
        vague_count = sum(1 for term in vague_terms if term in query.lower())
        vagueness_penalty = min(vague_count / 3.0, 0.5)
        
        specificity = max(0.0, sum(specificity_factors) / len(specificity_factors) - vagueness_penalty)
        
        return specificity
    
    def _detect_ambiguity(self, query: str) -> List[str]:
        """Detect ambiguity indicators in query"""
        
        ambiguity_indicators = []
        query_lower = query.lower()
        
        # Ambiguous pronouns
        ambiguous_pronouns = ["it", "this", "that", "they", "them"]
        for pronoun in ambiguous_pronouns:
            if re.search(rf'\b{pronoun}\b', query_lower):
                ambiguity_indicators.append(f"ambiguous_pronoun: {pronoun}")
        
        # Multiple possible interpretations
        if " or " in query_lower:
            ambiguity_indicators.append("multiple_options")
        
        # Vague quantifiers
        vague_quantifiers = ["some", "many", "few", "several", "most", "various"]
        for quantifier in vague_quantifiers:
            if re.search(rf'\b{quantifier}\b', query_lower):
                ambiguity_indicators.append(f"vague_quantifier: {quantifier}")
        
        # Context-dependent terms
        context_dependent = ["recent", "old", "new", "big", "small", "important", "relevant"]
        for term in context_dependent:
            if re.search(rf'\b{term}\b', query_lower):
                ambiguity_indicators.append(f"context_dependent: {term}")
        
        return ambiguity_indicators
    
    def _suggest_strategy(self, intent: QueryIntent, 
                         entities: List[ExtractedEntity],
                         relationships: List[str]) -> RetrievalStrategy:
        """Suggest optimal retrieval strategy based on query analysis"""
        
        # Intent-based strategy mapping
        intent_strategy_map = {
            QueryIntent.SEARCH: RetrievalStrategy.VECTOR_ONLY,
            QueryIntent.ENTITY_LOOKUP: RetrievalStrategy.GRAPH_GUIDED_VECTOR,
            QueryIntent.RELATIONSHIP: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.ANALYSIS: RetrievalStrategy.HYBRID_PARALLEL,
            QueryIntent.COMPARISON: RetrievalStrategy.HYBRID_PARALLEL,
            QueryIntent.COMPLIANCE: RetrievalStrategy.GRAPH_GUIDED_VECTOR,
            QueryIntent.RISK_ASSESSMENT: RetrievalStrategy.VECTOR_EXPANDED_GRAPH,
            QueryIntent.CLAUSE_EXTRACTION: RetrievalStrategy.GRAPH_GUIDED_VECTOR,
            QueryIntent.TIMELINE: RetrievalStrategy.GRAPH_ONLY,
            QueryIntent.SUMMARY: RetrievalStrategy.VECTOR_ONLY,
            QueryIntent.UNKNOWN: RetrievalStrategy.ADAPTIVE
        }
        
        base_strategy = intent_strategy_map.get(intent, RetrievalStrategy.ADAPTIVE)
        
        # Modify based on query characteristics
        if len(entities) > 3:
            # Many entities suggest graph-heavy approach
            if base_strategy == RetrievalStrategy.VECTOR_ONLY:
                base_strategy = RetrievalStrategy.HYBRID_PARALLEL
        
        if len(relationships) > 0:
            # Relationship indicators suggest graph involvement
            if base_strategy == RetrievalStrategy.VECTOR_ONLY:
                base_strategy = RetrievalStrategy.GRAPH_GUIDED_VECTOR
        
        return base_strategy
    
    def _suggest_filters(self, entities: List[ExtractedEntity],
                        temporal_expressions: List[str],
                        clause_types: List[str]) -> Dict[str, Any]:
        """Suggest metadata filters based on extracted information"""
        
        filters = {}
        
        # Entity-based filters
        companies = [e.canonical_form for e in entities if e.entity_type == "COMPANY"]
        if companies:
            filters["companies"] = companies
        
        persons = [e.canonical_form for e in entities if e.entity_type == "PERSON"]  
        if persons:
            filters["persons"] = persons
        
        # Date-based filters
        dates = [e.canonical_form for e in entities if e.entity_type == "DATE"]
        if dates:
            filters["dates"] = dates
        
        # Monetary filters
        amounts = [e.canonical_form for e in entities if e.entity_type == "MONEY"]
        if amounts:
            filters["monetary_amounts"] = amounts
        
        # Clause type filters
        if clause_types:
            filters["clause_types"] = clause_types
        
        # Temporal filters
        if temporal_expressions:
            filters["temporal_expressions"] = temporal_expressions
        
        return filters
    
    def _generate_expansion_terms(self, query: str, intent: QueryIntent,
                                 entities: List[ExtractedEntity]) -> List[str]:
        """Generate query expansion terms"""
        
        expansion_terms = []
        
        # Intent-based expansion
        intent_expansions = {
            QueryIntent.COMPLIANCE: ["regulation", "requirement", "standard", "rule", "law"],
            QueryIntent.RISK_ASSESSMENT: ["liability", "exposure", "threat", "vulnerability", "danger"],
            QueryIntent.CLAUSE_EXTRACTION: ["provision", "term", "condition", "section", "article"],
            QueryIntent.ANALYSIS: ["pattern", "trend", "insight", "statistics", "metric"]
        }
        
        if intent in intent_expansions:
            expansion_terms.extend(intent_expansions[intent])
        
        # Entity-based expansion
        for entity in entities:
            if entity.entity_type == "COMPANY":
                expansion_terms.extend(["corporation", "organization", "business", "entity"])
            elif entity.entity_type == "PERSON":
                expansion_terms.extend(["individual", "party", "signatory", "representative"])
            elif entity.entity_type == "DATE":
                expansion_terms.extend(["timeline", "schedule", "period", "duration"])
        
        # Remove duplicates and terms already in query
        query_words = set(query.lower().split())
        unique_terms = []
        for term in expansion_terms:
            if term.lower() not in query_words and term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms[:5]  # Limit to top 5 expansion terms
    
    def _update_stats(self, analysis: QueryAnalysis):
        """Update processing statistics"""
        
        self.stats["queries_processed"] += 1
        
        # Track intent accuracy (would need ground truth for real accuracy)
        intent = analysis.intent.value
        if intent not in self.stats["intent_accuracy"]:
            self.stats["intent_accuracy"][intent] = 0
        self.stats["intent_accuracy"][intent] += 1
        
        # Track entity extraction
        self.stats["entity_extraction_count"] += len(analysis.entities)
        
        # Track complexity
        total_complexity = (self.stats["average_complexity"] * (self.stats["queries_processed"] - 1) + 
                           analysis.complexity_score)
        self.stats["average_complexity"] = total_complexity / self.stats["queries_processed"]
    
    def to_hybrid_query(self, analysis: QueryAnalysis, 
                       additional_params: Dict[str, Any] = None) -> HybridQuery:
        """Convert query analysis to HybridQuery object"""
        
        # Extract entity hints
        entity_hints = [e.canonical_form for e in analysis.entities]
        
        # Build metadata filter
        metadata_filter = analysis.suggested_filters.copy()
        if additional_params and "metadata_filter" in additional_params:
            metadata_filter.update(additional_params["metadata_filter"])
        
        # Create HybridQuery
        hybrid_query = HybridQuery(
            text=analysis.cleaned_query,
            query_type=analysis.intent.value,
            entity_hints=entity_hints,
            relationship_hints=analysis.relationships,
            strategy=analysis.suggested_strategy,
            metadata_filter=metadata_filter,
            top_k=additional_params.get("top_k", 10) if additional_params else 10,
            include_context=True,
            include_explanations=True
        )
        
        # Add any additional parameters
        if additional_params:
            for key, value in additional_params.items():
                if hasattr(hybrid_query, key):
                    setattr(hybrid_query, key, value)
        
        return hybrid_query
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# Factory function
def create_query_processor() -> QueryProcessor:
    """Factory function to create query processor"""
    return QueryProcessor()