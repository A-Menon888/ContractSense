"""
Question Processing Component

Analyzes and preprocesses user questions for optimal retrieval and answer generation.
Handles question classification, entity extraction, and intent analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    from ..models.question_models import (
        QuestionType, QuestionIntent, ComplexityLevel, Entity, QuestionAnalysis
    )
    from ..utils.common import (
        generate_id, clean_text, extract_legal_entities, extract_keywords,
        extract_legal_concepts, calculate_confidence_score, Timer
    )
except ImportError:
    # Fallback for direct execution
    from provenance_qa.models.question_models import (
        QuestionType, QuestionIntent, ComplexityLevel, Entity, QuestionAnalysis
    )
    from provenance_qa.utils.common import (
        generate_id, clean_text, extract_legal_entities, extract_keywords,
        extract_legal_concepts, calculate_confidence_score, Timer
    )

logger = logging.getLogger(__name__)

class QuestionProcessor:
    """Comprehensive question analysis and preprocessing"""
    
    def __init__(self):
        self.question_type_patterns = self._build_question_type_patterns()
        self.intent_patterns = self._build_intent_patterns()
        self.complexity_indicators = self._build_complexity_indicators()
    
    def _build_question_type_patterns(self) -> Dict[QuestionType, List[str]]:
        """Build patterns for question type classification"""
        return {
            QuestionType.FACTUAL: [
                r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhere\s+is\b', r'\bwhen\s+is\b',
                r'\bhow\s+much\b', r'\bhow\s+many\b', r'\bwhich\b', r'\btell\s+me\b'
            ],
            QuestionType.COMPARATIVE: [
                r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b', r'\bsimilar\b',
                r'\bbetter\b', r'\bworse\b', r'\bversus\b', r'\bvs\b', r'\bbetween\b'
            ],
            QuestionType.ANALYTICAL: [
                r'\banalyze\b', r'\banalysis\b', r'\bevaluate\b', r'\bassess\b',
                r'\bexamine\b', r'\bwhat\s+are\s+the\s+implications\b', r'\bimpact\b'
            ],
            QuestionType.PROCEDURAL: [
                r'\bhow\s+to\b', r'\bhow\s+do\s+i\b', r'\bhow\s+can\s+i\b',
                r'\bsteps\b', r'\bprocess\b', r'\bprocedure\b', r'\bmethod\b'
            ],
            QuestionType.DEFINITIONAL: [
                r'\bdefine\b', r'\bdefinition\b', r'\bwhat\s+does\s+.+\s+mean\b',
                r'\bwhat\s+is\s+the\s+meaning\b', r'\bexplain\b'
            ],
            QuestionType.QUANTITATIVE: [
                r'\bhow\s+much\b', r'\bhow\s+many\b', r'\bamount\b', r'\bcost\b',
                r'\bprice\b', r'\bvalue\b', r'\bnumber\b', r'\bpercentage\b'
            ],
            QuestionType.TEMPORAL: [
                r'\bwhen\b', r'\bdate\b', r'\btime\b', r'\bdeadline\b',
                r'\bexpires?\b', r'\bexpiration\b', r'\bschedule\b'
            ],
            QuestionType.CONDITIONAL: [
                r'\bwhat\s+if\b', r'\bif\b', r'\bwhen\s+.+\s+happens\b',
                r'\bin\s+case\b', r'\bunder\s+what\s+circumstances\b'
            ]
        }
    
    def _build_intent_patterns(self) -> Dict[QuestionIntent, List[str]]:
        """Build patterns for intent classification"""
        return {
            QuestionIntent.INFORMATION_SEEKING: [
                r'\btell\s+me\b', r'\binformation\b', r'\bdetails\b', r'\bfacts\b'
            ],
            QuestionIntent.DECISION_SUPPORT: [
                r'\bshould\s+i\b', r'\brecommend\b', r'\badvise\b', r'\bchoose\b',
                r'\bdecision\b', r'\boption\b', r'\bbest\b'
            ],
            QuestionIntent.COMPLIANCE_CHECK: [
                r'\bcompliance\b', r'\bcompliant\b', r'\brequirements\b', r'\bmust\b',
                r'\bregulation\b', r'\blegal\b', r'\ballowed\b'
            ],
            QuestionIntent.RISK_ASSESSMENT: [
                r'\brisk\b', r'\brisky\b', r'\bdanger\b', r'\bproblem\b',
                r'\bissue\b', r'\bconcern\b', r'\bliability\b'
            ],
            QuestionIntent.VERIFICATION: [
                r'\bverify\b', r'\bconfirm\b', r'\bcheck\b', r'\bis\s+it\s+true\b',
                r'\baccurate\b', r'\bcorrect\b'
            ],
            QuestionIntent.EXPLANATION: [
                r'\bexplain\b', r'\bwhy\b', r'\bhow\s+does\b', r'\bunderstand\b',
                r'\bclarify\b', r'\breason\b'
            ],
            QuestionIntent.COMPARISON: [
                r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b', r'\bsimilar\b'
            ],
            QuestionIntent.PLANNING: [
                r'\bplan\b', r'\bstrategy\b', r'\bapproach\b', r'\bnext\s+steps\b'
            ]
        }
    
    def _build_complexity_indicators(self) -> Dict[ComplexityLevel, List[str]]:
        """Build indicators for complexity assessment"""
        return {
            ComplexityLevel.SIMPLE: [
                r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhen\s+is\b'
            ],
            ComplexityLevel.MODERATE: [
                r'\bhow\b', r'\bwhy\b', r'\bcompare\b', r'\blist\b'
            ],
            ComplexityLevel.COMPLEX: [
                r'\banalyze\b', r'\bevaluate\b', r'\bassess\b', r'\bimplications\b',
                r'\bimpact\b', r'\bconsequences\b'
            ],
            ComplexityLevel.EXPERT: [
                r'\bstrategic\b', r'\bcomprehensive\b', r'\bin-depth\b',
                r'\bdetailed\s+analysis\b', r'\bexpert\b'
            ]
        }
    
    def process_question(self, question: str) -> QuestionAnalysis:
        """Process question and return comprehensive analysis"""
        with Timer() as timer:
            # Clean and normalize the question
            normalized_question = clean_text(question)
            
            # Generate unique analysis ID
            analysis_id = generate_id("qa")
            
            # Create analysis object
            analysis = QuestionAnalysis(
                original_question=question,
                normalized_question=normalized_question
            )
            
            # Classify question type
            analysis.question_type = self._classify_question_type(normalized_question)
            
            # Determine intent
            analysis.intent = self._determine_intent(normalized_question)
            
            # Assess complexity
            analysis.complexity = self._assess_complexity(normalized_question)
            
            # Extract entities
            analysis.entities = self._extract_entities(normalized_question)
            
            # Extract keywords
            analysis.keywords = extract_keywords(normalized_question)
            
            # Extract legal concepts
            analysis.legal_concepts = extract_legal_concepts(normalized_question)
            
            # Calculate overall confidence
            analysis.confidence = self._calculate_analysis_confidence(analysis)
            
            # Record processing time
            analysis.processing_time = timer.elapsed()
            
            logger.info(f"Processed question: {question[:100]}... -> {analysis.question_type.value}")
            
            return analysis
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        type_scores = {}
        
        for question_type, patterns in self.question_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    score += 1
            
            if score > 0:
                type_scores[question_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return QuestionType.FACTUAL  # Default
    
    def _determine_intent(self, question: str) -> QuestionIntent:
        """Determine the intent behind the question"""
        question_lower = question.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return QuestionIntent.INFORMATION_SEEKING  # Default
    
    def _assess_complexity(self, question: str) -> ComplexityLevel:
        """Assess the complexity level of the question"""
        question_lower = question.lower()
        complexity_scores = {}
        
        for complexity, indicators in self.complexity_indicators.items():
            score = 0
            for indicator in indicators:
                if re.search(indicator, question_lower):
                    score += 1
            
            if score > 0:
                complexity_scores[complexity] = score
        
        # Additional complexity factors
        word_count = len(question.split())
        if word_count > 20:
            complexity_scores[ComplexityLevel.COMPLEX] = complexity_scores.get(ComplexityLevel.COMPLEX, 0) + 1
        elif word_count > 30:
            complexity_scores[ComplexityLevel.EXPERT] = complexity_scores.get(ComplexityLevel.EXPERT, 0) + 1
        
        # Check for multiple clauses
        if question.count(',') > 2 or question.count('and') > 1:
            complexity_scores[ComplexityLevel.MODERATE] = complexity_scores.get(ComplexityLevel.MODERATE, 0) + 1
        
        if complexity_scores:
            return max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return ComplexityLevel.SIMPLE  # Default
    
    def _extract_entities(self, question: str) -> List[Entity]:
        """Extract entities from the question"""
        raw_entities = extract_legal_entities(question)
        entities = []
        
        for raw_entity in raw_entities:
            entity = Entity(
                text=raw_entity["text"],
                entity_type=raw_entity["entity_type"],
                start_pos=raw_entity["start_pos"],
                end_pos=raw_entity["end_pos"],
                confidence=raw_entity["confidence"]
            )
            entities.append(entity)
        
        return entities
    
    def _calculate_analysis_confidence(self, analysis: QuestionAnalysis) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = {}
        
        # Entity extraction confidence
        if analysis.entities:
            avg_entity_confidence = sum(e.confidence for e in analysis.entities) / len(analysis.entities)
            confidence_factors["entities"] = avg_entity_confidence
        else:
            confidence_factors["entities"] = 0.5  # Neutral if no entities
        
        # Keywords confidence (based on count and relevance)
        keyword_confidence = min(1.0, len(analysis.keywords) / 10)
        confidence_factors["keywords"] = keyword_confidence
        
        # Legal concepts confidence
        concept_confidence = min(1.0, len(analysis.legal_concepts) / 5)
        confidence_factors["concepts"] = concept_confidence
        
        # Question clarity (based on length and structure)
        question_length = len(analysis.original_question.split())
        if 5 <= question_length <= 25:
            clarity_confidence = 0.9
        elif 3 <= question_length <= 50:
            clarity_confidence = 0.7
        else:
            clarity_confidence = 0.5
        
        confidence_factors["clarity"] = clarity_confidence
        
        # Overall confidence calculation
        weights = {
            "entities": 0.2,
            "keywords": 0.2,
            "concepts": 0.3,
            "clarity": 0.3
        }
        
        return calculate_confidence_score(confidence_factors, weights)
    
    def get_search_terms(self, analysis: QuestionAnalysis) -> List[str]:
        """Extract optimal search terms from question analysis"""
        search_terms = []
        
        # Add entity text
        for entity in analysis.entities:
            search_terms.append(entity.text)
        
        # Add important keywords (filter out very common ones)
        common_words = {'what', 'how', 'when', 'where', 'why', 'which', 'who'}
        filtered_keywords = [kw for kw in analysis.keywords if kw.lower() not in common_words]
        search_terms.extend(filtered_keywords[:10])  # Top 10 keywords
        
        # Add legal concepts
        search_terms.extend(analysis.legal_concepts)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in search_terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms[:20]  # Limit to top 20 terms
    
    def suggest_clarifications(self, analysis: QuestionAnalysis) -> List[str]:
        """Suggest clarifications if question is ambiguous"""
        suggestions = []
        
        # Low confidence suggestions
        if analysis.confidence < 0.6:
            suggestions.append("Could you provide more specific details about what you're looking for?")
        
        # Complex questions without entities
        if analysis.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT] and not analysis.entities:
            suggestions.append("It would be helpful to specify particular documents, clauses, or legal concepts you're interested in.")
        
        # Comparative questions without clear comparison targets
        if analysis.question_type == QuestionType.COMPARATIVE:
            if len(analysis.entities) < 2:
                suggestions.append("For comparison questions, please specify what documents or terms you want to compare.")
        
        # Quantitative questions without amount indicators
        if analysis.question_type == QuestionType.QUANTITATIVE and not any(e.entity_type == "money" for e in analysis.entities):
            suggestions.append("Could you clarify what specific amounts, percentages, or quantities you're asking about?")
        
        return suggestions