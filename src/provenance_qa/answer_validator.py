"""
Answer Validation Component

Validates generated answers for accuracy, completeness, and reliability.
Provides quality assessment and recommendations for answer improvement.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    from ..models.answer_models import Answer, ValidationResult, ValidationCheck, ValidationStatus
    from ..models.context_models import ContextWindow
    from ..models.question_models import QuestionAnalysis
    from ..models.provenance_models import QAResponse
    from ..utils.common import calculate_confidence_score, Timer
except ImportError:
    # Fallback for direct execution
    from provenance_qa.models.answer_models import Answer, ValidationResult, ValidationCheck, ValidationStatus
    from provenance_qa.models.context_models import ContextWindow
    from provenance_qa.models.question_models import QuestionAnalysis
    from provenance_qa.models.provenance_models import QAResponse
    from provenance_qa.utils.common import calculate_confidence_score, Timer

logger = logging.getLogger(__name__)

class AnswerValidator:
    """Comprehensive answer validation and quality assessment"""
    
    def __init__(self):
        self.validation_checks = self._build_validation_checks()
        self.quality_thresholds = self._build_quality_thresholds()
    
    def _build_validation_checks(self) -> Dict[str, callable]:
        """Build available validation check methods"""
        return {
            "factual_accuracy": self._check_factual_accuracy,
            "source_reliability": self._check_source_reliability,
            "logical_consistency": self._check_logical_consistency,
            "completeness": self._check_completeness,
            "relevance": self._check_relevance,
            "clarity": self._check_clarity,
            "citation_quality": self._check_citation_quality,
            "confidence_alignment": self._check_confidence_alignment
        }
    
    def _build_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Build quality thresholds for different validation aspects"""
        return {
            "factual_accuracy": {"excellent": 0.85, "good": 0.65, "acceptable": 0.45, "poor": 0.25},
            "source_reliability": {"excellent": 0.80, "good": 0.60, "acceptable": 0.40, "poor": 0.25},
            "logical_consistency": {"excellent": 0.85, "good": 0.65, "acceptable": 0.45, "poor": 0.25},
            "completeness": {"excellent": 0.80, "good": 0.60, "acceptable": 0.40, "poor": 0.25},
            "relevance": {"excellent": 0.85, "good": 0.65, "acceptable": 0.45, "poor": 0.25},
            "clarity": {"excellent": 0.80, "good": 0.60, "acceptable": 0.40, "poor": 0.25}
        }
    
    def validate_answer(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationResult:
        """Perform comprehensive answer validation"""
        
        with Timer() as timer:
            # Create validation result
            validation_result = ValidationResult(answer_id=answer.answer_id)
            
            # Run all validation checks
            for check_name, check_method in self.validation_checks.items():
                try:
                    validation_check = check_method(
                        answer, question_analysis, context_window, qa_response
                    )
                    validation_result.add_check(validation_check)
                    
                except Exception as e:
                    logger.error(f"Error in validation check {check_name}: {str(e)}")
                    # Add failed check
                    failed_check = ValidationCheck(
                        check_name=check_name,
                        status=ValidationStatus.FAILED,
                        score=0.0,
                        message=f"Validation check failed: {str(e)}"
                    )
                    validation_result.add_check(failed_check)
            
            # Calculate overall validation metrics
            self._calculate_overall_metrics(validation_result)
            
            # Determine overall status
            validation_result.overall_status = self._determine_overall_status(validation_result)
            
            # Generate recommendations
            validation_result.recommendations = self._generate_recommendations(validation_result)
            
            # Record validation time
            validation_result.validation_time = timer.elapsed()
            
            logger.info(f"Validated answer: {validation_result.overall_status.value} "
                       f"(score: {validation_result.overall_score:.2f})")
            
            return validation_result
    
    def _check_factual_accuracy(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check factual accuracy of the answer against source context"""
        
        check = ValidationCheck(check_name="factual_accuracy", status=ValidationStatus.PENDING)
        
        try:
            accuracy_factors = {}
            
            # Check alignment with context
            all_chunks = context_window.get_all_chunks()
            if all_chunks:
                # Calculate similarity to source material
                try:
                    from ..utils.common import calculate_text_similarity
                except ImportError:
                    from provenance_qa.utils.common import calculate_text_similarity
                all_context = " ".join(chunk.content for chunk in all_chunks)
                context_alignment = calculate_text_similarity(answer.text, all_context)
                accuracy_factors["context_alignment"] = context_alignment
            else:
                accuracy_factors["context_alignment"] = 0.0
                check.message = "No context available for fact-checking"
            
            # Check for contradictory statements
            contradiction_score = self._detect_contradictions(answer.text, all_chunks)
            accuracy_factors["no_contradictions"] = 1.0 - contradiction_score
            
            # Check for unsupported claims
            unsupported_score = self._detect_unsupported_claims(answer.text, all_chunks)
            accuracy_factors["supported_claims"] = 1.0 - unsupported_score
            
            # Check citation accuracy if available
            if qa_response and qa_response.citations:
                citation_accuracy = self._check_citation_accuracy(answer.text, qa_response.citations)
                accuracy_factors["citation_accuracy"] = citation_accuracy
            
            # Calculate overall factual accuracy
            weights = {
                "context_alignment": 0.4,
                "no_contradictions": 0.3,
                "supported_claims": 0.2,
                "citation_accuracy": 0.1
            }
            
            # Only use weights for factors that exist
            available_factors = {k: v for k, v in accuracy_factors.items() if k in weights}
            available_weights = {k: weights[k] for k in available_factors}
            
            if available_factors:
                check.score = calculate_confidence_score(available_factors, available_weights)
            else:
                check.score = 0.0
            
            # Determine status
            if check.score >= self.quality_thresholds["factual_accuracy"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Answer is factually accurate based on available sources"
            elif check.score >= self.quality_thresholds["factual_accuracy"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Answer appears mostly accurate but some concerns detected"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Significant factual accuracy concerns detected"
            
            check.details = accuracy_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking factual accuracy: {str(e)}"
        
        return check
    
    def _check_source_reliability(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check reliability of sources used in answer generation"""
        
        check = ValidationCheck(check_name="source_reliability", status=ValidationStatus.PENDING)
        
        try:
            reliability_factors = {}
            
            all_chunks = context_window.get_all_chunks()
            
            if all_chunks:
                # Average relevance score of sources
                avg_relevance = sum(chunk.relevance_score for chunk in all_chunks) / len(all_chunks)
                reliability_factors["source_relevance"] = avg_relevance
                
                # Diversity of sources (more sources = more reliable)
                unique_docs = len(context_window.get_unique_documents())
                source_diversity = min(1.0, unique_docs / 3)  # Normalize to 3 sources
                reliability_factors["source_diversity"] = source_diversity
                
                # Quality of retrieval methods used
                retrieval_quality = self._assess_retrieval_quality(all_chunks)
                reliability_factors["retrieval_quality"] = retrieval_quality
                
                # Consistency across sources
                source_consistency = self._assess_source_consistency(all_chunks)
                reliability_factors["source_consistency"] = source_consistency
                
            else:
                reliability_factors["source_relevance"] = 0.0
                reliability_factors["source_diversity"] = 0.0
                reliability_factors["retrieval_quality"] = 0.0
                reliability_factors["source_consistency"] = 0.0
            
            # Calculate overall reliability
            check.score = calculate_confidence_score(reliability_factors)
            
            # Determine status
            if check.score >= self.quality_thresholds["source_reliability"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Sources are reliable and well-suited for the answer"
            elif check.score >= self.quality_thresholds["source_reliability"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Sources are acceptable but could be improved"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Source reliability concerns detected"
            
            check.details = reliability_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking source reliability: {str(e)}"
        
        return check
    
    def _check_logical_consistency(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check logical consistency of the answer"""
        
        check = ValidationCheck(check_name="logical_consistency", status=ValidationStatus.PENDING)
        
        try:
            consistency_factors = {}
            
            # Internal consistency (no contradictory statements within answer)
            internal_consistency = self._check_internal_consistency(answer.text)
            consistency_factors["internal_consistency"] = internal_consistency
            
            # Consistency with question type
            type_consistency = self._check_answer_type_consistency(answer, question_analysis)
            consistency_factors["type_consistency"] = type_consistency
            
            # Logical flow and structure
            logical_flow = self._assess_logical_flow(answer.text)
            consistency_factors["logical_flow"] = logical_flow
            
            # Evidence-conclusion alignment
            evidence_alignment = self._assess_evidence_conclusion_alignment(answer, context_window)
            consistency_factors["evidence_alignment"] = evidence_alignment
            
            # Calculate overall consistency
            check.score = calculate_confidence_score(consistency_factors)
            
            # Determine status
            if check.score >= self.quality_thresholds["logical_consistency"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Answer is logically consistent and well-reasoned"
            elif check.score >= self.quality_thresholds["logical_consistency"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Answer is mostly consistent with minor logical issues"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Logical consistency issues detected"
            
            check.details = consistency_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking logical consistency: {str(e)}"
        
        return check
    
    def _check_completeness(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check completeness of the answer"""
        
        check = ValidationCheck(check_name="completeness", status=ValidationStatus.PENDING)
        
        try:
            completeness_factors = {}
            
            # Question entity coverage
            if question_analysis.entities:
                answer_lower = answer.text.lower()
                covered_entities = sum(1 for entity in question_analysis.entities 
                                     if entity.text.lower() in answer_lower)
                entity_coverage = covered_entities / len(question_analysis.entities)
                completeness_factors["entity_coverage"] = entity_coverage
            else:
                completeness_factors["entity_coverage"] = 0.8  # Neutral if no entities
            
            # Key concept coverage
            if question_analysis.legal_concepts:
                answer_lower = answer.text.lower()
                covered_concepts = sum(1 for concept in question_analysis.legal_concepts 
                                     if concept.lower() in answer_lower)
                concept_coverage = covered_concepts / len(question_analysis.legal_concepts)
                completeness_factors["concept_coverage"] = concept_coverage
            else:
                completeness_factors["concept_coverage"] = 0.8
            
            # Answer length appropriateness
            length_score = self._assess_answer_length_appropriateness(answer, question_analysis)
            completeness_factors["length_appropriateness"] = length_score
            
            # Key points extraction
            key_points_score = min(1.0, len(answer.key_points) / 3)  # Expect at least 3 key points
            completeness_factors["key_points"] = key_points_score
            
            # Calculate overall completeness
            check.score = calculate_confidence_score(completeness_factors)
            
            # Determine status
            if check.score >= self.quality_thresholds["completeness"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Answer is comprehensive and complete"
            elif check.score >= self.quality_thresholds["completeness"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Answer is mostly complete but some aspects could be expanded"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Answer appears incomplete"
            
            check.details = completeness_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking completeness: {str(e)}"
        
        return check
    
    def _check_relevance(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check relevance of answer to the question"""
        
        check = ValidationCheck(check_name="relevance", status=ValidationStatus.PENDING)
        
        try:
            # Use the existing relevance score from answer
            check.score = answer.relevance_score
            
            relevance_factors = {
                "answer_relevance_score": answer.relevance_score,
                "question_type_alignment": self._check_answer_type_consistency(answer, question_analysis)
            }
            
            # Determine status
            if check.score >= self.quality_thresholds["relevance"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Answer is highly relevant to the question"
            elif check.score >= self.quality_thresholds["relevance"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Answer is relevant but could be more focused"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Answer relevance concerns detected"
            
            check.details = relevance_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking relevance: {str(e)}"
        
        return check
    
    def _check_clarity(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check clarity and readability of the answer"""
        
        check = ValidationCheck(check_name="clarity", status=ValidationStatus.PENDING)
        
        try:
            # Use the existing clarity score from answer
            check.score = answer.clarity_score
            
            clarity_factors = {
                "clarity_score": answer.clarity_score,
                "structure_quality": 1.0 if answer.key_points else 0.6
            }
            
            # Determine status
            if check.score >= self.quality_thresholds["clarity"]["good"]:
                check.status = ValidationStatus.VALIDATED
                check.message = "Answer is clear and well-structured"
            elif check.score >= self.quality_thresholds["clarity"]["acceptable"]:
                check.status = ValidationStatus.WARNING
                check.message = "Answer clarity could be improved"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Answer clarity issues detected"
            
            check.details = clarity_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking clarity: {str(e)}"
        
        return check
    
    def _check_citation_quality(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check quality of citations if available"""
        
        check = ValidationCheck(check_name="citation_quality", status=ValidationStatus.PENDING)
        
        try:
            if qa_response and qa_response.citations:
                citation_factors = {}
                
                # Average citation quality
                avg_citation_quality = sum(c.quality_score for c in qa_response.citations) / len(qa_response.citations)
                citation_factors["average_quality"] = avg_citation_quality
                
                # Citation relevance
                avg_citation_relevance = sum(c.relevance_score for c in qa_response.citations) / len(qa_response.citations)
                citation_factors["average_relevance"] = avg_citation_relevance
                
                # Citation completeness (location info, etc.)
                complete_citations = sum(1 for c in qa_response.citations 
                                       if c.section_title and c.page_number)
                citation_completeness = complete_citations / len(qa_response.citations)
                citation_factors["completeness"] = citation_completeness
                
                check.score = calculate_confidence_score(citation_factors)
                check.details = citation_factors
                
                if check.score >= 0.75:
                    check.status = ValidationStatus.VALIDATED
                    check.message = "Citations are high quality and well-documented"
                elif check.score >= 0.6:
                    check.status = ValidationStatus.WARNING
                    check.message = "Citations are acceptable but could be improved"
                else:
                    check.status = ValidationStatus.FAILED
                    check.message = "Citation quality concerns detected"
                    
            else:
                check.status = ValidationStatus.SKIPPED
                check.score = 0.0
                check.message = "No citations available to validate"
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking citation quality: {str(e)}"
        
        return check
    
    def _check_confidence_alignment(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow,
        qa_response: Optional[QAResponse] = None
    ) -> ValidationCheck:
        """Check if confidence level aligns with answer quality"""
        
        check = ValidationCheck(check_name="confidence_alignment", status=ValidationStatus.PENDING)
        
        try:
            actual_quality = answer.get_overall_quality_score()
            stated_confidence = answer.confidence
            
            # Check alignment between confidence and quality
            confidence_diff = abs(actual_quality - stated_confidence)
            alignment_score = max(0.0, 1.0 - confidence_diff)
            
            check.score = alignment_score
            
            alignment_factors = {
                "confidence_quality_alignment": alignment_score,
                "actual_quality": actual_quality,
                "stated_confidence": stated_confidence
            }
            
            if alignment_score >= 0.8:
                check.status = ValidationStatus.VALIDATED
                check.message = "Confidence level appropriately reflects answer quality"
            elif alignment_score >= 0.6:
                check.status = ValidationStatus.WARNING
                check.message = "Minor misalignment between confidence and quality"
            else:
                check.status = ValidationStatus.FAILED
                check.message = "Significant misalignment between confidence and quality"
            
            check.details = alignment_factors
            
        except Exception as e:
            check.status = ValidationStatus.FAILED
            check.score = 0.0
            check.message = f"Error checking confidence alignment: {str(e)}"
        
        return check
    
    # Helper methods for validation checks
    
    def _detect_contradictions(self, answer_text: str, chunks: List) -> float:
        """Detect contradictory statements (simplified implementation)"""
        # This is a simplified implementation
        # In practice, would use more sophisticated NLP techniques
        
        contradiction_indicators = [
            "however", "but", "although", "nevertheless", "on the contrary",
            "in contrast", "unlike", "whereas", "while"
        ]
        
        answer_lower = answer_text.lower()
        contradiction_count = sum(1 for indicator in contradiction_indicators 
                                if indicator in answer_lower)
        
        # Normalize by answer length
        contradiction_score = min(1.0, contradiction_count / 3)
        return contradiction_score
    
    def _detect_unsupported_claims(self, answer_text: str, chunks: List) -> float:
        """Detect claims not supported by context"""
        # Simplified implementation
        # Would use more sophisticated fact-checking in practice
        
        if not chunks:
            return 1.0  # High unsupported score if no context
        
        all_context = " ".join(chunk.content for chunk in chunks)
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        
        similarity = calculate_text_similarity(answer_text, all_context)
        unsupported_score = max(0.0, 1.0 - similarity)
        
        return unsupported_score
    
    def _check_citation_accuracy(self, answer_text: str, citations: List) -> float:
        """Check accuracy of citations"""
        # Simplified implementation
        if not citations:
            return 0.0
        
        # Check if cited content appears in answer
        answer_lower = answer_text.lower()
        accurate_citations = 0
        
        for citation in citations:
            citation_content = citation.cited_text.lower()
            # Simple check for content overlap
            try:
                from ..utils.common import calculate_text_similarity
            except ImportError:
                from provenance_qa.utils.common import calculate_text_similarity
            similarity = calculate_text_similarity(citation_content, answer_lower)
            
            if similarity > 0.3:  # Threshold for relevance
                accurate_citations += 1
        
        return accurate_citations / len(citations) if citations else 0.0
    
    def _assess_retrieval_quality(self, chunks: List) -> float:
        """Assess quality of retrieval methods used"""
        if not chunks:
            return 0.0
        
        # Weight by retrieval method quality
        method_weights = {
            "cross_encoder": 1.0,
            "hybrid": 0.9,
            "vector_search": 0.8,
            "graph_traversal": 0.7,
            "keyword_match": 0.6
        }
        
        weighted_quality = 0.0
        total_weight = 0.0
        
        for chunk in chunks:
            method = chunk.retrieval_source.value
            weight = method_weights.get(method, 0.5)
            weighted_quality += weight * chunk.relevance_score
            total_weight += weight
        
        return weighted_quality / total_weight if total_weight > 0 else 0.0
    
    def _assess_source_consistency(self, chunks: List) -> float:
        """Assess consistency across different sources"""
        if len(chunks) <= 1:
            return 1.0  # Single source is consistent
        
        # Calculate pairwise similarity between chunks
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        
        similarities = []
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                similarity = calculate_text_similarity(chunk1.content, chunk2.content)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _check_internal_consistency(self, answer_text: str) -> float:
        """Check for internal contradictions within answer"""
        # Simplified implementation
        sentences = answer_text.split('.')
        
        if len(sentences) <= 1:
            return 1.0
        
        # Look for contradictory patterns
        contradiction_patterns = [
            ("yes", "no"), ("true", "false"), ("allowed", "prohibited"),
            ("required", "optional"), ("must", "may not")
        ]
        
        answer_lower = answer_text.lower()
        contradictions = 0
        
        for pos, neg in contradiction_patterns:
            if pos in answer_lower and neg in answer_lower:
                contradictions += 1
        
        consistency_score = max(0.0, 1.0 - (contradictions / 3))
        return consistency_score
    
    def _check_answer_type_consistency(self, answer: Answer, question_analysis: QuestionAnalysis) -> float:
        """Check if answer type matches question type"""
        from ..models.question_models import QuestionType
        from ..models.answer_models import AnswerType
        
        # Define expected answer types for question types
        type_mapping = {
            QuestionType.FACTUAL: [AnswerType.DIRECT, AnswerType.SUMMARY],
            QuestionType.COMPARATIVE: [AnswerType.COMPARISON, AnswerType.ANALYSIS],
            QuestionType.ANALYTICAL: [AnswerType.ANALYSIS, AnswerType.EXPLANATION],
            QuestionType.PROCEDURAL: [AnswerType.EXPLANATION, AnswerType.DIRECT],
            QuestionType.DEFINITIONAL: [AnswerType.EXPLANATION, AnswerType.DIRECT]
        }
        
        expected_types = type_mapping.get(question_analysis.question_type, [])
        
        if answer.answer_type in expected_types:
            return 1.0
        elif answer.answer_type == AnswerType.NOT_FOUND:
            return 0.3  # Low but not zero - might be appropriate
        else:
            return 0.6  # Partial credit for other types
    
    def _assess_logical_flow(self, answer_text: str) -> float:
        """Assess logical flow and structure of answer"""
        # Simple structural analysis
        flow_factors = {}
        
        # Presence of transition words
        transition_words = ["first", "then", "next", "finally", "however", "therefore", "thus", "consequently"]
        answer_lower = answer_text.lower()
        transitions = sum(1 for word in transition_words if word in answer_lower)
        
        flow_factors["transitions"] = min(1.0, transitions / 3)
        
        # Sentence length variation (good flow has varied sentences)
        sentences = [s.strip() for s in answer_text.split('.') if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            length_variation = sum(abs(l - avg_length) for l in lengths) / len(lengths)
            flow_factors["variation"] = min(1.0, length_variation / 5)  # Normalize
        else:
            flow_factors["variation"] = 0.0
        
        # Paragraph structure
        paragraphs = answer_text.count('\n\n') + 1
        if len(answer_text) > 200 and paragraphs == 1:
            flow_factors["structure"] = 0.6  # Long text without paragraphs
        else:
            flow_factors["structure"] = 1.0
        
        return calculate_confidence_score(flow_factors)
    
    def _assess_evidence_conclusion_alignment(self, answer: Answer, context_window: ContextWindow) -> float:
        """Assess alignment between evidence and conclusions"""
        # Simplified implementation
        if not context_window.get_all_chunks():
            return 0.5  # Neutral if no context
        
        # Use context relevance as proxy for evidence-conclusion alignment
        return context_window.relevance_score
    
    def _assess_answer_length_appropriateness(self, answer: Answer, question_analysis: QuestionAnalysis) -> float:
        """Assess if answer length is appropriate for question complexity"""
        
        length = len(answer.text)
        complexity = question_analysis.complexity
        
        # Expected length ranges by complexity
        from ..models.question_models import ComplexityLevel
        
        expected_ranges = {
            ComplexityLevel.SIMPLE: (50, 200),
            ComplexityLevel.MODERATE: (100, 400),
            ComplexityLevel.COMPLEX: (200, 800),
            ComplexityLevel.EXPERT: (300, 1200)
        }
        
        min_len, max_len = expected_ranges.get(complexity, (100, 400))
        
        if min_len <= length <= max_len:
            return 1.0
        elif length < min_len:
            return length / min_len
        else:
            return max(0.3, max_len / length)
    
    def _calculate_overall_metrics(self, validation_result: ValidationResult):
        """Calculate overall validation metrics from individual checks"""
        
        if not validation_result.checks:
            validation_result.overall_score = 0.0
            return
        
        # Extract category scores
        category_scores = {}
        
        for check in validation_result.checks:
            if check.check_name in ["factual_accuracy"]:
                validation_result.factual_accuracy = check.score
                category_scores["factual_accuracy"] = check.score
            elif check.check_name in ["source_reliability"]:
                validation_result.source_reliability = check.score
                category_scores["source_reliability"] = check.score
            elif check.check_name in ["logical_consistency"]:
                validation_result.logical_consistency = check.score
                category_scores["logical_consistency"] = check.score
            elif check.check_name in ["completeness"]:
                validation_result.completeness = check.score
                category_scores["completeness"] = check.score
        
        # Calculate overall score
        if category_scores:
            validation_result.overall_score = sum(category_scores.values()) / len(category_scores)
        else:
            validation_result.overall_score = sum(c.score for c in validation_result.checks) / len(validation_result.checks)
    
    def _determine_overall_status(self, validation_result: ValidationResult) -> ValidationStatus:
        """Determine overall validation status"""
        
        failed_checks = validation_result.get_failed_checks()
        warning_checks = validation_result.get_warning_checks()
        
        if failed_checks:
            # Add critical issues
            for check in failed_checks:
                validation_result.critical_issues.append(f"{check.check_name}: {check.message}")
            
            return ValidationStatus.FAILED
        
        elif warning_checks:
            # Add warnings
            for check in warning_checks:
                validation_result.warnings.append(f"{check.check_name}: {check.message}")
            
            return ValidationStatus.WARNING
        
        else:
            return ValidationStatus.VALIDATED
    
    def _generate_recommendations(self, validation_result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Recommendations based on failed checks
        for check in validation_result.get_failed_checks():
            if check.check_name == "factual_accuracy":
                recommendations.append("Review answer against source documents for factual correctness")
            elif check.check_name == "completeness":
                recommendations.append("Expand answer to address all aspects of the question")
            elif check.check_name == "relevance":
                recommendations.append("Focus answer more directly on the specific question asked")
            elif check.check_name == "clarity":
                recommendations.append("Improve answer structure and readability")
        
        # Recommendations based on low scores
        if validation_result.overall_score < 0.6:
            recommendations.append("Consider regenerating answer with different approach")
        
        if validation_result.source_reliability < 0.6:
            recommendations.append("Seek additional or higher-quality source material")
        
        return recommendations