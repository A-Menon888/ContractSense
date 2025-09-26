"""
Risk Aggregator Module

Aggregates individual clause risk assessments into document-level risk scores
and provides portfolio-wide risk analysis capabilities.
"""

import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .risk_calculator import ClauseRiskAssessment, DocumentRiskAssessment, RiskFactor
from .risk_config import RiskLevel, RiskCalculationConfig

logger = logging.getLogger(__name__)

@dataclass
class RiskDistribution:
    """Risk distribution across different categories"""
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    
    @property
    def total(self) -> int:
        return self.critical + self.high + self.medium + self.low
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "critical": self.critical,
            "high": self.high,
            "medium": self.medium,
            "low": self.low,
            "total": self.total
        }

@dataclass
class PortfolioRiskSummary:
    """Summary of risk across a portfolio of contracts"""
    total_documents: int
    total_clauses: int
    average_risk_score: float
    risk_distribution: RiskDistribution
    top_risk_factors: List[Dict[str, Any]]
    industry_benchmarks: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_clauses": self.total_clauses,
            "average_risk_score": self.average_risk_score,
            "risk_distribution": self.risk_distribution.to_dict(),
            "top_risk_factors": self.top_risk_factors,
            "industry_benchmarks": self.industry_benchmarks
        }

class RiskAggregator:
    """Aggregates clause-level risks into document and portfolio level insights"""
    
    def __init__(self, config: RiskCalculationConfig = None):
        self.config = config or RiskCalculationConfig()
    
    def aggregate_document_risk(self, 
                              document_id: str,
                              clause_assessments: List[ClauseRiskAssessment],
                              processing_time: float = 0.0) -> DocumentRiskAssessment:
        """Aggregate clause risk assessments into document-level risk assessment"""
        
        if not clause_assessments:
            return self._create_empty_assessment(document_id, processing_time)
        
        # Calculate overall risk score using multiple aggregation methods
        risk_scores = [assessment.risk_score for assessment in clause_assessments]
        
        # Primary method: Weighted average with risk level weighting
        overall_risk = self._calculate_weighted_risk_score(clause_assessments)
        
        # Alternative methods for comparison
        mean_risk = statistics.mean(risk_scores)
        median_risk = statistics.median(risk_scores)
        max_risk = max(risk_scores)
        
        # Use weighted average but cap based on maximum risk
        overall_risk = min(overall_risk, max_risk * 1.1)
        
        # Calculate confidence as average of clause confidences
        confidences = [assessment.confidence for assessment in clause_assessments]
        overall_confidence = statistics.mean(confidences)
        
        # Aggregate dimensional scores
        dimensional_scores = self._aggregate_dimensional_scores(clause_assessments)
        
        # Calculate risk distribution
        risk_distribution = self._calculate_risk_distribution(clause_assessments)
        
        # Determine overall risk level
        overall_risk_level = self.config.thresholds.get_risk_level(overall_risk)
        
        return DocumentRiskAssessment(
            document_id=document_id,
            overall_risk_score=round(overall_risk, 2),
            overall_risk_level=overall_risk_level,
            confidence=round(overall_confidence, 3),
            clause_assessments=clause_assessments,
            dimensional_scores={k: round(v, 2) for k, v in dimensional_scores.items()},
            risk_distribution=risk_distribution,
            processing_time=processing_time
        )
    
    def _calculate_weighted_risk_score(self, clause_assessments: List[ClauseRiskAssessment]) -> float:
        """Calculate weighted average risk score with higher weights for high-risk clauses"""
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for assessment in clause_assessments:
            # Calculate weight based on risk level and confidence
            weight = self._calculate_clause_weight(assessment)
            
            total_weighted_score += assessment.risk_score * weight
            total_weights += weight
        
        return total_weighted_score / total_weights if total_weights > 0 else 0.0
    
    def _calculate_clause_weight(self, assessment: ClauseRiskAssessment) -> float:
        """Calculate weight for a clause based on its characteristics"""
        base_weight = 1.0
        
        # Weight based on risk level
        risk_level_weights = {
            RiskLevel.CRITICAL: 2.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.LOW: 0.7
        }
        
        risk_weight = risk_level_weights.get(assessment.risk_level, 1.0)
        
        # Weight based on confidence (more confident assessments get higher weight)
        confidence_weight = 0.5 + (assessment.confidence * 0.5)  # Scale from 0.5 to 1.0
        
        # Weight based on clause type importance
        clause_type_weights = {
            "limitation_of_liability": 1.8,
            "indemnification": 1.7,
            "intellectual_property": 1.6,
            "termination": 1.5,
            "payment": 1.4,
            "confidentiality": 1.3,
            "warranties": 1.2,
            "governing_law": 1.1,
            "force_majeure": 1.0,
            "definitions": 0.8
        }
        
        type_weight = clause_type_weights.get(assessment.clause_type, 1.0)
        
        return base_weight * risk_weight * confidence_weight * type_weight
    
    def _aggregate_dimensional_scores(self, clause_assessments: List[ClauseRiskAssessment]) -> Dict[str, float]:
        """Aggregate dimensional risk scores across all clauses"""
        dimensions = ["financial", "legal", "operational", "strategic"]
        aggregated_scores = {}
        
        for dimension in dimensions:
            dimension_scores = []
            dimension_weights = []
            
            for assessment in clause_assessments:
                if dimension in assessment.dimensional_scores:
                    score = assessment.dimensional_scores[dimension]
                    weight = self._calculate_clause_weight(assessment)
                    
                    dimension_scores.append(score * weight)
                    dimension_weights.append(weight)
            
            if dimension_weights:
                aggregated_scores[dimension] = sum(dimension_scores) / sum(dimension_weights)
            else:
                aggregated_scores[dimension] = 0.0
        
        return aggregated_scores
    
    def _calculate_risk_distribution(self, clause_assessments: List[ClauseRiskAssessment]) -> Dict[str, int]:
        """Calculate distribution of clauses across risk levels"""
        distribution = RiskDistribution()
        
        for assessment in clause_assessments:
            if assessment.risk_level == RiskLevel.CRITICAL:
                distribution.critical += 1
            elif assessment.risk_level == RiskLevel.HIGH:
                distribution.high += 1
            elif assessment.risk_level == RiskLevel.MEDIUM:
                distribution.medium += 1
            else:
                distribution.low += 1
        
        return distribution.to_dict()
    
    def _create_empty_assessment(self, document_id: str, processing_time: float) -> DocumentRiskAssessment:
        """Create empty assessment for documents with no clauses"""
        return DocumentRiskAssessment(
            document_id=document_id,
            overall_risk_score=0.0,
            overall_risk_level=RiskLevel.LOW,
            confidence=0.0,
            clause_assessments=[],
            dimensional_scores={"financial": 0.0, "legal": 0.0, "operational": 0.0, "strategic": 0.0},
            risk_distribution={"critical": 0, "high": 0, "medium": 0, "low": 0},
            processing_time=processing_time
        )
    
    def analyze_portfolio_risk(self, 
                             document_assessments: List[DocumentRiskAssessment]) -> PortfolioRiskSummary:
        """Analyze risk across a portfolio of contract documents"""
        
        if not document_assessments:
            return PortfolioRiskSummary(
                total_documents=0,
                total_clauses=0,
                average_risk_score=0.0,
                risk_distribution=RiskDistribution(),
                top_risk_factors=[]
            )
        
        # Portfolio statistics
        total_documents = len(document_assessments)
        total_clauses = sum(len(doc.clause_assessments) for doc in document_assessments)
        
        # Calculate average risk score
        risk_scores = [doc.overall_risk_score for doc in document_assessments]
        average_risk_score = statistics.mean(risk_scores) if risk_scores else 0.0
        
        # Aggregate risk distribution
        portfolio_distribution = RiskDistribution()
        for doc in document_assessments:
            doc_dist = doc.risk_distribution
            portfolio_distribution.critical += doc_dist.get("critical", 0)
            portfolio_distribution.high += doc_dist.get("high", 0)
            portfolio_distribution.medium += doc_dist.get("medium", 0)
            portfolio_distribution.low += doc_dist.get("low", 0)
        
        # Analyze top risk factors across portfolio
        top_risk_factors = self._analyze_top_risk_factors(document_assessments)
        
        return PortfolioRiskSummary(
            total_documents=total_documents,
            total_clauses=total_clauses,
            average_risk_score=round(average_risk_score, 2),
            risk_distribution=portfolio_distribution,
            top_risk_factors=top_risk_factors
        )
    
    def _analyze_top_risk_factors(self, 
                                 document_assessments: List[DocumentRiskAssessment],
                                 top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify the most common and impactful risk factors across portfolio"""
        
        # Collect all risk factors with their frequencies and impacts
        factor_analysis = {}
        
        for doc_assessment in document_assessments:
            for clause_assessment in doc_assessment.clause_assessments:
                for risk_factor in clause_assessment.risk_factors:
                    factor_type = risk_factor.factor_type
                    
                    if factor_type not in factor_analysis:
                        factor_analysis[factor_type] = {
                            "count": 0,
                            "total_impact": 0.0,
                            "max_impact": 0.0,
                            "descriptions": [],
                            "severity_counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
                        }
                    
                    stats = factor_analysis[factor_type]
                    stats["count"] += 1
                    stats["total_impact"] += risk_factor.impact_score
                    stats["max_impact"] = max(stats["max_impact"], risk_factor.impact_score)
                    stats["severity_counts"][risk_factor.severity] += 1
                    
                    # Store unique descriptions
                    if risk_factor.description not in stats["descriptions"]:
                        stats["descriptions"].append(risk_factor.description)
        
        # Get total clauses count for percentage calculations
        total_clauses_in_portfolio = sum(len(doc.clause_assessments) for doc in document_assessments)
        
        # Calculate composite risk score for each factor type
        top_factors = []
        for factor_type, stats in factor_analysis.items():
            # Composite score based on frequency, impact, and severity
            frequency_score = min(stats["count"] / len(document_assessments), 1.0) * 3  # Max 3 points
            impact_score = stats["total_impact"] / stats["count"] if stats["count"] > 0 else 0  # Average impact
            severity_score = (
                stats["severity_counts"]["CRITICAL"] * 3 +
                stats["severity_counts"]["HIGH"] * 2 +
                stats["severity_counts"]["MEDIUM"] * 1 +
                stats["severity_counts"]["LOW"] * 0.5
            ) / stats["count"] if stats["count"] > 0 else 0
            
            composite_score = frequency_score + impact_score + severity_score
            
            top_factors.append({
                "factor_type": factor_type,
                "frequency": stats["count"],
                "frequency_percentage": round((stats["count"] / total_clauses_in_portfolio) * 100, 1) if total_clauses_in_portfolio > 0 else 0,
                "average_impact": round(stats["total_impact"] / stats["count"], 2) if stats["count"] > 0 else 0,
                "max_impact": round(stats["max_impact"], 2),
                "composite_score": round(composite_score, 2),
                "severity_distribution": stats["severity_counts"],
                "sample_description": stats["descriptions"][0] if stats["descriptions"] else ""
            })
        
        # Sort by composite score and return top N
        top_factors.sort(key=lambda x: x["composite_score"], reverse=True)
        return top_factors[:top_n]
    
    def compare_with_benchmarks(self, 
                              portfolio_summary: PortfolioRiskSummary,
                              industry_benchmarks: Dict[str, float]) -> Dict[str, Any]:
        """Compare portfolio risk metrics with industry benchmarks"""
        
        comparison = {
            "portfolio_score": portfolio_summary.average_risk_score,
            "industry_average": industry_benchmarks.get("average_risk_score", 5.0),
            "percentile": None,
            "recommendations": []
        }
        
        # Calculate percentile ranking
        industry_avg = industry_benchmarks.get("average_risk_score", 5.0)
        if portfolio_summary.average_risk_score < industry_avg:
            comparison["percentile"] = "Below Average (Better)"
            comparison["recommendations"].append("Portfolio risk is below industry average - good risk management")
        elif portfolio_summary.average_risk_score > industry_avg * 1.2:
            comparison["percentile"] = "Above Average (Higher Risk)"
            comparison["recommendations"].append("Portfolio risk is significantly above industry average - review needed")
        else:
            comparison["percentile"] = "Average"
            comparison["recommendations"].append("Portfolio risk is in line with industry standards")
        
        # Compare risk distribution
        if portfolio_summary.risk_distribution.critical > 0:
            comparison["recommendations"].append(f"High priority: Address {portfolio_summary.risk_distribution.critical} critical risk clauses")
        
        if portfolio_summary.risk_distribution.high > portfolio_summary.total_clauses * 0.2:
            comparison["recommendations"].append("Consider reviewing contract templates to reduce high-risk clause frequency")
        
        return comparison