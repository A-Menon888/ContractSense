"""
Financial Risk Analyzer

Analyzes contract clauses for financial risk factors including:
- Payment terms and conditions
- Financial exposure and liability limits
- Cost escalation and hidden fees
- Revenue impact and recognition risks
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FinancialRiskMetrics:
    """Financial risk analysis results"""
    payment_risk_score: float
    liability_risk_score: float
    cost_risk_score: float
    revenue_risk_score: float
    overall_financial_risk: float
    risk_factors: List[Dict[str, Any]]
    confidence: float

class FinancialRiskAnalyzer:
    """Analyzes financial risk aspects of contract clauses"""
    
    # Payment-related risk patterns
    PAYMENT_RISK_PATTERNS = {
        "late_payment_penalties": [
            r"late payment.*penalty|penalty.*late payment",
            r"interest.*overdue|overdue.*interest", 
            r"default interest|interest on default",
            r"penalty.*delay|delay.*penalty"
        ],
        "upfront_payments": [
            r"advance payment|payment in advance",
            r"upfront.*payment|payment.*upfront",
            r"pre-payment|prepayment",
            r"payment upon signing|upon execution.*payment"
        ],
        "payment_acceleration": [
            r"accelerate.*payment|payment.*acceleration",
            r"immediate payment|payment.*immediately",
            r"all amounts.*due.*immediately",
            r"acceleration.*obligations"
        ],
        "complex_payment_terms": [
            r"milestone.*payment|payment.*milestone",
            r"performance.*payment|payment.*performance", 
            r"conditional.*payment|payment.*conditional",
            r"tiered.*payment|payment.*tiered"
        ]
    }
    
    # Liability and financial exposure patterns
    LIABILITY_RISK_PATTERNS = {
        "unlimited_liability": [
            r"unlimited.*liability|liability.*unlimited",
            r"without limit.*liability|liability.*without limit",
            r"no cap.*liability|liability.*no cap",
            r"uncapped.*liability|liability.*uncapped"
        ],
        "broad_indemnification": [
            r"indemnify.*against.*all|all.*claims.*indemnify",
            r"defend.*hold harmless|hold harmless.*defend",
            r"broad.*indemnification|comprehensive.*indemnification",
            r"any.*all.*claims.*arising"
        ],
        "consequential_damages": [
            r"consequential.*damages|indirect.*damages",
            r"special.*damages|incidental.*damages",
            r"punitive.*damages|exemplary.*damages",
            r"loss.*profits|lost.*revenue"
        ],
        "financial_guarantees": [
            r"guarantee.*payment|payment.*guarantee",
            r"financial.*guarantee|guarantee.*financial",
            r"surety.*bond|performance.*bond",
            r"letter.*credit|credit.*facility"
        ]
    }
    
    # Cost-related risk patterns  
    COST_RISK_PATTERNS = {
        "cost_escalation": [
            r"cost.*increase|increase.*cost",
            r"price.*escalation|escalation.*price",
            r"inflation.*adjustment|adjustment.*inflation",
            r"annual.*increase|yearly.*increase"
        ],
        "hidden_costs": [
            r"additional.*charges|charges.*additional",
            r"supplemental.*fees|fees.*supplemental",
            r"out.*pocket.*expenses|expenses.*out.*pocket",
            r"incidental.*costs|costs.*incidental"
        ],
        "cost_reimbursement": [
            r"reimburse.*costs|costs.*reimbursement",
            r"expense.*reimbursement|reimbursement.*expense",
            r"actual.*costs|costs.*incurred",
            r"pass.*through.*costs"
        ],
        "change_order_costs": [
            r"change.*order|additional.*work",
            r"scope.*change|modification.*costs",
            r"extra.*work|work.*beyond.*scope",
            r"time.*materials|materials.*time"
        ]
    }
    
    # Revenue impact patterns
    REVENUE_RISK_PATTERNS = {
        "revenue_sharing": [
            r"revenue.*sharing|sharing.*revenue",
            r"profit.*sharing|sharing.*profit",
            r"royalty.*payment|payment.*royalty",
            r"percentage.*revenue|revenue.*percentage"
        ],
        "revenue_recognition": [
            r"revenue.*recognition|recognition.*revenue",
            r"payment.*milestone|milestone.*payment",
            r"acceptance.*criteria|criteria.*acceptance",
            r"delivery.*payment|payment.*delivery"
        ],
        "minimum_commitments": [
            r"minimum.*purchase|purchase.*minimum",
            r"minimum.*volume|volume.*minimum",
            r"guaranteed.*minimum|minimum.*guaranteed",
            r"take.*pay|pay.*take"
        ],
        "termination_impact": [
            r"termination.*fee|fee.*termination",
            r"early.*termination|termination.*early",
            r"cancellation.*charge|charge.*cancellation",
            r"wind.*down.*costs"
        ]
    }
    
    def __init__(self):
        self.risk_weights = {
            "payment": 0.30,
            "liability": 0.35, 
            "cost": 0.20,
            "revenue": 0.15
        }
    
    def analyze_financial_risk(self, 
                             clause_type: str,
                             clause_text: str,
                             contract_value: Optional[float] = None,
                             contract_duration: Optional[int] = None) -> FinancialRiskMetrics:
        """Perform comprehensive financial risk analysis on a clause"""
        
        # Analyze different financial risk dimensions
        payment_risk, payment_factors = self._analyze_payment_risk(clause_text)
        liability_risk, liability_factors = self._analyze_liability_risk(clause_text)
        cost_risk, cost_factors = self._analyze_cost_risk(clause_text)
        revenue_risk, revenue_factors = self._analyze_revenue_risk(clause_text)
        
        # Apply clause-type specific adjustments
        payment_risk *= self._get_clause_type_multiplier(clause_type, "payment")
        liability_risk *= self._get_clause_type_multiplier(clause_type, "liability")
        cost_risk *= self._get_clause_type_multiplier(clause_type, "cost")  
        revenue_risk *= self._get_clause_type_multiplier(clause_type, "revenue")
        
        # Calculate overall financial risk score
        overall_risk = (
            payment_risk * self.risk_weights["payment"] +
            liability_risk * self.risk_weights["liability"] + 
            cost_risk * self.risk_weights["cost"] +
            revenue_risk * self.risk_weights["revenue"]
        )
        
        # Apply contract context adjustments
        if contract_value:
            overall_risk *= self._get_contract_value_multiplier(contract_value)
        
        if contract_duration:
            overall_risk *= self._get_duration_multiplier(contract_duration)
        
        # Combine all risk factors
        all_factors = payment_factors + liability_factors + cost_factors + revenue_factors
        
        # Calculate confidence based on number of factors detected
        confidence = min(0.6 + (len(all_factors) * 0.08), 0.95)
        
        return FinancialRiskMetrics(
            payment_risk_score=round(payment_risk, 2),
            liability_risk_score=round(liability_risk, 2),
            cost_risk_score=round(cost_risk, 2),
            revenue_risk_score=round(revenue_risk, 2),
            overall_financial_risk=round(overall_risk, 2),
            risk_factors=all_factors,
            confidence=round(confidence, 3)
        )
    
    def _analyze_payment_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze payment-related financial risks"""
        text_lower = clause_text.lower()
        risk_score = 3.0  # Base payment risk
        risk_factors = []
        
        for risk_type, patterns in self.PAYMENT_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_payment_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Payment risk factor detected: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier
                    })
                    break  # Only count each risk type once
        
        # Look for payment amount indicators
        amount_risk = self._analyze_payment_amounts(clause_text)
        risk_score += amount_risk
        
        # Cap the risk score
        risk_score = min(risk_score, 10.0)
        
        return risk_score, risk_factors
    
    def _analyze_liability_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze liability and financial exposure risks"""
        text_lower = clause_text.lower()
        risk_score = 4.0  # Base liability risk
        risk_factors = []
        
        for risk_type, patterns in self.LIABILITY_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_liability_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Liability risk factor detected: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier
                    })
                    break
        
        # Look for liability cap indicators
        cap_adjustment = self._analyze_liability_caps(clause_text)
        risk_score *= cap_adjustment
        
        risk_score = min(risk_score, 10.0)
        
        return risk_score, risk_factors
    
    def _analyze_cost_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze cost escalation and hidden cost risks"""
        text_lower = clause_text.lower()
        risk_score = 2.5  # Base cost risk
        risk_factors = []
        
        for risk_type, patterns in self.COST_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_cost_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Cost risk factor detected: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier
                    })
                    break
        
        risk_score = min(risk_score, 10.0)
        
        return risk_score, risk_factors
    
    def _analyze_revenue_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze revenue impact and recognition risks"""
        text_lower = clause_text.lower()
        risk_score = 3.0  # Base revenue risk
        risk_factors = []
        
        for risk_type, patterns in self.REVENUE_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_revenue_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Revenue risk factor detected: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier
                    })
                    break
        
        risk_score = min(risk_score, 10.0)
        
        return risk_score, risk_factors
    
    def _analyze_payment_amounts(self, clause_text: str) -> float:
        """Analyze payment amounts for risk indicators"""
        text_lower = clause_text.lower()
        amount_risk = 0.0
        
        # Look for large percentage indicators
        if re.search(r"(100|ninety|eighty).*percent", text_lower):
            amount_risk += 1.5
        elif re.search(r"(fifty|60|70).*percent", text_lower):
            amount_risk += 1.0
        
        # Look for upfront payment percentages
        if re.search(r"upfront.*(50|sixty|seventy|80)", text_lower):
            amount_risk += 1.2
        
        return amount_risk
    
    def _analyze_liability_caps(self, clause_text: str) -> float:
        """Analyze liability caps and limitations"""
        text_lower = clause_text.lower()
        
        # Look for liability caps (reduce risk)
        if re.search(r"liability.*limited.*to|limited.*liability.*to", text_lower):
            return 0.7  # Reduce risk if liability is capped
        elif re.search(r"cap.*liability|liability.*cap", text_lower):
            return 0.8
        elif re.search(r"maximum.*liability|liability.*maximum", text_lower):
            return 0.75
        
        # Look for unlimited liability (increase risk)
        if re.search(r"unlimited.*liability|liability.*unlimited", text_lower):
            return 1.5
        
        return 1.0  # No adjustment
    
    def _get_payment_severity(self, risk_type: str) -> float:
        """Get severity multiplier for payment risks"""
        severity_map = {
            "late_payment_penalties": 1.3,
            "upfront_payments": 1.4,
            "payment_acceleration": 1.6,
            "complex_payment_terms": 1.2
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_liability_severity(self, risk_type: str) -> float:
        """Get severity multiplier for liability risks"""
        severity_map = {
            "unlimited_liability": 1.8,
            "broad_indemnification": 1.6,
            "consequential_damages": 1.7,
            "financial_guarantees": 1.4
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_cost_severity(self, risk_type: str) -> float:
        """Get severity multiplier for cost risks"""
        severity_map = {
            "cost_escalation": 1.3,
            "hidden_costs": 1.4,
            "cost_reimbursement": 1.2,
            "change_order_costs": 1.5
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_revenue_severity(self, risk_type: str) -> float:
        """Get severity multiplier for revenue risks"""
        severity_map = {
            "revenue_sharing": 1.3,
            "revenue_recognition": 1.2,
            "minimum_commitments": 1.4,
            "termination_impact": 1.5
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_clause_type_multiplier(self, clause_type: str, risk_dimension: str) -> float:
        """Get clause type multiplier for specific risk dimension"""
        multipliers = {
            "payment": {
                "payment": 1.5,
                "termination": 1.2,
                "force_majeure": 1.1,
                "warranties": 1.0,
                "confidentiality": 0.8
            },
            "liability": {
                "limitation_of_liability": 1.6,
                "indemnification": 1.5,
                "warranties": 1.3,
                "intellectual_property": 1.2,
                "payment": 1.0
            },
            "cost": {
                "payment": 1.4,
                "termination": 1.3,
                "change_orders": 1.5,
                "force_majeure": 1.2,
                "confidentiality": 0.9
            },
            "revenue": {
                "payment": 1.4,
                "termination": 1.5,
                "intellectual_property": 1.2,
                "exclusivity": 1.3,
                "confidentiality": 0.9
            }
        }
        
        return multipliers.get(risk_dimension, {}).get(clause_type, 1.0)
    
    def _get_contract_value_multiplier(self, contract_value: float) -> float:
        """Get risk multiplier based on contract value"""
        if contract_value >= 10000000:  # $10M+
            return 1.4
        elif contract_value >= 1000000:  # $1M+
            return 1.3
        elif contract_value >= 100000:   # $100K+
            return 1.2
        elif contract_value >= 10000:    # $10K+
            return 1.1
        else:
            return 1.0
    
    def _get_duration_multiplier(self, duration_months: int) -> float:
        """Get risk multiplier based on contract duration"""
        if duration_months >= 60:  # 5+ years
            return 1.3
        elif duration_months >= 36:  # 3+ years  
            return 1.2
        elif duration_months >= 24:  # 2+ years
            return 1.1
        else:
            return 1.0
    
    def _categorize_severity(self, multiplier: float) -> str:
        """Categorize severity based on multiplier value"""
        if multiplier >= 1.6:
            return "CRITICAL"
        elif multiplier >= 1.4:
            return "HIGH"
        elif multiplier >= 1.1:
            return "MEDIUM"
        else:
            return "LOW"