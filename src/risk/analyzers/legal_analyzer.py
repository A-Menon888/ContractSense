"""
Legal Risk Analyzer

Analyzes contract clauses for legal and regulatory risk factors including:
- Liability and indemnification risks
- Regulatory compliance issues  
- Dispute resolution and jurisdiction risks
- IP and confidentiality risks
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LegalRiskMetrics:
    """Legal risk analysis results"""
    liability_risk_score: float
    regulatory_risk_score: float
    dispute_risk_score: float
    ip_risk_score: float
    overall_legal_risk: float
    risk_factors: List[Dict[str, Any]]
    confidence: float

class LegalRiskAnalyzer:
    """Analyzes legal risk aspects of contract clauses"""
    
    # Liability and indemnification risk patterns
    LIABILITY_PATTERNS = {
        "unlimited_liability": [
            r"unlimited.*liability|liability.*unlimited",
            r"without limit.*liability|liability.*without limit", 
            r"no cap.*liability|liability.*no cap",
            r"total.*liability|aggregate.*liability"
        ],
        "broad_indemnification": [
            r"indemnify.*defend.*hold harmless",
            r"broad.*indemnification|comprehensive.*indemnification",
            r"any.*all.*claims|all.*claims.*arising",
            r"indemnify.*against.*any"
        ],
        "joint_liability": [
            r"joint.*several.*liability|jointly.*severally.*liable",
            r"joint.*liability|several.*liability",
            r"collectively.*liable|liable.*collectively"
        ],
        "successor_liability": [
            r"successor.*liability|liability.*successor",
            r"assigns.*liability|liability.*assigns",
            r"transferee.*liability|liability.*transferee"
        ]
    }
    
    # Regulatory compliance risk patterns
    REGULATORY_PATTERNS = {
        "data_protection": [
            r"personal.*data|data.*protection",
            r"gdpr|general.*data.*protection",
            r"privacy.*policy|policy.*privacy",
            r"data.*subject.*rights"
        ],
        "financial_compliance": [
            r"sox|sarbanes.*oxley|financial.*reporting",
            r"audit.*requirements|auditing.*standards",
            r"internal.*controls|controls.*internal",
            r"financial.*disclosure|disclosure.*financial"
        ],
        "industry_regulations": [
            r"fda|food.*drug.*administration",
            r"hipaa|health.*insurance.*portability",
            r"pci.*dss|payment.*card.*industry",
            r"regulatory.*approval|approval.*regulatory"
        ],
        "export_controls": [
            r"export.*control|itar|ear",
            r"export.*administration.*regulations",
            r"foreign.*persons|foreign.*nationals",
            r"technology.*transfer|transfer.*technology"
        ]
    }
    
    # Dispute resolution risk patterns
    DISPUTE_PATTERNS = {
        "exclusive_jurisdiction": [
            r"exclusive.*jurisdiction|jurisdiction.*exclusive",
            r"sole.*jurisdiction|jurisdiction.*sole",
            r"courts.*shall.*have.*exclusive",
            r"submit.*exclusive.*jurisdiction"
        ],
        "mandatory_arbitration": [
            r"mandatory.*arbitration|arbitration.*mandatory",
            r"shall.*arbitrate|arbitration.*required",
            r"binding.*arbitration|arbitration.*binding",
            r"disputes.*arbitration|arbitration.*disputes"
        ],
        "governing_law_risk": [
            r"governed.*by.*laws.*of",
            r"laws.*of.*state|state.*laws.*govern",
            r"applicable.*law|law.*applicable",
            r"subject.*to.*laws"
        ],
        "waiver_rights": [
            r"waive.*right.*jury|jury.*trial.*waived",
            r"waive.*right.*class|class.*action.*waived", 
            r"waive.*claims|waive.*causes.*action",
            r"release.*claims|release.*causes.*action"
        ]
    }
    
    # Intellectual property risk patterns
    IP_PATTERNS = {
        "broad_ip_assignment": [
            r"assign.*all.*intellectual.*property",
            r"transfer.*all.*rights|all.*rights.*transfer",
            r"work.*for.*hire|works.*made.*for.*hire",
            r"intellectual.*property.*created"
        ],
        "ip_indemnification": [
            r"indemnify.*intellectual.*property",
            r"ip.*indemnification|indemnification.*ip",
            r"patent.*indemnification|trademark.*indemnification",
            r"defend.*ip.*claims"
        ],
        "licensing_restrictions": [
            r"exclusive.*license|license.*exclusive",
            r"non.*transferable.*license|license.*non.*transferable",
            r"revocable.*license|license.*revocable",
            r"limited.*license|license.*limited"
        ],
        "confidentiality_risks": [
            r"perpetual.*confidentiality|confidentiality.*perpetual",
            r"broad.*confidentiality|confidentiality.*broad",
            r"residual.*knowledge|knowledge.*residual",
            r"confidential.*information.*includes"
        ]
    }
    
    def __init__(self):
        self.risk_weights = {
            "liability": 0.40,
            "regulatory": 0.30,
            "dispute": 0.15,
            "ip": 0.15
        }
    
    def analyze_legal_risk(self,
                          clause_type: str,
                          clause_text: str,
                          governing_law: Optional[str] = None,
                          industry: Optional[str] = None) -> LegalRiskMetrics:
        """Perform comprehensive legal risk analysis on a clause"""
        
        # Analyze different legal risk dimensions
        liability_risk, liability_factors = self._analyze_liability_risk(clause_text)
        regulatory_risk, regulatory_factors = self._analyze_regulatory_risk(clause_text, industry)
        dispute_risk, dispute_factors = self._analyze_dispute_risk(clause_text, governing_law)
        ip_risk, ip_factors = self._analyze_ip_risk(clause_text)
        
        # Apply clause-type specific adjustments
        liability_risk *= self._get_clause_type_multiplier(clause_type, "liability")
        regulatory_risk *= self._get_clause_type_multiplier(clause_type, "regulatory")
        dispute_risk *= self._get_clause_type_multiplier(clause_type, "dispute")
        ip_risk *= self._get_clause_type_multiplier(clause_type, "ip")
        
        # Calculate overall legal risk score
        overall_risk = (
            liability_risk * self.risk_weights["liability"] +
            regulatory_risk * self.risk_weights["regulatory"] +
            dispute_risk * self.risk_weights["dispute"] +
            ip_risk * self.risk_weights["ip"]
        )
        
        # Apply jurisdiction-specific adjustments
        if governing_law:
            overall_risk *= self._get_jurisdiction_multiplier(governing_law)
        
        # Combine all risk factors
        all_factors = liability_factors + regulatory_factors + dispute_factors + ip_factors
        
        # Calculate confidence based on factors and text length
        confidence = min(0.65 + (len(all_factors) * 0.07) + (len(clause_text) / 2000), 0.95)
        
        return LegalRiskMetrics(
            liability_risk_score=round(liability_risk, 2),
            regulatory_risk_score=round(regulatory_risk, 2),
            dispute_risk_score=round(dispute_risk, 2),
            ip_risk_score=round(ip_risk, 2),
            overall_legal_risk=round(overall_risk, 2),
            risk_factors=all_factors,
            confidence=round(confidence, 3)
        )
    
    def _analyze_liability_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze liability and indemnification risks"""
        text_lower = clause_text.lower()
        risk_score = 4.0  # Base liability risk
        risk_factors = []
        
        for risk_type, patterns in self.LIABILITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_liability_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Liability risk factor: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier,
                        "evidence": self._extract_evidence(clause_text, pattern)
                    })
                    break
        
        # Look for liability caps (reduce risk)
        if re.search(r"liability.*limited.*to|limited.*liability", text_lower):
            risk_score *= 0.7
            risk_factors.append({
                "type": "liability_cap",
                "description": "Liability appears to be capped or limited",
                "severity": "BENEFICIAL",
                "impact": 0.7
            })
        
        risk_score = min(risk_score, 10.0)
        return risk_score, risk_factors
    
    def _analyze_regulatory_risk(self, clause_text: str, industry: Optional[str] = None) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze regulatory compliance risks"""
        text_lower = clause_text.lower()
        risk_score = 3.0  # Base regulatory risk
        risk_factors = []
        
        for risk_type, patterns in self.REGULATORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_regulatory_severity(risk_type, industry)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Regulatory risk factor: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier,
                        "evidence": self._extract_evidence(clause_text, pattern)
                    })
                    break
        
        # Industry-specific regulatory risk adjustments
        if industry:
            industry_adjustment = self._get_industry_regulatory_multiplier(industry)
            risk_score *= industry_adjustment
        
        risk_score = min(risk_score, 10.0)
        return risk_score, risk_factors
    
    def _analyze_dispute_risk(self, clause_text: str, governing_law: Optional[str] = None) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze dispute resolution risks"""
        text_lower = clause_text.lower()
        risk_score = 3.5  # Base dispute risk
        risk_factors = []
        
        for risk_type, patterns in self.DISPUTE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_dispute_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"Dispute resolution risk: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier,
                        "evidence": self._extract_evidence(clause_text, pattern)
                    })
                    break
        
        risk_score = min(risk_score, 10.0)
        return risk_score, risk_factors
    
    def _analyze_ip_risk(self, clause_text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze intellectual property risks"""
        text_lower = clause_text.lower()
        risk_score = 3.5  # Base IP risk
        risk_factors = []
        
        for risk_type, patterns in self.IP_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity_multiplier = self._get_ip_severity(risk_type)
                    risk_score *= severity_multiplier
                    
                    risk_factors.append({
                        "type": risk_type,
                        "description": f"IP risk factor: {risk_type.replace('_', ' ')}",
                        "severity": self._categorize_severity(severity_multiplier),
                        "impact": severity_multiplier,
                        "evidence": self._extract_evidence(clause_text, pattern)
                    })
                    break
        
        risk_score = min(risk_score, 10.0)
        return risk_score, risk_factors
    
    def _get_liability_severity(self, risk_type: str) -> float:
        """Get severity multiplier for liability risks"""
        severity_map = {
            "unlimited_liability": 1.8,
            "broad_indemnification": 1.6,
            "joint_liability": 1.4,
            "successor_liability": 1.3
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_regulatory_severity(self, risk_type: str, industry: Optional[str] = None) -> float:
        """Get severity multiplier for regulatory risks"""
        base_severity = {
            "data_protection": 1.5,
            "financial_compliance": 1.6,
            "industry_regulations": 1.4,
            "export_controls": 1.7
        }
        
        multiplier = base_severity.get(risk_type, 1.0)
        
        # Industry-specific adjustments
        if industry == "healthcare" and risk_type == "industry_regulations":
            multiplier *= 1.2
        elif industry == "financial" and risk_type == "financial_compliance":
            multiplier *= 1.3
        elif industry == "technology" and risk_type == "export_controls":
            multiplier *= 1.2
        
        return multiplier
    
    def _get_dispute_severity(self, risk_type: str) -> float:
        """Get severity multiplier for dispute risks"""
        severity_map = {
            "exclusive_jurisdiction": 1.3,
            "mandatory_arbitration": 1.2,
            "governing_law_risk": 1.1,
            "waiver_rights": 1.5
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_ip_severity(self, risk_type: str) -> float:
        """Get severity multiplier for IP risks"""
        severity_map = {
            "broad_ip_assignment": 1.6,
            "ip_indemnification": 1.4,
            "licensing_restrictions": 1.3,
            "confidentiality_risks": 1.2
        }
        return severity_map.get(risk_type, 1.0)
    
    def _get_clause_type_multiplier(self, clause_type: str, risk_dimension: str) -> float:
        """Get clause type multiplier for specific legal risk dimension"""
        multipliers = {
            "liability": {
                "indemnification": 1.6,
                "limitation_of_liability": 1.5,
                "warranties": 1.3,
                "intellectual_property": 1.2,
                "termination": 1.1
            },
            "regulatory": {
                "confidentiality": 1.4,
                "data_processing": 1.5,
                "compliance": 1.6,
                "audit_rights": 1.3,
                "reporting": 1.2
            },
            "dispute": {
                "governing_law": 1.5,
                "dispute_resolution": 1.6,
                "jurisdiction": 1.4,
                "arbitration": 1.3,
                "termination": 1.1
            },
            "ip": {
                "intellectual_property": 1.6,
                "confidentiality": 1.4,
                "licensing": 1.5,
                "work_for_hire": 1.3,
                "non_disclosure": 1.2
            }
        }
        
        return multipliers.get(risk_dimension, {}).get(clause_type, 1.0)
    
    def _get_jurisdiction_multiplier(self, governing_law: str) -> float:
        """Get risk multiplier based on governing law jurisdiction"""
        # Higher risk jurisdictions
        high_risk_jurisdictions = ["delaware", "new york", "california"]
        medium_risk_jurisdictions = ["texas", "illinois", "florida"]
        
        law_lower = governing_law.lower()
        
        if any(jurisdiction in law_lower for jurisdiction in high_risk_jurisdictions):
            return 1.1  # Slightly higher risk due to sophisticated legal environment
        elif any(jurisdiction in law_lower for jurisdiction in medium_risk_jurisdictions):
            return 1.0
        else:
            return 0.95  # Slightly lower risk for less litigious jurisdictions
    
    def _get_industry_regulatory_multiplier(self, industry: str) -> float:
        """Get industry-specific regulatory risk multiplier"""
        industry_multipliers = {
            "healthcare": 1.4,
            "financial": 1.5,
            "pharmaceutical": 1.3,
            "technology": 1.2,
            "energy": 1.3,
            "defense": 1.4,
            "telecommunications": 1.2,
            "manufacturing": 1.1
        }
        return industry_multipliers.get(industry.lower(), 1.0)
    
    def _extract_evidence(self, clause_text: str, pattern: str) -> str:
        """Extract evidence text that matches the risk pattern"""
        match = re.search(pattern, clause_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 20)
            end = min(len(clause_text), match.end() + 20)
            return clause_text[start:end].strip()
        return ""
    
    def _categorize_severity(self, multiplier: float) -> str:
        """Categorize severity based on multiplier value"""
        if multiplier >= 1.6:
            return "CRITICAL"
        elif multiplier >= 1.4:
            return "HIGH"  
        elif multiplier >= 1.1:
            return "MEDIUM"
        elif multiplier < 1.0:
            return "BENEFICIAL"
        else:
            return "LOW"