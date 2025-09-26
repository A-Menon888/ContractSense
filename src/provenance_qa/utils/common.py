"""
Utility Functions for Provenance QA System

Common utilities for text processing, scoring, validation, and system operations.
"""

import re
import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

def generate_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = str(int(time.time()))[-6:]
    return f"{prefix}_{timestamp}_{unique_id}" if prefix else f"{timestamp}_{unique_id}"

def generate_hash(text: str) -> str:
    """Generate hash for text content"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]

def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from text length"""
    return max(1, len(text) // chars_per_token)

def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    if add_ellipsis:
        truncated += "..."
    
    return truncated

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def extract_legal_entities(text: str) -> List[Dict[str, Any]]:
    """Extract legal entities from text using simple patterns"""
    entities = []
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                "text": match.group(),
                "entity_type": "date",
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.8
            })
    
    # Money amounts
    money_pattern = r'\$[\d,]+(?:\.\d{2})?'
    matches = re.finditer(money_pattern, text)
    for match in matches:
        entities.append({
            "text": match.group(),
            "entity_type": "money",
            "start_pos": match.start(),
            "end_pos": match.end(),
            "confidence": 0.9
        })
    
    # Percentages
    percentage_pattern = r'\b\d+(?:\.\d+)?%'
    matches = re.finditer(percentage_pattern, text)
    for match in matches:
        entities.append({
            "text": match.group(),
            "entity_type": "percentage",
            "start_pos": match.start(),
            "end_pos": match.end(),
            "confidence": 0.85
        })
    
    # Company names (simple pattern)
    company_patterns = [
        r'\b[A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Corporation|Company|Co|Ltd|Limited)\.?\b',
        r'\b[A-Z][A-Za-z\s&]+(?:,\s*Inc|,\s*LLC|,\s*Corp)\.?\b'
    ]
    
    for pattern in company_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "text": match.group(),
                "entity_type": "organization",
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.7
            })
    
    return entities

def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract important keywords from text"""
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter and score words
    word_freq = {}
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

def extract_legal_concepts(text: str) -> List[str]:
    """Extract legal concepts and terms from text"""
    legal_terms = {
        # Contract terms
        'termination', 'breach', 'default', 'remedy', 'cure', 'notice', 'warranty',
        'representation', 'covenant', 'indemnification', 'liability', 'damages',
        'force majeure', 'assignment', 'novation', 'amendment', 'modification',
        
        # Business terms
        'consideration', 'payment', 'invoice', 'billing', 'revenue', 'profit',
        'loss', 'penalty', 'fee', 'commission', 'royalty', 'license',
        
        # Legal procedures
        'arbitration', 'litigation', 'jurisdiction', 'venue', 'governing law',
        'dispute resolution', 'mediation', 'settlement', 'judgment',
        
        # Time-related
        'effective date', 'expiration', 'renewal', 'extension', 'term',
        'duration', 'period', 'deadline', 'timeline',
        
        # Performance
        'deliverable', 'milestone', 'performance', 'completion', 'acceptance',
        'approval', 'consent', 'authorization', 'compliance'
    }
    
    text_lower = text.lower()
    found_concepts = []
    
    for concept in legal_terms:
        if concept in text_lower:
            found_concepts.append(concept)
    
    return list(set(found_concepts))  # Remove duplicates

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap"""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def calculate_confidence_score(factors: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate weighted confidence score from multiple factors"""
    if not factors:
        return 0.0
    
    if weights is None:
        # Default equal weights
        weights = {key: 1.0 for key in factors.keys()}
    
    weighted_sum = sum(factors[key] * weights.get(key, 1.0) for key in factors)
    total_weight = sum(weights.get(key, 1.0) for key in factors)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def validate_text_quality(text: str) -> Dict[str, Any]:
    """Validate text quality and return assessment"""
    if not text or not text.strip():
        return {
            "is_valid": False,
            "issues": ["Text is empty or only whitespace"],
            "score": 0.0
        }
    
    issues = []
    score_factors = {}
    
    # Length check
    text_length = len(text.strip())
    if text_length < 10:
        issues.append("Text is very short")
        score_factors["length"] = 0.3
    elif text_length < 50:
        score_factors["length"] = 0.6
    else:
        score_factors["length"] = 1.0
    
    # Character diversity
    unique_chars = len(set(text.lower()))
    char_diversity = min(1.0, unique_chars / 20)  # Normalize to reasonable range
    score_factors["diversity"] = char_diversity
    
    # Word count
    words = text.split()
    word_count = len(words)
    if word_count < 3:
        issues.append("Very few words")
        score_factors["word_count"] = 0.3
    elif word_count < 10:
        score_factors["word_count"] = 0.6
    else:
        score_factors["word_count"] = 1.0
    
    # Basic grammar (simple check for sentence structure)
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences == 0 and word_count > 10:
        issues.append("No sentence endings found")
        score_factors["grammar"] = 0.5
    else:
        score_factors["grammar"] = 0.8
    
    overall_score = calculate_confidence_score(score_factors)
    
    return {
        "is_valid": overall_score >= 0.5 and len(issues) == 0,
        "issues": issues,
        "score": overall_score,
        "details": score_factors
    }

def format_timestamp(timestamp: Optional[float] = None, include_date: bool = True) -> str:
    """Format timestamp for human readability"""
    if timestamp is None:
        timestamp = time.time()
    
    dt = datetime.fromtimestamp(timestamp)
    
    if include_date:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.strftime("%H:%M:%S")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to specified range"""
    return max(min_val, min(max_val, score))

def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any], conflict_resolution: str = "second") -> Dict[str, Any]:
    """Merge two dictionaries with conflict resolution strategy"""
    merged = dict1.copy()
    
    for key, value in dict2.items():
        if key not in merged:
            merged[key] = value
        else:
            if conflict_resolution == "second":
                merged[key] = value
            elif conflict_resolution == "first":
                pass  # Keep original value
            elif conflict_resolution == "combine" and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = merged[key] + value
            elif conflict_resolution == "max" and isinstance(merged[key], (int, float)) and isinstance(value, (int, float)):
                merged[key] = max(merged[key], value)
    
    return merged

class Timer:
    """Simple timer utility for performance measurement"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """Get elapsed time (without stopping)"""
        if self.start_time is None:
            return 0.0
        
        current_time = self.end_time if self.end_time else time.time()
        return current_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_proceed(self) -> bool:
        """Check if we can make another call"""
        current_time = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if current_time - call_time < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a new API call"""
        self.calls.append(time.time())
    
    def time_until_next_call(self) -> float:
        """Get time in seconds until next call is allowed"""
        if self.can_proceed():
            return 0.0
        
        current_time = time.time()
        oldest_call = min(self.calls) if self.calls else current_time
        
        return max(0.0, self.time_window - (current_time - oldest_call))