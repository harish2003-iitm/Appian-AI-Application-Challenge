"""
Layer 4: Governance, Guardrails, and Audit
- Hallucination prevention
- PII detection and redaction
- Fact-checking rails
- Content filtering
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai


@dataclass
class GuardrailResult:
    """Result of guardrail check"""
    passed: bool
    violations: List[str]
    warnings: List[str]
    redacted_content: Optional[str] = None
    confidence: float = 1.0


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information.
    """

    PII_PATTERNS = {
        "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN REDACTED]"),
        "ssn_no_dash": (r'\b\d{9}\b', "[SSN REDACTED]"),
        "phone": (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "[PHONE REDACTED]"),
        "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL REDACTED]"),
        "credit_card": (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "[CARD REDACTED]"),
        "dob": (r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b', "[DOB REDACTED]"),
        "member_id": (r'\bMEM\d{8,12}\b', "[MEMBER_ID REDACTED]"),
        "policy_holder_name": (r'\b[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', None),  # Don't auto-redact names
        "address": (r'\b\d+\s+[A-Za-z]+\s+(St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Rd|Road)\b', "[ADDRESS REDACTED]"),
        "ip_address": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "[IP REDACTED]"),
        "npi": (r'\b\d{10}\b', None),  # Don't redact NPI - it's public info
    }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        found = {}
        for pii_type, (pattern, _) in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found[pii_type] = matches
        return found

    def redact(self, text: str, types_to_redact: List[str] = None) -> Tuple[str, Dict]:
        """Redact PII from text"""
        redacted = text
        redactions = {}

        for pii_type, (pattern, replacement) in self.PII_PATTERNS.items():
            if replacement is None:
                continue
            if types_to_redact and pii_type not in types_to_redact:
                continue

            matches = re.findall(pattern, redacted, re.IGNORECASE)
            if matches:
                redactions[pii_type] = len(matches)
                redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

        return redacted, redactions


class HallucinationDetector:
    """
    Detects potential hallucinations by verifying claims against sources.
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Skip questions and meta-statements
                if not sentence.startswith(('I ', 'Based on', 'According to', '?')):
                    claims.append(sentence)

        return claims

    def verify_claim(self, claim: str, sources: List[str]) -> Tuple[bool, float, str]:
        """Verify a single claim against sources"""
        sources_text = "\n---\n".join(sources[:5])  # Limit sources

        prompt = f"""Verify if this claim is supported by the provided sources.

CLAIM: "{claim}"

SOURCES:
{sources_text}

Respond in JSON:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "evidence": "Quote from source that supports/contradicts" or "No relevant evidence found",
    "reasoning": "Brief explanation"
}}
"""
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            return (
                result.get("supported", False),
                result.get("confidence", 0.5),
                result.get("evidence", "")
            )
        except:
            return (False, 0.0, "Verification failed")

    def check_answer(self, answer: str, sources: List[str]) -> GuardrailResult:
        """Check entire answer for hallucinations"""
        claims = self.extract_claims(answer)
        violations = []
        warnings = []

        for claim in claims[:5]:  # Limit to 5 claims for performance
            supported, confidence, evidence = self.verify_claim(claim, sources)

            if not supported:
                if confidence < 0.3:
                    violations.append(f"Unsupported claim: '{claim[:100]}...'")
                else:
                    warnings.append(f"Weakly supported: '{claim[:100]}...'")

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            confidence=1.0 - (len(violations) * 0.2)
        )


class TopicalGuardrail:
    """
    Ensures responses stay on-topic for insurance/healthcare domain.
    Prevents off-topic discussions.
    """

    ALLOWED_TOPICS = [
        "insurance", "policy", "claim", "coverage", "deductible", "premium",
        "healthcare", "medical", "patient", "provider", "diagnosis", "treatment",
        "regulatory", "compliance", "hipaa", "aca", "cms",
        "flood", "auto", "property", "liability", "workers compensation",
        "appeal", "denial", "authorization", "benefits", "eligibility"
    ]

    BLOCKED_TOPICS = [
        "politics", "election", "president", "congress", "democrat", "republican",
        "religion", "god", "church", "prayer",
        "violence", "weapon", "attack", "terrorism",
        "gambling", "casino", "betting",
        "adult", "explicit", "nsfw"
    ]

    def check_query(self, query: str) -> GuardrailResult:
        """Check if query is on-topic"""
        query_lower = query.lower()
        violations = []
        warnings = []

        # Check for blocked topics
        for topic in self.BLOCKED_TOPICS:
            if topic in query_lower:
                violations.append(f"Off-topic query: contains '{topic}'")

        # Check if query relates to allowed topics
        has_allowed = any(topic in query_lower for topic in self.ALLOWED_TOPICS)

        if not has_allowed and not violations:
            warnings.append("Query may be off-topic for insurance/healthcare domain")

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )

    def check_response(self, response: str) -> GuardrailResult:
        """Check if response stays on-topic"""
        response_lower = response.lower()
        violations = []

        for topic in self.BLOCKED_TOPICS:
            if topic in response_lower:
                violations.append(f"Response contains blocked topic: '{topic}'")

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=[]
        )


class SafetyGuardrail:
    """
    Detects and blocks harmful content, jailbreak attempts.
    """

    JAILBREAK_PATTERNS = [
        r'ignore\s+(previous|all)\s+instructions',
        r'pretend\s+you\s+are',
        r'act\s+as\s+if',
        r'bypass\s+(safety|restrictions)',
        r'DAN\s+mode',
        r'developer\s+mode',
        r'no\s+restrictions',
        r'unlimited\s+mode'
    ]

    HARMFUL_PATTERNS = [
        r'how\s+to\s+(hack|steal|fraud)',
        r'create\s+(fake|forged)',
        r'insurance\s+fraud',
        r'falsify\s+(documents|records)',
    ]

    def check_input(self, text: str) -> GuardrailResult:
        """Check input for jailbreak or harmful content"""
        violations = []

        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Potential jailbreak attempt detected")
                break

        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Potentially harmful request detected")
                break

        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            warnings=[]
        )


class GuardrailsPipeline:
    """
    Main guardrails pipeline that orchestrates all checks.
    """

    def __init__(self, api_key: str, enable_hallucination_check: bool = True):
        self.pii_detector = PIIDetector()
        self.topical_guard = TopicalGuardrail()
        self.safety_guard = SafetyGuardrail()
        self.hallucination_detector = HallucinationDetector(api_key) if enable_hallucination_check else None

    def check_input(self, query: str, redact_pii: bool = True) -> Dict:
        """Run all input guardrails"""
        results = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "query": query,
            "redacted_query": query
        }

        # Safety check
        safety_result = self.safety_guard.check_input(query)
        if not safety_result.passed:
            results["passed"] = False
            results["violations"].extend(safety_result.violations)

        # Topical check
        topical_result = self.topical_guard.check_query(query)
        if not topical_result.passed:
            results["passed"] = False
            results["violations"].extend(topical_result.violations)
        results["warnings"].extend(topical_result.warnings)

        # PII detection and redaction
        if redact_pii:
            redacted, redactions = self.pii_detector.redact(query)
            results["redacted_query"] = redacted
            if redactions:
                results["pii_redacted"] = redactions

        return results

    def check_output(self, answer: str, sources: List[str] = None, redact_pii: bool = True) -> Dict:
        """Run all output guardrails"""
        results = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "answer": answer,
            "redacted_answer": answer
        }

        # Topical check
        topical_result = self.topical_guard.check_response(answer)
        if not topical_result.passed:
            results["passed"] = False
            results["violations"].extend(topical_result.violations)

        # PII redaction
        if redact_pii:
            redacted, redactions = self.pii_detector.redact(answer)
            results["redacted_answer"] = redacted
            if redactions:
                results["pii_redacted"] = redactions

        # Hallucination check (if enabled and sources provided)
        if self.hallucination_detector and sources:
            hallucination_result = self.hallucination_detector.check_answer(answer, sources)
            if not hallucination_result.passed:
                results["passed"] = False
                results["violations"].extend(hallucination_result.violations)
            results["warnings"].extend(hallucination_result.warnings)
            results["hallucination_confidence"] = hallucination_result.confidence

        return results

    def process(self, query: str, answer: str, sources: List[str] = None) -> Dict:
        """Full guardrails processing"""
        input_check = self.check_input(query)
        output_check = self.check_output(answer, sources)

        return {
            "input_check": input_check,
            "output_check": output_check,
            "overall_passed": input_check["passed"] and output_check["passed"],
            "safe_query": input_check["redacted_query"],
            "safe_answer": output_check["redacted_answer"]
        }
