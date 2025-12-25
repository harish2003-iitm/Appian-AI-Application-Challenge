"""
Query Analyzer - Intelligently selects RAG mode based on query complexity
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    query: str
    complexity: str  # "simple", "medium", "complex"
    recommended_mode: str  # "Standard", "Graph RAG", "Agentic RAG"
    confidence: float
    reasoning: str
    detected_features: List[str]


class IntelligentQueryAnalyzer:
    """
    Analyzes queries to determine the best RAG mode to use.

    RAG Modes:
    - Standard: Simple factual queries, direct lookups
    - Graph RAG: Relationship-heavy queries, entity connections
    - Agentic RAG: Complex multi-step queries, calculations, web research
    """

    # Keywords that indicate different query types
    SIMPLE_KEYWORDS = [
        r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bdescribe\b',
        r'\blist\b', r'\bshow\b', r'\btell me about\b'
    ]

    RELATIONSHIP_KEYWORDS = [
        r'\brelationship\b', r'\bconnect', r'\brelated to\b', r'\bassociated with\b',
        r'\blink', r'\bdependen', r'\baffect', r'\bimpact', r'\bcorrelat',
        r'\bhow does .* relate to\b', r'\bconnection between\b'
    ]

    ENTITY_KEYWORDS = [
        r'\bpolicy\b', r'\bregulation\b', r'\borganization\b', r'\bcompany\b',
        r'\blocation\b', r'\bcoverage\b', r'\bclaim\b', r'\brequirement\b'
    ]

    COMPLEX_KEYWORDS = [
        r'\bcalculate\b', r'\bcompute\b', r'\bcompare .* and\b', r'\bdifference between\b',
        r'\bwhy\b', r'\bhow does .* work\b', r'\bwhat would happen if\b',
        r'\bfind .* and .* and\b', r'\bsearch .* then\b', r'\bfirst .* then\b'
    ]

    WEB_KEYWORDS = [
        r'\blatest\b', r'\brecent\b', r'\bcurrent\b', r'\btoday\b',
        r'\bnews\b', r'\bupdate\b', r'\bweb\b', r'\bonline\b', r'\burl\b'
    ]

    MULTI_STEP_PATTERNS = [
        r'\band then\b', r'\bafter that\b', r'\bfollowed by\b',
        r'\bstep by step\b', r'\bfirst.*second.*third\b'
    ]

    CALCULATION_PATTERNS = [
        r'\d+\s*[+\-*/]\s*\d+', r'\bpremium\b.*\bcalculate\b',
        r'\btotal\b.*\bcost\b', r'\bsum\b', r'\baverage\b', r'\bpercentage\b'
    ]

    def __init__(self):
        self.query_history = []

    def analyze_query(self, query: str, case_context: Dict = None) -> QueryAnalysis:
        """
        Analyze query and recommend best RAG mode.

        Args:
            query: User's query string
            case_context: Optional case context (case_type, state, etc.)

        Returns:
            QueryAnalysis with recommended mode and reasoning
        """
        query_lower = query.lower()
        detected_features = []
        scores = {
            "Standard": 1.0,      # Base score
            "Graph RAG": 0.0,
            "Agentic RAG": 0.0
        }

        # 1. Check query length and structure
        word_count = len(query.split())
        if word_count < 5:
            scores["Standard"] += 2.0
            detected_features.append("short_query")
        elif word_count > 20:
            scores["Agentic RAG"] += 1.5
            detected_features.append("long_query")

        # 2. Check for simple lookup patterns
        for pattern in self.SIMPLE_KEYWORDS:
            if re.search(pattern, query_lower):
                scores["Standard"] += 1.5
                detected_features.append("simple_lookup")
                break

        # 3. Check for relationship queries
        relationship_count = 0
        for pattern in self.RELATIONSHIP_KEYWORDS:
            if re.search(pattern, query_lower):
                relationship_count += 1
        if relationship_count > 0:
            scores["Graph RAG"] += 2.0 * relationship_count
            detected_features.append(f"relationship_query_x{relationship_count}")

        # 4. Check for entity mentions (boosts Graph RAG)
        entity_count = 0
        for pattern in self.ENTITY_KEYWORDS:
            entity_count += len(re.findall(pattern, query_lower))
        if entity_count >= 2:
            scores["Graph RAG"] += 1.0
            detected_features.append(f"multiple_entities_x{entity_count}")

        # 5. Check for complex reasoning patterns
        for pattern in self.COMPLEX_KEYWORDS:
            if re.search(pattern, query_lower):
                scores["Agentic RAG"] += 2.0
                detected_features.append("complex_reasoning")
                break

        # 6. Check for web/latest info needs
        for pattern in self.WEB_KEYWORDS:
            if re.search(pattern, query_lower):
                scores["Agentic RAG"] += 2.5
                detected_features.append("web_search_needed")
                break

        # 7. Check for multi-step queries
        for pattern in self.MULTI_STEP_PATTERNS:
            if re.search(pattern, query_lower):
                scores["Agentic RAG"] += 3.0
                detected_features.append("multi_step")
                break

        # 8. Check for calculations
        for pattern in self.CALCULATION_PATTERNS:
            if re.search(pattern, query_lower):
                scores["Agentic RAG"] += 2.5
                detected_features.append("calculation_needed")
                break

        # 9. Check for multiple questions (AND, OR)
        if ' and ' in query_lower or ' or ' in query_lower:
            and_count = query_lower.count(' and ')
            or_count = query_lower.count(' or ')
            if and_count + or_count >= 2:
                scores["Agentic RAG"] += 2.0
                detected_features.append("multiple_conditions")

        # 10. Check for comparison queries
        if 'compare' in query_lower or 'versus' in query_lower or ' vs ' in query_lower:
            scores["Graph RAG"] += 1.5
            detected_features.append("comparison_query")

        # 11. Question mark count (multiple questions)
        question_marks = query.count('?')
        if question_marks > 1:
            scores["Agentic RAG"] += 1.5 * question_marks
            detected_features.append(f"multiple_questions_x{question_marks}")

        # Determine winner
        recommended_mode = max(scores, key=scores.get)
        max_score = scores[recommended_mode]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.33

        # Determine complexity
        if scores["Agentic RAG"] > 3.0:
            complexity = "complex"
        elif scores["Graph RAG"] > 2.0 or scores["Agentic RAG"] > 1.0:
            complexity = "medium"
        else:
            complexity = "simple"

        # Generate reasoning
        reasoning = self._generate_reasoning(
            recommended_mode, detected_features, scores, complexity
        )

        analysis = QueryAnalysis(
            query=query,
            complexity=complexity,
            recommended_mode=recommended_mode,
            confidence=confidence,
            reasoning=reasoning,
            detected_features=detected_features
        )

        # Store in history
        self.query_history.append(analysis)

        return analysis

    def _generate_reasoning(
        self, mode: str, features: List[str], scores: Dict[str, float], complexity: str
    ) -> str:
        """Generate human-readable reasoning for mode selection"""

        reasons = []

        if "simple_lookup" in features or "short_query" in features:
            reasons.append("simple factual query")

        if any("relationship" in f for f in features):
            reasons.append("involves entity relationships")

        if any("entities" in f for f in features):
            reasons.append("mentions multiple entities")

        if "complex_reasoning" in features:
            reasons.append("requires complex reasoning")

        if "web_search_needed" in features:
            reasons.append("needs latest/web information")

        if "multi_step" in features:
            reasons.append("multi-step query")

        if "calculation_needed" in features:
            reasons.append("involves calculations")

        if "multiple_conditions" in features:
            reasons.append("multiple conditions/questions")

        if "comparison_query" in features:
            reasons.append("comparison between entities")

        reason_str = ", ".join(reasons) if reasons else "default selection"

        return f"Selected {mode} (complexity: {complexity}) - {reason_str}. Scores: {scores}"

    def get_query_stats(self) -> Dict:
        """Get statistics about analyzed queries"""
        if not self.query_history:
            return {"total_queries": 0}

        mode_counts = {}
        complexity_counts = {}

        for analysis in self.query_history:
            mode_counts[analysis.recommended_mode] = mode_counts.get(analysis.recommended_mode, 0) + 1
            complexity_counts[analysis.complexity] = complexity_counts.get(analysis.complexity, 0) + 1

        avg_confidence = sum(a.confidence for a in self.query_history) / len(self.query_history)

        return {
            "total_queries": len(self.query_history),
            "mode_distribution": mode_counts,
            "complexity_distribution": complexity_counts,
            "average_confidence": avg_confidence
        }
