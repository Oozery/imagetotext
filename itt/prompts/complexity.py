"""
=============================================================================
Prompt Complexity Analyzer
=============================================================================
Analyzes prompt difficulty to determine the optimal optimization strategy.

Simple prompts (e.g., "A red apple on a table") can skip the expensive
LLM rewriting stage and go directly to Promptist, saving API calls and
time. Complex prompts (spatial, counting, negation) get the full pipeline.

This is a key efficiency optimization — routing prompts through the
minimum necessary processing stages based on their compositional complexity.

Complexity Signals:
  - Word count and sentence structure
  - Presence of spatial keywords (left, right, between, above, below)
  - Counting words (numbers, "exactly", "several")
  - Negation patterns ("no", "without", "but not")
  - Multi-object detection (conjunctions, lists)
  - Abstract/metaphorical language
=============================================================================
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("hybrid_prompt_opt.complexity")


class ComplexityLevel(Enum):
    """Prompt complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class ComplexityReport:
    """Detailed breakdown of a prompt's complexity analysis."""
    prompt: str
    level: ComplexityLevel
    score: float  # 0.0 to 1.0
    word_count: int
    signals: Dict[str, float]
    recommended_stages: List[str]
    skip_stage1: bool
    reasoning: str


# ============================================================================
# Signal Detection Patterns
# ============================================================================

SPATIAL_KEYWORDS = {
    "left", "right", "above", "below", "between", "behind", "front",
    "beside", "next to", "on top of", "underneath", "adjacent",
    "facing", "opposite", "center", "corner", "edge", "positioned",
    "leaning against", "perched on",
}

COUNTING_PATTERNS = [
    r"\b(exactly|precisely)\s+\w+",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
    r"\b\d+\b",
    r"\b(several|multiple|many|few|couple|pair)\b",
]

NEGATION_PATTERNS = [
    r"\bno\s+\w+",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bbut\s+no\b",
    r"\bdevoid\b",
    r"\bempty\b",
    r"\babsence\b",
    r"\bnone\b",
]

ABSTRACT_KEYWORDS = {
    "concept", "feeling", "emotion", "visualized", "represented",
    "depicted", "metaphor", "abstract", "symbolize", "embody",
    "essence", "spirit", "atmosphere", "mood", "aura",
}

STYLE_KEYWORDS = {
    "style of", "in the style", "watercolor", "oil painting",
    "pencil sketch", "impressionism", "cubism", "surrealism",
    "digital art", "cyberpunk", "anime", "photorealistic",
    "minimalist", "baroque", "art nouveau",
}

MULTI_OBJECT_INDICATORS = {
    "and", "with", "beside", "next to", "together",
    "alongside", "accompanied by",
}


class PromptComplexityAnalyzer:
    """
    Analyzes prompt complexity to determine optimal optimization routing.
    
    Scoring weights can be tuned based on empirical results — categories
    where the LLM rewriter provides the most improvement should have
    higher weights to ensure they always get full pipeline treatment.
    
    Args:
        simple_threshold: Score below which prompts are classified as simple.
        complex_threshold: Score above which prompts are classified as complex.
    """

    def __init__(
        self,
        simple_threshold: float = 0.25,
        complex_threshold: float = 0.55,
    ):
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold

        # Weights for each signal type (tuned from empirical results)
        self.weights = {
            "spatial": 0.25,
            "counting": 0.25,
            "negation": 0.20,
            "multi_object": 0.10,
            "abstract": 0.08,
            "style": 0.05,
            "length": 0.07,
        }

    def analyze(self, prompt: str) -> ComplexityReport:
        """
        Perform full complexity analysis on a prompt.
        
        Returns a ComplexityReport with score, classification, and
        recommended pipeline stages.
        """
        prompt_lower = prompt.lower().strip()
        words = prompt_lower.split()
        word_count = len(words)

        # Compute individual signal scores
        signals = {
            "spatial": self._score_spatial(prompt_lower),
            "counting": self._score_counting(prompt_lower),
            "negation": self._score_negation(prompt_lower),
            "multi_object": self._score_multi_object(prompt_lower),
            "abstract": self._score_abstract(prompt_lower),
            "style": self._score_style(prompt_lower),
            "length": self._score_length(word_count),
        }

        # Weighted composite score
        total_score = sum(
            signals[key] * self.weights[key]
            for key in signals
        )
        total_score = min(1.0, total_score)

        # Classification
        if total_score < self.simple_threshold:
            level = ComplexityLevel.SIMPLE
        elif total_score < self.complex_threshold:
            level = ComplexityLevel.MODERATE
        elif total_score < 0.80:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX

        # Routing decision
        skip_stage1 = level == ComplexityLevel.SIMPLE
        recommended = self._recommend_stages(level, signals)
        reasoning = self._build_reasoning(signals, level)

        report = ComplexityReport(
            prompt=prompt,
            level=level,
            score=round(total_score, 4),
            word_count=word_count,
            signals={k: round(v, 4) for k, v in signals.items()},
            recommended_stages=recommended,
            skip_stage1=skip_stage1,
            reasoning=reasoning,
        )

        logger.debug(
            f"Complexity: {level.value} ({total_score:.3f}) — {prompt[:60]}..."
        )
        return report

    def batch_analyze(self, prompts: List[str]) -> List[ComplexityReport]:
        """Analyze a batch of prompts and return sorted by complexity."""
        reports = [self.analyze(p) for p in prompts]
        reports.sort(key=lambda r: r.score, reverse=True)
        return reports

    def get_routing_summary(self, prompts: List[str]) -> Dict[str, int]:
        """
        Summarize how many prompts would be routed to each pipeline path.
        Useful for estimating API costs before running the full pipeline.
        """
        reports = [self.analyze(p) for p in prompts]
        summary = {
            "full_pipeline": sum(1 for r in reports if not r.skip_stage1),
            "promptist_only": sum(1 for r in reports if r.skip_stage1),
            "total": len(reports),
            "avg_complexity": round(
                sum(r.score for r in reports) / len(reports), 4
            ) if reports else 0.0,
        }
        return summary

    # ------------------------------------------------------------------
    # Signal Scoring Functions
    # ------------------------------------------------------------------

    def _score_spatial(self, text: str) -> float:
        """Score spatial relationship complexity (0.0–1.0)."""
        matches = sum(1 for kw in SPATIAL_KEYWORDS if kw in text)
        return min(1.0, matches * 0.4)

    def _score_counting(self, text: str) -> float:
        """Score counting/numerical complexity."""
        matches = 0
        for pattern in COUNTING_PATTERNS:
            matches += len(re.findall(pattern, text))
        return min(1.0, matches * 0.35)

    def _score_negation(self, text: str) -> float:
        """Score negation complexity."""
        matches = 0
        for pattern in NEGATION_PATTERNS:
            matches += len(re.findall(pattern, text))
        return min(1.0, matches * 0.5)

    def _score_multi_object(self, text: str) -> float:
        """Score multi-object scene complexity."""
        matches = sum(1 for kw in MULTI_OBJECT_INDICATORS if kw in text)
        return min(1.0, matches * 0.3)

    def _score_abstract(self, text: str) -> float:
        """Score abstract/metaphorical language."""
        matches = sum(1 for kw in ABSTRACT_KEYWORDS if kw in text)
        return min(1.0, matches * 0.4)

    def _score_style(self, text: str) -> float:
        """Score style transfer requirements."""
        matches = sum(1 for kw in STYLE_KEYWORDS if kw in text)
        return min(1.0, matches * 0.5)

    def _score_length(self, word_count: int) -> float:
        """Score based on prompt length (longer = potentially more complex)."""
        if word_count <= 6:
            return 0.0
        elif word_count <= 12:
            return 0.3
        elif word_count <= 20:
            return 0.6
        else:
            return 1.0

    # ------------------------------------------------------------------
    # Routing Logic
    # ------------------------------------------------------------------

    def _recommend_stages(
        self, level: ComplexityLevel, signals: Dict[str, float]
    ) -> List[str]:
        """Determine which pipeline stages to apply."""
        if level == ComplexityLevel.SIMPLE:
            return ["promptist"]

        stages = ["compositional_rewrite", "promptist"]

        # Very complex prompts get additional processing
        if level == ComplexityLevel.VERY_COMPLEX:
            stages.insert(0, "complexity_decomposition")

        return stages

    def _build_reasoning(
        self, signals: Dict[str, float], level: ComplexityLevel
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        active_signals = [
            name for name, score in signals.items() if score > 0.2
        ]

        if not active_signals:
            return f"Simple prompt with no significant complexity signals."

        signal_str = ", ".join(active_signals)
        return (
            f"Classified as {level.value} due to: {signal_str}. "
            f"{'Full pipeline recommended.' if level != ComplexityLevel.SIMPLE else 'Promptist-only sufficient.'}"
        )
