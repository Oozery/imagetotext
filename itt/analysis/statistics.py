"""
=============================================================================
Statistical Analysis Module
=============================================================================
Provides rigorous statistical testing for the prompt optimization results.

Goes beyond simple mean comparisons to include:
  - Wilcoxon signed-rank tests (non-parametric paired comparison)
  - Confidence intervals for mean differences
  - Effect size computation (Cohen's d)
  - Ablation study framework (Stage 1 only vs Stage 2 only vs both)
  - Pearson correlation between CLIP and human scores

These tests are essential for a credible final year project — they
demonstrate that observed improvements are statistically significant
rather than due to random variation.
=============================================================================
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger("hybrid_prompt_opt.statistics")


class StatisticalAnalyzer:
    """
    Statistical analysis for prompt optimization results.
    
    Provides hypothesis testing, effect size computation, and
    correlation analysis to validate experimental findings.
    
    Args:
        significance_level: Alpha threshold for hypothesis tests (default 0.05).
        confidence_level: Confidence level for intervals (default 0.95).
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        confidence_level: float = 0.95,
    ):
        self.alpha = significance_level
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # Paired Comparison Tests
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        label_a: str = "Method A",
        label_b: str = "Method B",
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Non-parametric alternative to paired t-test. Appropriate for
        Likert scale data and small sample sizes.
        
        Tests H0: No difference between methods.
        Tests H1: Methods produce different scores.
        
        Args:
            scores_a: Scores from method A.
            scores_b: Scores from method B.
            label_a: Display name for method A.
            label_b: Display name for method B.
            
        Returns:
            Dict with test statistic, p-value, and interpretation.
        """
        from scipy import stats

        a = np.array(scores_a)
        b = np.array(scores_b)

        if len(a) != len(b):
            raise ValueError(f"Score arrays must have equal length: {len(a)} vs {len(b)}")

        # Remove ties (pairs where a == b)
        diff = a - b
        non_zero = diff[diff != 0]

        if len(non_zero) < 2:
            return {
                "test": "Wilcoxon signed-rank",
                "comparison": f"{label_a} vs {label_b}",
                "statistic": None,
                "p_value": 1.0,
                "significant": False,
                "interpretation": "Insufficient non-tied pairs for testing.",
                "mean_diff": round(float(np.mean(diff)), 4),
            }

        stat, p_value = stats.wilcoxon(a, b)

        significant = p_value < self.alpha
        mean_diff = float(np.mean(b - a))

        if significant:
            direction = "higher" if mean_diff > 0 else "lower"
            interpretation = (
                f"{label_b} scores are significantly {direction} than {label_a} "
                f"(p={p_value:.4f} < α={self.alpha})"
            )
        else:
            interpretation = (
                f"No significant difference between {label_a} and {label_b} "
                f"(p={p_value:.4f} ≥ α={self.alpha})"
            )

        result = {
            "test": "Wilcoxon signed-rank",
            "comparison": f"{label_a} vs {label_b}",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": significant,
            "mean_diff": round(mean_diff, 4),
            "interpretation": interpretation,
        }

        logger.info(f"Wilcoxon: {label_a} vs {label_b} — p={p_value:.4f}, sig={significant}")
        return result

    # ------------------------------------------------------------------
    # Effect Size
    # ------------------------------------------------------------------

    def cohens_d(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> Dict[str, Any]:
        """
        Compute Cohen's d effect size for paired samples.
        
        Interpretation:
          |d| < 0.2  — Negligible
          0.2 ≤ |d| < 0.5 — Small
          0.5 ≤ |d| < 0.8 — Medium
          |d| ≥ 0.8  — Large
        """
        a = np.array(scores_a)
        b = np.array(scores_b)

        diff = b - a
        d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return {
            "cohens_d": round(d, 4),
            "magnitude": magnitude,
            "interpretation": f"Effect size is {magnitude} (d={d:.3f})",
        }

    # ------------------------------------------------------------------
    # Confidence Intervals
    # ------------------------------------------------------------------

    def confidence_interval(
        self,
        scores: List[float],
    ) -> Dict[str, Any]:
        """
        Compute confidence interval for the mean.
        
        Uses t-distribution for small samples.
        """
        from scipy import stats

        data = np.array(scores)
        n = len(data)
        mean = float(np.mean(data))
        se = float(stats.sem(data))

        ci = stats.t.interval(
            self.confidence_level, df=n - 1, loc=mean, scale=se
        )

        return {
            "mean": round(mean, 4),
            "std": round(float(np.std(data, ddof=1)), 4),
            "se": round(se, 4),
            "ci_lower": round(float(ci[0]), 4),
            "ci_upper": round(float(ci[1]), 4),
            "confidence_level": self.confidence_level,
            "n": n,
        }

    # ------------------------------------------------------------------
    # Correlation Analysis
    # ------------------------------------------------------------------

    def pearson_correlation(
        self,
        x: List[float],
        y: List[float],
        label_x: str = "X",
        label_y: str = "Y",
    ) -> Dict[str, Any]:
        """
        Compute Pearson correlation coefficient with significance test.
        
        Used to validate CLIP scores against human ratings.
        """
        from scipy import stats

        r, p = stats.pearsonr(x, y)

        if abs(r) < 0.3:
            strength = "weak"
        elif abs(r) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if r > 0 else "negative"

        return {
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "significant": p < self.alpha,
            "strength": strength,
            "direction": direction,
            "interpretation": (
                f"{strength.capitalize()} {direction} correlation between "
                f"{label_x} and {label_y} (r={r:.3f}, p={p:.4f})"
            ),
        }

    # ------------------------------------------------------------------
    # Ablation Study
    # ------------------------------------------------------------------

    def ablation_analysis(
        self,
        baseline_scores: List[float],
        stage1_only_scores: List[float],
        promptist_only_scores: List[float],
        hybrid_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Ablation study comparing individual and combined stages.
        
        Tests whether the hybrid approach (Stage 1 + Stage 2) provides
        statistically significant improvement over each stage alone.
        
        This is critical for justifying the two-stage architecture.
        """
        results = {
            "stage1_vs_baseline": self.wilcoxon_test(
                baseline_scores, stage1_only_scores,
                "Baseline", "Stage 1 Only"
            ),
            "promptist_vs_baseline": self.wilcoxon_test(
                baseline_scores, promptist_only_scores,
                "Baseline", "Promptist Only"
            ),
            "hybrid_vs_baseline": self.wilcoxon_test(
                baseline_scores, hybrid_scores,
                "Baseline", "Hybrid"
            ),
            "hybrid_vs_stage1": self.wilcoxon_test(
                stage1_only_scores, hybrid_scores,
                "Stage 1 Only", "Hybrid"
            ),
            "hybrid_vs_promptist": self.wilcoxon_test(
                promptist_only_scores, hybrid_scores,
                "Promptist Only", "Hybrid"
            ),
            "effect_sizes": {
                "hybrid_vs_baseline": self.cohens_d(baseline_scores, hybrid_scores),
                "hybrid_vs_promptist": self.cohens_d(promptist_only_scores, hybrid_scores),
            },
        }

        # Summary
        hybrid_wins = sum(
            1 for key in ["hybrid_vs_baseline", "hybrid_vs_stage1", "hybrid_vs_promptist"]
            if results[key]["significant"] and results[key]["mean_diff"] > 0
        )

        results["summary"] = {
            "hybrid_significant_wins": hybrid_wins,
            "total_comparisons": 3,
            "conclusion": (
                "Hybrid approach shows significant improvement over all individual methods."
                if hybrid_wins == 3
                else f"Hybrid approach shows significant improvement in {hybrid_wins}/3 comparisons."
            ),
        }

        return results

    # ------------------------------------------------------------------
    # Full Analysis Report
    # ------------------------------------------------------------------

    def full_analysis(
        self,
        baseline_scores: List[float],
        promptist_scores: List[float],
        hybrid_scores: List[float],
        clip_scores: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete statistical analysis suite.
        
        Returns a comprehensive report with all tests, effect sizes,
        and confidence intervals.
        """
        report = {
            "pairwise_tests": {
                "hybrid_vs_baseline": self.wilcoxon_test(
                    baseline_scores, hybrid_scores, "Baseline", "Hybrid"
                ),
                "hybrid_vs_promptist": self.wilcoxon_test(
                    promptist_scores, hybrid_scores, "Promptist", "Hybrid"
                ),
                "promptist_vs_baseline": self.wilcoxon_test(
                    baseline_scores, promptist_scores, "Baseline", "Promptist"
                ),
            },
            "effect_sizes": {
                "hybrid_vs_baseline": self.cohens_d(baseline_scores, hybrid_scores),
                "hybrid_vs_promptist": self.cohens_d(promptist_scores, hybrid_scores),
            },
            "confidence_intervals": {
                "baseline": self.confidence_interval(baseline_scores),
                "promptist": self.confidence_interval(promptist_scores),
                "hybrid": self.confidence_interval(hybrid_scores),
            },
        }

        # CLIP-Human correlation if available
        if clip_scores:
            all_human = baseline_scores + promptist_scores + hybrid_scores
            all_clip = (
                clip_scores.get("baseline", []) +
                clip_scores.get("promptist", []) +
                clip_scores.get("hybrid", [])
            )
            if len(all_human) == len(all_clip) and len(all_clip) > 2:
                report["clip_human_correlation"] = self.pearson_correlation(
                    all_human, all_clip,
                    "Human Fidelity", "CLIP Score"
                )

        return report
