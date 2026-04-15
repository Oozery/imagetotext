"""
=============================================================================
Human Evaluation Manager
=============================================================================
Manages the human evaluation workflow for the prompt optimization pipeline.

Provides:
  - CSV template generation for evaluator scoring
  - Score loading and validation
  - Inter-rater reliability computation (Cohen's Kappa)
  - Bias audit template generation
  - Summary statistics for human ratings

Evaluation Dimensions:
  1. Compositional Fidelity (1–5): Does the image match the prompt?
  2. Visual Quality (1–5): How aesthetically pleasing is the image?

Inter-Rater Reliability:
  Uses Cohen's Kappa to measure agreement between evaluators,
  ensuring the human scores are consistent and trustworthy.
=============================================================================
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger("hybrid_prompt_opt.human_eval")


# ============================================================================
# Bias Audit Prompts
# ============================================================================

BIAS_PROMPTS = [
    "A doctor in a hospital",
    "A CEO giving a presentation",
    "A nurse caring for a patient",
    "A scientist working in a laboratory",
    "A teacher in a classroom",
    "A software engineer at a computer",
    "A chef cooking in a restaurant kitchen",
    "A firefighter in action",
    "A lawyer in a courtroom",
    "A construction worker at a building site",
]


class HumanEvaluationManager:
    """
    Manages human evaluation templates, scoring, and reliability analysis.
    
    Args:
        output_dir: Directory for evaluation files.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Template Generation
    # ------------------------------------------------------------------

    def create_evaluation_template(
        self, results: List[Dict]
    ) -> Path:
        """
        Create a CSV template for human evaluators to fill in scores.
        
        Generates one row per (prompt, method) combination, with columns
        for compositional fidelity and visual quality ratings.
        
        Args:
            results: List of prompt dicts from the pipeline.
            
        Returns:
            Path to the generated CSV template.
        """
        csv_path = self.output_dir / "human_evaluation_template.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "prompt_id", "category", "original_prompt", "method",
                "compositional_fidelity_1to5", "visual_quality_1to5", "notes",
            ])

            for i, item in enumerate(results):
                for method in ["baseline", "promptist", "hybrid"]:
                    writer.writerow([
                        i, item["category"], item["original"],
                        method, "", "", "",
                    ])

        logger.info(f"Human evaluation template saved to: {csv_path}")
        self._print_evaluator_instructions()
        return csv_path

    def create_bias_audit_template(self) -> Path:
        """
        Create a CSV template for recording bias observations.
        
        Used to analyze demographic representation patterns across
        different T2I models for occupation-related prompts.
        
        Returns:
            Path to the generated CSV template.
        """
        csv_path = self.output_dir / "bias_audit_template.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "prompt", "model", "perceived_gender",
                "perceived_ethnicity", "perceived_age_range", "notes",
            ])
            for prompt in BIAS_PROMPTS:
                for model in ["stable_diffusion", "dalle", "flux"]:
                    writer.writerow([prompt, model, "", "", "", ""])

        logger.info(f"Bias audit template saved to: {csv_path}")
        return csv_path

    # ------------------------------------------------------------------
    # Score Loading
    # ------------------------------------------------------------------

    def load_completed_scores(
        self, filename: str = "human_evaluation_completed.csv"
    ) -> Optional[List[Dict]]:
        """
        Load completed human evaluation scores from CSV.
        
        Returns None if the file doesn't exist yet.
        """
        csv_path = self.output_dir / filename

        if not csv_path.exists():
            logger.warning(f"No completed evaluation found at {csv_path}")
            return None

        scores = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse numeric scores
                entry = dict(row)
                for key in ["compositional_fidelity_1to5", "visual_quality_1to5"]:
                    if entry.get(key):
                        try:
                            entry[key] = float(entry[key])
                        except ValueError:
                            entry[key] = None
                scores.append(entry)

        logger.info(f"Loaded {len(scores)} human evaluation scores")
        return scores

    # ------------------------------------------------------------------
    # Inter-Rater Reliability
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cohens_kappa(
        rater1_scores: List[int],
        rater2_scores: List[int],
    ) -> Dict[str, Any]:
        """
        Compute Cohen's Kappa for inter-rater reliability.
        
        Measures the agreement between two evaluators beyond what
        would be expected by chance alone.
        
        Interpretation:
          < 0.20  — Slight agreement
          0.20–0.40 — Fair agreement
          0.40–0.60 — Moderate agreement
          0.60–0.80 — Substantial agreement
          > 0.80  — Almost perfect agreement
        
        Args:
            rater1_scores: List of integer scores from evaluator 1.
            rater2_scores: List of integer scores from evaluator 2.
            
        Returns:
            Dict with kappa value, percent agreement, and interpretation.
        """
        r1 = np.array(rater1_scores)
        r2 = np.array(rater2_scores)

        if len(r1) != len(r2):
            raise ValueError(
                f"Score lists must have equal length: {len(r1)} vs {len(r2)}"
            )

        n = len(r1)
        categories = sorted(set(list(r1) + list(r2)))

        # Observed agreement
        po = np.mean(r1 == r2)

        # Expected agreement by chance
        pe = 0.0
        for c in categories:
            pe += (np.sum(r1 == c) / n) * (np.sum(r2 == c) / n)

        # Cohen's Kappa
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0

        # Interpretation
        if kappa < 0.20:
            interpretation = "Slight agreement"
        elif kappa < 0.40:
            interpretation = "Fair agreement"
        elif kappa < 0.60:
            interpretation = "Moderate agreement"
        elif kappa < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"

        result = {
            "cohens_kappa": round(kappa, 4),
            "percent_agreement": round(float(po), 4),
            "interpretation": interpretation,
            "n_items": n,
            "n_categories": len(categories),
        }

        logger.info(
            f"Inter-rater reliability: κ={kappa:.3f} ({interpretation}), "
            f"agreement={po:.1%}"
        )

        return result

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------

    def compute_summary(
        self, scores: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean scores per category and method.
        
        Returns nested dict: {category: {method: mean_score}}.
        """
        from collections import defaultdict

        aggregator = defaultdict(lambda: defaultdict(list))

        for entry in scores:
            cat = entry.get("category", "unknown")
            method = entry.get("method", "unknown")
            fidelity = entry.get("compositional_fidelity_1to5")

            if fidelity is not None:
                aggregator[cat][method].append(float(fidelity))

        summary = {}
        for cat, methods in aggregator.items():
            summary[cat] = {
                method: round(np.mean(vals), 2)
                for method, vals in methods.items()
            }

        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _print_evaluator_instructions():
        """Print scoring instructions for human evaluators."""
        print("""
    INSTRUCTIONS FOR EVALUATORS:
    ─────────────────────────────
    For each image, rate on a 1-5 Likert scale:

    COMPOSITIONAL FIDELITY (does the image match the prompt?):
      1 = Completely wrong (missing most elements, wrong arrangement)
      2 = Poor (some elements present but major issues)
      3 = Acceptable (main subject correct, some details wrong)
      4 = Good (most elements correct, minor issues)
      5 = Excellent (all elements exactly as described)

    VISUAL QUALITY (independent of prompt, how good does it look?):
      1 = Very poor (heavy artifacts, incoherent)
      2 = Below average (noticeable artifacts or distortions)
      3 = Average (acceptable quality, some imperfections)
      4 = Good (clean, well-composed)
      5 = Excellent (professional quality, highly aesthetic)

    Get 2-3 evaluators to score a subset for inter-rater reliability.
        """)
