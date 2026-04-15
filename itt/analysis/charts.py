"""
=============================================================================
Chart Generator — Visualization for Prompt Optimization Results
=============================================================================
Produces publication-quality charts for the project presentation:

  Chart 1: Grouped bar chart — Human compositional fidelity by category
  Chart 2: Grouped bar chart — CLIP scores by category
  Chart 3: Improvement heatmap — % gains across methods
  Chart 4: Radar chart — Model strengths across categories
  Chart 5: Scatter plot — CLIP vs human score correlation
  Chart 6: Efficiency comparison — Generation time vs quality
  Chart 7: Pipeline overhead breakdown — Time spent per stage

All charts use a consistent color scheme:
  - Baseline:  #94A3B8 (slate gray)
  - Promptist: #8B5CF6 (purple)
  - Hybrid:    #06B6D4 (cyan)
=============================================================================
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("hybrid_prompt_opt.charts")

# Consistent color scheme across all charts
COLORS = {
    "baseline": "#94A3B8",
    "promptist": "#8B5CF6",
    "hybrid": "#06B6D4",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "promptist": "Promptist",
    "hybrid": "Hybrid (Ours)",
}


class ChartGenerator:
    """
    Generates all visualization charts for the project.
    
    Supports both real data and demo/placeholder data for development.
    All charts are saved as high-DPI PNGs with timestamps to avoid
    overwriting previous versions.
    
    Args:
        output_dir: Directory to save chart images.
        categories: Ordered list of category names.
        category_labels: Human-readable category labels.
    """

    def __init__(
        self,
        output_dir: Path,
        categories: List[str],
        category_labels: List[str],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.categories = categories
        self.category_labels = category_labels

        # Demo data for development/testing
        self._demo_human = {
            "single_object":         {"baseline": 3.8, "promptist": 4.2, "hybrid": 4.3},
            "multi_object_spatial":  {"baseline": 2.1, "promptist": 2.5, "hybrid": 3.7},
            "counting":              {"baseline": 1.8, "promptist": 2.0, "hybrid": 3.1},
            "text_rendering":        {"baseline": 1.6, "promptist": 1.8, "hybrid": 1.9},
            "abstract_metaphorical": {"baseline": 2.4, "promptist": 3.4, "hybrid": 3.5},
            "cultural_specific":     {"baseline": 2.8, "promptist": 3.2, "hybrid": 3.4},
            "style_transfer":        {"baseline": 3.0, "promptist": 3.8, "hybrid": 3.9},
            "negation":              {"baseline": 1.5, "promptist": 1.7, "hybrid": 2.4},
        }

        self._demo_clip = {
            "single_object":         {"baseline": 0.28, "promptist": 0.31, "hybrid": 0.31},
            "multi_object_spatial":  {"baseline": 0.22, "promptist": 0.24, "hybrid": 0.29},
            "counting":              {"baseline": 0.20, "promptist": 0.21, "hybrid": 0.27},
            "text_rendering":        {"baseline": 0.18, "promptist": 0.19, "hybrid": 0.20},
            "abstract_metaphorical": {"baseline": 0.24, "promptist": 0.29, "hybrid": 0.30},
            "cultural_specific":     {"baseline": 0.23, "promptist": 0.26, "hybrid": 0.27},
            "style_transfer":        {"baseline": 0.26, "promptist": 0.30, "hybrid": 0.31},
            "negation":              {"baseline": 0.18, "promptist": 0.19, "hybrid": 0.23},
        }

    def _timestamp_path(self, base_name: str) -> Path:
        """Generate a timestamped output path."""
        ts = int(time.time())
        return self.output_dir / f"{base_name}_{ts}.png"

    # ------------------------------------------------------------------
    # Chart 1: Human Compositional Fidelity
    # ------------------------------------------------------------------

    def chart_human_fidelity(
        self,
        baseline_scores: Optional[List[float]] = None,
        promptist_scores: Optional[List[float]] = None,
        hybrid_scores: Optional[List[float]] = None,
    ) -> Path:
        """
        Grouped bar chart of human compositional fidelity ratings.
        
        Falls back to demo data if no scores provided.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if baseline_scores is None:
            baseline_scores = [self._demo_human[c]["baseline"] for c in self.categories]
            promptist_scores = [self._demo_human[c]["promptist"] for c in self.categories]
            hybrid_scores = [self._demo_human[c]["hybrid"] for c in self.categories]

        x = np.arange(len(self.categories))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width, baseline_scores, width,
                       label=METHOD_LABELS["baseline"], color=COLORS["baseline"],
                       edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x, promptist_scores, width,
                       label=METHOD_LABELS["promptist"], color=COLORS["promptist"],
                       edgecolor="white", linewidth=0.5)
        bars3 = ax.bar(x + width, hybrid_scores, width,
                       label=METHOD_LABELS["hybrid"], color=COLORS["hybrid"],
                       edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Prompt Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Avg Compositional Fidelity (1-5)", fontsize=12, fontweight="bold")
        ax.set_title("Human Evaluation: Compositional Fidelity by Category",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(self.category_labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 5.5)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        path = self._timestamp_path("chart1_human_compositional_fidelity")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 1 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 2: CLIP Scores
    # ------------------------------------------------------------------

    def chart_clip_scores(
        self,
        clip_baseline: Optional[List[float]] = None,
        clip_promptist: Optional[List[float]] = None,
        clip_hybrid: Optional[List[float]] = None,
    ) -> Path:
        """Grouped bar chart of CLIP scores by category."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if clip_baseline is None:
            clip_baseline = [self._demo_clip[c]["baseline"] for c in self.categories]
            clip_promptist = [self._demo_clip[c]["promptist"] for c in self.categories]
            clip_hybrid = [self._demo_clip[c]["hybrid"] for c in self.categories]

        x = np.arange(len(self.categories))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width, clip_baseline, width,
                       label=METHOD_LABELS["baseline"], color=COLORS["baseline"],
                       edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x, clip_promptist, width,
                       label=METHOD_LABELS["promptist"], color=COLORS["promptist"],
                       edgecolor="white", linewidth=0.5)
        bars3 = ax.bar(x + width, clip_hybrid, width,
                       label=METHOD_LABELS["hybrid"], color=COLORS["hybrid"],
                       edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Prompt Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("CLIP Score", fontsize=12, fontweight="bold")
        ax.set_title("Automated Evaluation: CLIP Score by Category",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(self.category_labels, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0.10, 0.38)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        path = self._timestamp_path("chart2_clip_scores")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 2 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 3: Improvement Heatmap
    # ------------------------------------------------------------------

    def chart_improvement_heatmap(
        self,
        baseline_scores: List[float],
        promptist_scores: List[float],
        hybrid_scores: List[float],
    ) -> Path:
        """Heatmap showing percentage improvement across methods."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        improvement_data = {}
        for i, c in enumerate(self.categories):
            if baseline_scores[i] > 0:
                improvement_data[self.category_labels[i]] = {
                    "Promptist vs Baseline": round(
                        (promptist_scores[i] - baseline_scores[i]) / baseline_scores[i] * 100, 1
                    ),
                    "Hybrid vs Baseline": round(
                        (hybrid_scores[i] - baseline_scores[i]) / baseline_scores[i] * 100, 1
                    ),
                    "Hybrid vs Promptist": round(
                        (hybrid_scores[i] - promptist_scores[i]) / promptist_scores[i] * 100, 1
                    ) if promptist_scores[i] > 0 else 0,
                }

        df_heatmap = pd.DataFrame(improvement_data).T

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="YlGnBu",
                    linewidths=1, linecolor="white",
                    cbar_kws={"label": "% Improvement"},
                    ax=ax, annot_kws={"fontsize": 11})
        ax.set_title("Percentage Improvement Over Baseline/Promptist",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("")

        plt.tight_layout()
        path = self._timestamp_path("chart3_improvement_heatmap")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 3 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 4: Radar Chart
    # ------------------------------------------------------------------

    def chart_radar(
        self,
        baseline_scores: List[float],
        promptist_scores: List[float],
        hybrid_scores: List[float],
    ) -> Path:
        """Radar chart showing model strengths across categories."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        angles = np.linspace(0, 2 * np.pi, len(self.categories), endpoint=False).tolist()
        angles += angles[:1]

        bl = baseline_scores + [baseline_scores[0]]
        pr = promptist_scores + [promptist_scores[0]]
        hy = hybrid_scores + [hybrid_scores[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, bl, "o-", linewidth=2, label=METHOD_LABELS["baseline"],
                color=COLORS["baseline"])
        ax.fill(angles, bl, alpha=0.1, color=COLORS["baseline"])
        ax.plot(angles, pr, "o-", linewidth=2, label=METHOD_LABELS["promptist"],
                color=COLORS["promptist"])
        ax.fill(angles, pr, alpha=0.1, color=COLORS["promptist"])
        ax.plot(angles, hy, "o-", linewidth=2, label=METHOD_LABELS["hybrid"],
                color=COLORS["hybrid"])
        ax.fill(angles, hy, alpha=0.15, color=COLORS["hybrid"])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.category_labels, fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_title("Compositional Fidelity Across Categories",
                     fontsize=14, fontweight="bold", pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

        plt.tight_layout()
        path = self._timestamp_path("chart4_radar")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 4 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 5: CLIP vs Human Correlation
    # ------------------------------------------------------------------

    def chart_clip_vs_human(
        self,
        baseline_scores: List[float],
        promptist_scores: List[float],
        hybrid_scores: List[float],
        clip_baseline: List[float],
        clip_promptist: List[float],
        clip_hybrid: List[float],
    ) -> Path:
        """Scatter plot showing correlation between CLIP and human scores."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats

        all_human = baseline_scores + promptist_scores + hybrid_scores
        all_clip = clip_baseline + clip_promptist + clip_hybrid
        methods = (
            ["Baseline"] * len(self.categories) +
            ["Promptist"] * len(self.categories) +
            ["Hybrid"] * len(self.categories)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for method_name in ["Baseline", "Promptist", "Hybrid"]:
            method_key = method_name.lower()
            if method_key == "hybrid":
                color = COLORS["hybrid"]
            else:
                color = COLORS.get(method_key, "#999999")

            h_scores = [all_human[i] for i in range(len(methods)) if methods[i] == method_name]
            c_scores = [all_clip[i] for i in range(len(methods)) if methods[i] == method_name]
            ax.scatter(h_scores, c_scores, label=method_name, color=color,
                      s=80, edgecolor="white", linewidth=0.5, zorder=3)

        # Pearson correlation
        r, p = stats.pearsonr(all_human, all_clip)

        ax.set_xlabel("Human Compositional Fidelity (1-5)", fontsize=12, fontweight="bold")
        ax.set_ylabel("CLIP Score", fontsize=12, fontweight="bold")
        ax.set_title(f"CLIP Score vs Human Rating (Pearson r = {r:.3f}, p = {p:.4f})",
                     fontsize=13, fontweight="bold", pad=15)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Trend line
        z = np.polyfit(all_human, all_clip, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(all_human), max(all_human), 100)
        ax.plot(x_line, p_line(x_line), "--", color="#EF4444", alpha=0.6, linewidth=1.5)

        plt.tight_layout()
        path = self._timestamp_path("chart5_clip_vs_human")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 5 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 6: Efficiency Comparison
    # ------------------------------------------------------------------

    def chart_efficiency_comparison(
        self,
        efficiency_data: Dict[str, Dict[str, Any]],
    ) -> Path:
        """
        Bar chart comparing quality-per-second across methods.
        
        This is the key efficiency visualization showing that optimized
        prompts achieve better quality per unit of compute time.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods = list(efficiency_data.keys())
        qps_values = [efficiency_data[m].get("quality_per_second", 0) for m in methods]
        time_values = [efficiency_data[m].get("avg_generation_time_s", 0) for m in methods]
        clip_values = [efficiency_data[m].get("avg_clip_score", 0) for m in methods]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Quality per second
        colors = [COLORS.get(m, "#999999") for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]

        axes[0].bar(labels, qps_values, color=colors, edgecolor="white", linewidth=0.5)
        axes[0].set_title("Quality per Second (CLIP/s)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("CLIP Score / Generation Time")
        axes[0].grid(axis="y", alpha=0.3)

        # Average generation time
        axes[1].bar(labels, time_values, color=colors, edgecolor="white", linewidth=0.5)
        axes[1].set_title("Avg Generation Time", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Seconds")
        axes[1].grid(axis="y", alpha=0.3)

        # Average CLIP score
        axes[2].bar(labels, clip_values, color=colors, edgecolor="white", linewidth=0.5)
        axes[2].set_title("Avg CLIP Score", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("CLIP Score")
        axes[2].grid(axis="y", alpha=0.3)

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("Efficiency Analysis: Quality vs Compute Cost",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = self._timestamp_path("chart6_efficiency_comparison")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 6 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Chart 7: Pipeline Overhead Breakdown
    # ------------------------------------------------------------------

    def chart_pipeline_overhead(
        self,
        pipeline_timings: Dict[str, Dict[str, float]],
    ) -> Path:
        """
        Stacked bar chart showing time breakdown per pipeline stage.
        
        Visualizes the overhead of prompt optimization relative to
        the total generation time.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods = list(pipeline_timings.keys())
        labels = [METHOD_LABELS.get(m, m) for m in methods]

        stage1_times = [pipeline_timings[m].get("stage1_rewrite_s", 0) for m in methods]
        stage2_times = [pipeline_timings[m].get("stage2_promptist_s", 0) for m in methods]
        gen_times = [pipeline_timings[m].get("generation_s", 0) for m in methods]
        eval_times = [pipeline_timings[m].get("clip_evaluation_s", 0) for m in methods]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(labels, stage1_times, label="Stage 1: LLM Rewrite", color="#F59E0B")
        ax.bar(labels, stage2_times, bottom=stage1_times,
               label="Stage 2: Promptist", color="#8B5CF6")
        ax.bar(labels, gen_times,
               bottom=[s1 + s2 for s1, s2 in zip(stage1_times, stage2_times)],
               label="Image Generation", color="#06B6D4")
        ax.bar(labels, eval_times,
               bottom=[s1 + s2 + g for s1, s2, g in zip(stage1_times, stage2_times, gen_times)],
               label="CLIP Evaluation", color="#10B981")

        ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Pipeline Time Breakdown by Method",
                     fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add overhead percentage labels
        for i, m in enumerate(methods):
            overhead_pct = pipeline_timings[m].get("overhead_pct", 0)
            total = pipeline_timings[m].get("total_s", 0)
            if total > 0:
                ax.annotate(f"Overhead: {overhead_pct:.1f}%",
                           xy=(i, total), xytext=(0, 8),
                           textcoords="offset points", ha="center",
                           fontsize=9, fontweight="bold", color="#374151")

        plt.tight_layout()
        path = self._timestamp_path("chart7_pipeline_overhead")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Chart 7 saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------

    def print_summary_table(
        self,
        baseline_scores: List[float],
        promptist_scores: List[float],
        hybrid_scores: List[float],
    ):
        """Print a formatted summary results table to console."""
        print("\n" + "=" * 70)
        print("SUMMARY RESULTS TABLE")
        print("=" * 70)
        print(f"{'Category':<25} {'Baseline':>8} {'Promptist':>10} {'Hybrid':>8} {'Improv.':>8}")
        print("-" * 70)

        for i, c in enumerate(self.categories):
            label = self.category_labels[i]
            bl_s = baseline_scores[i]
            pr_s = promptist_scores[i]
            hy_s = hybrid_scores[i]
            imp = ((hy_s - bl_s) / bl_s * 100) if bl_s > 0 else 0
            print(f"{label:<25} {bl_s:>8.1f} {pr_s:>10.1f} {hy_s:>8.1f} {imp:>+7.0f}%")

        overall_bl = np.mean(baseline_scores)
        overall_pr = np.mean(promptist_scores)
        overall_hy = np.mean(hybrid_scores)
        overall_imp = ((overall_hy - overall_bl) / overall_bl * 100) if overall_bl > 0 else 0

        print("-" * 70)
        print(f"{'OVERALL AVERAGE':<25} {overall_bl:>8.1f} {overall_pr:>10.1f} {overall_hy:>8.1f} {overall_imp:>+7.0f}%")
        print(f"\nAll charts saved to: {self.output_dir}/")

    # ------------------------------------------------------------------
    # Generate All Charts
    # ------------------------------------------------------------------

    def generate_all(
        self,
        baseline_scores: Optional[List[float]] = None,
        promptist_scores: Optional[List[float]] = None,
        hybrid_scores: Optional[List[float]] = None,
        clip_baseline: Optional[List[float]] = None,
        clip_promptist: Optional[List[float]] = None,
        clip_hybrid: Optional[List[float]] = None,
        efficiency_data: Optional[Dict] = None,
        pipeline_timings: Optional[Dict] = None,
    ) -> List[Path]:
        """Generate all charts and return list of saved paths."""
        paths = []

        # Use demo data if not provided
        if baseline_scores is None:
            baseline_scores = [self._demo_human[c]["baseline"] for c in self.categories]
            promptist_scores = [self._demo_human[c]["promptist"] for c in self.categories]
            hybrid_scores = [self._demo_human[c]["hybrid"] for c in self.categories]

        if clip_baseline is None:
            clip_baseline = [self._demo_clip[c]["baseline"] for c in self.categories]
            clip_promptist = [self._demo_clip[c]["promptist"] for c in self.categories]
            clip_hybrid = [self._demo_clip[c]["hybrid"] for c in self.categories]

        paths.append(self.chart_human_fidelity(baseline_scores, promptist_scores, hybrid_scores))
        paths.append(self.chart_clip_scores(clip_baseline, clip_promptist, clip_hybrid))
        paths.append(self.chart_improvement_heatmap(baseline_scores, promptist_scores, hybrid_scores))
        paths.append(self.chart_radar(baseline_scores, promptist_scores, hybrid_scores))
        paths.append(self.chart_clip_vs_human(
            baseline_scores, promptist_scores, hybrid_scores,
            clip_baseline, clip_promptist, clip_hybrid,
        ))

        if efficiency_data:
            paths.append(self.chart_efficiency_comparison(efficiency_data))

        if pipeline_timings:
            paths.append(self.chart_pipeline_overhead(pipeline_timings))

        self.print_summary_table(baseline_scores, promptist_scores, hybrid_scores)

        logger.info(f"Generated {len(paths)} charts")
        return paths
