"""
=============================================================================
Efficiency Benchmarking Module
=============================================================================
Measures and compares the computational efficiency of the prompt
optimization pipeline. This module provides the quantitative evidence
for the project's core thesis:

  "Optimized prompts improve the efficiency of text-to-image generation
   by reducing wasted computation on failed/low-quality outputs."

Efficiency Metrics:
  1. Generation Time — Wall-clock time per image
  2. Retry Rate — How many generations needed to achieve acceptable quality
  3. API Cost — LLM calls saved via caching and complexity routing
  4. Memory Usage — Peak RAM during generation
  5. Quality-per-Second — CLIP score divided by generation time
  6. First-Acceptable-Generation — Steps to reach CLIP threshold

The key insight is that prompt optimization has a small upfront cost
(LLM API call + Promptist inference) but saves significant downstream
computation by producing better results on the first try.
=============================================================================
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("hybrid_prompt_opt.efficiency")


@dataclass
class GenerationBenchmark:
    """Metrics for a single image generation attempt."""
    method: str
    prompt_index: int
    category: str
    generation_time_s: float
    clip_score: Optional[float] = None
    peak_memory_mb: float = 0.0
    inference_steps: int = 50
    was_acceptable: bool = False
    attempt_number: int = 1


@dataclass
class PipelineTimings:
    """End-to-end timing breakdown for the full pipeline."""
    stage1_rewrite_s: float = 0.0
    stage2_promptist_s: float = 0.0
    generation_s: float = 0.0
    clip_evaluation_s: float = 0.0
    total_s: float = 0.0
    overhead_pct: float = 0.0  # Optimization overhead as % of total


class EfficiencyBenchmark:
    """
    Comprehensive efficiency benchmarking for the optimization pipeline.
    
    Collects timing, memory, and quality metrics across all pipeline
    stages and methods, then computes comparative statistics.
    
    Args:
        output_dir: Directory to save benchmark results.
        clip_threshold: Minimum CLIP score to consider a generation "acceptable".
    """

    def __init__(
        self,
        output_dir: Path,
        clip_threshold: float = 0.25,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clip_threshold = clip_threshold

        self._benchmarks: List[GenerationBenchmark] = []
        self._pipeline_timings: Dict[str, PipelineTimings] = {}
        self._api_call_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_generation(
        self,
        method: str,
        prompt_index: int,
        category: str,
        generation_time_s: float,
        clip_score: Optional[float] = None,
        peak_memory_mb: float = 0.0,
        inference_steps: int = 50,
        attempt_number: int = 1,
    ):
        """Record metrics for a single generation attempt."""
        was_acceptable = (
            clip_score is not None and clip_score >= self.clip_threshold
        )

        benchmark = GenerationBenchmark(
            method=method,
            prompt_index=prompt_index,
            category=category,
            generation_time_s=generation_time_s,
            clip_score=clip_score,
            peak_memory_mb=peak_memory_mb,
            inference_steps=inference_steps,
            was_acceptable=was_acceptable,
            attempt_number=attempt_number,
        )

        self._benchmarks.append(benchmark)

    def record_pipeline_timing(
        self,
        method: str,
        stage1_time: float = 0.0,
        stage2_time: float = 0.0,
        generation_time: float = 0.0,
        clip_eval_time: float = 0.0,
    ):
        """Record end-to-end pipeline timing for a method."""
        total = stage1_time + stage2_time + generation_time + clip_eval_time
        overhead = stage1_time + stage2_time
        overhead_pct = (overhead / total * 100) if total > 0 else 0.0

        self._pipeline_timings[method] = PipelineTimings(
            stage1_rewrite_s=round(stage1_time, 3),
            stage2_promptist_s=round(stage2_time, 3),
            generation_s=round(generation_time, 3),
            clip_evaluation_s=round(clip_eval_time, 3),
            total_s=round(total, 3),
            overhead_pct=round(overhead_pct, 1),
        )

    def record_api_call(
        self,
        provider: str,
        latency_ms: float,
        cached: bool = False,
        tokens_used: int = 0,
    ):
        """Record an LLM API call for cost analysis."""
        self._api_call_log.append({
            "provider": provider,
            "latency_ms": round(latency_ms, 1),
            "cached": cached,
            "tokens_used": tokens_used,
        })

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_method_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare efficiency metrics across methods (baseline/promptist/hybrid).
        
        Returns per-method statistics including:
          - Average generation time
          - Average CLIP score
          - Quality-per-second ratio
          - Acceptance rate (% above CLIP threshold)
          - Average retry count
        """
        methods = set(b.method for b in self._benchmarks)
        comparison = {}

        for method in methods:
            method_benchmarks = [
                b for b in self._benchmarks if b.method == method
            ]

            times = [b.generation_time_s for b in method_benchmarks]
            scores = [
                b.clip_score for b in method_benchmarks
                if b.clip_score is not None
            ]
            memories = [b.peak_memory_mb for b in method_benchmarks]
            acceptable = [b.was_acceptable for b in method_benchmarks]

            avg_time = sum(times) / len(times) if times else 0
            avg_score = sum(scores) / len(scores) if scores else 0
            avg_memory = sum(memories) / len(memories) if memories else 0
            acceptance_rate = sum(acceptable) / len(acceptable) if acceptable else 0

            # Quality-per-second: how much CLIP score you get per second of compute
            qps = avg_score / avg_time if avg_time > 0 else 0

            comparison[method] = {
                "n_generations": len(method_benchmarks),
                "avg_generation_time_s": round(avg_time, 3),
                "avg_clip_score": round(avg_score, 4),
                "quality_per_second": round(qps, 4),
                "acceptance_rate": round(acceptance_rate, 4),
                "avg_peak_memory_mb": round(avg_memory, 2),
                "total_compute_time_s": round(sum(times), 2),
            }

        return comparison

    def compute_category_efficiency(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze efficiency gains per prompt category.
        
        Shows which categories benefit most from optimization,
        supporting the argument that complex prompts (spatial, counting,
        negation) see the largest efficiency improvements.
        """
        categories = set(b.category for b in self._benchmarks)
        analysis = {}

        for category in categories:
            cat_benchmarks = [
                b for b in self._benchmarks if b.category == category
            ]

            by_method = {}
            for method in ["baseline", "promptist", "hybrid"]:
                method_b = [b for b in cat_benchmarks if b.method == method]
                if method_b:
                    scores = [b.clip_score for b in method_b if b.clip_score]
                    times = [b.generation_time_s for b in method_b]
                    by_method[method] = {
                        "avg_clip": round(sum(scores) / len(scores), 4) if scores else 0,
                        "avg_time_s": round(sum(times) / len(times), 3) if times else 0,
                    }

            # Compute improvement ratios
            baseline_score = by_method.get("baseline", {}).get("avg_clip", 0)
            hybrid_score = by_method.get("hybrid", {}).get("avg_clip", 0)

            improvement = (
                (hybrid_score - baseline_score) / baseline_score * 100
                if baseline_score > 0
                else 0
            )

            analysis[category] = {
                "methods": by_method,
                "hybrid_improvement_pct": round(improvement, 1),
            }

        return analysis

    def compute_api_cost_savings(self) -> Dict[str, Any]:
        """
        Analyze API call efficiency and cost savings from caching.
        
        Quantifies how many API calls were avoided through the
        prompt cache and complexity-based routing.
        """
        total_calls = len(self._api_call_log)
        cached_calls = sum(1 for c in self._api_call_log if c["cached"])
        actual_calls = total_calls - cached_calls

        total_tokens = sum(c["tokens_used"] for c in self._api_call_log)
        total_latency = sum(c["latency_ms"] for c in self._api_call_log)

        return {
            "total_api_calls": total_calls,
            "cached_calls": cached_calls,
            "actual_api_calls": actual_calls,
            "cache_savings_pct": round(
                cached_calls / total_calls * 100 if total_calls > 0 else 0, 1
            ),
            "total_tokens_used": total_tokens,
            "total_api_latency_ms": round(total_latency, 1),
            "avg_call_latency_ms": round(
                total_latency / actual_calls if actual_calls > 0 else 0, 1
            ),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive efficiency report."""
        report = {
            "method_comparison": self.compute_method_comparison(),
            "category_efficiency": self.compute_category_efficiency(),
            "api_cost_savings": self.compute_api_cost_savings(),
            "pipeline_timings": {
                k: asdict(v) for k, v in self._pipeline_timings.items()
            },
            "summary": self._compute_summary(),
        }
        return report

    def save_report(self, filename: str = "efficiency_report.json"):
        """Save the efficiency report to JSON."""
        report = self.generate_report()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Efficiency report saved to {output_path}")
        return output_path

    def print_summary(self):
        """Print a formatted efficiency summary to console."""
        report = self.generate_report()
        summary = report["summary"]

        print("\n" + "=" * 60)
        print("EFFICIENCY BENCHMARK SUMMARY")
        print("=" * 60)

        comparison = report["method_comparison"]
        print(f"\n{'Method':<15} {'Avg Time':>10} {'Avg CLIP':>10} {'QPS':>10} {'Accept%':>10}")
        print("-" * 55)

        for method, stats in comparison.items():
            print(
                f"{method:<15} "
                f"{stats['avg_generation_time_s']:>9.2f}s "
                f"{stats['avg_clip_score']:>10.4f} "
                f"{stats['quality_per_second']:>10.4f} "
                f"{stats['acceptance_rate']:>9.1%}"
            )

        if summary.get("efficiency_gain_pct"):
            print(f"\nOverall efficiency gain (hybrid vs baseline): "
                  f"{summary['efficiency_gain_pct']:+.1f}%")

        api_savings = report["api_cost_savings"]
        if api_savings["total_api_calls"] > 0:
            print(f"\nAPI calls saved by cache: {api_savings['cached_calls']} "
                  f"({api_savings['cache_savings_pct']:.1f}%)")

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute high-level summary statistics."""
        comparison = self.compute_method_comparison()

        baseline_qps = comparison.get("baseline", {}).get("quality_per_second", 0)
        hybrid_qps = comparison.get("hybrid", {}).get("quality_per_second", 0)

        efficiency_gain = (
            (hybrid_qps - baseline_qps) / baseline_qps * 100
            if baseline_qps > 0
            else 0
        )

        return {
            "total_benchmarks": len(self._benchmarks),
            "methods_compared": list(comparison.keys()),
            "efficiency_gain_pct": round(efficiency_gain, 1),
            "clip_threshold": self.clip_threshold,
        }
