"""
=============================================================================
Hybrid Prompt Optimization for Text-to-Image Generation
=============================================================================
Main Pipeline Orchestrator

This script coordinates the full hybrid prompt optimization pipeline:

  Stage 1: LLM-based compositional rewriting (spatial, counting, negation)
  Stage 2: Promptist aesthetic optimization (style tokens via RL-trained GPT-2)
  Stage 3: Image generation via Stable Diffusion
  Stage 4: Automated evaluation (CLIP scores)
  Stage 5: Human evaluation template generation
  Stage 6: Statistical analysis and visualization
  Stage 7: Efficiency benchmarking and reporting

The pipeline is designed to be run incrementally — each stage saves its
output to disk, so you can resume from any point without re-running
earlier stages.

REQUIREMENTS:
  pip install transformers torch Pillow matplotlib pandas seaborn scipy
  pip install accelerate       # for Promptist (Stage 2)
  pip install openai           # for Groq/OpenAI LLM rewriting
  pip install anthropic        # if using Claude for rewriting
  pip install diffusers        # for Stable Diffusion generation

HOW TO RUN:
  1. Set your API key as an environment variable (e.g., GROQ_API_KEY)
  2. Uncomment the stages you want to run in the __main__ block
  3. Run: python main.py
=============================================================================
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# --- Project Modules ---
from config import Config
from prompts.suite import PROMPT_SUITE, ALL_PROMPTS, get_category_names, get_category_display_labels
from prompts.cache import PromptCache
from prompts.complexity import PromptComplexityAnalyzer
from optimizers.compositional import CompositionalRewriter
from optimizers.promptist_optimizer import PromptistOptimizer
from generation.generator import ImageGenerator
from evaluation.clip_eval import CLIPEvaluator
from evaluation.human_eval import HumanEvaluationManager
from evaluation.efficiency import EfficiencyBenchmark
from analysis.charts import ChartGenerator
from analysis.statistics import StatisticalAnalyzer
from utils.helpers import save_json, load_json, save_csv, format_duration


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class Pipeline:
    """
    Main pipeline orchestrator for the hybrid prompt optimization project.
    
    Coordinates all stages from prompt rewriting through evaluation and
    analysis, with efficiency tracking throughout.
    
    Args:
        config: Pipeline configuration object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = config.get_logger()

        # Ensure all output directories exist
        config.ensure_directories()

        # Initialize components
        self.cache = PromptCache(config.cache_dir)
        self.complexity_analyzer = PromptComplexityAnalyzer()
        self.rewriter = CompositionalRewriter(config, cache=self.cache)
        self.promptist = PromptistOptimizer(
            model_name=config.promptist.model_name,
            tokenizer_name=config.promptist.tokenizer_name,
            max_new_tokens=config.promptist.max_new_tokens,
            num_beams=config.promptist.num_beams,
            length_penalty=config.promptist.length_penalty,
        )
        self.generator = ImageGenerator(config)
        self.clip_evaluator = CLIPEvaluator(config.evaluation.clip_model_name)
        self.human_eval = HumanEvaluationManager(config.output_dir)
        self.efficiency = EfficiencyBenchmark(
            config.benchmarks_dir,
            clip_threshold=config.efficiency.acceptable_clip_threshold,
        )
        self.chart_gen = ChartGenerator(
            config.output_dir,
            get_category_names(),
            get_category_display_labels(),
        )
        self.stats = StatisticalAnalyzer(
            significance_level=config.evaluation.significance_level,
            confidence_level=config.evaluation.confidence_interval,
        )

        self.logger.info(f"Pipeline initialized: {config}")

    # ------------------------------------------------------------------
    # Stage 0: Complexity Analysis (Pre-processing)
    # ------------------------------------------------------------------

    def run_complexity_analysis(self) -> dict:
        """
        Analyze prompt complexity to determine optimal routing.
        
        Simple prompts skip Stage 1 (LLM rewriting) and go directly
        to Promptist, saving API calls and time.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 0: Prompt Complexity Analysis")
        self.logger.info("=" * 60)

        prompts = [item["prompt"] for item in ALL_PROMPTS]
        routing = self.complexity_analyzer.get_routing_summary(prompts)

        self.logger.info(f"Routing summary: {routing}")
        self.logger.info(
            f"  Full pipeline: {routing['full_pipeline']} prompts"
        )
        self.logger.info(
            f"  Promptist-only: {routing['promptist_only']} prompts"
        )
        self.logger.info(
            f"  Avg complexity: {routing['avg_complexity']:.3f}"
        )

        # Detailed per-prompt analysis
        reports = []
        for item in ALL_PROMPTS:
            report = self.complexity_analyzer.analyze(item["prompt"])
            reports.append({
                "category": item["category"],
                "prompt": item["prompt"],
                "complexity_level": report.level.value,
                "complexity_score": report.score,
                "skip_stage1": report.skip_stage1,
                "signals": report.signals,
                "reasoning": report.reasoning,
            })

        save_json(reports, self.config.output_dir / "complexity_analysis.json")
        return routing

    # ------------------------------------------------------------------
    # Stage 1: LLM Compositional Rewriting
    # ------------------------------------------------------------------

    def run_stage1(self, use_complexity_routing: bool = False) -> list:
        """
        Run Stage 1: LLM compositional rewriting on all prompts.
        
        Args:
            use_complexity_routing: If True, skip rewriting for simple prompts.
            
        Returns:
            List of dicts with original and rewritten prompts.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: LLM Compositional Rewriting")
        self.logger.info("=" * 60)

        start_time = time.time()
        results = []

        for i, item in enumerate(ALL_PROMPTS):
            self.logger.info(
                f"[{i+1}/{len(ALL_PROMPTS)}] {item['category']}: {item['prompt'][:50]}..."
            )

            # Complexity-based routing
            if use_complexity_routing:
                report = self.complexity_analyzer.analyze(item["prompt"])
                if report.skip_stage1:
                    self.logger.info(f"  Skipping Stage 1 (simple prompt)")
                    results.append({
                        "category": item["category"],
                        "original": item["prompt"],
                        "stage1_rewritten": item["prompt"],
                        "stage1_skipped": True,
                    })
                    continue

            rewritten = self.rewriter.rewrite(item["prompt"])
            if rewritten is None:
                rewritten = item["prompt"]
                self.logger.warning("  Using original as fallback")

            results.append({
                "category": item["category"],
                "original": item["prompt"],
                "stage1_rewritten": rewritten,
                "stage1_skipped": False,
            })

        elapsed = time.time() - start_time

        # Save results
        save_json(results, self.config.output_dir / "stage1_rewrites.json")

        # Save cache
        self.cache.save()

        # Log efficiency stats
        rewriter_stats = self.rewriter.get_efficiency_stats()
        cache_stats = self.cache.get_stats()
        self.logger.info(f"Stage 1 complete in {format_duration(elapsed)}")
        self.logger.info(f"  Rewriter stats: {rewriter_stats}")
        self.logger.info(f"  Cache stats: {cache_stats}")

        # Record timing for efficiency benchmark
        self.efficiency.record_pipeline_timing(
            "stage1", stage1_time=elapsed
        )

        return results

    # ------------------------------------------------------------------
    # Stage 2: Promptist Aesthetic Optimization
    # ------------------------------------------------------------------

    def run_stage2(self, stage1_results: list) -> list:
        """
        Run Stage 2: Promptist optimization on original and rewritten prompts.
        
        Returns enriched results with 'promptist_only' and 'hybrid' keys.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: Promptist Aesthetic Optimization")
        self.logger.info("=" * 60)

        start_time = time.time()
        results = self.promptist.batch_optimize(stage1_results)
        elapsed = time.time() - start_time

        # Save results
        save_json(results, self.config.output_dir / "all_prompts.json")
        save_csv(
            results,
            self.config.output_dir / "all_prompts.csv",
            fieldnames=["category", "original", "stage1_rewritten", "promptist_only", "hybrid"],
        )

        # Log stats
        promptist_stats = self.promptist.get_efficiency_stats()
        self.logger.info(f"Stage 2 complete in {format_duration(elapsed)}")
        self.logger.info(f"  Promptist stats: {promptist_stats}")

        return results

    # ------------------------------------------------------------------
    # Stage 3: Image Generation
    # ------------------------------------------------------------------

    def run_stage3(self, all_results: list) -> list:
        """
        Run Stage 3: Generate images using Stable Diffusion.
        
        Generates baseline, promptist, and hybrid images for each prompt.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: Image Generation (Stable Diffusion)")
        self.logger.info("=" * 60)

        start_time = time.time()
        metrics = self.generator.batch_generate(
            all_results, self.config.images_dir
        )
        elapsed = time.time() - start_time

        # Save generation metrics
        save_json(metrics, self.config.benchmarks_dir / "generation_metrics.json")

        gen_stats = self.generator.get_efficiency_summary()
        self.logger.info(f"Stage 3 complete in {format_duration(elapsed)}")
        self.logger.info(f"  Generation stats: {gen_stats}")

        return metrics

    # ------------------------------------------------------------------
    # Stage 4: CLIP Evaluation
    # ------------------------------------------------------------------

    def run_stage4_full(self, all_results: list) -> list:
        """Run CLIP evaluation on all generated images."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: CLIP Score Evaluation")
        self.logger.info("=" * 60)

        start_time = time.time()
        clip_results = self.clip_evaluator.evaluate_all(
            all_results, self.config.images_dir
        )
        elapsed = time.time() - start_time

        self.clip_evaluator.save_results(
            clip_results, self.config.output_dir / "clip_scores.json"
        )

        self.logger.info(f"Stage 4 complete in {format_duration(elapsed)}")
        return clip_results

    def run_stage4_single(self, item: dict, images_dir: str) -> dict:
        """Run CLIP evaluation on a single prompt's images."""
        self.logger.info("CLIP evaluation (single prompt)")
        return self.clip_evaluator.evaluate_single(item, Path(images_dir))

    # ------------------------------------------------------------------
    # Stage 5: Human Evaluation Templates
    # ------------------------------------------------------------------

    def run_stage5(self, all_results: list):
        """Generate human evaluation and bias audit templates."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: Human Evaluation Templates")
        self.logger.info("=" * 60)

        self.human_eval.create_evaluation_template(all_results)
        self.human_eval.create_bias_audit_template()

    # ------------------------------------------------------------------
    # Stage 6: Analysis and Visualization
    # ------------------------------------------------------------------

    def run_stage6(self):
        """
        Run full analysis: charts, statistical tests, and summary tables.
        Works with either real data or demo data.
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: Analysis and Visualization")
        self.logger.info("=" * 60)

        # Load data (real or demo)
        categories = get_category_names()

        # Try loading real human scores
        human_scores = self.human_eval.load_completed_scores()
        if human_scores:
            summary = self.human_eval.compute_summary(human_scores)
            baseline_scores = [summary.get(c, {}).get("baseline", 0) for c in categories]
            promptist_scores = [summary.get(c, {}).get("promptist", 0) for c in categories]
            hybrid_scores = [summary.get(c, {}).get("hybrid", 0) for c in categories]
        else:
            self.logger.info("Using demo data for human scores")
            baseline_scores = None
            promptist_scores = None
            hybrid_scores = None

        # Try loading real CLIP scores
        clip_path = self.config.output_dir / "clip_scores.json"
        clip_baseline = None
        clip_promptist = None
        clip_hybrid = None

        if clip_path.exists():
            clip_data = load_json(clip_path)
            import numpy as np
            clip_baseline = []
            clip_promptist = []
            clip_hybrid = []
            for c in categories:
                cat_items = [x for x in clip_data if x["category"] == c]
                clip_baseline.append(
                    np.mean([x["clip_baseline"] for x in cat_items if x["clip_baseline"]])
                    if cat_items else 0
                )
                clip_promptist.append(
                    np.mean([x["clip_promptist"] for x in cat_items if x["clip_promptist"]])
                    if cat_items else 0
                )
                clip_hybrid.append(
                    np.mean([x["clip_hybrid"] for x in cat_items if x["clip_hybrid"]])
                    if cat_items else 0
                )

        # Try loading efficiency data
        efficiency_data = None
        pipeline_timings = None
        efficiency_report_path = self.config.benchmarks_dir / "efficiency_report.json"
        if efficiency_report_path.exists():
            eff_report = load_json(efficiency_report_path)
            efficiency_data = eff_report.get("method_comparison")
            pipeline_timings = eff_report.get("pipeline_timings")

        # Generate all charts
        chart_paths = self.chart_gen.generate_all(
            baseline_scores=baseline_scores,
            promptist_scores=promptist_scores,
            hybrid_scores=hybrid_scores,
            clip_baseline=clip_baseline,
            clip_promptist=clip_promptist,
            clip_hybrid=clip_hybrid,
            efficiency_data=efficiency_data,
            pipeline_timings=pipeline_timings,
        )

        # Run statistical analysis if we have real scores
        if baseline_scores and promptist_scores and hybrid_scores:
            stat_report = self.stats.full_analysis(
                baseline_scores, promptist_scores, hybrid_scores,
                clip_scores={
                    "baseline": clip_baseline or [],
                    "promptist": clip_promptist or [],
                    "hybrid": clip_hybrid or [],
                } if clip_baseline else None,
            )
            save_json(stat_report, self.config.reports_dir / "statistical_analysis.json")
            self.logger.info("Statistical analysis saved")

        self.logger.info(f"Generated {len(chart_paths)} charts")

    # ------------------------------------------------------------------
    # Stage 7: Efficiency Report
    # ------------------------------------------------------------------

    def run_stage7(self):
        """Generate and save the efficiency benchmark report."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 7: Efficiency Benchmarking Report")
        self.logger.info("=" * 60)

        self.efficiency.save_report()
        self.efficiency.print_summary()

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_all(self):
        """Run the complete pipeline end-to-end."""
        self.logger.info("Starting full pipeline run")
        pipeline_start = time.time()

        # Save config for reproducibility
        self.config.save()

        # Pre-processing
        self.run_complexity_analysis()

        # Stage 1 & 2
        stage1_results = self.run_stage1(use_complexity_routing=True)
        all_results = self.run_stage2(stage1_results)

        # Stage 3 (image generation)
        self.run_stage3(all_results)

        # Stage 4 (CLIP evaluation)
        self.run_stage4_full(all_results)

        # Stage 5 (human eval templates)
        self.run_stage5(all_results)

        # Stage 6 (analysis)
        self.run_stage6()

        # Stage 7 (efficiency report)
        self.run_stage7()

        total_time = time.time() - pipeline_start
        self.logger.info(f"Full pipeline complete in {format_duration(total_time)}")


# ============================================================================
# Print Generation Instructions (for manual image generation)
# ============================================================================

def print_generation_instructions(results: list):
    """Print instructions for manually generating images."""
    print("\n" + "=" * 60)
    print("IMAGE GENERATION INSTRUCTIONS")
    print("=" * 60)
    print("""
    For each prompt, generate ONE image with each variant:
      a) Baseline: Use the 'original' or 'stage1_rewritten' prompt
      b) Promptist: Use the 'promptist_only' prompt
      c) Hybrid: Use the 'hybrid' prompt

    Save images as:
      project_outputs/generated_images/baseline/{category}_{index}.png
      project_outputs/generated_images/promptist/{category}_{index}.png
      project_outputs/generated_images/hybrid/{category}_{index}.png
    """)

    print("\n--- ALL PROMPTS FOR COPY-PASTE ---\n")
    for i, item in enumerate(results):
        print(f"[{i+1}] Category: {item['category']}")
        print(f"    BASELINE:  {item['original']}")
        print(f"    PROMPTIST: {item['promptist_only']}")
        print(f"    HYBRID:    {item['hybrid']}")
        print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID PROMPT OPTIMIZATION PIPELINE")
    print("=" * 60)
    print(f"Total prompts in test suite: {len(ALL_PROMPTS)}")
    print(f"Categories: {get_category_names()}")
    print()

    # --- Initialize pipeline ---
    cfg = Config(output_base="project_outputs")
    warnings = cfg.validate()
    for w in warnings:
        print(f"  [WARNING] {w}")

    pipeline = Pipeline(cfg)

    # =================================================================
    # Uncomment the stages you want to run.
    # Each stage saves output to disk, so you can resume from any point.
    # =================================================================

    # --- STAGE 0: Complexity Analysis ---
    # pipeline.run_complexity_analysis()

    # --- STAGE 1 & 2: Generate all prompt variants ---
    # stage1_results = pipeline.run_stage1(use_complexity_routing=True)
    # all_results = pipeline.run_stage2(stage1_results)
    # print_generation_instructions(all_results)

    # --- Load from saved files (if stages 1 & 2 already done) ---
    stage1_results = load_json(cfg.output_dir / "stage1_rewrites.json")
    all_results = load_json(cfg.output_dir / "all_prompts.json")

    # --- STAGE 3: Image Generation (Stable Diffusion) ---
    # pipeline.run_stage3(all_results)

    # --- STAGE 4: CLIP Evaluation (full suite) ---
    # clip_results = pipeline.run_stage4_full(all_results)

    # --- STAGE 4: CLIP Evaluation (single prompt for testing) ---
    images_dir = r"C:\Users\Kaashvi\Desktop\NITW\4_2\FinalYearProject\newest\basecodeFYproj\project_outputs\generated_images"
    single_item = all_results[1]
    pipeline.run_stage4_single(single_item, images_dir)

    # --- STAGE 5: Human Evaluation Templates ---
    # pipeline.run_stage5(all_results)

    # --- STAGE 6: Analysis and Charts (works with demo data) ---
    # pipeline.run_stage6()

    # --- STAGE 7: Efficiency Report ---
    # pipeline.run_stage7()

    # --- FULL PIPELINE (runs everything) ---
    # pipeline.run_all()

    # --- Inter-rater reliability example ---
    # from evaluation.human_eval import HumanEvaluationManager
    # rater1 = [4, 3, 5, 2, 3, 4, 1, 3, 4, 5]
    # rater2 = [4, 2, 5, 2, 4, 4, 1, 3, 3, 5]
    # result = HumanEvaluationManager.compute_cohens_kappa(rater1, rater2)
    # print(result)
