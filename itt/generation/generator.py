"""
=============================================================================
Image Generator — Stable Diffusion Inference with Efficiency Tracking
=============================================================================
Wraps the HuggingFace diffusers pipeline for Stable Diffusion with:

  - Automatic negative prompt extraction from [NEGATIVE: ...] tags
  - Seed control for reproducible generation
  - Per-image timing and memory profiling
  - Batch generation with progress logging
  - CPU/GPU device management
  - Configurable inference steps for speed/quality tradeoff

The efficiency tracking here is central to the project thesis:
  "Better prompts → fewer wasted generations → more efficient pipeline"

By measuring generation time and tracking retry counts, we can
quantify the efficiency gains from prompt optimization.
=============================================================================
"""

import os
import re
import time
import logging
import tracemalloc
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

from config import Config

logger = logging.getLogger("hybrid_prompt_opt.generator")


class ImageGenerator:
    """
    Stable Diffusion image generator with efficiency instrumentation.
    
    Manages the diffusers pipeline lifecycle and provides detailed
    per-generation metrics for efficiency analysis.
    
    Args:
        config: Pipeline configuration object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.gen_config = config.generation

        self._pipe = None
        self._is_loaded = False
        self._load_time = 0.0

        # Per-generation metrics
        self._generation_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pipeline Loading
    # ------------------------------------------------------------------

    def load(self):
        """
        Load the Stable Diffusion pipeline.
        
        Uses the primary model ID from config, with fallback to the
        lighter turbo model if loading fails.
        """
        if self._is_loaded:
            return

        from diffusers import StableDiffusionPipeline
        import torch

        logger.info(f"Loading Stable Diffusion: {self.gen_config.model_id}")
        logger.info(f"Device: {self.gen_config.device}")

        start = time.time()

        try:
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.gen_config.model_id
            ).to(self.gen_config.device)
        except Exception as e:
            logger.warning(
                f"Failed to load {self.gen_config.model_id}: {e}. "
                f"Falling back to {self.gen_config.fallback_model_id}"
            )
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.gen_config.fallback_model_id
            ).to(self.gen_config.device)

        # CPU optimization
        if self.gen_config.device == "cpu":
            self._pipe.enable_attention_slicing()
            logger.info("Attention slicing enabled for CPU inference")

        self._load_time = time.time() - start
        self._is_loaded = True
        logger.info(f"Pipeline loaded in {self._load_time:.1f}s")

    def ensure_loaded(self):
        """Load pipeline if not already loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Prompt Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def split_prompt(prompt: str) -> Tuple[str, str]:
        """
        Extract main prompt and negative prompt from tagged text.
        
        Parses the [NEGATIVE: ...] tag appended by the compositional
        rewriter and returns (main_prompt, negative_prompt).
        
        Args:
            prompt: Full prompt text, possibly containing [NEGATIVE: ...]
            
        Returns:
            Tuple of (main_prompt, negative_prompt).
        """
        if "[NEGATIVE:" in prompt:
            main, neg = prompt.split("[NEGATIVE:")
            neg = neg.replace("]", "").strip()
            return main.strip(), neg
        return prompt.strip(), ""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single image from a prompt.
        
        Args:
            prompt: Full prompt text (may include [NEGATIVE: ...]).
            output_path: File path to save the generated image.
            seed: Random seed for reproducibility (overrides config).
            num_steps: Inference steps (overrides config).
            
        Returns:
            Dict with generation metadata and timing information.
        """
        import torch

        self.ensure_loaded()

        main_prompt, neg_prompt = self.split_prompt(prompt)
        actual_seed = seed if seed is not None else self.gen_config.seed
        actual_steps = num_steps if num_steps is not None else self.gen_config.num_inference_steps

        # Set seed for reproducibility
        generator = None
        if actual_seed is not None:
            generator = torch.Generator(device=self.gen_config.device)
            generator.manual_seed(actual_seed)

        # Memory tracking
        tracemalloc.start()
        start_time = time.time()

        image = self._pipe(
            main_prompt,
            negative_prompt=neg_prompt if neg_prompt else None,
            height=self.gen_config.height,
            width=self.gen_config.width,
            num_inference_steps=actual_steps,
            guidance_scale=self.gen_config.guidance_scale,
            generator=generator,
        ).images[0]

        elapsed = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        # Log metrics
        metrics = {
            "prompt": main_prompt[:100],
            "negative_prompt": neg_prompt[:100] if neg_prompt else "",
            "output_path": str(output_path),
            "seed": actual_seed,
            "steps": actual_steps,
            "resolution": f"{self.gen_config.width}x{self.gen_config.height}",
            "generation_time_s": round(elapsed, 3),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "device": self.gen_config.device,
        }

        self._generation_log.append(metrics)
        logger.info(
            f"Generated: {output_path} ({elapsed:.1f}s, "
            f"{peak_mem / (1024*1024):.0f}MB peak)"
        )

        return metrics

    def batch_generate(
        self,
        results: List[Dict],
        images_dir: Path,
        methods: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate images for all prompt variants in the results.
        
        For each prompt, generates images using:
          - baseline: Stage 1 rewritten prompt
          - promptist: Promptist-only prompt
          - hybrid: Full hybrid pipeline prompt
        
        Args:
            results: List of prompt dicts with all variant keys.
            images_dir: Base directory for generated images.
            methods: Which methods to generate (default: all three).
            
        Returns:
            List of generation metric dicts.
        """
        if methods is None:
            methods = ["baseline", "promptist", "hybrid"]

        method_key_map = {
            "baseline": "stage1_rewritten",
            "promptist": "promptist_only",
            "hybrid": "hybrid",
        }

        all_metrics = []
        total = len(results)

        for idx, item in enumerate(results):
            category = item.get("category", "unknown")
            logger.info(f"[{idx+1}/{total}] Generating images for: {category}")

            for method in methods:
                prompt_key = method_key_map[method]
                prompt = item.get(prompt_key, item.get("original", ""))

                output_path = images_dir / method / f"img_{idx}.png"

                try:
                    metrics = self.generate(prompt, str(output_path))
                    metrics["method"] = method
                    metrics["category"] = category
                    metrics["prompt_index"] = idx
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Generation failed ({method}, idx={idx}): {e}")
                    all_metrics.append({
                        "method": method,
                        "category": category,
                        "prompt_index": idx,
                        "error": str(e),
                    })

            logger.info(f"  Done generating for prompt {idx+1}")

        return all_metrics

    # ------------------------------------------------------------------
    # Efficiency Stats
    # ------------------------------------------------------------------

    def get_generation_log(self) -> List[Dict[str, Any]]:
        """Return the full generation log with per-image metrics."""
        return self._generation_log

    def get_efficiency_summary(self) -> Dict[str, Any]:
        """
        Compute aggregate efficiency statistics across all generations.
        
        Returns timing, memory, and throughput metrics.
        """
        if not self._generation_log:
            return {"total_generations": 0}

        times = [m["generation_time_s"] for m in self._generation_log if "generation_time_s" in m]
        mems = [m["peak_memory_mb"] for m in self._generation_log if "peak_memory_mb" in m]

        return {
            "model": self.gen_config.model_id,
            "device": self.gen_config.device,
            "model_load_time_s": round(self._load_time, 2),
            "total_generations": len(self._generation_log),
            "total_time_s": round(sum(times), 2),
            "avg_time_per_image_s": round(sum(times) / len(times), 3) if times else 0,
            "min_time_s": round(min(times), 3) if times else 0,
            "max_time_s": round(max(times), 3) if times else 0,
            "avg_peak_memory_mb": round(sum(mems) / len(mems), 2) if mems else 0,
            "max_peak_memory_mb": round(max(mems), 2) if mems else 0,
        }
