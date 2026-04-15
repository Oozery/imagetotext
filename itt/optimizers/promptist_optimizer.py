"""
=============================================================================
Stage 2 — Promptist Aesthetic Optimizer
=============================================================================
Applies Microsoft's Promptist model to add aesthetic style tokens that
improve visual quality in Stable Diffusion outputs.

Promptist is an RL-fine-tuned GPT-2 model trained to append style tokens
(e.g., "trending on artstation, highly detailed, 8k") that maximize
aesthetic scores from a CLIP-based reward model.

Reference:
  Hao et al., "Optimizing Prompts for Text-to-Image Generation" (2022)
  https://arxiv.org/abs/2212.09611

This module handles:
  - Model loading with caching (avoids re-downloading)
  - Batch optimization with progress tracking
  - Memory-efficient inference (no gradient computation)
  - Timing instrumentation for efficiency analysis
=============================================================================
"""

import time
import logging
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger("hybrid_prompt_opt.promptist")


class PromptistOptimizer:
    """
    Stage 2 optimizer using Microsoft's Promptist for aesthetic enhancement.
    
    Loads the pretrained Promptist model and applies it to prompts to
    append style tokens that improve Stable Diffusion output quality.
    
    The model is loaded lazily on first use and cached for subsequent calls.
    
    Args:
        model_name: HuggingFace model identifier for Promptist.
        tokenizer_name: Tokenizer to use (default: gpt2).
        max_new_tokens: Maximum tokens to generate.
        num_beams: Beam search width.
        length_penalty: Penalty for output length (negative = shorter).
    """

    def __init__(
        self,
        model_name: str = "microsoft/Promptist",
        tokenizer_name: str = "gpt2",
        max_new_tokens: int = 75,
        num_beams: int = 8,
        length_penalty: float = -1.0,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.length_penalty = length_penalty

        # Lazy-loaded model and tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        # Efficiency tracking
        self._call_count = 0
        self._total_latency = 0.0
        self._load_time = 0.0

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load(self):
        """
        Load the Promptist model and tokenizer from HuggingFace.
        
        Downloads ~500MB on first run, cached locally afterwards.
        """
        if self._is_loaded:
            logger.debug("Promptist already loaded, skipping.")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Promptist model: {self.model_name}")
        logger.info("(First run downloads ~500MB, subsequent runs use cache)")

        start = time.time()

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._load_time = time.time() - start
        self._is_loaded = True

        logger.info(f"Promptist loaded in {self._load_time:.1f}s")

    def ensure_loaded(self):
        """Load model if not already loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self, prompt: str) -> str:
        """
        Optimize a single prompt using Promptist.
        
        Appends " Rephrase:" to the input and generates an aesthetically
        enhanced version using beam search.
        
        Args:
            prompt: Input prompt to optimize.
            
        Returns:
            Aesthetically optimized prompt string.
        """
        import torch

        self.ensure_loaded()

        start = time.time()

        input_text = prompt.strip() + " Rephrase:"
        input_ids = self._tokenizer(
            input_text, return_tensors="pt"
        ).input_ids

        eos_id = self._tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                num_return_sequences=1,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                length_penalty=self.length_penalty,
            )

        output_text = self._tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        # Strip the input prefix from the output
        result = output_text.replace(input_text, "").strip()

        elapsed = time.time() - start
        self._call_count += 1
        self._total_latency += elapsed

        logger.debug(f"Promptist ({elapsed:.2f}s): {prompt[:40]}... → {result[:40]}...")

        return result

    def batch_optimize(
        self,
        stage1_results: List[Dict],
    ) -> List[Dict]:
        """
        Run Promptist on both original and Stage 1 rewritten prompts.
        
        For each prompt, generates:
          - promptist_only: Promptist applied to the original prompt
          - hybrid: Promptist applied to the Stage 1 rewrite
        
        This allows direct comparison of:
          1. Baseline (original prompt, no optimization)
          2. Promptist-only (aesthetic optimization only)
          3. Hybrid (compositional + aesthetic optimization)
        
        Args:
            stage1_results: List of dicts from Stage 1 with 'original'
                           and 'stage1_rewritten' keys.
        
        Returns:
            Enriched list of dicts with 'promptist_only' and 'hybrid' keys.
        """
        self.ensure_loaded()

        results = []
        total = len(stage1_results)

        for i, item in enumerate(stage1_results):
            logger.info(f"[{i+1}/{total}] Promptist optimization ({item.get('category', '?')})")

            # Promptist on original (for Promptist-only baseline)
            promptist_only = self.optimize(item["original"])
            logger.info(f"  Promptist-only: {promptist_only[:80]}...")

            # Promptist on Stage 1 rewrite (the hybrid approach)
            hybrid = self.optimize(item["stage1_rewritten"])
            logger.info(f"  Hybrid: {hybrid[:80]}...")

            results.append({
                **item,
                "promptist_only": promptist_only,
                "hybrid": hybrid,
            })

        return results

    # ------------------------------------------------------------------
    # Efficiency Stats
    # ------------------------------------------------------------------

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Return timing and usage statistics."""
        avg_latency = (
            self._total_latency / self._call_count
            if self._call_count > 0
            else 0.0
        )
        return {
            "model": self.model_name,
            "model_load_time_s": round(self._load_time, 2),
            "total_optimizations": self._call_count,
            "total_latency_s": round(self._total_latency, 3),
            "avg_latency_ms": round(avg_latency * 1000, 1),
        }

    def __repr__(self):
        status = "loaded" if self._is_loaded else "not loaded"
        return f"PromptistOptimizer({self.model_name}, {status})"
