"""
=============================================================================
CLIP Score Evaluation
=============================================================================
Computes CLIP (Contrastive Language-Image Pre-training) scores to measure
text-image alignment between generated images and their source prompts.

CLIP Score quantifies how well an image matches its text description,
providing an automated proxy for compositional fidelity. While not a
perfect substitute for human evaluation, it enables large-scale
comparison across methods and categories.

Methodology:
  1. Encode the text prompt using CLIP's text encoder
  2. Encode the generated image using CLIP's vision encoder
  3. Compute cosine similarity between the two embeddings
  4. Normalize to a 0–1 scale

Higher scores indicate better text-image alignment.

Reference:
  Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision" (2021) — https://arxiv.org/abs/2103.00020
=============================================================================
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger("hybrid_prompt_opt.clip_eval")


class CLIPEvaluator:
    """
    CLIP-based automated evaluation for text-image alignment.
    
    Loads the CLIP model lazily and caches it for repeated evaluations.
    Supports both single-image and batch evaluation modes.
    
    Args:
        model_name: HuggingFace CLIP model identifier.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._load_time = 0.0

        # Evaluation tracking
        self._eval_count = 0
        self._total_latency = 0.0

    def load(self):
        """Load CLIP model and processor from HuggingFace."""
        if self._is_loaded:
            return

        from transformers import CLIPProcessor, CLIPModel

        logger.info(f"Loading CLIP model: {self.model_name}")
        start = time.time()

        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

        self._load_time = time.time() - start
        self._is_loaded = True
        logger.info(f"CLIP loaded in {self._load_time:.1f}s")

    def ensure_loaded(self):
        """Load model if not already loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Single Image Evaluation
    # ------------------------------------------------------------------

    def compute_score(self, image_path: str, prompt: str) -> Optional[float]:
        """
        Compute CLIP score between an image and a text prompt.
        
        Args:
            image_path: Path to the generated image file.
            prompt: Text prompt to compare against.
            
        Returns:
            Normalized CLIP score (0–1), or None on error.
        """
        import torch
        from PIL import Image

        self.ensure_loaded()

        start = time.time()

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Normalized cosine similarity
            score = outputs.logits_per_image.item() / 100.0
            score = round(score, 4)

            elapsed = time.time() - start
            self._eval_count += 1
            self._total_latency += elapsed

            logger.debug(f"CLIP score: {score:.4f} — {Path(image_path).name}")
            return score

        except Exception as e:
            logger.error(f"CLIP evaluation error ({image_path}): {e}")
            return None

    # ------------------------------------------------------------------
    # Batch Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        results: List[Dict],
        images_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Compute CLIP scores for all generated images across all methods.
        
        Evaluates baseline, promptist, and hybrid images for each prompt
        in the results list.
        
        Args:
            results: List of prompt dicts with all variant keys.
            images_dir: Base directory containing method subdirectories.
            
        Returns:
            List of dicts with CLIP scores per method.
        """
        self.ensure_loaded()

        clip_results = []
        total = len(results)

        for i, item in enumerate(results):
            category = item.get("category", "unknown")
            idx = i % 5  # Index within category

            logger.info(f"[{i+1}/{total}] CLIP evaluation: {category}_{idx}")

            scores = {}
            for method, prompt_key in [
                ("baseline", "original"),
                ("promptist", "promptist_only"),
                ("hybrid", "hybrid"),
            ]:
                img_path = images_dir / method / f"{category}_{idx}.png"
                if img_path.exists():
                    score = self.compute_score(str(img_path), item["original"])
                    scores[method] = score
                    logger.info(f"  [{method}] CLIP={score}")
                else:
                    scores[method] = None
                    logger.warning(f"  [{method}] Image not found: {img_path}")

            clip_results.append({
                "category": category,
                "prompt": item["original"],
                "clip_baseline": scores.get("baseline"),
                "clip_promptist": scores.get("promptist"),
                "clip_hybrid": scores.get("hybrid"),
            })

        return clip_results

    def evaluate_single(
        self,
        item: Dict,
        images_dir: Path,
    ) -> Dict[str, Any]:
        """
        Compute CLIP scores for a single prompt across all methods.
        
        Used for quick testing with individual prompts.
        
        Args:
            item: Prompt dict with variant keys.
            images_dir: Base directory containing method subdirectories.
            
        Returns:
            Dict with CLIP scores per method.
        """
        self.ensure_loaded()

        idx = 0
        category = item.get("category", "single")
        images_dir = Path(images_dir)

        scores = {}
        for method, prompt_key in [
            ("baseline", "stage1_rewritten"),
            ("promptist", "promptist_only"),
            ("hybrid", "hybrid"),
        ]:
            img_path = images_dir / method / f"img_{idx}.png"
            if img_path.exists():
                score = self.compute_score(str(img_path), item[prompt_key])
                scores[method] = score
                logger.info(f"[{method}] CLIP={score}")
            else:
                scores[method] = None
                logger.warning(f"[{method}] Image not found: {img_path}")

        result = {
            "category": category,
            "prompt": item.get("original", ""),
            "clip_baseline": scores.get("baseline"),
            "clip_promptist": scores.get("promptist"),
            "clip_hybrid": scores.get("hybrid"),
        }

        logger.info(f"CLIP results: {result}")
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, results: List[Dict], output_path: Path):
        """Save CLIP evaluation results to JSON."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"CLIP scores saved to {output_path}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return evaluation performance statistics."""
        avg_latency = (
            self._total_latency / self._eval_count
            if self._eval_count > 0
            else 0.0
        )
        return {
            "model": self.model_name,
            "load_time_s": round(self._load_time, 2),
            "total_evaluations": self._eval_count,
            "total_latency_s": round(self._total_latency, 3),
            "avg_latency_ms": round(avg_latency * 1000, 1),
        }
