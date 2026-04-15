"""
=============================================================================
Stage 1 — LLM Compositional Rewriter
=============================================================================
Rewrites user prompts to improve compositional accuracy in text-to-image
generation. Targets the primary failure modes of diffusion models:

  - Incorrect object counts
  - Wrong spatial relationships
  - Ignored negation constraints
  - Ambiguous object descriptions

Supports multiple LLM backends (Claude, OpenAI, Groq) with automatic
fallback and retry logic. Integrates with the prompt cache to avoid
redundant API calls.

Architecture:
  CompositionalRewriter
    ├── _rewrite_claude()     → Anthropic Claude API
    ├── _rewrite_openai()     → OpenAI ChatCompletion API
    ├── _rewrite_groq()       → Groq (OpenAI-compatible) API
    └── _clean_output()       → Post-processing and sanitization
=============================================================================
"""

import re
import time
import logging
from typing import Optional, Dict, Any

from config import Config, LLMProviderConfig
from prompts.cache import PromptCache

logger = logging.getLogger("hybrid_prompt_opt.compositional")


# ============================================================================
# System Prompt for Compositional Rewriting
# ============================================================================

COMPOSITIONAL_SYSTEM_PROMPT = """You are a text-to-image prompt optimizer focused on COMPOSITIONAL ACCURACY.

Given a user's image description, rewrite it to maximize the chance that a text-to-image model
will generate ALL the described elements correctly. Specifically:

1. COUNTING: Make object counts explicit and emphatic
   - Preserve exact wording like "exactly", "at least", "no more than"
   - "some birds" → "exactly four birds, each clearly visible and distinct"

2. SPATIAL RELATIONSHIPS: Add precise spatial anchors
   - "a cat and dog" → "a cat positioned on the left side, a dog on the right side"

3. NEGATION: Convert negative instructions into positive framing where possible
   - "no people" → "an empty scene, completely devoid of any human figures"

4. OBJECT CLARITY: Ensure each object is described distinctly
   - "flowers" → "three red tulips, each with visible stems and petals"

5. SCENE GROUNDING: Add background/environment context
   - "on a table" → "on a rustic wooden table, indoor setting, soft ambient lighting"

6. NEGATIVE PROMPT SUGGESTIONS: Do NOT include any NEGATIVE prompt or bracketed text.


RULES:

CRITICAL:
Do NOT introduce ANY new attributes, environments, actions, or conditions that are not explicitly present in the original prompt.
If unsure, leave it unspecified.

- Keep the rewritten prompt under 80 words (excluding the negative prompt)
- Do NOT add elements the user didn't imply or mention
- Do NOT change the artistic intent or mood
- Preserve the user's core subject matter exactly
- NEVER introduce specific numbers (e.g., "three", "six") unless explicitly stated in the original prompt
- Do NOT add artistic styles (e.g., "well-painted", "oil painting") unless explicitly mentioned
- Do NOT introduce new attributes or conditions (e.g., "broken", "dirty", "partially eaten") that were not mentioned
- Do NOT use quotation marks in the output
- Avoid over-specifying details that were not implied (e.g., textures, conditions, or environment constraints beyond the prompt)
- Output ONLY the rewritten prompt, nothing else"""


class CompositionalRewriter:
    """
    Stage 1 optimizer that rewrites prompts for compositional accuracy.
    
    Uses an LLM to restructure prompts so that text-to-image models
    are more likely to generate all described elements correctly.
    
    Features:
      - Multi-provider support (Claude, OpenAI, Groq)
      - Automatic fallback on provider failure
      - Prompt caching to avoid redundant API calls
      - Output sanitization and length enforcement
      - Timing and token usage tracking for efficiency analysis
    
    Args:
        config: Pipeline configuration object.
        cache: Optional prompt cache for deduplication.
    """

    def __init__(self, config: Config, cache: Optional[PromptCache] = None):
        self.config = config
        self.cache = cache
        self.provider = config.get_active_provider()

        # Efficiency tracking
        self._call_count = 0
        self._cache_hits = 0
        self._total_latency = 0.0
        self._total_tokens = 0
        self._errors = []

    def rewrite(self, prompt: str) -> Optional[str]:
        """
        Rewrite a prompt for improved compositional accuracy.
        
        Checks cache first, then calls the configured LLM provider.
        Falls back to the original prompt on failure.
        
        Args:
            prompt: Original user prompt.
            
        Returns:
            Rewritten prompt string, or None on complete failure.
        """
        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(prompt)
            if cached is not None:
                self._cache_hits += 1
                logger.info(f"Cache hit for: {prompt[:50]}...")
                return cached

        # Call LLM
        start_time = time.time()
        result = self._dispatch(prompt)
        elapsed = time.time() - start_time

        self._call_count += 1
        self._total_latency += elapsed

        if result is not None:
            # Clean and validate output
            result = self._clean_output(result)

            # Store in cache
            if self.cache is not None:
                self.cache.put(prompt, result, metadata={
                    "provider": self.provider.name,
                    "latency_ms": round(elapsed * 1000),
                })

            logger.info(
                f"Rewrite complete ({elapsed:.2f}s): {prompt[:40]}... → {result[:40]}..."
            )
        else:
            logger.warning(f"Rewrite failed for: {prompt[:50]}...")

        return result

    def batch_rewrite(self, prompts: list) -> list:
        """
        Rewrite a batch of prompts with progress logging.
        
        Returns list of dicts with original and rewritten prompts.
        """
        results = []
        total = len(prompts)

        for i, item in enumerate(prompts):
            prompt = item["prompt"] if isinstance(item, dict) else item
            category = item.get("category", "unknown") if isinstance(item, dict) else "unknown"

            logger.info(f"[{i+1}/{total}] Rewriting ({category}): {prompt[:50]}...")

            rewritten = self.rewrite(prompt)
            if rewritten is None:
                rewritten = prompt  # Fallback to original
                logger.warning(f"  Using original as fallback")

            results.append({
                "category": category,
                "original": prompt,
                "stage1_rewritten": rewritten,
            })

        return results

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """Return efficiency metrics for this rewriter session."""
        avg_latency = (
            self._total_latency / self._call_count
            if self._call_count > 0
            else 0.0
        )
        return {
            "provider": self.provider.name,
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "total_latency_s": round(self._total_latency, 3),
            "avg_latency_ms": round(avg_latency * 1000, 1),
            "errors": len(self._errors),
            "calls_saved_by_cache": self._cache_hits,
        }

    # ------------------------------------------------------------------
    # Provider Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, prompt: str) -> Optional[str]:
        """Route to the configured LLM provider."""
        provider_map = {
            "claude": self._rewrite_claude,
            "openai": self._rewrite_openai,
            "groq": self._rewrite_groq,
        }

        handler = provider_map.get(self.provider.name)
        if handler is None:
            raise ValueError(f"Unknown provider: {self.provider.name}")

        try:
            return handler(prompt)
        except Exception as e:
            self._errors.append({"prompt": prompt[:50], "error": str(e)})
            logger.error(f"Provider error ({self.provider.name}): {e}")
            return None

    def _rewrite_claude(self, prompt: str) -> Optional[str]:
        """Rewrite using Anthropic Claude API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.provider.api_key)
        response = client.messages.create(
            model=self.provider.model_name,
            max_tokens=self.provider.max_tokens,
            system=COMPOSITIONAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _rewrite_openai(self, prompt: str) -> Optional[str]:
        """Rewrite using OpenAI ChatCompletion API."""
        import openai

        client = openai.OpenAI(api_key=self.provider.api_key)
        response = client.chat.completions.create(
            model=self.provider.model_name,
            max_tokens=self.provider.max_tokens,
            messages=[
                {"role": "system", "content": COMPOSITIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _rewrite_groq(self, prompt: str) -> Optional[str]:
        """Rewrite using Groq API (OpenAI-compatible endpoint)."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url,
        )
        response = client.chat.completions.create(
            model=self.provider.model_name,
            max_tokens=self.provider.max_tokens,
            temperature=self.provider.temperature,
            stop=self.provider.stop_sequences or None,
            messages=[
                {"role": "system", "content": COMPOSITIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Output Cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Sanitize LLM output to ensure clean prompt text.
        
        Removes:
          - Surrounding quotes
          - Newlines
          - Bracketed content (e.g., [NEGATIVE: ...])
          - Common LLM prefixes ("Rewritten:", etc.)
          - Stray tokens and artifacts
        
        Appends a standard negative prompt suffix for Stable Diffusion.
        """
        text = text.strip()

        # Remove surrounding quotes
        text = text.strip('"').strip("'")

        # Remove newlines
        text = text.replace("\n", " ")

        # Remove anything inside brackets
        text = re.sub(r"\[.*?\]", "", text)

        # Remove common LLM prefixes
        text = re.sub(r"(?i)^rewritten:\s*", "", text)
        text = re.sub(r"(?i)^optimized prompt:\s*", "", text)
        text = re.sub(r"(?i)^here is the rewritten.*?:\s*", "", text)

        # Remove stray tokens
        text = re.sub(r"\b[A-Z]*NATIVE:\b", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Hard truncate for safety
        text = text[:400]

        # Append standard negative prompt
        text += " [NEGATIVE: blurry, extra limbs, duplicate objects, text artifacts]"

        return text
