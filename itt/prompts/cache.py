"""
=============================================================================
Prompt Cache — Avoid Redundant LLM Calls for Similar Prompts
=============================================================================
Implements a similarity-based caching layer that stores successful prompt
rewrites and retrieves them when a sufficiently similar prompt is encountered.

This directly improves pipeline efficiency by:
  - Reducing API calls (cost savings)
  - Eliminating redundant computation for near-duplicate prompts
  - Providing instant results for cached prompts

Cache Strategy:
  - Exact match: Hash-based O(1) lookup
  - Fuzzy match: Token overlap similarity with configurable threshold
  - Persistence: JSON-backed file storage for cross-session reuse
=============================================================================
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import OrderedDict

logger = logging.getLogger("hybrid_prompt_opt.cache")


class PromptCache:
    """
    Similarity-aware prompt cache with LRU eviction and persistence.
    
    Stores prompt → rewrite mappings and supports both exact and fuzzy
    retrieval to minimize redundant LLM API calls.
    
    Args:
        cache_dir: Directory for persistent cache storage.
        max_size: Maximum number of cached entries (LRU eviction).
        similarity_threshold: Minimum token overlap ratio for fuzzy match (0.0–1.0).
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size: int = 500,
        similarity_threshold: float = 0.85,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "prompt_cache.json"
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        # Ordered dict for LRU behavior
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "fuzzy_hits": 0, "evictions": 0}

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> Optional[str]:
        """
        Look up a cached rewrite for the given prompt.
        
        Tries exact match first, then fuzzy match if enabled.
        Returns None on cache miss.
        """
        key = self._hash(prompt)

        # Exact match
        if key in self._cache:
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            logger.debug(f"Cache HIT (exact): {prompt[:50]}...")
            return self._cache[key]["rewrite"]

        # Fuzzy match
        best_match = self._fuzzy_lookup(prompt)
        if best_match is not None:
            self._stats["fuzzy_hits"] += 1
            logger.debug(f"Cache HIT (fuzzy): {prompt[:50]}...")
            return best_match

        self._stats["misses"] += 1
        return None

    def put(self, prompt: str, rewrite: str, metadata: Optional[dict] = None):
        """
        Store a prompt → rewrite mapping in the cache.
        
        Args:
            prompt: Original prompt text.
            rewrite: Optimized/rewritten prompt.
            metadata: Optional metadata (provider, timestamp, etc.).
        """
        key = self._hash(prompt)

        entry = {
            "prompt": prompt,
            "rewrite": rewrite,
            "tokens": self._tokenize(prompt),
            "metadata": metadata or {},
        }

        self._cache[key] = entry
        self._cache.move_to_end(key)

        # LRU eviction
        while len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Cache eviction: {evicted_key[:16]}...")

    def save(self):
        """Persist cache to disk."""
        serializable = {}
        for key, entry in self._cache.items():
            serializable[key] = {
                "prompt": entry["prompt"],
                "rewrite": entry["rewrite"],
                "metadata": entry.get("metadata", {}),
            }

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Cache saved: {len(self._cache)} entries → {self.cache_file}"
        )

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "fuzzy_hits": 0, "evictions": 0}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared.")

    def get_stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._stats["hits"] + self._stats["fuzzy_hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] + self._stats["fuzzy_hits"]) / total
            if total > 0
            else 0.0
        )
        return {
            **self._stats,
            "total_lookups": total,
            "hit_rate": round(hit_rate, 4),
            "cache_size": len(self._cache),
        }

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _load(self):
        """Load cache from disk if available."""
        if not self.cache_file.exists():
            logger.info("No existing cache found. Starting fresh.")
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, entry in data.items():
                self._cache[key] = {
                    "prompt": entry["prompt"],
                    "rewrite": entry["rewrite"],
                    "tokens": self._tokenize(entry["prompt"]),
                    "metadata": entry.get("metadata", {}),
                }

            logger.info(f"Cache loaded: {len(self._cache)} entries from {self.cache_file}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache file corrupted, starting fresh: {e}")
            self._cache.clear()

    def _hash(self, text: str) -> str:
        """Generate a deterministic hash key for a prompt."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def _tokenize(self, text: str) -> set:
        """Simple whitespace tokenization for similarity comparison."""
        return set(text.strip().lower().split())

    def _fuzzy_lookup(self, prompt: str) -> Optional[str]:
        """
        Find the most similar cached prompt using token overlap.
        
        Returns the cached rewrite if similarity exceeds threshold,
        otherwise None.
        """
        query_tokens = self._tokenize(prompt)
        if not query_tokens:
            return None

        best_score = 0.0
        best_rewrite = None

        for entry in self._cache.values():
            cached_tokens = entry["tokens"]
            if not cached_tokens:
                continue

            # Jaccard similarity
            intersection = len(query_tokens & cached_tokens)
            union = len(query_tokens | cached_tokens)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > best_score:
                best_score = similarity
                best_rewrite = entry["rewrite"]

        if best_score >= self.similarity_threshold:
            return best_rewrite

        return None

    def __len__(self):
        return len(self._cache)

    def __contains__(self, prompt: str):
        return self._hash(prompt) in self._cache

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"PromptCache(size={stats['cache_size']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
