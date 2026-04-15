"""
=============================================================================
Configuration Module — Hybrid Prompt Optimization Pipeline
=============================================================================
Centralized configuration for all pipeline stages, model parameters,
evaluation settings, and output paths.

This module provides a single source of truth for:
  - API keys and provider selection
  - Model hyperparameters for each stage
  - Directory structure and output paths
  - Evaluation thresholds and parameters
  - Logging configuration
  - Efficiency benchmarking settings

Usage:
    from config import Config
    cfg = Config()
    cfg.validate()
=============================================================================
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ============================================================================
# Logging Setup
# ============================================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure project-wide logging with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files.
        level: Logging level (default INFO).
    
    Returns:
        Root logger instance configured for the project.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger("hybrid_prompt_opt")
    logger.setLevel(level)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    # File handler — captures everything
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    # Console handler — info and above
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# Provider Configuration
# ============================================================================

@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider used in Stage 1 rewriting."""
    name: str
    api_key: Optional[str] = None
    model_name: str = ""
    max_tokens: int = 300
    temperature: float = 0.2
    base_url: Optional[str] = None
    stop_sequences: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if this provider has a valid API key configured."""
        return self.api_key is not None and len(self.api_key) > 0


# ============================================================================
# Promptist (Stage 2) Configuration
# ============================================================================

@dataclass
class PromptistConfig:
    """Hyperparameters for the Promptist aesthetic optimization model."""
    model_name: str = "microsoft/Promptist"
    tokenizer_name: str = "gpt2"
    max_new_tokens: int = 75
    num_beams: int = 8
    num_return_sequences: int = 1
    length_penalty: float = -1.0
    do_sample: bool = False


# ============================================================================
# Image Generation Configuration
# ============================================================================

@dataclass
class GenerationConfig:
    """Settings for Stable Diffusion image generation."""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    fallback_model_id: str = "stabilityai/sd-turbo"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    use_attention_slicing: bool = True
    device: str = "cpu"  # "cuda" if GPU available
    seed: Optional[int] = 42  # For reproducibility; None for random


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Parameters for CLIP and human evaluation."""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    likert_scale_min: int = 1
    likert_scale_max: int = 5
    min_raters_for_reliability: int = 2
    significance_level: float = 0.05
    confidence_interval: float = 0.95


# ============================================================================
# Efficiency Benchmarking Configuration
# ============================================================================

@dataclass
class EfficiencyConfig:
    """Settings for efficiency measurement and benchmarking."""
    enable_timing: bool = True
    enable_memory_profiling: bool = True
    warmup_runs: int = 1
    benchmark_runs: int = 3
    track_api_calls: bool = True
    track_token_usage: bool = True
    max_retries_baseline: int = 5
    max_retries_optimized: int = 5
    acceptable_clip_threshold: float = 0.25


# ============================================================================
# Main Configuration Class
# ============================================================================

class Config:
    """
    Master configuration object for the entire pipeline.
    
    Loads API keys from environment variables, sets up directory structure,
    and provides validated access to all sub-configurations.
    
    Usage:
        cfg = Config()
        cfg.validate()
        logger = cfg.get_logger()
    """

    def __init__(self, output_base: str = "project_outputs"):
        # --- API Keys ---
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # --- Active LLM provider ---
        self.llm_provider = os.getenv("LLM_PROVIDER", "groq")

        # --- Directory Structure ---
        self.output_dir = Path(output_base)
        self.images_dir = self.output_dir / "generated_images"
        self.logs_dir = self.output_dir / "logs"
        self.cache_dir = self.output_dir / "prompt_cache"
        self.benchmarks_dir = self.output_dir / "benchmarks"
        self.reports_dir = self.output_dir / "reports"

        # --- Sub-configurations ---
        self.providers = self._init_providers()
        self.promptist = PromptistConfig()
        self.generation = GenerationConfig()
        self.evaluation = EvaluationConfig()
        self.efficiency = EfficiencyConfig()

        # --- Logger (lazy init) ---
        self._logger = None

    def _init_providers(self) -> Dict[str, LLMProviderConfig]:
        """Initialize all supported LLM provider configurations."""
        return {
            "claude": LLMProviderConfig(
                name="claude",
                api_key=self.anthropic_api_key,
                model_name="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.0,
            ),
            "openai": LLMProviderConfig(
                name="openai",
                api_key=self.openai_api_key,
                model_name="gpt-4o-mini",
                max_tokens=300,
                temperature=0.0,
            ),
            "groq": LLMProviderConfig(
                name="groq",
                api_key=self.groq_api_key,
                model_name="llama-3.1-8b-instant",
                max_tokens=200,
                temperature=0.2,
                base_url="https://api.groq.com/openai/v1",
                stop_sequences=["[NEGATIVE", "\n\n"],
            ),
        }

    def get_active_provider(self) -> LLMProviderConfig:
        """Return the currently selected LLM provider config."""
        if self.llm_provider not in self.providers:
            raise ValueError(
                f"Unknown LLM provider '{self.llm_provider}'. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[self.llm_provider]

    def get_logger(self) -> logging.Logger:
        """Get or create the project logger."""
        if self._logger is None:
            self._logger = setup_logging(self.logs_dir)
        return self._logger

    def ensure_directories(self):
        """Create all required output directories."""
        dirs = [
            self.output_dir,
            self.images_dir,
            self.logs_dir,
            self.cache_dir,
            self.benchmarks_dir,
            self.reports_dir,
        ]
        for method in ["baseline", "promptist", "hybrid"]:
            dirs.append(self.images_dir / method)

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of warnings.
        Raises ValueError if critical configuration is missing.
        """
        warnings = []
        provider = self.get_active_provider()

        if not provider.is_available():
            raise ValueError(
                f"API key for provider '{self.llm_provider}' is not set. "
                f"Set the corresponding environment variable."
            )

        # Non-critical warnings
        for name, prov in self.providers.items():
            if not prov.is_available():
                warnings.append(f"Provider '{name}' API key not configured.")

        if self.generation.device == "cpu":
            warnings.append(
                "Running on CPU — image generation will be slow. "
                "Set generation.device='cuda' if GPU is available."
            )

        if self.generation.seed is None:
            warnings.append(
                "No random seed set — results will not be reproducible."
            )

        return warnings

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary for logging/saving."""
        return {
            "llm_provider": self.llm_provider,
            "output_dir": str(self.output_dir),
            "generation": {
                "model_id": self.generation.model_id,
                "height": self.generation.height,
                "width": self.generation.width,
                "steps": self.generation.num_inference_steps,
                "guidance_scale": self.generation.guidance_scale,
                "device": self.generation.device,
                "seed": self.generation.seed,
            },
            "promptist": {
                "model": self.promptist.model_name,
                "beams": self.promptist.num_beams,
                "max_tokens": self.promptist.max_new_tokens,
            },
            "evaluation": {
                "clip_model": self.evaluation.clip_model_name,
                "significance_level": self.evaluation.significance_level,
            },
            "efficiency": {
                "timing": self.efficiency.enable_timing,
                "memory_profiling": self.efficiency.enable_memory_profiling,
                "benchmark_runs": self.efficiency.benchmark_runs,
            },
        }

    def save(self, path: Optional[Path] = None):
        """Save current configuration to JSON."""
        if path is None:
            path = self.output_dir / "pipeline_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from a saved JSON file (partial restore)."""
        cfg = cls()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            cfg.llm_provider = data.get("llm_provider", cfg.llm_provider)
            gen = data.get("generation", {})
            cfg.generation.device = gen.get("device", cfg.generation.device)
            cfg.generation.seed = gen.get("seed", cfg.generation.seed)
            cfg.generation.num_inference_steps = gen.get("steps", cfg.generation.num_inference_steps)
        return cfg

    def __repr__(self):
        return (
            f"Config(provider={self.llm_provider}, "
            f"device={self.generation.device}, "
            f"seed={self.generation.seed}, "
            f"output={self.output_dir})"
        )
