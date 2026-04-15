"""
=============================================================================
Utility Functions
=============================================================================
Common helper functions used across the pipeline modules.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("hybrid_prompt_opt.utils")


def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")


def load_json(path: Path) -> Any:
    """Load data from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict],
    path: Path,
    fieldnames: List[str] = None,
):
    """Save a list of dicts to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        logger.warning(f"No data to save to {path}")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"Saved CSV ({len(data)} rows): {path}")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_banner(title: str, width: int = 60):
    """Print a formatted section banner."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_progress(current: int, total: int, label: str = ""):
    """Print a progress indicator."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) {label}", end="", flush=True)
    if current == total:
        print()
