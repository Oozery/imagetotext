"""
=============================================================================
Prompt Test Suite — 40 Prompts Across 8 Compositional Categories
=============================================================================
Each category targets a specific weakness in current text-to-image models:

  1. single_object        — Baseline; should be easy for all models
  2. multi_object_spatial — Tests spatial relationship understanding
  3. counting             — Tests numerical accuracy
  4. text_rendering       — Tests in-image text generation
  5. abstract_metaphorical — Tests conceptual/abstract understanding
  6. cultural_specific    — Tests cultural knowledge and representation
  7. style_transfer       — Tests artistic style adherence
  8. negation             — Tests ability to exclude elements

The suite is designed to provide a balanced evaluation across difficulty
levels, with 5 prompts per category for statistical reliability.
=============================================================================
"""

from typing import List, Dict, Optional


# ============================================================================
# Category Metadata
# ============================================================================

CATEGORY_METADATA = {
    "single_object": {
        "display_name": "Single Object",
        "difficulty": "easy",
        "description": "Simple single-subject prompts to establish baseline performance.",
        "known_failure_modes": ["attribute binding errors", "background hallucination"],
        "optimization_priority": "low",
    },
    "multi_object_spatial": {
        "display_name": "Multi-Object Spatial",
        "difficulty": "hard",
        "description": "Prompts requiring correct spatial arrangement of multiple objects.",
        "known_failure_modes": ["object merging", "incorrect positioning", "missing objects"],
        "optimization_priority": "high",
    },
    "counting": {
        "display_name": "Counting",
        "difficulty": "hard",
        "description": "Prompts specifying exact object counts.",
        "known_failure_modes": ["wrong count", "duplicate objects", "merged objects"],
        "optimization_priority": "high",
    },
    "text_rendering": {
        "display_name": "Text Rendering",
        "difficulty": "very_hard",
        "description": "Prompts requiring legible text within the generated image.",
        "known_failure_modes": ["garbled text", "missing characters", "wrong spelling"],
        "optimization_priority": "medium",
    },
    "abstract_metaphorical": {
        "display_name": "Abstract / Metaphorical",
        "difficulty": "medium",
        "description": "Conceptual prompts requiring visual metaphor interpretation.",
        "known_failure_modes": ["literal interpretation", "incoherent composition"],
        "optimization_priority": "medium",
    },
    "cultural_specific": {
        "display_name": "Cultural Specific",
        "difficulty": "medium",
        "description": "Prompts requiring cultural knowledge for accurate depiction.",
        "known_failure_modes": ["cultural inaccuracy", "stereotyping", "missing elements"],
        "optimization_priority": "medium",
    },
    "style_transfer": {
        "display_name": "Style Transfer",
        "difficulty": "medium",
        "description": "Prompts specifying a particular artistic style.",
        "known_failure_modes": ["style not applied", "content distortion"],
        "optimization_priority": "low",
    },
    "negation": {
        "display_name": "Negation",
        "difficulty": "hard",
        "description": "Prompts that explicitly exclude certain elements.",
        "known_failure_modes": ["negated element appears", "empty scene"],
        "optimization_priority": "high",
    },
}


# ============================================================================
# Prompt Test Suite
# ============================================================================

PROMPT_SUITE: Dict[str, List[str]] = {
    "single_object": [
        "A red apple on a wooden table",
        "A golden retriever sitting on grass",
        "A blue ceramic vase with sunflowers",
        "A vintage typewriter on a desk",
        "A steaming cup of coffee on a rainy windowsill",
    ],
    "multi_object_spatial": [
        "A cat sitting to the left of a dog on a park bench",
        "A red ball in front of a green box, with a blue triangle behind them",
        "A child standing between two tall trees in a meadow",
        "A bicycle leaning against a brick wall with a potted plant beside it",
        "A bird perched on top of a lamppost with a car parked below",
    ],
    "counting": [
        "Exactly three red roses in a glass vase",
        "Five colorful balloons floating in a blue sky",
        "Two cats and one dog sleeping together on a couch",
        "Four books stacked on a wooden shelf",
        "Seven stars arranged in a circle pattern in the night sky",
    ],
    "text_rendering": [
        "A coffee mug with the word HELLO printed on it",
        "A neon sign that reads OPEN in a dark alley",
        "A birthday cake with HAPPY BIRTHDAY written in icing",
        "A chalkboard with the equation E equals mc squared written on it",
        "A wooden signpost that says WELCOME TO THE FOREST",
    ],
    "abstract_metaphorical": [
        "The concept of loneliness visualized as a landscape",
        "Time passing, depicted as a melting clock in a desert",
        "The feeling of hope represented as light breaking through storm clouds",
        "Chaos and order existing side by side in a single image",
        "The weight of knowledge shown as a person carrying glowing books",
    ],
    "cultural_specific": [
        "A traditional South Indian wedding ceremony with flower decorations",
        "A Japanese tea ceremony in a minimalist tatami room",
        "A Mexican Day of the Dead altar with marigolds and candles",
        "A traditional Chinese dragon boat festival scene on a river",
        "An Indian Diwali celebration with diyas and rangoli patterns",
    ],
    "style_transfer": [
        "A modern cityscape in the style of a watercolor painting",
        "A portrait of an elderly woman in the style of Impressionism",
        "A forest scene rendered as a pencil sketch",
        "A futuristic city in the style of cyberpunk digital art",
        "A still life of fruits in the style of oil painting with thick brushstrokes",
    ],
    "negation": [
        "A park with green grass and trees but no people",
        "A kitchen table with plates and cups but no food",
        "A night sky full of stars with no moon visible",
        "A highway stretching into the distance with no cars",
        "An empty classroom with desks and a chalkboard but no students",
    ],
}


# ============================================================================
# Flattened Prompt List
# ============================================================================

def _build_flat_prompts() -> List[Dict[str, str]]:
    """Flatten the prompt suite into a list of dicts with category labels."""
    flat = []
    for category, prompts in PROMPT_SUITE.items():
        for prompt in prompts:
            flat.append({
                "category": category,
                "prompt": prompt,
            })
    return flat


ALL_PROMPTS: List[Dict[str, str]] = _build_flat_prompts()


# ============================================================================
# Utility Functions
# ============================================================================

def get_prompts_by_category(category: str) -> List[str]:
    """Retrieve all prompts for a given category."""
    if category not in PROMPT_SUITE:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Available: {list(PROMPT_SUITE.keys())}"
        )
    return PROMPT_SUITE[category]


def get_category_metadata(category: str) -> dict:
    """Get metadata (difficulty, description, etc.) for a category."""
    return CATEGORY_METADATA.get(category, {})


def get_high_priority_categories() -> List[str]:
    """Return categories marked as high optimization priority."""
    return [
        cat for cat, meta in CATEGORY_METADATA.items()
        if meta.get("optimization_priority") == "high"
    ]


def get_prompt_count() -> int:
    """Total number of prompts in the test suite."""
    return len(ALL_PROMPTS)


def get_category_names() -> List[str]:
    """Return ordered list of category names."""
    return list(PROMPT_SUITE.keys())


def get_category_display_labels() -> List[str]:
    """Return human-readable category labels for charts."""
    return [
        CATEGORY_METADATA[c]["display_name"]
        for c in PROMPT_SUITE.keys()
    ]
