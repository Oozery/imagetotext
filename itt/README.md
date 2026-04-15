# Hybrid Prompt Optimization for Text-to-Image Generation

A two-stage prompt optimization pipeline that improves the efficiency and quality of text-to-image generation by rewriting prompts for better compositional accuracy and aesthetic appeal.

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────┐
│  Complexity Analyzer    │ ← Routes simple prompts directly to Stage 2
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Stage 1: LLM Rewriter  │ ← Compositional accuracy (spatial, counting, negation)
│  (Groq / Claude / GPT)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Stage 2: Promptist     │ ← Aesthetic style tokens (RL-trained GPT-2)
│  (Microsoft Promptist)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Stable Diffusion       │ ← Image generation with optimized prompt
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Evaluation             │ ← CLIP scores + Human Likert ratings
│  + Efficiency Metrics   │
└─────────────────────────┘
```

## Project Structure

```
├── main.py                    # Pipeline orchestrator
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
├── prompts/
│   ├── suite.py               # 40-prompt test suite (8 categories)
│   ├── cache.py               # Similarity-based prompt cache
│   └── complexity.py          # Complexity analyzer for routing
├── optimizers/
│   ├── compositional.py       # Stage 1: LLM compositional rewriter
│   └── promptist_optimizer.py # Stage 2: Promptist aesthetic optimizer
├── generation/
│   └── generator.py           # Stable Diffusion wrapper with profiling
├── evaluation/
│   ├── clip_eval.py           # CLIP score computation
│   ├── human_eval.py          # Human evaluation + Cohen's Kappa
│   └── efficiency.py          # Efficiency benchmarking
├── analysis/
│   ├── charts.py              # 7 visualization charts
│   └── statistics.py          # Wilcoxon, Cohen's d, confidence intervals
└── utils/
    └── helpers.py             # I/O and formatting utilities
```

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"  # or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

## Usage

```bash
# Run specific stages (uncomment in main.py)
python main.py

# Or use the Pipeline class programmatically
from main import Pipeline
from config import Config

cfg = Config()
cfg.validate()
pipeline = Pipeline(cfg)

# Run individual stages
pipeline.run_complexity_analysis()
stage1 = pipeline.run_stage1(use_complexity_routing=True)
all_results = pipeline.run_stage2(stage1)
pipeline.run_stage6()  # Analysis (works with demo data)
```

## Evaluation

- **CLIP Score**: Automated text-image alignment measurement
- **Human Evaluation**: 1-5 Likert scale for compositional fidelity and visual quality
- **Inter-Rater Reliability**: Cohen's Kappa for evaluator agreement
- **Efficiency Metrics**: Generation time, memory usage, quality-per-second, API cost savings
- **Statistical Tests**: Wilcoxon signed-rank, Cohen's d effect size, confidence intervals

## Test Suite Categories

| Category | Difficulty | Focus |
|----------|-----------|-------|
| Single Object | Easy | Baseline performance |
| Multi-Object Spatial | Hard | Spatial relationships |
| Counting | Hard | Numerical accuracy |
| Text Rendering | Very Hard | In-image text |
| Abstract/Metaphorical | Medium | Conceptual understanding |
| Cultural Specific | Medium | Cultural knowledge |
| Style Transfer | Medium | Artistic style |
| Negation | Hard | Element exclusion |
