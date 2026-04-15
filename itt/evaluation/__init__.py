"""
Evaluation module.

Provides automated (CLIP score) and human (Likert scale) evaluation
tools, plus efficiency benchmarking for the optimization pipeline.
"""

from evaluation.clip_eval import CLIPEvaluator
from evaluation.human_eval import HumanEvaluationManager
from evaluation.efficiency import EfficiencyBenchmark
