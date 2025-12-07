# Evaluators Module
#
# Pluggable evaluation functions for scoring code execution results.
#
# Quick Start:
#     from src.environment.evaluators import EvaluatorFactory
#     
#     # Create evaluator by name
#     evaluator = EvaluatorFactory.create("regex_pattern", pattern=r"Accuracy: ([\d.]+)")
#     
#     # Use evaluator
#     result = evaluator.evaluate(output, file_path)
#     score = result.score
#     
#     # List available evaluators
#     EvaluatorFactory.list_evaluators()
#
# Available Evaluators:
#     - no_score: No scoring, always returns 0
#     - regex_pattern: Parse score from output using regex
#     - file_json: Read score from JSON file
#     - multi_metric: Weighted combination of multiple metrics
#     - llm_judge: LLM-based evaluation
#     - llm_comparison: LLM comparison against expected output
#     - composite: Combine multiple evaluators

# Base classes
from src.environment.evaluators.base import (
    Evaluator,
    EvaluationResult,
)

# Factory and decorator
from src.environment.evaluators.factory import (
    EvaluatorFactory,
    register_evaluator,
)

# Built-in evaluators (import to register them)
from src.environment.evaluators.builtin import (
    NoScoreEvaluator,
    RegexPatternEvaluator,
    FileJsonEvaluator,
    MultiMetricEvaluator,
    LLMJudgeEvaluator,
    LLMComparisonEvaluator,
    CompositeEvaluator,
)

__all__ = [
    # Base
    "Evaluator",
    "EvaluationResult",
    # Factory
    "EvaluatorFactory",
    "register_evaluator",
    # Built-in evaluators
    "NoScoreEvaluator",
    "RegexPatternEvaluator",
    "FileJsonEvaluator",
    "MultiMetricEvaluator",
    "LLMJudgeEvaluator",
    "LLMComparisonEvaluator",
    "CompositeEvaluator",
]

