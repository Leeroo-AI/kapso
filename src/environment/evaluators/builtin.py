# Built-in Evaluators
#
# Ready-to-use evaluators for common evaluation patterns.
# All evaluators are automatically registered via @register_evaluator decorator.
#
# Available evaluators:
# - no_score: No scoring, always returns 0
# - regex_pattern: Parse score from output using regex
# - file_json: Read score from JSON file
# - multi_metric: Weighted combination of multiple metrics
# - llm_judge: LLM-based evaluation
# - llm_comparison: LLM comparison against expected output
# - composite: Combine multiple evaluators

import json
import os
import re
from typing import Any, Dict, List, Optional

from src.environment.evaluators.base import Evaluator, EvaluationResult
from src.environment.evaluators.factory import register_evaluator


# =============================================================================
# Basic Evaluators
# =============================================================================

@register_evaluator("no_score")
class NoScoreEvaluator(Evaluator):
    """
    No scoring - always returns 0.
    
    Use when you only care about successful execution, not scoring.
    """
    
    description = "No scoring available and evaluation must be done based on solution and output. Provided experiment final score is fixed and invalid."
    requires_llm = False
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        return EvaluationResult(score=0.0, raw_output=output)


@register_evaluator("regex_pattern")
class RegexPatternEvaluator(Evaluator):
    """
    Extract score using regex pattern.
    
    Params:
        pattern: Regex with capture group for score (default: r'SCORE:\\s*([\\d.]+)')
        default_score: Score if pattern not found (default: 0.0)
    
    Example:
        evaluator = EvaluatorFactory.create(
            "regex_pattern",
            pattern=r"Accuracy: ([\\d.]+)%"
        )
    """
    
    description = "Extract score using regex pattern"
    requires_llm = False
    
    def __init__(
        self, 
        pattern: str = r'SCORE:\s*([-+]?\d*\.?\d+)',
        default_score: float = 0.0,
        **params
    ):
        super().__init__(**params)
        self.pattern = pattern
        self.default_score = default_score
        self._compiled = re.compile(pattern, re.IGNORECASE)
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        match = self._compiled.search(output)
        if match:
            score = float(match.group(1))
            return EvaluationResult(
                score=score,
                details={"matched": match.group(0)},
                raw_output=output,
            )
        return EvaluationResult(
            score=self.default_score,
            feedback="No score pattern found in output",
            raw_output=output,
        )


@register_evaluator("file_json")
class FileJsonEvaluator(Evaluator):
    """
    Read score from a JSON file created by the code.
    
    Params:
        filename: JSON file to read (default: "results.json")
        score_key: Key path to score (dot notation, default: "score")
        default_score: Score if file/key not found (default: 0.0)
        
    Example:
        evaluator = EvaluatorFactory.create(
            "file_json",
            filename="metrics.json",
            score_key="evaluation.f1_score"
        )
    """
    
    description = "Read score from JSON file"
    requires_llm = False
    
    def __init__(
        self,
        filename: str = "results.json",
        score_key: str = "score",
        default_score: float = 0.0,
        **params
    ):
        super().__init__(**params)
        self.filename = filename
        self.score_key = score_key
        self.default_score = default_score
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        file_path_full = os.path.join(file_path, self.filename)
        
        if not os.path.exists(file_path_full):
            return EvaluationResult(
                score=self.default_score,
                feedback=f"File not found: {self.filename}",
                raw_output=output,
            )
        
        try:
            with open(file_path_full) as f:
                data = json.load(f)
            
            # Navigate dot notation (e.g., "metrics.accuracy")
            score = data
            for key in self.score_key.split("."):
                score = score[key]
            
            return EvaluationResult(
                score=float(score),
                details=data,
                raw_output=output,
            )
        except Exception as e:
            return EvaluationResult(
                score=self.default_score,
                feedback=f"Error reading score: {e}",
                raw_output=output,
            )


@register_evaluator("multi_metric")
class MultiMetricEvaluator(Evaluator):
    """
    Extract multiple metrics and compute weighted score.
    
    Params:
        patterns: Dict of metric_name -> (pattern, weight)
        invert: List of metric names where lower is better
        
    Example:
        evaluator = EvaluatorFactory.create(
            "multi_metric",
            patterns={
                "accuracy": (r"Accuracy: ([\\d.]+)", 0.5),
                "f1": (r"F1: ([\\d.]+)", 0.3),
                "speed": (r"Time: ([\\d.]+)s", 0.2),
            },
            invert=["speed"],  # Lower time is better
        )
    """
    
    description = "Weighted combination of multiple regex metrics"
    requires_llm = False
    
    def __init__(
        self,
        patterns: Dict[str, tuple],  # {name: (pattern, weight)}
        invert: Optional[List[str]] = None,
        **params
    ):
        super().__init__(**params)
        self.patterns = {
            name: (re.compile(pat, re.IGNORECASE), weight)
            for name, (pat, weight) in patterns.items()
        }
        self.invert = set(invert or [])
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, (pattern, weight) in self.patterns.items():
            match = pattern.search(output)
            if match:
                value = float(match.group(1))
                # Invert if lower is better (assuming 0-1 scale)
                if name in self.invert:
                    value = 1.0 - min(value, 1.0)
                metrics[name] = value
                weighted_sum += value * weight
                total_weight += weight
        
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return EvaluationResult(
            score=score,
            details={
                "metrics": metrics, 
                "weights": {n: w for n, (_, w) in self.patterns.items()}
            },
            raw_output=output,
        )


# =============================================================================
# LLM-based Evaluators
# =============================================================================

@register_evaluator("llm_judge")
class LLMJudgeEvaluator(Evaluator):
    """
    Use LLM to evaluate code output quality.
    
    Params:
        criteria: What to evaluate (e.g., "correctness, efficiency, code quality")
        model: LLM model to use (default: "gpt-4.1-mini")
        scale: Score scale (default: 10)
        include_code: Whether to include source code in evaluation
        
    Example:
        evaluator = EvaluatorFactory.create(
            "llm_judge",
            criteria="correctness of the algorithm and output format",
            model="gpt-4.1",
        )
    """
    
    description = "LLM-based evaluation (flexible criteria)"
    requires_llm = True
    
    def __init__(
        self,
        criteria: str = "correctness and quality",
        model: str = "gpt-4.1-mini",
        scale: int = 10,
        include_code: bool = False,
        **params
    ):
        super().__init__(**params)
        self.criteria = criteria
        self.model = model
        self.scale = scale
        self.include_code = include_code
        
        # Lazy import to avoid circular dependency
        from src.core.llm import LLMBackend
        self.llm = LLMBackend()
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        # Build context for LLM
        problem = context.get("problem", "Not provided")
        solution = context.get("solution", "Not provided")
        
        # Optionally include code
        code_section = ""
        if self.include_code:
            main_file = os.path.join(file_path, "main.py")
            if os.path.exists(main_file):
                with open(main_file) as f:
                    code_section = f"\n\n<code>\n{f.read()}\n</code>"
        
        system_prompt = f"""You are an expert code evaluator. 
Evaluate the code execution output based on these criteria: {self.criteria}

Provide:
1. A score from 0 to {self.scale} (where {self.scale} is perfect)
2. Brief feedback explaining the score

Format your response as:
SCORE: <number>
FEEDBACK: <one paragraph explanation>
"""
        
        # Truncate output for context limits
        truncated_output = output[:3000] if len(output) > 3000 else output
        
        user_prompt = f"""
<problem>
{problem}
</problem>

<solution_description>
{solution}
</solution_description>

<execution_output>
{truncated_output}
</execution_output>
{code_section}

Evaluate this execution output.
"""
        
        try:
            response = self.llm.llm_completion_with_system_prompt(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_prompt,
            )
            
            # Parse response
            score_match = re.search(r'SCORE:\s*([\d.]+)', response)
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.0
            feedback = feedback_match.group(1).strip() if feedback_match else response
            
            # Normalize to 0-1 scale
            normalized_score = score / self.scale
            
            return EvaluationResult(
                score=normalized_score,
                feedback=feedback,
                details={"raw_score": score, "scale": self.scale, "model": self.model},
                raw_output=output,
            )
            
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                feedback=f"LLM evaluation failed: {e}",
                raw_output=output,
            )


@register_evaluator("llm_comparison")
class LLMComparisonEvaluator(Evaluator):
    """
    Use LLM to compare output against expected output or criteria.
    
    Params:
        expected: Expected output or description of correct behavior
        model: LLM model to use
        strict: If True, requires exact match; if False, allows semantic equivalence
    """
    
    description = "LLM-based comparison against expected output"
    requires_llm = True
    
    def __init__(
        self,
        expected: str = "",
        model: str = "gpt-4.1-mini",
        strict: bool = False,
        **params
    ):
        super().__init__(**params)
        self.expected = expected
        self.model = model
        self.strict = strict
        
        # Lazy import to avoid circular dependency
        from src.core.llm import LLMBackend
        self.llm = LLMBackend()
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        expected = self.expected or context.get("expected_output", "")
        
        strict_instruction = "Require exact match." if self.strict else \
            "Allow semantic equivalence - the meaning should match even if formatting differs."
        
        system_prompt = f"""You are a code output validator.
Compare the actual output to the expected output.

{strict_instruction}

Respond with:
MATCH: YES or NO or PARTIAL
SCORE: 0.0 to 1.0
REASON: Brief explanation
"""
        
        # Truncate for context limits
        truncated_output = output[:2000] if len(output) > 2000 else output
        
        user_prompt = f"""
<expected>
{expected}
</expected>

<actual>
{truncated_output}
</actual>
"""
        
        try:
            response = self.llm.llm_completion_with_system_prompt(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_prompt,
            )
            
            score_match = re.search(r'SCORE:\s*([\d.]+)', response)
            score = float(score_match.group(1)) if score_match else 0.0
            
            return EvaluationResult(
                score=score,
                feedback=response,
                raw_output=output,
            )
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                feedback=f"Comparison failed: {e}",
                raw_output=output,
            )


# =============================================================================
# Composite Evaluator
# =============================================================================

@register_evaluator("composite")
class CompositeEvaluator(Evaluator):
    """
    Combine multiple evaluators with weights.
    
    Params:
        evaluators: List of (evaluator_type, params, weight) tuples
        aggregation: "weighted_avg", "min", "max", "product"
        
    Example:
        evaluator = EvaluatorFactory.create(
            "composite",
            evaluators=[
                ("regex_pattern", {"pattern": r"Accuracy: ([\\d.]+)"}, 0.6),
                ("llm_judge", {"criteria": "code quality"}, 0.4),
            ],
            aggregation="weighted_avg",
        )
    """
    
    description = "Combine multiple evaluators"
    requires_llm = False  # Depends on sub-evaluators
    
    def __init__(
        self,
        evaluators: List[tuple],  # [(type, params, weight), ...]
        aggregation: str = "weighted_avg",
        **params
    ):
        super().__init__(**params)
        
        # Import factory here to avoid circular import
        from src.environment.evaluators.factory import EvaluatorFactory
        
        self.sub_evaluators = [
            (EvaluatorFactory.create(eval_type, **eval_params), weight)
            for eval_type, eval_params, weight in evaluators
        ]
        self.aggregation = aggregation
        
        # Check if any sub-evaluator requires LLM
        self.requires_llm = any(
            eval.requires_llm for eval, _ in self.sub_evaluators
        )
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        results = []
        for evaluator, weight in self.sub_evaluators:
            result = evaluator.evaluate(output, file_path, **context)
            results.append((result, weight))
        
        scores = [r.score for r, _ in results]
        weights = [w for _, w in results]
        
        # Aggregate scores based on method
        if self.aggregation == "weighted_avg":
            score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        elif self.aggregation == "min":
            score = min(scores)
        elif self.aggregation == "max":
            score = max(scores)
        elif self.aggregation == "product":
            score = 1.0
            for s in scores:
                score *= s
        else:
            # Default to average
            score = sum(scores) / len(scores)
        
        return EvaluationResult(
            score=score,
            details={
                "sub_scores": {
                    f"evaluator_{i}": {"score": res.score, "weight": w}
                    for i, ((eval, w), (res, _)) in enumerate(
                        zip(self.sub_evaluators, results)
                    )
                },
                "aggregation": self.aggregation,
            },
            raw_output=output,
        )

