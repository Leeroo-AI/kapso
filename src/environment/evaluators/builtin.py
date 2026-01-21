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
# - f1_score: Extract F1 score from output (common ML metric)
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


@register_evaluator("f1_score")
class F1ScoreEvaluator(Evaluator):
    """
    Extract F1 score from output.
    
    Supports multiple common output formats:
    - "F1: 0.85" or "F1 Score: 0.85"
    - "f1_score: 0.85" or "f1-score: 0.85"
    - "F1=0.85" or "f1 = 0.85"
    - Percentage format: "F1: 85%" (auto-converts to 0.85)
    - JSON in output: {"f1": 0.85} or {"f1_score": 0.85}
    
    Params:
        default_score: Score if F1 not found (default: 0.0)
        source: Where to look - "output", "file", or "both" (default: "both")
        filename: JSON file to check if source includes "file" (default: "results.json")
        json_key: Key path in JSON file (default: "f1_score", also tries "f1")
        
    Example:
        # Simple usage - just pass evaluator="f1_score"
        solution = kapso.evolve(
            goal="Train model with F1 > 0.85",
            evaluator="f1_score",
            stop_condition="threshold",
            stop_condition_params={"threshold": 0.85},
        )
        
        # With custom JSON file
        solution = kapso.evolve(
            goal="Train model",
            evaluator="f1_score",
            evaluator_params={"source": "file", "filename": "metrics.json"},
            stop_condition="threshold",
            stop_condition_params={"threshold": 0.90},
        )
    """
    
    description = "Extract F1 score from output or file"
    requires_llm = False
    
    # Common patterns for F1 score in output
    # Ordered by specificity - more specific patterns first
    F1_PATTERNS = [
        # Explicit F1 patterns with various separators
        r'f1[_\-\s]*score\s*[:=]\s*([\d.]+)\s*%?',  # f1_score: 0.85 or f1-score = 85%
        r'f1\s*[:=]\s*([\d.]+)\s*%?',               # F1: 0.85 or f1 = 85%
        r'f1[_\-\s]*macro\s*[:=]\s*([\d.]+)\s*%?',  # f1_macro: 0.85
        r'f1[_\-\s]*micro\s*[:=]\s*([\d.]+)\s*%?',  # f1_micro: 0.85
        r'f1[_\-\s]*weighted\s*[:=]\s*([\d.]+)\s*%?', # f1_weighted: 0.85
        # JSON-like patterns in output
        r'"f1_score"\s*:\s*([\d.]+)',               # "f1_score": 0.85
        r'"f1"\s*:\s*([\d.]+)',                     # "f1": 0.85
        r"'f1_score'\s*:\s*([\d.]+)",               # 'f1_score': 0.85
        r"'f1'\s*:\s*([\d.]+)",                     # 'f1': 0.85
    ]
    
    # Keys to try when reading from JSON file
    JSON_KEYS = [
        "f1_score", "f1", "f1-score",
        "f1_macro", "f1_micro", "f1_weighted",
        "metrics.f1_score", "metrics.f1",
        "evaluation.f1_score", "evaluation.f1",
        "results.f1_score", "results.f1",
    ]
    
    def __init__(
        self,
        default_score: float = 0.0,
        source: str = "both",  # "output", "file", or "both"
        filename: str = "results.json",
        json_key: Optional[str] = None,  # If None, tries common keys
        **params
    ):
        super().__init__(**params)
        self.default_score = default_score
        self.source = source.lower()
        self.filename = filename
        self.json_key = json_key
        
        # Compile patterns for efficiency
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.F1_PATTERNS
        ]
    
    def _extract_from_output(self, output: str) -> Optional[float]:
        """Try to extract F1 score from stdout/stderr output."""
        for pattern in self._compiled_patterns:
            match = pattern.search(output)
            if match:
                value = float(match.group(1))
                # Convert percentage to decimal if needed
                if value > 1.0:
                    value = value / 100.0
                return value
        return None
    
    def _extract_from_file(self, file_path: str) -> Optional[float]:
        """Try to extract F1 score from JSON file."""
        file_path_full = os.path.join(file_path, self.filename)
        
        if not os.path.exists(file_path_full):
            return None
        
        try:
            with open(file_path_full) as f:
                data = json.load(f)
            
            # If specific key provided, use it
            keys_to_try = [self.json_key] if self.json_key else self.JSON_KEYS
            
            for key in keys_to_try:
                if key is None:
                    continue
                try:
                    # Navigate dot notation (e.g., "metrics.f1")
                    value = data
                    for k in key.split("."):
                        value = value[k]
                    
                    score = float(value)
                    # Convert percentage to decimal if needed
                    if score > 1.0:
                        score = score / 100.0
                    return score
                except (KeyError, TypeError):
                    continue
                    
        except (json.JSONDecodeError, IOError):
            pass
        
        return None
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        score = None
        source_found = None
        
        # Try output first (usually more immediate feedback)
        if self.source in ("output", "both"):
            score = self._extract_from_output(output)
            if score is not None:
                source_found = "output"
        
        # Try file if not found in output or if source is "file"
        if score is None and self.source in ("file", "both"):
            score = self._extract_from_file(file_path)
            if score is not None:
                source_found = f"file:{self.filename}"
        
        # Return result
        if score is not None:
            return EvaluationResult(
                score=score,
                details={"source": source_found, "f1_score": score},
                raw_output=output,
            )
        
        return EvaluationResult(
            score=self.default_score,
            feedback=f"F1 score not found in {self.source}",
            details={"source": None, "searched": self.source},
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


# =============================================================================
# Script Evaluator (Default)
# =============================================================================

@register_evaluator("script")
class ScriptEvaluator(Evaluator):
    """
    Default evaluator that runs a custom evaluation script written by the coding agent.
    
    The agent creates evaluate.py that:
    1. Computes the evaluation metric
    2. Prints "SCORE: <float>" (required)
    3. Optionally prints "STOP: true" when goal is achieved
    
    The system (LLM) generates rich feedback based on the score and code.
    
    This is the DEFAULT evaluator - no configuration needed.
    
    Example agent-written evaluate.py:
        from sklearn.metrics import accuracy_score
        import json
        
        with open("results.json") as f:
            results = json.load(f)
        
        score = accuracy_score(results["labels"], results["predictions"])
        print(f"SCORE: {score:.4f}")
        
        if score >= 0.90:
            print("STOP: true")
    
    Params:
        script_path: Path to evaluation script (default: "evaluate.py")
        timeout: Execution timeout in seconds (default: 120)
        feedback_model: LLM model for feedback generation (default: "gpt-4.1")
        generate_feedback: Whether to generate LLM feedback (default: True)
    """
    
    description = "Run evaluate.py (agent-written) for score; LLM generates feedback"
    requires_llm = True  # For feedback generation
    
    DEFAULT_SCRIPT_PATH = "evaluate.py"
    DEFAULT_TIMEOUT = 120
    
    def __init__(
        self,
        script_path: str = DEFAULT_SCRIPT_PATH,
        timeout: int = DEFAULT_TIMEOUT,
        feedback_model: str = "gpt-4.1",
        generate_feedback: bool = True,
        **params
    ):
        super().__init__(**params)
        self.script_path = script_path
        self.timeout = timeout
        self.feedback_model = feedback_model
        self.generate_feedback = generate_feedback
        self._llm = None  # Lazy init
    
    @property
    def llm(self):
        """Lazy-load LLM backend to avoid import issues."""
        if self._llm is None and self.generate_feedback:
            from src.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    def evaluate(self, output: str, file_path: str, **context) -> EvaluationResult:
        """
        Run evaluate.py and generate feedback.
        
        Returns EvaluationResult with:
        - score: From evaluate.py output
        - feedback: LLM-generated improvement suggestions
        - details["should_stop"]: True if evaluate.py printed "STOP: true"
        """
        # Step 1: Run evaluate.py to get score and stop signal
        score, should_stop, eval_output, error = self._run_eval_script(file_path)
        
        if error:
            return EvaluationResult(
                score=0.0,
                feedback=f"Evaluation failed: {error}",
                details={"should_stop": False, "eval_output": eval_output},
                raw_output=output,
            )
        
        # Step 2: Generate LLM feedback (system-owned, not agent-written)
        feedback = ""
        if self.generate_feedback and self.llm:
            feedback = self._generate_feedback(
                problem=context.get("problem", ""),
                solution=context.get("solution", ""),
                code=self._read_main_code(file_path),
                score=score,
                eval_output=eval_output,
                execution_output=output,
            )
        
        return EvaluationResult(
            score=score,
            feedback=feedback,
            details={
                "should_stop": should_stop,
                "eval_output": eval_output,
                "script_path": self.script_path,
            },
            raw_output=output,
        )
    
    def _run_eval_script(self, file_path: str) -> tuple:
        """
        Run evaluate.py and parse SCORE and STOP from output.
        
        Returns:
            (score, should_stop, eval_output, error)
        """
        import subprocess
        
        script_full_path = os.path.join(file_path, self.script_path)
        
        # Check if script exists
        if not os.path.exists(script_full_path):
            return None, False, "", f"Missing {self.script_path} - agent must create evaluation script"
        
        try:
            result = subprocess.run(
                ["python", self.script_path],
                cwd=file_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            eval_output = result.stdout + result.stderr
            
            # Parse score (required)
            score_match = re.search(r'SCORE:\s*([-+]?\d*\.?\d+)', eval_output, re.IGNORECASE)
            if not score_match:
                return None, False, eval_output, f"No SCORE found in {self.script_path} output"
            
            score = float(score_match.group(1))
            
            # Parse stop signal (optional)
            stop_match = re.search(r'STOP:\s*(true|yes|1)', eval_output, re.IGNORECASE)
            should_stop = bool(stop_match)
            
            return score, should_stop, eval_output, None
            
        except subprocess.TimeoutExpired:
            return None, False, "", f"Evaluation script timed out after {self.timeout}s"
        except Exception as e:
            return None, False, "", str(e)
    
    def _generate_feedback(
        self,
        problem: str,
        solution: str,
        code: str,
        score: float,
        eval_output: str,
        execution_output: str,
    ) -> str:
        """Generate actionable feedback using LLM."""
        
        system_prompt = """You are an expert code reviewer analyzing a solution's evaluation results.

Provide 2-3 specific, actionable areas of improvement:
- Focus on what would most improve the score
- Be specific about the issue and its impact on performance
- Do NOT provide the fix, just identify the problem clearly

Format your response as:
<feedback>
Area 1: [specific issue and why it hurts the score]
Area 2: [specific issue and why it hurts the score]
Area 3: [specific issue and why it hurts the score] (optional)
</feedback>

If the score is very high (>0.95), you can acknowledge good performance and suggest minor optimizations."""

        # Truncate long content to fit context limits
        problem_truncated = problem[:2000] if problem else "Not provided"
        solution_truncated = solution[:1000] if solution else "Not provided"
        code_truncated = code[:3000] if code else "Not provided"
        eval_truncated = eval_output[:1000] if eval_output else "Not provided"
        exec_truncated = execution_output[:500] if execution_output else "Not provided"

        user_prompt = f"""<problem>
{problem_truncated}
</problem>

<solution_approach>
{solution_truncated}
</solution_approach>

<code>
{code_truncated}
</code>

<execution_output>
{exec_truncated}
</execution_output>

<evaluation_output>
{eval_truncated}
</evaluation_output>

<score>{score}</score>

Analyze why the score is {score} and identify specific areas that could improve it."""

        try:
            response = self.llm.llm_completion_with_system_prompt(
                model=self.feedback_model,
                system_prompt=system_prompt,
                user_message=user_prompt,
            )
            
            # Extract feedback section if present
            match = re.search(r'<feedback>(.*?)</feedback>', response, re.DOTALL)
            return match.group(1).strip() if match else response
            
        except Exception as e:
            return f"Feedback generation failed: {e}"
    
    def _read_main_code(self, file_path: str) -> str:
        """Read main.py for feedback context."""
        main_file = os.path.join(file_path, "main.py")
        if os.path.exists(main_file):
            try:
                with open(main_file) as f:
                    return f.read()
            except Exception:
                pass
        return ""
