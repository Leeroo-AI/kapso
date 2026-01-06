# Generic Problem Handler
#
# A flexible problem handler for any arbitrary problem.
# Uses pluggable Evaluator and StopCondition classes from the new system.
#
# Usage:
#     from src.environment.handlers.generic import GenericProblemHandler
#     from src.environment.evaluators import EvaluatorFactory
#     from src.environment.stop_conditions import StopConditionFactory
#     
#     handler = GenericProblemHandler(
#         problem_description="Build a web scraper...",
#         evaluator=EvaluatorFactory.create("regex_pattern", pattern=r"Accuracy: ([\d.]+)"),
#         stop_condition=StopConditionFactory.create("threshold", threshold=0.95),
#     )

import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

from src.environment.handlers.base import ProblemHandler, ProblemRunResult
from src.environment.evaluators import Evaluator, EvaluatorFactory
from src.environment.stop_conditions import StopCondition, StopConditionFactory
from src.core import llm as llm_utils


class GenericProblemHandler(ProblemHandler):
    """
    Generic problem handler for any arbitrary problem.
    
    Uses pluggable Evaluator and StopCondition classes for flexibility.
    
    Args:
        problem_description: Main problem description/prompt
        main_file: Entry point file (default: main.py)
        language: Programming language (python, cpp, node, bash)
        timeout: Execution timeout in seconds
        debug_timeout: Timeout for debug mode
        data_dir: Optional data directory path
        output_file: Expected output file (if any)
        additional_context: Extra context to append (tips, requirements, etc.)
        maximize_scoring: True if higher score is better
        evaluator: Evaluator instance or name (string) to create via factory
        evaluator_params: Parameters for evaluator if using name
        stop_condition: StopCondition instance or name (string) to create via factory
        stop_condition_params: Parameters for stop_condition if using name
        
    Examples:
        # Simple - just run code (no scoring)
        handler = GenericProblemHandler(
            problem_description="Write a prime number finder..."
        )
        
        # With evaluator by name
        handler = GenericProblemHandler(
            problem_description="Build a classifier...",
            evaluator="regex_pattern",
            evaluator_params={"pattern": r"Accuracy: ([\\d.]+)"},
        )
        
        # With evaluator instance
        from src.environment.evaluators import EvaluatorFactory
        handler = GenericProblemHandler(
            problem_description="Build a classifier...",
            evaluator=EvaluatorFactory.create("llm_judge", criteria="correctness"),
        )
        
        # With stop condition
        handler = GenericProblemHandler(
            problem_description="Achieve 95% accuracy...",
            stop_condition="threshold",
            stop_condition_params={"threshold": 0.95},
        )
        
        # Composite: stop if score >= 0.95 OR after 50 iterations
        handler = GenericProblemHandler(
            problem_description="...",
            stop_condition="composite",
            stop_condition_params={
                "conditions": [
                    ("threshold", {"threshold": 0.95}),
                    ("max_iterations", {"max_iter": 50}),
                ],
                "mode": "any",
            },
        )
    """
    
    def __init__(
        self,
        problem_description: str,
        main_file: str = "main.py",
        language: str = "python",
        timeout: int = 300,
        debug_timeout: int = 60,
        data_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        additional_context: str = "",
        maximize_scoring: bool = True,
        evaluator: Optional[Union[Evaluator, str]] = None,
        evaluator_params: Optional[Dict[str, Any]] = None,
        stop_condition: Optional[Union[StopCondition, str]] = None,
        stop_condition_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize generic problem handler."""
        # Parent init with additional_context
        super().__init__(additional_context=additional_context)
        
        # Core config
        self.problem_description = problem_description
        self.main_file = main_file
        self.language = language
        self.timeout = timeout
        self.debug_timeout = debug_timeout
        self.data_dir = data_dir
        self.output_file = output_file
        self.maximize_scoring = maximize_scoring
        
        # Create evaluator - accepts instance or name
        if evaluator is None:
            self.evaluator = EvaluatorFactory.create("no_score")
        elif isinstance(evaluator, str):
            params = evaluator_params or {}
            self.evaluator = EvaluatorFactory.create(evaluator, **params)
        else:
            self.evaluator = evaluator
        
        # Create stop condition - accepts instance or name
        if stop_condition is None:
            self.stop_condition_obj = StopConditionFactory.create("never")
        elif isinstance(stop_condition, str):
            params = stop_condition_params or {}
            self.stop_condition_obj = StopConditionFactory.create(stop_condition, **params)
        else:
            self.stop_condition_obj = stop_condition
        
        # State tracking
        self.llm = llm_utils.LLMBackend()
        self.best_score: Optional[float] = None
        self.iteration_count: int = 0
        self.last_result: Optional[ProblemRunResult] = None
        
        # Build context once
        self._problem_context = self._build_problem_context()
    
    def _build_problem_context(self) -> str:
        """Build the full problem context string."""
        parts = [
            "# Problem Description",
            self.problem_description,
            "",
            "# Requirements",
            f"- Main file: {self.main_file}",
            f"- Language: {self.language}",
            f"- Timeout: {self.timeout} seconds",
            f"- Final evaluation score logic: {self.evaluator.description}",
        ]
        
        if self.output_file:
            parts.append(f"- Output file: {self.output_file}")
        
        if self.data_dir and os.path.exists(self.data_dir):
            parts.extend([
                "",
                "# Data",
                f"Data directory: {self.data_dir}",
            ])
        
        if self.additional_context:
            parts.extend([
                "",
                "# Additional Context",
                self.additional_context,
            ])
        
        parts.extend([
            "",
            "# Execution Notes",
            "- Do not use interactive outputs (tqdm, progress bars)",
            "- Print meaningful progress to stdout",
        ])
        
        return "\n".join(parts)
    
    def _get_run_command(self) -> List[str]:
        """Get the command to execute the main file."""
        lang = self.language.lower()
        
        commands = {
            "python": ["python", self.main_file],
            "cpp": ["bash", "-c", f"g++ -O2 -o main {self.main_file} && ./main"],
            "c++": ["bash", "-c", f"g++ -O2 -o main {self.main_file} && ./main"],
            "node": ["node", self.main_file],
            "javascript": ["node", self.main_file],
            "js": ["node", self.main_file],
            "bash": ["bash", self.main_file],
        }
        
        return commands.get(lang, [self.main_file])
    
    def _run_command(
        self, 
        file_path: str, 
        command: List[str], 
        timeout: int
    ) -> tuple:
        """Execute command and return (had_error, error_details, output, time)."""
        start_time = time.time()
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        try:
            # `subprocess.Popen(..., stdout=PIPE)` creates a file object for stdout.
            # We use a context manager so it always closes deterministically.
            #
            # Why:
            # - Prevent `ResourceWarning: unclosed file <_io.TextIOWrapper ...>` noise.
            # - Avoid leaking file descriptors in long-running sessions.
            with subprocess.Popen(
                command,
                cwd=file_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            ) as process:
                
                output_lines = []
                if process.stdout:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        output_lines.append(line)
                        if len(output_lines) > 5000:
                            process.kill()
                            break
                
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    # Reap the process so we don't leak resources.
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        pass
                
                execution_time = time.time() - start_time
                
                # Truncate long output
                if len(output_lines) > 200:
                    output_lines = output_lines[:100] + [" ...\n"] + output_lines[-100:]
                
                output = ''.join(output_lines)
                had_error = process.returncode != 0
                
                return had_error, output if had_error else "", output, execution_time
            
        except Exception as e:
            return True, str(e), "", time.time() - start_time
    
    def run(
        self, 
        file_path: str, 
        run_data_dir: str = "",
        debug: bool = False, 
        **kwargs
    ) -> ProblemRunResult:
        """Execute code and evaluate results."""
        self.iteration_count += 1
        
        timeout = self.debug_timeout if debug else self.timeout
        command = self._get_run_command()
        
        had_error, error_details, output, exec_time = self._run_command(
            file_path, command, timeout
        )
        
        # Check timeout
        if exec_time >= timeout - 1:
            had_error = True
            error_details = f"Execution timed out after {timeout} seconds"
        
        # Evaluate using evaluator
        score = 0.0
        feedback = ""
        if not had_error:
            # Pass context to evaluator for richer evaluation
            eval_result = self.evaluator.evaluate(
                output, 
                file_path,
                problem=self.problem_description,
                solution=kwargs.get("solution", ""),
                iteration=self.iteration_count,
            )
            score = eval_result.score
            feedback = eval_result.feedback
            
            # Update best score
            if self.best_score is None:
                self.best_score = score
            elif self.maximize_scoring:
                self.best_score = max(self.best_score, score)
            else:
                self.best_score = min(self.best_score, score)
        
        result = ProblemRunResult(
            score=score,
            output=output,
            detailed_output=output,
            run_had_error=had_error,
            error_message=error_details if had_error else "",  # Actual error, not generic string
            error_details=error_details,
            feedbacks=feedback,
            continue_debugging=True,
        )
        
        self.last_result = result
        return result
    
    def final_evaluate(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Final evaluation - returns summary of all runs."""
        return {
            "best_score": self.best_score,
            "last_score": self.last_result.score if self.last_result else None,
            "iterations": self.iteration_count,
            "had_error": self.last_result.run_had_error if self.last_result else None,
            "evaluator": self.evaluator.name,
        }
    
    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        """Return problem context."""
        return self._problem_context
    
    def stop_condition(self, **kwargs) -> bool:
        """Check if we should stop early using the stop condition."""
        if self.best_score is None or self.last_result is None:
            return False
        
        # Pass additional context to stop condition
        decision = self.stop_condition_obj.check(
            best_score=self.best_score,
            current_score=self.last_result.score,
            iteration=self.iteration_count,
            had_error=self.last_result.run_had_error,
            **kwargs
        )
        
        return decision.should_stop
