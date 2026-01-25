# Generic Problem Handler
#
# A flexible problem handler for any arbitrary problem.
# In the new design, the developer agent is responsible for building and
# running evaluation. This handler provides problem context and execution utilities.
#
# Usage:
#     from src.environment.handlers.generic import GenericProblemHandler
#     
#     handler = GenericProblemHandler(
#         problem_description="Build a web scraper...",
#     )

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from src.environment.handlers.base import ProblemHandler, ProblemRunResult
from src.core import llm as llm_utils


class GenericProblemHandler(ProblemHandler):
    """
    Generic problem handler for any arbitrary problem.
    
    In the new design:
    - Developer agent builds evaluation in kapso_evaluation/
    - Developer agent runs evaluation and reports results
    - FeedbackGenerator decides when to stop
    
    This handler provides:
    - Problem context/description
    - Basic code execution utilities
    - State tracking for iterations
    
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
        
    Examples:
        # Simple - just provide problem description
        handler = GenericProblemHandler(
            problem_description="Write a prime number finder..."
        )
        
        # With data directory
        handler = GenericProblemHandler(
            problem_description="Build a classifier...",
            data_dir="./datasets/",
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
        ]
        
        # Add evaluation instructions for new design
        parts.extend([
            "",
            "# Evaluation",
            "You are responsible for building and running evaluation.",
            "Create evaluation code in the `kapso_evaluation/` directory.",
            "The evaluation should:",
            "1. Test your solution against the goal criteria",
            "2. Output a clear score or success/failure indication",
            "3. Be fair and actually test what it claims to test",
            "",
            "Example evaluation structure:",
            "```",
            "kapso_evaluation/",
            "  ├── evaluate.py      # Main evaluation script",
            "  ├── test_cases/      # Test data (if needed)",
            "  └── README.md        # Evaluation description",
            "```",
        ])
        
        if self.output_file:
            parts.append(f"- Output file: {self.output_file}")
        
        if self.data_dir and os.path.exists(self.data_dir):
            parts.extend([
                "",
                "# Data",
                f"Data directory: {self.data_dir}",
                "Use `kapso_datasets/` for any datasets provided.",
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
            "- Run your evaluation and report the result before completing the iteration",
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
            # Use context manager to prevent resource leaks
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
        """
        Execute code and return results.
        
        NOTE: In the new design, the developer agent is responsible for
        running evaluation. This method is kept for backward compatibility
        and basic code execution.
        """
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
        
        # In new design, score is extracted by FeedbackGenerator
        # Here we just return the execution result
        score = 0.0
        
        result = ProblemRunResult(
            score=score,
            output=output,
            detailed_output=output,
            run_had_error=had_error,
            error_message=error_details if had_error else "",
            error_details=error_details,
            feedbacks="",
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
        }
    
    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        """Return problem context."""
        return self._problem_context
    
    def stop_condition(self, **kwargs) -> bool:
        """
        Check if we should stop early.
        
        NOTE: In the new design, FeedbackGenerator decides when to stop.
        This method is kept for backward compatibility and always returns False.
        """
        return False
