# Generic Problem Handler
#
# A flexible problem handler for any arbitrary problem.
# In the new design, the developer agent is responsible for building and
# running evaluation. This handler provides problem context.
#
# Usage:
#     from src.environment.handlers.generic import GenericProblemHandler
#     
#     handler = GenericProblemHandler(
#         problem_description="Build a web scraper...",
#     )

import os
from typing import Any, Dict, Optional

from src.environment.handlers.base import ProblemHandler


class GenericProblemHandler(ProblemHandler):
    """
    Generic problem handler for any arbitrary problem.
    
    In the new design:
    - Developer agent builds evaluation in kapso_evaluation/
    - Developer agent runs evaluation and reports results
    - FeedbackGenerator decides when to stop
    
    This handler provides:
    - Problem context/description
    
    Args:
        problem_description: Main problem description/prompt
        main_file: Entry point file (default: main.py)
        language: Programming language (python, cpp, node, bash)
        timeout: Execution timeout in seconds
        data_dir: Optional data directory path
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
        data_dir: Optional[str] = None,
        additional_context: str = "",
        maximize_scoring: bool = True,
    ):
        """Initialize generic problem handler."""
        super().__init__(additional_context=additional_context)
        
        # Core config
        self.problem_description = problem_description
        self.main_file = main_file
        self.language = language
        self.timeout = timeout
        self.data_dir = data_dir
        self.maximize_scoring = maximize_scoring
        
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
            "After running evaluation, write results to `kapso_evaluation/result.json`:",
            "```json",
            "{",
            '  "evaluation_script_path": "kapso_evaluation/evaluate.py",',
            '  "evaluation_output": "Full output from running evaluation",',
            '  "score": 0.95',
            "}",
            "```",
        ])
        
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
            "- Run your evaluation and report the result before completing",
        ])
        
        return "\n".join(parts)
    
    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        """Return problem context."""
        return self._problem_context
    
    def final_evaluate(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Final evaluation - returns empty dict (no separate test set)."""
        return {}
