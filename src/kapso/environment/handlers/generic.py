# Generic Problem Handler
#
# A flexible problem handler for any arbitrary problem.
# In the new design, the developer agent is responsible for building and
# running evaluation. This handler provides problem context.
#
# Usage:
#     from kapso.environment.handlers.generic import GenericProblemHandler
#     
#     handler = GenericProblemHandler(
#         problem_description="Build a web scraper...",
#     )

import os
from typing import Any, Dict, Optional

from kapso.environment.handlers.base import ProblemHandler


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
        eval_dir: Optional evaluation directory path (copied to kapso_evaluation/)
        data_dir: Optional data directory path (copied to kapso_datasets/)
        additional_context: Extra context to append (tips, requirements, etc.)
        
    Examples:
        # Simple - just provide problem description
        handler = GenericProblemHandler(
            problem_description="Write a prime number finder..."
        )
        
        # With eval and data directories
        handler = GenericProblemHandler(
            problem_description="Build a classifier...",
            eval_dir="./evaluation/",
            data_dir="./datasets/",
        )
    """
    
    # Always maximize scoring for generic handler
    maximize_scoring: bool = True
    
    def __init__(
        self,
        problem_description: str,
        eval_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        additional_context: str = "",
    ):
        """Initialize generic problem handler."""
        super().__init__(additional_context=additional_context)
        
        # Core config
        self.problem_description = problem_description
        self.eval_dir = eval_dir
        self.data_dir = data_dir
        
        # Build context once
        self._problem_context = self._build_problem_context()
    
    def _build_problem_context(self) -> str:
        """Build the full problem context string."""
        parts = [
            "# Problem Description",
            self.problem_description,
        ]
        
        # Add evaluation instructions
        parts.extend([
            "",
            "# Evaluation",
        ])
        
        # If eval_dir provided, mention it
        if self.eval_dir and os.path.exists(self.eval_dir):
            parts.extend([
                "Evaluation scripts are provided in `kapso_evaluation/`.",
                "Review and run the existing evaluation code.",
            ])
        else:
            parts.extend([
                "You are responsible for building and running evaluation.",
                "Create evaluation code in the `kapso_evaluation/` directory.",
            ])
        
        parts.extend([
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
        
        # Mention data directory if provided
        if self.data_dir and os.path.exists(self.data_dir):
            parts.extend([
                "",
                "# Data",
                "Datasets are provided in `kapso_datasets/`.",
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
