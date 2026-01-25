"""
Problem Handler Base

Abstract base class for all problem handlers.
Each benchmark implements its own handler with specific evaluation logic.

NOTE: In the new design, the developer agent is responsible for building and
running evaluation. The handler now primarily provides problem context and
basic execution utilities. The stop_condition is handled by the FeedbackGenerator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProblemRunResult:
    """Result of running code on a problem."""
    score: float = 0
    output: str = ""
    detailed_output: str = ""
    run_had_error: bool = False
    error_message: str = ""
    error_details: str = ""
    feedbacks: str = ""
    continue_debugging: bool = True
    # New fields for feedback generator
    evaluation_script_path: str = ""  # Path to evaluation script (from developer agent)


class ProblemHandler(ABC):
    """
    Abstract base class for problem handlers.
    
    Subclasses must implement:
    - run(): Execute code and return results (for legacy/benchmark compatibility)
    - final_evaluate(): Final evaluation on private test set
    - get_problem_context(): Return problem description
    
    NOTE: stop_condition() is now optional and only used for backward compatibility
    with existing benchmarks. The new design uses FeedbackGenerator for stop decisions.
    """
    
    def __init__(self, additional_context: str = ""):
        """
        Initialize problem handler.
        
        Args:
            additional_context: Extra context to include (tips, domain knowledge, etc.)
        """
        self.additional_context = additional_context
    
    @abstractmethod
    def run(self, file_path: str, debug: bool = False, **kwargs) -> ProblemRunResult:
        """Execute code and return results."""
        pass
        
    @abstractmethod
    def final_evaluate(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Final evaluation on private/held-out test set."""
        pass

    @abstractmethod
    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        """Return problem description (may vary with budget progress)."""
        pass
    
    def stop_condition(self, **kwargs) -> bool:
        """
        Return True if search should stop early.
        
        NOTE: This method is optional and only used for backward compatibility
        with existing benchmarks. The new design uses FeedbackGenerator for
        stop decisions. Default implementation always returns False.
        """
        return False
