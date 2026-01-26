"""
Problem Handler Base

Abstract base class for all problem handlers.
Provides problem context to the developer agent.

NOTE: In the new design, the developer agent is responsible for building and
running evaluation. The handler primarily provides problem context.
The FeedbackGenerator handles stop decisions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProblemRunResult:
    """
    Result of running code on a problem.
    
    DEPRECATED: This class is kept for backward compatibility with benchmarks.
    In the new design, the developer agent handles evaluation and returns
    results via kapso_evaluation/result.json.
    """
    score: float = 0
    output: str = ""
    detailed_output: str = ""
    run_had_error: bool = False
    error_message: str = ""
    error_details: str = ""
    feedbacks: str = ""
    continue_debugging: bool = True


class ProblemHandler(ABC):
    """
    Abstract base class for problem handlers.
    
    Subclasses must implement:
    - get_problem_context(): Return problem description
    
    Optional methods:
    - final_evaluate(): Final evaluation on private test set (for benchmarks)
    """
    
    # Whether higher scores are better (used by search strategies)
    maximize_scoring: bool = True
    
    def __init__(self, additional_context: str = ""):
        """
        Initialize problem handler.
        
        Args:
            additional_context: Extra context to include (tips, domain knowledge, etc.)
        """
        self.additional_context = additional_context

    @abstractmethod
    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        """Return problem description (may vary with budget progress)."""
        pass
    
    def final_evaluate(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Final evaluation on private/held-out test set.
        
        Override this for benchmarks that have a separate test set.
        Default implementation returns empty dict.
        """
        return {}
