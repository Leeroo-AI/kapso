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
from typing import Any, Dict, Optional


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

    def finalize_run_selection(self, manifest: Dict[str, Any], valid: bool) -> None:
        """Stamp run-selection eligibility once a node's score of record is known.

        Called by the search strategy exactly when it resolves a session's
        registered manifest (the node's canonical measurement). `manifest` is
        that parsed manifest line; `valid` is the node's judged validity
        (feedback verdict minus integrity flags). Handlers whose archives
        distinguish registered finals from intermediate evaluations override
        this to label the archive; the default is a no-op for handlers with
        no run archive.
        """
        return None

    def deliverable_ready_reserve_seconds(self) -> Optional[float]:
        """Reserve still needed once a CONFIRMED deliverable exists on disk.

        None (default) means no confirmed deliverable — the campaign keeps
        its full finalization reserve and admission floor. A float means the
        deliverable is banked and verified, so the endgame insurance is
        already paid: the orchestrator shrinks the reserve to this many
        seconds (what freeze still costs) and applies the insured admission
        floor, keeping late iterations available. Override in handlers whose
        benchmark maintains a verified best-so-far submission.
        """
        return None
