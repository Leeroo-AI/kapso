# Execution Types
#
# Common types used across the execution module.
# Moved from context_manager/types.py for broader use.

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol


@dataclass
class ContextData:
    """
    Output data structure from context gathering.
    
    Contains all context needed for solution generation:
    - problem: The problem description
    - additional_info: Experiment history and other info
    - kg_results: Knowledge graph text results
    - kg_code_results: Knowledge graph code snippets
    """
    problem: str
    additional_info: str
    kg_results: Optional[str] = ""
    kg_code_results: Optional[str] = ""


class ExperimentHistoryProvider(Protocol):
    """
    Protocol for components that provide experiment history.
    
    Used by ContextManager to get experiment history without
    depending on the full SearchStrategy class.
    """
    
    def get_experiment_history(self, best_last: bool = False) -> List[Any]:
        """
        Get experiment history.
        
        Args:
            best_last: If True, sort by score (best last)
            
        Returns:
            List of experiment results
        """
        ...
