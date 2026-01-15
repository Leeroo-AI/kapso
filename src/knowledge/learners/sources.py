# Knowledge Sources
#
# Typed wrappers for knowledge inputs.
# These define what can be passed to Tinkerer.learn() and Learner.learn().
#
# Usage:
#     from src.knowledge.learners import Source
#     
#     tinkerer.learn(
#         Source.Repo("https://github.com/user/repo"),
#         target_kg="https://skills.leeroo.com",
#     )

from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

# Avoid circular import
if TYPE_CHECKING:
    from src.tinkerer import SolutionResult


class Source:
    """
    Namespace for knowledge source types.
    
    Each source type is a typed wrapper that tells Tinkerer.learn() 
    how to process the input. The type determines which Learner
    is used to extract knowledge.
    
    Usage:
        tinkerer.learn(
            Source.Repo("https://github.com/user/repo"),
            target_kg="https://skills.leeroo.com",
        )
    """
    
    @dataclass
    class Repo:
        """
        Source from a Git repository.
        
        Processed by: RepoLearner
        Extracts: README, code patterns, docstrings, structure
        """
        url: str
        branch: str = "main"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"url": self.url, "branch": self.branch}
    
    @dataclass
    class Solution:
        """
        Source from a completed Solution (experiment logs).
        
        Processed by: ExperimentLearner
        Extracts: Goal-outcome pairs, successful patterns, failures to avoid
        """
        obj: "SolutionResult"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"solution": self.obj}

