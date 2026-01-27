# Knowledge Sources
#
# Typed wrappers for knowledge inputs.
# These define what can be passed to Kapso.learn() and Learner.learn().
#
# Usage:
#     from src.knowledge.learners import Source
#     
#     kapso.learn(
#         Source.Repo("https://github.com/user/repo"),
#         wiki_dir="data/wikis",
#     )
#
# For research outputs, use the types from src.knowledge.researcher:
#     from src.knowledge.researcher import Idea, Implementation, ResearchReport
#     
#     ideas = kapso.research("How to fine-tune LLMs?", mode="idea", top_k=5)
#     for idea in ideas:
#         kapso.learn(idea, wiki_dir="data/wikis")

from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

# Avoid circular import
if TYPE_CHECKING:
    from src.kapso import SolutionResult


class Source:
    """
    Namespace for knowledge source types.
    
    Each source type is a typed wrapper that tells Kapso.learn() 
    how to process the input. The type determines which Learner
    is used to extract knowledge.
    
    Usage:
        kapso.learn(
            Source.Repo("https://github.com/user/repo"),
            wiki_dir="data/wikis",
        )
    
    For research outputs (Idea, Implementation, ResearchReport),
    import directly from src.knowledge.researcher.
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
