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
#         wiki_dir="data/wikis",
#     )
#
#     # Public web research (deep search)
#     research = tinkerer.research("How to pick LoRA rank?", mode="idea")
#     pipeline.run(research)

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
            wiki_dir="data/wikis",
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

    @dataclass
    class Research:
        """
        Source from public web research.
        
        Produced by: Tinkerer.research()
        Consumed by: KnowledgePipeline.run(...) via ResearchIngestor
        
        Notes:
        - `report_markdown` is the only required artifact. It should include URLs inline.
        - `mode` controls the shape of the report: "idea" | "implementation" | "both".
        """
        objective: str
        mode: str  # "idea" | "implementation" | "both"
        report_markdown: str
        
        def to_context_string(self) -> str:
            """
            Format research for LLM prompt context.
            
            Keep it simple and copy-paste friendly.
            """
            header = f"# Web Research ({self.mode})\nObjective: {self.objective}\n"
            body = (self.report_markdown or "").strip()
            return (header + "\n" + body).strip() if body else header.strip()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dict (useful for logging/serialization)."""
            return {
                "objective": self.objective,
                "mode": self.mode,
                "report_markdown": self.report_markdown,
            }

