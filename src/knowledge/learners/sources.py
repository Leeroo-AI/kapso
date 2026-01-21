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
#     # Public web research (deep search)
#     research = kapso.research("How to pick LoRA rank?", mode="idea")
#     pipeline.run(research)
#
#     # Get ideas from research findings
#     ideas = research_findings.ideas(top_k=20)
#     for idea in ideas:
#         print(idea.to_context_string())

from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

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
    class Idea:
        """
        A single research idea/insight extracted from web research.
        
        Produced by: ResearchFindings.ideas(top_k)
        Used in: kapso.evolve(context=[...])
        
        Has to_context_string() for converting to LLM-friendly text.
        """
        content: str
        source_url: str = ""  # Optional URL where this idea came from
        
        def to_context_string(self) -> str:
            """Format idea for LLM prompt context."""
            if self.source_url:
                return f"- {self.content} ({self.source_url})"
            return f"- {self.content}"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"content": self.content, "source_url": self.source_url}
        
        def __str__(self) -> str:
            """String representation uses to_context_string()."""
            return self.to_context_string()

    @dataclass
    class Research:
        """
        Source from public web research.
        
        Produced by: Kapso.research()
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


# Helper class for a list of ideas with to_context_string()
class IdeaList(list):
    """
    A list of Source.Idea objects with to_context_string() method.
    
    This allows:
        ideas = research_findings.ideas(top_k=20)
        context_str = ideas.to_context_string()  # or str(ideas)
    """
    
    def __init__(self, ideas: List[Source.Idea], objective: str = ""):
        super().__init__(ideas)
        self.objective = objective
    
    def to_context_string(self) -> str:
        """Convert all ideas to a single context string."""
        if not self:
            return ""
        
        header = f"# Research Insights"
        if self.objective:
            header += f": {self.objective}"
        header += "\n\n"
        
        body = "\n".join(idea.to_context_string() for idea in self)
        return header + body
    
    def __str__(self) -> str:
        """String representation uses to_context_string()."""
        return self.to_context_string()

