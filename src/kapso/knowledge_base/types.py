# Knowledge Types
#
# Unified type definitions for all knowledge sources.
# Single source of truth for source types used across the knowledge subsystem.
#
# Usage:
#     from kapso.knowledge_base.types import Source
#     
#     # Repository source
#     repo = Source.Repo("https://github.com/user/repo")
#     
#     # Research outputs
#     idea = Source.Idea(query="...", source="...", content="...")
#     impl = Source.Implementation(query="...", source="...", content="...")
#     report = Source.ResearchReport(query="...", content="...")
#     
#     # Solution from experiments
#     solution = Source.Solution(obj=solution_result)
#
# All types work with the IngestorFactory via class name matching.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Avoid circular import for SolutionResult
if TYPE_CHECKING:
    from kapso.kapso import SolutionResult


class Source:
    """
    Namespace for all knowledge source types.
    
    Each source type tells the learning pipeline how to process the input.
    The type determines which Ingestor is used to extract knowledge.
    
    Source Types:
        - Repo: Git repository (README, code patterns, docstrings)
        - Solution: Experiment results (goal-outcome pairs, patterns)
        - Idea: Research idea from web search
        - Implementation: Code implementation from web search
        - ResearchReport: Comprehensive research report
    """
    
    @dataclass
    class Repo:
        """
        Source from a Git repository.
        
        Processed by: RepoIngestor
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
        
        Processed by: ExperimentIngestor
        Extracts: Goal-outcome pairs, successful patterns, failures to avoid
        """
        obj: "SolutionResult"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"solution": self.obj}
    
    @dataclass
    class Idea:
        """
        A single research idea from web research.
        
        Produced by: researcher.research(query, mode="idea")
        Processed by: IdeaIngestor
        """
        query: str      # Original research query
        source: str     # URL where this idea came from
        content: str    # Full content with sections
        
        def to_string(self) -> str:
            """Format idea as context string for LLM prompts."""
            return f"# Research Idea\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"query": self.query, "source": self.source, "content": self.content}
        
        def __str__(self) -> str:
            return self.to_string()
    
    @dataclass
    class Implementation:
        """
        A single implementation from web research.
        
        Produced by: researcher.research(query, mode="implementation")
        Processed by: ImplementationIngestor
        """
        query: str      # Original research query
        source: str     # URL where this implementation came from
        content: str    # Full content with code snippet
        
        def to_string(self) -> str:
            """Format implementation as context string for LLM prompts."""
            return f"# Implementation\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"query": self.query, "source": self.source, "content": self.content}
        
        def __str__(self) -> str:
            return self.to_string()
    
    @dataclass
    class ResearchReport:
        """
        A comprehensive research report (academic paper style).
        
        Produced by: researcher.research(query, mode="study")
        Processed by: ResearchReportIngestor
        """
        query: str      # Original research query
        content: str    # Full markdown report
        
        def to_string(self) -> str:
            """Format report as context string for LLM prompts."""
            return f"# Research Report\nQuery: {self.query}\n\n{self.content}"
        
        def to_dict(self) -> Dict[str, Any]:
            return {"query": self.query, "content": self.content}
        
        def __str__(self) -> str:
            return self.to_string()


@dataclass
class ResearchFindings:
    """
    Wrapper for multi-mode research results.
    
    Produced by: researcher.research(query, mode=["idea", "implementation"])
    Contains results from multiple modes in a single object.
    """
    query: str
    ideas: List[Source.Idea] = field(default_factory=list)
    implementations: List[Source.Implementation] = field(default_factory=list)
    report: Optional[Source.ResearchReport] = None
    
    def to_string(self) -> str:
        """Format all findings as context string for LLM prompts."""
        parts = [f"# Research Findings\nQuery: {self.query}\n"]
        
        if self.ideas:
            parts.append("\n## Ideas\n")
            for idea in self.ideas:
                parts.append(f"### {idea.source}\n{idea.content}\n")
        
        if self.implementations:
            parts.append("\n## Implementations\n")
            for impl in self.implementations:
                parts.append(f"### {impl.source}\n{impl.content}\n")
        
        if self.report:
            parts.append("\n## Report\n")
            parts.append(self.report.content)
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "ideas": [i.to_dict() for i in self.ideas],
            "implementations": [i.to_dict() for i in self.implementations],
            "report": self.report.to_dict() if self.report else None,
        }
    
    def __str__(self) -> str:
        return self.to_string()
