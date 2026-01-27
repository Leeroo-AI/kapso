# Public Web Research (Knowledge)
#
# This package provides deep public web research utilities that return
# `ResearchFindings` objects with structured results.
#
# Exports:
# - Researcher: main entry point (OpenAI Responses API + web_search tool)
# - ResearchFindings: wrapper with .ideas, .implementations, .report
# - IdeaResult: single idea from research
# - ImplementationResult: single implementation from research
# - ResearchReport: freeform research report
# - ResearchMode: "idea" | "implementation" | "freeform"
# - ResearchDepth: "light" | "deep"

from src.knowledge.researcher.researcher import (
    Researcher,
    ResearchDepth,
    ResearchMode,
    ResearchModeInput,
)
from src.knowledge.researcher.research_findings import (
    ResearchFindings,
    IdeaResult,
    ImplementationResult,
    ResearchReport,
)

__all__ = [
    "Researcher",
    "ResearchFindings",
    "IdeaResult",
    "ImplementationResult",
    "ResearchReport",
    "ResearchMode",
    "ResearchModeInput",
    "ResearchDepth",
]
