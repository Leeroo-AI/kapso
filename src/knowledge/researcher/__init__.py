# Public Web Research (Knowledge)
#
# This package provides deep public web research utilities.
#
# Exports:
# - Researcher: main entry point (OpenAI Responses API + web_search tool)
# - Idea: single idea from research
# - Implementation: single implementation from research
# - ResearchReport: study mode research report
# - ResearchFindings: wrapper for multi-mode results
# - ResearchMode: "idea" | "implementation" | "study"
# - ResearchModeInput: single mode or list of modes
# - ResearchDepth: "light" | "deep"

from src.knowledge.researcher.researcher import (
    Researcher,
    ResearchDepth,
)
from src.knowledge.researcher.research_findings import (
    Idea,
    Implementation,
    ResearchReport,
    ResearchFindings,
    ResearchMode,
    ResearchModeInput,
)

__all__ = [
    "Researcher",
    "Idea",
    "Implementation",
    "ResearchReport",
    "ResearchFindings",
    "ResearchMode",
    "ResearchModeInput",
    "ResearchDepth",
]
