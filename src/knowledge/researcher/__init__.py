# Public Web Research (Knowledge)
#
# This package provides deep public web research utilities.
#
# Exports:
# - Researcher: main entry point (OpenAI Responses API + web_search tool)
# - ResearchMode: "idea" | "implementation" | "study"
# - ResearchModeInput: single mode or list of modes
# - ResearchDepth: "light" | "deep"
#
# For source types (Idea, Implementation, ResearchReport), use:
#     from src.knowledge.types import Source

from src.knowledge.researcher.researcher import (
    Researcher,
    ResearchDepth,
)
from src.knowledge.researcher.research_findings import (
    ResearchMode,
    ResearchModeInput,
)

__all__ = [
    "Researcher",
    "ResearchMode",
    "ResearchModeInput",
    "ResearchDepth",
]
