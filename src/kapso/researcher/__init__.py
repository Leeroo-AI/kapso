# Public Web Research
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
#     from kapso.knowledge_base.types import Source

from kapso.researcher.researcher import (
    Researcher,
    ResearchDepth,
)
from kapso.researcher.research_findings import (
    ResearchMode,
    ResearchModeInput,
)

__all__ = [
    "Researcher",
    "ResearchMode",
    "ResearchModeInput",
    "ResearchDepth",
]
