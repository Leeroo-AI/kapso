# Public Web Research (Knowledge)
#
# This package provides deep public web research utilities that return
# `ResearchFindings` objects with fluent accessors.
#
# Exports:
# - Researcher: main entry point (OpenAI Responses API + web_search tool)
# - ResearchFindings: wrapper with .repos() and .ideas() methods
# - ResearchMode: "idea" | "implementation" | "both"
# - ResearchDepth: "light" | "deep"

from src.knowledge.web_research.researcher import Researcher, ResearchDepth, ResearchMode
from src.knowledge.web_research.research_findings import ResearchFindings

__all__ = ["Researcher", "ResearchFindings", "ResearchDepth", "ResearchMode"]

