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

from src.knowledge.researcher.researcher import Researcher, ResearchDepth, ResearchMode
from src.knowledge.researcher.research_findings import ResearchFindings

__all__ = ["Researcher", "ResearchFindings", "ResearchDepth", "ResearchMode"]

