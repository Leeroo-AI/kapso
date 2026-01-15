# Public Web Research (Knowledge)
#
# This package provides deep public web research utilities that return
# `Source.Research` artifacts by default.
#
# Exports:
# - DeepWebResearch: main entry point (OpenAI Responses API + web_search tool)
# - ResearchMode: "idea" | "implementation" | "both"
# - ResearchDepth: "light" | "deep"

from src.knowledge.web_research.deep_web_research import DeepWebResearch, ResearchDepth, ResearchMode

__all__ = ["DeepWebResearch", "ResearchDepth", "ResearchMode"]

