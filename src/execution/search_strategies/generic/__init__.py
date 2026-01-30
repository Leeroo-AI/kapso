# Generic Search Strategy
#
# Agent-based search strategy for general problem solving.
# Uses Claude Code as the ideation and implementation agent.
#
# Components:
# - strategy.py: Main GenericSearch class
# - feedback_generator/: LLM-based feedback generation
# - prompts/: Prompt templates for ideation and implementation

from .strategy import GenericSearch
from .feedback_generator import FeedbackGenerator, FeedbackResult

__all__ = ["GenericSearch", "FeedbackGenerator", "FeedbackResult"]
