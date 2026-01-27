# Feedback Generator Module
#
# LLM-based feedback generator that validates evaluation and decides stop/continue.
# Uses a coding agent (default: claude_code) to analyze evaluation results.

from .feedback_generator import FeedbackGenerator, FeedbackResult

__all__ = ["FeedbackGenerator", "FeedbackResult"]
