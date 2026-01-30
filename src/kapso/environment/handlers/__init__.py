# Problem Handlers
#
# Base and concrete implementations for problem handling.
#
# In the new design:
# - Developer agent builds evaluation in kapso_evaluation/
# - Developer agent runs evaluation and reports results
# - FeedbackGenerator decides when to stop
#
# The handler provides problem context.

from kapso.environment.handlers.base import ProblemHandler, ProblemRunResult
from kapso.environment.handlers.generic import GenericProblemHandler

__all__ = [
    # Base
    "ProblemHandler",
    "ProblemRunResult",  # Deprecated, kept for benchmark compatibility
    # Generic handler
    "GenericProblemHandler",
]
