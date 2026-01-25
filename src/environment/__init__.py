# Environment Module - Problem Environment
#
# Handles problem definition and code execution.
#
# In the new design:
# - Developer agent builds evaluation in kapso_evaluation/
# - Developer agent runs evaluation and reports results
# - FeedbackGenerator decides when to stop
#
# Submodules:
#   - handlers: Problem handlers (base, generic)

# Handlers
from src.environment.handlers import (
    ProblemHandler,
    ProblemRunResult,
    GenericProblemHandler,
)

__all__ = [
    # Handlers
    "ProblemHandler",
    "ProblemRunResult",
    "GenericProblemHandler",
]
