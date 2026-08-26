# Environment Module - Problem Environment
#
# Handles problem definition.
#
# In the new design:
# - Developer agent builds evaluation in kapso_evaluation/
# - Developer agent runs evaluation and reports results
# - FeedbackGenerator decides when to stop
#
# Submodules:
#   - handlers: Problem handlers (base, generic)

# Handlers
from kapso.environment.handlers import (
    ProblemHandler,
    ProblemRunResult,  # Benchmark handlers' run-result contract (live)
    GenericProblemHandler,
)

__all__ = [
    # Handlers
    "ProblemHandler",
    "ProblemRunResult",  # Deprecated
    "GenericProblemHandler",
]
