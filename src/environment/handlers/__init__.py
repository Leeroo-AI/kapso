# Problem Handlers
#
# Base and concrete implementations for problem handling.
#
# The handler system uses pluggable evaluators and stop conditions.
# See src/environment/evaluators/ and src/environment/stop_conditions/

from src.environment.handlers.base import ProblemHandler, ProblemRunResult
from src.environment.handlers.generic import GenericProblemHandler

__all__ = [
    # Base
    "ProblemHandler",
    "ProblemRunResult",
    # Generic handler
    "GenericProblemHandler",
]
