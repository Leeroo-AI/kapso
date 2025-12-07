# Environment Module - Problem Environment
#
# Handles problem definition, code execution, and evaluation.
#
# Submodules:
#   - handlers: Problem handlers (base, generic)
#   - evaluators: Pluggable evaluators for scoring
#   - stop_conditions: Pluggable stop conditions

# Handlers
from src.environment.handlers import (
    ProblemHandler,
    ProblemRunResult,
    GenericProblemHandler,
)

# Evaluators
from src.environment.evaluators import (
    Evaluator,
    EvaluationResult,
    EvaluatorFactory,
    register_evaluator,
)

# Stop Conditions
from src.environment.stop_conditions import (
    StopCondition,
    StopDecision,
    StopConditionFactory,
    register_stop_condition,
)

__all__ = [
    # Handlers
    "ProblemHandler",
    "ProblemRunResult",
    "GenericProblemHandler",
    # Evaluators
    "Evaluator",
    "EvaluationResult",
    "EvaluatorFactory",
    "register_evaluator",
    # Stop Conditions
    "StopCondition",
    "StopDecision",
    "StopConditionFactory",
    "register_stop_condition",
]
