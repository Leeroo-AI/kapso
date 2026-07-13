# Execution Module - Execution Engine
#
# Coordinates the experimentation loop: orchestrator, search strategies,
# coding agents, developer sessions, and context management.

# SolutionResult has minimal dependencies, import first
from kapso.execution.solution import SolutionResult
from kapso.execution.run_checkpoint import (
    RunCheckpointCompletedError,
    RunCheckpointCorruptError,
    RunCheckpointError,
    RunCheckpointIncompatibleError,
    RunCheckpointMissingError,
)
from kapso.execution.iteration_evaluator import (
    IterationEvaluationContext,
    IterationEvaluationError,
    IterationEvaluationResult,
    IterationEvaluationValidationError,
    IterationEvaluator,
)

# OrchestratorAgent has heavy dependencies (git, etc.), import lazily
def __getattr__(name):
    if name == "OrchestratorAgent":
        from kapso.execution.orchestrator import OrchestratorAgent
        return OrchestratorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "OrchestratorAgent",
    "SolutionResult",
    "RunCheckpointError",
    "RunCheckpointMissingError",
    "RunCheckpointCorruptError",
    "RunCheckpointIncompatibleError",
    "RunCheckpointCompletedError",
    "IterationEvaluationContext",
    "IterationEvaluationResult",
    "IterationEvaluationError",
    "IterationEvaluationValidationError",
    "IterationEvaluator",
]
