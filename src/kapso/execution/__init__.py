# Execution Module - Execution Engine
#
# Coordinates the experimentation loop: orchestrator, search strategies,
# coding agents, developer sessions, and context management.
#
# Every public symbol loads lazily through __getattr__, the same pattern as
# the top-level package. Eager re-exports pulled the whole execution tree in
# on `import kapso.execution.<anything>` — and kapso.core.cli_inference
# imports kapso.execution.coding_agents, while that tree's commit-message
# generator imports kapso.core.cli_inference. With eager exports a cold
# `import kapso.core.cli_inference` (kapso.researcher, the gate server's
# research backend, a test run alone) raised ImportError on the partially
# initialized module; lazily, a submodule import loads only that submodule.

import importlib

_LAZY_IMPORTS = {
    "OrchestratorAgent": ("kapso.execution.orchestrator", "OrchestratorAgent"),
    "SolutionResult": ("kapso.execution.solution", "SolutionResult"),
    "RunCheckpointError": ("kapso.execution.run_checkpoint", "RunCheckpointError"),
    "RunCheckpointMissingError": ("kapso.execution.run_checkpoint", "RunCheckpointMissingError"),
    "RunCheckpointCorruptError": ("kapso.execution.run_checkpoint", "RunCheckpointCorruptError"),
    "RunCheckpointIncompatibleError": ("kapso.execution.run_checkpoint", "RunCheckpointIncompatibleError"),
    "RunCheckpointCompletedError": ("kapso.execution.run_checkpoint", "RunCheckpointCompletedError"),
    "IterationEvaluationContext": ("kapso.execution.iteration_evaluator", "IterationEvaluationContext"),
    "IterationEvaluationResult": ("kapso.execution.iteration_evaluator", "IterationEvaluationResult"),
    "IterationEvaluationError": ("kapso.execution.iteration_evaluator", "IterationEvaluationError"),
    "IterationEvaluationValidationError": ("kapso.execution.iteration_evaluator", "IterationEvaluationValidationError"),
    "IterationEvaluator": ("kapso.execution.iteration_evaluator", "IterationEvaluator"),
    "EvaluationIntegrityError": ("kapso.execution.evaluation_integrity", "EvaluationIntegrityError"),
    "EvaluationIntegrityReport": ("kapso.execution.evaluation_integrity", "EvaluationIntegrityReport"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attribute = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_path), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
