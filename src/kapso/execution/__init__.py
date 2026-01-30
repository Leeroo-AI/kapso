# Execution Module - Execution Engine
#
# Coordinates the experimentation loop: orchestrator, search strategies,
# coding agents, developer sessions, and context management.

# SolutionResult has minimal dependencies, import first
from kapso.execution.solution import SolutionResult

# OrchestratorAgent has heavy dependencies (git, etc.), import lazily
def __getattr__(name):
    if name == "OrchestratorAgent":
        from kapso.execution.orchestrator import OrchestratorAgent
        return OrchestratorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "OrchestratorAgent",
    "SolutionResult",
]

