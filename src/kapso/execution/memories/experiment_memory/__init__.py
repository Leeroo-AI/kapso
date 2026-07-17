# Experiment Memory Module
#
# Stores and retrieves experiment history for the evolve loop.
# Provides MCP-compatible tools for agents to query past experiments.
# The lesson artifact per experiment is the implementor-authored
# technical_difficulties field (fallback-generated when missing) — there
# is no separate insight-extraction pass.

from kapso.execution.memories.experiment_memory.store import (
    ExperimentHistoryStore,
    ExperimentRecord,
)

__all__ = [
    "ExperimentHistoryStore",
    "ExperimentRecord",
]
