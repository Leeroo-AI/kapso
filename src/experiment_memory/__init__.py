# Experiment Memory Module
#
# Stores and retrieves experiment history for the evolve loop.
# Provides MCP-compatible tools for agents to query past experiments.
#
# Storage:
# - JSON file for basic retrieval (top, recent)
# - Weaviate for semantic search (optional)
#
# Usage:
#   from src.experiment_memory import ExperimentHistoryStore
#   
#   store = ExperimentHistoryStore(json_path=".kapso/experiment_history.json")
#   store.add_experiment(node)
#   top = store.get_top_experiments(k=5)

from src.experiment_memory.store import (
    ExperimentHistoryStore,
    ExperimentRecord,
)

__all__ = [
    "ExperimentHistoryStore",
    "ExperimentRecord",
]
