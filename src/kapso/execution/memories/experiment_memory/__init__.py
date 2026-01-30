# Experiment Memory Module
#
# Stores and retrieves experiment history for the evolve loop.
# Provides MCP-compatible tools for agents to query past experiments.
#
# Features:
# - JSON file for basic retrieval (top, recent)
# - Weaviate for semantic search (optional)
# - LLM-based insight extraction from errors and successes
# - Duplicate detection to prevent redundant insights
#
# Usage:
#   from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
#   
#   store = ExperimentHistoryStore(
#       json_path=".kapso/experiment_history.json",
#       goal="Fine-tune LLaMA with LoRA",
#   )
#   store.add_experiment(node)  # Automatically extracts insights
#   top = store.get_top_experiments(k=5)
#   insights = store.get_experiments_with_insights(k=10)

from kapso.execution.memories.experiment_memory.store import (
    ExperimentHistoryStore,
    ExperimentRecord,
)
from kapso.execution.memories.experiment_memory.insight_extractor import (
    InsightExtractor,
    ExtractedInsight,
    InsightType,
)

__all__ = [
    # Store
    "ExperimentHistoryStore",
    "ExperimentRecord",
    # Insight extraction
    "InsightExtractor",
    "ExtractedInsight",
    "InsightType",
]
