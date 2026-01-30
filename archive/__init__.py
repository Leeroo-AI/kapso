# Context Manager Module
#
# Modular context managers for gathering experiment context.
#
# Quick Start:
#   cm = ContextManagerFactory.create("kg_enriched", problem_handler, search_strategy)
#   context = cm.get_context(budget_progress=50)
#
# Add New Context Manager:
#   1. Copy _template.py to my_context_manager.py
#   2. Implement get_context() with your logic
#   3. Add presets in context_manager.yaml
#   4. Done! Auto-discovered on import.
#
# See README.md for full documentation.

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import (
    ContextManagerFactory,
    register_context_manager,
)

__all__ = [
    # Types
    "ContextData",
    "ExperimentHistoryProvider",
    # Base classes
    "ContextManager",
    # Factory
    "ContextManagerFactory",
    "register_context_manager",
]
