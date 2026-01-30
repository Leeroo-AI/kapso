# Search Strategies Module
#
# Provides modular search strategies for experiment generation.
#
# Available strategies:
# - generic: Agent-based search for general problem solving
# - benchmark_tree_search: Tree search with handler-based evaluation
#
# To add a new strategy:
# 1. Create a new file in this directory (e.g., my_strategy.py)
# 2. Subclass SearchStrategy from base.py
# 3. Use @register_strategy("my_name") decorator
# 4. Add configuration presets in strategies.yaml
# 5. Configure in benchmark config.yaml:
#    search_strategy:
#      type: "my_name"
#      params: {...}

from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
    ExperimentResult,
)
from kapso.execution.search_strategies.factory import (
    SearchStrategyFactory,
    register_strategy,
)

# Import strategies to register them
from kapso.execution.search_strategies.generic import GenericSearch
from kapso.execution.search_strategies.benchmark_tree_search import BenchmarkTreeSearch

__all__ = [
    "SearchStrategy",
    "SearchStrategyConfig",
    "SearchNode",
    "ExperimentResult",
    "SearchStrategyFactory",
    "register_strategy",
    "GenericSearch",
    "BenchmarkTreeSearch",
]
