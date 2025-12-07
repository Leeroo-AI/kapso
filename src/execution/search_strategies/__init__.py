# Search Strategies Module
#
# Provides modular search strategies for experiment generation.
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

from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
)
from src.execution.search_strategies.factory import (
    SearchStrategyFactory,
    register_strategy,
)

__all__ = [
    "SearchStrategy",
    "SearchStrategyConfig", 
    "ExperimentResult",
    "SearchStrategyFactory",
    "register_strategy",
]
