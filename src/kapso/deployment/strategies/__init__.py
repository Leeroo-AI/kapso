# Deployment Strategies Module
#
# Each deployment option is a self-contained package with:
# - selector_instruction.md: When to choose this strategy (for selector LLM)
# - adapter_instruction.md: How to adapt/deploy (for adapter LLM)
# - runner.py: Runtime execution interface
#
# The StrategyRegistry auto-discovers all strategies from subdirectories.
#
# To add a new deployment strategy:
# 1. Create strategies/{name}/ directory
# 2. Add selector_instruction.md describing when to use it
# 3. Add adapter_instruction.md with DEPLOY COMMAND and RUN INTERFACE
# 4. Add runner.py with a class inheriting from Runner

from kapso.deployment.strategies.base import Runner, StrategyRegistry, DeployStrategyConfig

__all__ = ["Runner", "StrategyRegistry", "DeployStrategyConfig"]
