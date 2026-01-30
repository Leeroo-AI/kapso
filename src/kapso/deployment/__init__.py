# Deployment Module
#
# Provides unified deployment for solutions across different infrastructures.
# The system uses a Selector -> Adapter -> Runner pipeline to deploy code.
#
# Architecture:
# 1. StrategyRegistry: Auto-discovers deployment strategies from strategies/
# 2. Selector: Analyzes solution and selects optimal deployment strategy
# 3. Adapter: Uses coding agents to transform repo for deployment
# 4. Runner: Handles actual execution (function call, HTTP, subprocess)
# 5. Software: Unified interface for users (.run(), .stop(), etc.)
#
# Adding a New Strategy:
#   Create a new directory under strategies/ with:
#   - selector_instruction.md: When to choose this strategy
#   - adapter_instruction.md: How to adapt code for this strategy
#   - runner.py: Runtime execution class inheriting from Runner
#
# Usage:
#     from kapso.deployment import DeploymentFactory, DeployStrategy, DeployConfig
#     
#     config = DeployConfig(code_path="./solution", goal="My Goal")
#     software = DeploymentFactory.create(DeployStrategy.AUTO, config)
#     
#     # Unified interface - works the same for LOCAL, DOCKER, MODAL, etc.
#     result = software.run({"input": "data"})
#     software.stop()

from kapso.deployment.base import (
    # Core classes
    Software,
    DeployConfig,
    DeployStrategy,
    # Internal types (for advanced use)
    DeploymentSetting,
    AdaptationResult,
    DeploymentInfo,
)

from kapso.deployment.factory import DeploymentFactory
from kapso.deployment.software import DeployedSoftware

# Strategy registry and Runner base class
from kapso.deployment.strategies import Runner, StrategyRegistry, DeployStrategyConfig

# Selector components (LLM-based)
from kapso.deployment.selector.agent import SelectorAgent

# Adapter components
from kapso.deployment.adapter import AdapterAgent, AdaptationValidator

# Strategy runners (for direct access)
from kapso.deployment.strategies.local.runner import LocalRunner
from kapso.deployment.strategies.docker.runner import DockerRunner
from kapso.deployment.strategies.modal.runner import ModalRunner
from kapso.deployment.strategies.bentoml.runner import BentoMLRunner
from kapso.deployment.strategies.langgraph.runner import LangGraphRunner

# Backwards compatibility aliases
FunctionRunner = LocalRunner
HTTPRunner = DockerRunner
BentoCloudRunner = BentoMLRunner

__all__ = [
    # Main entry points
    "DeploymentFactory",
    "DeployStrategy",
    "DeployConfig",
    "Software",
    
    # Strategy registry
    "StrategyRegistry",
    "DeployStrategyConfig",
    "Runner",
    
    # Implementation (for advanced use)
    "DeployedSoftware",
    "DeploymentInfo",
    
    # Selector (LLM-based)
    "SelectorAgent",
    "DeploymentSetting",
    
    # Adapter
    "AdapterAgent",
    "AdaptationValidator",
    "AdaptationResult",
    
    # Strategy runners
    "LocalRunner",
    "DockerRunner",
    "ModalRunner",
    "BentoMLRunner",
    "LangGraphRunner",
    
    # Backwards compatibility
    "FunctionRunner",
    "HTTPRunner",
    "BentoCloudRunner",
]
