"""Discovery contract of the deployment strategy registry.

The regression this pins: the registry scanned for selector_instruction.md
while every shipped strategy carries selector_instruction.txt, so
list_strategies() returned [] and every deploy() path raised (fixed
2026-09-02). If the instruction-file naming ever drifts again — on either
side — these fail instead of deploy() dying at runtime.
"""

import pytest

from kapso.deployment.factory import DeploymentFactory
from kapso.deployment.strategies.base import StrategyRegistry

SHIPPED = ["bentoml", "docker", "langgraph", "local", "modal"]


@pytest.fixture(autouse=True)
def fresh_registry():
    # The registry is a lazy singleton; other tests may have populated it.
    StrategyRegistry.reset()
    yield
    StrategyRegistry.reset()


def test_registry_discovers_every_shipped_strategy():
    assert DeploymentFactory.list_strategies() == SHIPPED


def test_each_discovered_strategy_serves_instructions_and_a_runner():
    registry = StrategyRegistry.get()
    for name in SHIPPED:
        config = registry.get_strategy(name)
        assert config.get_selector_instruction().strip(), name
        assert config.get_adapter_instruction().strip(), name
        assert config.has_runner(), name


def test_explain_strategy_reads_the_selector_summary():
    description = DeploymentFactory.explain_strategy("modal")
    assert "Unknown strategy" not in description
    assert description.strip()
