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


def test_deploy_config_defaults_are_the_packaged_deployment_block():
    """The selector and adapter used to hardcode claude-opus-4-5: a model
    the config never named, and one the codex agent cannot serve. The
    defaults now come from config.yaml's deployment block, one source."""
    from kapso.core.config import load_deployment_defaults
    from kapso.deployment.base import DeployConfig
    from kapso.execution.solution import SolutionResult

    block = load_deployment_defaults()
    config = DeployConfig(solution=SolutionResult(goal="g", code_path="/nowhere"))
    assert (config.coding_agent, config.model) == (block["coding_agent"], block["model"])
    assert block["model"] == "claude-opus-5"


def test_factory_threads_agent_and_model_to_selector_and_adapter(monkeypatch):
    """Every session deploy() runs is built from the DeployConfig, so a
    per-call or per-config agent and model reach the selector and adapter."""
    import kapso.deployment.adapter.agent as adapter_module
    import kapso.deployment.selector.agent as selector_module
    from kapso.deployment.base import DeployConfig, DeploymentSetting
    from kapso.execution.solution import SolutionResult

    built = []

    class FakeSelector:
        def __init__(self, coding_agent_type, model):
            built.append(("selector", coding_agent_type, model))

        def select(self, solution, allowed_strategies=None):
            return DeploymentSetting(strategy="local", provider="local", resources={}, interface={}, reasoning="fake")

    class FakeAdapter:
        def __init__(self, coding_agent_type, model, max_retries):
            built.append(("adapter", coding_agent_type, model))

        def adapt(self, solution, setting, allowed_strategies=None):
            return "adapted"

    monkeypatch.setattr(selector_module, "SelectorAgent", FakeSelector)
    monkeypatch.setattr(adapter_module, "AdapterAgent", FakeAdapter)
    config = DeployConfig(
        solution=SolutionResult(goal="g", code_path="/nowhere"),
        coding_agent="codex", model="gpt-5.6-sol",
    )
    setting = DeploymentFactory._select_strategy(config)
    assert DeploymentFactory._adapt_repo(config, setting) == "adapted"
    assert built == [
        ("selector", "codex", "gpt-5.6-sol"),
        ("adapter", "codex", "gpt-5.6-sol"),
    ]
