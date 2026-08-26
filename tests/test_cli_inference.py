"""Hermetic tests for CLI-only inference (cli-only-inference design).

CliInference replaced LLMBackend's completion surface: every completion
runs as a read-only coding-agent CLI session. These tests fake the agent
factory at the seam CodingAgentFactory.create() and pin the contracts
the conversion depends on: role-spec resolution from the packaged
config, full-content prompt flattening, fail-loud on failed/empty
sessions, the codex web-search flag, and cost aggregation across the
CLI sessions plus the delegated embedding backend.
"""

from types import SimpleNamespace

import pytest

import kapso.core.llm as llm_module
from kapso.core.cli_inference import CliInference, default_inference_config
from kapso.execution.coding_agents.base import CodingResult


INFERENCE = {
    "default": {
        "cli": "codex",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "sandbox": "read-only",
        "timeout_seconds": 900,
    },
    "roles": {
        "research": {"web_search": True, "timeout_seconds": 1800},
        "commit_message": {"effort": "low", "timeout_seconds": 180},
        "claude_role": {"cli": "claude_code", "model": "opus"},
    },
}


class FakeAgent:
    def __init__(self, config, result, cost=0.0):
        self.config = config
        self.result = result
        self.cost = cost
        self.prompts = []

    def initialize(self, workspace_dir):
        self.workspace_dir = workspace_dir

    def generate_code(self, prompt, timeout_seconds=None):
        self.prompts.append(prompt)
        self.timeout_seconds = timeout_seconds
        return self.result

    def get_cumulative_cost(self):
        return self.cost

    def cleanup(self):
        self.cleaned = True


class FakeFactory:
    """Stands in for CodingAgentFactory: records configs, scripts results."""

    def __init__(self, outputs=("session output",), cost=0.0):
        self.outputs = list(outputs)
        self.cost = cost
        self.agents = []

    def create(self, config):
        result = self.outputs.pop(0)
        if not isinstance(result, CodingResult):
            result = CodingResult(success=True, output=result)
        agent = FakeAgent(config, result, cost=self.cost)
        self.agents.append(agent)
        return agent


def cli(factory=None, inference=None):
    return CliInference(
        inference=inference or INFERENCE,
        agent_factory=factory or FakeFactory(),
    )


def test_role_spec_merges_overrides_onto_default_and_unknown_role_raises():
    backend = cli()
    spec = backend._role_spec("research")
    # Overrides win; unset keys inherit the default spec.
    assert spec["web_search"] is True and spec["timeout_seconds"] == 1800
    assert spec["cli"] == "codex" and spec["effort"] == "xhigh"
    assert backend._role_spec(None)["timeout_seconds"] == 900

    with pytest.raises(ValueError, match="unknown inference role 'nope'"):
        backend._role_spec("nope")


def test_session_config_carries_the_role_spec_and_codex_sandbox():
    factory = FakeFactory()
    backend = cli(factory)
    backend.llm_completion(
        messages=[{"role": "user", "content": "q"}], role="commit_message"
    )
    agent = factory.agents[0]
    assert agent.config.agent_type == "codex"
    assert agent.config.model == "gpt-5.6-sol"
    assert agent.config.agent_specific["effort"] == "low"
    assert agent.config.agent_specific["sandbox"] == "read-only"
    assert agent.config.agent_specific["web_search"] is False
    assert agent.config.agent_specific["streaming"] is False
    assert agent.timeout_seconds == 180
    assert agent.cleaned  # scratch session torn down


def test_non_codex_role_gets_auth_mode_instead_of_sandbox():
    factory = FakeFactory()
    backend = cli(factory)
    backend.llm_completion(
        messages=[{"role": "user", "content": "q"}], role="claude_role"
    )
    spec = factory.agents[0].config.agent_specific
    assert factory.agents[0].config.agent_type == "claude_code"
    assert spec["auth_mode"] == "oauth"
    assert "sandbox" not in spec and "web_search" not in spec


def test_flatten_puts_system_first_and_carries_full_content():
    # Rule 6: content crosses into the session WHOLE — a long user
    # message must arrive byte-identical, system content hoisted first.
    factory = FakeFactory()
    backend = cli(factory)
    long_user = "pasted traceback " * 4000
    backend.llm_completion_with_system_prompt(
        system_prompt="you are a judge",
        user_message=long_user,
        role="commit_message",
    )
    prompt = factory.agents[0].prompts[0]
    assert prompt == "you are a judge\n\n" + long_user


def test_failed_or_empty_session_raises_loud():
    # Rule 2: a silent empty completion is how the research gate once
    # poisoned an E2E run — both failure shapes must raise.
    failed = FakeFactory(
        outputs=[CodingResult(success=False, output="", error="cli exploded")]
    )
    with pytest.raises(RuntimeError, match="cli exploded"):
        cli(failed).llm_completion(messages=[{"role": "user", "content": "q"}])

    empty = FakeFactory(
        outputs=[CodingResult(success=True, output="   \n")]
    )
    with pytest.raises(RuntimeError, match="empty output"):
        cli(empty).llm_completion(messages=[{"role": "user", "content": "q"}])


def test_web_search_completion_sets_codex_flag_and_depth_note():
    factory = FakeFactory()
    backend = cli(factory)
    backend.llm_completion_with_web_search(
        messages=[{"role": "user", "content": "find sota"}],
        search_context_size="high",
        role="research",
    )
    agent = factory.agents[0]
    assert agent.config.agent_specific["web_search"] is True
    assert agent.prompts[0].startswith("find sota")
    assert "primary sources" in agent.prompts[0]


def test_cumulative_cost_sums_cli_sessions_and_embedding_backend(monkeypatch):
    def fake_embedding(**kwargs):
        return SimpleNamespace(
            data=[{"embedding": [0.5]}],
            _hidden_params={"response_cost": 0.25},
        )

    monkeypatch.setattr(llm_module, "embedding", fake_embedding)
    factory = FakeFactory(outputs=["a", "b"], cost=1.5)
    backend = cli(factory)
    backend.llm_completion(messages=[{"role": "user", "content": "q"}])
    backend.llm_completion(messages=[{"role": "user", "content": "q"}])
    assert backend.create_embedding("text") == [0.5]
    assert backend.get_cumulative_cost() == pytest.approx(1.5 + 1.5 + 0.25)


def test_packaged_config_defines_every_role_the_code_selects():
    # The config/code contract: every role= literal at a call site must
    # resolve in the packaged inference block, else that seam dies at
    # runtime with "unknown inference role".
    config = default_inference_config()
    for key in ("cli", "model", "effort", "sandbox", "timeout_seconds"):
        assert key in config["default"], key
    for role in (
        "research",       # researcher / research gate
        "kg_rerank",      # kg_graph_search reranker
        "kg_navigate",    # kg_llm_navigation_search
        "repo_memory",    # repo-memory builders
        "commit_message", # commit message generator
        "tree_search",    # benchmark tree search
    ):
        assert role in config["roles"], role
