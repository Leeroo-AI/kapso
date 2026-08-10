"""Hermetic tests for the Codex adapter's child-process environment.

Pins the judge-task fix: OPENAI_API_KEY is stripped from the codex child ONLY
when env_strip asks for it (solve.sh does that on non-judge tasks). Judge-scored
tasks (arenahardwriting/healthbench) leave env_strip empty so the agent's own
OpenAI GPT-judge win-rate/rubric eval keeps the key it needs — the regression
that made an arena-hard run unable to self-score.
"""

import os

import pytest

from kapso.execution.coding_agents.adapters.codex_agent import CodexCodingAgent
from kapso.execution.coding_agents.base import CodingAgentConfig


@pytest.fixture(autouse=True)
def codex_on_path(monkeypatch):
    # Adapter __init__ requires the codex binary; stub the lookup so the test
    # does not depend on it being installed.
    monkeypatch.setattr(
        "kapso.execution.coding_agents.adapters.codex_agent.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )


def make_agent(**agent_specific):
    return CodexCodingAgent(
        CodingAgentConfig(
            agent_type="codex",
            model="gpt-5.6-sol",
            debug_model="gpt-5.6-sol",
            agent_specific=agent_specific,
        )
    )


def test_openai_key_passes_through_on_judge_tasks(monkeypatch):
    # Judge-scored task: solve.sh leaves env_strip empty, so the key survives
    # to the agent's Bash tools (its GPT-judge eval needs it).
    monkeypatch.setenv("OPENAI_API_KEY", "scaffold-key")
    env = make_agent(env_strip=[])._child_env()
    assert env["OPENAI_API_KEY"] == "scaffold-key"


def test_openai_key_stripped_only_when_env_strip_requests_it(monkeypatch):
    # Non-judge task: solve.sh passes --strip-agent-env OPENAI_API_KEY.
    monkeypatch.setenv("OPENAI_API_KEY", "scaffold-key")
    env = make_agent(env_strip=["OPENAI_API_KEY", "ABSENT_VAR"])._child_env()
    assert "OPENAI_API_KEY" not in env
    # The orchestrating process keeps its own credential untouched.
    assert os.environ["OPENAI_API_KEY"] == "scaffold-key"


def test_env_overrides_set_and_defaults_fill_gaps_only(monkeypatch):
    monkeypatch.setenv("AMBIENT_VAR", "keep")
    env = make_agent(
        env_overrides={"PINNED_VAR": "on"},
        env_defaults={"AMBIENT_VAR": "ignored", "GAP_VAR": "filled"},
    )._child_env()
    assert env["PINNED_VAR"] == "on"
    assert env["AMBIENT_VAR"] == "keep"   # ambient wins over configured default
    assert env["GAP_VAR"] == "filled"     # only fills when absent
