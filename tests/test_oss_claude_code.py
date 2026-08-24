"""Hermetic tests for the oss_claude_code adapter and its ensemble wiring.

Contract under test: an OSS member drives the Claude Code CLI against an
Anthropic-compatible endpoint using ONLY the configured credential (named env
var), with every first-party escape hatch stripped from the child env — and
the strategy only accepts endpoint wiring on oss_claude_code members.
"""

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters.oss_claude_code_agent import (
    FIRST_PARTY_ENV_VARS,
    OssClaudeCodeCodingAgent,
)
from kapso.execution.search_strategies.generic.ideation import (
    ENSEMBLE_MEMBER_CLIS,
    normalize_ensemble_member,
)

GLM_MODEL = "accounts/fireworks/models/glm-5p2"
BASE_URL = "https://api.fireworks.ai/inference"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in FIRST_PARTY_ENV_VARS + ("ANTHROPIC_AUTH_TOKEN", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "kapso.execution.coding_agents.adapters.oss_claude_code_agent.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )


def make_agent(monkeypatch=None, **overrides):
    agent_specific = {
        "base_url": BASE_URL,
        "auth_token_env": "FIREWORKS_API_KEY",
        **overrides,
    }
    return OssClaudeCodeCodingAgent(
        CodingAgentConfig(
            agent_type="oss_claude_code",
            model=GLM_MODEL,
            debug_model=GLM_MODEL,
            agent_specific=agent_specific,
        )
    )


def test_child_env_gets_endpoint_auth_and_loses_first_party(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-first-party")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    env = make_agent()._get_env()
    assert env["ANTHROPIC_BASE_URL"] == BASE_URL
    assert env["ANTHROPIC_AUTH_TOKEN"] == "fw-secret"
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_MODEL"):
        assert name not in env
    # Internal CLI slots must not route to Anthropic model ids the endpoint
    # cannot serve.
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == GLM_MODEL
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == GLM_MODEL


def test_missing_provider_key_fails_loud_at_init(monkeypatch):
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        make_agent()


def test_missing_endpoint_wiring_fails_loud(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret")
    with pytest.raises(ValueError, match="base_url"):
        make_agent(base_url="")
    with pytest.raises(ValueError, match="auth_token_env"):
        make_agent(auth_token_env=None)


def test_env_strip_and_defaults_still_apply(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    env = make_agent(
        env_strip=["OPENAI_API_KEY"],
        env_defaults={"BASH_DEFAULT_TIMEOUT_MS": "900000"},
    )._get_env()
    assert "OPENAI_API_KEY" not in env
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == "900000"


def test_ensemble_accepts_oss_member_and_requires_wiring():
    assert "oss_claude_code" in ENSEMBLE_MEMBER_CLIS
    member = normalize_ensemble_member(
        {
            "cli": "oss_claude_code",
            "model": GLM_MODEL,
            "effort": "max",
            "base_url": BASE_URL,
            "auth_token_env": "FIREWORKS_API_KEY",
        },
        role="ideation_ensemble[2]",
    )
    assert member["base_url"] == BASE_URL
    with pytest.raises(ValueError, match="auth_token_env|base_url"):
        normalize_ensemble_member(
            {"cli": "oss_claude_code", "model": GLM_MODEL},
            role="ideation_ensemble[2]",
        )


def test_endpoint_wiring_rejected_on_first_party_members():
    with pytest.raises(ValueError, match="only valid for"):
        normalize_ensemble_member(
            {"cli": "claude_code", "model": "claude-fable-5", "base_url": BASE_URL},
            role="ideation_ensemble[1]",
        )
