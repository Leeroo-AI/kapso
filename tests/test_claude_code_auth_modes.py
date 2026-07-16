"""Hermetic tests for Claude Code authentication mode selection."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)
from kapso.execution.coding_agents.base import CodingAgentConfig


AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_PROFILE",
    "AWS_REGION",
)


@pytest.fixture(autouse=True)
def isolated_auth_environment(monkeypatch):
    for name in AUTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "kapso.execution.coding_agents.adapters.claude_code_agent.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )


def make_config(**agent_specific):
    return CodingAgentConfig(
        agent_type="claude_code",
        model="claude-opus-4-6",
        debug_model="claude-opus-4-6",
        agent_specific=agent_specific,
    )


def auth_status(*, logged_in=True, auth_method="claude.ai"):
    return SimpleNamespace(
        returncode=0 if logged_in else 1,
        stdout=json.dumps(
            {"loggedIn": logged_in, "authMethod": auth_method}
        ),
        stderr="",
    )


def test_explicit_bedrock_uses_aws_and_removes_direct_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="bedrock"))
    env = agent._get_env()

    assert agent._auth_mode == "bedrock"
    assert agent._use_bedrock is True
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-east-1"
    assert "ANTHROPIC_API_KEY" not in env


def test_explicit_api_key_wins_over_ambient_provider_flags(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "higher-precedence-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="api_key"))
    env = agent._get_env()

    assert agent._auth_mode == "api_key"
    assert env["ANTHROPIC_API_KEY"] == "anthropic"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env


def test_explicit_oauth_uses_cli_status_and_sanitizes_status_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-win")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return auth_status()

    monkeypatch.setattr(subprocess, "run", fake_run)

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="oauth"))
    env = agent._get_env()

    assert captured["command"] == ["claude", "auth", "status"]
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "CLAUDE_CODE_USE_BEDROCK" not in captured["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert agent.get_capabilities()["oauth"] is True


def test_oauth_token_does_not_require_status_subprocess(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("auth status should not run with an OAuth token")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="oauth"))

    assert agent._auth_mode == "oauth"
    assert agent._get_env()["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth"


def test_auto_prefers_bedrock_then_api_key(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "kapso")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth")

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="auto"))

    assert agent._auth_mode == "bedrock"


def test_auto_prefers_api_key_over_oauth(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth")

    agent = ClaudeCodeCodingAgent(make_config())

    assert agent._requested_auth_mode == "auto"
    assert agent._auth_mode == "api_key"


def test_auto_falls_back_to_stored_oauth_login(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: auth_status())

    agent = ClaudeCodeCodingAgent(make_config(auth_mode="auto"))

    assert agent._auth_mode == "oauth"


def test_auto_fails_with_actionable_error_when_no_credentials(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: auth_status(logged_in=False),
    )

    with pytest.raises(ValueError, match="No Claude Code credentials found"):
        ClaudeCodeCodingAgent(make_config(auth_mode="auto"))


def test_oauth_rejects_status_that_only_reports_an_api_key(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: auth_status(auth_method="api_key"),
    )

    with pytest.raises(ValueError, match="OAuth credentials not found"):
        ClaudeCodeCodingAgent(make_config(auth_mode="oauth"))


def test_invalid_auth_mode_is_rejected_before_credential_checks():
    with pytest.raises(ValueError, match="Invalid Claude Code auth_mode"):
        ClaudeCodeCodingAgent(make_config(auth_mode="magic"))


@pytest.mark.parametrize(
    ("use_bedrock", "env_name", "expected"),
    [
        (True, "AWS_PROFILE", "bedrock"),
        (False, "ANTHROPIC_API_KEY", "api_key"),
    ],
)
def test_use_bedrock_alias_preserves_behavior_and_warns(
    monkeypatch, use_bedrock, env_name, expected
):
    monkeypatch.setenv(env_name, "configured")

    with pytest.warns(DeprecationWarning, match="use_bedrock is deprecated"):
        agent = ClaudeCodeCodingAgent(make_config(use_bedrock=use_bedrock))

    assert agent._auth_mode == expected


def test_explicit_auth_mode_wins_when_deprecated_alias_is_also_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")

    with pytest.warns(DeprecationWarning):
        agent = ClaudeCodeCodingAgent(
            make_config(auth_mode="api_key", use_bedrock=True)
        )

    assert agent._auth_mode == "api_key"


def test_env_overrides_participate_in_validation_and_resolution():
    agent = ClaudeCodeCodingAgent(
        make_config(
            auth_mode="api_key",
            env_overrides={"ANTHROPIC_API_KEY": "from-config"},
        )
    )

    assert agent._get_env()["ANTHROPIC_API_KEY"] == "from-config"


def test_env_strip_removes_named_vars_from_agent_env_only(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "scaffold-key")
    agent = ClaudeCodeCodingAgent(
        make_config(
            auth_mode="api_key",
            env_strip=["OPENAI_API_KEY", "ABSENT_VAR"],
        )
    )

    env = agent._get_env()

    assert "OPENAI_API_KEY" not in env
    assert env["ANTHROPIC_API_KEY"] == "anthropic"
    # The orchestrating process keeps its own credential untouched.
    assert os.environ["OPENAI_API_KEY"] == "scaffold-key"


def test_agent_env_passes_openai_key_through_without_env_strip(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "scaffold-key")
    agent = ClaudeCodeCodingAgent(make_config(auth_mode="api_key"))

    assert agent._get_env()["OPENAI_API_KEY"] == "scaffold-key"
