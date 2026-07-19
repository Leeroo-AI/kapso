"""Tests for capability-aware MCP gate resolution."""

import logging
import sys

import pytest

from kapso.gated_mcp import (
    GATES,
    GateCapabilityError,
    GateDefinition,
    get_allowed_tools_for_gates,
    get_mcp_config,
    resolve_gates,
)
from kapso.gated_mcp.server import _resolve_configuration


CAPABILITY_ENV = (
    "KG_INDEX_PATH",
    "OPENAI_API_KEY",
    "EXPERIMENT_HISTORY_PATH",
    "REPO_MEMORY_ROOT",
    "LEEROOPEDIA_API_KEY",
    "MCP_ENABLED_GATES",
    "MCP_GATE_FAILURE_POLICY",
)


@pytest.fixture(autouse=True)
def isolated_capabilities(monkeypatch):
    for name in CAPABILITY_ENV:
        monkeypatch.delenv(name, raising=False)


def command_lookup(*available):
    available_commands = set(available)
    return lambda command: f"/bin/{command}" if command in available_commands else None


def test_registry_declares_environment_and_command_requirements():
    assert GATES["research"].required_env == ["OPENAI_API_KEY"]
    assert GATES["experiment_history"].required_env == [
        "EXPERIMENT_HISTORY_PATH"
    ]
    assert GATES["leeroopedia"].required_env == ["LEEROOPEDIA_API_KEY"]
    assert GATES["leeroopedia"].required_commands == ["leeroopedia-mcp"]


def test_legacy_env_keys_are_folded_into_required_environment():
    definition = GateDefinition(tools=["tool"], env_keys=["LEGACY_KEY"])

    assert definition.required_env == ["LEGACY_KEY"]


def test_resolution_preserves_order_deduplicates_and_reports_every_gate():
    resolution = resolve_gates(
        ["repo_memory", "research", "repo_memory"],
        policy="skip",
        env={},
    )

    assert resolution.requested_gates == ("repo_memory", "research")
    assert resolution.enabled_gates == ("repo_memory",)
    assert resolution.unavailable_gates == ("research",)
    assert [item.reason for item in resolution.diagnostics] == [
        "available",
        "missing environment: OPENAI_API_KEY",
    ]
    assert resolution.to_dict()["unavailable_gates"] == ["research"]


def test_warn_policy_logs_and_omits_unavailable_gate(caplog):
    with caplog.at_level(logging.WARNING):
        resolution = resolve_gates(["research"], policy="warn", env={})

    assert resolution.enabled_gates == ()
    assert "Skipping unavailable MCP gate 'research'" in caplog.text
    assert "OPENAI_API_KEY" in caplog.text


def test_error_policy_aggregates_missing_capabilities():
    with pytest.raises(GateCapabilityError) as exc_info:
        resolve_gates(
            ["research", "leeroopedia"],
            policy="error",
            env={},
            command_resolver=command_lookup(),
        )

    error = exc_info.value
    assert [item.gate_name for item in error.diagnostics] == [
        "research",
        "leeroopedia",
    ]
    assert "OPENAI_API_KEY" in str(error)
    assert "leeroopedia-mcp" in str(error)


@pytest.mark.parametrize("policy", ["ignore", "", "WARN_AND_CONTINUE"])
def test_invalid_policy_is_a_configuration_error(policy):
    with pytest.raises(ValueError, match="Invalid gate failure policy"):
        resolve_gates([], policy=policy, env={})


@pytest.mark.parametrize(
    "operation",
    [
        lambda: resolve_gates(["typo"], env={}),
        lambda: get_allowed_tools_for_gates(["typo"], "server"),
        lambda: get_mcp_config(["typo"]),
    ],
)
def test_unknown_gate_is_always_a_configuration_error(operation):
    with pytest.raises(ValueError, match="Unknown gate"):
        operation()


def test_explicit_paths_satisfy_internal_gate_requirements(tmp_path):
    index_path = tmp_path / "knowledge.index"
    history_path = tmp_path / "history.json"

    servers, tools = get_mcp_config(
        ["idea", "experiment_history", "repo_memory"],
        project_root=tmp_path,
        kg_index_path=str(index_path),
        experiment_history_path=str(history_path),
        repo_root=str(tmp_path),
        gate_failure_policy="error",
        include_base_tools=False,
    )

    server = servers["gated-knowledge"]
    assert server["command"] == sys.executable
    assert server["env"]["MCP_ENABLED_GATES"] == (
        "idea,experiment_history,repo_memory"
    )
    assert server["env"]["KG_INDEX_PATH"] == str(index_path)
    assert server["env"]["EXPERIMENT_HISTORY_PATH"] == str(history_path)
    assert server["env"]["REPO_MEMORY_ROOT"] == str(tmp_path)
    assert server["env"]["MCP_GATE_FAILURE_POLICY"] == "error"
    assert "mcp__gated-knowledge__wiki_idea_search" in tools
    assert "mcp__gated-knowledge__get_top_experiments" in tools


def test_warn_config_keeps_available_gates_and_removes_missing_tools(tmp_path):
    servers, tools = get_mcp_config(
        ["research", "repo_memory"],
        project_root=tmp_path,
        repo_root=str(tmp_path),
        gate_failure_policy="warn",
        include_base_tools=False,
    )

    assert servers["gated-knowledge"]["env"]["MCP_ENABLED_GATES"] == (
        "repo_memory"
    )
    assert all("research_" not in tool for tool in tools)
    assert "mcp__gated-knowledge__get_repo_memory_summary" in tools


def test_only_available_external_gate_does_not_spawn_empty_internal_server(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "secret")

    servers, tools = get_mcp_config(
        ["leeroopedia"],
        project_root=tmp_path,
        gate_failure_policy="error",
        command_resolver=command_lookup("leeroopedia-mcp"),
        include_base_tools=False,
    )

    assert set(servers) == {"leeroopedia"}
    assert servers["leeroopedia"] == {
        "command": "leeroopedia-mcp",
        "env": {"LEEROOPEDIA_API_KEY": "secret"},
    }
    assert "mcp__leeroopedia__search_knowledge" in tools


def test_skipping_all_gates_returns_only_requested_base_tools(tmp_path):
    servers, tools = get_mcp_config(
        ["research"],
        project_root=tmp_path,
        gate_failure_policy="skip",
        include_base_tools=True,
    )

    assert servers == {}
    assert tools == ["Read", "Write", "Bash"]


def test_bundled_server_rejects_unknown_and_external_gate_names(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED_GATES", "typo")
    with pytest.raises(ValueError, match="Unknown gate"):
        _resolve_configuration()

    monkeypatch.setenv("MCP_ENABLED_GATES", "leeroopedia")
    with pytest.raises(ValueError, match="External gate"):
        _resolve_configuration()


def test_bundled_server_applies_capability_policy(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED_GATES", "research,repo_memory")
    monkeypatch.setenv("MCP_GATE_FAILURE_POLICY", "skip")

    configs = _resolve_configuration()

    assert set(configs) == {"repo_memory"}


def test_embedding_model_is_forwarded_to_experiment_history_gate(tmp_path):
    """The gate process learns its semantic-search model through the server
    env (the transport across the MCP process boundary); no gate → no
    forward."""
    history_path = tmp_path / "history.json"

    servers, _ = get_mcp_config(
        ["experiment_history"],
        project_root=tmp_path,
        experiment_history_path=str(history_path),
        experiment_embedding_model="text-embedding-3-small",
        gate_failure_policy="error",
        include_base_tools=False,
    )
    env = servers["gated-knowledge"]["env"]
    assert env["EXPERIMENT_EMBEDDING_MODEL"] == "text-embedding-3-small"

    servers_without, _ = get_mcp_config(
        ["repo_memory"],
        project_root=tmp_path,
        repo_root=str(tmp_path),
        experiment_embedding_model="text-embedding-3-small",
        gate_failure_policy="error",
        include_base_tools=False,
    )
    assert (
        "EXPERIMENT_EMBEDDING_MODEL"
        not in servers_without["gated-knowledge"]["env"]
    )
