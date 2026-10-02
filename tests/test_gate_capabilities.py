"""Tests for capability-aware MCP gate resolution."""

import logging
import sys

import pytest

from kapso.gated_mcp import (
    GATES,
    GateCapabilityError,
    GateDefinition,
    gate_usage,
    get_allowed_tools_for_gates,
    get_mcp_config,
    knowledge_tools_block,
    resolve_gates,
    user_setup_gaps,
)
from kapso.core.prompt_loader import load_prompt
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


def test_registry_declares_environment_requirements():
    assert GATES["research"].required_env == ["OPENAI_API_KEY"]
    assert GATES["experiment_history"].required_env == [
        "EXPERIMENT_HISTORY_PATH"
    ]
    assert GATES["leeroopedia"].required_env == ["LEEROOPEDIA_API_KEY"]
    # Hosted, nothing to install: the only requirement is the key, and the
    # address comes from the packaged config (Rule 1).
    assert GATES["leeroopedia"].url == (
        "https://mcp.leeroopedia.com/mcp?token={LEEROOPEDIA_API_KEY}"
    )


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
        )

    error = exc_info.value
    assert [item.gate_name for item in error.diagnostics] == [
        "research",
        "leeroopedia",
    ]
    assert "OPENAI_API_KEY" in str(error)
    assert "LEEROOPEDIA_API_KEY" in str(error)


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


def test_hosted_gate_is_an_http_server_carrying_the_key_and_spawns_no_bundled_server(
    monkeypatch, tmp_path
):
    """The hosted server takes the key on the query string (a Bearer header
    is refused at initialize, verified 2026-10-01), so the launch fills the
    URL template from the env rather than passing env to a subprocess. The
    per-call timeout rides along: the CLI default of 60s cut the two
    slowest tools at exactly 60.0s in the live run of 2026-10-02."""
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "secret")

    servers, tools = get_mcp_config(
        ["leeroopedia"],
        project_root=tmp_path,
        gate_failure_policy="error",
        include_base_tools=False,
    )

    assert set(servers) == {"leeroopedia"}
    assert servers["leeroopedia"] == {
        "type": "http",
        "url": "https://mcp.leeroopedia.com/mcp?token=secret",
        "timeout": 600_000,
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


def test_mcp_sdk_matches_the_server_api_and_stays_pinned():
    """mcp 2.0 removed the 1.x low-level decorator API; an unpinned box install
    pulled it and the gate server crashed at startup in every production
    session (AttributeError: 'Server' object has no attribute 'list_tools')
    while dev environments kept passing — all gates silently dead, 2026-08-03.
    """
    from mcp.server import Server

    server = Server("contract-probe")
    for decorator in ("list_tools", "call_tool"):
        assert hasattr(server, decorator), (
            f"mcp SDK lacks Server.{decorator} — the installed mcp is "
            "incompatible with kapso.gated_mcp.server (2.x API?); migrate "
            "server.py before lifting the pyproject ceiling"
        )

    import pathlib
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    assert '"mcp>=1.9,<2"' in pyproject.read_text(), (
        "the mcp<2 ceiling left pyproject without a server.py migration"
    )


def test_research_gate_failures_propagate(monkeypatch):
    # Regression (E2E review 2026-08-26, R6): the research handlers'
    # swallows (plus the server dispatch catch) turned every backend
    # failure into research-shaped prose the agent trusted. All three
    # handlers must propagate. (The RESEARCH_WEB_SEARCH_MODEL route this
    # test also pinned died with the CLI-only inference conversion —
    # research now runs as a codex --search session, no route threading.)
    import asyncio

    import kapso.gated_mcp.backends as backends
    from kapso.gated_mcp.gates.research_gate import ResearchGate

    class Boom:
        def research(self, *args, **kwargs):
            raise RuntimeError("provider 400")

    monkeypatch.setattr(backends, "_researcher_backend", Boom())
    gate = ResearchGate()
    for tool in ("research_idea", "research_implementation", "research_study"):
        with pytest.raises(RuntimeError, match="provider 400"):
            asyncio.run(gate.handle_call(tool, {"query": "q"}))
    monkeypatch.setattr(backends, "_researcher_backend", None)


def test_gate_usage_attributes_by_server_and_tool_and_reads_credit_balances():
    """A session's tool calls become per-gate telemetry keys. `search_knowledge`
    exists on two servers (the bundled kg gate and hosted Leeroopedia), so
    attribution is by (server, tool); built-in tools and a user's own MCP
    servers are not knowledge calls. Leeroopedia answers end with a balance
    footer (live 2026-10-02: "---\n*Credits remaining: 761*"); the first and
    last balance of the session are kept, an error answer reports none."""
    footer = "\n\n---\n*Credits remaining: {}*"
    usage = gate_usage([
        {"name": "Bash", "input": {"command": "ls"}, "result": "file.py"},
        {"name": "mcp__leeroopedia__search_knowledge", "input": {"query": "q"}, "result": "# LoRA" + footer.format(761)},
        {"name": "mcp__leeroopedia__search_knowledge", "input": {"query": "r"}, "result": "The operation timed out."},
        {"name": "mcp__leeroopedia__build_plan", "input": {"goal": "g"}, "result": "# Plan" + footer.format(747)},
        {"name": "mcp__gated-knowledge__search_knowledge", "input": {"query": "q"}, "result": "pages"},
        {"name": "mcp__gated-knowledge__get_repo_memory_summary", "input": {}},
        {"name": "mcp__brightdata__scrape_as_markdown", "input": {"url": "u"}, "result": "html"},
    ])
    assert usage == {
        "leeroopedia_calls": 3,
        "leeroopedia.search_knowledge": 2,
        "leeroopedia.build_plan": 1,
        "leeroopedia_credits_first": 761,
        "leeroopedia_credits_last": 747,
        "kg_calls": 1,
        "kg.search_knowledge": 1,
        "repo_memory_calls": 1,
        "repo_memory.get_repo_memory_summary": 1,
    }
    assert gate_usage([]) == {}


def test_user_setup_gaps_name_only_what_the_user_must_provide(monkeypatch):
    """Injected env (set by Kapso at launch) is never a gap; a hosted gate's
    gap carries its own setup hint, a bundled gate's the variable to set."""
    gaps = user_setup_gaps(["repo_memory", "research", "leeroopedia", "inbox", "bank"])
    assert [(gate, missing) for gate, _, missing in gaps] == [
        ("research", ("OPENAI_API_KEY",)),
        ("leeroopedia", ("LEEROOPEDIA_API_KEY",)),
    ]
    fixes = {gate: fix for gate, fix, _ in gaps}
    assert fixes["research"] == "set OPENAI_API_KEY in .env"
    assert "LEEROOPEDIA_API_KEY" in fixes["leeroopedia"] and "nothing to install" in fixes["leeroopedia"]

    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_x")
    assert [gate for gate, _, _ in user_setup_gaps(["leeroopedia", "repo_memory"])] == []


def test_every_declared_gate_guidance_file_exists_and_names_its_tools():
    """Guidance is the gate's own prompt text, one file per phase; a file
    that does not load, or that never names the gate's tools, is a
    registry bug, not something a campaign should discover."""
    declared = {(gate, phase): path for gate, d in GATES.items() for phase, path in d.guidance.items()}
    assert ("leeroopedia", "ideation") in declared and ("leeroopedia", "implementation") in declared
    assert ("repo_memory", "implementation") in declared and ("experiment_history", "ideation") in declared
    for (gate, phase), path in declared.items():
        text = load_prompt(path)
        assert any(f"**{tool}**" in text for tool in GATES[gate].tools), (gate, phase)


def test_knowledge_tools_block_describes_only_the_mounted_gates():
    """The prompt names a gate's tools only when the session can call them:
    the shipped ideation whitelist carries repo memory, experiment history,
    research and Leeroopedia; it never carried the wiki_* tools the old static
    prompt advertised, and a session with no MCP (a codex ensemble member)
    gets no gate text at all."""
    servers, allowed = get_mcp_config(
        ["research", "experiment_history", "repo_memory", "leeroopedia"],
        experiment_history_path="/tmp/history.json",
        repo_root="/tmp/repo",
        gate_failure_policy="skip",
        include_base_tools=False,
    )
    # No OPENAI_API_KEY or LEEROOPEDIA_API_KEY in this test's env: those two
    # gates did not resolve and must not be described.
    block = knowledge_tools_block("ideation", allowed)
    assert "### Experiment History" in block and "**get_top_experiments**" in block
    assert "### RepoMemory Access" in block
    assert "Leeroopedia" not in block and "research_idea" not in block
    assert "wiki_idea_search" not in block
    # Registry order: experiment history (a MUST) precedes repo memory.
    assert block.index("### Experiment History") < block.index("### RepoMemory Access")

    with_leeroopedia = allowed + ["mcp__leeroopedia__build_plan", "mcp__leeroopedia__get_page"]
    block = knowledge_tools_block("implementation", with_leeroopedia)
    assert "### Leeroopedia" in block and "**build_plan**" in block
    assert "Experiment History" not in block  # no implementation guidance for that gate

    assert knowledge_tools_block("ideation", []) == ""
    assert knowledge_tools_block("ideation", ["Read", "Bash", "WebSearch"]) == ""
