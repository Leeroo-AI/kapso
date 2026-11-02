"""The web as the engine's boundary (docs/plans/redesign/wp4-config.md,
change 6): `ideation.web` and `implementation.web` decide, on every
session path, whether Claude Code's native web tools are on the
allow-list or disallowed and whether a Codex session gets `--search`;
the selector, the ensemble members and the lens planner follow
ideation's switch."""

from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.factory as factory_module
import kapso.gated_mcp as gated_mcp_module
from kapso.core.roles import resolve_optional, resolve_role
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.generic.ideation import WEB_TOOLS
from kapso.execution.search_strategies.generic.implementation import run_implementation
from kapso.execution.search_strategies.generic.lens_planning import run_lens_planner_session
from implementation_support import MCP_TOOL, FakeWorkspace, no_staging
from strategy_support import generic_params, packaged_hosts
from test_ensemble_ideation import CLAUDE_MEMBER, CODEX_MEMBER, SELECTOR, _plan, make_ensemble_strategy

HOSTS = packaged_hosts()


def captured_agent(monkeypatch, output="done"):
    configs = []

    class Agent:
        def __init__(self, config):
            configs.append(config)

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt, **kwargs):
            return SimpleNamespace(success=True, output=output, error=None, metadata={"tool_calls": []})

        def get_cumulative_cost(self):
            return 0.0

        def cleanup(self):
            pass

    monkeypatch.setattr(factory_module.CodingAgentFactory, "create", classmethod(lambda cls, config: Agent(config)))
    return configs


def web_tools_follow(config, web: bool) -> None:
    """The allow-list carries the tools when the switch is on; off, they
    are out of it and disallowed (binding under skip-permissions)."""
    allowed = config.agent_specific["allowed_tools"]
    assert all((tool in allowed) is web for tool in WEB_TOOLS)
    assert config.agent_specific["disallowed_tools"] == ([] if web else WEB_TOOLS)


@pytest.mark.parametrize("host", ["claude_code", "codex"])
@pytest.mark.parametrize("web", [True, False])
def test_implementation_follows_its_switch(tmp_path, monkeypatch, host, web):
    configs = captured_agent(monkeypatch)
    monkeypatch.setattr(gated_mcp_module, "get_mcp_config", lambda **kw: ({}, [MCP_TOOL]))
    monkeypatch.setattr(RepoMemoryManager, "ensure_exists_in_worktree", classmethod(lambda cls, folder: {}))
    monkeypatch.setattr(RepoMemoryManager, "render_summary_and_toc", classmethod(lambda cls, doc, max_chars=2500: "memory"))
    role = resolve_role(HOSTS, generic_params(host=host, implementation={"web": web}), "implementation")
    run_implementation(
        stage_harness_for=no_staging, solution="the plan", problem="the problem", branch_name="candidate-1",
        parent_branch_name="main", ideation_repo_memory_sections_consulted=None, lane_index=0,
        workspace=FakeWorkspace(str(tmp_path)), llm=None, registered_evaluation_manifest=None,
        sync_registered_evaluation=lambda folder: None, implementation_gates=["repo_memory"],
        gate_failure_policy="skip", implementation_role=role, implementation_fallback_model=None,
        env_strip=[], env_defaults={}, lane_env=None, clamped_timeout=lambda seconds: seconds,
        session_stream_path=lambda branch: str(tmp_path / f"{branch}.jsonl"), build_prompt=lambda **kw: "the prompt",
        previous_errors_text="", lane_brief="", note_session_started=lambda: None,
        note_session_end_facts=lambda facts: None, note_session_spend=lambda telemetry: None,
        await_registered_evaluation=lambda output: None,
    )
    (config,) = configs
    assert config.agent_type == host
    web_tools_follow(config, web)
    # Codex reads `web_search` (--search); the same key rides along on
    # Claude Code, where the tool lists are the boundary.
    assert config.agent_specific["web_search"] is web


@pytest.mark.parametrize("web", [True, False])
def test_the_members_and_the_selector_follow_ideations_switch(tmp_path, monkeypatch, web):
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)], selector=dict(SELECTOR),
        claude_output=f"<solution>{_plan('claude')}</solution>",
        codex_output=f"<solution>{_plan('codex')}</solution>",
        selector_output=f"<solution>{_plan('winner')}</solution>", web=web,
    )
    strategy._generate_solution("problem", "main")
    # The Codex member's runner is told whether to pass --search.
    assert [call["web_search"] for call in events["codex_calls"]] == [web]
    member, selector = events["configs"]
    web_tools_follow(member, web)
    web_tools_follow(selector, web)


@pytest.mark.parametrize("web", [True, False])
def test_the_lens_planner_follows_ideations_switch(tmp_path, monkeypatch, web):
    configs = captured_agent(monkeypatch, output="<axes>\n1. a\n</axes><lens_1>x</lens_1>")
    params = generic_params(host="claude_code", ideation={"web": web})
    planner = resolve_optional(HOSTS, params, {"cli": "claude_code", "effort": "max"}, name="lens_planner")
    assert planner.web is web
    run_lens_planner_session(
        "plan", str(tmp_path), planner=planner, bank_serving=None, env_strip=[], env_defaults={},
        clamped_timeout=lambda seconds: seconds, artifacts_dir=str(tmp_path), staged=None,
    )
    (config,) = configs
    web_tools_follow(config, web)
