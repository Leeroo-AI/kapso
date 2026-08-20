"""Hermetic tests for the `implementation_web` knob (decision #3).

Pins the contract: the knob gates live-web access in implementation
sessions on BOTH CLIs — WebSearch/WebFetch in the claude whitelist and
--search on codex — and is independent of the ideation `web_search` knob
(ideation web on + implementation web off is expressible).
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
import kapso.execution.coding_agents.factory as factory_module
import kapso.gated_mcp as gated_mcp_module
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.generic.implementation import (
    run_implementation,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch

MCP_TOOL = "mcp__repo_memory__get_repo_memory_summary"


class FakeSession:
    def __init__(self, folder):
        self.session_folder = folder

    def schedule_repo_memory_update(self, *, solution_spec, run_result):
        pass


class FakeWorkspace:
    def __init__(self, folder):
        self._folder = folder

    def create_experiment_session(self, branch_name, parent_branch_name, llm=None):
        return FakeSession(self._folder)

    def finalize_session(self, session):
        pass


def run_fake_implementation(tmp_path, monkeypatch, *, implementation_cli,
                            implementation_web):
    """Drive run_implementation with fakes; return the captured agent config."""
    captured = {}

    class FakeAgent:
        def __init__(self, config):
            captured["config"] = config

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt):
            return SimpleNamespace(
                success=True,
                output="<score>1.0</score>",
                metadata={},
                error=None,
            )

        def get_cumulative_cost(self):
            return 0.0

        def cleanup(self):
            pass

    monkeypatch.setattr(claude_module, "ClaudeCodeCodingAgent", FakeAgent)
    monkeypatch.setattr(
        factory_module.CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: FakeAgent(config)),
    )
    monkeypatch.setattr(
        gated_mcp_module, "get_mcp_config", lambda **kw: ({}, [MCP_TOOL])
    )
    monkeypatch.setattr(
        RepoMemoryManager,
        "ensure_exists_in_worktree",
        classmethod(lambda cls, folder: {}),
    )
    monkeypatch.setattr(
        RepoMemoryManager,
        "render_summary_and_toc",
        classmethod(lambda cls, doc, max_chars=2500: "memory"),
    )

    run_implementation(
        solution="the plan",
        problem="the problem",
        branch_name="candidate-1",
        parent_branch_name="main",
        ideation_repo_memory_sections_consulted=None,
        lane_index=0,
        workspace=FakeWorkspace(str(tmp_path)),
        llm=None,
        registered_evaluation_manifest=None,
        sync_registered_evaluation=lambda folder: None,
        implementation_gates=["repo_memory"],
        gate_failure_policy="skip",
        implementation_cli=implementation_cli,
        implementation_model="the-model",
        implementation_fallback_model=None,
        implementation_web=implementation_web,
        claude_auth_settings={"auth_mode": "oauth"},
        env_strip=[],
        env_defaults={},
        aws_region="us-east-1",
        lane_env=None,
        session_effort=None,
        clamped_timeout=lambda seconds: seconds,
        implementation_timeout=600,
        session_stream_path=lambda branch: str(tmp_path / f"{branch}.jsonl"),
        build_prompt=lambda **kw: "the prompt",
        previous_errors_text="",
        lane_brief="",
        note_session_started=lambda: None,
        note_session_end_facts=lambda facts: None,
        await_registered_evaluation=lambda output: None,
    )
    return captured["config"]


def test_web_on_adds_web_tools_to_claude_whitelist(tmp_path, monkeypatch):
    config = run_fake_implementation(
        tmp_path, monkeypatch,
        implementation_cli="claude_code", implementation_web=True,
    )
    assert config.agent_specific["allowed_tools"] == [
        "Read", "Write", "Edit", "Bash", "WebSearch", "WebFetch", MCP_TOOL,
    ]


def test_web_off_keeps_claude_whitelist_web_free(tmp_path, monkeypatch):
    config = run_fake_implementation(
        tmp_path, monkeypatch,
        implementation_cli="claude_code", implementation_web=False,
    )
    assert config.agent_specific["allowed_tools"] == [
        "Read", "Write", "Edit", "Bash", MCP_TOOL,
    ]


@pytest.mark.parametrize("web", [True, False])
def test_codex_search_follows_the_knob(tmp_path, monkeypatch, web):
    # The codex adapter defaults web_search to True, so the knob must be
    # threaded EXPLICITLY — otherwise implementation_web: false would leave
    # a --search side-door on codex-primary benchmarks (relbench).
    config = run_fake_implementation(
        tmp_path, monkeypatch,
        implementation_cli="codex", implementation_web=web,
    )
    assert config.agent_specific["web_search"] is web


@contextmanager
def _patched_super_init(workspace_dir):
    from kapso.execution.search_strategies.base import SearchStrategy

    original = SearchStrategy.__init__

    def fake_init(self, config, wd=None, import_from_checkpoint=False):
        self.params = config.params or {}
        self.workspace_dir = workspace_dir
        self.feedback_generator = None

    SearchStrategy.__init__ = fake_init
    yield
    SearchStrategy.__init__ = original


def test_ideation_and_implementation_web_knobs_are_independent(tmp_path):
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(
            SimpleNamespace(
                params={"web_search": True, "implementation_web": False}
            ),
            str(tmp_path),
        )
    assert strategy.ideation_web_search is True
    assert strategy._web_disallowed_tools == []
    assert strategy.implementation_web is False


def test_implementation_web_defaults_true(tmp_path):
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(SimpleNamespace(params={}), str(tmp_path))
    assert strategy.implementation_web is True
