"""A build session that died is a fact the lane carries: run_implementation
returns the session's error beside its output, so the node can be marked
rather than read from the prose the error was folded into. The session
that retries it on the fallback model is a new session: its prompt is
built when it starts, so it reads the notes left while the first ran."""

from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.factory as factory_module
import kapso.gated_mcp as gated_mcp_module
from kapso.core.roles import SessionRole
from kapso.execution.experiment_workspace.experiment_session import SessionPushError
from kapso.execution.inbox import leave_note, notes_path, read_notes
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.generic.implementation import (
    render_campaign_notes,
    run_implementation,
)

from implementation_support import MCP_TOOL, FakeWorkspace, no_staging


def run_session(
    tmp_path,
    monkeypatch,
    result,
    workspace=None,
    note_session_spend=lambda telemetry: None,
    fallback_model=None,
    build_prompt=lambda **kw: "the prompt",
):
    """Drive run_implementation with an agent whose one session ends as
    ``result``; a callable ``result`` is given each session's prompt."""

    class FakeAgent:
        def __init__(self, config):
            pass

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt):
            return result(prompt) if callable(result) else result

        def get_cumulative_cost(self):
            return 2.5

        def cleanup(self):
            pass

    monkeypatch.setattr(factory_module.CodingAgentFactory, "create", classmethod(lambda cls, config: FakeAgent(config)))
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
    return run_implementation(
        stage_harness_for=no_staging,
        solution="the plan",
        problem="the problem",
        branch_name="candidate-1",
        parent_branch_name="main",
        ideation_repo_memory_sections_consulted=None,
        lane_index=0,
        workspace=workspace or FakeWorkspace(str(tmp_path)),
        llm=None,
        registered_evaluation_manifest=None,
        sync_registered_evaluation=lambda folder: None,
        implementation_gates=["repo_memory"],
        gate_failure_policy="skip",
        implementation_role=SessionRole(
            name="implementation", cli="claude_code", model="the-model",
            effort=None, timeout=600, settings={"auth_mode": "oauth"}, web=False,
        ),
        implementation_fallback_model=fallback_model,
        env_strip=[],
        env_defaults={},
        lane_env=None,
        clamped_timeout=lambda seconds: seconds,
        session_stream_path=lambda branch: str(tmp_path / f"{branch}.jsonl"),
        build_prompt=build_prompt,
        previous_errors_text="",
        lane_brief="",
        note_session_started=lambda: None,
        note_session_end_facts=lambda facts: None,
        note_session_spend=note_session_spend,
        await_registered_evaluation=lambda output: None,
    )


def test_a_session_bill_is_noted_before_the_close_that_can_fail(tmp_path, monkeypatch):
    class PushFails(FakeWorkspace):
        def finalize_session(self, session):
            raise SessionPushError("origin rejected the push")

    noted = []
    with pytest.raises(SessionPushError):
        run_session(
            tmp_path,
            monkeypatch,
            SimpleNamespace(
                success=True,
                output="done",
                metadata={"tool_calls": []},
                error=None,
            ),
            workspace=PushFails(str(tmp_path)),
            note_session_spend=noted.append,
        )
    assert [telemetry["cost_usd"] for telemetry in noted] == [2.5]


def test_dead_session_returns_its_error_beside_its_output(tmp_path, monkeypatch):
    output, _, _, suspended, build_error = run_session(
        tmp_path, monkeypatch, SimpleNamespace(
            success=False, output="partial work",
            metadata={"tool_calls": []}, error="provider kill",
        ),
    )
    assert build_error == "provider kill"
    assert output == "Implementation failed: provider kill\n\npartial work"
    assert suspended is None


def test_finished_session_returns_no_error(tmp_path, monkeypatch):
    output, _, _, suspended, build_error = run_session(
        tmp_path, monkeypatch, SimpleNamespace(
            success=True, output="<score>null</score>",
            metadata={"tool_calls": []}, error=None,
        ),
    )
    assert build_error is None
    assert output == "<score>null</score>"
    assert suspended is None


def test_the_crash_retry_reads_a_note_left_while_the_first_session_ran(tmp_path, monkeypatch):
    """The person is told the running session will not see the note and
    every session that starts afterwards will. The retry on the fallback
    model starts afterwards, and it is the session that then builds."""
    note = "Stop writing to PROD: use DEV_DB.\n" + "\n".join(f"- schema {index} is read-only" for index in range(200))
    prompts = []

    def session(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            leave_note(notes_path(tmp_path), note)   # the person, while the first session runs
            return SimpleNamespace(success=False, output="", metadata={"tool_calls": []}, error="provider kill")
        return SimpleNamespace(success=True, output="<score>null</score>", metadata={"tool_calls": []}, error=None)

    output, _, _, _, build_error = run_session(
        tmp_path, monkeypatch, session, fallback_model="the-fallback-model",
        build_prompt=lambda **kw: "the prompt\n\n" + render_campaign_notes(read_notes(notes_path(tmp_path))),
    )

    first, retry = prompts
    assert first == "the prompt\n\n" and note not in first
    assert retry.startswith("the prompt\n\n## Notes from the person running this campaign") and note in retry
    assert "NOTE: a previous session on this lane ended prematurely (provider kill)" in retry
    assert (output, build_error) == ("<score>null</score>", None)


def test_dead_session_without_an_error_message_fails_loud(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="failed without an error message"):
        run_session(
            tmp_path, monkeypatch, SimpleNamespace(
                success=False, output="", metadata={"tool_calls": []}, error=None,
            ),
        )
