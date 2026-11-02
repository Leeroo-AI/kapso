"""Tests for the technical_difficulties capture chain.

Contract: the implementor is the primary author (a mandatory output tag);
when the tag is missing the strategy runs a fallback reconstruction — a
purely mechanical trigger, never score- or outcome-based. The field rides
the node into the experiment store and its tool renderings.
"""

from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.execution.search_strategies.generic import difficulties_generator
from kapso.core.roles import SessionRole
from kapso.gated_mcp.gates.experiment_history_gate import (
    ExperimentHistoryGate,
)
from kapso.execution.memories.experiment_memory.store import ExperimentRecord


def make_strategy_stub():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.harness = None
    return strategy


def test_extract_agent_result_parses_the_new_tag():
    strategy = make_strategy_stub()
    output = (
        "<code_changes_summary>built it</code_changes_summary>\n"
        "<evaluation_script_path>eval.py</evaluation_script_path>\n"
        "<evaluation_output>ok</evaluation_output>\n"
        "<score>0.9</score>\n"
        "<technical_difficulties>OOM at batch 16; fixed with batch 8"
        "</technical_difficulties>"
    )
    result = strategy._extract_agent_result(output)
    assert result["technical_difficulties"] == (
        "OOM at batch 16; fixed with batch 8"
    )
    assert result["score"] == 0.9


def test_fallback_prompt_renders_and_tag_is_parsed(tmp_path, monkeypatch):
    """The fallback module extracts the tag from its session's output."""
    captured = {}

    class FakeAgent:
        def __init__(self, config):
            captured["config"] = config

        def initialize(self, workspace):
            captured["workspace"] = workspace

        def generate_code(self, prompt):
            captured["prompt"] = prompt
            from types import SimpleNamespace

            return SimpleNamespace(
                success=True,
                output=(
                    "narration...\n<technical_difficulties>\nreconstructed: "
                    "session died mid-eval\n</technical_difficulties>"
                ),
            )

        def get_cumulative_cost(self):
            return 0.4

        def cleanup(self):
            pass

    monkeypatch.setattr(
        difficulties_generator.CodingAgentFactory, "create", classmethod(lambda cls, config: FakeAgent(config))
    )
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"type":"assistant"}\n')

    text, telemetry = difficulties_generator.generate_technical_difficulties(
        role=SessionRole(
            name="implementation", cli="claude_code", model="test-model",
            effort="xhigh", timeout=None, settings={"auth_mode": "oauth"},
        ),
        env_strip=["OPENAI_API_KEY"],
        timeout_seconds=600,
        workspace_dir=str(tmp_path),
        solution="the plan",
        stream_artifact_path=str(stream),
        branch_name="generic_exp_3",
        parent_branch_name="main",
        staged=None,
    )

    assert text == "reconstructed: session died mid-eval"
    assert telemetry["cost_usd"] == 0.4
    assert str(stream) in captured["prompt"]
    assert "the plan" in captured["prompt"]
    # The prompt reads the branch before it concludes what was lost.
    assert "git log --oneline main..generic_exp_3" in captured["prompt"]
    assert "git diff --stat main..generic_exp_3" in captured["prompt"]
    assert captured["config"].agent_specific["env_strip"] == [
        "OPENAI_API_KEY"
    ]
    assert captured["config"].agent_specific["allowed_tools"] == [
        "Read",
        "Bash",
    ]
    # The session runs on the role's host with the host entry's settings.
    assert captured["config"].agent_type == "claude_code" and captured["config"].model == "test-model"
    assert captured["config"].agent_specific["auth_mode"] == "oauth"
    assert captured["config"].agent_specific["effort"] == "xhigh"


def test_fallback_returns_empty_on_session_failure(monkeypatch, tmp_path):
    class DeadAgent:
        def __init__(self, config):
            pass

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt):
            from types import SimpleNamespace

            return SimpleNamespace(success=False, output="")

        def get_cumulative_cost(self):
            return 0.4

        def cleanup(self):
            pass

    monkeypatch.setattr(
        difficulties_generator.CodingAgentFactory, "create", classmethod(lambda cls, config: DeadAgent(config))
    )
    text, telemetry = difficulties_generator.generate_technical_difficulties(
        role=SessionRole(name="implementation", cli="claude_code", model="m", effort=None, timeout=None),
        env_strip=[],
        timeout_seconds=60,
        workspace_dir=str(tmp_path),
        solution="s",
        stream_artifact_path=str(tmp_path / "absent.jsonl"),
        branch_name="b",
        parent_branch_name="main",
        staged=None,
    )
    assert text == ""
    # The session that produced nothing was still paid for.
    assert telemetry["cost_usd"] == 0.4


def test_gate_renders_difficulties_block():
    gate = ExperimentHistoryGate.__new__(ExperimentHistoryGate)
    record = ExperimentRecord(
        node_id=1,
        solution="s",
        score=0.9,
        feedback="fine",
        branch_name="b",
        had_error=False,
        error_message="",
        timestamp="t",
        technical_difficulties="hit the special-token trap; re-init fixed it",
    )
    rendered = gate._format_experiments([record], "Title")
    assert "Technical difficulties:" in rendered
    assert "special-token trap" in rendered
