"""Tests for the technical_difficulties capture chain.

Contract: the implementor is the primary author (a mandatory output tag);
when the tag is missing the strategy runs a fallback reconstruction — a
purely mechanical trigger, never score- or outcome-based. The field rides
the node into the experiment store and its tool renderings.
"""

from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.execution.search_strategies.generic import difficulties_generator
from kapso.gated_mcp.gates.experiment_history_gate import (
    ExperimentHistoryGate,
)
from kapso.execution.memories.experiment_memory.store import ExperimentRecord


def make_strategy_stub():
    strategy = GenericSearch.__new__(GenericSearch)
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

        def cleanup(self):
            pass

    monkeypatch.setattr(
        difficulties_generator, "ClaudeCodeCodingAgent", FakeAgent
    )
    stream = tmp_path / "stream.jsonl"
    stream.write_text('{"type":"assistant"}\n')

    text = difficulties_generator.generate_technical_difficulties(
        model="test-model",
        claude_auth_settings={"auth_mode": "oauth"},
        aws_region="us-east-1",
        env_strip=["OPENAI_API_KEY"],
        effort="xhigh",
        timeout_seconds=600,
        workspace_dir=str(tmp_path),
        solution="the plan",
        stream_artifact_path=str(stream),
    )

    assert text == "reconstructed: session died mid-eval"
    assert str(stream) in captured["prompt"]
    assert "the plan" in captured["prompt"]
    assert captured["config"].agent_specific["env_strip"] == [
        "OPENAI_API_KEY"
    ]
    assert captured["config"].agent_specific["allowed_tools"] == [
        "Read",
        "Bash",
    ]


def test_fallback_returns_empty_on_session_failure(monkeypatch, tmp_path):
    class DeadAgent:
        def __init__(self, config):
            pass

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt):
            from types import SimpleNamespace

            return SimpleNamespace(success=False, output="")

        def cleanup(self):
            pass

    monkeypatch.setattr(
        difficulties_generator, "ClaudeCodeCodingAgent", DeadAgent
    )
    text = difficulties_generator.generate_technical_difficulties(
        model="m",
        claude_auth_settings={"auth_mode": "oauth"},
        aws_region="us-east-1",
        env_strip=[],
        effort=None,
        timeout_seconds=60,
        workspace_dir=str(tmp_path),
        solution="s",
        stream_artifact_path=str(tmp_path / "absent.jsonl"),
    )
    assert text == ""


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
