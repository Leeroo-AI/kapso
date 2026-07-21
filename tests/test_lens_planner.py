"""Contract tests for task-aware ideation lens planning.

What must hold: the planner block validates strictly (claude_code only — it
needs native web tools), static member lenses are forbidden while the planner
owns them, planner output parses fail-loud, and the once-per-campaign plan is
persisted and reused (a stale plan with the wrong member count raises rather
than silently mis-assigning lenses).
"""

import json
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    LENS_PLAN_FILENAME,
    normalize_ideation_lens_planner,
    parse_lens_plan,
    validate_lens_planner_against_ensemble,
)

PLANNER = {"cli": "claude_code", "model": "claude-fable-5", "effort": "max"}
MEMBERS = [
    {"cli": "codex", "model": "gpt-5.6-sol"},
    {"cli": "claude_code", "model": "claude-fable-5"},
]


def test_planner_block_validation():
    assert normalize_ideation_lens_planner(None) is None
    assert normalize_ideation_lens_planner(PLANNER)["model"] == "claude-fable-5"
    with pytest.raises(ValueError, match="claude_code"):
        normalize_ideation_lens_planner({"cli": "codex", "model": "m"})
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_ideation_lens_planner({**PLANNER, "web": True})
    with pytest.raises(ValueError, match="model"):
        normalize_ideation_lens_planner({"cli": "claude_code", "model": " "})
    with pytest.raises(ValueError, match="timeout"):
        normalize_ideation_lens_planner({**PLANNER, "timeout": 0})


def test_static_lenses_forbidden_with_planner():
    validate_lens_planner_against_ensemble(None, MEMBERS)
    validate_lens_planner_against_ensemble(PLANNER, MEMBERS)
    with pytest.raises(ValueError, match="static lens keys"):
        validate_lens_planner_against_ensemble(
            PLANNER, [{**MEMBERS[0], "lens": "math first"}, MEMBERS[1]]
        )
    with pytest.raises(ValueError, match="requires ideation_ensemble"):
        validate_lens_planner_against_ensemble(PLANNER, None)


def test_parse_lens_plan():
    output = (
        "reasoning...\n<lens_1>decision-theoretic attack</lens_1>\n"
        "<lens_2>measurement fidelity attack</lens_2>\n"
        "<sources>\n- https://example.org/paper\n</sources>"
    )
    plan = parse_lens_plan(output, 2)
    assert plan["lenses"] == [
        "decision-theoretic attack",
        "measurement fidelity attack",
    ]
    assert "example.org" in plan["sources"]
    with pytest.raises(ValueError, match="lens_3"):
        parse_lens_plan(output, 3)
    with pytest.raises(ValueError, match="lens_1"):
        parse_lens_plan("<lens_1>  </lens_1>", 1)


def make_stub(tmp_path, planner=PLANNER):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(tmp_path)
    strategy.ideation_ensemble = [dict(m) for m in MEMBERS]
    strategy.ideation_lens_planner = dict(planner) if planner else None
    strategy.iteration_count = 1
    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    return strategy


def test_resolver_disabled_without_planner(tmp_path):
    strategy = make_stub(tmp_path, planner=None)
    assert strategy._resolve_member_lenses("problem", str(tmp_path)) is None


def test_resolver_reuses_persisted_plan_without_a_session(tmp_path):
    strategy = make_stub(tmp_path)
    plan_dir = tmp_path / ".kapso"
    plan_dir.mkdir()
    (plan_dir / LENS_PLAN_FILENAME).write_text(
        json.dumps({"lenses": ["alpha", "beta"], "sources": ""})
    )
    assert strategy._resolve_member_lenses("problem", str(tmp_path)) == [
        "alpha",
        "beta",
    ]


def test_resolver_rejects_stale_plan_with_wrong_member_count(tmp_path):
    strategy = make_stub(tmp_path)
    plan_dir = tmp_path / ".kapso"
    plan_dir.mkdir()
    (plan_dir / LENS_PLAN_FILENAME).write_text(
        json.dumps({"lenses": ["only-one"], "sources": ""})
    )
    with pytest.raises(ValueError, match="delete it to replan"):
        strategy._resolve_member_lenses("problem", str(tmp_path))


def test_resolver_runs_planner_and_persists(tmp_path, monkeypatch):
    strategy = make_stub(tmp_path)
    strategy._claude_auth_settings = {"auth_mode": "oauth"}
    strategy.env_strip = []
    strategy.env_defaults = {}
    strategy.aws_region = "us-east-1"
    strategy.session_effort = "xhigh"

    captured = {}

    class FakeAgent:
        def __init__(self, config):
            captured["config"] = config

        def initialize(self, workspace):
            captured["workspace"] = workspace

        def generate_code(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(
                success=True,
                error=None,
                output=(
                    "<lens_1>literature-transfer attack</lens_1>"
                    "<lens_2>failure-mode attack</lens_2>"
                    "<sources>- s</sources>"
                ),
            )

        def cleanup(self):
            captured["cleaned"] = True

    import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module

    monkeypatch.setattr(claude_module, "ClaudeCodeCodingAgent", FakeAgent)

    lenses = strategy._resolve_member_lenses("the problem", str(tmp_path))
    assert lenses == ["literature-transfer attack", "failure-mode attack"]
    # Web tools granted, effort threaded, prompt carries roster + problem.
    assert captured["config"].agent_specific["allowed_tools"] == [
        "Read", "WebSearch", "WebFetch",
    ]
    assert captured["config"].agent_specific["effort"] == "max"
    assert "member 1: cli=codex" in captured["prompt"]
    assert "the problem" in captured["prompt"]
    # Persisted for reuse.
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["lenses"] == lenses and saved["planner_model"] == "claude-fable-5"
