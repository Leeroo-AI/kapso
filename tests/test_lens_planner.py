"""Contract tests for task-aware ideation lens planning.

What must hold: the planner block validates strictly (claude_code only — it
needs native web tools), static member lenses are forbidden while the planner
owns them, planner output parses fail-loud at iteration 1, and — since the
per-iteration redesign — every later iteration runs a keep-or-revise session
against the campaign evidence: revise overwrites the plan, keep bumps its
iteration, an invalid revision falls back LOUDLY to the previous validated
plan (never killing the campaign), and every decision lands in the
lens_plan_history.jsonl audit trail.
"""

import json
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.lens_planning import (
    DESIGN_AXES_DEFAULT,
    LENS_PLAN_FILENAME,
    LENS_PLAN_HISTORY_FILENAME,
    normalize_design_axes,
    normalize_ideation_lens_planner,
    parse_lens_plan,
    parse_lens_revision,
    validate_lens_planner_against_ensemble,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch

PLANNER = {"cli": "claude_code", "model": "claude-fable-5", "effort": "max"}
MEMBERS = [
    {"cli": "codex", "model": "gpt-5.6-sol"},
    {"cli": "claude_code", "model": "claude-fable-5"},
]

REVISE_OUTPUT = (
    "<revision_rationale>ratings line exhausted at 2.63; pretrained family "
    "has higher ceiling</revision_rationale>\n"
    "<lens_1>fine-tune an open relational checkpoint</lens_1>\n"
    "<lens_2>drift-focused rank mechanics</lens_2>\n"
    "<sources>\n- https://example.org/rt\n</sources>"
)


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


def test_parse_lens_revision_matrix():
    keep = parse_lens_revision("<keep>line still has return</keep>", 2)
    assert keep == {"kind": "keep", "rationale": "line still has return"}

    revise = parse_lens_revision(REVISE_OUTPUT, 2)
    assert revise["kind"] == "revise"
    assert revise["lenses"] == [
        "fine-tune an open relational checkpoint",
        "drift-focused rank mechanics",
    ]
    assert "example.org" in revise["sources"]
    assert "exhausted" in revise["rationale"]

    # Rationale/sources optional on a complete lens set.
    bare = parse_lens_revision("<lens_1>a a a</lens_1><lens_2>b b b</lens_2>", 2)
    assert bare["kind"] == "revise" and bare["rationale"] == ""

    # Incomplete lens set without keep -> invalid, never a raise.
    partial = parse_lens_revision("<lens_1>only one</lens_1>", 2)
    assert partial["kind"] == "invalid" and "lens_2" in partial["reason"]
    assert parse_lens_revision("", 2)["kind"] == "invalid"
    assert parse_lens_revision("<keep>  </keep>", 2)["kind"] == "invalid"


class FakePlannerAgent:
    """Session double: records config/prompt, replies from a script."""

    outputs: list = []
    calls: list = []

    def __init__(self, config):
        FakePlannerAgent.calls.append({"config": config})

    def initialize(self, workspace):
        FakePlannerAgent.calls[-1]["workspace"] = workspace

    def generate_code(self, prompt):
        FakePlannerAgent.calls[-1]["prompt"] = prompt
        spec = FakePlannerAgent.outputs.pop(0)
        return SimpleNamespace(
            success=spec.get("success", True),
            error=spec.get("error"),
            output=spec.get("output", ""),
        )

    def get_cumulative_cost(self):
        return 1.25

    def cleanup(self):
        pass


def make_stub(tmp_path, planner=PLANNER, iteration=1, node_history=()):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(tmp_path)
    strategy.ideation_ensemble = [dict(m) for m in MEMBERS]
    strategy.ideation_lens_planner = dict(planner) if planner else None
    strategy.iteration_count = iteration
    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    # Mirror __init__-set attributes the planner-session path reads (stub
    # gotcha: new GenericSearch instance attributes must be added here too).
    strategy.design_axes = DESIGN_AXES_DEFAULT
    strategy.ideation_web_search = True
    strategy._web_disallowed_tools = []
    strategy._claude_auth_settings = {"auth_mode": "oauth"}
    strategy.env_strip = []
    strategy.env_defaults = {}
    strategy.aws_region = "us-east-1"
    strategy.session_effort = "xhigh"
    strategy.node_history = list(node_history)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=False)
    strategy.bank_serving = None
    return strategy


@pytest.fixture
def fake_planner(monkeypatch):
    import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module

    FakePlannerAgent.outputs = []
    FakePlannerAgent.calls = []
    monkeypatch.setattr(claude_module, "ClaudeCodeCodingAgent", FakePlannerAgent)
    return FakePlannerAgent


def _write_plan(tmp_path, lenses, iteration=1):
    plan_dir = tmp_path / ".kapso"
    plan_dir.mkdir(exist_ok=True)
    (plan_dir / LENS_PLAN_FILENAME).write_text(
        json.dumps(
            {
                "lenses": lenses,
                "sources": "- prior source",
                "planner_model": "claude-fable-5",
                "iteration": iteration,
                "decision": "initial",
                "rationale": "",
            }
        )
    )


def _history(tmp_path):
    path = tmp_path / ".kapso" / LENS_PLAN_HISTORY_FILENAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resolver_disabled_without_planner(tmp_path):
    strategy = make_stub(tmp_path, planner=None)
    assert strategy._resolve_member_lenses("problem", str(tmp_path)) == (None, 0.0)


def test_same_iteration_reuses_plan_without_a_session(tmp_path, fake_planner):
    strategy = make_stub(tmp_path, iteration=3)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=3)
    assert strategy._resolve_member_lenses("problem", str(tmp_path)) == (
        ["alpha", "beta"],
        0.0,
    )
    assert fake_planner.calls == []


def test_resolver_rejects_stale_plan_with_wrong_member_count(tmp_path):
    strategy = make_stub(tmp_path)
    plan_dir = tmp_path / ".kapso"
    plan_dir.mkdir()
    (plan_dir / LENS_PLAN_FILENAME).write_text(
        json.dumps({"lenses": ["only-one"], "sources": ""})
    )
    with pytest.raises(ValueError, match="delete it to replan"):
        strategy._resolve_member_lenses("problem", str(tmp_path))


def test_initial_plan_runs_persists_and_records_history(tmp_path, fake_planner):
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [
        {
            "output": (
                "<lens_1>literature-transfer attack</lens_1>"
                "<lens_2>failure-mode attack</lens_2>"
                "<sources>- s</sources>"
            )
        }
    ]
    lenses, cost = strategy._resolve_member_lenses("the problem", str(tmp_path))
    assert lenses == ["literature-transfer attack", "failure-mode attack"]
    assert cost == 1.25
    call = fake_planner.calls[0]
    assert call["config"].agent_specific["allowed_tools"] == [
        "Read", "WebSearch", "WebFetch",
    ]
    assert call["config"].agent_specific["effort"] == "max"
    assert "member 1: cli=codex" in call["prompt"]
    assert "the problem" in call["prompt"]
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["lenses"] == lenses
    assert saved["decision"] == "initial" and saved["iteration"] == 1
    assert [h["decision"] for h in _history(tmp_path)] == ["initial"]


def test_initial_planner_failure_still_raises(tmp_path, fake_planner):
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [{"success": False, "error": "boom"}]
    with pytest.raises(RuntimeError, match="lens planner session failed"):
        strategy._resolve_member_lenses("problem", str(tmp_path))


def test_later_iteration_revises_with_campaign_evidence(tmp_path, fake_planner):
    node = SearchNode(node_id=0, branch_name="e0", score=2.63)
    node.feedback = "family X closed: plateau at 2.63; reopen if drift model appears"
    node.solution = "the incumbent cohort-ranking solution"
    strategy = make_stub(tmp_path, iteration=4, node_history=[node])
    _write_plan(tmp_path, ["old lens one", "old lens two"], iteration=3)
    fake_planner.outputs = [{"output": REVISE_OUTPUT}]

    lenses, cost = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == [
        "fine-tune an open relational checkpoint",
        "drift-focused rank mechanics",
    ]
    assert cost == 1.25
    prompt = fake_planner.calls[0]["prompt"]
    # The replanner sees the evidence, in full (no truncation).
    assert "old lens one" in prompt
    assert "champion score: 2.63" in prompt
    assert "family X closed" in prompt
    assert "the incumbent cohort-ranking solution" in prompt
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["decision"] == "revise" and saved["iteration"] == 4
    assert saved["lenses"] == lenses
    assert [h["decision"] for h in _history(tmp_path)] == ["revise"]


def test_keep_decision_bumps_iteration_and_skips_next_session(
    tmp_path, fake_planner
):
    strategy = make_stub(tmp_path, iteration=2)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=1)
    fake_planner.outputs = [{"output": "<keep>line still paying</keep>"}]

    lenses, _ = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == ["alpha", "beta"]
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["decision"] == "keep" and saved["iteration"] == 2
    assert saved["rationale"] == "line still paying"
    assert [h["decision"] for h in _history(tmp_path)] == ["keep"]
    # Same iteration again (resume/retry): no second session.
    assert strategy._resolve_member_lenses("problem", str(tmp_path)) == (
        ["alpha", "beta"],
        0.0,
    )
    assert len(fake_planner.calls) == 1


def test_invalid_revision_falls_back_loudly_to_previous_plan(
    tmp_path, fake_planner
):
    strategy = make_stub(tmp_path, iteration=2)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=1)
    fake_planner.outputs = [{"output": "no tags at all"}]

    lenses, cost = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == ["alpha", "beta"]
    assert cost == 1.25
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    # Iteration NOT bumped: a same-iteration retry replans.
    assert saved["iteration"] == 1
    history = _history(tmp_path)
    assert history[-1]["decision"] == "failed"
    assert "lens_1" in history[-1]["reason"]
    assert history[-1]["raw_output"] == "no tags at all"


def test_failed_revision_session_falls_back_loudly(tmp_path, fake_planner):
    strategy = make_stub(tmp_path, iteration=5)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=4)
    fake_planner.outputs = [{"success": False, "error": "rate limited"}]

    lenses, _ = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == ["alpha", "beta"]
    assert _history(tmp_path)[-1]["decision"] == "failed"
    assert "rate limited" in _history(tmp_path)[-1]["reason"]


def test_normalize_design_axes():
    assert normalize_design_axes(None) == DESIGN_AXES_DEFAULT
    assert normalize_design_axes(["a", " b "]) == ("a", "b")
    with pytest.raises(ValueError):
        normalize_design_axes([])
    with pytest.raises(ValueError):
        normalize_design_axes(["ok", ""])
    with pytest.raises(ValueError):
        normalize_design_axes("not-a-list")


def test_planner_and_replanner_prompts_carry_design_axes(
    tmp_path, fake_planner
):
    """Anti-freeze contract (user-directed 2026-07-27): both lens sessions
    receive the design-axis map, and the replanner additionally carries the
    axis-status contract (ACTIVE / SATURATED-with-evidence / DEFERRED) so
    freezing an axis becomes a dated, evidence-bearing claim — never a
    silent default (the run4 feature-matrix freeze, run_0003..run_0011)."""
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [
        {
            "output": (
                "<lens_1>a</lens_1><lens_2>b</lens_2><sources>- s</sources>"
            )
        }
    ]
    strategy._resolve_member_lenses("problem", str(tmp_path))
    planner_prompt = fake_planner.calls[0]["prompt"]
    assert "Design axes of the solution space" in planner_prompt
    assert "input representation" in planner_prompt

    strategy2 = make_stub(tmp_path, iteration=2)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=1)
    fake_planner.outputs = [{"output": "<keep>still paying</keep>"}]
    strategy2._resolve_member_lenses("problem", str(tmp_path))
    replanner_prompt = fake_planner.calls[-1]["prompt"]
    assert "Design axes of the solution space" in replanner_prompt
    assert "Axis-coverage contract" in replanner_prompt
    assert "SATURATED" in replanner_prompt
    assert "DEFERRED" in replanner_prompt


def test_planner_session_mounts_bank_gate_when_serving(tmp_path, fake_planner):
    # Regression (serving v2 §5): the lens planner is the direction-setter —
    # with bank_serving staged its session must carry the bank MCP tools;
    # without it the session stays a plain Read/WebSearch/WebFetch session
    # (asserted by the existing initial-plan test via bank_serving=None).
    strategy = make_stub(tmp_path)
    strategy.bank_serving = {
        "KAPSO_BANK_DIR": str(tmp_path),
        "KAPSO_BANK_HEAD": "abc",
        "KAPSO_SERVING_PULL_LOG": str(tmp_path / "pull.jsonl"),
        "KAPSO_TASK_FAMILY": "entity_binary_classification",
        "KAPSO_PROBE_BUDGET": "1",
    }
    fake_planner.outputs = [
        {
            "output": (
                "<lens_1>lens one one</lens_1>"
                "<lens_2>lens two two</lens_2>"
                "<sources>- s</sources>"
            )
        }
    ]
    lenses, _ = strategy._resolve_member_lenses("the problem", str(tmp_path))
    assert lenses == ["lens one one", "lens two two"]
    tools = fake_planner.calls[0]["config"].agent_specific["allowed_tools"]
    assert any("bank_index" in name for name in tools)
    assert fake_planner.calls[0]["config"].agent_specific["mcp_servers"]
