"""Contract tests for task-aware ideation lens planning.

What must hold: the planner block validates strictly (claude_code only — it
needs native web tools), static member lenses are forbidden while the planner
owns them, planner output parses fail-loud at iteration 1, and — since the
per-iteration redesign — every later iteration runs a keep-or-revise session
against the campaign evidence: revise overwrites the plan, keep bumps its
iteration, an invalid revision falls back LOUDLY to the previous validated
plan (never killing the campaign), and every decision lands in the
lens_plan_history.jsonl audit trail. The design axes are the planner's:
the first plan names them, a revision statuses every one, a plan without
them is refused on reload, and the judge reads them from the plan.
"""

import json
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.base import SearchNode
from kapso.core.roles import resolve_members, resolve_optional
from kapso.execution.search_strategies.generic.lens_planning import (
    LENS_PLAN_FILENAME,
    LENS_PLAN_HISTORY_FILENAME,
    current_plan_axes,
    parse_lens_plan,
    parse_lens_revision,
    validate_lens_planner,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from strategy_support import generic_params, packaged_hosts

HOSTS = packaged_hosts()
PARAMS = generic_params()
PLANNER = {"cli": "claude_code", "model": "claude-fable-5", "effort": "max"}
MEMBERS = [
    {"cli": "codex", "model": "gpt-5.6-sol"},
    {"cli": "claude_code", "model": "claude-fable-5"},
]


def planner_role(spec=PLANNER):
    return resolve_optional(HOSTS, PARAMS, spec, name="lens_planner")


def member_roles(members=MEMBERS):
    return resolve_members(HOSTS, PARAMS, members)

AXES = ["axis one", "axis two"]
STATUSED_AXES = ["axis one — SATURATED: ratings plateau at 2.63", "axis two — ACTIVE (lens 2)"]
REVISE_OUTPUT = (
    "<revision_rationale>ratings line exhausted at 2.63; pretrained family "
    "has higher ceiling</revision_rationale>\n"
    "<axes>\n1. axis one — SATURATED: ratings plateau at 2.63\n2. axis two — ACTIVE (lens 2)\n</axes>\n"
    "<lens_1>fine-tune an open relational checkpoint</lens_1>\n"
    "<lens_2>drift-focused rank mechanics</lens_2>\n"
    "<sources>\n- https://example.org/rt\n</sources>"
)


def test_planner_block_validation():
    assert planner_role(None) is None
    assert planner_role().model == "claude-fable-5" and planner_role().name == "lens_planner"
    with pytest.raises(ValueError, match="Claude Code host"):
        validate_lens_planner(planner_role({"cli": "codex", "model": "m"}), member_roles())
    with pytest.raises(ValueError, match="unknown key web"):
        planner_role({**PLANNER, "web": True})
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        planner_role({"cli": "claude_code", "model": " "})
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        planner_role({**PLANNER, "timeout": 0})


def test_static_lenses_forbidden_with_planner():
    validate_lens_planner(None, member_roles())
    validate_lens_planner(planner_role(), member_roles())
    with pytest.raises(ValueError, match="static lens keys"):
        validate_lens_planner(
            planner_role(), member_roles([{**MEMBERS[0], "lens": "math first"}, MEMBERS[1]])
        )
    with pytest.raises(ValueError, match="needs an ensemble"):
        validate_lens_planner(planner_role(), None)


def test_parse_lens_plan():
    lenses = (
        "<lens_1>decision-theoretic attack</lens_1>\n"
        "<lens_2>measurement fidelity attack</lens_2>\n"
        "<sources>\n- https://example.org/paper\n</sources>"
    )
    output = "reasoning...\n<axes>\n1. decision theory\n\n2) measurement\n</axes>\n" + lenses
    plan = parse_lens_plan(output, 2)
    assert plan["axes"] == ["decision theory", "measurement"]
    assert plan["lenses"] == [
        "decision-theoretic attack",
        "measurement fidelity attack",
    ]
    assert "example.org" in plan["sources"]
    with pytest.raises(ValueError, match="lens_3"):
        parse_lens_plan(output, 3)
    with pytest.raises(ValueError, match="lens_1"):
        parse_lens_plan("<lens_1>  </lens_1>", 1)
    # The axes are the planner's to name: a plan without them, or with a
    # list not numbered from 1, fails loud at iteration 1.
    with pytest.raises(ValueError, match="no <axes> tag"):
        parse_lens_plan(lenses, 2)
    with pytest.raises(ValueError, match="numbered from 1"):
        parse_lens_plan("<axes>\n2. second first\n</axes>\n" + lenses, 2)
    with pytest.raises(ValueError, match="names no axis"):
        parse_lens_plan("<axes>\n</axes>\n" + lenses, 2)


def test_parse_lens_revision_matrix():
    keep = parse_lens_revision("<keep>line still has return</keep>", 2, 2)
    assert keep == {"kind": "keep", "rationale": "line still has return"}

    revise = parse_lens_revision(REVISE_OUTPUT, 2, 2)
    assert revise["kind"] == "revise"
    assert revise["axes"] == STATUSED_AXES
    assert revise["lenses"] == [
        "fine-tune an open relational checkpoint",
        "drift-focused rank mechanics",
    ]
    assert "example.org" in revise["sources"]
    assert "exhausted" in revise["rationale"]

    # Rationale/sources optional on a complete lens set with its axes.
    bare = parse_lens_revision(
        "<axes>1. a — ACTIVE (lens 1)\n2. b — DEFERRED: later\n3. c — ACTIVE (lens 2)</axes>"
        "<lens_1>a a a</lens_1><lens_2>b b b</lens_2>", 2, 2,
    )
    assert bare["kind"] == "revise" and bare["rationale"] == "" and len(bare["axes"]) == 3

    # A revision that drops an axis, or statuses none, is discarded: an
    # unexplored dimension cannot be erased unstatused.
    fewer = parse_lens_revision(
        "<axes>1. a — ACTIVE (lens 1)</axes><lens_1>a a a</lens_1><lens_2>b b b</lens_2>", 2, 2,
    )
    assert fewer["kind"] == "invalid" and "every axis" in fewer["reason"]
    unstatused = parse_lens_revision("<lens_1>a a a</lens_1><lens_2>b b b</lens_2>", 2, 2)
    assert unstatused["kind"] == "invalid" and "<axes>" in unstatused["reason"]

    # Incomplete lens set without keep -> invalid, never a raise.
    partial = parse_lens_revision("<lens_1>only one</lens_1>", 2, 2)
    assert partial["kind"] == "invalid" and "lens_2" in partial["reason"]
    assert parse_lens_revision("", 2, 2)["kind"] == "invalid"
    assert parse_lens_revision("<keep>  </keep>", 2, 2)["kind"] == "invalid"


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
    strategy.harness = None
    strategy.workspace_dir = str(tmp_path)
    strategy.ideation_ensemble = member_roles()
    strategy.ideation_lens_planner = planner_role(planner) if planner else None
    strategy.iteration_count = iteration
    strategy.harness = None
    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    # Mirror __init__-set attributes the planner-session path reads (stub
    # gotcha: new GenericSearch instance attributes must be added here too).
    strategy.env_strip = []
    strategy.env_defaults = {}
    strategy.budget_snapshot = None
    strategy.budget_snapshot_monotonic = None
    strategy.node_history = list(node_history)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=False)
    strategy.bank_serving = None
    strategy.kg_index_path = None
    strategy.experience_intro_path = None
    strategy.hosted = None
    strategy.ideation_bills = []
    return strategy


@pytest.fixture
def fake_planner(monkeypatch):
    import kapso.execution.coding_agents.factory as factory_module

    FakePlannerAgent.outputs = []
    FakePlannerAgent.calls = []
    monkeypatch.setattr(factory_module.CodingAgentFactory, "create", classmethod(lambda cls, config: FakePlannerAgent(config)))
    return FakePlannerAgent


def _write_plan(tmp_path, lenses, iteration=1, axes=AXES):
    plan_dir = tmp_path / ".kapso"
    plan_dir.mkdir(exist_ok=True)
    (plan_dir / LENS_PLAN_FILENAME).write_text(
        json.dumps(
            {
                "axes": list(axes),
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
        json.dumps({"axes": ["a"], "lenses": ["only-one"], "sources": ""})
    )
    with pytest.raises(ValueError, match="delete it to replan"):
        strategy._resolve_member_lenses("problem", str(tmp_path))


@pytest.mark.parametrize("axes", [None, [], ["ok", " "], "not a list"])
def test_a_plan_without_axes_is_refused_on_reload(tmp_path, axes):
    """A plan file predating the planner's axes, or hand-edited without
    them, is refused by name: the judge and the replanner read them."""
    strategy = make_stub(tmp_path, iteration=2)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=1, axes=axes or [])
    if axes is not None:
        plan = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
        plan["axes"] = axes
        (tmp_path / ".kapso" / LENS_PLAN_FILENAME).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="has no axes — delete it to replan"):
        strategy._resolve_member_lenses("problem", str(tmp_path))
    with pytest.raises(ValueError, match="has no axes"):
        current_plan_axes(str(tmp_path))


def test_the_judge_reads_the_axes_from_the_plan(tmp_path):
    """No plan (no planner, or none yet): no axes and no frontier duty;
    a plan: its axes, the statused ones after a revision."""
    assert current_plan_axes(str(tmp_path)) == []
    _write_plan(tmp_path, ["alpha", "beta"], axes=STATUSED_AXES)
    assert current_plan_axes(str(tmp_path)) == STATUSED_AXES


def test_initial_plan_runs_persists_and_records_history(tmp_path, fake_planner):
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [
        {
            "output": (
                "<axes>\n1. axis one\n2. axis two\n</axes>"
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
    # The planner names the axes; nobody hands it a list.
    assert "name the design axes of THIS task's solution space" in call["prompt"]
    assert "{{" not in call["prompt"]
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["axes"] == AXES and saved["lenses"] == lenses
    assert saved["decision"] == "initial" and saved["iteration"] == 1
    assert [h["decision"] for h in _history(tmp_path)] == ["initial"]
    assert _history(tmp_path)[0]["axes"] == AXES


def test_initial_planner_failure_still_raises(tmp_path, fake_planner):
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [{"success": False, "error": "boom"}]
    with pytest.raises(RuntimeError, match="lens planner session failed"):
        strategy._resolve_member_lenses("problem", str(tmp_path))


def test_an_initial_plan_without_axes_raises(tmp_path, fake_planner):
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [{"output": "<lens_1>a a</lens_1><lens_2>b b</lens_2><sources>- s</sources>"}]
    with pytest.raises(ValueError, match="lens planner output: no <axes> tag"):
        strategy._resolve_member_lenses("problem", str(tmp_path))
    assert not (tmp_path / ".kapso" / LENS_PLAN_FILENAME).exists()


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
    # The replanner sees the evidence, in full (no truncation), and the
    # axes it named.
    assert "old lens one" in prompt
    assert "champion score: 2.63" in prompt
    assert "family X closed" in prompt
    assert "the incumbent cohort-ranking solution" in prompt
    assert "1. axis one\n2. axis two" in prompt
    saved = json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())
    assert saved["decision"] == "revise" and saved["iteration"] == 4
    assert saved["lenses"] == lenses
    # The revision's statused axes replace the plan's.
    assert saved["axes"] == STATUSED_AXES
    assert [h["decision"] for h in _history(tmp_path)] == ["revise"]
    assert _history(tmp_path)[0]["axes"] == STATUSED_AXES


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
    # A keep carries the axes along, in the plan and in its record.
    assert saved["axes"] == AXES
    assert [h["decision"] for h in _history(tmp_path)] == ["keep"]
    assert _history(tmp_path)[0]["axes"] == AXES
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

    # A complete lens set that statuses no axis is a malformed revision
    # too: the previous plan, axes included, stays in force.
    fake_planner.outputs = [{"output": "<lens_1>new one</lens_1><lens_2>new two</lens_2>"}]
    lenses, _ = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == ["alpha", "beta"]
    assert "every axis" in _history(tmp_path)[-1]["reason"]
    assert json.loads((tmp_path / ".kapso" / LENS_PLAN_FILENAME).read_text())["axes"] == AXES


def test_failed_revision_session_falls_back_loudly(tmp_path, fake_planner):
    strategy = make_stub(tmp_path, iteration=5)
    _write_plan(tmp_path, ["alpha", "beta"], iteration=4)
    fake_planner.outputs = [{"success": False, "error": "rate limited"}]

    lenses, _ = strategy._resolve_member_lenses("problem", str(tmp_path))
    assert lenses == ["alpha", "beta"]
    assert _history(tmp_path)[-1]["decision"] == "failed"
    assert "rate limited" in _history(tmp_path)[-1]["reason"]


def test_the_planner_names_the_axes_and_the_replanner_statuses_them(
    tmp_path, fake_planner
):
    """Anti-freeze contract (user-directed 2026-07-27, the axes the
    planner's since WP4): the first plan derives the axes from the goal,
    and the replanner carries the plan's axes with the axis-status
    contract (ACTIVE / SATURATED-with-evidence / DEFERRED) so freezing an
    axis becomes a dated, evidence-bearing claim — never a silent default
    (the run4 feature-matrix freeze, run_0003..run_0011)."""
    strategy = make_stub(tmp_path)
    fake_planner.outputs = [
        {
            "output": (
                "<axes>\n1. the feature matrix\n2. the ranking mechanism\n</axes>"
                "<lens_1>a</lens_1><lens_2>b</lens_2><sources>- s</sources>"
            )
        }
    ]
    strategy._resolve_member_lenses("problem", str(tmp_path))
    planner_prompt = fake_planner.calls[0]["prompt"]
    assert "Design axes of the solution space" in planner_prompt
    assert "name the design axes" in planner_prompt and "<axes>" in planner_prompt

    strategy2 = make_stub(tmp_path, iteration=2)
    fake_planner.outputs = [{"output": "<keep>still paying</keep>"}]
    strategy2._resolve_member_lenses("problem", str(tmp_path))
    replanner_prompt = fake_planner.calls[-1]["prompt"]
    assert "Design axes of the solution space (the ones you named)" in replanner_prompt
    assert "1. the feature matrix\n2. the ranking mechanism" in replanner_prompt
    assert "Axis-coverage contract" in replanner_prompt
    assert "SATURATED" in replanner_prompt
    assert "DEFERRED" in replanner_prompt
    assert "restates every axis above with its current status" in replanner_prompt


def test_planner_session_mounts_bank_gate_when_serving(tmp_path, fake_planner):
    # Regression (serving v2 §5): the lens planner is the direction-setter —
    # with bank_serving staged its session must carry the bank MCP tools;
    # without it the session stays a plain Read/WebSearch/WebFetch session
    # (asserted by the existing initial-plan test via bank_serving=None).
    strategy = make_stub(tmp_path)
    strategy.bank_serving = {
        "KAPSO_BANKS": '[{"reference": "mine", "head": "abc", "dir": "' + str(tmp_path) + '", "provenance": "measured by you", "scope": "harness"}]',
        "KAPSO_SERVING_PULL_LOG": str(tmp_path / "pull.jsonl"),
        "KAPSO_TASK_HARNESS": "none",
        "KAPSO_PROBE_BUDGET": "1",
    }
    fake_planner.outputs = [
        {
            "output": (
                "<axes>\n1. axis one\n</axes>"
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
