"""`kapso trace` (wp6-observability.md, change 11): the campaign's record
read into one view and rendered. Goldens over three fixture campaigns
rebuilt from archived proof runs by `tests/fixtures/trace/build_fixture.py`
(WP3's trading build: a void with its findings, a timeout, served cards
and a delegate; WP3's churn build: a delivered candidate with no harness;
WP2's first machine: a delivered candidate with a valid verification); a
campaign still running, one with no status yet, and one whose record
predates the checkpoint render partial; a directory that is not a
campaign is refused by name; `--branch` narrows to one attempt in full;
`--json` round-trips through `TraceView(data)`; the generator reproduces
the fixtures from the archives when they are present."""
import filecmp
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.cli.trace import cmd_trace
from kapso.execution.coding_agents.session_record import EXIT_ERROR, EXIT_SUCCESS, EXIT_TIMEOUT, run_entry
from kapso.execution.coding_agents.transcripts import CLAUDE_HOST, CODEX_HOST, digest_claude
from kapso.execution.inbox import write_launch_record
from kapso.execution.observability import BuildStatus
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.trace import COMPLETE, PARTIAL, SCHEMA_VERSION, TraceView, failed_runs, read_trace
from kapso.handles import _ERROR_HEADLINE, Solution
from kapso.harness.staging import SESSION_RECORD_FILE, SESSIONS_DIR

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "trace"
# The fixtures' generator, a script beside them, loaded as a module.
_GENERATOR = importlib.util.spec_from_file_location("build_fixture", FIXTURES / "build_fixture.py")
build_fixture = importlib.util.module_from_spec(_GENERATOR)
_GENERATOR.loader.exec_module(build_fixture)
ARCHIVES = {
    "trading": "wp3/campaign1.tgz",
    "churn": "wp3/campaign2.tgz",
    "trading-delivered": "wp2-claude/campaign.tgz",
    # The WP4 proof run: the facts the older archives cannot carry.
    "churn-unjudged": "wp4/campaign-c12.tgz",
    "churn-codex": "wp4/campaign-devbox-a.tgz",
    "trading-void": "wp4/campaign-devbox-b.tgz",
}
RUNS = Path(__file__).resolve().parents[1] / "scripts" / "redesign" / "runs"
PLACEHOLDER = "<campaign>"
HARNESS = {"name": "trading", "version": "0.1.0", "hash": "83b6c0b60d6a" + "0" * 52, "source": "/h/trading"}
MINE = {"reference": "mine", "kind": "bank", "resolved": "/me/.kapso/bank.git", "head": "a9ddad3d302d" + "0" * 28,
        "provenance": "measured by you", "scope": "harness"}


def rendered(campaign: Path) -> str:
    return read_trace(campaign).explain().replace(str(campaign.resolve()), PLACEHOLDER) + "\n"


# ------------------------------------------------------------------ goldens

@pytest.mark.parametrize("name", list(ARCHIVES))
def test_the_text_is_the_golden(name):
    assert rendered(FIXTURES / name) == (FIXTURES / name / "trace.txt").read_text(encoding="utf-8")


def test_the_trading_build_lays_out_the_void_the_timeout_and_the_cards():
    view = read_trace(FIXTURES / "trading")
    data = view.data
    assert data["state"] == COMPLETE and data["branch"] is None
    header = data["header"]
    assert header["harness"]["name"] == "trading" and header["experience"] == ["mine at a9ddad3d302d, measured by you"]
    # Three runs whose spend no figure covers: the two killed at their
    # deadlines, and the judge's first run on the sixty-second floor, whose
    # retry was a fresh host session; the two invocations that ended
    # without a result and were resumed are covered by the resumed runs'
    # cumulative figures.
    assert header["cost"] == {"usd": 13.1687975, "coverage": "all", "unpriced_runs": 3}
    assert (header["stopped_reason"], header["stop_detail"], header["experiments"], header["rejected"]) == ("budget_exhausted", "time_budget", 2, 1)
    assert header["delivered"] == {"branch": None, "commit": None} and header["status"]["state"] == "done"
    first, second = view.attempts
    assert [session["label"] for session in first["sessions"]] == [
        "ideation/iter1/claude_code", "generic_exp_0", "verification/generic_exp_0", "feedback/generic_exp_0",
    ]
    # The implementation ran as three host invocations (the first ended
    # without a result event while its delegate still ran, and the session
    # was resumed twice), the verification as two: one run entry per
    # invocation, the cost the delta of the host's cumulative figure, the
    # first resumed run carrying the whole figure so far.
    assert [len(session["runs"]) for session in first["sessions"]] == [1, 3, 2, 1]
    implementation = first["sessions"][1]
    assert [run["exit"] for run in implementation["runs"]] == [EXIT_ERROR, EXIT_SUCCESS, EXIT_SUCCESS]
    assert [run["cost_usd"] for run in implementation["runs"]] == [None, 3.92427, 0.229677]
    assert [run["duration_seconds"] is None for run in implementation["runs"]] == [True, False, False]
    assert len({run["cli_session_id"] for run in implementation["runs"]}) == 1
    [delegate] = implementation["runs"][0]["delegates"]
    assert delegate["name"] == "lookahead-auditor" and delegate["report"].startswith("## Outcome")
    assert implementation["runs"][0]["hooks"][0]["decision"] == "context"
    assert first["verification"]["verdict"] == "void" and first["evaluation"]["valid"] is False
    assert first["judge"]["stop"] is False and first["judge"]["feedback"].startswith("# Iteration 1: VOID")
    assert len(first["cards"]["read"]) == 5 and "horizon-grid-locates-live-band" in first["cards"]["read"]
    assert len(first["cards"]["load_bearing"]) == 11
    assert second["evaluation"]["score"] is None and second["verification"]["verdict"] == "unverified"
    [lesson] = data["lessons"]
    assert lesson["served"] == 15 and list(lesson["read_by"]) == ["ideation/iter1"]
    assert lesson["load_bearing_by"] == {"0": first["cards"]["load_bearing"]}
    [void] = data["voided"]
    # The harness's void verdict is a rejection in the design's two words (voided
    # is the integrity check's), so a reader tells the two apart by the reason.
    assert void["node"] == 0 and void["reason"].startswith("rejected: the verification found ") and len(void["evidence"]) == 2
    # The header counts by the same two words (the engine's `rejected_candidates`
    # counts an integrity void as rejected, and a reader followed it).
    assert (data["header"]["voided"], data["header"]["rejected"]) == (0, 1)
    assert [(failure["label"], failure["exit"], failure["every_run"]) for failure in data["failures"]] == [
        ("feedback/generic_exp_1", EXIT_ERROR, False), ("generic_exp_0", EXIT_ERROR, False),
        ("generic_exp_1", EXIT_TIMEOUT, False), ("verification/generic_exp_0", EXIT_ERROR, False),
        ("verification/generic_exp_1", EXIT_TIMEOUT, False),
    ]
    assert data["delivered"] is None
    [setup] = data["sessions"]
    assert setup["label"] == "harness_setup" and setup["module"] == "setup"


def test_the_churn_build_delivers_without_a_harness():
    view = read_trace(FIXTURES / "churn")
    assert view.data["header"]["harness"] is None and view.data["voided"] == []
    [attempt] = view.attempts
    assert attempt["verification"] is None and attempt["evaluation"]["score"] == 0.846
    assert view.delivered["branch"] == "generic_exp_0" and view.delivered["verification"] is None
    [lesson] = view.data["lessons"]
    assert lesson["served"] == 0 and lesson["read_by"] == {} and lesson["load_bearing_by"] == {}


def test_the_delivered_build_carries_its_verification_and_files():
    view = read_trace(FIXTURES / "trading-delivered")
    assert view.data["header"]["stopped_reason"] == "goal_achieved" and view.data["header"]["experience"] == []
    assert view.delivered["verification"]["verdict"] == "valid"
    assert "strategy.py" in view.delivered["files_changed"]
    text = view.explain()
    assert "integrity       provided; intact" in text
    assert "hooks           4 fired in its sessions, none denied" in text
    # Its three sessions each began with an invocation that ended without
    # a result and was resumed: in its failures, as the record has them.
    assert "failures        3 in its sessions" in text


# ----------------------------------------------------------- the narrowing

def test_narrowing_to_a_branch_prints_its_evidence_in_full():
    view = read_trace(FIXTURES / "trading")
    narrowed = view.narrow("generic_exp_0")
    assert narrowed.data["branch"] == "generic_exp_0" and len(narrowed.attempts) == 1
    assert narrowed.data["lessons"] == view.data["lessons"]
    text = narrowed.explain()
    assert "showing         the attempt on generic_exp_0, in full" in text
    assert "# Solution Steps" in text                         # the idea, whole
    assert "## 1. Verdict and what it does and does not mean" in text   # the judge's words, whole
    assert "delegate lookahead-auditor (completed)" in text and "## Outcome" in text
    assert "hook SessionStart SessionStart:startup -> context" in text
    assert "closing message" in text
    assert "- the 2024 bars are read by the evaluation alone" in text    # a finding with its evidence
    with pytest.raises(ValueError, match="no attempt on branch 'nowhere'; the attempts are: generic_exp_0, generic_exp_1"):
        view.narrow("nowhere")


def test_json_round_trips_through_the_view():
    view = read_trace(FIXTURES / "trading")
    again = TraceView(json.loads(json.dumps(view.data)))
    assert again.explain() == view.explain() and again.data == view.data
    with pytest.raises(ValueError, match="not a trace of schema 2"):
        TraceView({"schema_version": 0})


# -------------------------------------------------------- partial campaigns

def running_campaign(tmp_path: Path, *, with_record: bool = False) -> Path:
    """A campaign mid-run: a status, a launch record, a checkpoint with one
    node, one implementation session with a run that errored and a second
    in flight."""
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    status = BuildStatus(campaign / ".kapso" / "status.json", heartbeat_seconds=60, harness=HARNESS, experience=[MINE])
    status.phase("implementation", budget={"elapsed_min": 12.0, "total_min": 40.0})
    write_launch_record(campaign, {
        "brief": {"objective": "Raise the Sharpe", "budget": {"minutes": None}}, "minutes": 40.0,
        "harness": HARNESS, "experience": [MINE], "output_path": str(campaign),
    })
    node = SearchNode(
        node_id=0, branch_name="generic_exp_0", parent_branch_name="main", solution="# Idea\nTrend following",
        score=1.2, feedback="Good but not there", evaluation_valid=True, evaluation_provenance="provided",
        cost_usd=2.5, duration_seconds=600.0,
    ).to_dict()
    RunCheckpointStore(str(campaign)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Raise the Sharpe", config_fingerprint="f", status="running",
        completed_iterations=1, cumulative_cost=2.5, current_feedback=None,
        strategy_state={"node_history": [node]}, harness=HARNESS, experience=[MINE],
    ))
    record_dir = campaign / SESSIONS_DIR / "generic_exp_1"
    record_dir.mkdir(parents=True)
    failed = run_entry(
        digest_claude([]), model="claude-opus-5", cli_session_id="s1", exit=EXIT_ERROR, error="Quota exceeded",
        files_changed=[], started="2026-10-25T10:00:00+00:00", ended="2026-10-25T10:00:30+00:00", duration_seconds=30.0,
    )
    (record_dir / SESSION_RECORD_FILE).write_text(json.dumps({
        "module": "implementation", "host": CODEX_HOST, "harness": HARNESS, "session_dir": str(campaign), "pages": [],
        "skills": [], "mcp_servers": [], "agents": [], "hooks": [], "prepared": [], "stream": "", "runs": [failed],
    }))
    return campaign


def test_a_running_campaign_renders_partial_from_the_status_and_the_checkpoint(tmp_path):
    campaign = running_campaign(tmp_path)
    view = read_trace(campaign)
    data = view.data
    assert data["state"] == PARTIAL
    header = data["header"]
    assert header["objective"] == "Raise the Sharpe" and header["harness"] == HARNESS
    assert header["experience"] == ["mine at a9ddad3d302d, measured by you"]
    assert header["cost"] == {"usd": 2.5, "coverage": "claude_code", "unpriced_runs": 0}   # a Codex session ran: its cost is not in the figure
    assert header["experiments"] == 1 and header["delivered"] == {"branch": None, "commit": None}
    assert header["status"] == {"state": "running", "phase": "implementation", "error": None}
    [attempt] = view.attempts
    assert attempt["branch"] == "generic_exp_0" and attempt["sessions"] == []
    # The session of the node in flight belongs to no attempt yet.
    [in_flight] = data["sessions"]
    assert in_flight["label"] == "generic_exp_1" and data["failures"][0]["every_run"] is True
    text = view.explain()
    assert text.startswith(f"trace of {campaign.resolve()} — partial: the campaign is running (phase implementation)")
    assert "cost            $2.500 (Claude Code sessions only; Codex reports no cost)" in text
    assert "every run failed: implementation on codex" in text
    assert "delivered       none succeeded" in text


def test_a_campaign_with_no_status_yet_says_so(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    view = read_trace(campaign)
    assert view.data["state"] == PARTIAL and view.data["header"]["status"] is None
    text = view.explain()
    assert "— partial: no status yet" in text and "objective       not recorded" in text
    assert "ATTEMPTS\n  none recorded yet" in text and "LESSONS\n  nothing served" in text


def test_a_record_older_than_the_checkpoint_renders_partial(tmp_path):
    campaign = tmp_path / "churn"
    shutil.copytree(FIXTURES / "churn", campaign)
    assert read_trace(campaign).data["state"] == COMPLETE
    store = RunCheckpointStore(str(campaign))
    checkpoint = store.load()
    checkpoint.completed_iterations = 2
    store.save(checkpoint)
    view = read_trace(campaign)
    assert view.data["state"] == PARTIAL
    assert "— partial: the record predates the checkpoint's last write" in view.explain()


def test_a_failed_campaign_names_its_error(tmp_path):
    campaign = running_campaign(tmp_path)
    data = json.loads((campaign / ".kapso" / "status.json").read_text())
    data["state"], data["error"] = "failed", "HarnessError: the harness changed since launch"
    (campaign / ".kapso" / "status.json").write_text(json.dumps(data))
    assert "— partial: the campaign failed: HarnessError: the harness changed since launch" in read_trace(campaign).explain()


def test_a_directory_that_is_not_a_campaign_is_refused_by_name(tmp_path, capsys):
    with pytest.raises(FileNotFoundError, match="is not a campaign directory"):
        read_trace(tmp_path)
    with pytest.raises(SystemExit) as stop:
        cmd_trace(SimpleNamespace(campaign=str(tmp_path), branch=None, json=False))
    assert stop.value.code == 1
    assert capsys.readouterr().err == f"Error: {tmp_path} is not a campaign directory (no .kapso/ inside)\n"


# ------------------------------------------------------------ the command

def test_the_command_prints_the_text_the_json_and_refuses_an_unknown_branch(capsys):
    fixture = FIXTURES / "churn"
    cmd_trace(SimpleNamespace(campaign=str(fixture), branch=None, json=False))
    assert capsys.readouterr().out == read_trace(fixture).explain() + "\n"
    cmd_trace(SimpleNamespace(campaign=str(fixture), branch=None, json=True))
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == SCHEMA_VERSION and data == read_trace(fixture).data
    cmd_trace(SimpleNamespace(campaign=str(fixture), branch="generic_exp_0", json=True))
    assert json.loads(capsys.readouterr().out)["branch"] == "generic_exp_0"
    with pytest.raises(SystemExit) as stop:
        cmd_trace(SimpleNamespace(campaign=str(fixture), branch="generic_exp_9", json=False))
    assert stop.value.code == 1
    assert capsys.readouterr().err == "Error: no attempt on branch generic_exp_9; the attempts are: generic_exp_0\n"


# ------------------------------------------------- the record's failed sessions

def test_the_failed_runs_reach_the_record_and_its_summary_line():
    failures = failed_runs(FIXTURES / "trading")
    assert [(failure["label"], failure["module"], failure["host"], failure["exit"]) for failure in failures] == [
        ("feedback/generic_exp_1", "feedback", CLAUDE_HOST, EXIT_ERROR),
        ("generic_exp_0", "implementation", CLAUDE_HOST, EXIT_ERROR),
        ("generic_exp_1", "implementation", CLAUDE_HOST, EXIT_TIMEOUT),
        ("verification/generic_exp_0", "verification", CLAUDE_HOST, EXIT_ERROR),
        ("verification/generic_exp_1", "verification", CLAUDE_HOST, EXIT_TIMEOUT),
    ]
    assert failures[2]["error"] == "Claude Code CLI exceeded its 182.63159913599975s deadline and was terminated"
    solution = Solution.open(FIXTURES / "trading")
    assert solution.session_failures == failures
    lines = solution.explain().splitlines()
    [line] = [line for line in lines if line.startswith("Sessions failed:")]
    # One clause per kind of session, with its count and the last error's
    # first line, clipped for the summary's width.
    def headline(error: str) -> str:
        return error if len(error) <= _ERROR_HEADLINE else error[: _ERROR_HEADLINE - 1] + "…"

    assert line == (
        f"Sessions failed: feedback on claude_code ×1 ({headline(failures[0]['error'])}); "
        f"implementation on claude_code ×2 ({headline(failures[2]['error'])}); "
        f"verification on claude_code ×2 ({headline(failures[4]['error'])})"
    )
    [experiments] = [index for index, text in enumerate(lines) if text.startswith("Experiments:")]
    assert lines.index(line) == experiments + 1
    assert "Sessions failed:" not in Solution.open(FIXTURES / "churn").explain()


# --------------------------------------------------------------- the generator

@pytest.mark.parametrize("name", list(ARCHIVES))
def test_the_generator_reproduces_the_fixture_from_its_archive(name, tmp_path):
    archive = RUNS / ARCHIVES[name]
    if not archive.is_file():
        pytest.skip(f"the archive {archive} is not on this machine")
    rebuilt = tmp_path / name
    build_fixture.build(archive, rebuilt)
    build_fixture.render(rebuilt)
    committed = sorted(path.relative_to(FIXTURES / name) for path in (FIXTURES / name).rglob("*") if path.is_file())
    fresh = sorted(path.relative_to(rebuilt) for path in rebuilt.rglob("*") if path.is_file())
    assert fresh == committed
    differing = [str(path) for path in committed if not filecmp.cmp(FIXTURES / name / path, rebuilt / path, shallow=False)]
    assert differing == []


# ------------------------------------------------- the WP4 proof run's facts

def test_an_unjudged_attempt_with_its_commit_and_the_engines_failed_session():
    """The twelve-minute churn build on the fresh machine (wp4-config.md,
    change 13): the judge cut at the budget's deadline leaves the attempt
    unjudged with the reason and its commit, nothing delivered, the stop
    `finishing_cut`, the reserve on the budget line, and the engine's own
    session (on a Codex key over its quota) under the other sessions."""
    view = read_trace(FIXTURES / "churn-unjudged")
    header = view.data["header"]
    assert header["stopped_reason"] == "budget_exhausted" and header["stop_detail"] == "finishing_cut"
    assert header["budget"] == {"elapsed_min": 11.5, "total_min": 12.0, "reserve_min": 1.5}
    assert header["budget_source"] == "the launch's over the brief's 45.0 min"
    attempt, = view.attempts
    assert attempt["commit"].startswith("60cb08e3876d")
    assert attempt["unjudged"] == "the judge's session ended at the budget's deadline"
    assert attempt["evaluation"]["score"] == 0.8454 and attempt["judge"] == {"stop": False, "valid": True, "feedback": ""}
    assert view.delivered is None
    engine, = view.data["sessions"]
    assert engine["label"] == "engine/repo_memory/1" and engine["module"] == "engine" and engine["host"] == "codex"
    assert engine["runs"][0]["exit"] == "error" and "Quota exceeded" in engine["runs"][0]["error"]
    text = view.explain()
    assert "judge         unjudged: the judge's session ended at the budget's deadline" in text
    assert "budget          11.5 of 12.0 min, 1.5 min reserved for finishing (the launch's over the brief's 45.0 min)" in text
    assert (
        "stopped         budget_exhausted (finishing_cut): a finishing session was cut: "
        "node 0 unjudged: the judge's session ended at the budget's deadline; 1 experiment"
    ) in text


def test_every_session_on_the_host_the_launch_named():
    """The churn build with `--coding-agent codex` on the packaged config:
    every session of the attempt and the engine's own run on Codex with
    the host's model, the second round refused by admission, the attempt
    delivered."""
    view = read_trace(FIXTURES / "churn-codex")
    attempt, = view.attempts
    sessions = attempt["sessions"] + view.data["sessions"]
    assert {session["host"] for session in sessions} == {"codex"}
    assert {run["model"] for session in sessions for run in session["runs"]} == {"gpt-5.6-sol"}
    assert [session["label"] for session in attempt["sessions"]] == ["ideation/iter1/codex_gpt-5.6-sol", "generic_exp_0", "feedback/generic_exp_0"]
    engine, = view.data["sessions"]
    assert engine["label"] == "engine/repo_memory/1" and engine["runs"][0]["exit"] == "success"
    assert view.data["header"]["stop_detail"] == "attempt_would_not_fit"
    assert view.delivered["branch"] == "generic_exp_0" and attempt["commit"] == view.delivered["commit"]


def test_a_voided_attempt_on_codex_with_the_judge_on_claude():
    """The trading build with the harness, `host: codex` and `judge: {cli:
    claude_code}`: the setup, ideation, implementation and verification on
    Codex, the judge on Claude Code; the verifier's void on a file added
    under the protected directory, the attempt rejected and not delivered,
    the judge's delegate stopped."""
    view = read_trace(FIXTURES / "trading-void")
    attempt, = view.attempts
    hosts = {session["label"]: session["host"] for session in attempt["sessions"] + view.data["sessions"]}
    assert hosts == {
        "ideation/iter1/codex_gpt-5.6-sol": "codex", "generic_exp_0": "codex", "verification/generic_exp_0": "codex",
        "feedback/generic_exp_0": "claude_code", "engine/repo_memory/1": "codex", "harness_setup": "codex",
    }
    assert attempt["verification"]["verdict"] == "void" and len(attempt["verification"]["findings"]) == 1
    assert attempt["evaluation"]["score"] is None and attempt["evaluation"]["valid"] is False
    assert attempt["judge"]["valid"] is False and attempt["unjudged"] == ""
    assert view.data["header"]["rejected"] == 1 and view.delivered is None
    judge_run, = next(session for session in attempt["sessions"] if session["label"] == "feedback/generic_exp_0")["runs"]
    assert judge_run["model"] == "claude-opus-5" and judge_run["delegates"][0]["status"] == "stopped"
    assert view.data["voided"][0]["branch"] == "generic_exp_0"
