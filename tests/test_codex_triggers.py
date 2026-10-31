"""The Codex trigger runner's reading of a session (docs/plans/redesign/
wp7-plugin.md, change 9): the two shapes a skill load takes, pinned by
the captured fixture; a changed shape raises instead of reading nothing;
the cases read as `claude plugin eval` reads them; the Skill graders
applied to the loaded skills as the host applies them to the tool's
input."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "codex_trigger_events.jsonl"
spec = importlib.util.spec_from_file_location("codex_triggers", ROOT / "scripts" / "redesign" / "codex_triggers.py")
codex_triggers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codex_triggers)


def _fixture_lines():
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    stream = [line for line in lines if json.loads(line).get("type") in ("thread.started", "item.completed")]
    rollout = [line for line in lines if "payload" in json.loads(line)]
    return stream, rollout


def test_the_fixture_pins_both_shapes_of_a_skill_load():
    stream, rollout = _fixture_lines()
    assert codex_triggers.loaded_skills(stream, []) == ["kapso-build"]        # the command that read the file
    assert codex_triggers.loaded_skills([], rollout) == ["kapso-build"]       # the explicit invocation's message
    assert codex_triggers.loaded_skills(stream, rollout) == ["kapso-build"]   # once each
    assert codex_triggers.thread_id(stream) == json.loads(stream[0])["thread_id"]


def test_a_changed_shape_raises_instead_of_reading_nothing():
    stream, rollout = _fixture_lines()
    item = json.loads(stream[1])
    del item["item"]["command"]
    with pytest.raises(KeyError):
        codex_triggers.loaded_skills([json.dumps(item)], [])
    message = json.loads(rollout[0])
    del message["payload"]["role"]
    with pytest.raises(KeyError):
        codex_triggers.loaded_skills([], [json.dumps(message)])
    with pytest.raises(ValueError, match="no thread.started"):
        codex_triggers.thread_id(stream[1:])


def test_the_cases_are_read_as_the_host_reads_them_and_graded_on_the_loads():
    found = {case["name"]: case for case in codex_triggers.cases(tag="main")}
    assert "build-c01" in found and "holdout-01" not in found
    confusion = found["build-c01"]
    assert [g["name"] for g in confusion["graders"]] == ["not-kapso-build", "skill-fired"]
    assert codex_triggers.grade(confusion["graders"], ["kapso"]) == [
        {"name": "not-kapso-build", "count": 0, "passed": True, "read_after": False}, {"name": "skill-fired", "count": 1, "passed": True},
    ]
    # Codex reads skills as references: the skill under test read after
    # the expected one is a pass, said as such; read first, the session
    # routed to it, and the refusal fails; alone, both fail.
    later = codex_triggers.grade(confusion["graders"], ["kapso", "kapso-build"])
    assert [g["passed"] for g in later] == [True, True] and later[0]["read_after"] is True
    assert [g["passed"] for g in codex_triggers.grade(confusion["graders"], ["kapso-build", "kapso"])] == [False, True]
    assert [g["passed"] for g in codex_triggers.grade(confusion["graders"], ["kapso-build"])] == [False, False]
    none = found["build-n01"]
    assert [g["passed"] for g in codex_triggers.grade(none["graders"], [])] == [True]
    assert [g["passed"] for g in codex_triggers.grade(none["graders"], ["kapso-review"])] == [False]
    # A run that never started is no evidence for a refusal.
    assert [g["passed"] for g in codex_triggers.grade(none["graders"], [], started=False)] == [False]
    assert [case["name"] for case in codex_triggers.cases(tag="main", pattern="setup-e*")] == ["setup-e01", "setup-e02"]


def test_a_load_is_a_read_that_succeeded():
    """A command that names a skill file without reading it, or that
    failed, loads nothing."""
    def command(text, exit_code=0):
        return json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": f"/bin/bash -lc '{text}'", "exit_code": exit_code, "status": "completed"}})
    stream = [
        command("echo .agents/skills/kapso-build/SKILL.md"),
        command("cat .agents/skills/kapso-review/SKILL.md", exit_code=1),
        command("sed -n 1,80p .agents/skills/kapso-setup/SKILL.md"),
    ]
    assert codex_triggers.loaded_skills(stream, []) == ["kapso-setup"]


def test_a_sandbox_that_never_started_is_the_machines_failure_not_a_miss(tmp_path):
    # s2-codex-5 on GCP: 123 of 138 failed runs said "bwrap: loopback: Failed
    # RTM_NEWADDR: Operation not permitted" — the session announced the skill
    # and could not read it. Such a run is classified from its own words and
    # is never a pass, never a miss: the bar is not established on it.
    stream = tmp_path / "streams" / "build-c01-1.jsonl"
    stream.parent.mkdir()
    stream.write_text('{"type":"thread.started","thread_id":"t1"}\n', encoding="utf-8")
    stream.with_suffix(".err").write_text("ERROR: bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n", encoding="utf-8")
    assert codex_triggers.sandbox_failure(stream) == "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
    clean = tmp_path / "streams" / "build-c01-2.jsonl"
    clean.write_text('{"type":"thread.started","thread_id":"t2"}\n', encoding="utf-8")
    assert codex_triggers.sandbox_failure(clean) is None
    results = [
        {"case": "build-c01", "run": 1, "started": True, "loaded": [], "graders": [{"name": "kapso-build", "count": 0, "passed": False}],
         "environment": "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted", "passed": False},
        {"case": "build-c01", "run": 2, "started": True, "loaded": ["kapso-build"], "graders": [{"name": "kapso-build", "count": 1, "passed": True}],
         "environment": None, "passed": True},
    ]
    text = codex_triggers.report(results, [])
    assert text.splitlines()[2].startswith("| build-c01 | 2 | 1/2 |") and text.splitlines()[2].endswith("| 0 | 1 |")
    assert "1 run(s) failed in the sandbox before any skill could be read" in text


def test_a_run_whose_sandbox_failed_is_tried_again_before_it_counts(monkeypatch):
    # The release gate's S2 on Codex: 151 of 178 runs failed in the sandbox,
    # intermittently, at two sessions at once — the machine's failure; a run is
    # tried again up to the suite's count, and the last try is the run.
    outcomes = iter(["bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted", "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted", None])
    monkeypatch.setattr(codex_triggers.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(codex_triggers, "_run_case_once", lambda *args: {"environment": next(outcomes), "passed": False})
    assert codex_triggers.run_case({"name": "build-c01"}, 1, None, "m", "medium", 600, "read-only") == {"environment": None, "passed": False, "tries": 3}
    always = iter(["bwrap: x"] * 10)
    monkeypatch.setattr(codex_triggers, "_run_case_once", lambda *args: {"environment": next(always), "passed": False})
    tries = int(codex_triggers.SUITE["triggers"]["sandbox_tries"])
    assert codex_triggers.run_case({"name": "build-c01"}, 1, None, "m", "medium", 600, "read-only") == {"environment": "bwrap: x", "passed": False, "tries": tries}


def test_a_run_the_deadline_passed_is_not_reached_and_each_run_lands_in_the_ledger_as_it_ends(tmp_path, monkeypatch, capsys):
    # rel-s2-codex: one session at a time ran past the step's two hours, the
    # shell's timeout ended the driver, and no result was written. The probe
    # keeps the clock itself: a run not started by the deadline is recorded as
    # not reached (never a pass, never a miss), each run's row lands as it
    # ends, and the result and the report are written all the same.
    case = {"name": "build-c01", "tags": [], "prompt": "p", "graders": [{"name": "build", "input_match": "build", "min": 1, "max": None}]}
    monkeypatch.setattr(codex_triggers, "cases", lambda tag, pattern=None: [case])
    monkeypatch.setattr(codex_triggers, "probe_sandbox", lambda sandbox, out: None)
    clock = iter([0.0, 0.0, 100.0])   # the deadline's reading, the first run's check, the second's
    monkeypatch.setattr(codex_triggers.time, "monotonic", lambda: next(clock, 100.0))
    monkeypatch.setattr(codex_triggers, "run_case", lambda case, run, *rest: {
        "case": case["name"], "run": run, "exit": 0, "seconds": 1.0, "started": True, "loaded": ["kapso-build"],
        "graders": codex_triggers.grade(case["graders"], ["kapso-build"]), "environment": None, "passed": True})
    out = tmp_path / "probe"
    assert codex_triggers.main(["--out", str(out), "--runs", "2", "-j", "1", "--deadline-seconds", "60"]) == 1
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["run"], row["passed"], row["started"]) for row in rows] == [(1, True, True), (2, False, False)]
    assert rows[1]["not_reached"] == "the probe's deadline of 60 s passed before this run started"
    document = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [row.get("not_reached") is not None for row in document["results"]] == [False, True]
    assert "1 run(s) not reached before the probe's deadline" in capsys.readouterr().out
    # Graded again, a run not reached stays not established: no stream to read, never a pass.
    assert codex_triggers.regrade(out) == 1
    graded = json.loads((out / "results.json").read_text(encoding="utf-8"))["results"]
    assert [row["passed"] for row in graded] == [True, False] and graded[1]["environment"] is None


def test_the_probe_decides_the_sandbox_from_several_sessions_and_falls_back_when_bubblewrap_fails(tmp_path, monkeypatch):
    # One probe passed on the release gate's machine and 113 of 139 runs then
    # failed in bubblewrap's loopback setup, one session at a time with three
    # tries each. The probe runs the suite's count of sessions; any failure
    # sends the suite to the fallback sandbox (which must itself start), the
    # decision written beside the results and said in the record.
    outcomes = iter([None, "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted", None, None])
    monkeypatch.setattr(codex_triggers, "_probe_once", lambda sandbox, project: next(outcomes))
    assert codex_triggers.probe_sandbox("read-only", tmp_path) == codex_triggers.SUITE["triggers"]["sandbox_fallback"]
    decision = json.loads((tmp_path / "sandbox.json").read_text(encoding="utf-8"))
    assert decision["requested"] == "read-only" and decision["used"] == "danger-full-access" and len(decision["probes"]) == 3
    monkeypatch.setattr(codex_triggers, "_probe_once", lambda sandbox, project: None)
    assert codex_triggers.probe_sandbox("read-only", tmp_path) == "read-only"
    always = iter(["bwrap: x"] * 10)
    monkeypatch.setattr(codex_triggers, "_probe_once", lambda sandbox, project: next(always))
    with pytest.raises(RuntimeError, match="no sandbox starts"):
        codex_triggers.probe_sandbox("read-only", tmp_path)


def test_the_runs_switch_to_the_fallback_after_a_streak_of_sandbox_failures_and_the_failed_runs_are_run_again(tmp_path, monkeypatch, capsys):
    # The fourth launch of the gate's S2 on Codex: three probes started under
    # read-only, then 15 of 18 runs failed in bubblewrap. A streak of the
    # suite's length sends the rest of the suite to the fallback, the runs the
    # sandbox failed are run again under it at the end, and the record says so.
    cases = [{"name": f"build-c0{index}", "tags": [], "prompt": "p", "graders": [{"name": "build", "input_match": "build", "min": 1, "max": None}]} for index in range(1, 6)]
    monkeypatch.setattr(codex_triggers, "cases", lambda tag, pattern=None: cases)
    monkeypatch.setattr(codex_triggers, "_probe_once", lambda sandbox, project: None)
    seen = []

    def fake_run(case, run, out, model, effort, timeout, sandbox):
        seen.append((case["name"], sandbox))
        failed = sandbox == "read-only" and len([s for s in seen if s[1] == "read-only"]) <= 3
        loaded = [] if failed else ["kapso-build"]
        return {"case": case["name"], "run": run, "exit": 1 if failed else 0, "seconds": 1.0, "started": not failed, "loaded": loaded,
                "graders": codex_triggers.grade(case["graders"], loaded, not failed), "environment": "bwrap: loopback: Failed RTM_NEWADDR" if failed else None, "passed": not failed}

    monkeypatch.setattr(codex_triggers, "run_case", fake_run)
    out = tmp_path / "probe"
    assert codex_triggers.main(["--out", str(out), "--runs", "1", "-j", "1"]) == 0
    assert [s[1] for s in seen] == ["read-only"] * 3 + ["danger-full-access"] * 5, "the streak switches the rest, and the three failed runs are run again under the fallback"
    document = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert document["sandbox"] == "danger-full-access" and document["sandbox_requested"] == "read-only" and document["switched_after_runs"] == 3
    assert all(row["passed"] and row["sandbox"] == "danger-full-access" for row in document["results"])
    assert [row.get("first_sandbox_failure") for row in document["results"]][:3] == ["bwrap: loopback: Failed RTM_NEWADDR"] * 3
    decision = json.loads((out / "sandbox.json").read_text(encoding="utf-8"))
    assert decision["used"] == "danger-full-access" and decision["switched_after_runs"] == 3
    assert "the rest run under danger-full-access" in capsys.readouterr().out
