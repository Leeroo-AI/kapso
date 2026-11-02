"""S5's tooling (docs/plans/redesign/wp9-semantic-suite.md, change 8): the
provider-limit shim passes every call through and answers the
reconstructed limit only for an armed session turn, N times, counting (a
real local subprocess); the enrichment service answers the table to the
key alone and logs every request; the timeline sampler reads the stored
state beside the observed liveness; "truthful within one heartbeat" is
read from a timeline; the request's text names the variable and never a
value; the diagnosis is read for the class and the fix; the run-level
metrics aggregate the injections."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "redesign"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "redesign" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s5 = _load("s5_injections")
service = _load("enrichment_service")
timeline = _load("status_timeline")


def test_the_shim_passes_through_and_answers_an_armed_session_turn_n_times(tmp_path):
    shim = tmp_path / "shim"
    shim.mkdir()
    real = tmp_path / "real-claude"
    real.write_text("#!/usr/bin/env bash\necho real: \"$@\"\n")
    real.chmod(real.stat().st_mode | stat.S_IEXEC)
    (shim / "claude").write_text(s5.SHIM)
    (shim / "claude").chmod(0o755)
    (shim / "real").write_text(str(real))
    (shim / "limit").write_text("2")
    (shim / "count").write_text("0")
    (shim / "exit").write_text("1")
    (shim / "stdout.jsonl").write_text("\n".join(json.dumps(row) for row in s5.RECONSTRUCTED_LIMIT["stdout"]) + "\n")
    (shim / "stderr.txt").write_text(s5.RECONSTRUCTED_LIMIT["stderr"] + "\n")

    def call(*args):
        return subprocess.run([str(shim / "claude"), *args], capture_output=True, text=True)

    assert call("--version").stdout == "real: --version\n", "a version call passes through"
    assert call("-p", "hello", "--output-format", "stream-json").stdout.startswith("real:"), "unarmed, a session turn passes through"
    (shim / "armed").write_text("")
    assert call("auth", "status").stdout.startswith("real:"), "armed, a non-session call still passes through"
    first = call("-p", "hello", "--output-format", "stream-json")
    rows = [json.loads(line) for line in first.stdout.splitlines()]
    assert first.returncode == 1 and rows[-1]["is_error"] and "rate_limit_error" in rows[-1]["result"] and "429" in first.stderr
    second = call("-p", "hello", "--output-format", "stream-json")
    assert second.returncode == 1 and (shim / "count").read_text().strip() == "2" and len((shim / "fired").read_text().splitlines()) == 2
    third = call("-p", "hello", "--output-format", "stream-json")
    assert third.returncode == 0 and third.stdout.startswith("real:"), "after N the CLI runs"


def test_the_enrichment_service_answers_the_key_alone_logs_with_time_and_covers_the_fixture_range(tmp_path):
    log = tmp_path / "requests.log"
    server = service.serve(0, "the-key", log, rows=13000)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/table")
    assert refused.value.code == 401
    request = urllib.request.Request(f"http://127.0.0.1:{port}/table", headers={"Authorization": "Bearer the-key"})
    body = urllib.request.urlopen(request).read().decode()
    # The table covers the hold-out ids the evaluator scores (C010001…C013000), not an arbitrary 3000.
    assert body.startswith("id,segment,risk_score\nC000001,") and body.count("\n") == 13001 and "\nC013000," in body
    request = urllib.request.Request(f"http://127.0.0.1:{port}/enrich?customer_id=C010001", headers={"Authorization": "Bearer the-key"})
    assert json.loads(urllib.request.urlopen(request).read())["id"] == "C010001"
    server.shutdown()
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [entry["authorized"] for entry in entries] == [False, True, True] and entries[0]["status"] == 401
    assert all("ts" in entry for entry in entries), "every request is timestamped, so a fetch can be placed after the reply"
    assert service.row("C000001") == service.row("C000001") and service.row("C000001") != service.row("C000002")


def test_shown_state_reads_the_verdict_kapso_watch_prints():
    assert timeline.shown_state("churn-campaign   DEAD   120s ago") == "DEAD"
    assert timeline.shown_state("state: RUNNING (phase implementation)") == "RUNNING"
    assert timeline.shown_state("nothing here") is None
    # The single-campaign screen: the stored word, then the verdict as a
    # marker (s5-claude-5's killed writer read RUNNING for five samples).
    assert timeline.shown_state("RUNNING ♥ 289s ago   ⚠ DEAD (pid 5554 is gone)      pid 5554\nbudget 9%") == "DEAD"
    assert timeline.shown_state("RUNNING ♥ 200s ago   ⚠ STALLED?      pid 5554") == "STALLED"
    assert timeline.shown_state("RUNNING ♥ 8s ago      pid 6754") == "RUNNING"
    # A detached resume whose child has not written its status yet: watch
    # says so above the earlier run's screen, DEAD marker and all; the
    # product's verdict is the top line (rel-s5-codex's killed writer read
    # DEAD a minute after the resume, and the bar fell on the sampler).
    resuming = "starting again (pid 18047, launched 14s ago; log: /x/s5-killed-writer.log); the status below is the earlier run's\nRUNNING ♥ 90s ago   ⚠ DEAD (pid 16983 is gone)      pid 16983"
    assert timeline.shown_state(resuming) == "STARTING"


def test_the_sampler_records_what_watch_shows_beside_the_stored_status(tmp_path, monkeypatch):
    beat = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(timespec="seconds")
    calls = {"human": "churn   DEAD   200s ago", "json": json.dumps({"state": "running", "phase": "implementation", "pid": 2 ** 22 - 1, "heartbeat_at": beat, "best": None})}

    def fake_run(cmd, capture_output, text):
        out = calls["json"] if "--json" in cmd else calls["human"]
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(timeline.subprocess, "run", fake_run)
    row = timeline.sample(tmp_path / "campaign")
    assert row["shown"] == "DEAD" and row["stored"]["state"] == "running" and row["pid_alive"] is False


def _row(seconds_after, shown, base):
    return {"ts": (base + timedelta(seconds=seconds_after)).isoformat(timespec="seconds"), "shown": shown}


def test_truthful_within_one_heartbeat_is_read_from_what_watch_showed():
    base = datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc)

    def row(seconds, shown, alive, age=1.0, state="running"):
        return {"ts": datetime.fromtimestamp(base.timestamp() + seconds, timezone.utc).isoformat(), "shown": shown, "pid_alive": alive, "heartbeat_age_s": age, "stored": {"state": state}}

    cadence = 60.0
    # The first heartbeat after the injection may be in flight; the samples
    # between one and two cadences are judged, each against the liveness
    # the sampler observed beside it; a killed writer must be shown gone.
    assert s5.truthful_within([row(-60, "RUNNING", True), row(30, "RUNNING", False), row(90, "DEAD", False)], base, cadence, expect_terminal=True) is True
    assert s5.truthful_within([row(-60, "RUNNING", True), row(90, "RUNNING", False), row(200, "DEAD", False)], base, cadence, expect_terminal=True) is False, "shown RUNNING with the pid gone"
    assert s5.truthful_within([row(-60, "RUNNING", True), row(90, "RUNNING", True)], base, cadence, expect_terminal=False) is True, "a living writer shown running"
    assert s5.truthful_within([row(-60, "RUNNING", True), row(90, "WAITING", True)], base, cadence, expect_terminal=False) is True, "a writer paused on the inbox is shown WAITING, truthfully"
    assert s5.truthful_within([row(-60, "RUNNING", True), row(90, "STALLED", True, age=200.0)], base, cadence, expect_terminal=False) is True, "STALLED with a late heartbeat is the truth"
    assert s5.truthful_within([row(-60, "RUNNING", True), row(90, "STALLED", True, age=5.0)], base, cadence, expect_terminal=False) is False, "STALLED with a fresh heartbeat is not"
    assert s5.truthful_within([row(-60, "RUNNING", True), row(100, "DONE", False, state="done")], base, cadence, expect_terminal=False) is True
    assert s5.truthful_within([row(-60, "RUNNING", True), row(200, "FAILED", False, state="failed")], base, cadence, expect_terminal=False) is None, "no sample in the judged window: unobserved"
    assert s5.row_truthful({"shown": "DEAD", "ts": "x"}, cadence) is None, "a sample without the observation decides nothing"


def test_bank_state_reads_a_bare_repo_and_fails_unreadable(tmp_path):

    bare = tmp_path / "bank.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    run = lambda *a: subprocess.run(["git", "-C", str(work), *a], capture_output=True, text=True, check=True)
    run("config", "user.email", "x@x")
    run("config", "user.name", "x")
    (work / "insights").mkdir()
    (work / "insights" / "a-card.md").write_text("# card\n")
    (work / "insights" / "index.md").write_text("# index\n")
    run("add", "-A")
    run("commit", "-q", "-m", "one card")
    run("push", "-q", "origin", "HEAD")
    state = s5.bank_state(bare)
    assert state["readable"] is True and state["cards"] == 1 and state["head"], "the bare bank's one card, index.md excluded"
    assert s5.bank_state(tmp_path / "nope")["readable"] is False


def test_the_seeded_bank_is_the_fixture_as_one_commit_in_a_bare_repo(tmp_path):
    # bank-compare compared nothing in every S5 arm before 2026-10-28: no
    # bank was loaded, so "intact" was vacuous. The seed is the fixture's
    # tree, three cards, read back by the same reader the comparison uses.
    state = s5.seed_bank(ROOT / "tests" / "fixtures" / "bank_seed", tmp_path / "bank-seed.git")
    assert state["readable"] is True and state["cards"] == 3 and state["head"]
    assert (tmp_path / "bank-seed.git" / "HEAD").is_file() and not (tmp_path / "bank-seed.git.src").exists()
    # The engine reads a bank's `main` branch (s5-claude-5 was refused a seed
    # committed on git's default branch): the skeleton is there on main.
    on_main = subprocess.run(["git", "-C", str(tmp_path / "bank-seed.git"), "ls-tree", "--name-only", "main"], capture_output=True, text=True)
    assert on_main.returncode == 0 and {"index.md", "insights", "procedures"} <= set(on_main.stdout.split())
    assert s5.seed_bank(ROOT / "tests" / "fixtures" / "bank_seed", tmp_path / "bank-seed.git") == state, "re-seeding replaces the bank; the tree is the same"
    with pytest.raises(FileNotFoundError):
        s5.seed_bank(tmp_path / "no-such-fixture", tmp_path / "other.git")


def test_a_crash_is_read_from_the_attempts_own_sessions(tmp_path):
    # evaluator-crash read 0 crashed attempts in four arms: the session ran
    # the evaluator, and the crash sat in the session's stream, not in the
    # trace row.
    campaign = tmp_path / "campaign"
    folder = campaign / ".kapso" / "sessions" / "generic_exp_1"
    folder.mkdir(parents=True)
    (folder / "stream.jsonl").write_text(json.dumps({"type": "tool_result", "text": "FileNotFoundError: private/holdout_labels.csv"}) + "\n", encoding="utf-8")
    attempt = {"node": 1, "sessions": [{"label": "generic_exp_1", "module": "implementation"}], "evaluation": {"score": None}}
    assert s5.session_mentions(campaign, attempt, "holdout_labels.csv")
    assert not s5.session_mentions(campaign, attempt, "ENOSPC")
    assert not s5.session_mentions(campaign, {"sessions": [{"label": "missing"}]}, "holdout_labels.csv")
    # The crash counts even when a capable agent then restored the labels and
    # the attempt went on to score (rel-s5-codex: the crash sat in the session,
    # the attempt scored 0.8481, and the plugin named the crash): the injection
    # bit and the plugin had a crash to diagnose, which is what is under test.
    assert s5.evaluator_crashed(campaign, attempt) and s5.crash_occurred(campaign, {"attempts": [attempt]})
    scored = {"node": 1, "sessions": [{"label": "generic_exp_1", "module": "implementation"}], "evaluation": {"score": 0.8481}}
    assert not s5.evaluator_crashed(campaign, scored), "a scored attempt is not a scoreless crash"
    assert s5.crash_occurred(campaign, {"attempts": [scored]}), "but the crash in its session still occurred"


def test_a_finished_rate_limited_campaign_is_continued_not_waited_for():
    # rel-s5-codex, sixth launch: the rate-limited first attempt counted
    # against the two experiments and the campaign stopped at max_iterations
    # before the question; the plugin's fix was a larger allowance, not a
    # wait, and the vocabulary takes it.
    text = ("The first attempt hit 429 rate-limit errors during ideation before producing any implementation turns. "
            "Kapso counted it anyway. Continuing requires increasing the experiment allowance.")
    assert s5.names_class_and_fix(text, "provider-limit") == {"names_class": True, "names_fix": True}
    assert s5.names_class_and_fix("the first attempt hit 429 errors; wait for the window, then resume", "provider-limit") == {"names_class": True, "names_fix": True}


def test_the_request_names_the_variable_and_never_a_value_and_the_diagnosis_names_its_class(tmp_path):
    inbox = tmp_path / "inbox.jsonl"
    assert s5.request_names_no_value(inbox, "secret-value") is None
    inbox.write_text(json.dumps({"event": "requested", "key": "env:DATA_API_KEY", "hit": "the key is missing"}) + "\n")
    assert s5.request_names_no_value(inbox, "secret-value") is True
    inbox.write_text(json.dumps({"event": "requested", "key": "env:DATA_API_KEY", "hit": "found secret-value in a log"}) + "\n")
    assert s5.request_names_no_value(inbox, "secret-value") is False
    assert s5.names_class_and_fix("The campaign's writer is dead, not running; resume it with --resume.", "killed-writer") == {"names_class": True, "names_fix": True}
    assert s5.names_class_and_fix("All good, nothing to see.", "full-disk") == {"names_class": False, "names_fix": False}


def test_the_run_metrics_aggregate_the_injections():
    record = {"steps": [
        {"name": "provider-limit", "kind": "injection", "metrics": {"user_messages": 2, "truthful": True}},
        {"name": "missing-key", "kind": "injection", "metrics": {"user_messages": 2, "truthful": True, "request_names_no_value": True}},
        {"name": "killed-session", "kind": "injection", "metrics": {"user_messages": 1, "truthful": False}},
        {"name": "bank-compare", "kind": "shell", "metrics": {"bank_intact": True}},
    ]}
    assert s5.run_metrics(record) == {"user_messages_max": 2, "truthful_within_heartbeat": 2 / 3, "truthfulness_unobserved": 0, "requests_name_no_value": 1.0, "bank_intact": 1.0}
    # An injection with no sample in its judged window leaves the ratio and is counted, where the bar wants zero.
    record["steps"].append({"name": "full-disk", "kind": "injection", "metrics": {"user_messages": 2, "truthful": None}})
    metrics = s5.run_metrics(record)
    assert metrics["truthful_within_heartbeat"] == 2 / 3 and metrics["truthfulness_unobserved"] == 1
    assert "user_messages_max" not in s5.SUITE["tests"]["S5"]["bars"] and s5.SUITE["tests"]["S5"]["bars"]["truthfulness_unobserved"] == {"max": 0}
    assert s5.STEP_NAMES == s5.SUITE["tests"]["S5"]["required_steps"]["claude_code"]


def test_the_diagnosis_vocabularies_are_disjoint_and_name_the_class(tmp_path):
    # One word must never satisfy both halves of the bar: "you can resume
    # it" named the killed writer's class and fix at once under a looser
    # table; "the evaluator is fine" satisfied both halves of the crash.
    assert s5.names_class_and_fix("you can resume it", "killed-writer") == {"names_class": False, "names_fix": True}
    assert s5.names_class_and_fix("the writer is dead (pid 19524 is gone); run kapso build --output ../s5-killed-writer --resume", "killed-writer") == {"names_class": True, "names_fix": True}
    # rel-s5-codex's plugin named the class as plainly, in other words.
    assert s5.names_class_and_fix("The campaign's writer process died at the start of iteration 1's implementation. `watch` correctly detected that PID 16983 was gone. I resumed the existing campaign with `--resume --detach`.", "killed-writer") == {"names_class": True, "names_fix": True}
    # rel-s5-codex's two phrasings, both correct: the campaign/pid gone, the campaign recovered and finished.
    assert s5.names_class_and_fix("The campaign died at iteration 1's implementation phase: PID 16763 disappeared, leaving a stale 'running' status. I resumed it from its checkpoint.", "killed-writer") == {"names_class": True, "names_fix": True}
    assert s5.names_class_and_fix("First attempt: terminated by SIGKILL (exit -9). The campaign recovered and finished. Second attempt: delivered a three-model ensemble.", "killed-session") == {"names_class": True, "names_fix": True}
    assert s5.names_class_and_fix("the evaluator is fine", "evaluator-crash") == {"names_class": False, "names_fix": False}
    assert s5.names_class_and_fix("the evaluator crashed: holdout_labels.csv was missing; restore it, then resume", "evaluator-crash") == {"names_class": True, "names_fix": True}
    # s5-claude-8's engineer, rightly: the session got SIGKILL, the campaign finished normally, nothing to resume.
    assert s5.names_class_and_fix("The only thing killed was one coding session: it got SIGKILL 1.6 s in. The campaign finished normally; there's nothing to resume or stop.", "killed-session") == {"names_class": True, "names_fix": True}
    for name, class_regex in s5.CLASS_WORDS.items():
        assert not class_regex.search("resume"), name
    # The crash is the evaluator's own failure line, never the file named in a prompt.
    assert not s5.evaluator_crashed(tmp_path, {"evaluation": {"score": None, "output": "I read the brief: the labels live in private/holdout_labels.csv."}})
    assert s5.evaluator_crashed(tmp_path, {"evaluation": {"score": None, "output": "Traceback (most recent call last):\n  FileNotFoundError: [Errno 2] No such file or directory: 'private/holdout_labels.csv'"}})
    assert not s5.evaluator_crashed(tmp_path, {"evaluation": {"score": 0.84, "output": "FileNotFoundError: holdout_labels.csv"}}), "a scored attempt did not crash"


def test_an_archived_run_is_re_read_where_the_archive_allows(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "killed-writer-answers.txt").write_text("The writer is dead: pid 1 is gone; resume it with --resume.\n\n---\n\nResumed.", encoding="utf-8")
    record = {"steps": [
        {"name": "killed-writer", "kind": "injection", "completed": True, "notes": [], "metrics": {"user_messages": 2, "truthful": True, "names_class": True, "names_fix": True, "attempts_kept": True, "attempts_before": 0}},
        {"name": "full-disk", "kind": "injection", "completed": True, "notes": [], "metrics": {"user_messages": 2, "truthful": None, "attempts_kept": True, "attempts_before": 1}},
        {"name": "bank-compare", "kind": "shell", "metrics": {"bank_intact": True}},
    ], "metrics": {}, "notes": []}
    out = s5.rederive(record, tmp_path)
    kept = {step["name"]: step["metrics"] for step in out["steps"]}
    assert kept["killed-writer"]["attempts_kept"] is None, "no attempt had landed: retention not exercised"
    assert kept["full-disk"]["attempts_kept"] is True and kept["full-disk"]["truthful"] is None, "the truthfulness verdict stays as recorded"
    assert kept["killed-writer"]["names_class"] and kept["killed-writer"]["names_fix"]
    assert out["metrics"]["truthfulness_unobserved"] == 1 and out["steps"][0]["notes"][-1].startswith("re-read 20") and " from the archive:" in out["steps"][0]["notes"][-1]


def test_a_score_after_the_reply_may_land_on_the_paused_node_itself():
    # The campaign pauses on node 0 before it scores; after the reply node 0
    # scores. A rule on node identity alone read that as nothing new
    # (s5-claude-8's missing-key class); the comparison is of scores.
    before = {0: None}
    assert s5.scored_after(before, [{"node": 0, "evaluation": {"score": 0.85}}])
    assert s5.scored_after(before, [{"node": 0, "evaluation": {"score": None}}, {"node": 1, "evaluation": {"score": 0.8}}])
    assert not s5.scored_after({0: 0.84}, [{"node": 0, "evaluation": {"score": 0.84}}]), "a score the record held before the reply is not one after it"
    assert not s5.scored_after(before, [{"node": 0, "evaluation": {"score": None}}])


def test_a_resume_is_read_from_the_campaigns_records_not_from_a_running_state_after_the_turn(tmp_path):
    # The second S5 of the release gate read three resumes as none: the plugin
    # waited inside its turn until the resumed campaign had finished, and the
    # driver looked for `running` afterwards. The records say when the campaign
    # was launched again: the status's started_at, or the launch receipt's.
    campaign = tmp_path / "s5-full-disk"
    (campaign / ".kapso").mkdir(parents=True)
    asked = datetime(2026, 10, 30, 8, 43, tzinfo=timezone.utc)
    (campaign / ".kapso" / "status.json").write_text(json.dumps({"state": "done", "started_at": "2026-10-30T07:54:00+00:00"}), encoding="utf-8")
    assert not s5.launched_after(campaign, asked), "the first launch, before the ask, is no resume"
    (campaign / ".kapso" / "status.json").write_text(json.dumps({"state": "done", "started_at": "2026-10-30T08:45:10+00:00"}), encoding="utf-8")
    assert s5.launched_after(campaign, asked), "a campaign relaunched after the ask and already done is a resume"
    (campaign / ".kapso" / "status.json").write_text(json.dumps({"state": "running", "started_at": "2026-10-30T07:54:00+00:00"}), encoding="utf-8")
    (tmp_path / "s5-full-disk.launch.json").write_text(json.dumps({"pid": 1, "log": "x", "argv": [], "started_at": "2026-10-30T08:44:00+00:00"}), encoding="utf-8")
    assert s5.launched_after(campaign, asked), "the launch receipt's time counts before the status is rewritten"
