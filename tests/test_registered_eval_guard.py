"""Tests for the registered-evaluation teardown guard (relbench Issue 2).

Contract: runs BEFORE finalize_session's rmtree; when the session output
carries no manifest, it waits for a live grader (bounded) and recovers the
manifest line from the durable run archive the grader wrote outside the
workspace.
"""

import os
import subprocess
import sys
import time

import kapso.execution.search_strategies.generic.strategy as strategy_module
from kapso.core import processes
from kapso.core.processes import process_commands
from kapso.core.roles import SessionRole
from kapso.execution.evaluation_archive_sandbox import write_selection_label
from kapso.execution.evaluation_maintainer.maintainer import MANIFEST_MARKER
from kapso.execution.search_strategies.generic import registered_evaluation
from kapso.execution.search_strategies.generic.strategy import GenericSearch


# Parked until released, then archives its run the way the grader does.
GRADER = """
import pathlib, sys
sys.stdin.read()
run = pathlib.Path(sys.argv[1]) / "run_0001"
(run / "private").mkdir(parents=True)
(run / "private" / "selection.json").write_text('{"session": "candidate"}')
(run / "manifest.txt").write_text(sys.argv[2] + chr(10))
"""


def make_stub(tmp_path, archive_glob):
    s = GenericSearch.__new__(GenericSearch)
    s.harness = None
    s.registered_evaluation_command = (
        "python kapso_evaluation/kapso_eval.py --fidelity full"
    )
    s.registered_evaluation_archive_glob = archive_glob
    s.implementation_role = SessionRole(name="implementation", cli="claude_code", model="m", effort=None, timeout=10)
    s.budget_snapshot = None
    s.budget_snapshot_monotonic = None
    s._last_session_started_ts = time.time() - 3600
    return s


def _write_archive(tmp_path, name, line, session="candidate"):
    runs = tmp_path / "work" / "runs"
    run_dir = runs / name
    (run_dir / "private").mkdir(parents=True)
    write_selection_label(run_dir, session=session, evaluator_id="evaluator")
    (run_dir / "manifest.txt").write_text(line + "\n")
    return runs


def test_recovers_manifest_from_durable_archive(tmp_path):
    line = MANIFEST_MARKER + ' {"fidelity": "full", "score": 0.2334}'
    _write_archive(tmp_path, "run_0001", line)
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))

    recovered = stub._await_registered_evaluation("session died, no manifest", "candidate")

    assert recovered == line


def test_manifest_already_in_output_short_circuits(tmp_path):
    _write_archive(tmp_path, "run_0001", MANIFEST_MARKER + " {}")
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))

    out = MANIFEST_MARKER + ' {"score": 0.5}'
    assert stub._await_registered_evaluation(out, "candidate") is None


def test_stale_archives_are_ignored(tmp_path):
    line = MANIFEST_MARKER + ' {"score": 0.9}'
    runs = _write_archive(tmp_path, "run_0001", line)
    # Archive predates the session: must not be recovered.
    old = time.time() - 7200
    os.utime(runs / "run_0001", (old, old))
    stub = make_stub(tmp_path, str(runs))
    stub._last_session_started_ts = time.time() - 60

    assert stub._await_registered_evaluation("no manifest", "candidate") is None


def test_no_registered_command_is_a_noop(tmp_path):
    stub = make_stub(tmp_path, None)
    stub.registered_evaluation_command = None
    assert stub._await_registered_evaluation("anything", "candidate") is None


def test_a_lane_recovers_only_its_own_sessions_archive(tmp_path, monkeypatch):
    """Parallel lanes share one archive root: a lane whose session died
    without its manifest must not take a sibling lane's score of record."""
    sibling = MANIFEST_MARKER + ' {"fidelity": "full", "score": 0.9}'
    _write_archive(tmp_path, "run_0001", sibling, session="generic_exp_1")
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))
    stub.__dict__.update(
        expansion_lane_env=None, workspace=None, llm=None,
        registered_evaluation_manifest=None, implementation_gates=[],
        gate_failure_policy="skip", bank_serving=None, kg_index_path=None, hosted=None,
        implementation_role=SessionRole(name="implementation", cli="claude_code", model="m", effort=None, timeout=None, web=False),
        implementation_fallback_model=None, env_strip=[], env_defaults={},
        previous_errors=[], recent_error_count=3,
        node_expansion_value=2,
    )

    def died_without_manifest(**kwargs):
        recovered = kwargs["await_registered_evaluation"]("session died")
        return "session died", {}, recovered, None, "provider kill"

    monkeypatch.setattr(strategy_module, "run_implementation", died_without_manifest)

    def recovered_for(branch_name):
        return stub._implement(
            solution="s", problem="p", branch_name=branch_name,
        )[2]

    assert recovered_for("generic_exp_0") is None
    assert recovered_for("generic_exp_1") == sibling


def test_waits_for_a_real_grader_before_teardown(tmp_path, monkeypatch, capsys):
    """Teardown removes the grader's working tree, so a grader still running
    must be found, on every kernel the guard runs on, or its score is lost."""
    line = MANIFEST_MARKER + ' {"fidelity": "full", "score": 0.5}'
    runs = tmp_path / "work" / "runs"
    grader = subprocess.Popen(
        [sys.executable, "-c", GRADER, str(runs), line, "kapso_evaluation/kapso_eval.py"],
        stdin=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while "kapso_eval.py" not in process_commands().get(grader.pid, ""):
        assert time.monotonic() < deadline
    real_sleep = time.sleep

    def release_the_grader(seconds):
        if not grader.stdin.closed:
            grader.stdin.close()
        real_sleep(0.05)

    monkeypatch.setattr(registered_evaluation.time, "sleep", release_the_grader)
    stub = make_stub(tmp_path, str(runs))
    stub.implementation_timeout = 3600

    recovered = stub._await_registered_evaluation("no manifest", "candidate")

    release_the_grader(0)
    grader.wait()
    assert f"still running (pid {grader.pid})" in capsys.readouterr().out
    assert recovered == line


def test_recovers_when_the_kernel_cannot_list_processes(tmp_path, monkeypatch):
    # With no way to see a grader there is none to wait on, and the guard must
    # not fail: it runs before finalize_session, so raising here loses the
    # session's commit. Recovery from the archive still runs.
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path / "no-proc"))
    monkeypatch.setattr(processes.sys, "platform", "freebsd14")
    line = MANIFEST_MARKER + ' {"fidelity": "full", "score": 0.5}'
    _write_archive(tmp_path, "run_0001", line)
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))

    assert stub._await_registered_evaluation("no manifest", "candidate") == line
