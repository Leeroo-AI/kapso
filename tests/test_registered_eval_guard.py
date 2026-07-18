"""Tests for the registered-evaluation teardown guard (relbench Issue 2).

Contract: runs BEFORE finalize_session's rmtree; when the session output
carries no manifest, it waits for a live grader (bounded) and recovers the
manifest line from the durable run archive the grader wrote outside the
workspace.
"""

import os
import time

from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    MANIFEST_MARKER,
)


def make_stub(tmp_path, archive_glob):
    s = GenericSearch.__new__(GenericSearch)
    s.registered_evaluation_command = (
        "python kapso_evaluation/kapso_eval.py --fidelity full"
    )
    s.registered_evaluation_archive_glob = archive_glob
    s.implementation_timeout = 10
    s.budget_snapshot = None
    s.budget_snapshot_monotonic = None
    s._last_session_started_ts = time.time() - 3600
    return s


def _write_archive(tmp_path, name, line):
    runs = tmp_path / "work" / "runs"
    run_dir = runs / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.txt").write_text(line + "\n")
    return runs


def test_recovers_manifest_from_durable_archive(tmp_path):
    line = MANIFEST_MARKER + ' {"fidelity": "full", "score": 0.2334}'
    _write_archive(tmp_path, "run_0001", line)
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))

    recovered = stub._await_registered_evaluation("session died, no manifest")

    assert recovered == line


def test_manifest_already_in_output_short_circuits(tmp_path):
    _write_archive(tmp_path, "run_0001", MANIFEST_MARKER + " {}")
    stub = make_stub(tmp_path, str(tmp_path / "work" / "runs"))

    out = MANIFEST_MARKER + ' {"score": 0.5}'
    assert stub._await_registered_evaluation(out) is None


def test_stale_archives_are_ignored(tmp_path):
    line = MANIFEST_MARKER + ' {"score": 0.9}'
    runs = _write_archive(tmp_path, "run_0001", line)
    # Archive predates the session: must not be recovered.
    old = time.time() - 7200
    os.utime(runs / "run_0001", (old, old))
    stub = make_stub(tmp_path, str(runs))
    stub._last_session_started_ts = time.time() - 60

    assert stub._await_registered_evaluation("no manifest") is None


def test_no_registered_command_is_a_noop(tmp_path):
    stub = make_stub(tmp_path, None)
    stub.registered_evaluation_command = None
    assert stub._await_registered_evaluation("anything") is None
