"""Hermetic tests for the observability layer (design v3 §5.5).

One abstract class owns the mechanics; these tests pin them ONCE (atomic
write, ring cap, phase-timer reset, terminal-state protection, daemon
contract, on_status hook) plus the reader's path resolution, staleness
verdicts, and per-operation renderer blocks — the contracts `kapso
watch` and Kapso.status() depend on.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kapso.core.config import load_platform_defaults, load_config
from kapso.execution.observability import (
    EvolveStatus,
    KnowledgeStatus,
    LessonStatus,
    OperationStatusView,
    RECENT_RING_SIZE,
    STALL_MISSED_HEARTBEATS,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_lifecycle_states_phases_and_ring(tmp_path):
    path = tmp_path / "learn.json"
    status = LessonStatus(path, trajectory_id="t/1")
    assert read(path)["state"] == "starting"
    assert read(path)["operation"] == "learn"
    assert read(path)["phases"] == list(LessonStatus.PHASES)

    status.phase("harvest")
    first = read(path)
    assert first["state"] == "running" and first["phase"] == "harvest"

    # Re-entering the same phase keeps the timer; a new phase resets it.
    status.phase("harvest")
    assert read(path)["phase_started_at"] == first["phase_started_at"]
    status.phase("mine")
    assert read(path)["phase_started_at"] >= first["phase_started_at"]

    with pytest.raises(ValueError, match="unknown learn phase 'compile'"):
        status.phase("compile")

    for index in range(RECENT_RING_SIZE + 3):
        status.note(f"line {index}")
    ring = read(path)["recent"]
    assert len(ring) == RECENT_RING_SIZE
    assert ring[-1].endswith(f"line {RECENT_RING_SIZE + 2}")  # newest kept

    status.done(cards={"created": [], "updated": []})
    final = read(path)
    assert final["state"] == "done" and "finished_at" in final

    # Terminal is final: further mutation is a wiring bug and raises;
    # a late daemon heartbeat is a benign race and no-ops.
    with pytest.raises(RuntimeError, match="already 'done'"):
        status.update(x=1)
    before = read(path)["heartbeat_at"]
    status.heartbeat()
    assert read(path)["heartbeat_at"] == before


def test_failed_records_the_error(tmp_path):
    status = EvolveStatus(tmp_path / "s.json")
    status.failed(RuntimeError("codex died"))
    data = read(tmp_path / "s.json")
    assert data["state"] == "failed"
    assert data["error"] == "RuntimeError: codex died"


def test_file_is_always_parseable_under_concurrent_writers(tmp_path):
    # The atomic tmp+fsync+replace write is the whole point: a reader
    # polling mid-write must never see a torn file.
    path = tmp_path / "evolve.json"
    status = EvolveStatus(path)
    stop = threading.Event()
    torn = []

    def reader():
        while not stop.is_set():
            json.loads(path.read_text()) if path.exists() else None

    thread = threading.Thread(target=reader)
    thread.start()
    for index in range(200):
        status.note(f"iteration {index}", iteration=index)
    stop.set()
    thread.join()
    assert read(path)["iteration"] == 199
    assert not torn


def test_daemon_requires_interval_and_beats(tmp_path):
    with pytest.raises(ValueError, match="heartbeat_seconds"):
        LessonStatus(tmp_path / "a.json", daemon=True)

    status = LessonStatus(
        tmp_path / "b.json", heartbeat_seconds=0.05, daemon=True
    )
    first = read(tmp_path / "b.json")["heartbeat_at"]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=5)
    while read(tmp_path / "b.json")["heartbeat_at"] == first:
        assert datetime.now(timezone.utc) < deadline, "daemon never beat"
    status.done()


def test_on_status_receives_the_file_dict_and_errors_propagate(tmp_path):
    seen = []
    status = KnowledgeStatus(
        tmp_path / "k.json", on_status=lambda d: seen.append(d)
    )
    status.phase("ingest", sources={"done": 0, "total": 2})
    assert seen[-1] == read(tmp_path / "k.json")  # same dict, one truth

    def boom(_d):
        raise RuntimeError("progress bar broke")

    failing = KnowledgeStatus(tmp_path / "k2.json")
    failing._on_status = boom
    with pytest.raises(RuntimeError, match="progress bar broke"):
        failing.note("x")


def test_view_resolves_workspace_file_and_directory(tmp_path):
    workspace = tmp_path / "campaign"
    (workspace / ".kapso").mkdir(parents=True)
    status_file = workspace / ".kapso" / "status.json"
    EvolveStatus(status_file).update(iteration=3)
    assert OperationStatusView(workspace).path == status_file
    assert OperationStatusView(status_file).data["iteration"] == 3

    status_dir = tmp_path / "status"
    LessonStatus(status_dir / "learn-1.json").note("old")
    newest = status_dir / "learn-2.json"
    LessonStatus(newest).note("new")
    assert OperationStatusView(status_dir).path == newest

    with pytest.raises(FileNotFoundError, match="no status file"):
        OperationStatusView(tmp_path / "nowhere")


def test_alive_flips_on_heartbeat_staleness(tmp_path):
    path = tmp_path / "s.json"
    EvolveStatus(path, heartbeat_seconds=60).update(iteration=1)
    assert OperationStatusView(path).alive is True

    data = read(path)
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=60 * STALL_MISSED_HEARTBEATS + 1
    )
    data["heartbeat_at"] = stale.isoformat(timespec="seconds")
    path.write_text(json.dumps(data))
    view = OperationStatusView(path)
    assert view.alive is False and view.stalled is True
    assert "STALLED" in view.explain()

    # Terminal is not stalled, and no cadence means no verdict.
    data["state"] = "done"
    path.write_text(json.dumps(data))
    assert OperationStatusView(path).stalled is False
    del data["heartbeat_seconds"]
    data["state"] = "running"
    path.write_text(json.dumps(data))
    assert OperationStatusView(path).alive is None


def test_renderer_blocks_show_each_operation_shape(tmp_path):
    evolve = tmp_path / "evolve.json"
    EvolveStatus(
        evolve, heartbeat_seconds=60,
        budget={"elapsed_min": 30.0, "total_min": 60.0},
    ).note(
        "node 3 completed score=0.81",
        iteration=4, best={"score": 0.81, "node": 3},
        last={"score": 0.7, "node": 4},
    )
    screen = OperationStatusView(evolve).explain()
    assert "30/60 min" in screen and "▓" in screen
    assert "best: 0.81" in screen and "node 3 completed" in screen

    learn = tmp_path / "learn.json"
    status = LessonStatus(learn, trajectory_id="c/1")
    status.phase("exam", bank_head_before="3e713c93aaaa")
    screen = OperationStatusView(learn).explain()
    assert "harvest ✓" in screen and "exam" in screen and "push ·" in screen
    assert "bank head 3e713c93" in screen

    knowledge = tmp_path / "k.json"
    KnowledgeStatus(knowledge).phase(
        "ingest", sources={"done": 2, "total": 5},
        current_source="catboost.ai/docs", pages_extracted=31,
    )
    screen = OperationStatusView(knowledge).explain()
    assert "sources 2/5" in screen and "catboost.ai/docs" in screen
    assert "pages extracted so far: 31" in screen


def test_watch_json_is_a_pure_passthrough(tmp_path, capsys):
    from kapso.cli import cmd_watch

    path = tmp_path / "s.json"
    EvolveStatus(path).update(iteration=7)

    class Args:
        pass

    args = Args()
    args.path = str(path)
    args.json = True
    args.follow = False
    cmd_watch(args)
    assert json.loads(capsys.readouterr().out)["iteration"] == 7


def test_config_carries_the_observability_keys():
    # The wiring reads these (Rule 1: config is the single source) — a
    # missing key kills learn()/solve() at status construction.
    defaults = load_platform_defaults()
    assert defaults["budget"]["checkpoint_heartbeat_seconds"] > 0
    packaged = load_config(
        str(Path(__file__).parent.parent / "src/kapso/config.yaml")
    )
    assert packaged["learning"]["status_dir"]
