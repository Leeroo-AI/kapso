"""Campaign pipeline contracts: queue integrity, gating, goal derivation,
and the wall-clock budget threading. Hermetic — no LLMs, no downloads."""

import json
from pathlib import Path

import pytest
import yaml

# The relbench package is installed separately (pip install relbench); it is
# not a declared extra. Without it these skip rather than erroring at import.
pytest.importorskip("relbench")

from benchmarks.relbench.campaign import (CPU_LOCAL_QUEUE, CPU_SAFE_DATASETS,
                                          PROTOCOL_SENSITIVE_TASKS, ROI_QUEUE,
                                          derive_goal, select_tasks)
from benchmarks.relbench.runner import _write_runtime_config


def test_queue_integrity():
    from relbench.tasks import get_task_names

    assert len(ROI_QUEUE) == 65 and len(set(ROI_QUEUE)) == 65
    assert "rel-mimic/patient-iculengthofstay" not in ROI_QUEUE
    for task_id in ROI_QUEUE:
        ds, task = task_id.split("/")
        assert task in get_task_names(ds), f"{task_id} not in relbench registry"
    assert len(CPU_LOCAL_QUEUE) == 39
    assert all(t.split("/")[0] in CPU_SAFE_DATASETS for t in CPU_LOCAL_QUEUE)


def test_selection_gates(tmp_path):
    done = "rel-event/user-attendance"
    (tmp_path / "rel-event--user-attendance").mkdir()
    (tmp_path / "rel-event--user-attendance" / "final_report.json").write_text("{}")
    queue = [done, "rel-f1/driver-position", "rel-hm/user-churn", "rel-f1/driver-dnf",
             "rel-avito/ad-ctr"]
    chosen = select_tasks(queue, "cpu", tmp_path, allow_sensitive=False, explicit=None)
    # done skipped; hm needs GPU; rolling tasks pass — their harness is verified
    assert chosen == ["rel-f1/driver-position", "rel-f1/driver-dnf", "rel-avito/ad-ctr"]
    chosen_gpu = select_tasks(queue, "gpu", tmp_path, allow_sensitive=True, explicit=None)
    assert chosen_gpu == ["rel-f1/driver-position", "rel-hm/user-churn", "rel-f1/driver-dnf",
                          "rel-avito/ad-ctr"]
    assert PROTOCOL_SENSITIVE_TASKS == {"rel-f1/driver-position", "rel-f1/driver-dnf",
                                        "rel-f1/driver-top3"}
    from benchmarks.relbench.campaign import ROLLING_VERIFIED

    # an unverified rolling task would be blocked: simulate by construction
    assert ROLLING_VERIFIED <= PROTOCOL_SENSITIVE_TASKS


def test_derive_goal_explicit_only():
    # Symbolic board targets were retired with data/sota.json +
    # data/baselines.json; only explicit numeric targets remain.
    explicit, desc = derive_goal("rel-f1/driver-position", "2.6")
    assert explicit == 2.6 and "explicit" in desc
    for retired in ("beat-best", "beat-kumo"):
        with pytest.raises(ValueError, match="retired"):
            derive_goal("rel-f1/driver-position", retired)


def test_goal_none_omits_target(tmp_path, capsys):
    import argparse

    from benchmarks.relbench.campaign import run_one

    args = argparse.Namespace(goal="none", iterations=100, mode="RELBENCH_GENERIC",
                              hours_per_task=4.0, dry_run=True)
    verdict = run_one("rel-f1/driver-position", args)
    out = capsys.readouterr().out
    assert verdict["status"] == "dry-run"
    assert "--target-val" not in out and "budget-bound" in out
    assert "--knowledge-file" in out  # no-stop note rides the knowledge channel


def test_time_budget_threading(tmp_path):
    path = _write_runtime_config("RELBENCH_GENERIC", str(tmp_path), tmp_path,
                                 time_budget_hours=7.5)
    config = yaml.safe_load(Path(path).read_text())
    assert config["modes"]["RELBENCH_GENERIC"]["budget"]["time_budget_minutes"] == 450.0
    # no override -> config default untouched
    path = _write_runtime_config("RELBENCH_GENERIC", str(tmp_path), tmp_path)
    config = yaml.safe_load(Path(path).read_text())
    assert config["modes"]["RELBENCH_GENERIC"]["budget"]["time_budget_minutes"] == 600
