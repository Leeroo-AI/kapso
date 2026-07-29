"""Campaign pipeline contracts: queue integrity, gating, goal derivation,
and the wall-clock budget threading. Hermetic — no LLMs, no downloads."""

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.relbench.campaign import (CPU_LOCAL_QUEUE, CPU_SAFE_DATASETS,
                                          ROI_QUEUE, derive_goal, select_tasks)
from benchmarks.relbench.runner import _write_runtime_config
from benchmarks.relbench.scorecard import PROTOCOL_SENSITIVE_TASKS, _campaign_orders

DATA = Path(__file__).parents[1] / "benchmarks" / "relbench" / "data"


def test_queue_integrity():
    from relbench.tasks import get_task_names

    assert len(ROI_QUEUE) == 65 and len(set(ROI_QUEUE)) == 65
    assert "rel-mimic/patient-iculengthofstay" not in ROI_QUEUE
    for task_id in ROI_QUEUE:
        ds, task = task_id.split("/")
        assert task in get_task_names(ds), f"{task_id} not in relbench registry"
    assert len(CPU_LOCAL_QUEUE) == 39
    assert all(t.split("/")[0] in CPU_SAFE_DATASETS for t in CPU_LOCAL_QUEUE)
    # scorecard sources the same lists (the single-home contract)
    roi, cpu = _campaign_orders()
    assert roi == ROI_QUEUE and cpu == set(CPU_LOCAL_QUEUE)


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
    from benchmarks.relbench.scorecard import ROLLING_VERIFIED

    # an unverified rolling task would be blocked: simulate by construction
    assert ROLLING_VERIFIED <= PROTOCOL_SENSITIVE_TASKS


def test_derive_goal_units():
    baselines = json.loads((DATA / "baselines.json").read_text())
    div = baselines["_meta"]["train_std_divisors_nmae"]["rel-f1/driver-position"]
    kumo, desc = derive_goal("rel-f1/driver-position", "beat-kumo")
    assert kumo == pytest.approx(2.731) and "raw MAE" in desc
    best, desc = derive_goal("rel-f1/driver-position", "beat-best")
    assert best == pytest.approx(0.3745 * div, rel=1e-6)  # NMAE -> raw MAE
    auroc, _ = derive_goal("rel-event/user-repeat", "beat-best")
    assert auroc == pytest.approx(83.6)  # percentages pass through
    explicit, _ = derive_goal("rel-f1/driver-position", "2.6")
    assert explicit == 2.6


def test_time_budget_threading(tmp_path):
    path = _write_runtime_config("RELBENCH_GENERIC", str(tmp_path), tmp_path,
                                 time_budget_hours=7.5)
    config = yaml.safe_load(Path(path).read_text())
    assert config["modes"]["RELBENCH_GENERIC"]["budget"]["time_budget_minutes"] == 450.0
    # no override -> config default untouched
    path = _write_runtime_config("RELBENCH_GENERIC", str(tmp_path), tmp_path)
    config = yaml.safe_load(Path(path).read_text())
    assert config["modes"]["RELBENCH_GENERIC"]["budget"]["time_budget_minutes"] == 600
