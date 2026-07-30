"""Rolling-harness contracts: sensitivity detection matches the ⚠ set, and the
grader's per-tick assembly reconstructs the official row order. Hermetic."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.relbench.scorecard import PROTOCOL_SENSITIVE_TASKS

GRADER = Path(__file__).parents[1] / "benchmarks" / "relbench" / "data" / "generic_eval" / "grader.py"


def test_rolling_detection_matches_protocol_set():
    """The handler's rolling trigger (num_eval_timestamps > 1) must identify
    exactly the tasks the protocol doc gates — no drift between the two."""
    from relbench.tasks import get_task_names, task_registry

    rolling = set()
    for ds in ("rel-amazon", "rel-avito", "rel-event", "rel-f1", "rel-hm",
               "rel-stack", "rel-trial", "rel-arxiv", "rel-salt", "rel-ratebeer"):
        for name in get_task_names(ds):
            cls = task_registry[ds][name][0]
            if getattr(cls, "num_eval_timestamps", 1) > 1:
                rolling.add(f"{ds}/{name}")
    assert rolling == PROTOCOL_SENSITIVE_TASKS


def test_rolling_contract_note_in_context():
    """Regression: the rolling contract must exist and be wired into
    build_problem_context (a constants-block edit once deleted it and only
    live rolling-task runs would have caught the NameError)."""
    import inspect

    from benchmarks.relbench.context import ROLLING_CONTRACT_NOTE, build_problem_context

    assert "ONCE PER TICK" in ROLLING_CONTRACT_NOTE
    assert "ROLLING_CONTRACT_NOTE" in inspect.getsource(build_problem_context)


def test_grader_rolling_assembly(tmp_path):
    """Two fake ticks, interleaved official positions: the assembled vector must
    place each tick's predictions at its indices.npy positions."""
    sys.dont_write_bytecode = True  # never leave __pycache__ inside the eval suite
    spec = importlib.util.spec_from_file_location("grader", GRADER)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)

    root = tmp_path / "rolling"
    ticks = {"val_000_2005": [0, 2, 4], "val_001_2005b": [1, 3],
             "test_000_2010": [1, 0]}
    for name, idx in ticks.items():
        task_dir = root / name / "rel-x" / "tasks" / "t"
        task_dir.mkdir(parents=True)
        pd.DataFrame({"date": pd.to_datetime(["2005-01-01"] * len(idx)),
                      "entity": range(len(idx))}).to_parquet(task_dir / "test.parquet")
        np.save(task_dir / "indices.npy", np.array(idx))

    # stub candidate: predicts 100*i + row for tick ordinal i (from env marker)
    candidate = tmp_path / "repo"
    (candidate / "kapso_evaluation").mkdir(parents=True)
    (candidate / "main.py").write_text(
        "import os, numpy as np, pandas as pd\n"
        "cache = os.environ['RELBENCH_CACHE_DIR']\n"
        "df = pd.read_parquet(f'{cache}/rel-x/tasks/t/test.parquet')\n"
        "tick = float(os.path.basename(os.path.dirname(cache)).split('_')[1] if False else 0)\n"
        "idx = np.load(f'{cache}/rel-x/tasks/t/indices.npy')\n"
        "np.save(os.path.join(os.environ['KAPSO_RUN_DATA_DIR'], 'test_predictions.npy'),"
        " idx.astype(float) * 10.0)\n"
    )

    out = tmp_path / "out"
    out.mkdir()
    env = dict(os.environ, RELBENCH_FULL_TIMEOUT="120", RELBENCH_DEBUG_TIMEOUT="60")
    grader.os.environ.update(env)
    grader._repo_root = lambda: candidate
    grader.run_candidate_rolling("full", out, root)

    val = np.load(out / "val_predictions.npy")
    test = np.load(out / "test_predictions.npy")
    # each official position i must hold 10*i (stub predicts 10*official_index)
    assert np.array_equal(val, np.arange(5) * 10.0)
    assert np.array_equal(test, np.arange(2) * 10.0)


def test_sandbox_rolling_rejects_non_rolling_task():
    """Fail-loud contract: --rolling on a span-0 task must error."""
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.relbench.sandbox",
         "--dataset", "rel-event", "--task", "user-attendance",
         "--dest", "/tmp/nonexistent_rolling_test", "--rolling"],
        capture_output=True, text=True,
        cwd=Path(__file__).parents[1],
    )
    assert proc.returncode != 0
    assert "not a rolling task" in (proc.stdout + proc.stderr)
