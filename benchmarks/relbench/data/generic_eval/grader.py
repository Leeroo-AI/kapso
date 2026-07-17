"""RelBench provided grader — the immutable scoring logic for generic search.

Runs the candidate's `main.py` as an isolated child subprocess, scores the
resulting validation predictions with the OFFICIAL relbench metrics against
the sanitized cache (test labels are physically absent from it), archives
full-fidelity runs for final selection, and prints the machine-readable
manifest line the search parses as the score of record.

Self-contained by design: imports only stdlib + numpy + relbench, never
candidate modules — candidate code runs in the child process only.

CLI (the registered entrypoint delegates here verbatim):
    python kapso_evaluation/grader.py --fidelity fast|full [--fraction F] [--seed S]

Fidelity semantics for THIS suite (see README.md): the evaluation cost lives
in the candidate's training build, not in scoring items. `fast` runs
`main.py --debug` (the contract's cheap-build mode); `full` runs `main.py`.
Both score the complete official validation split; the item set never
shrinks. `fraction`/`seed` are echoed into the manifest for grant validation.

Environment (exported by the RelBench handler at startup):
    RELBENCH_DATASET / RELBENCH_TASK      task identity
    RELBENCH_PRIMARY_METRIC              e.g. "mae", "link_prediction_map"
    RELBENCH_WORK_DIR                    archive root (runs/run_NNNN/)
    RELBENCH_FULL_TIMEOUT / RELBENCH_DEBUG_TIMEOUT   child caps, seconds
    RELBENCH_CACHE_DIR                   sanitized cache (set process-wide)
    KAPSO_SHARED_CACHE_DIR               cross-experiment cache for candidates
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MANIFEST_MARKER = "KAPSO_EVAL_MANIFEST"


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        print(f"[grader] missing required environment variable: {name}")
        sys.exit(2)
    return value


def _repo_root() -> Path:
    """The candidate repo root: the parent of this suite's directory."""
    return Path(__file__).resolve().parent.parent


def run_candidate(fidelity: str, run_data_dir: Path) -> None:
    root = _repo_root()
    main_py = root / "main.py"
    if not main_py.exists():
        print("[grader] main.py not found at repo root — the solution must be main.py")
        sys.exit(3)

    timeout = int(
        _env("RELBENCH_DEBUG_TIMEOUT") if fidelity == "fast" else _env("RELBENCH_FULL_TIMEOUT")
    )
    cmd = [sys.executable, "main.py"] + (["--debug"] if fidelity == "fast" else [])
    env = os.environ.copy()
    env["KAPSO_RUN_DATA_DIR"] = str(run_data_dir)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[grader] running candidate: {' '.join(cmd)} (timeout {timeout}s)")
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    tail: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    try:
        proc.wait(timeout=max(1, int(timeout - (time.time() - start))))
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), 9)
        print(f"[grader] candidate exceeded {timeout}s and was killed")
        sys.exit(4)
    if proc.returncode != 0:
        print(f"[grader] candidate exited non-zero ({proc.returncode})")
        sys.exit(5)


def load_and_score(run_data_dir: Path):
    from relbench.base import RecommendationTask, TaskType
    from relbench.tasks import get_task

    dataset_name = _env("RELBENCH_DATASET")
    task_name = _env("RELBENCH_TASK")
    primary = _env("RELBENCH_PRIMARY_METRIC")

    task = get_task(dataset_name, task_name, download=False)
    val_table = task.get_table("val", mask_input_cols=False)
    n_val = len(val_table)
    n_test = len(task.get_table("test"))

    val_path = run_data_dir / "val_predictions.npy"
    test_path = run_data_dir / "test_predictions.npy"
    for p, n in ((val_path, n_val), (test_path, n_test)):
        if not p.exists():
            print(f"[grader] contract violation: {p.name} was not written")
            sys.exit(6)

    val_pred = np.load(val_path, allow_pickle=False)
    test_pred = np.load(test_path, allow_pickle=False)

    if isinstance(task, RecommendationTask):
        expected = (n_val, task.eval_k)
        if val_pred.shape != expected or not np.issubdtype(val_pred.dtype, np.integer):
            print(f"[grader] contract violation: val predictions must be int {expected}, got {val_pred.shape} {val_pred.dtype}")
            sys.exit(6)
    else:
        val_arr = np.asarray(val_pred, dtype=np.float64)
        if val_arr.ndim == 2 and val_arr.shape[1] == 1 and task.task_type != TaskType.MULTICLASS_CLASSIFICATION:
            val_pred = val_arr[:, 0]
        if len(val_pred) != n_val or not np.all(np.isfinite(np.asarray(val_pred, dtype=np.float64))):
            print(f"[grader] contract violation: val predictions wrong length or non-finite")
            sys.exit(6)

    val_metrics = {k: float(v) for k, v in task.evaluate(val_pred, val_table).items()}
    if primary not in val_metrics:
        print(f"[grader] primary metric {primary} missing from {sorted(val_metrics)}")
        sys.exit(7)
    return val_metrics, val_pred, test_pred, n_val


def archive_full_run(run_data_dir: Path, val_metrics: dict) -> str:
    """Persist a full-fidelity run for final selection. Fast runs are never
    archived: their debug-mode predictions are pipeline checks, not results."""
    runs_root = Path(_env("RELBENCH_WORK_DIR")) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(runs_root.glob("run_*"))
    next_index = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    run_dir = runs_root / f"run_{next_index:04d}"
    private = run_dir / "private"
    private.mkdir(parents=True)
    for name in ("val_predictions.npy", "test_predictions.npy"):
        shutil.copy2(run_data_dir / name, run_dir / name)
    (private / "metrics.json").write_text(
        json.dumps({"val": val_metrics, "test": {}}, indent=2)
    )
    # Snapshot the candidate's code for the final audit and claims package
    # (same filters as the tree-path archiver: no git internals, no
    # evaluation suite, no oversized files).
    root = _repo_root()
    code_dir = run_dir / "code"
    for f in root.rglob("*"):
        if not f.is_file() or ".git" in f.parts or "kapso_evaluation" in f.parts or "__pycache__" in f.parts:
            continue
        if f.stat().st_size > 2_000_000:
            continue
        dest = code_dir / f.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    return run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", required=True, choices=["fast", "full"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    primary = _env("RELBENCH_PRIMARY_METRIC")
    run_data_dir = Path(tempfile.mkdtemp(prefix="relbench_eval_"))
    try:
        run_candidate(args.fidelity, run_data_dir)
        val_metrics, _val_pred, _test_pred, n_val = load_and_score(run_data_dir)

        archived = ""
        if args.fidelity == "full":
            archived = archive_full_run(run_data_dir, val_metrics)

        print(f"[grader] OFFICIAL VALIDATION METRICS: {json.dumps(val_metrics)}")
        if archived:
            print(f"[grader] archived as {archived}")
        manifest = {
            "fidelity": args.fidelity,
            "fraction": args.fraction,
            "seed": args.seed,
            "items": n_val,
            "total_items": n_val,
            "score": val_metrics[primary],
        }
        print(f"{MANIFEST_MARKER} {json.dumps(manifest)}")
    finally:
        shutil.rmtree(run_data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
