"""RelBench campaign pipeline — pass tasks one by one into Kapso.

You specify only: hour limit per task, goal, and hardware. Everything else
(queue order, per-task bars, tier timeouts, regime gates, recording) is derived
from the repo's own data files.

    PYTHONPATH=src:. python -m benchmarks.relbench.campaign \
        --hours-per-task 10 --goal beat-best --hardware gpu [--dry-run]

Per task: select (queue ∩ hardware ∩ not-done ∩ not regime-gated) → derive the
val target from data/sota.json / data/baselines.json → run the runner with the
time budget threaded into the mode config → on exit regenerate RESULTS.md via
the scorecard and print the goal verdict. Git commits stay with the operator.

The ROI/CPU queues live here as data; the scorecard imports them (single
source — the old scripts/run_relbench_campaign.sh regex parsing is gone).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kapso.core.config import load_config
from kapso.kapso import DEFAULT_CONFIG_PATH
from kapso.learning.harvest import harvest_campaign
from kapso.learning.trajectory_store import TrajectoryStore

NO_STOP_NOTE = """# Budget-bound campaign — do not stop early

This campaign has a wall-clock budget and NO score target. There is no
validation value at which stopping is correct: published bars are context,
not finish lines. Never conclude "good enough", "target reached", or
"diminishing returns" — every remaining minute goes to attempting further
improvement (new features, new mechanisms, better selection), and the only
legitimate stop is the clock. Feedback verdicts must set stop=False while
budget remains.
"""

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
REPO_ROOT = BENCH_DIR.parents[1]

# ROI-sorted flat queue: expected claim value x win probability / compute cost.
# Tier S = tiny DBs; A = medium; B = 8h-tier; C = outlier/saturated (match only).
# rel-mimic/patient-iculengthofstay is excluded (credentialed PhysioNet access).
ROI_QUEUE = [
    # --- Tier S ---
    "rel-event/user-attendance", "rel-f1/driver-circuit-compete",
    "rel-f1/results-position", "rel-f1/qualifying-position",
    "rel-f1/driver-position", "rel-event/event_interest-interested",
    "rel-event/event_interest-not_interested", "rel-event/users-birthyear",
    "rel-event/user-repeat", "rel-event/user-ignore", "rel-f1/driver-dnf",
    # --- Tier A ---
    "rel-salt/sales-group", "rel-salt/sales-payterms", "rel-salt/sales-shipcond",
    "rel-salt/sales-incoterms", "rel-salt/item-incoterms",
    "rel-trial/studies-enrollment", "rel-trial/studies-has_dmc",
    "rel-avito/searchinfo-isuserloggedon", "rel-avito/searchstream-click",
    "rel-trial/study-adverse", "rel-avito/ad-ctr", "rel-trial/study-outcome",
    "rel-trial/condition-sponsor-run", "rel-avito/user-ad-visit",
    "rel-avito/user-clicks", "rel-trial/site-success", "rel-trial/site-sponsor-run",
    "rel-ratebeer/user-count", "rel-arxiv/author-publication",
    "rel-ratebeer/beer-churn", "rel-ratebeer/brewer-dormant",
    "rel-arxiv/paper-citation", "rel-ratebeer/user-churn",
    "rel-ratebeer/beer_ratings-total_score", "rel-ratebeer/user-beer-liked",
    "rel-ratebeer/user-place-liked", "rel-ratebeer/user-beer-favorite",
    "rel-arxiv/author-category", "rel-arxiv/paper-paper-cocitation",
    "rel-trial/eligibilities-adult", "rel-trial/eligibilities-child",
    # --- Tier B ---
    "rel-stack/badges-class", "rel-hm/transactions-price",
    "rel-amazon/review-rating", "rel-hm/user-churn", "rel-hm/item-sales",
    "rel-hm/user-item-purchase", "rel-stack/user-engagement",
    "rel-stack/user-badge", "rel-stack/post-votes", "rel-stack/user-post-comment",
    "rel-stack/post-post-related", "rel-amazon/user-churn",
    "rel-amazon/item-churn", "rel-amazon/user-ltv", "rel-amazon/item-ltv",
    "rel-amazon/user-item-purchase", "rel-amazon/user-item-rate",
    "rel-amazon/user-item-review",
    # --- Tier C ---
    "rel-avito/user-visits", "rel-f1/driver-top3", "rel-salt/item-plant",
    "rel-salt/item-shippoint", "rel-salt/sales-office",
]

# ROI order restricted to datasets safe on a CPU-only ~32 GB box (db.zip sizes
# measured 2026-07-14; excludes rel-stack/rel-ratebeer/rel-amazon/rel-hm).
CPU_SAFE_DATASETS = {"rel-f1", "rel-salt", "rel-event", "rel-arxiv", "rel-avito", "rel-trial"}
CPU_LOCAL_QUEUE = [t for t in ROI_QUEUE if t.split("/")[0] in CPU_SAFE_DATASETS]


def derive_goal(task_id: str, goal_spec: str) -> tuple:
    """Return (target_in_val_units, description). Board units are converted to
    the runner's raw primary-metric units (NMAE -> raw MAE via the stored
    train-std divisors; AUROC/acc/MAP percentages pass through)."""
    baselines = json.loads((DATA_DIR / "baselines.json").read_text())
    divisors = baselines["_meta"]["train_std_divisors_nmae"]

    if goal_spec not in ("beat-best", "beat-kumo"):
        return float(goal_spec), f"explicit {goal_spec}"

    if goal_spec == "beat-kumo":
        ku = baselines["kumorfm_fine_tuned"]
        if task_id in ku["v1_regression_mae"]:
            return float(ku["v1_regression_mae"][task_id]["test"]), "KumoRFM-ft raw MAE"
        if task_id in ku["v1_classification_auroc_pct"]:
            return float(ku["v1_classification_auroc_pct"][task_id]), "KumoRFM-ft AUROC %"
        if task_id in ku["v1_recommendation_map_pct"]:
            return float(ku["v1_recommendation_map_pct"][task_id]), "KumoRFM-ft MAP %"
        raise ValueError(f"{task_id}: no KumoRFM-ft baseline recorded — use --goal beat-best or a number")

    sota = json.loads((DATA_DIR / "sota.json").read_text())
    if task_id not in sota:
        raise ValueError(f"{task_id}: no sota.json entry — pass an explicit --goal value")
    entry = sota[task_id]
    value, metric = float(entry["value"]), entry["metric"]
    if metric == "nmae":
        return value * float(divisors[task_id]), f"best-known {entry.get('method', '?')} (NMAE->raw MAE)"
    return value, f"best-known {entry.get('method', '?')} ({metric})"


def select_tasks(queue: list, hardware: str, work_root: Path,
                 allow_sensitive: bool, explicit: list | None) -> list:
    from benchmarks.relbench.scorecard import PROTOCOL_SENSITIVE_TASKS, ROLLING_VERIFIED

    blocked = PROTOCOL_SENSITIVE_TASKS - ROLLING_VERIFIED
    chosen = []
    pool = explicit if explicit else queue
    for task_id in pool:
        ds, task = task_id.split("/")
        if hardware == "cpu" and ds not in CPU_SAFE_DATASETS:
            print(f"  skip {task_id}: needs GPU-tier hardware")
            continue
        if task_id in blocked and not allow_sensitive:
            print(f"  skip {task_id}: ⚠ rolling harness not yet verified for this task "
                  f"(EVALUATION_PROTOCOL.md); pass --allow-regime-sensitive to override")
            continue
        if (work_root / f"{ds}--{task}" / "final_report.json").exists():
            print(f"  skip {task_id}: already done (final_report.json exists)")
            continue
        chosen.append(task_id)
    return chosen


def _harvest_trajectory(ds: str, task: str, lane, workspace: str, log_path: Path) -> str:
    """Save the finished campaign into the trajectory store (P1.G).

    save_trajectory is the harvest step and the evolve->learn bridge (design
    §3.4): work dir minus the shared cache (model caches are not evidence),
    the campaign log, the workspace .kapso artifacts, and the living
    documents copied to the bundle root per the strict-contract layout. A
    contract violation raises after the verdict/scorecard have landed — loud,
    and the campaign result itself is untouched.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    if not config["learning"]["harvest"]["enabled"]:
        return "harvest-disabled"
    work_dir = REPO_ROOT / "tmp" / "relbench" / f"{ds}--{task}"
    shared_cache = work_dir / "shared_cache"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    trajectory_id = f"{ds}--{task}/{stamp}_{lane or 'local'}"
    kapso_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    living = {"features_history.md": str(shared_cache / "features_history.md")}
    for optional in ("table_information.md", "artifacts.json"):
        if (shared_cache / optional).is_file():
            living[optional] = str(shared_cache / optional)
    return harvest_campaign(
        TrajectoryStore.from_config(config),
        trajectory_id,
        work_dir=str(work_dir),
        campaign_log=str(log_path),
        workspace_dir=str(REPO_ROOT / workspace),
        living_documents=living,
        work_dir_exclude=("shared_cache",),
        kapso_commit=kapso_commit,
    )


def run_one(task_id: str, args) -> dict:
    ds, task = task_id.split("/")
    extra_args = []
    if args.goal == "none":
        # Budget-bound: no early stop — the search uses the full hour budget
        # and keeps its best. The verdict still reports vs best-known. The
        # note rides the knowledge-file channel (the proven stop-rule path)
        # so the feedback generator never volunteers a stop of its own.
        target, target_desc = None, "budget-bound (no early stop)"
        note_path = REPO_ROOT / "tmp" / "campaign_no_stop_note.md"
        note_path.parent.mkdir(exist_ok=True)
        note_path.write_text(NO_STOP_NOTE)
        extra_args = ["--knowledge-file", str(note_path)]
    else:
        target, target_desc = derive_goal(task_id, args.goal)
        extra_args = ["--target-val", str(target)]
    workspace = f"tmp/search_strategy_workspace/{uuid.uuid4()}"
    log_path = REPO_ROOT / "tmp" / f"campaign_{ds}--{task}.log"
    cmd = [
        sys.executable, "-m", "benchmarks.relbench.runner",
        "-s", ds, "-t", task, "-i", str(args.iterations), "-m", args.mode,
        "--workspace", workspace,
        "--time-budget-hours", str(args.hours_per_task),
    ] + extra_args
    shown = "—" if target is None else f"{target:.6g}"
    print(f"\n{'=' * 70}\nCAMPAIGN TASK: {task_id}\n"
          f"  target-val {shown} ({target_desc}) | {args.hours_per_task}h | log {log_path}\n{'=' * 70}")
    if args.dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        return {"task": task_id, "status": "dry-run"}
    # Record what actually runs (surfaces in RESULTS.md HW/Cap columns); the
    # trace-archive URI is added post-hoc by the harvest step as
    # artifact_archive.json in the same directory.
    meta_path = REPO_ROOT / "tmp" / "relbench" / f"{ds}--{task}" / "campaign_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "hardware": args.hardware_desc or args.hardware,
        "cap_hours": args.hours_per_task,
        "lane": args.lane,
        "goal": args.goal,
        "workspace": workspace,
    }, indent=1))
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    report_path = REPO_ROOT / "tmp" / "relbench" / f"{ds}--{task}" / "final_report.json"
    verdict = {"task": task_id, "exit": proc.returncode, "status": "failed"}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        verdict.update({"status": "done", "val": report.get("val_metrics"),
                        "test": report.get("test_metrics"), "target": target})
    subprocess.run(
        [sys.executable, "-m", "benchmarks.relbench.scorecard", "--reference"],
        cwd=REPO_ROOT,
    )
    if verdict["status"] == "done":
        verdict["trajectory_id"] = _harvest_trajectory(ds, task, args.lane, workspace, log_path)
    print(f"  VERDICT: {json.dumps(verdict, default=str)}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours-per-task", type=float, required=True)
    parser.add_argument("--goal", type=str, default="beat-best",
                        help="beat-best | beat-kumo | none (budget-bound, no early stop) | explicit value in raw val units")
    parser.add_argument("--hardware", type=str, choices=["cpu", "gpu"], required=True)
    parser.add_argument("--tasks", type=str, default=None,
                        help="comma-separated task ids; default = ROI queue")
    parser.add_argument("--limit", type=int, default=None, help="max tasks this invocation")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", type=str, default="RELBENCH_GENERIC")
    parser.add_argument("--allow-regime-sensitive", action="store_true")
    parser.add_argument("--lane", type=str, default=None,
                        help="label for this campaign lane (e.g. lane-a); recorded in campaign_meta")
    parser.add_argument("--hardware-desc", type=str, default=None,
                        help="human hardware description recorded in RESULTS.md (e.g. 4xA100)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work_root = REPO_ROOT / "tmp" / "relbench"
    explicit = args.tasks.split(",") if args.tasks else None
    queue = ROI_QUEUE if args.hardware == "gpu" else CPU_LOCAL_QUEUE
    tasks = select_tasks(queue, args.hardware, work_root, args.allow_regime_sensitive, explicit)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"\nCampaign: {len(tasks)} task(s) | {args.hours_per_task}h each | "
          f"goal={args.goal} | hardware={args.hardware}")

    results = [run_one(task_id, args) for task_id in tasks]
    print(f"\n{'=' * 70}\nCAMPAIGN SUMMARY\n{'=' * 70}")
    for r in results:
        print(f"  {r['task']}: {r['status']}")
    if any(r["status"] == "failed" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
