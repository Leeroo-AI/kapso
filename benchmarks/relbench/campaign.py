"""RelBench campaign pipeline — pass tasks one by one into Kapso.

You specify only: hour limit per task, goal, and hardware. Everything else
(queue order, per-task bars, tier timeouts, regime gates, recording) is derived
from the repo's own data files.

    PYTHONPATH=src:. python -m benchmarks.relbench.campaign \
        --hours-per-task 10 --goal beat-best --hardware gpu [--dry-run]

Per task: select (queue ∩ hardware ∩ not-done ∩ not regime-gated) → run the
runner with the time budget threaded into the mode config → print the goal
verdict. Git commits stay with the operator. Symbolic goal targets and the
scorecard were retired with the campaign's reference data (claims/ and the
sota/baseline snapshots live in GCS + git history).

The ROI/CPU queues live here as data (single
source — the old scripts/run_relbench_campaign.sh regex parsing is gone).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

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


PROTOCOL_SENSITIVE_TASKS = {
    "rel-f1/driver-position",
    "rel-f1/driver-dnf",
    "rel-f1/driver-top3",
}

# Rolling harness verified per task (campaign gate opens only for these):
# driver-position — full acceptance 2026-07-29: reference B-rolling through the
#   real grader+cascade scored test MAE 2.6516 (band 2.653±0.015; bar 2.731);
# driver-dnf / driver-top3 — cascade invariants verified same day: every test
#   snapshot leak-clean, snapshot-relabeled train == official labels exactly.
ROLLING_VERIFIED = set(PROTOCOL_SENSITIVE_TASKS)


def derive_goal(task_id: str, goal_spec: str) -> tuple:
    """Return (target_in_val_units, description).

    Only explicit numeric targets remain: the symbolic beat-best/beat-kumo
    specs were retired with data/sota.json + data/baselines.json (the board
    snapshots live in git history; run archives in GCS)."""
    if goal_spec in ("beat-best", "beat-kumo"):
        raise ValueError(
            f"{task_id}: goal spec {goal_spec!r} was retired with the "
            "sota/baseline snapshots — pass an explicit value in raw val units"
        )
    return float(goal_spec), f"explicit {goal_spec}"


def select_tasks(queue: list, hardware: str, work_root: Path,
                 allow_sensitive: bool, explicit: list | None) -> list:
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
    print(f"  VERDICT: {json.dumps(verdict, default=str)}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours-per-task", type=float, required=True)
    parser.add_argument("--goal", type=str, default="none",
                        help="none (budget-bound, no early stop) | explicit value in raw val units")
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
