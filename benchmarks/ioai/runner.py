#!/usr/bin/env python3
"""IOAI 2026 Home Task 3 runner (expert-ioai).

Runs the Kapso agent on the Animal Deduction home task inside a prepared run
root (see data/prepare_animal_deduction.py), then scores the submitted
solution on the held-out split with a pristine harness.

Usage:
    python -m benchmarks.ioai.runner --root /path/to/run_root --hours 2
    python -m benchmarks.ioai.runner --root /path/to/run_root --final-eval-only
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import yaml

from kapso.execution.orchestrator import OrchestratorAgent
from benchmarks.ioai.handler import AnimalDeductionHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

HARNESS_FILES = ["interactor.py", "evaluate.py", "animals_pool.txt",
                 "questions_pool.txt", "dev.csv"]

# Source patterns that indicate the solution reads oracle internals instead of
# playing the game. Matches are reported, not silently fatal — a human reads
# the flagged lines before trusting (or zeroing) the score.
AUDIT_PATTERNS = [
    r"\.gold\b",
    r"gold_animal",
    r"\.history\b",
    r"_llm_yes_no",
    r"__dict__",
    r"getattr\s*\(\s*interactor",
    r"vars\s*\(\s*interactor",
]


def shape_session_timeouts(mode_cfg: dict, total_run_seconds: float) -> dict:
    """Scale per-session deadlines to the run size (posttrain runner pattern:
    fractions of the total budget, floored, with the config caps as ceilings)."""
    knobs = mode_cfg["session_budget"]
    params = mode_cfg["search_strategy"]["params"]
    return {
        "ideation_timeout": int(min(
            params["ideation_timeout"],
            max(
                knobs["ideation_min_seconds"],
                total_run_seconds * knobs["ideation_fraction"],
            ),
        )),
        "implementation_timeout": int(min(
            params["implementation_timeout"],
            max(
                knobs["implementation_min_seconds"],
                total_run_seconds * knobs["implementation_fraction"],
            ),
        )),
    }


def build_runtime_config(mode: str, coding_model: "str | None",
                         task_dir: str, session_timeouts: dict,
                         shared_cache_dir: "str | None" = None,
                         node_expansion: "int | None" = None) -> str:
    """Write the per-run config: shaped session deadlines + model override."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    mode_cfg = config["modes"][mode]
    params = mode_cfg["search_strategy"]["params"]
    params.update(session_timeouts)
    if shared_cache_dir:
        # Persistent task-level cache: artifacts (and their registry offer)
        # carry across campaigns instead of dying with the workspace.
        params["shared_cache_dir"] = os.path.abspath(shared_cache_dir)
    if node_expansion and node_expansion > 1:
        params["node_expansion_value"] = node_expansion
        # One GPU per lane on multi-GPU boxes.
        params["expansion_lane_env"] = [
            {"CUDA_VISIBLE_DEVICES": str(i)} for i in range(node_expansion)
        ]
    if coding_model:
        params["idea_generation_model"] = coding_model
        params["implementation_model"] = coding_model
        for section in ("coding_agent", "feedback_generator"):
            mode_cfg[section]["model"] = coding_model
            mode_cfg[section]["debug_model"] = coding_model
    runtime_dir = os.path.join(task_dir, ".kapso_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_path = os.path.join(runtime_dir, "config.yaml")
    with open(runtime_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return runtime_path


def audit_submission(submission_dir: str) -> list:
    """Static scan of submission sources for oracle-internal access."""
    findings = []
    for dirpath, _, filenames in os.walk(submission_dir):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    for pattern in AUDIT_PATTERNS:
                        if re.search(pattern, line):
                            findings.append(
                                f"{os.path.relpath(path, submission_dir)}:"
                                f"{lineno}: [{pattern}] {line.strip()}"
                            )
    return findings


def parse_eval_output(stdout: str) -> dict:
    """Pull the summary block out of evaluate.py's stdout."""
    metrics = {}
    for key, pattern in (
        ("mean_score", r"Mean score:\s*([\d.]+)"),
        ("solved_rate", r"Solved rate:\s*([\d.]+)%"),
        ("mean_queries", r"Mean queries:\s*([\d.]+)"),
    ):
        match = re.search(pattern, stdout)
        if not match:
            raise ValueError(
                f"evaluate.py output missing '{key}' — output was:\n{stdout}"
            )
        metrics[key] = float(match.group(1))
    metrics["solved_rate"] /= 100.0
    return metrics


def run_final_eval(root: str, timeout_seconds: int, task_python: str) -> dict:
    """Score the submission on dev + held-out test1 with a pristine harness."""
    pristine = os.path.join(root, "private", "dataset_pristine")
    submission_src = os.path.join(root, "task", "submission")
    solution = os.path.join(submission_src, "solution.py")
    if not os.path.isfile(solution):
        return {"error": "no submission/solution.py — nothing to score"}

    eval_dir = os.path.join(root, "final_eval")
    shutil.rmtree(eval_dir, ignore_errors=True)
    os.makedirs(eval_dir)
    for name in HARNESS_FILES:
        shutil.copy2(os.path.join(pristine, name), os.path.join(eval_dir, name))
    shutil.copytree(submission_src, os.path.join(eval_dir, "submission"))

    report = {"audit": audit_submission(submission_src)}
    for split, csv_path in (
        ("dev", os.path.join(eval_dir, "dev.csv")),
        ("test1", os.path.join(root, "private", "test1.csv")),
    ):
        print(f"[final-eval] scoring {split} ...", flush=True)
        proc = subprocess.run(
            [task_python, "evaluate.py", "--csv", csv_path,
             "--solution", "submission/solution.py:MySolution"],
            cwd=eval_dir, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            report[split] = {
                "error": f"evaluate.py exited {proc.returncode}",
                "stderr_tail": proc.stderr[-4000:],
            }
        else:
            report[split] = parse_eval_output(proc.stdout)
        print(f"[final-eval] {split}: {report[split]}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Kapso on IOAI 2026 Home Task 3")
    parser.add_argument("--root", required=True,
                        help="Run root from prepare_animal_deduction.py")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--guard-minutes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", default="IOAI")
    parser.add_argument("--coding-agent", default="claude_code")
    parser.add_argument("--coding-model", default=None)
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--final-eval-only", action="store_true")
    parser.add_argument("--node-expansion", type=int, default=None,
                        help="K parallel implementation lanes per round "
                             "(selector emits top-K; one GPU per lane)")
    parser.add_argument("--shared-cache-dir", default=None,
                        help="Persistent task-level shared cache (artifact "
                             "registry offers carry across campaigns); "
                             "default keeps the per-campaign cache")
    parser.add_argument("--task-python", default=sys.executable,
                        help="Python with the task's ML stack (torch/"
                             "transformers) used to run evaluate.py; kapso "
                             "itself may live in a different env")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    task_dir = os.path.join(root, "task")
    statement_path = os.path.join(task_dir, "dataset", "statement.md")
    if not os.path.isfile(statement_path):
        sys.exit(f"{statement_path} missing — run prepare_animal_deduction.py first")

    with open(CONFIG_PATH) as f:
        mode_cfg = yaml.safe_load(f)["modes"][args.mode]

    if args.final_eval_only:
        report = run_final_eval(root, mode_cfg["final_eval"]["timeout_seconds"],
                                args.task_python)
        results_path = os.path.join(root, "results.json")
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    statement = open(statement_path, encoding="utf-8").read()

    total_run_seconds = args.hours * 3600
    deadline_ts = time.time() + total_run_seconds
    knobs = mode_cfg["session_budget"]
    guard_minutes = (args.guard_minutes if args.guard_minutes is not None
                     else knobs["guard_minutes"])
    budget_minutes = max(5, int(args.hours * 60) - guard_minutes)
    reserve_minutes = min(
        knobs["finalization_reserve_max_minutes"],
        max(
            knobs["finalization_reserve_min_minutes"],
            budget_minutes * knobs["finalization_reserve_fraction"],
        ),
    )
    session_timeouts = shape_session_timeouts(mode_cfg, total_run_seconds)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        print("WARNING: neither ANTHROPIC_API_KEY nor CLAUDE_CODE_OAUTH_TOKEN is set")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — utility-LLM roles will fail")

    config_path = build_runtime_config(args.mode, args.coding_model, task_dir,
                                       session_timeouts,
                                       shared_cache_dir=args.shared_cache_dir,
                                       node_expansion=args.node_expansion)

    print(f"root={root}")
    print(f"budget={budget_minutes} min (guard={guard_minutes} min, "
          f"finalization reserve={reserve_minutes:.0f} min), "
          f"iterations<={args.iterations}")
    print(f"session caps: ideation={session_timeouts['ideation_timeout']}s "
          f"implementation={session_timeouts['implementation_timeout']}s")
    print(f"config={config_path} mode={args.mode} "
          f"coding_model={args.coding_model}")

    handler = AnimalDeductionHandler(
        task_dir=task_dir,
        statement=statement,
        deadline_ts=deadline_ts,
        session_caps=session_timeouts,
        contest_economics=mode_cfg["contest_economics"],
    )

    orchestrator = OrchestratorAgent(
        handler,
        config_path=config_path,
        mode=args.mode,
        coding_agent=args.coding_agent,
        is_kg_active=False,
        workspace_dir=os.path.join(task_dir, "kapso_campaign"),
        goal=statement,
        resume=args.resume,
    )

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        orchestrator.solve(
            experiment_max_iter=args.iterations,
            time_budget_minutes=budget_minutes,
            cost_budget=args.cost_budget,
            finalization_reserve_minutes=reserve_minutes,
        )
    finally:
        print("\n=== campaign done ===")
        print(orchestrator.search_strategy.get_experiment_history())
        print(f"cumulative agent cost: ${orchestrator.get_cumulative_cost():.2f}")

    summary = {
        "root": root,
        "hours": args.hours,
        "submission_present": os.path.isfile(
            os.path.join(task_dir, "submission", "solution.py")),
    }
    if not args.skip_final_eval:
        summary["final"] = run_final_eval(
            root, mode_cfg["final_eval"]["timeout_seconds"], args.task_python)
    results_path = os.path.join(root, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
