#!/usr/bin/env python3
"""Generic local IOAI-task runner (cross-year learning-harvest campaigns).

Runs the Kapso agent on a prepared task root (see data/prepare_*.py), then
scores the submitted solution on a HELD-OUT labeled split with the task's
evaluator. Ground truth is local; there is no external submission.

Usage:
    python -m benchmarks.ioai_tasks.runner --root /path/to/run_root --hours 2
    python -m benchmarks.ioai_tasks.runner --root /path/to/run_root --final-eval-only
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import yaml

from kapso.execution.orchestrator import OrchestratorAgent
from benchmarks.ioai_tasks.handler import LocalTaskHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def shape_session_timeouts(mode_cfg: dict, total_run_seconds: float) -> dict:
    knobs = mode_cfg["session_budget"]
    params = mode_cfg["search_strategy"]["params"]
    return {
        "ideation_timeout": int(min(
            params["ideation_timeout"],
            max(knobs["ideation_min_seconds"],
                total_run_seconds * knobs["ideation_fraction"]),
        )),
        "implementation_timeout": int(min(
            params["implementation_timeout"],
            max(knobs["implementation_min_seconds"],
                total_run_seconds * knobs["implementation_fraction"]),
        )),
    }


def build_runtime_config(mode: str, task_dir: str, session_timeouts: dict,
                         shared_cache_dir: "str | None" = None,
                         node_expansion: "int | None" = None) -> str:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    mode_cfg = config["modes"][mode]
    params = mode_cfg["search_strategy"]["params"]
    params.update(session_timeouts)
    if shared_cache_dir:
        params["shared_cache_dir"] = os.path.abspath(shared_cache_dir)
    if node_expansion and node_expansion > 1:
        params["node_expansion_value"] = node_expansion
        params["expansion_lane_env"] = [
            {"CUDA_VISIBLE_DEVICES": str(i)} for i in range(node_expansion)
        ]
    runtime_dir = os.path.join(task_dir, ".kapso_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_path = os.path.join(runtime_dir, "config.yaml")
    with open(runtime_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return runtime_path


def parse_metric(stdout: str, metric_name: str) -> float:
    """Pull the metric line the task evaluator prints. Missing -> raise."""
    match = re.search(rf"{re.escape(metric_name)}:\s*([\d.]+)", stdout)
    if not match:
        raise ValueError(
            f"evaluator output missing '{metric_name}:' — output was:\n{stdout}"
        )
    return float(match.group(1))


def run_final_eval(root: str, task_meta: dict, timeout_seconds: int,
                   task_python: str) -> dict:
    """Score submission/solution.py on the held-out split."""
    task_dir = os.path.join(root, "task")
    solution = os.path.join(task_dir, "submission", "solution.py")
    if not os.path.isfile(solution):
        return {"error": "no submission/solution.py — nothing to score"}
    command = [task_python] + task_meta["final_eval_command"]
    print(f"[final-eval] {' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=task_dir, capture_output=True,
                          text=True, timeout=timeout_seconds)
    if proc.returncode != 0:
        return {"error": f"evaluator exited {proc.returncode}",
                "stderr_tail": proc.stderr[-4000:]}
    report = {"held_out": parse_metric(proc.stdout, task_meta["metric_name"]),
              "metric": task_meta["metric_name"], "stdout_tail": proc.stdout[-1500:]}
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Kapso on a local IOAI task")
    parser.add_argument("--root", required=True)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--guard-minutes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", default="LOCAL")
    parser.add_argument("--coding-agent", default="claude_code")
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--final-eval-only", action="store_true")
    parser.add_argument("--node-expansion", type=int, default=None)
    parser.add_argument("--shared-cache-dir", default=None)
    parser.add_argument("--task-python", default=sys.executable,
                        help="Python with the task's ML stack for the evaluator")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    task_dir = os.path.join(root, "task")
    statement_path = os.path.join(task_dir, "dataset", "statement.md")
    meta_path = os.path.join(task_dir, "task_meta.json")
    for required in (statement_path, meta_path):
        if not os.path.isfile(required):
            sys.exit(f"{required} missing — run the task's prepare_*.py first")
    with open(meta_path) as f:
        task_meta = json.load(f)

    with open(CONFIG_PATH) as f:
        mode_cfg = yaml.safe_load(f)["modes"][args.mode]

    if args.final_eval_only:
        report = run_final_eval(root, task_meta,
                                mode_cfg["final_eval"]["timeout_seconds"],
                                args.task_python)
        with open(os.path.join(root, "results.json"), "w") as f:
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
        max(knobs["finalization_reserve_min_minutes"],
            budget_minutes * knobs["finalization_reserve_fraction"]),
    )
    session_timeouts = shape_session_timeouts(mode_cfg, total_run_seconds)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        print("WARNING: neither ANTHROPIC_API_KEY nor CLAUDE_CODE_OAUTH_TOKEN is set")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — utility-LLM roles will fail")

    config_path = build_runtime_config(args.mode, task_dir, session_timeouts,
                                       shared_cache_dir=args.shared_cache_dir,
                                       node_expansion=args.node_expansion)

    print(f"root={root} task={task_meta['task_name']}")
    print(f"budget={budget_minutes} min (guard={guard_minutes} min, "
          f"finalization reserve={reserve_minutes:.0f} min)")

    handler = LocalTaskHandler(
        task_dir=task_dir,
        statement=statement,
        deadline_ts=deadline_ts,
        session_caps=session_timeouts,
        eval_spec={
            "self_check_command": task_meta["self_check_command"],
            "metric_name": task_meta["metric_name"],
        },
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

    summary = {"root": root, "hours": args.hours,
               "task": task_meta["task_name"],
               "submission_present": os.path.isfile(
                   os.path.join(task_dir, "submission", "solution.py"))}
    if not args.skip_final_eval:
        summary["final"] = run_final_eval(
            root, task_meta, mode_cfg["final_eval"]["timeout_seconds"],
            args.task_python)
    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
