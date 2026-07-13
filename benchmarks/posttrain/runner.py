#!/usr/bin/env python3
"""
PostTrainBench Runner

Runs the Kapso agent on one PostTrainBench task (post-train a base LLM for a
target benchmark on a single H100 within a wall-clock budget).

Designed to be invoked by the PostTrainBench harness via agents/kapso/solve.sh
inside the task container, but equally runnable by hand in any prepared task
directory (one containing evaluate.py, timer.sh and templates/).

Usage (inside the harness — everything is inferred from $PROMPT and timer.sh):
    expert-posttrain --task-dir "$PWD" --prompt-env PROMPT --coding-model "$AGENT_CONFIG"

Usage (by hand):
    python -m benchmarks.posttrain.runner \
        --task-dir /path/to/task --prompt-file prompt.txt \
        --model Qwen/Qwen3-4B-Base --benchmark-id gsm8k --hours 10
"""

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import yaml

from kapso.execution.orchestrator import OrchestratorAgent
from benchmarks.posttrain.handler import PostTrainBenchHandler, ITERATION_EVAL_LIMITS

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# How much wall-clock to hold back from the orchestrator budget so the final
# consolidation (verify/copy final_model) always has room before the harness's
# hard `timeout` kills the process tree.
DEFAULT_RESERVE_MINUTES = 20


def parse_timer(task_dir: str):
    """Extract the absolute deadline from the harness-generated timer.sh."""
    timer_path = os.path.join(task_dir, "timer.sh")
    if not os.path.isfile(timer_path):
        return None, None
    text = open(timer_path).read()
    hours = re.search(r"^NUM_HOURS=(\d+)", text, re.M)
    created = re.search(r"^CREATION_DATE=(\d+)", text, re.M)
    if hours and created:
        return int(created.group(1)) + int(hours.group(1)) * 3600, int(hours.group(1))
    return None, None


# Substring (of the normalized benchmark name) -> eval task id. Order matters.
BENCHMARK_ALIASES = [
    ("arenahard", "arenahardwriting"),
    ("aime", "aime2025"),
    ("gpqa", "gpqamain"),
    ("gsm8k", "gsm8k"),
    ("healthbench", "healthbench"),
    ("humaneval", "humaneval"),
    ("bfcl", "bfcl"),
    ("functioncalling", "bfcl"),
]


def parse_prompt(prompt: str):
    """Pull the base-model id and benchmark name out of the official prompt.

    The prompt template escapes backticks (LLM \\`{model}\\`), so tolerate
    backslashes around and inside the captured id.
    """
    model = re.search(r"train the small LLM \\?`([^`]+)`", prompt)
    benchmark = re.search(r"to excel at (.+?)\.\s*$", prompt.splitlines()[0] if prompt else "")
    model_id = model.group(1).strip("\\") if model else ""
    return model_id, (benchmark.group(1) if benchmark else "")


def resolve_benchmark_id(name: str) -> str:
    """Map a benchmark name ('GSM8K (Grade School Math 8K)') to a task id."""
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    for alias, task_id in BENCHMARK_ALIASES:
        if alias in normalized:
            return task_id
    return ""


def build_runtime_config(mode: str, coding_model: str, task_dir: str) -> str:
    """Write a config copy with every agent model set to the requested one."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    mode_cfg = config["modes"][mode]
    params = mode_cfg["search_strategy"]["params"]
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


def is_loadable_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def consolidate_final_model(task_dir: str) -> str:
    """Guarantee task_dir/final_model holds a loadable model if any exists.

    The agent is instructed to maintain final_model continuously; this is the
    safety net for crashes/kills between updates.
    """
    final_model = os.path.join(task_dir, "final_model")
    if is_loadable_model_dir(final_model):
        return f"final_model present: {final_model}"

    artifacts = os.path.join(task_dir, "artifacts")
    candidates = []

    score_log = os.path.join(task_dir, "best_score.log")
    if os.path.isfile(score_log):
        for line in reversed(open(score_log).read().splitlines()):
            parts = line.split()
            if len(parts) >= 3:
                candidate = os.path.join(artifacts, parts[-1])
                if is_loadable_model_dir(candidate):
                    candidates.append(candidate)
                    break

    if not candidates and os.path.isdir(artifacts):
        dirs = [
            os.path.join(artifacts, d)
            for d in os.listdir(artifacts)
            if is_loadable_model_dir(os.path.join(artifacts, d))
        ]
        if dirs:
            candidates.append(max(dirs, key=os.path.getmtime))

    if not candidates:
        return "WARNING: no loadable model found anywhere — final_model missing"

    tmp = final_model + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(candidates[0], tmp)
    shutil.rmtree(final_model, ignore_errors=True)
    os.replace(tmp, final_model)
    return f"final_model restored from {candidates[0]}"


def main():
    parser = argparse.ArgumentParser(description="Run Kapso on a PostTrainBench task")
    parser.add_argument("--task-dir", default=os.getcwd())
    parser.add_argument("--prompt", default=None, help="Official task prompt text")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-env", default="PROMPT",
                        help="Env var holding the prompt (harness default: PROMPT)")
    parser.add_argument("--model", default=None, help="Base model HF id (else parsed from prompt)")
    parser.add_argument("--benchmark-id", default=None,
                        help="Task id, e.g. gsm8k (else inferred from prompt)")
    parser.add_argument("--hours", type=float, default=None,
                        help="Budget if timer.sh is absent (default 10)")
    parser.add_argument("--reserve-minutes", type=int, default=DEFAULT_RESERVE_MINUTES)
    parser.add_argument("--iterations", type=int, default=40,
                        help="Iteration ceiling; the time budget is the real governor")
    parser.add_argument("--mode", default="POSTTRAIN")
    parser.add_argument("--coding-agent", default="claude_code")
    parser.add_argument("--coding-model", default=None,
                        help="Model for ideation/implementation/feedback (harness $AGENT_CONFIG)")
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)

    prompt = args.prompt
    if not prompt and args.prompt_file:
        prompt = open(args.prompt_file).read()
    if not prompt and args.prompt_env:
        prompt = os.environ.get(args.prompt_env, "")
    if not prompt:
        sys.exit("No task prompt: pass --prompt/--prompt-file or set $PROMPT")

    parsed_model, parsed_benchmark = parse_prompt(prompt)
    model_id = args.model or parsed_model
    benchmark_id = args.benchmark_id or resolve_benchmark_id(parsed_benchmark)

    deadline_ts, timer_hours = parse_timer(task_dir)
    if deadline_ts is None:
        hours = args.hours if args.hours is not None else 10.0
        deadline_ts = time.time() + hours * 3600
    budget_minutes = max(5, int((deadline_ts - time.time()) / 60) - args.reserve_minutes)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        print("WARNING: neither ANTHROPIC_API_KEY nor CLAUDE_CODE_OAUTH_TOKEN is set")

    config_path = CONFIG_PATH
    if args.coding_model:
        config_path = build_runtime_config(args.mode, args.coding_model, task_dir)

    print(f"task_dir={task_dir}")
    print(f"model={model_id!r} benchmark={parsed_benchmark!r} (id={benchmark_id!r})")
    print(f"deadline in {budget_minutes} min (timer hours={timer_hours}, "
          f"reserve={args.reserve_minutes} min), iterations<={args.iterations}")
    print(f"config={config_path} mode={args.mode} coding_model={args.coding_model}")

    handler = PostTrainBenchHandler(
        task_dir=task_dir,
        official_prompt=prompt,
        model_id=model_id,
        benchmark_name=parsed_benchmark,
        benchmark_id=benchmark_id,
        deadline_ts=deadline_ts,
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
    )

    orchestrator = OrchestratorAgent(
        handler,
        config_path=config_path,
        mode=args.mode,
        coding_agent=args.coding_agent,
        is_kg_active=False,
        workspace_dir=os.path.join(task_dir, "kapso_campaign"),
        goal=prompt,
        resume=args.resume,
    )

    # The harness sends SIGTERM at the deadline (SIGKILL 30s later): convert to
    # SystemExit so the finally-block still consolidates final_model.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        orchestrator.solve(
            experiment_max_iter=args.iterations,
            time_budget_minutes=budget_minutes,
            cost_budget=args.cost_budget,
        )
    finally:
        print("\n=== consolidation ===")
        print(consolidate_final_model(task_dir))
        try:
            print(orchestrator.search_strategy.get_experiment_history())
            print(f"cumulative agent cost: ${orchestrator.get_cumulative_cost():.2f}")
        except Exception as exc:  # never let reporting mask the run outcome
            print(f"(history/cost reporting failed: {exc})")

    summary = {
        "task_dir": task_dir,
        "model": model_id,
        "benchmark_id": benchmark_id,
        "final_model_present": is_loadable_model_dir(os.path.join(task_dir, "final_model")),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
