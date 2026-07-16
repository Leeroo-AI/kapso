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


def parse_timer(task_dir: str):
    """Extract the absolute deadline from the harness-generated timer.sh."""
    timer_path = os.path.join(task_dir, "timer.sh")
    if not os.path.isfile(timer_path):
        return None, None
    with open(timer_path) as timer_file:
        text = timer_file.read()
    hours = re.search(r"^NUM_HOURS=(\d+)", text, re.M)
    created = re.search(r"^CREATION_DATE=(\d+)", text, re.M)
    if hours and created:
        return int(created.group(1)) + int(hours.group(1)) * 3600, int(hours.group(1))
    return None, None


def shape_session_timeouts(mode_cfg: dict, total_run_seconds: float) -> dict:
    """Scale per-session deadlines to the run size.

    A fixed cap dominates short runs (live run #6: the 30-minute ideation
    cap consumed 77% of a 39-minute budget); fractions of the run keep the
    ratio sane at every scale while the config caps still bound long runs.
    Derived from the run's TOTAL budget, not live remaining, so a resumed
    campaign recomputes identical values and passes strict resume checks.
    """
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


def build_runtime_config(
    mode: str,
    coding_model: "str | None",
    task_dir: str,
    session_timeouts: dict,
    agent_env_strip: "list[str] | None" = None,
) -> str:
    """Write the per-run config: shaped session deadlines + model override."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    mode_cfg = config["modes"][mode]
    params = mode_cfg["search_strategy"]["params"]
    params.update(session_timeouts)
    if coding_model:
        # Ensemble members/selector are pinned by the config on purpose:
        # --coding-model (the harness $AGENT_CONFIG) labels and drives the
        # implementation/feedback agents, not the ideation ensemble.
        params["idea_generation_model"] = coding_model
        params["implementation_model"] = coding_model
        for section in ("coding_agent", "feedback_generator"):
            mode_cfg[section]["model"] = coding_model
            mode_cfg[section]["debug_model"] = coding_model
    if agent_env_strip:
        # Credential containment (non-judge tasks): kapso's own process keeps
        # OPENAI_API_KEY for its utility-LLM roles, but the agent sessions it
        # spawns must look exactly like an official non-judge environment —
        # no OpenAI key. solve.sh decides per-task and passes the names here.
        for section in ("coding_agent", "feedback_generator"):
            mode_cfg[section].setdefault("agent_specific", {})[
                "env_strip"
            ] = list(agent_env_strip)

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
        # Trainer checkpoints live nested (artifacts/<exp>/checkpoint-N/), so
        # walk a few levels instead of only the top (run #7 review, F7).
        found = []
        for root, dirs, files in os.walk(artifacts):
            if os.path.relpath(root, artifacts).count(os.sep) > 2:
                dirs[:] = []
                continue
            if "config.json" in files:
                found.append(root)
                dirs[:] = []  # a model dir; don't descend further
        if found:
            candidates.append(max(found, key=os.path.getmtime))

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
    parser.add_argument("--guard-minutes", type=int, default=None,
                        help="Wall-clock held outside the orchestrator budget "
                             "for final consolidation (default: config "
                             "session_budget.guard_minutes)")
    parser.add_argument("--iterations", type=int, default=40,
                        help="Iteration ceiling; the time budget is the real governor")
    parser.add_argument("--mode", default="POSTTRAIN")
    parser.add_argument("--coding-agent", default="claude_code")
    parser.add_argument("--coding-model", default=None,
                        help="Model for ideation/implementation/feedback (harness $AGENT_CONFIG)")
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--strip-agent-env", action="append", default=None,
                        metavar="VAR",
                        help="Env var stripped from coding-agent/feedback "
                             "sessions (repeatable). solve.sh passes "
                             "OPENAI_API_KEY on non-judge tasks so agents "
                             "never inherit kapso's own LLM credential.")
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
    total_run_hours = timer_hours
    if deadline_ts is None:
        total_run_hours = args.hours if args.hours is not None else 10.0
        deadline_ts = time.time() + total_run_hours * 3600
    total_run_seconds = total_run_hours * 3600

    with open(CONFIG_PATH) as f:
        mode_cfg = yaml.safe_load(f)["modes"][args.mode]
    knobs = mode_cfg["session_budget"]

    guard_minutes = (
        args.guard_minutes if args.guard_minutes is not None
        else knobs["guard_minutes"]
    )
    budget_minutes = max(5, int((deadline_ts - time.time()) / 60) - guard_minutes)
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
        print("WARNING: OPENAI_API_KEY is not set — utility-LLM roles "
              "(repo memory, insight extraction) will fail")

    config_path = build_runtime_config(
        args.mode, args.coding_model, task_dir, session_timeouts,
        agent_env_strip=args.strip_agent_env,
    )

    print(f"task_dir={task_dir}")
    print(f"model={model_id!r} benchmark={parsed_benchmark!r} (id={benchmark_id!r})")
    print(f"budget={budget_minutes} min of a {total_run_hours}h run "
          f"(guard={guard_minutes} min, finalization reserve={reserve_minutes:.0f} min), "
          f"iterations<={args.iterations}")
    print(f"session caps: ideation={session_timeouts['ideation_timeout']}s "
          f"implementation={session_timeouts['implementation_timeout']}s")
    print(f"config={config_path} mode={args.mode} coding_model={args.coding_model} "
          f"agent_env_strip={args.strip_agent_env or []}")

    handler = PostTrainBenchHandler(
        task_dir=task_dir,
        official_prompt=prompt,
        model_id=model_id,
        benchmark_name=parsed_benchmark,
        benchmark_id=benchmark_id,
        deadline_ts=deadline_ts,
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
        session_caps=session_timeouts,
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
            finalization_reserve_minutes=reserve_minutes,
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
