#!/usr/bin/env python3
"""Task-brief -> run-root preflight for Kaggle competitions.

Turns the organizer's starter prompt (or a competition URL, for tasks that
publish none) into the run root the runner consumes:
    <root>/task/starter_prompt.txt   the launch input, verbatim
    <root>/task/dataset/             competition data + statement.md
    <root>/task/kaggle.json          {"competition": <slug>}

The brief is OPAQUE to this code: no slug parsing here. One `codex exec`
session receives it verbatim, identifies the competition it names, downloads
the data with the authenticated kaggle CLI, writes kaggle.json, and authors
dataset/statement.md per preflight_spec.md. This code only stages input and
validates the artifacts afterwards — fail loud on anything missing.

Usage:
    python -m benchmarks.ioai2026.preflight \
        --task 'Solve the Kaggle competition <slug>. ...' \
        --root ~/kaggle_run
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

load_dotenv()

from kapso.execution.coding_agents.factory import CodingAgentFactory

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
SPEC_PATH = os.path.join(os.path.dirname(__file__), "preflight_spec.md")
RULES_PATH = os.path.join(os.path.dirname(__file__), "RULES.md")


def build_prompt(task_brief: str, task_dir: str, dataset_dir: str,
                 statement_path: str, rules_path: str) -> str:
    """Dynamic per-competition header + the fixed statement-authoring spec.

    The brief goes in VERBATIM and IN FULL — it is organizer instruction text
    and may carry directives beyond naming the competition.
    """
    spec = open(SPEC_PATH, encoding="utf-8").read()
    header = (
        "# Preflight assignment\n\n"
        "Your launch input — usually the organizer's starter prompt, possibly "
        "a plain competition URL — verbatim between the markers:\n"
        "<<<TASK BRIEF\n"
        f"{task_brief}\n"
        "TASK BRIEF>>>\n\n"
        "A copy of it sits at task/starter_prompt.txt.\n\n"
        f"Task directory: {task_dir}\n"
        f"Dataset directory (you download the data into it): {dataset_dir}\n"
        f"Binding rules, which override the competition pages: {rules_path}\n"
        f"Write the statement to: {statement_path}\n\n"
    )
    return header + spec


def validate_root(task_dir: str) -> str:
    """Fail-loud artifact checks after the authoring session; returns the slug."""
    dataset_dir = os.path.join(task_dir, "dataset")
    statement_path = os.path.join(dataset_dir, "statement.md")
    kaggle_json = os.path.join(task_dir, "kaggle.json")
    if not (os.path.isfile(statement_path) and os.path.getsize(statement_path) > 0):
        sys.exit(f"{statement_path} missing or empty after preflight")
    if not os.path.isdir(dataset_dir):
        sys.exit(f"{dataset_dir} missing after preflight")
    data_entries = [n for n in os.listdir(dataset_dir) if n != "statement.md"]
    if not data_entries:
        sys.exit(f"{dataset_dir} holds no competition data beyond statement.md")
    if not os.path.isfile(kaggle_json):
        sys.exit(f"{kaggle_json} missing after preflight")
    with open(kaggle_json) as f:
        slug = json.load(f).get("competition", "")
    if not slug:
        sys.exit(f"{kaggle_json} carries no competition slug")
    return slug


def run_preflight(task_brief: str, root: str, mode: str = "KAGGLE") -> str:
    root = os.path.abspath(root)
    task_dir = os.path.join(root, "task")
    dataset_dir = os.path.join(task_dir, "dataset")
    statement_path = os.path.join(dataset_dir, "statement.md")
    os.makedirs(dataset_dir, exist_ok=True)

    # The clock starts HERE, at brief-in. A real competition's window opens
    # when the organizers hand over the starter prompt, so the download and
    # the statement-authoring session are inside the budget, not free time
    # before it; the runner reads this stamp as its origin instead of
    # starting a fresh clock of its own.
    meta_path = os.path.join(root, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"run_started_utc": datetime.now(timezone.utc).isoformat()},
                  f, indent=2)

    with open(os.path.join(task_dir, "starter_prompt.txt"), "w",
              encoding="utf-8") as f:
        f.write(task_brief)
    # Staged here as well as by the runner: the authoring agent is told these
    # rules override the competition pages, so it has to be able to read them.
    rules_path = os.path.join(task_dir, "RULES.md")
    shutil.copy2(RULES_PATH, rules_path)

    with open(CONFIG_PATH) as f:
        preflight_cfg = yaml.safe_load(f)["modes"][mode]["preflight"]

    cli = preflight_cfg["cli"]
    if cli not in ("codex", "claude_code"):
        raise ValueError(f"preflight.cli must be codex or claude_code, got {cli!r}")
    agent_specific = {
        "effort": preflight_cfg["effort"],
        "timeout": preflight_cfg["timeout"],
        "streaming": True,
    }
    if cli == "codex":
        agent_specific["web_search"] = preflight_cfg["web_search"]
    else:
        # The claude CLI takes an explicit toolset: the preflight must read the
        # brief and downloaded files, run the kaggle CLI, browse the
        # competition pages, and write the statement + kaggle.json.
        agent_specific["auth_mode"] = preflight_cfg["auth_mode"]
        agent_specific["allowed_tools"] = [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        ] + (["WebSearch", "WebFetch"] if preflight_cfg["web_search"] else [])
    config = CodingAgentFactory.build_config(
        agent_type=cli,
        model=preflight_cfg["model"],
        agent_specific=agent_specific,
    )
    agent = CodingAgentFactory.create(config)
    agent.initialize(task_dir)
    result = agent.generate_code(
        build_prompt(task_brief, task_dir, dataset_dir, statement_path,
                     rules_path),
        timeout_seconds=preflight_cfg["timeout"],
    )

    if not result.success:
        sys.exit(f"preflight {cli} run failed: {result.error}")
    slug = validate_root(task_dir)

    data_entries = [n for n in os.listdir(dataset_dir) if n != "statement.md"]
    print(f"\nrun root ready: {root}")
    print(f"  competition: {slug}")
    print(f"  statement:   {statement_path} "
          f"({os.path.getsize(statement_path)} bytes)")
    print(f"  data:        {sorted(data_entries)[:8]}"
          f"{' ...' if len(data_entries) > 8 else ''}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True,
                        help="Task brief: the organizer's starter prompt "
                             "verbatim, or a competition URL")
    parser.add_argument("--root", required=True,
                        help="Run root to create (consumed by runner.py --root)")
    parser.add_argument("--mode", default="KAGGLE")
    args = parser.parse_args()
    run_preflight(args.task, args.root, args.mode)


if __name__ == "__main__":
    main()
