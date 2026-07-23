#!/usr/bin/env python3
"""Learning-harvest runner — extract + aggregate as claude_code CLI sessions.

Both stages run the `claude` CLI in print mode with the reusable briefs
(extract_prompt.md / aggregate_prompt.md), Fable 5 at max reasoning, OAuth.
The CLI reads CLAUDE_CODE_OAUTH_TOKEN from the environment itself; load_dotenv
populates it from .env and the subprocess inherits it — our code never reads
the secret. Model / effort / timeout / tools are config-sourced (Rule 1).

    # per finished run:
    python -m benchmarks.ioai_tasks.harvest.harvest_runner extract \
        --run-root <run_root> --out <learning.json>
    # once, after the extractions exist:
    python -m benchmarks.ioai_tasks.harvest.harvest_runner aggregate \
        --extractions a.json b.json --current-learnings LEARNINGS.md \
        --out LEARNINGS.new.md
"""

import argparse
import json
import os
import shutil
import subprocess

import yaml
from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(HERE, "..", "config.yaml")
EXTRACT_BRIEF = os.path.join(HERE, "extract_prompt.md")
AGGREGATE_BRIEF = os.path.join(HERE, "aggregate_prompt.md")


def _harvest_config() -> dict:
    with open(CONFIG_PATH) as f:
        harvest = yaml.safe_load(f)["harvest"]
    for key in ("model", "effort", "timeout_seconds", "allowed_tools"):
        if key not in harvest:
            raise ValueError(f"config harvest block missing '{key}'")
    return harvest


def _claude_bin() -> str:
    claude = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(claude):
        raise FileNotFoundError("claude CLI not found on PATH or ~/.local/bin")
    return claude


def _run_claude(prompt: str, cwd: str) -> str:
    """One claude_code print-mode session. Fail loud on non-zero exit."""
    cfg = _harvest_config()
    claude = _claude_bin()
    cmd = [claude, "-p", "--model", cfg["model"], "--effort", cfg["effort"],
           "--dangerously-skip-permissions",
           "--allowedTools", *cfg["allowed_tools"]]
    # --allowedTools is variadic and swallows a positional prompt, so the
    # prompt goes via stdin.
    proc = subprocess.run(cmd, input=prompt, cwd=cwd, capture_output=True,
                          text=True, timeout=cfg["timeout_seconds"])
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}\nstderr:\n{proc.stderr[-3000:]}"
        )
    return proc.stdout


def extract(run_root: str, out_path: str) -> str:
    run_root = os.path.abspath(run_root)
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"run root {run_root} does not exist")
    out_path = os.path.abspath(out_path)
    brief = open(EXTRACT_BRIEF, encoding="utf-8").read()
    prompt = (
        f"{brief}\n\n---\n"
        f"The completed run root is at: {run_root}\n"
        "Read its artifacts: run.log (or run_*.log), results.json, "
        "task/kapso_campaign/experiment_history.json, "
        "task/kapso_campaign/.kapso/lens_plan.json, every "
        "task/kapso_campaign/sessions/*/ (PLAN.md, changes.log, "
        "result.json, eval_profile.md), and task/submission/solution.py. "
        "Grep the big logs rather than reading them whole. When done, WRITE "
        f"your structured learning JSON (only the JSON) to: {out_path}\n"
        "Print a one-line confirmation to stdout; do not print the JSON."
    )
    _run_claude(prompt, cwd=run_root)
    with open(out_path, encoding="utf-8") as f:
        json.load(f)  # validate — malformed raises (fail loud)
    return out_path


def aggregate(extraction_paths, current_learnings: str, out_path: str,
              reference_solutions=None) -> str:
    reference_solutions = reference_solutions or []
    for path in [current_learnings, *extraction_paths, *reference_solutions]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing")
    out_path = os.path.abspath(out_path)
    brief = open(AGGREGATE_BRIEF, encoding="utf-8").read()
    listed = "\n".join(f"  - {os.path.abspath(p)}" for p in extraction_paths)
    refs = "\n".join(f"  - {os.path.abspath(p)}" for p in reference_solutions)
    refs_block = (f"Gold reference solutions to mine directly:\n{refs}\n"
                  if reference_solutions else
                  "Gold reference solutions: none supplied.\n")
    prompt = (
        f"{brief}\n\n---\n"
        f"Current Night Watch LEARNINGS.md: {os.path.abspath(current_learnings)}\n"
        f"Per-task extraction JSONs to merge:\n{listed}\n"
        f"{refs_block}"
        "Read them all, then WRITE the full updated LEARNINGS.md to: "
        f"{out_path}\nPrint a one-line confirmation to stdout."
    )
    _run_claude(prompt, cwd=os.path.dirname(out_path) or ".")
    if not os.path.isfile(out_path):
        raise RuntimeError(f"aggregate did not write {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run-root", required=True)
    e.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--extractions", nargs="+", required=True)
    a.add_argument("--current-learnings", required=True)
    a.add_argument("--reference-solutions", nargs="*", default=[],
                   help="gold reference-solution files to mine directly")
    a.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.cmd == "extract":
        path = extract(args.run_root, args.out)
        print(f"[harvest] extraction written: {path}")
    else:
        path = aggregate(args.extractions, args.current_learnings, args.out,
                         reference_solutions=args.reference_solutions)
        print(f"[harvest] aggregated LEARNINGS written: {path}")


if __name__ == "__main__":
    main()
