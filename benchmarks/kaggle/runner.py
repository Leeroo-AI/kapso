#!/usr/bin/env python3
"""Kaggle code-competition runner (IOAI AI Models Track practice tasks).

Runs the Kapso agent against a prepared run root (see preflight.py),
then reads the ground truth from the Kaggle leaderboard: best publicScore
among submissions made during the run window, plus a static audit of the
submitted kernel source for external-resource violations.

Usage:
    python -m benchmarks.kaggle.runner --root /path/to/run_root --hours 2
    python -m benchmarks.kaggle.runner --root /path/to/run_root --final-eval-only
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import yaml

from kapso.execution.orchestrator import OrchestratorAgent
from benchmarks.kaggle.handler import KaggleNotebookHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
RULES_PATH = os.path.join(os.path.dirname(__file__), "RULES.md")

# Source patterns that indicate the kernel pulls external pretrained
# resources or data. Matches are reported, not silently fatal — a human
# reads the flagged lines before trusting (or zeroing) the score.
AUDIT_PATTERNS = [
    r"hf_hub_download",
    r"snapshot_download",
    r"torch\.hub",
    r"\btimm\b",
    r"load_dataset",
    r"https?://",
    r"from_pretrained\(\s*[\"'](?!\.|/kaggle/input|model|dataset/model)",
]


def shape_session_timeouts(mode_cfg: dict, total_run_seconds: float) -> dict:
    """Per-session ceilings, bounded only by the run itself.

    There is deliberately no per-phase fraction: a phase sub-budget starved
    ideation (a member delivered 1 of 5 candidates, the selector timed out and
    the pool went unranked) while the sessions still had run budget left. The
    single enforcer is the strategy's dynamic clamp against the searchable
    budget that remains — so a session that finishes early hands its unused
    time to the next phase instead of forfeiting it.
    """
    params = mode_cfg["search_strategy"]["params"]
    return {
        "ideation_timeout": int(min(
            params["ideation_timeout"], total_run_seconds)),
        "implementation_timeout": int(min(
            params["implementation_timeout"], total_run_seconds)),
    }


def build_runtime_config(mode: str, task_dir: str, session_timeouts: dict,
                         shared_cache_dir: "str | None" = None,
                         node_expansion: "int | None" = None) -> str:
    """Write the per-run config with shaped session deadlines."""
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


def audit_kernel(kernel_dir: str) -> list:
    """Static scan of the kernel sources for external-resource pulls."""
    findings = []
    for dirpath, _, filenames in os.walk(kernel_dir):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    for pattern in AUDIT_PATTERNS:
                        if re.search(pattern, line):
                            findings.append(
                                f"{os.path.relpath(path, kernel_dir)}:"
                                f"{lineno}: [{pattern}] {line.strip()}"
                            )
    return findings


def parse_submissions_json(raw: str) -> list:
    """Parse `kaggle competitions submissions --format json` output.

    The CLI may prefix the JSON with pagination noise ("Next Page Token =
    ..."); the payload starts at the first bracket. Anything unparseable
    raises — fail loud.
    """
    start_candidates = [i for i in (raw.find("["), raw.find("{")) if i >= 0]
    if not start_candidates:
        raise ValueError(f"no JSON payload in submissions output: {raw[:200]!r}")
    payload = json.loads(raw[min(start_candidates):])
    if isinstance(payload, dict):
        payload = payload.get("submissions", [])
    if not isinstance(payload, list):
        raise ValueError("submissions payload is neither a list nor "
                         "a {'submissions': [...]} object")
    return payload


def best_public_score(submissions: list, since_utc_iso: str) -> dict:
    """Best publicScore among submissions at/after the run start."""
    since_key = since_utc_iso.replace("T", " ")[:19]
    considered, best = [], None
    for sub in submissions:
        date_key = str(sub.get("date", "")).replace("T", " ")[:19]
        if date_key < since_key:
            continue
        raw_score = str(sub.get("publicScore", "") or "").strip()
        entry = {
            "date": date_key,
            "status": str(sub.get("status", "")),
            "description": str(sub.get("description", "")),
            "publicScore": raw_score,
        }
        considered.append(entry)
        if re.fullmatch(r"-?\d+(\.\d+)?", raw_score):
            score = float(raw_score)
            if best is None or score > best["score"]:
                best = {"score": score, **entry}
    return {"best": best, "submissions": considered}


def discover_run_kernels(task_dir: str) -> list:
    """Kernel refs this run pushed, read from the lanes' kernel-metadata.json.

    K-way lanes namespace their own submission directories, so there is no
    single canonical kernel path to look in; the metadata files each lane wrote
    are the authoritative record of what this run created.
    """
    refs = []
    for dirpath, _, filenames in os.walk(os.path.join(task_dir, "submission")):
        if "kernel-metadata.json" not in filenames:
            continue
        meta_path = os.path.join(dirpath, "kernel-metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("id"):
            raise ValueError(f"{meta_path} has no 'id' field")
        refs.append(meta["id"])
    return sorted(set(refs))


def kernels_run_since(kaggle_bin: str, since_utc_iso: str, page_size: int,
                      timeout_seconds: int) -> list:
    """Our own kernels whose last run started at/after the campaign did.

    Local metadata alone is not enough: a lane can push a kernel and never
    record it under submission/ (run 3's lane 3 did exactly that, leaving a
    COMPLETE kernel discoverable only from the account listing).
    """
    proc = subprocess.run(
        [kaggle_bin, "kernels", "list", "-m", "--sort-by", "dateRun",
         "--page-size", str(page_size), "--format", "json"],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels list exited {proc.returncode}: {proc.stderr[-500:]}")
    since_key = since_utc_iso.replace("T", " ")[:19]
    return sorted(
        entry["ref"] for entry in json.loads(proc.stdout)
        if str(entry.get("lastRunTime", "")).replace("T", " ")[:19] >= since_key
    )


def kernel_status(kaggle_bin: str, ref: str, timeout_seconds: int) -> str:
    """KernelWorkerStatus for a kernel ref (COMPLETE / RUNNING / ...)."""
    proc = subprocess.run(
        [kaggle_bin, "kernels", "status", ref],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", proc.stdout + proc.stderr)
    return match.group(1) if match else "UNKNOWN"


def submission_matches_template(candidate_path: str, template_path: str) -> bool:
    """Same header, row count and id column (in order) as the sample file."""
    with open(template_path, newline="", encoding="utf-8") as f:
        template = list(csv.reader(f))
    with open(candidate_path, newline="", encoding="utf-8") as f:
        candidate = list(csv.reader(f))
    if len(candidate) != len(template) or candidate[0] != template[0]:
        return False
    return all(c[0] == t[0] for c, t in zip(candidate[1:], template[1:]))


def submit_kernel_output(kaggle_bin: str, competition: str, ref: str,
                         message: str, max_version: int,
                         timeout_seconds: int) -> "int | None":
    """Submit the NEWEST submittable version of a kernel; None if none took.

    `-v` is mandatory for code competitions and the CLI exposes the version
    nowhere (neither `kernels list --format json` nor `kernels status` carries
    it), so the version is discovered by probing. The probe runs DOWNWARD:
    upward would stop at the oldest submittable version instead of the newest.
    """
    for version in range(max_version, 0, -1):
        proc = subprocess.run(
            [kaggle_bin, "competitions", "submit", "-c", competition,
             "-k", ref, "-v", str(version), "-f", "submission.csv",
             "-m", message],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        blob = proc.stdout + proc.stderr
        if proc.returncode == 0 and "403" not in blob and "Error" not in blob:
            return version
    return None


def harvest_unsubmitted_kernels(root: str, competition: str,
                                run_started_utc: str,
                                final_eval_cfg: dict) -> dict:
    """Submit every COMPLETE kernel this run pushed, before reading scores.

    A kernel and its output live on Kaggle independently of the run box, so a
    campaign that ends before firing `competitions submit` has still produced a
    scoreable artifact — run 2 left one worth 0.83626 unshipped. Duplicate
    submissions are harmless (Kaggle scores best-of) and attributing a past
    submission to a kernel is not possible from the API, so candidates are
    deduped by output content alone rather than by guessing what was already
    sent.
    """
    timeout_seconds = final_eval_cfg["timeout_seconds"]
    task_dir = os.path.join(root, "task")
    template = os.path.join(task_dir, "dataset", "submission.csv")
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise FileNotFoundError("kaggle CLI not on PATH — cannot harvest")
    if not os.path.isfile(template):
        raise FileNotFoundError(f"{template} missing — cannot validate outputs")

    refs = sorted(set(discover_run_kernels(task_dir)) | set(kernels_run_since(
        kaggle_bin, run_started_utc,
        final_eval_cfg["harvest_kernel_list_size"], timeout_seconds)))
    report = {"kernels_found": len(refs), "submitted": [], "skipped": []}
    workdir = os.path.join(root, ".harvest")
    seen_digests = {}

    for ref in refs:
        status = kernel_status(kaggle_bin, ref, timeout_seconds)
        if status != "COMPLETE":
            report["skipped"].append({"kernel": ref, "reason": f"status {status}"})
            continue
        dest = os.path.join(workdir, ref.replace("/", "__"))
        os.makedirs(dest, exist_ok=True)
        subprocess.run(
            [kaggle_bin, "kernels", "output", ref, "-p", dest, "--force"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        produced = os.path.join(dest, "submission.csv")
        if not os.path.isfile(produced):
            report["skipped"].append({"kernel": ref, "reason": "no submission.csv"})
            continue
        if not submission_matches_template(produced, template):
            report["skipped"].append({"kernel": ref, "reason": "shape mismatch"})
            continue
        with open(produced, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest in seen_digests:
            report["skipped"].append(
                {"kernel": ref, "reason": f"identical to {seen_digests[digest]}"})
            continue
        seen_digests[digest] = ref
        version = submit_kernel_output(
            kaggle_bin, competition, ref,
            f"harvest: {ref.split('/')[-1]} (pushed by the run, not submitted)",
            final_eval_cfg["harvest_max_kernel_version"], timeout_seconds,
        )
        if version is None:
            report["skipped"].append({"kernel": ref, "reason": "no version took"})
            continue
        report["submitted"].append({"kernel": ref, "version": version})

    print(f"[harvest] kernels={report['kernels_found']} "
          f"submitted={len(report['submitted'])} skipped={len(report['skipped'])}")
    return report


def run_final_eval(root: str, competition: str, final_eval_cfg: dict) -> dict:
    """Read the leaderboard truth: best publicScore in the run window."""
    timeout_seconds = final_eval_cfg["timeout_seconds"]
    meta_path = os.path.join(root, "run_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} missing — the campaign runner writes it at launch"
        )
    with open(meta_path) as f:
        run_started = json.load(f)["run_started_utc"]

    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise FileNotFoundError("kaggle CLI not on PATH — cannot read scores")

    # Ship before reading: a kernel left unsubmitted scores nothing, and the
    # harvested entries must be in flight before the leaderboard is polled.
    harvest = None
    if final_eval_cfg["harvest_unsubmitted"]:
        harvest = harvest_unsubmitted_kernels(
            root, competition, run_started, final_eval_cfg)
        if harvest["submitted"]:
            wait = final_eval_cfg["harvest_score_wait_seconds"]
            print(f"[harvest] waiting {wait}s for harvested submissions to score")
            time.sleep(wait)

    proc = subprocess.run(
        [kaggle_bin, "competitions", "submissions", competition,
         "--format", "json", "-q"],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        return {
            "error": f"kaggle submissions exited {proc.returncode}",
            "stderr_tail": proc.stderr[-2000:],
        }
    report = best_public_score(parse_submissions_json(proc.stdout), run_started)
    report["audit"] = audit_kernel(
        os.path.join(root, "task", "submission", "kernel")
    )
    if harvest is not None:
        report["harvest"] = harvest
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run Kapso on a Kaggle code competition")
    parser.add_argument("--root", required=True,
                        help="Run root from preflight.py")
    parser.add_argument("--hours", type=float, default=None,
                        help="Run wall-clock hours (default: config run_defaults.hours)")
    parser.add_argument("--guard-minutes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", default="KAGGLE")
    parser.add_argument("--coding-agent", default=None,
                        help="Override the config's coding_agent.type "
                             "(default: None — the config block wins, codex here)")
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--final-eval-only", action="store_true")
    parser.add_argument("--node-expansion", type=int, default=None,
                        help="K parallel implementation lanes per round "
                             "(default: config run_defaults.node_expansion)")
    parser.add_argument("--shared-cache-dir", default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    task_dir = os.path.join(root, "task")
    statement_path = os.path.join(task_dir, "dataset", "statement.md")
    kaggle_meta_path = os.path.join(task_dir, "kaggle.json")
    for required in (statement_path, kaggle_meta_path):
        if not os.path.isfile(required):
            sys.exit(f"{required} missing — run preflight.py first")
    with open(kaggle_meta_path) as f:
        competition = json.load(f)["competition"]

    with open(CONFIG_PATH) as f:
        mode_cfg = yaml.safe_load(f)["modes"][args.mode]

    if args.final_eval_only:
        report = run_final_eval(root, competition, mode_cfg["final_eval"])
        results_path = os.path.join(root, "results.json")
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    statement = open(statement_path, encoding="utf-8").read()

    # Stage the organizers' binding rules beside the task so every session can
    # read them; copied fresh each launch so an edited RULES.md always wins.
    shutil.copy2(RULES_PATH, os.path.join(task_dir, "RULES.md"))

    run_defaults = mode_cfg["run_defaults"]
    hours = args.hours if args.hours is not None else run_defaults["hours"]
    node_expansion = (args.node_expansion if args.node_expansion is not None
                      else run_defaults["node_expansion"])

    total_run_seconds = hours * 3600
    deadline_ts = time.time() + total_run_seconds
    knobs = mode_cfg["session_budget"]
    guard_minutes = (args.guard_minutes if args.guard_minutes is not None
                     else knobs["guard_minutes"])
    budget_minutes = max(5, int(hours * 60) - guard_minutes)
    # Sized to ONE submission round trip (push -> kernel run -> submit ->
    # score), not to a fraction of the run: a campaign that ends before
    # shipping scores nothing. The handler hands most of it back once a public
    # score is banked (deliverable_ready_reserve_seconds).
    reserve_minutes = knobs["finalization_reserve_minutes"]
    session_timeouts = shape_session_timeouts(mode_cfg, total_run_seconds)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        print("WARNING: neither ANTHROPIC_API_KEY nor CLAUDE_CODE_OAUTH_TOKEN is set")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — utility-LLM roles will fail")
    if not shutil.which("kaggle"):
        print("WARNING: kaggle CLI not on PATH — submissions will fail")

    with open(os.path.join(root, "run_meta.json"), "w") as f:
        json.dump({"run_started_utc":
                   datetime.now(timezone.utc).isoformat()}, f)

    config_path = build_runtime_config(args.mode, task_dir, session_timeouts,
                                       shared_cache_dir=args.shared_cache_dir,
                                       node_expansion=node_expansion)

    print(f"root={root} competition={competition} K={node_expansion} hours={hours}")
    print(f"budget={budget_minutes} min (guard={guard_minutes} min, "
          f"finalization reserve={reserve_minutes:.0f} min), "
          f"iterations<={args.iterations}")
    print(f"session caps: ideation={session_timeouts['ideation_timeout']}s "
          f"implementation={session_timeouts['implementation_timeout']}s")

    handler = KaggleNotebookHandler(
        task_dir=task_dir,
        statement=statement,
        deadline_ts=deadline_ts,
        session_caps=session_timeouts,
        kaggle={"competition": competition},
        insured_reserve_seconds=knobs["insured_reserve_minutes"] * 60,
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
        "hours": hours,
        "competition": competition,
        "kernel_present": os.path.isfile(
            os.path.join(task_dir, "submission", "kernel", "script.py")),
    }
    if not args.skip_final_eval:
        summary["final"] = run_final_eval(
            root, competition, mode_cfg["final_eval"])
    results_path = os.path.join(root, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
