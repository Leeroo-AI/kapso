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
import glob
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
from benchmarks.kaggle import kernel_slots
from benchmarks.kaggle.handler import KaggleNotebookHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
RULES_PATH = os.path.join(os.path.dirname(__file__), "RULES.md")
SLOTS_PATH = os.path.join(os.path.dirname(__file__), "kernel_slots.py")
# The repo's kaggle-cli-submission skill doubles as the lanes' submission
# playbook. Claude CLIs only discover it natively when cwd is this repo, and
# codex has no skill loader at all — while lanes run in isolated session
# clones on whichever CLI the config picks. So it is staged into the task dir
# and every coding-agent CLI gets the same absolute path from the handler
# context (the RULES.md pattern).
SKILL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    ".claude", "skills", "kaggle-cli-submission", "SKILL.md"))

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
    if proc.stdout.strip() == "Not found":
        # The CLI's empty-state for an account with no kernels (exit 0,
        # non-JSON) — a real possibility on a fresh account after a
        # zero-push run, not a corrupt payload.
        return []
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


def classify_submit_output(stdout: str, stderr: str) -> str:
    """One label per submit attempt; the CLI exits 0 even on rejection.

    Verified live 2026-08-02: a rejected code submission prints `403 Client
    Error ... CreateCodeSubmission` and exits 0, so the exit code carries no
    signal — the output text is the only discriminator. "accepted" means no
    rejection signature; anything else is the signature found.
    """
    blob = stdout + stderr
    if "403" in blob:
        return "rejected-403"          # version invalid OR no session capacity
    if "400" in blob:
        return "rejected-400"          # wrong endpoint/modality for this comp
    if "Error" in blob or "error:" in blob:
        return "rejected-error"
    return "accepted"


def submit_kernel_output(kaggle_bin: str, competition: str, ref: str,
                         message: str, max_version: int,
                         timeout_seconds: int) -> tuple:
    """Submit the NEWEST submittable version; (version|None, attempt log).

    `-v` is mandatory for code competitions and the CLI exposes the version
    nowhere (neither `kernels list --format json` nor `kernels status` carries
    it), so the version is discovered by probing. The probe runs DOWNWARD:
    upward would stop at the oldest submittable version instead of the newest.
    Every attempt's output is kept — run 5 lost six kernels to rejections
    whose stderr was discarded, leaving only "no version took" behind.
    """
    attempts = []
    for version in range(max_version, 0, -1):
        proc = subprocess.run(
            [kaggle_bin, "competitions", "submit", "-c", competition,
             "-k", ref, "-v", str(version), "-f", "submission.csv",
             "-m", message],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        verdict = classify_submit_output(proc.stdout, proc.stderr)
        attempts.append({"version": version, "verdict": verdict,
                         "output": (proc.stdout + proc.stderr).strip()})
        if verdict == "accepted":
            return version, attempts
    return None, attempts


def banked_kernel_refs(task_dir: str) -> set:
    """Kernel refs with a public score in best_score.log.

    The board line is `<public_score> <iso-time> <kernel-ref> <idea>`; the ref
    field is what makes never-scored kernels computable. A line whose third
    field is not a ref (a lane that skipped it) simply attributes nothing —
    that only affects harvest ORDER, never correctness. Missing log = nothing
    banked; a malformed score field raises upstream where it is read.
    """
    score_log = os.path.join(task_dir, "best_score.log")
    if not os.path.isfile(score_log):
        return set()
    refs = set()
    with open(score_log, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 3 and kernel_slots.KERNEL_REF_PATTERN.match(fields[2]):
                refs.add(fields[2])
    return refs


def rank_harvest_candidates(refs: list, banked: set) -> list:
    """Never-scored kernels first — alphabet cost run 5 two lanes' scores.

    A kernel with no banked score is one submission away from turning finished
    work into leaderboard value; a banked one only offers a stochastic re-roll
    (scoring re-runs the kernel). Stable within each group.
    """
    return ([r for r in refs if r not in banked]
            + [r for r in refs if r in banked])


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
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise FileNotFoundError("kaggle CLI not on PATH — cannot harvest")

    refs = sorted(set(discover_run_kernels(task_dir)) | set(kernels_run_since(
        kaggle_bin, run_started_utc,
        final_eval_cfg["harvest_kernel_list_size"], timeout_seconds)))
    report = {"kernels_found": len(refs), "submitted": [], "skipped": []}
    if not refs:
        # Not every task is a code competition; one scored from an uploaded
        # file pushes no kernels, and there is nothing here to ship. Returning
        # empty keeps the leaderboard readout alive for those tasks.
        print("[harvest] no kernels from this run — nothing to harvest")
        return report

    # The sample file's directory varies per competition (timed-deps nests it
    # under dataset/archive/ — that fixed-path raise killed two runs' entire
    # leaderboard readouts). Search for it; without a template the candidates
    # are skipped loudly below, but the readout always survives.
    template_name = final_eval_cfg["submission_template"]
    template_matches = sorted(
        glob.glob(os.path.join(task_dir, "dataset", "**", template_name),
                  recursive=True))
    template = template_matches[0] if template_matches else None
    if template is None:
        print(f"[harvest] no {template_name} under dataset/ — "
              f"{len(refs)} kernels skipped unvalidated")
        report["skipped"] = [
            {"kernel": ref, "reason": f"no {template_name} template found"}
            for ref in refs
        ]
        return report
    workdir = os.path.join(root, ".harvest")
    seen_digests = {}

    # Never-scored kernels first: run 5 spent both scoring slots re-rolling
    # already-scored kernels (alphabetical order) while two unscored ones —
    # finished work one submission away from value — were bounced.
    refs = rank_harvest_candidates(refs, banked_kernel_refs(task_dir))

    # Gate every candidate down to something submittable before spending a
    # scoring slot on it.
    ready = []
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
        ready.append(ref)

    # Scoring scheduler. A code submission RE-RUNS the kernel, occupying one
    # of the account's 2 GPU sessions for about the kernel's runtime — run 5's
    # harvest submitted two, filled the pool, and misread the next 72
    # rejections as "no version took". So: take a score ticket per submission
    # (ship priority — this is the reserved endgame window), fill both slots,
    # and submit the next candidate only when one scores.
    deadline = time.time() + final_eval_cfg["harvest_budget_seconds"]
    poll_seconds = final_eval_cfg["harvest_poll_seconds"]
    retry_seconds = final_eval_cfg["harvest_retry_seconds"]
    pending = []          # [{kernel, version, ticket, submitted_at}]
    retried = set()

    def poll_scored():
        """Release tickets of harvest submissions that finished scoring."""
        proc = subprocess.run(
            [kaggle_bin, "competitions", "submissions", competition,
             "--format", "json", "-q"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        rows = parse_submissions_json(proc.stdout)
        still = []
        for entry in pending:
            # Window by date: a previous run's harvest used the same message
            # for the same slug, and its scored row must not satisfy this one.
            since = datetime.fromtimestamp(
                entry["submitted_at"] - 60, timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            row = next(
                (r for r in rows
                 if entry["kernel"].split("/")[-1] in str(r.get("description", ""))
                 and str(r.get("date", ""))[:19] >= since
                 and "pending" not in str(r.get("status", "")).lower()), None)
            if row is None:
                still.append(entry)
                continue
            kernel_slots.release(task_dir, entry.pop("ticket"))
            entry["publicScore"] = str(row.get("publicScore", ""))
            report["submitted"].append(entry)
        pending[:] = still

    while (ready or pending) and time.time() < deadline:
        while ready and time.time() < deadline:
            ref = ready[0]
            wait = min(300.0, max(0.0, deadline - time.time()))
            ticket = kernel_slots.acquire_blocking(
                task_dir, "gpu", "score", ref, lane="harvest",
                priority="ship", wait_seconds=wait)
            if ticket is None:
                # No scoring slot inside the budget — record the remainder
                # rather than losing the whole leaderboard readout.
                for leftover in ready:
                    report["skipped"].append(
                        {"kernel": leftover,
                         "reason": "no scoring slot within harvest budget"})
                ready.clear()
                break
            version, attempts = submit_kernel_output(
                kaggle_bin, competition, ref,
                f"harvest: {ref.split('/')[-1]} (pushed by the run, not submitted)",
                final_eval_cfg["harvest_max_kernel_version"], timeout_seconds,
            )
            if version is None:
                kernel_slots.release(task_dir, ticket)
                if ref not in retried and time.time() + retry_seconds < deadline:
                    # A rejection of every version is indistinguishable from a
                    # transient capacity block; one paced retry before giving up.
                    retried.add(ref)
                    ready.append(ready.pop(0))
                    print(f"[harvest] {ref}: all versions rejected — "
                          f"retrying once after {retry_seconds}s")
                    time.sleep(retry_seconds)
                else:
                    ready.pop(0)
                    report["skipped"].append(
                        {"kernel": ref, "reason": "no version took",
                         "attempts": attempts})
                continue
            ready.pop(0)
            pending.append({"kernel": ref, "version": version,
                            "ticket": ticket, "submitted_at": time.time()})
        if pending:
            time.sleep(min(poll_seconds, max(0.0, deadline - time.time())))
            poll_scored()

    poll_scored()
    for entry in pending:
        # Budget ran out mid-scoring: the submission is in flight and counts;
        # the slot frees itself when scoring ends (the reap verifies).
        kernel_slots.release(task_dir, entry.pop("ticket"))
        entry["publicScore"] = ""
        report["submitted"].append(entry)

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
        # The harvest waits for its own scores inside its budget, so the
        # leaderboard is read immediately after it returns.
        harvest = harvest_unsubmitted_kernels(
            root, competition, run_started, final_eval_cfg)

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

    # Stage the CLI playbook: the statement names the modality and the file
    # but never the commands, so this is where any lane learns the
    # push/poll/submit/score flow (run 5's lanes had neither and burned their
    # first submission rediscovering -k/-v from --help).
    shutil.copy2(SKILL_PATH, os.path.join(task_dir, "KAGGLE_CLI.md"))

    # Stage the curated knowledge bank (config knowledge_bank_dir,
    # repo-root-relative): the shared-learning book every module searches
    # FIRST. Configured-but-missing is a launch defect, not an option — a box
    # that did not receive the bank must die here, not run blind.
    bank_rel = mode_cfg.get("knowledge_bank_dir")
    if bank_rel:
        bank_src = os.path.join(REPO_ROOT, bank_rel)
        if not os.path.isdir(bank_src):
            raise FileNotFoundError(
                f"knowledge_bank_dir={bank_rel!r} resolved to {bank_src} which "
                "does not exist — ship the bank to this machine or unset the key")
        bank_dst = os.path.join(task_dir, "knowledge_bank")
        if os.path.isdir(bank_dst):
            shutil.rmtree(bank_dst)
        shutil.copytree(bank_src, bank_dst)

    # Stage the slot ticket office + its limits. Lanes run in isolated session
    # clones, so this has to be reachable by path rather than by import; a stale
    # ledger from a previous launch would hold phantom tickets, so it goes too.
    shutil.copy2(SLOTS_PATH, os.path.join(task_dir, "kernel_slots.py"))
    with open(os.path.join(task_dir, ".kernel_slots_config.json"), "w") as f:
        json.dump(mode_cfg["session_budget"]["kernel_slots"], f, indent=2)
    stale_ledger = os.path.join(task_dir, ".kernel_slots.json")
    if os.path.isfile(stale_ledger):
        os.remove(stale_ledger)

    run_defaults = mode_cfg["run_defaults"]
    hours = args.hours if args.hours is not None else run_defaults["hours"]
    node_expansion = (args.node_expansion if args.node_expansion is not None
                      else run_defaults["node_expansion"])

    total_run_seconds = hours * 3600
    # The clock starts at URL-in, not here: the preflight stamps run_meta.json
    # before it downloads, so its ~8 min is billed to the run rather than being
    # free time ahead of it (a real competition's window opens once, and the
    # download and statement authoring happen inside it). Direct runner
    # invocations with no preflight stamp start their clock now.
    meta_path = os.path.join(root, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            started_utc = json.load(f)["run_started_utc"]
        origin_ts = datetime.fromisoformat(started_utc).timestamp()
    else:
        origin_ts = time.time()
        with open(meta_path, "w") as f:
            json.dump({"run_started_utc":
                       datetime.now(timezone.utc).isoformat()}, f, indent=2)

    deadline_ts = origin_ts + total_run_seconds
    spent_minutes = (time.time() - origin_ts) / 60
    knobs = mode_cfg["session_budget"]
    guard_minutes = (args.guard_minutes if args.guard_minutes is not None
                     else knobs["guard_minutes"])
    # The campaign returns early enough for the harvest's submissions to be
    # SCORED before the deadline, not merely sent: a submission counts only if
    # it finished scoring in time, so a sweep that fires at the buzzer produces
    # entries that never reach the leaderboard. The harvest only submits
    # already-COMPLETE kernels, so this window covers the submit calls plus
    # Kaggle's scoring latency — not another kernel run.
    harvest_window = knobs["harvest_window_minutes"]
    budget_minutes = int(
        hours * 60 - spent_minutes - guard_minutes - harvest_window
    )
    if budget_minutes < knobs["min_campaign_minutes"]:
        sys.exit(
            f"only {budget_minutes} min of the {hours}h budget remain after "
            f"{spent_minutes:.1f} min of preflight — below the "
            f"{knobs['min_campaign_minutes']} min floor; nothing scoreable "
            "could finish"
        )
    # Sized to ONE submission round trip (push -> kernel run -> submit ->
    # score), not to a fraction of the run: a campaign that ends before
    # shipping scores nothing. The handler hands most of it back once a public
    # score is banked (deliverable_ready_reserve_seconds).
    reserve_minutes = knobs["finalization_reserve_minutes"]
    session_timeouts = shape_session_timeouts(mode_cfg, budget_minutes * 60)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")):
        print("WARNING: neither ANTHROPIC_API_KEY nor CLAUDE_CODE_OAUTH_TOKEN is set")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — utility-LLM roles will fail")
    if not shutil.which("kaggle"):
        print("WARNING: kaggle CLI not on PATH — submissions will fail")

    config_path = build_runtime_config(args.mode, task_dir, session_timeouts,
                                       shared_cache_dir=args.shared_cache_dir,
                                       node_expansion=node_expansion)

    print(f"root={root} competition={competition} K={node_expansion} hours={hours}")
    print(f"clock started {spent_minutes:.1f} min ago (preflight); "
          f"deadline {datetime.fromtimestamp(deadline_ts, timezone.utc):%H:%M:%S} UTC")
    print(f"budget={budget_minutes} min (guard={guard_minutes} min, "
          f"harvest window={harvest_window} min, "
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
        # The statement plus the one fact the feedback judge needs: this goal
        # has no "fully achieved" — the radar rehearsal's judges voted stop at
        # 0.982/1.0 and ended the campaign 25 min early. The handler's
        # honor_agent_stop=False is the hard guarantee; this line keeps the
        # judges' own feedback from telling lanes "no further iteration is
        # required".
        goal=(statement
              + "\n\nThis is an open-ended competition: the goal is never"
              " 'fully achieved'. However high the score, keep improving the"
              " best public score until the time budget expires — judge each"
              " iteration on score progress, never on completion."),
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
