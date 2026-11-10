"""The evaluation Kapso protects: the development conversations through the harness, scored by the runner.

    python kapso_evaluation/evaluate.py            # the 20 development conversations once; prints `pass_1: <share>` last
    python kapso_evaluation/evaluate.py --fast     # the five fast conversations once, a smoke check: prints
                                                   # `fast_pass_1: <share>` last, not comparable with a full score,
                                                   # and records no score
    python kapso_evaluation/evaluate.py --check    # the integrity checks alone, no run

Run from the workspace root, the directory holding harness/ and bench/. In order:

1. Integrity. bench/run_dev.py, bench/adapter.py, bench/endpoint.json, bench/dev_tasks.json,
   bench/protection.json and bench/fixed_conditions.json must be byte-identical to the copies under
   protected/. The runner's files and trees that fixed.json records (the run script, the runner's
   pyproject.toml and uv.lock, every .py under its source tree, the banking domain's database,
   prompts and documents) must hash as recorded. The model endpoint must list the fixed model with
   the fixed context length. A mismatch is a failure, exit 2, not a score.
2. The run. bench/run_dev.py, unchanged, from the workspace root (--fast passed through), its output
   streamed here. It starts harness/serve.py, waits for the endpoint, runs the conversations at the
   fixed settings, writes results/ and the logs under bench/runs/<save>/.
3. The reading. The paired flips against the recorded reference runs (protected/references.json),
   then the metric as the last line, and result.json beside this script: evaluation_script_path,
   evaluation_output, score, the harness version and settings that ran, the run's summary.

Exit codes: 0 scored; 2 an integrity mismatch, an endpoint that does not answer, a runner failure or
an incomplete run; 3 harness/serve.py is missing. Nothing here opens the runner's private files:
bench/run_dev.py reads the runner's results file as it always has, and no other protected path is read.
Standard library only.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTECTED = HERE / "protected"
FIXED = HERE / "fixed.json"
RESULT = HERE / "result.json"
REFERENCES = PROTECTED / "references.json"
BENCH_FILES = ("run_dev.py", "adapter.py", "endpoint.json", "dev_tasks.json", "protection.json", "fixed_conditions.json")
METRIC = "pass_1"
SCRIPT_PATH = "kapso_evaluation/evaluate.py"

printed = []


def say(line=""):
    printed.append(line)
    print(line, flush=True)


def fail(code, reason):
    say(f"evaluation failed: {reason}")
    sys.exit(code)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root, pattern, exclude=()):
    """(file count, one digest) over every file under root matching pattern, by relative path, in
    sorted order; a path with a component in exclude is skipped and never opened."""
    root = Path(root)
    files = sorted(path for path in root.rglob(pattern)
                   if path.is_file() and not any(part in exclude for part in path.relative_to(root).parts))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return len(files), digest.hexdigest()


def fetch_json(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:
            return json.loads(reply.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def workspace_root():
    cwd = Path.cwd()
    if (cwd / "bench" / "run_dev.py").is_file() and (cwd / "harness").is_dir():
        return cwd
    parent = HERE.parent
    if (parent / "bench" / "run_dev.py").is_file() and (parent / "harness").is_dir():
        return parent
    fail(2, "run from the workspace root, the directory holding harness/ and bench/")


def check_bench(workspace):
    problems = []
    for name in BENCH_FILES:
        theirs = workspace / "bench" / name
        if not theirs.is_file():
            problems.append(f"bench/{name} is missing")
        elif theirs.read_bytes() != (PROTECTED / name).read_bytes():
            problems.append(f"bench/{name} differs from the protected copy")
    return problems


def check_runner(fixed):
    problems = []
    for path, digest in fixed["runner_files"].items():
        if not Path(path).is_file():
            problems.append(f"{path} is missing")
        elif sha256_file(path) != digest:
            problems.append(f"{path} has changed")
    for root, spec in fixed["runner_trees"].items():
        if not Path(root).is_dir():
            problems.append(f"{root} is missing")
            continue
        count, digest = tree_digest(root, spec["pattern"], tuple(spec.get("exclude", ())))
        if (count, digest) != (spec["files"], spec["sha256"]):
            problems.append(f"{root} ({spec['pattern']}) has changed: {count} files now, {spec['files']} recorded")
    return problems


def check_endpoint(fixed, endpoint, wait_seconds, poll_seconds):
    want = fixed["endpoint"]
    deadline = time.monotonic() + wait_seconds
    listing = fetch_json(endpoint["chat_base_url"] + "/models", 10)
    while not listing and time.monotonic() < deadline:
        time.sleep(poll_seconds)
        listing = fetch_json(endpoint["chat_base_url"] + "/models", 10)
    if not listing:
        return [f"the model endpoint {endpoint['chat_base_url']} did not answer within {wait_seconds} s"]
    served = next((item for item in listing.get("data", []) if item.get("id") == endpoint["model"]), None)
    if served is None:
        return [f"the endpoint does not list `{endpoint['model']}`"]
    return [f"the endpoint's {key} is {served.get(key)!r}; fixed at {want[key]!r}"
            for key in ("root", "max_model_len") if served.get(key) != want[key]]


def integrity(workspace, fixed, endpoint, wait_seconds):
    problems = check_bench(workspace) + check_runner(fixed) + check_endpoint(fixed, endpoint, wait_seconds, endpoint["endpoint_poll_seconds"])
    if problems:
        for problem in problems:
            say(f"integrity: {problem}")
        fail(2, f"{len(problems)} integrity problem(s); the fixed conditions of the measurement do not hold")
    say(f"integrity: ok; bench/ files identical to the protected copies, {len(fixed['runner_files'])} runner files and "
        f"{len(fixed['runner_trees'])} trees as fixed.json records, endpoint serves {fixed['endpoint']['root']} "
        f"at max_model_len {fixed['endpoint']['max_model_len']}")


def run_dev(workspace, fast):
    """bench/run_dev.py streamed line by line; (exit code, its lines). Its metric line is held back
    and re-printed last under the right name."""
    command = [sys.executable, "bench/run_dev.py"] + (["--fast"] if fast else [])
    process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for raw in process.stdout:
        line = raw.rstrip("\n")
        lines.append(line)
        if not line.startswith(f"{METRIC}:"):
            say(line)
    process.wait()
    return process.returncode, lines


def flips(workspace, references):
    """The per-conversation comparison with each recorded reference run: gained, lost, flips, net."""
    outcomes = {}
    with open(workspace / "results" / "episodes.jsonl", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            outcomes[row["id"]] = outcomes.get(row["id"], False) or bool(row["passed"])
    comparison = {}
    for name, reference in references.items():
        per_task = reference["per_task"]
        gained = sorted(task for task, passed in outcomes.items() if passed and not per_task.get(task))
        lost = sorted(task for task, count in per_task.items() if count and not outcomes.get(task))
        comparison[name] = {"reference_passes": reference["passes_total"], "passes": sum(outcomes.values()),
                            "gained": gained, "lost": lost, "flips": len(gained) + len(lost), "net": len(gained) - len(lost)}
        say(f"against {reference['label']} ({reference['passes_total']} of {reference['tasks']}, recorded {reference['recorded']}): "
            f"gained {', '.join(gained) or 'none'}; lost {', '.join(lost) or 'none'}; "
            f"flips {len(gained) + len(lost)}, net {len(gained) - len(lost):+d}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="the five fast conversations once; a smoke check, no score recorded")
    parser.add_argument("--check", action="store_true", help="the integrity checks alone")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    workspace = workspace_root()
    fixed = json.loads(FIXED.read_text(encoding="utf-8"))
    endpoint = json.loads((PROTECTED / "endpoint.json").read_text(encoding="utf-8"))
    references = json.loads(REFERENCES.read_text(encoding="utf-8"))
    mode = "check" if args.check else "fast" if args.fast else "full"
    say(f"evaluate.py: mode {mode}, workspace {workspace}, started {started.isoformat(timespec='seconds')}")
    integrity(workspace, fixed, endpoint, 30 if args.check else endpoint["endpoint_wait_seconds"])
    if args.check:
        say("check: ok")
        return

    code, lines = run_dev(workspace, args.fast)
    if code != 0:
        fail(code, f"bench/run_dev.py exited {code}")
    metric_lines = [line for line in lines if line.startswith(f"{METRIC}:")]
    if not metric_lines:
        fail(2, f"bench/run_dev.py printed no {METRIC} line")
    score = float(metric_lines[-1].split(":", 1)[1])
    finished = datetime.now(timezone.utc)
    minutes = round((finished - started).total_seconds() / 60, 1)

    if args.fast:
        say(f"fast: the five fast conversations once in {minutes} min; a smoke check, not comparable with a full score; no result.json written")
        say(f"fast_{METRIC}: {score:.4f}")
        return

    comparison = flips(workspace, references)
    run = json.loads((workspace / "results" / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((workspace / "results" / "summary.json").read_text(encoding="utf-8"))["summary"]
    say(f"full: {summary['tasks']} conversations x {summary['trials']} trial, harness {run['harness']['version']}, "
        f"{minutes} min, finished {finished.isoformat(timespec='seconds')}")
    say(f"{METRIC}: {score:.4f}")
    RESULT.write_text(json.dumps({
        "evaluation_script_path": SCRIPT_PATH,
        "evaluation_output": "\n".join(printed),
        "score": score,
        "metric": METRIC,
        "mode": "full",
        "harness": run["harness"],
        "save_name": run["save_name"],
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "minutes": minutes,
        "summary": summary,
        "comparison": comparison,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
