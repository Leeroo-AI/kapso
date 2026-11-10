"""Run the development conversations through the harness and score them.

    python bench/run_dev.py [--fast]

Run from the repository root, where harness/ is. It starts the harness service, runs the
development conversations through it against the fixed model endpoint and simulated customer,
converts the runner's output with the adapter to results/episodes.jsonl, writes
results/summary.json and results/run.json, and prints the share of conversations resolved as
its last line:

    pass_1: 0.4750

Higher is better. Exit codes: 0 scored; 2 the model endpoint or the harness did not answer, the
runner failed, or the run is incomplete; 3 the repository has no harness/serve.py.
--fast runs the five fast conversations once: a smoke check whose number is not comparable with a
full score. --tasks <file> runs the conversations the file lists under `conversations` once, into
results/heldout/ (the held-out conversations, handed over after a build). endpoint.json beside this script names the endpoint, the runner and the trials;
dev_tasks.json lists the development conversations. A full run takes about an hour and a half
at six conversations at a time. The runner's own output stays in its own directory, which holds
the task definitions and their expected outcomes and never enters this repository; the
harness's and the runner's logs for each run land under bench/runs/.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import convert, summarize  # noqa: E402

SERVICE_MODULE = "harness.serve"


def fail(code, reason):
    print(f"evaluation failed: {reason}")
    sys.exit(code)


def fetch_json(url, timeout):
    """The JSON a URL answers, or None when it does not answer or answers an error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:
            return json.loads(reply.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def wait_for_endpoint(config):
    """The model endpoint lists the model, within the configured wait."""
    deadline = time.monotonic() + config["endpoint_wait_seconds"]
    while time.monotonic() < deadline:
        listing = fetch_json(config["chat_base_url"] + "/models", 10)
        if listing and any(item.get("id") == config["model"] for item in listing.get("data", [])):
            return
        time.sleep(config["endpoint_poll_seconds"])
    fail(2, f"the model endpoint {config['chat_base_url']} did not list `{config['model']}` within {config['endpoint_wait_seconds']} s")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def start_service(workspace, port, upstream, log_path, wait_seconds):
    """The harness service from the workspace, healthy; the process and its health document."""
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", SERVICE_MODULE, "--port", str(port), "--upstream", upstream],
                               cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + wait_seconds
    health = None
    while time.monotonic() < deadline and process.poll() is None:
        health = fetch_json(f"http://127.0.0.1:{port}/health", 15)
        if health and health.get("status") == "ok":
            return process, health
        time.sleep(1)
    stop_service(process)
    log.close()
    tail = "\n".join(Path(log_path).read_text(encoding="utf-8").splitlines()[-12:])
    fail(2, f"the harness service did not answer /health within {wait_seconds} s (last health: {health}):\n{tail}")


def stop_service(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_runner(config, save_name, task_ids, port, log_path):
    """The runner on the tasks through the service; exit 2 on a non-zero exit."""
    agent_args = json.dumps({"api_base": f"http://127.0.0.1:{port}/v1", "temperature": 0.0})
    command = [sys.executable, "-I", config["runner"], save_name, config["retrieval_config"], str(config["trials"]),
               *task_ids, "--", "--agent-llm-args", agent_args, "--max-concurrency", str(config["max_concurrency"]),
               "--auto-resume"]
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = "\n".join(Path(log_path).read_text(encoding="utf-8").splitlines()[-15:])
        fail(2, f"the runner exited {completed.returncode}:\n{tail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="the five fast tasks once; not comparable with a full score")
    parser.add_argument("--tasks", type=Path, default=None,
                        help="run the conversations this JSON file lists under `conversations` once, into results/heldout/")
    args = parser.parse_args()
    workspace = Path.cwd()   # the repository root
    here = Path(__file__).resolve().parent
    if not (workspace / "harness" / "serve.py").is_file():
        fail(3, "harness/serve.py is missing from the repository root")
    config = json.loads((here / "endpoint.json").read_text(encoding="utf-8"))
    tasks = json.loads((here / "dev_tasks.json").read_text(encoding="utf-8"))
    task_ids = (json.loads(args.tasks.read_text(encoding="utf-8"))["conversations"] if args.tasks
                else tasks["fast"] if args.fast else tasks["development"])
    started = datetime.now(timezone.utc)
    save_name = f"greywell_{'heldout' if args.tasks else 'fast' if args.fast else 'full'}_{started.strftime('%Y%m%dT%H%M%SZ')}"
    runs = here / "runs" / save_name
    runs.mkdir(parents=True)

    wait_for_endpoint(config)
    port = free_port()
    process, health = start_service(workspace, port, config["chat_base_url"], runs / "harness.log", config["service_wait_seconds"])
    print(f"harness {health['harness']['version']} on port {port}, upstream {config['chat_base_url']}; "
          f"{len(task_ids)} tasks x {config['trials']} trial(s), save {save_name}", flush=True)
    try:
        run_runner(config, save_name, task_ids, port, runs / "runner.log")
    finally:
        stop_service(process)

    results_path = Path(config["simulations_dir"]) / save_name / "results.json"
    if not results_path.is_file():
        fail(2, f"the runner left no results file at {results_path}")
    rows = convert(results_path, set(task_ids))
    expected = {(task, trial) for task in task_ids for trial in range(config["trials"])}
    present = {(row["id"], row["trial"]) for row in rows}
    if present != expected:
        fail(2, f"the run is incomplete: {len(present)} of {len(expected)} episodes present; "
                f"missing {sorted(expected - present)[:10]}")
    summary = summarize(rows, k=config["trials"])

    results_dir = workspace / "results" / "heldout" if args.tasks else workspace / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (results_dir / "summary.json").write_text(json.dumps(
        {"source": "results/episodes.jsonl", "against": None, "summary": summary, "comparison": None}, indent=2) + "\n",
        encoding="utf-8")
    (results_dir / "run.json").write_text(json.dumps({
        "save_name": save_name, "fast": args.fast, "tasks": task_ids, "trials": config["trials"],
        "harness": health["harness"], "upstream": config["chat_base_url"], "upstream_models": health["upstream_models"],
        "retrieval_config": config["retrieval_config"], "max_concurrency": config["max_concurrency"],
        "runner_output": str(results_path), "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    lines = [f"tasks {summary['tasks']}  trials {summary['trials']}  episodes {summary['episodes']}  errors {summary['errors']}",
             "per task: " + ", ".join(f"{task}={count}/{summary['trials']}" for task, count in sorted(summary["per_task"].items()))]
    if args.fast:
        lines.append("fast: the five fast tasks once; not comparable with a full score")
    if args.tasks:
        lines.append(f"held-out: {len(task_ids)} conversations once, listed by {args.tasks}")
    if summary["pass_k"] is not None and summary["trials"] > 1:
        lines.append(f"pass_{summary['k']}: {summary['pass_k']:.4f}")
    lines.append(f"pass_1: {summary['pass_1']:.4f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
