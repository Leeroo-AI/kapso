"""Write the files of this example that name your machine, from the place the runner was installed.

    python runner/configure.py <runner dir> [--chat-url http://localhost:18000/v1] [--embedding-url http://localhost:18001/v1]
                               [--customer-model azure/gpt-5.2] [--customer-env ~/.secrets/customer.env]

Run it after runner/install.sh (and after runner/seal.py, when a held-out set is kept). It writes:

  <runner dir>/runner.json                    the settings of run_conversations.py
  repo/bench/endpoint.json                    where the model endpoint and the runner are
  repo/bench/protection.json                  the runner's private files, by absolute path
  evaluation/protected/endpoint.json, protection.json    the evaluation's copies of those two, byte for byte
  evaluation/fixed.json                       the hashes of the runner's files and the endpoint's identity
  data/baseline-qwen-harness-0.1.0/           the pass-through's recorded run, as the build receives it

and compares your runner with the hashes of the one this example was made with (runner/recorded_hashes.json),
so you know whether you are measuring under the same runner. Nothing here opens a task definition.
"""
import argparse
import importlib.util
import json
import pathlib
import shutil
from datetime import datetime, timezone

EXAMPLE = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = pathlib.Path(__file__).resolve().parent / "templates"
BENCH, PROTECTED = EXAMPLE / "repo" / "bench", EXAMPLE / "evaluation" / "protected"
DOMAIN = pathlib.Path("data") / "tau2" / "domains" / "banking_knowledge"
# The four bench files the evaluation holds copies of beside the two written here.
COPIED = ("run_dev.py", "adapter.py", "dev_tasks.json", "fixed_conditions.json")
# The endpoint's identity under the fixed conditions (repo/bench/fixed_conditions.json).
MODEL, MODEL_ROOT, MAX_MODEL_LEN = "qwen", "Qwen/Qwen3.8-27B-FP8", 262144
HASHED_FILES = ("pyproject.toml", "uv.lock", str(DOMAIN / "db.json"))
HASHED_TREES = (("src/tau2", "*.py"), (str(DOMAIN / "prompts"), "*"), (str(DOMAIN / "documents"), "*"))
EXCLUDED = ["__pycache__"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runner_dir", type=pathlib.Path, help="where runner/install.sh put the runner")
    parser.add_argument("--chat-url", default="http://localhost:18000/v1", help="the model endpoint's base URL")
    parser.add_argument("--embedding-url", default="http://localhost:18001/v1", help="the embedding endpoint's base URL")
    parser.add_argument("--customer-model", default="azure/gpt-5.2", help="the simulated customer, as a LiteLLM model")
    parser.add_argument("--customer-env", default="~/.secrets/customer.env", help="the file holding its credentials")
    args = parser.parse_args()
    runner = args.runner_dir.resolve()
    source = runner / "tau2-bench-2.0.0"
    for needed in (source / "pyproject.toml", runner / "run_conversations.py"):
        if not needed.is_file():
            raise SystemExit(f"{needed} is missing; run runner/install.sh {runner} first")
    for name in COPIED:
        if (BENCH / name).read_bytes() != (PROTECTED / name).read_bytes():
            raise SystemExit(f"repo/bench/{name} differs from evaluation/protected/{name}; the evaluation would refuse to run")

    (runner / "runner.json").write_text(json.dumps({
        "chat_base_url": args.chat_url, "embedding_base_url": args.embedding_url,
        "agent_model": f"hosted_vllm/{MODEL}", "embedder_model": "qwen-embed",
        "customer_model": args.customer_model, "customer_env_file": args.customer_env,
    }, indent=2) + "\n", encoding="utf-8")
    for name in ("endpoint.json", "protection.json"):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        text = text.replace("@RUNNER_DIR@", str(runner)).replace("@CHAT_BASE_URL@", args.chat_url)
        json.loads(text)   # a path with a quote or a backslash in it would break the file: fail here, not in a run
        (BENCH / name).write_text(text, encoding="utf-8")
        (PROTECTED / name).write_text(text, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("example_evaluate", EXAMPLE / "evaluation" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)   # the evaluator's own hashing, so what is written here is what it checks
    files = {str(runner / name): evaluate.sha256_file(runner / name) for name in ("run_conversations.py", "runner.json")}
    files.update({str(source / relative): evaluate.sha256_file(source / relative) for relative in HASHED_FILES})
    trees = {}
    for relative, pattern in HASHED_TREES:
        count, digest = evaluate.tree_digest(source / relative, pattern, tuple(EXCLUDED))
        trees[str(source / relative)] = {"pattern": pattern, "exclude": EXCLUDED, "files": count, "sha256": digest}
    (EXAMPLE / "evaluation" / "fixed.json").write_text(json.dumps({
        "recorded": datetime.now(timezone.utc).date().isoformat(),
        "note": "The runner-side conditions a run depends on, hashed so the evaluation refuses to score under changed "
                "conditions. No protected path (bench/protection.json) is listed or read; the task definitions are never opened.",
        "runner_files": files,
        "runner_trees": trees,
        "endpoint": {"base_url": args.chat_url, "model": MODEL, "root": MODEL_ROOT, "max_model_len": MAX_MODEL_LEN},
    }, indent=2) + "\n", encoding="utf-8")

    baseline = EXAMPLE / "data" / "baseline-qwen-harness-0.1.0"
    if baseline.exists():
        shutil.rmtree(baseline)
    shutil.copytree(BENCH / "reference" / "qwen-harness-0.1.0", baseline)

    print(f"written: {runner / 'runner.json'}, repo/bench/endpoint.json and protection.json with the evaluation's copies, "
          "evaluation/fixed.json, data/baseline-qwen-harness-0.1.0/")
    recorded = json.loads((pathlib.Path(__file__).resolve().parent / "recorded_hashes.json").read_text(encoding="utf-8"))
    different = [relative for relative, digest in recorded["files"].items() if files[str(source / relative)] != digest]
    different += [relative for relative, spec_ in recorded["trees"].items()
                  if (trees[str(source / relative)]["files"], trees[str(source / relative)]["sha256"]) != (spec_["files"], spec_["sha256"])]
    compared = len(recorded["files"]) + len(recorded["trees"])
    if different:
        print(f"the runner differs from the one this example was made with in {len(different)} of {compared} hashed parts: {', '.join(different)}; "
              "your numbers are measured under another runner")
    else:
        print(f"the runner hashes as the one this example was made with in all {compared} parts (its lock file, its source, the domain's database, prompts and documents)")
    definitions = sum(1 for _ in (source / DOMAIN / "tasks").glob("*.json"))   # counted by file name, never opened
    development = len(json.loads((BENCH / "dev_tasks.json").read_text(encoding="utf-8"))["development"])
    print(f"the runner holds {definitions} per-task definition files; the development set is {development}"
          + ("" if definitions == development else ": to keep a held-out set, seal the rest before a build (runner/split.py, runner/seal.py)"))
    print("next: kapso doctor build --config repo/kapso-config.yaml, then the smoke check: (cd repo && python bench/run_dev.py --fast)")


if __name__ == "__main__":
    main()
