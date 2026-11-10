"""Run conversations of the banking knowledge domain through the conversation runner (τ²-bench 2.0).

    python run_conversations.py <save name> <retrieval config> <trials> <task id> [<task id> ...] [-- extra runner arguments]

The agent and the embeddings are the locally served models; the simulated customer is the model `runner.json`
names, called through LiteLLM with the credentials of the customer env file. `bench/run_dev.py` calls this
script and appends `--agent-llm-args` with the harness service's address, so the runner talks to the harness and
the harness to the model. A reference run of another agent is this script with `-- --agent-llm <a LiteLLM model>`.

`runner.json` beside this file holds the settings (`runner/configure.py` writes it): the two endpoints, the
model names, and the path of the customer env file. That file holds `NAME=value` lines in LiteLLM's own
variable names for the customer's provider (for Azure: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION). It is
read by path and nothing of it is printed.
"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TAU2 = HERE / "tau2-bench-2.0.0"
SETTINGS = ("chat_base_url", "embedding_base_url", "agent_model", "embedder_model", "customer_model", "customer_env_file")


def credentials(path):
    """The NAME=value lines of the customer env file; blank lines and comments skipped, quotes around a value dropped."""
    pairs = {}
    for line in pathlib.Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        pairs[name.strip()] = value.strip().strip('"').strip("'")
    return pairs


def runner_command(config, save_name, retrieval, trials, task_ids, extra):
    """The runner's command line: the banking knowledge domain, the retrieval configuration, the locally served
    agent and embedder, the simulated customer at temperature 0. Arguments after `--` come last, so a later
    `--agent-llm-args` (the harness's address) wins over the one given here."""
    return ["uv", "run", "tau2", "run", "--benchmark", "none", "--domain", "banking_knowledge",
            "--retrieval-config", retrieval,
            "--retrieval-config-kwargs", json.dumps({"embedder_model": config["embedder_model"]}),
            "--agent-llm", config["agent_model"],
            "--agent-llm-args", json.dumps({"api_base": config["chat_base_url"], "temperature": 0.0}),
            "--user-llm", config["customer_model"], "--user-llm-args", json.dumps({"temperature": 0.0}),
            "--num-trials", trials, "--task-ids", *task_ids, "--save-to", save_name, *extra]


def main():
    args = sys.argv[1:]
    extra = args[args.index("--") + 1:] if "--" in args else []
    args = args[:args.index("--")] if "--" in args else args
    if len(args) < 4:
        raise SystemExit("usage: run_conversations.py <save name> <retrieval config> <trials> <task id> [<task id> ...] "
                         "[-- extra runner arguments]")
    config = json.loads((HERE / "runner.json").read_text(encoding="utf-8"))
    missing = [name for name in SETTINGS if not config.get(name)]
    if missing:
        raise SystemExit(f"runner.json lacks {', '.join(missing)}; run runner/configure.py")
    if not (TAU2 / "pyproject.toml").is_file():
        raise SystemExit(f"the runner's source is not at {TAU2}; run runner/install.sh")
    save_name, retrieval, trials, task_ids = args[0], args[1], args[2], args[3:]
    env = {**os.environ, **credentials(config["customer_env_file"]),
           # The sandbox runtime of the runner's shell tool is installed under ~/.local.
           "PATH": f"{pathlib.Path.home() / '.local' / 'bin'}{os.pathsep}{os.environ['PATH']}",
           "HOSTED_VLLM_API_BASE": config["chat_base_url"],
           # The runner's embedder is the OpenAI SDK with no base URL of its own, so the SDK's own variables
           # point it at the local embedding server. A simulated customer therefore cannot use the plain OpenAI
           # route; reach it through another LiteLLM provider (Azure, OpenRouter).
           "OPENAI_BASE_URL": config["embedding_base_url"], "OPENAI_API_KEY": "none"}
    print("running the runner with", retrieval, "on", len(task_ids), "tasks x", trials, "trials", flush=True)
    sys.exit(subprocess.run(runner_command(config, save_name, retrieval, trials, task_ids, extra), cwd=TAU2, env=env).returncode)


if __name__ == "__main__":
    main()
