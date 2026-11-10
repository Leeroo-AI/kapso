"""Hand the sealed conversations over after a build has finished, for the check on what it delivered.

    python runner/handover.py <runner dir> <vault dir> <folder> [--reference-results <results.json> --reference-name fable-5.1]

Puts the sealed conversations' definitions back into the runner's registry beside the development ones and
writes <folder>/tasks.json, the list `bench/run_dev.py --tasks` takes. Run it only once the build is over:
from this moment the machine holds the sealed set.

With --reference-results, a run of another agent on the sealed conversations (a results file the runner
wrote, recorded where the vault lives) is converted with the repository's adapter into
<folder>/reference/<name>/: `episodes.jsonl`, `summary.json` and the transcripts, the shape of the
repository's own `bench/reference/`. Prints counts only, never a task id.
"""
import argparse
import importlib.util
import json
import pathlib
import shutil

DOMAIN = pathlib.Path("tau2-bench-2.0.0") / "data" / "tau2" / "domains" / "banking_knowledge"
ADAPTER = pathlib.Path(__file__).resolve().parents[1] / "repo" / "bench" / "adapter.py"


def transcript(simulation):
    """One conversation as it ran, with the tool calls: the form of the repository's reference transcripts."""
    reward = (simulation.get("reward_info") or {}).get("reward") or 0.0
    lines = [f"# {simulation['task_id']} · trial {simulation.get('trial', 0)} · "
             f"{'resolved' if reward >= 1.0 else 'not resolved'} · ended: {simulation.get('termination_reason')}", ""]
    for message in simulation.get("messages") or []:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "assistant":
            if content:
                lines += [f"**assistant:** {content}", ""]
            for call in message.get("tool_calls") or []:
                name = call.get("name") or (call.get("function") or {}).get("name")
                arguments = call.get("arguments") if call.get("arguments") is not None else (call.get("function") or {}).get("arguments")
                lines += [f"**assistant → {name}:** {json.dumps(arguments) if not isinstance(arguments, str) else arguments}", ""]
        elif role == "user":
            lines += [f"**user:** {content}", ""]
        elif role == "tool":
            lines += [f"**tool:** {content}", ""]
    return "\n".join(lines) + "\n"


def write_reference(folder, name, results_path, sealed):
    """The recorded run of another agent on the sealed conversations, converted with the repository's adapter."""
    spec = importlib.util.spec_from_file_location("bench_adapter", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    data = json.loads(pathlib.Path(results_path).read_text(encoding="utf-8"))
    simulations = [simulation for simulation in data["simulations"]
                   if simulation.get("task_id") in sealed and simulation.get("trial", 0) == 0]
    if len(simulations) != len(sealed):
        raise SystemExit(f"{results_path} holds a first trial of {len(simulations)} of the {len(sealed)} sealed conversations")
    reference = folder / "reference" / name
    (reference / "transcripts").mkdir(parents=True)
    filtered = reference / "filtered_results.json"
    filtered.write_text(json.dumps({**{key: value for key, value in data.items() if key != "simulations"},
                                    "simulations": simulations}), encoding="utf-8")
    rows = adapter.convert(filtered, set(sealed))
    filtered.unlink()
    (reference / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = adapter.summarize(rows, k=1)
    (reference / "summary.json").write_text(json.dumps(
        {"source": "episodes.jsonl", "against": None, "summary": summary, "comparison": None}, indent=2) + "\n", encoding="utf-8")
    for simulation in simulations:
        (reference / "transcripts" / f"{simulation['task_id']}.md").write_text(transcript(simulation), encoding="utf-8")
    return summary["passes_total"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runner_dir", type=pathlib.Path, help="where runner/install.sh put the runner")
    parser.add_argument("vault_dir", type=pathlib.Path, help="the vault runner/seal.py filled")
    parser.add_argument("folder", type=pathlib.Path, help="the hand-over folder to write, beside the repository")
    parser.add_argument("--reference-results", type=pathlib.Path, default=None,
                        help="a results file of the runner: another agent's run on the sealed conversations")
    parser.add_argument("--reference-name", default="reference", help="the name of that run's directory")
    args = parser.parse_args()
    runner, vault, folder = args.runner_dir.resolve(), args.vault_dir.resolve(), args.folder.resolve()
    split = json.loads((vault / "split.json").read_text(encoding="utf-8"))
    development, sealed = split["dev"], split["sealed"]
    everything = json.loads((vault / "banking_knowledge.tasks.json").read_text(encoding="utf-8"))
    registry_path = runner / DOMAIN / "tasks.json"
    present = {task["id"] for task in json.loads(registry_path.read_text(encoding="utf-8"))}
    if not set(development) <= present:
        raise SystemExit("the runner's registry does not hold the development conversations; is this the sealed runner?")
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f"{folder} is not empty")

    wanted = set(development) | set(sealed)
    registry_path.write_text(json.dumps([task for task in everything if task["id"] in wanted], indent=2) + "\n", encoding="utf-8")
    for identifier in sealed:
        shutil.copy(vault / "banking_knowledge.tasks" / f"{identifier}.json", runner / DOMAIN / "tasks" / f"{identifier}.json")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tasks.json").write_text(json.dumps({"conversations": sorted(sealed)}, indent=2) + "\n", encoding="utf-8")

    registry = {task["id"] for task in json.loads(registry_path.read_text(encoding="utf-8"))}
    missing_files = [identifier for identifier in sealed if not (runner / DOMAIN / "tasks" / f"{identifier}.json").is_file()]
    print(f"the runner's registry now holds {len(registry)} task definitions: {len(development)} development, {len(sealed)} sealed")
    print(f"sealed conversations missing from the registry: {len(set(sealed) - registry)}; missing a per-task file: {len(missing_files)}")
    print(f"{folder / 'tasks.json'} lists {len(sealed)} conversations")
    if args.reference_results is not None:
        passed = write_reference(folder, args.reference_name, args.reference_results, sealed)
        print(f"reference/{args.reference_name}: {passed} of {len(sealed)} resolved in the recorded run")
    if set(sealed) - registry or missing_files:
        raise SystemExit("the hand-over is incomplete")
    print("handed over; from the repository: python bench/run_dev.py --tasks <folder>/tasks.json")


if __name__ == "__main__":
    main()
