"""Seal the runner before a build: leave only the development conversations' definitions in its directory and
move everything else that defines a task into a vault.

    python runner/seal.py <runner dir> <vault dir>

Needs the split of runner/split.py. Moved to the vault: the split itself, the full task registry, the
per-task files of every task outside the development set, the voice variant of the registry, the domain's
task-subset list, and the places of the runner's source that restate task definitions (the translated task
sets and their tests, the published submissions, the release notes). The registry left behind holds the
development tasks alone, so the runner cannot run a sealed conversation and nothing on disk describes one.

Move the vault off the machine the build runs on, or at least out of the reach of the account it runs as:
on one machine with one account, what keeps a session away from it is the brief's rule and the harness's
auditors, not the file system. Prints counts and file names, never a task id.
"""
import argparse
import json
import pathlib
import re
import shutil

DOMAIN = pathlib.Path("data") / "tau2" / "domains" / "banking_knowledge"
# Places outside the domain's own directory that restate task definitions or cite task ids.
ELSEWHERE = ("data/tau2/multilingual", "data/tau2/multilingual_extensions", "tests/test_multilingual", "web",
             "CHANGELOG.md", "RELEASE_NOTES.md", "github-release-body.md")
SKIPPED_IN_SCAN = (".venv", "__pycache__")


def move(source, vault, relative):
    destination = vault / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runner_dir", type=pathlib.Path, help="where runner/install.sh put the runner")
    parser.add_argument("vault_dir", type=pathlib.Path, help="an empty or new directory outside the runner and the repository")
    args = parser.parse_args()
    runner, vault = args.runner_dir.resolve(), args.vault_dir.resolve()
    source = runner / "tau2-bench-2.0.0"
    split_path = runner / "split.json"
    if not split_path.is_file():
        raise SystemExit(f"no split at {split_path}; run runner/split.py first")
    if vault == runner or runner in vault.parents:
        raise SystemExit("the vault must be outside the runner's directory")
    if vault.exists() and any(vault.iterdir()):
        raise SystemExit(f"{vault} is not empty: a vault is filled once")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development, sealed = set(split["dev"]), set(split["sealed"])
    registry_path = source / DOMAIN / "tasks.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    known = {task["id"] for task in registry}
    if not (development | sealed) <= known or development & sealed:
        raise SystemExit("the split does not fit the registry: is this the full registry, sealed once already?")
    vault.mkdir(parents=True, exist_ok=True)

    shutil.copy(registry_path, vault / "banking_knowledge.tasks.json")
    kept = [task for task in registry if task["id"] in development]
    registry_path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    moved_files = 0
    for task_file in sorted((source / DOMAIN / "tasks").glob("*.json")):
        if task_file.stem not in development:
            move(task_file, vault, pathlib.Path("banking_knowledge.tasks") / task_file.name)
            moved_files += 1
    moved_places = []
    voice = source / DOMAIN / "tasks_voice.json"
    if voice.is_file():
        move(voice, vault, pathlib.Path("elsewhere") / DOMAIN / "tasks_voice.json")
        moved_places.append(str(DOMAIN / "tasks_voice.json"))
    for subset in sorted((source / "data" / "tau2" / "task_subsets").glob("banking_knowledge*.json")):
        move(subset, vault, pathlib.Path("elsewhere") / subset.relative_to(source))
        moved_places.append(str(subset.relative_to(source)))
    for relative in ELSEWHERE:
        if (source / relative).exists():
            move(source / relative, vault, pathlib.Path("elsewhere") / relative)
            moved_places.append(relative)
    move(split_path, vault, "split.json")

    # What still names a sealed conversation: file names and counts only.
    pattern = re.compile("(?:" + "|".join(re.escape(identifier) for identifier in sorted(sealed)) + r")(?!\d)")
    remaining = {}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in SKIPPED_IN_SCAN for part in relative.parts):
            continue
        found = len(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
        if found:
            remaining[str(relative)] = found
    print(f"registry cut to {len(kept)} development task definitions (it held {len(registry)}); "
          f"{moved_files} per-task files and {len(moved_places)} other places moved to {vault}")
    for place in moved_places:
        print(f"  moved: {place}")
    if remaining:
        print(f"{len(remaining)} file(s) of the runner still mention a sealed id. An id alone defines nothing, but a saved "
              "run under data/simulations holds whole conversations and belongs in the vault. Read each and decide:")
        for relative, found in sorted(remaining.items()):
            print(f"  {found:3d}  {re.sub(r'task_[0-9]+', 'task_<id>', relative)}")
    else:
        print("no file of the runner mentions a sealed id")
    print(f"sealed: {len(sealed)} conversations are defined in the vault only; move {vault} off this machine before a build")


if __name__ == "__main__":
    main()
