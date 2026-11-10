"""Prepare one branch per unit of the manifest, then open one pull request per branch. Run by deploy.sh from
the workspace, which is never switched to another commit: branches are made in worktrees of their own.

    open_pull_requests.py prepare <manifest> <prepared.json> <base> <base commit> <branch prefix>
    open_pull_requests.py open <prepared.json> <opened.json> <repo> <remote url> <base> <label> <branch checks.json> [<verification report>]

The manifest names units and their proofs, nothing else: {"units": [{"unit": "<ticket>", "proof": "<path>"}]}.
A unit is named after the ticket of its proof. What it ships comes from that proof: `changed_files` are its files
and `needs` the one unit whose change its code cannot build without. Titles and bodies are written here, at
publication, by the pull-request-proof skill's renderer, with the verdict it computes from the evidence, the
verification report and the branch's own check; nothing typed into a file beforehand is published. A title
carries no tag: the body says in a sentence whether the change is verified or which points are open, and
the label itself is kept in <opened.json> for the deployment's summary.

prepare: for each unit, a branch <prefix>/<unit> holding the unit's files as they are in the workspace (a
file absent there is removed), in one commit. A unit that names another in `needs` starts from that unit's
branch, so it builds on its own, and its pull request is opened against that branch, so its diff shows the
unit alone; every other unit starts from the base commit. A file is taken whole, so a file two units both
changed can only go into one of them: it goes into the unit the others are stacked on, and both pull requests
say so; units that share a file and are not stacked that way are refused, and so is a unit left with no file
of its own (deliver such tickets as one). A path that is not one file, and a unit whose files do not differ
from where its branch starts, are refused.

open: every branch must have passed its own check (verify_branches.py) at the commit prepared here; a task
that pins no check able to run on a branch alone has none, and its pull requests carry that as an open point. Each branch
is pushed (that branch only, and only over a branch this deployment pushed before), its title and body
rendered, and its pull request opened or, on a second run, updated. A stacked pull request says which one it
is stacked on, and the one under a stack lists what is stacked on it; both say how to merge without losing
the stack (the host closes a pull request whose target branch is deleted from the command line, and moves it
onto the base branch only when it deletes the branch itself). <opened.json> keeps one entry per unit across
runs and is rewritten after every pull request, so a run that stops midway loses no record. Standard library
only; every git and gh failure stops the script with the command's own message.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HARNESS = pathlib.Path(os.environ["KAPSO_HARNESS_ROOT"]) if os.environ.get("KAPSO_HARNESS_ROOT") else pathlib.Path(__file__).resolve().parents[3]
RENDER = HARNESS / "skills" / "pull-request-proof" / "scripts" / "render_proof.py"
ENVIRONMENT = pathlib.Path("prepared/environment.json")
PUSHED = pathlib.Path("deploy/pull_requests/pushed.json")
COMMAND_LINE_DELETION = "`gh pr merge --delete-branch`, `git push --delete`"
SINGLE_TICKET = "change"


def run(command, check=True):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and completed.returncode != 0:
        raise SystemExit(f"{' '.join(command[:4])} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return completed


def git(*arguments, check=True):
    """One git command that reads every path as a literal file name, never as a pattern."""
    return run(["git", "--literal-pathspecs", *arguments], check=check)


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Written whole or not at all: a run that stops midway never leaves half a record."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_name(path.name + ".part")
    draft.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(draft, path)


def ticket_of(proof_file):
    """The ticket a proof file is about, by where it sits: the renderer and the evidence read it the same way."""
    path = pathlib.Path(proof_file)
    return SINGLE_TICKET if path.name == "proof.json" else path.stem


def units_of(manifest_path):
    """The manifest's units, each with what its proof says it ships, ordered so a unit follows the one it needs."""
    manifest = read_json(manifest_path)
    listed = manifest.get("units") if isinstance(manifest, dict) else None
    if not isinstance(listed, list) or not listed:
        raise SystemExit(f"{manifest_path} holds no `units` list")
    units = []
    for item in listed:
        name, proof_file = (item.get("unit"), item.get("proof")) if isinstance(item, dict) else (None, None)
        if not isinstance(name, str) or not name or not isinstance(proof_file, str) or not pathlib.Path(proof_file).is_file():
            raise SystemExit(f"every unit is {{\"unit\": \"<name>\", \"proof\": \"<an existing proof file>\"}}, not {item!r}")
        proof = read_json(proof_file)
        if ticket_of(proof_file) != name or proof.get("ticket") not in (None, name):
            raise SystemExit(f"unit {name}: {proof_file} is the proof of `{proof.get('ticket') or ticket_of(proof_file)}`; a unit is named "
                             "after the ticket whose proof it ships, so its label is the one that ticket earned")
        if proof.get("change_kind") == "no_change":
            raise SystemExit(f"unit {name}: a no_change outcome changes no file and ships no pull request; its evidence goes in the summary")
        files = proof.get("changed_files")
        if not isinstance(files, list) or not files or not all(isinstance(path, str) and path for path in files):
            raise SystemExit(f"unit {name}: its proof ({proof_file}) lists no `changed_files`")
        units.append({"unit": name, "proof": proof_file, "files": list(dict.fromkeys(files)), "needs": proof.get("needs") or None,
                      "summary": str((proof.get("root_cause") or {}).get("summary") or proof.get("summary") or "the change").strip()})
    names = [unit["unit"] for unit in units]
    if len(set(names)) != len(names):
        raise SystemExit("every unit needs its own name")
    for unit in units:
        if unit["needs"] is not None and (unit["needs"] not in names or unit["needs"] == unit["unit"]):
            raise SystemExit(f"unit {unit['unit']}: its proof's `needs` names one other unit of the manifest, not {unit['needs']!r}")
    placed, order = set(), []
    while len(order) < len(units):
        ready = [unit for unit in units if unit["unit"] not in placed and (unit["needs"] is None or unit["needs"] in placed)]
        if not ready:
            raise SystemExit("the units' `needs` form a cycle")
        for unit in ready:
            placed.add(unit["unit"])
            order.append(unit)
    return order


def assign_shared_files(units):
    """A file is taken whole from the workspace, so it ships in one unit. A file several units name goes into
    the one all the others are stacked on, directly or through another unit; each side records it, and both
    pull requests say so. Units that share a file and are not stacked that way are refused."""
    owners = {}
    for unit in units:
        for path in unit["files"]:
            owners.setdefault(path, []).append(unit["unit"])
    by_name = {unit["unit"]: unit for unit in units}

    def stacked_on(name):
        """Every unit a unit is stacked on, nearest first."""
        chain = []
        while by_name[name]["needs"] is not None:
            name = by_name[name]["needs"]
            chain.append(name)
        return chain

    for path, names in owners.items():
        if len(names) == 1:
            continue
        under = [name for name in names if all(name in stacked_on(other) for other in names if other != name)]
        if len(under) != 1:
            raise SystemExit(f"{path} is changed by {', '.join(names)}, and none of them is the one all the others are stacked on. A file "
                             "ships whole in one pull request: name the unit the others need in their proofs' `needs`, or deliver them as "
                             "one ticket")
        for name in names:
            if name != under[0]:
                by_name[name]["files"].remove(path)
                by_name[name].setdefault("shared", []).append({"file": path, "ships_in": under[0]})
                by_name[under[0]].setdefault("carries", []).append({"file": path, "of": name})
    for unit in units:
        if not unit["files"]:
            raise SystemExit(f"unit {unit['unit']}: every file it names also belongs to `{unit['needs']}` and ships there, so it has nothing of "
                             "its own to open. Two tickets whose whole change is one file are one review: deliver them as one ticket")


def prepare(manifest_path, prepared_path, base, base_commit, prefix):
    units = units_of(manifest_path)
    assign_shared_files(units)
    git("worktree", "prune")
    prepared = []
    with tempfile.TemporaryDirectory(prefix="pull-request-branches-") as root:
        for unit in units:
            name = unit["unit"]
            branch = f"{prefix}/{name}"
            parent_branch = f"{prefix}/{unit['needs']}" if unit["needs"] else None
            start = parent_branch or base_commit
            tree = pathlib.Path(root) / name
            git("worktree", "add", "-q", "-B", branch, str(tree), start)
            for path in unit["files"]:
                pure = pathlib.PurePosixPath(path)
                source, target = pathlib.Path(path), tree / path
                if pure.is_absolute() or ".." in pure.parts or path in (".", "") or source.is_symlink() or source.is_dir() or target.is_dir():
                    raise SystemExit(f"unit {name}: {path} is not one file inside the repository")
                if source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(source, target)
                elif target.is_file():
                    target.unlink()
                else:
                    raise SystemExit(f"unit {name}: {path} is a file neither in the workspace nor where the branch starts")
                git("-C", str(tree), "add", "--", path)
            if git("-C", str(tree), "diff", "--cached", "--quiet", check=False).returncode == 0:
                raise SystemExit(f"unit {name}: its files do not differ from {unit['needs'] or base}; nothing to open")
            git("-C", str(tree), "commit", "-q", "-m", f"{name}: {unit['summary']}"[:200])
            commit = git("-C", str(tree), "rev-parse", "HEAD").stdout.strip()
            git("worktree", "remove", "--force", str(tree))
            prepared.append({"unit": name, "branch": branch, "needs": unit["needs"], "parent_branch": parent_branch, "proof": unit["proof"],
                             "files": unit["files"], "shared": unit.get("shared", []), "carries": unit.get("carries", []), "commit": commit})
            print(f"prepared {branch}" + (f" on {parent_branch}" if parent_branch else f" on {base}"))
    write_json(prepared_path, {"base": base, "base_commit": base_commit, "branches": prepared})


def with_notes(notes, body):
    """The body under its notes, each note a paragraph of one quoted block."""
    return ("\n>\n".join(notes) + "\n\n" if notes else "") + body


def checked_alone(prepared, checks_path):
    """{unit: check} when every prepared branch passed its own check at the commit prepared; an empty mapping
    when the task pins no check that can run on a branch alone, and only then. Anything else stops the
    publication."""
    if not pathlib.Path(checks_path).is_file():
        raise SystemExit(f"no {checks_path}: every branch is checked on its own (verify_branches.py) before anything is pushed")
    report = read_json(checks_path)
    settings = read_json(ENVIRONMENT) if ENVIRONMENT.is_file() else {}
    if report.get("checked") is not True:
        if settings.get("acceptance_command") and "acceptance_at_a_time" in settings:
            raise SystemExit(f"{checks_path} says the branches were not checked, and {ENVIRONMENT} pins a check that runs on a branch "
                             "alone: run verify_branches.py; nothing was pushed")
        return {}
    checks = report.get("branches", {})
    for entry in prepared:
        check = checks.get(entry["unit"])
        if check is None or check.get("verdict") != "pass" or check.get("commit") != entry["commit"]:
            raise SystemExit(f"unit {entry['unit']}: its branch has no passing check at the commit prepared "
                             f"({(check or {}).get('verdict', 'not checked')}); nothing was pushed")
    return checks


def push(remote, branch, commit):
    """Push one branch, only when the remote has no branch of this name or holds a commit this deployment
    pushed there. The commit is recorded before it is pushed: a run that stops between the two finds its own
    branch on the next one, and no other deployment can have made that commit."""
    pushed = read_json(PUSHED) if PUSHED.is_file() else {}
    listed = run(["git", "ls-remote", remote, f"refs/heads/{branch}"]).stdout.split()
    on_remote = listed[0] if listed else ""
    if on_remote and on_remote not in pushed.get(branch, []):
        raise SystemExit(f"{branch} exists on the remote at {on_remote[:12]}, which this deployment did not push: not overwritten")
    pushed[branch] = pushed.get(branch, []) + [commit]
    write_json(PUSHED, pushed)
    run(["git", "push", "-q", f"--force-with-lease={branch}:{on_remote}", remote, f"{commit}:refs/heads/{branch}"])


def open_all(prepared_path, opened_path, repo, remote, base, label, checks_path, verification_path=None):
    prepared = read_json(prepared_path)["branches"]
    checks = checked_alone(prepared, checks_path)
    # The host moves a stacked pull request onto the base branch only when it deletes the merged branch itself.
    host_deletes = run(["gh", "api", f"repos/{repo}", "--jq", ".delete_branch_on_merge"]).stdout.strip() == "true"
    # One entry per unit, kept across runs: a unit an earlier run opened stays on record until this run reaches it.
    journal = {entry["unit"]: entry for entry in (read_json(opened_path).get("pull_requests", []) if pathlib.Path(opened_path).is_file() else [])}
    bodies = {}

    def record():
        ordered = [journal[entry["unit"]] for entry in prepared if entry["unit"] in journal]
        ordered += [entry for unit, entry in journal.items() if unit not in {item["unit"] for item in prepared}]
        write_json(opened_path, {"repo": repo, "base": base, "label": label, "host_deletes_merged_branches": host_deletes, "pull_requests": ordered})

    for entry in prepared:
        name, branch = entry["unit"], entry["branch"]
        check = checks.get(name)
        folder = pathlib.Path("deploy/pull_requests") / name
        folder.mkdir(parents=True, exist_ok=True)
        render = [sys.executable, str(RENDER), entry["proof"], "--title-out", str(folder / "title.txt"), "--body-out", str(folder / "body.md"),
                  "--label-out", str(folder / "label.json"), "--branch-check", check["verdict"] if check else "not_run"]
        run(render + (["--verification", verification_path] if verification_path and pathlib.Path(verification_path).is_file() else []))
        title = (folder / "title.txt").read_text(encoding="utf-8").strip().splitlines()[0]
        body = (folder / "body.md").read_text(encoding="utf-8")
        verdict = read_json(folder / "label.json")   # the label stays in the record; the pull request says it in a sentence
        notes = []
        if entry["needs"]:
            under = journal[entry["needs"]]["number"]
            after = (f"this repository deletes a merged branch itself, and GitHub then moves this pull request onto `{base}`" if host_deletes else
                     f"this repository keeps merged branches, so move this pull request onto `{base}` (Edit, base branch) once #{under} is merged")
            notes.append(f"> **Stacked on #{under} (`{entry['needs']}`).** This branch builds only with that change, so it starts from it and "
                         f"this pull request targets its branch. Merge #{under} first and leave its branch to GitHub: {after}. Deleting that "
                         f"branch from the command line ({COMMAND_LINE_DELETION}) closes this pull request.")
        for shared in entry["shared"]:
            notes.append(f"> `{shared['file']}` also carries this ticket's change and ships in #{journal[shared['ships_in']]['number']} "
                         f"(`{shared['ships_in']}`), since a file is delivered whole.")
        notes.append(f"> Acceptance check on this branch alone: **{check['verdict']}**" + (f" ({check['detail']})" if check.get("detail") else "") + "."
                     if check else "> This branch was not checked on its own: the task pins no acceptance check that can run on a branch alone.")
        full_body = folder / "body.opened.md"
        bodies[name] = (full_body, notes, body)
        full_body.write_text(with_notes(notes, body), encoding="utf-8")
        push(remote, branch, entry["commit"])
        target = entry["parent_branch"] or base
        # A second run of deploy.sh (after a branch was fixed) meets the pull requests the first one opened.
        existing = run(["gh", "pr", "view", branch, "--repo", repo, "--json", "url,state"], check=False)
        if existing.returncode == 0 and json.loads(existing.stdout)["state"] == "OPEN":
            url = json.loads(existing.stdout)["url"]
            run(["gh", "pr", "edit", url, "--title", title, "--body-file", str(full_body), "--base", target])
        else:
            created = run(["gh", "pr", "create", "--repo", repo, "--base", target, "--head", branch, "--title", title,
                           "--body-file", str(full_body), "--label", label])
            url = created.stdout.strip().splitlines()[-1]
        journal[name] = {"unit": name, "branch": branch, "commit": entry["commit"], "base": target, "needs": entry["needs"],
                         "number": int(url.rstrip("/").rsplit("/", 1)[-1]), "url": url, "title": title,
                         "label": verdict["label"], "open_points": verdict["open_points"],
                         "acceptance_on_branch": (check or {}).get("verdict", "not_run")}
        record()
        stated = "verified" if not verdict["open_points"] else f"{len(verdict['open_points'])} open point(s)"
        print(f"opened #{journal[name]['number']} for {name} against {target} ({stated}): {title}")
    # The pull request under a stack is the one its merger reads: it lists what is stacked on it and how to merge.
    for under in sorted({entry["needs"] for entry in prepared if entry["needs"]}):
        stacked = [journal[entry["unit"]]["number"] for entry in prepared if entry["needs"] == under]
        number = journal[under]["number"]
        after = (f"GitHub deletes the branch after the merge and moves them onto `{base}`" if host_deletes else
                 f"this repository keeps merged branches, so move each of them onto `{base}` (Edit, base branch) after the merge")
        note = (f"> **{len(stacked)} pull request(s) are stacked on this one: {', '.join(f'#{item}' for item in stacked)}.** Merge this one "
                f"first, with the merge button or `gh pr merge {number} --merge`; {after}. Never delete this branch from the command "
                f"line ({COMMAND_LINE_DELETION}): that closes every pull request stacked on it.")
        carried = [f"> `{item['file']}` also carries the change of #{journal[item['of']]['number']} (`{item['of']}`): a file is delivered "
                   "whole, so that part of it is reviewed and merged here."
                   for entry in prepared if entry["unit"] == under for item in entry["carries"]]
        full_body, notes, body = bodies[under]
        full_body.write_text(with_notes([note] + carried + notes, body), encoding="utf-8")
        run(["gh", "pr", "edit", str(number), "--repo", repo, "--body-file", str(full_body)])
        print(f"#{number} ({under}) carries {len(stacked)} stacked pull request(s): merge it first, and leave its branch to the host")
    record()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "open"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "prepare":
        if len(sys.argv) != 7:
            raise SystemExit(__doc__)
        prepare(*sys.argv[2:7])
    else:
        if len(sys.argv) not in (9, 10):
            raise SystemExit(__doc__)
        open_all(*sys.argv[2:])
