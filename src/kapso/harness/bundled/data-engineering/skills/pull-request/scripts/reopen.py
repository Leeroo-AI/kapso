"""Reopen the closed pull requests of this deployment and put each on the base it belongs on. Run by start.sh
from the workspace.

    reopen.py <opened.json> <repo> <base>

A pull request the deployment's stop closed is reopened. A stacked one the host closed because the branch
under it was deleted from the command line can be neither reopened nor moved while that branch is gone: when
the pull request it was stacked on is merged, the branch is put back at the commit this deployment recorded,
the pull request reopened, and the branch removed again once no open pull request of the repository targets
it. Whether it was reopened, was open all along or needed the branch back, a stacked pull request ends on the
branch of the nearest unit under it that is not merged yet, or on the base branch when every unit under it is
merged: moved any further down, its diff would show changes nobody merged. A merged pull request stays merged.
Exits 1 naming what could not be reopened. Standard library and `gh` only.
"""
import json
import pathlib
import subprocess
import sys


def gh(*arguments, check=True):
    completed = subprocess.run(["gh", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and completed.returncode != 0:
        raise SystemExit(f"gh {' '.join(arguments[:3])} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return completed


def view(number, repo, field):
    return gh("pr", "view", str(number), "--repo", repo, "--json", field, "--jq", f".{field}").stdout.strip()


def belongs_on(entry, by_unit, state, base):
    """The branch of the nearest unit under this one whose pull request is not merged, or the base branch."""
    under = by_unit.get(entry.get("needs"))
    while under is not None and state[under["number"]] == "MERGED":
        under = by_unit.get(under.get("needs"))
    return under["branch"] if under is not None else base


def main(opened_path, repo, base):
    journal = json.loads(pathlib.Path(opened_path).read_text(encoding="utf-8"))
    entries = journal["pull_requests"]
    by_unit = {entry["unit"]: entry for entry in entries}
    state = {entry["number"]: view(entry["number"], repo, "state") for entry in entries}
    restored, failed = {}, []
    for entry in entries:   # a unit comes after the one it needs, so what it is stacked on is settled first
        number = entry["number"]
        under = by_unit.get(entry.get("needs"))
        if state[number] == "CLOSED":
            reopened = gh("pr", "reopen", str(number), "--repo", repo, check=False).returncode == 0
            if not reopened and under is not None and state[under["number"]] == "MERGED":
                if gh("api", f"repos/{repo}/git/ref/heads/{under['branch']}", check=False).returncode != 0:
                    gh("api", f"repos/{repo}/git/refs", "-f", f"ref=refs/heads/{under['branch']}", "-f", f"sha={under['commit']}")
                    restored[under["branch"]] = under["commit"]
                reopened = gh("pr", "reopen", str(number), "--repo", repo, check=False).returncode == 0
            if not reopened:
                failed.append(number)
                continue
            state[number] = "OPEN"
            print(f"reopened #{number}")
        else:
            print(f"#{number} is {state[number]}; not reopened")
        target = belongs_on(entry, by_unit, state, base)
        if state[number] == "OPEN" and view(number, repo, "baseRefName") != target:
            gh("pr", "edit", str(number), "--repo", repo, "--base", target)
            print(f"moved #{number} onto {target}: what it was stacked on is merged")
        if state[number] == "OPEN" and entry.get("base") != target:
            # The record follows the pull request: its diff is still the unit alone, now against this base.
            entry["base"] = target
            pathlib.Path(opened_path).write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    for branch, commit in restored.items():
        # Any open pull request of the repository may target the branch, not only this deployment's.
        targeting = gh("pr", "list", "--repo", repo, "--state", "open", "--base", branch, "--json", "number", "--jq", ".[].number").stdout.split()
        tip = gh("api", f"repos/{repo}/git/ref/heads/{branch}", "--jq", ".object.sha", check=False).stdout.strip()
        if not targeting and tip == commit:
            gh("api", "-X", "DELETE", f"repos/{repo}/git/refs/heads/{branch}")
            print(f"removed {branch} again: no open pull request targets it")
        else:
            print(f"left {branch} in place: " + (f"#{targeting[0]} targets it" if targeting else "it moved since it was put back"))
    if failed:
        raise SystemExit(f"could not reopen {', '.join('#' + str(number) for number in failed)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])
