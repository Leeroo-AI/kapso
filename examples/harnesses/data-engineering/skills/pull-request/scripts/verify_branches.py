"""Run the task's acceptance check on every prepared branch, each on its own, before anything is pushed.

    verify_branches.py <prepared.json> <branch checks.json> [<environment.json>]

A pull request that passes only together with another one is not what its title says. For every branch
prepare made, this checks the branch out in a worktree of its own and runs the acceptance command the setup
pinned in prepared/environment.json (`acceptance_command`, with `{ticket}` for the unit's name). The branch
passes when the command exits 0 and, where the setup pinned `acceptance_pass_pattern`, its output matches
it (and not `acceptance_fail_pattern`, when one is pinned); both are regular expressions with `{ticket}`
filled in, and without a pass pattern the exit status alone is the verdict. One
branch runs at a time unless the setup pinned `acceptance_at_a_time` (the task says its check may run several
at once, each on its own state). Writes one entry per unit: the verdict (`pass`, `fail`, or `not_run` when
its worktree could not be made), the last line of the command's output and the commit checked. Exit status 1
unless every branch passed, so deploy.sh stops before opening: a failing unit needs another unit's change
(`needs` in its proof) or is not ready. A branch can be checked alone only by a check that builds a state of
its own from the tree it is run in, which the setup declares by pinning `acceptance_at_a_time`: the command
runs in a checkout that holds the repository's tracked files and nothing else, and whatever it needs beyond
them (a profile, installed packages, its own state of the warehouse) it establishes itself. A task with no
acceptance command, or with one that reads the build namespace, cannot be checked this way: the report says
so (`checked: false`) and every pull request then carries that as an open point. Standard library only.
"""
import concurrent.futures
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

prepared = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["branches"]
out = pathlib.Path(sys.argv[2])
environment_file = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "prepared/environment.json")
settings = json.loads(environment_file.read_text(encoding="utf-8")) if environment_file.is_file() else {}
command_template, pass_template = settings.get("acceptance_command"), settings.get("acceptance_pass_pattern")
fail_template = settings.get("acceptance_fail_pattern")
at_a_time = settings.get("acceptance_at_a_time", 1)

if not command_template or "acceptance_at_a_time" not in settings:
    why = (f"no acceptance_command in {environment_file}" if not command_template else
           f"the acceptance check reads the build namespace (no acceptance_at_a_time in {environment_file}), which holds no branch's build")
    out.write_text(json.dumps({"checked": False, "why": why, "branches": {}}, indent=2) + "\n", encoding="utf-8")
    print(f"the branches cannot be checked on their own ({why}): every pull request carries that as an open point")
    sys.exit(0)


def check(entry, root):
    tree = pathlib.Path(root) / entry["unit"]
    added = subprocess.run(["git", "worktree", "add", "-q", "--detach", str(tree), entry["commit"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if added.returncode != 0:
        return entry["unit"], {"verdict": "not_run", "detail": f"no worktree: {added.stderr.strip()[-160:]}", "commit": entry["commit"]}
    completed = subprocess.run(shlex.split(command_template.replace("{ticket}", entry["unit"])), cwd=tree, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = completed.stdout or ""

    def matches(template):
        return re.search(template.replace("{ticket}", re.escape(entry["unit"])), output, re.MULTILINE) is not None

    passed = completed.returncode == 0 and (not pass_template or matches(pass_template)) and not (fail_template and matches(fail_template))
    last = [line.strip() for line in output.splitlines() if line.strip()]
    subprocess.run(["git", "worktree", "remove", "--force", str(tree)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return entry["unit"], {"verdict": "pass" if passed else "fail", "detail": (last[-1] if last else "no output")[:160], "commit": entry["commit"]}


with tempfile.TemporaryDirectory(prefix="branch-checks-") as root:
    with concurrent.futures.ThreadPoolExecutor(max_workers=at_a_time if isinstance(at_a_time, int) and at_a_time > 0 else 1) as executor:
        results = dict(executor.map(lambda entry: check(entry, root), prepared))
subprocess.run(["git", "worktree", "prune"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)   # a checkout a failed check left behind
out.write_text(json.dumps({"checked": True, "branches": results}, indent=2) + "\n", encoding="utf-8")
not_passing = sorted(unit for unit, result in results.items() if result["verdict"] != "pass")
for unit, result in results.items():
    print(f"  {result['verdict']:<8} {unit}  {result.get('detail', '')}")
if not_passing:
    print(f"{len(not_passing)} branch(es) do not pass the acceptance check on their own: {', '.join(not_passing)}. Name the unit each needs "
          "(`needs` in its proof) when another unit's change is what it lacks, then run deploy.sh again.")
    sys.exit(1)
