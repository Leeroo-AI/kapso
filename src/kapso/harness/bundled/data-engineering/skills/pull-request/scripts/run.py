"""One call of the pull-request deployment: the state of every pull request it opened.

Reads the call's inputs as one JSON document on stdin (none are used) and writes one JSON document on stdout:
the repository, the base branch, and per pull request its number, unit, state, mergeability, review decision,
checks summary and URL. The pull requests are the ones named in KAPSO_VALUE_PR_NUMBERS, or in
deploy/pull_requests/opened.json when the engine has not recorded values yet. Standard library and `gh` only.
"""
import json
import os
import pathlib
import subprocess
import sys

OPENED = pathlib.Path("deploy/pull_requests/opened.json")

json.load(sys.stdin)
opened = json.loads(OPENED.read_text(encoding="utf-8")) if OPENED.is_file() else {}
units = {entry["number"]: entry["unit"] for entry in opened.get("pull_requests", [])}
repo = os.environ.get("KAPSO_VALUE_REPO") or opened.get("repo")
recorded = os.environ.get("KAPSO_VALUE_PR_NUMBERS")
numbers = [int(number) for number in recorded.split(",")] if recorded else sorted(units)
if not repo or not numbers:
    sys.exit("this deployment has no recorded pull requests: neither KAPSO_VALUE_PR_NUMBERS nor deploy/pull_requests/opened.json names any")

pull_requests = []
for number in numbers:
    completed = subprocess.run(["gh", "pr", "view", str(number), "--repo", repo, "--json",
                                "number,title,state,mergeable,reviewDecision,url,statusCheckRollup,baseRefName"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        sys.exit(f"gh pr view {number} failed: {completed.stderr.strip()}")
    view = json.loads(completed.stdout)
    checks = view.get("statusCheckRollup") or []
    pull_requests.append({
        "number": view["number"], "unit": units.get(view["number"]), "title": view["title"], "state": view["state"],
        "targets": view.get("baseRefName"),
        "mergeable": view.get("mergeable"), "review": view.get("reviewDecision") or None,
        "checks": {"total": len(checks), "failing": sum(1 for check in checks if (check.get("conclusion") or check.get("state") or "").upper()
                                                        in ("FAILURE", "TIMED_OUT", "CANCELLED", "ERROR", "ACTION_REQUIRED", "STARTUP_FAILURE"))},
        "url": view["url"],
    })
print(json.dumps({"repo": repo, "base": os.environ.get("KAPSO_VALUE_BASE") or opened.get("base"), "pull_requests": pull_requests}))
