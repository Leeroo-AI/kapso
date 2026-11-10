"""Release what is merged on `main` into the production warehouse, through the data platform.

    python checks/release.py request [--commit SHA] [--also MODEL ...]   stage a release and wait for it
    python checks/release.py status <release>                            where a release stands
    python checks/release.py publish <release>                           build the staged release in production
    python checks/release.py rollback <release>                          put production back to before it
    python checks/release.py production                                  the commit production stands at

Production takes only code that is on `main` of this repository on GitHub: the platform fetches the commit
itself, so merging a pull request is the approval and nothing in a working tree is ever released.

  request   The platform puts the staging database (RETAIL_DW_STAGING) back to production's state and builds
            there every model and snapshot the commit changed since the last release. Production is untouched.
            Staging is readable (dbt target `staging`): audit it before publishing.
  publish   Only for a staged release in which everything built: the platform backs up every relation the
            release replaces, then runs the same build on production (RETAIL_DW).
  rollback  The platform restores what the release replaced, drops what it created, and verifies production
            against its state before the release.

`request` builds the models and snapshots whose own files the commit changed. `--also` names models to build
with them although their files did not change (a macro they call changed, a model they read was fixed):
`<model>` in the warehouse project, `<project>/<model>` otherwise.

Add `--json` to print the release's record as one JSON object; a request the platform refused prints
`{"state": "refused", "error": ...}`. Exit codes: 0 the step reached its end state (staged, published, rolled
back); 1 it did not (the record says why); 2 the service could not be reached or refused.
"""
import argparse
import json
import sys
import time

from platform_service import SERVICE, call, require_service

END_STATES = {"request": ("staged", "stage_failed", "stage_error"),
              "publish": ("published", "publish_failed", "publish_error"),
              "rollback": ("rolled_back", "rollback_incomplete", "rollback_error")}
GOOD = {"request": "staged", "publish": "published", "rollback": "rolled_back"}

parser = argparse.ArgumentParser(description="Release what is merged on main into the production warehouse.")
parser.add_argument("step", choices=("request", "status", "publish", "rollback", "production"))
parser.add_argument("release", nargs="?")
parser.add_argument("--commit", default=None, help="a commit on main; default: its tip")
parser.add_argument("--also", nargs="*", default=[],
                    help="models to build although their files did not change: <model> (warehouse) or <project>/<model>")
parser.add_argument("--json", action="store_true")
arguments = parser.parse_args()


def refused(status, answer):
    if arguments.json:
        print(json.dumps({"release": arguments.release, "state": "refused", "error": answer.get("error"), "http_status": status}, indent=1))
    else:
        print(f"release: the platform service refused ({status}): {answer.get('error')}")
    sys.exit(2)


def show(record):
    if arguments.json:
        print(json.dumps(record, indent=1))
        return
    print(f"release {record['release']} at {record['commit'][:12]} ({record['source']}): {record['state']}")
    for label in ("staged", "published"):
        for entry in record.get(label) or []:
            where = f" ({entry['relation']})" if entry.get("relation") else ""
            print(f"  {label:<9} {entry['status']:<9} {entry['project']} {entry['resource']}{where}")
            for line in entry.get("errors") or []:
                print(f"            {line}")
    if record.get("backed_up"):
        print(f"  backed up before publishing: {', '.join(record['backed_up'])}")
    if record.get("rollback"):
        print(f"  rollback: {'verified against the state before the release' if record['rollback']['clean'] else record['rollback']['problems']}")
    if record.get("error"):
        print(f"  {record['error']}")


def wait_for(release, step):
    deadline = time.time() + SERVICE["release_timeout_seconds"]
    while True:
        status, record = call("GET", f"/releases/{release}")
        if status != 200:
            refused(status, record)
        if record["state"] in END_STATES[step]:
            show(record)
            sys.exit(0 if record["state"] == GOOD[step] else 1)
        if time.time() > deadline:
            record["error"] = f"the {step} did not finish within {SERVICE['release_timeout_seconds']} s (state {record['state']})"
            show(record)
            sys.exit(1)
        time.sleep(SERVICE["poll_seconds"])


require_service("release")
if arguments.step == "production":
    status, answer = call("GET", "/production")
    if status != 200:
        refused(status, answer)
    print(json.dumps(answer, indent=1) if arguments.json else
          f"production stands at {answer['released_commit'][:12]}; {len(answer['releases'])} release(s) published")
    sys.exit(0)
if arguments.step == "request":
    payload = {"also": arguments.also}
    if arguments.commit:
        payload["commit"] = arguments.commit
    status, answer = call("POST", "/releases", payload)
    if status != 202:
        refused(status, answer)
    wait_for(answer["release"], "request")
if not arguments.release:
    sys.exit(f"release: `{arguments.step}` needs the release's id")
if arguments.step == "status":
    status, answer = call("GET", f"/releases/{arguments.release}")
    if status != 200:
        refused(status, answer)
    show(answer)
    sys.exit(0)
status, answer = call("POST", f"/releases/{arguments.release}/{arguments.step}")
if status != 202:
    refused(status, answer)
wait_for(arguments.release, arguments.step)
