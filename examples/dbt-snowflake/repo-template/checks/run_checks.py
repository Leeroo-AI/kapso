"""Run the ticket checks on the project as it stands in this working tree.

    python checks/run_checks.py                          # every ticket in the backlog
    python checks/run_checks.py --ticket daily-revenue   # one ticket (repeat the option for more)
    python checks/run_checks.py --fast                   # the smoke subset named in checks/tickets.json

The checks are the data platform's: this script sends `warehouse/` and `marketing/` to the check service, which
builds each ticket's models from them in a fresh clone of the warehouse and runs that ticket's acceptance
checks on the result. The acceptance checks are not in this repository. What comes back per ticket is pass or
fail, how the build went, and, when the build failed, the kind of each error with the names involved. The
build's log itself is not returned: the same build in the development database shows the full message.

It prints one line per ticket, then the count, then the score as its last line:

    tickets_passing: 0.2353          a full run
    fast_tickets_passing: 0.2500     a --fast or --ticket run, which is a spot check and not the score

Details land in checks/results/ (ignored by git). Exit codes: 0 the checks ran; 2 the service could not be
reached or refused the submission; 3 on a full run when no ticket's build succeeded: the project itself does not build.
"""
import argparse
import io
import json
import pathlib
import socket
import sys
import tarfile
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SERVICE = json.loads((HERE / "service.json").read_text())
BACKLOG = json.loads((HERE / "tickets.json").read_text())
PROJECTS = ("warehouse", "marketing")
LEFT_OUT = {"target", "dbt_packages", "logs", "__pycache__", ".venv", ".user.yml", "profiles.yml"}


def project_root():
    """The directory that holds both dbt projects: the current directory or one above it, else one above this
    script, so the checks run the same from the repository root and from a copy of this folder inside it."""
    for start in (pathlib.Path.cwd(), HERE):
        for directory in (start, *start.parents):
            if all((directory / project / "dbt_project.yml").is_file() for project in PROJECTS):
                return directory
    sys.exit("checks: no warehouse/ and marketing/ project found from the current directory or from this script")


ROOT = project_root()


def keep(member):
    """Build output, installed packages, logs and connection profiles stay out of the submission."""
    if any(part in LEFT_OUT for part in pathlib.PurePosixPath(member.name).parts):
        return None
    if not (member.isfile() or member.isdir()):
        return None
    return member


def submission():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for project in PROJECTS:
            archive.add(ROOT / project, arcname=project, filter=keep)
    return buffer.getvalue()


def call(method, path, body=None):
    request = urllib.request.Request(SERVICE["url"] + path, data=body, method=method,
                                     headers={"Content-Type": "application/gzip"} if body else {})
    opener = urllib.request.build_opener(Quiet())
    with opener.open(request, timeout=SERVICE["request_timeout_seconds"]) as response:
        return response.status, json.loads(response.read().decode())


class Quiet(urllib.request.HTTPErrorProcessor):
    """Hand every answer back, whatever its status, so a refusal is read rather than raised."""

    def http_response(self, request, response):
        return response

    https_response = http_response


parser = argparse.ArgumentParser(description="Run the ticket checks on this working tree.")
parser.add_argument("--ticket", action="append", default=[], help="a ticket to check (repeatable); default: all")
parser.add_argument("--fast", action="store_true", help="the smoke subset; a spot check, not the score")
arguments = parser.parse_args()

known = [ticket["key"] for ticket in BACKLOG["tickets"]]
unknown = [name for name in arguments.ticket if name not in known]
if unknown:
    sys.exit(f"checks: unknown ticket(s) {', '.join(unknown)}; the backlog is: {', '.join(known)}")
selected = arguments.ticket or (BACKLOG["smoke"] if arguments.fast else known)
full_run = selected == known

address = urllib.parse.urlparse(SERVICE["url"])


def connect_error():
    """0 when the service accepts a connection, the socket's error number otherwise."""
    probe = socket.socket()
    probe.settimeout(SERVICE["connect_timeout_seconds"])
    error = probe.connect_ex((address.hostname, address.port))
    probe.close()
    return error


unreachable = connect_error()
if unreachable != 0:
    print(f"checks: the check service at {SERVICE['url']} is not reachable (connect error {unreachable}); "
          "nothing was checked. See README.md, 'The ticket checks'.")
    sys.exit(2)

status, accepted = call("POST", "/runs?tickets=" + ",".join(selected), submission())
if status != 202:
    print(f"checks: the check service refused the submission ({status}): {accepted.get('error')}")
    sys.exit(2)
run = accepted["run"]
print(f"ticket checks, run {run}: {len(selected)} ticket(s) submitted", flush=True)

deadline = time.time() + SERVICE["run_timeout_seconds"]
reported = set()
silent_since = None
view = {"state": "queued", "tickets": []}
while view["state"] != "done":
    if time.time() > deadline:
        print(f"checks: run {run} did not finish within {SERVICE['run_timeout_seconds']} s")
        sys.exit(2)
    time.sleep(SERVICE["poll_seconds"])
    if connect_error() != 0:
        # The door to the service can close for a moment and reopen; the run goes on behind it.
        if silent_since is None:
            silent_since = time.time()
            print("checks: the check service is not answering; waiting for it to come back", flush=True)
        if time.time() - silent_since > SERVICE["reconnect_seconds"]:
            print(f"checks: the check service stayed unreachable for {SERVICE['reconnect_seconds']} s; run {run} was not read")
            sys.exit(2)
        continue
    silent_since = None
    status, view = call("GET", f"/runs/{run}")
    if status != 200:
        print(f"checks: the check service lost run {run} ({status}): {view.get('error')}")
        sys.exit(2)
    for entry in view["tickets"]:
        if "passed" in entry and entry["ticket"] not in reported:
            reported.add(entry["ticket"])
            note = "" if entry["build"] == "ok" else f"   build {entry['build']}"
            if entry.get("check_error"):
                note = f"   {entry['check_error']}"
            print(f"  {'PASS' if entry['passed'] else 'FAIL'}  {entry['ticket']}{note}", flush=True)
            for line in entry.get("build_errors") or []:
                print(f"        {line}", flush=True)

results_dir = ROOT / "checks" / "results"
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / f"{run}.json").write_text(json.dumps(view, indent=2) + "\n", encoding="utf-8")
failed_builds = [entry["ticket"] for entry in view["tickets"] if entry["build"] != "ok"]
print(f"{view['passed']} of {view['total']} tickets pass")
if full_run and len(failed_builds) == view["total"]:
    print("checks: no ticket's build succeeded; the project does not build, so this is not a score")
    sys.exit(3)
share = view["passed"] / view["total"]
print(f"{'tickets_passing' if full_run else 'fast_tickets_passing'}: {share:.4f}")
