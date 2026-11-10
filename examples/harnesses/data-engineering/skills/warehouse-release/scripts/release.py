"""Release merged changes into the production warehouse through the repository's release command, audited with
each ticket's own checks before and after. Run from the deployment workspace, where results/ is.

    release.py deploy     request the release, audit the rehearsal, record production as it is, publish, check
                          production; writes deploy/release/released.json and prints the KAPSO_ lines
    release.py restart    the same for the commit this deployment released before (the deployment's start)
    release.py run        the proof call: every released ticket's checks on production as it stands now
    release.py rollback   put production back to before this release, and check that it is (the stop)
    release.py health     exit 0 while production stands at the released commit
    release.py logs       the platform's record of the release

What it needs from prepared/environment.json (the setup session pins them from the repository's
documentation): `release_command`, a command that speaks the contract of this skill's page (`request [--commit
C]`, `publish <id>`, `rollback <id>`, `production`, `status <id>`, each with `--json` printing one record);
`rehearsal_target`, the target that reads where a release is rehearsed; `reference_target`, the target that
reads production. A session never writes to production itself: the platform builds what is merged, and this
script decides, from the tickets' checks, whether a rehearsed release is published and whether a published
one stays.

Every audit runs on an export of the released commit, never on the workspace's own project: a ticket's checks
then read the relations the released code names, whatever unmerged change the workspace still holds.

A ticket is in the release when the released commit holds every file its proof names exactly as the workspace
does, and the platform built each of its datasets. A ticket whose files differ there is left out and named,
with how many of them the commit does hold. A ticket whose files are merged while the platform did not build
one of its datasets stops the release: production would carry the merge without the fix. A dataset the
platform built that no released ticket covers stops the release too, unless the call passes the option
allow_unaudited=yes, and is then named in the record. A rehearsal that fails a released ticket's check is not
published.

Before publishing, every dataset the release builds is fingerprinted on production, or shown absent there; one
that cannot be measured stops the release, since a rollback could not be checked afterwards, and so does a
production that moved to another commit while the release was rehearsed. From then on the record says
`publishing`, and the publication with the checks after it runs as a process of its own: however it ends short
of "published and every check passes" (a failed build, a command that prints no record, a failing check, a
crash), this script reads the record and compensates. When production still stands at the earlier commit and
every dataset fingerprints as recorded, nothing was published and the record says so. Otherwise the platform
is asked for the rollback, and the rollback counts only when the platform confirms it, production stands at
the earlier commit and every dataset the release built fingerprints as it did before the publication (a
fingerprint is a screen: counts, sums and distinct counts, not row equality). The script exits 1 either way.

What this script cannot do is outlive its own end: when it is killed together with the publication (a
deployment's time limit), the record still says `publishing`. While the record says `publishing`,
`rolling_back` or `rollback_failed`, production may hold part of a release, and this deployment releases
nothing new until its stop has run. Production that stands at neither this release's commit nor the one before
it holds a later release and is not rolled back from here. The record is written whole or not at all, and the
releases this deployment made earlier stay in it. Other skills' scripts are found through KAPSO_HARNESS_ROOT.
Standard library only; dbt, git and tar on PATH.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS = Path(os.environ["KAPSO_HARNESS_ROOT"]) if os.environ.get("KAPSO_HARNESS_ROOT") else Path(__file__).resolve().parents[3]
INVARIANTS = HARNESS / "skills" / "invariants" / "scripts" / "invariants.py"
FINGERPRINT = HARNESS / "skills" / "data-diff" / "scripts" / "fingerprint.py"
ENVIRONMENT = Path("prepared/environment.json")
RESULTS = Path("results")
FOLDER = Path("deploy/release")
STATE = FOLDER / "released.json"
LEDGER = RESULTS / "backlog.json"
MAY_HOLD_A_CHANGE = ("publishing", "rolling_back", "rollback_failed")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Written whole or not at all: a process that dies while writing leaves the record as it was."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_name(path.name + ".part")
    draft.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(draft, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def settings():
    for script in (INVARIANTS, FINGERPRINT):
        if not script.is_file():
            raise SystemExit(f"the harness's {script.name} is not at {script}: KAPSO_HARNESS_ROOT names the harness directory")
    if not ENVIRONMENT.is_file():
        raise SystemExit(f"{ENVIRONMENT} does not exist: the setup session pins the release command and the targets there")
    data = read_json(ENVIRONMENT)
    missing = [key for key in ("release_command", "rehearsal_target", "reference_target") if not data.get(key)]
    if missing:
        raise SystemExit(f"{ENVIRONMENT} names no {', '.join(missing)}: this task has no release path the harness can drive")
    return data


def platform(verb_arguments):
    """(record, exit code, what it printed): one verb of the release command. The record is the JSON object on
    its stdout, or None when it printed none; the caller says what a missing record means."""
    command = shlex.split(settings()["release_command"]) + verb_arguments + ["--json"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = completed.stdout.strip()
    start = text.find("{")
    record = json.loads(text[start:]) if start >= 0 else None
    return (record if isinstance(record, dict) else None), completed.returncode, (text or completed.stderr.strip())[-300:]


def production_stands_at():
    """The commit production stands at, or None when the platform did not answer with one."""
    record, code, _ = platform(["production"])
    return record.get("released_commit") if code == 0 and record else None


def git(*arguments):
    completed = subprocess.run(["git", "--literal-pathspecs", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def export_released(commit, tree):
    """The released commit's files under `tree`, fetched first when this workspace does not hold the commit."""
    if git("cat-file", "-e", f"{commit}^{{commit}}")[0] != 0:
        remote = os.environ.get("KAPSO_SOURCE_REMOTE")
        if not remote or git("fetch", "-q", remote, commit)[0] != 0:
            raise SystemExit(f"the released commit {commit[:12]} could not be fetched from {remote or 'an unknown remote'}: "
                             "which tickets it holds cannot be established")
    archive = subprocess.run(["git", "archive", "--format=tar", commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if archive.returncode != 0 or subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE).returncode != 0:
        raise SystemExit(f"the released commit {commit[:12]} could not be exported: nothing is audited on other code than the released one")


def project_at(tree, project_dir):
    """One dbt project of the exported commit, made runnable: the installed packages and an untracked profile
    copied from the workspace's project of the same path, the packages installed when the workspace has none."""
    project = Path(tree) / project_dir
    if not project.is_dir():
        raise SystemExit(f"the released commit holds no {project_dir}")
    for name in ("dbt_packages", "profiles.yml"):
        source = Path(project_dir) / name
        if source.is_dir() and not (project / name).exists():
            shutil.copytree(source, project / name)
        elif source.is_file() and not (project / name).exists():
            shutil.copy(source, project / name)
    if (project / "packages.yml").is_file() and not (project / "dbt_packages").is_dir():
        installed = subprocess.run(["dbt", "deps"], cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if installed.returncode != 0:
            raise SystemExit(f"the packages of {project_dir} could not be installed for the audit: {installed.stdout.strip()[-200:]}")
    return project


def units():
    """Every done ticket with its project, datasets, specs and the files its proof names, from the ledger, the
    tickets' evidence and their proofs. A spec that changed since it was measured stops the script."""
    done = None
    if LEDGER.is_file():
        document = read_json(LEDGER)
        rows = document.get("tickets") if isinstance(document, dict) else document
        done = {row["ticket"]: row for row in rows if isinstance(row, dict) and row.get("status") == "done"}
    found = {}
    for evidence_file in sorted(RESULTS.glob("*/evidence.json")):
        evidence = read_json(evidence_file)
        ticket = evidence["ticket"]
        if done is not None and ticket not in done:
            continue
        proof_file = RESULTS / "proofs" / f"{ticket}.json" if (RESULTS / "proofs" / f"{ticket}.json").is_file() else RESULTS / "proof.json"
        proof = read_json(proof_file) if proof_file.is_file() else {}
        datasets = evidence["select"] + evidence["snapshots"]
        specs = {}
        for name in datasets:
            entry = evidence.get("specs", {}).get(name)
            if entry is None or not Path(entry["path"]).is_file() or sha256(entry["path"]) != entry["sha256"]:
                raise SystemExit(f"{ticket}: the spec of {name} is missing or is not the one its evidence was measured with; nothing is released on it")
            specs[name] = entry["path"]
        # The platform names a built resource by its project's directory in the repository and its name.
        found[ticket] = {"project": os.path.normpath(evidence["project_dir"]), "datasets": datasets, "specs": specs,
                         "files": [str(path) for path in proof.get("changed_files") or []]}
    if not found:
        raise SystemExit("no ticket has evidence (results/<ticket>/evidence.json): nothing to audit a release with")
    return found


def files_at(commit, unit):
    """The files of a unit that the released commit does not hold exactly as the workspace does."""
    different = []
    for path in unit["files"]:
        code, released, _ = git("rev-parse", "--verify", "--quiet", f"{commit}:{path}")
        here = git("hash-object", "--", path)[1] if Path(path).is_file() else ""
        if (released if code == 0 else "") != here:
            different.append(path)
    return different


def run_checks(ticket, unit, target, tree):
    """Every spec of one ticket on one target, run in the released commit's project: {dataset: {passed, not_passing}}."""
    outcome = {}
    for name, spec in unit["specs"].items():
        out = FOLDER / f"{target}_{ticket}_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        completed = subprocess.run([sys.executable, str(INVARIANTS), str(Path(spec).resolve()), "--project-dir", str(project_at(tree, unit["project"])),
                                    "--target", target, "--out", str(out.resolve())], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if not out.is_file():
            outcome[name] = {"passed": 0, "not_passing": [f"the spec could not be run: {completed.stdout.strip()[-200:]}"]}
            continue
        report = read_json(out)
        outcome[name] = {"passed": report["passed"], "not_passing": [r["name"] for r in report["results"] if r["status"] != "pass"]}
    return outcome


def all_pass(checks, unit):
    """Every dataset of the unit has a report with at least one check, and none that does not pass."""
    return set(checks) == set(unit["datasets"]) and all(result["passed"] > 0 and result["not_passing"] == [] for result in checks.values())


def fingerprints(state, label, tree):
    """{project: report}: every resource the release builds, fingerprinted on production through the released
    commit's project, one report a project."""
    names = {}
    for entry in state["staged"]:
        names.setdefault(entry["project"], set()).add(entry["resource"])
    reports = {}
    for index, (project, resources) in enumerate(sorted(names.items())):
        out = FOLDER / state["release"] / f"{label}_{index}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        subprocess.run([sys.executable, str(FINGERPRINT), "--select", " ".join(sorted(resources)), "--project-dir", str(project_at(tree, project)),
                        "--target", settings()["reference_target"], "--save", str(out.resolve())], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        reports[project] = str(out)
    return reports


def not_measured(reports, staged):
    """The resources whose state on production a report does not hold: neither a fingerprint with its row
    count nor an absence the adapter showed."""
    missing = []
    for entry in staged:
        path = reports.get(entry["project"])
        measured = read_json(path)["models"].get(entry["resource"]) if path and Path(path).is_file() else None
        rows = (measured or {}).get("rows")
        counted = isinstance(rows, int) and not isinstance(rows, bool) and not (measured or {}).get("error")
        if measured is None or not (measured.get("absent") is True or counted):
            missing.append(f"{entry['project']} {entry['resource']}" + (f" ({measured['error'][:120]})" if measured and measured.get("error") else ""))
    return missing


def differences(before, after):
    """What of the release's datasets on production does not read as it did before the publication."""
    wrong = []
    for project, path in before.items():
        completed = subprocess.run([sys.executable, str(FINGERPRINT), "--compare", path, after.get(project, ""), "--json"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            wrong.append(f"{project}: production could not be compared with its record from before the release ({completed.stderr.strip()[-200:]})")
            continue
        result = json.loads(completed.stdout)
        wrong += [f"{project} {item['model']}: rows {item['rows_before']} before the release, {item['rows_after']} now"
                  + (f"; differs in {', '.join(item['columns'][:4])}" if item["columns"] else "") for item in result["changed"]]
        wrong += [f"{project} {name}: recorded before the release and not measured now" for name in result["removed"]]
    return wrong


def roll_back(state, why):
    """(production is as before, what happened): settle a release that may have changed production. When
    production still stands at the earlier commit and every dataset fingerprints as recorded, nothing was
    published and nothing is undone. Otherwise the record says `rolling_back`, the platform is asked, and the
    rollback counts only with the platform's confirmation, the earlier commit and equal fingerprints."""
    release, earlier = state["release"], state["production_before"]
    stands_at = production_stands_at()
    if stands_at not in (state["commit"], earlier):
        return False, (f"production stands at {stands_at[:12] if stands_at else 'a commit the platform did not name'}, neither release {release}'s "
                       "commit nor the one before it: a later release may hold, and rolling this one back from here could undo it. "
                       "Nothing was rolled back; this is for the platform's owner")
    with tempfile.TemporaryDirectory(prefix="released-") as tree:
        export_released(state["commit"], tree)
        if stands_at == earlier:
            now = fingerprints(state, "before_rollback", tree)
            if not not_measured(now, state["staged"]) and not differences(state["before"], now):
                state.update(state="not_published", settled_at=stamp(), not_published_because=why)
                write_json(STATE, state)
                return True, (f"nothing of release {release} is in production: it stands at {earlier[:12]} and every dataset the release "
                              "would have built fingerprints as recorded before the publication, so there was nothing to roll back")
        state.update(state="rolling_back", rolling_back_since=stamp(), rolled_back_because=why)
        write_json(STATE, state)
        record, code, printed = platform(["rollback", release])
        after = fingerprints(state, "after_rollback", tree)
        wrong = [f"not measured after the rollback: {name}" for name in not_measured(after, state["staged"])] or differences(state["before"], after)
    stands_at = production_stands_at()
    said = (record or {}).get("state") or f"no record: {printed}"
    confirmed = code == 0 and (record or {}).get("state") == "rolled_back"
    restored = confirmed and stands_at == earlier and not wrong
    state.update(state="rolled_back" if restored else "rollback_failed", rolled_back_at=stamp(),
                 rollback={"platform": said, "platform_exit": code, "production_stands_at": stands_at, "problems": wrong})
    write_json(STATE, state)
    if restored:
        return True, (f"release {release} was rolled back: the platform confirms it, production stands at {earlier[:12]} again and every "
                      "dataset the release built fingerprints as it did before the publication")
    return False, (f"the rollback of release {release} is not established (the platform: {said}, exit {code}; production stands at "
                   f"{str(stands_at)[:12]}; {'; '.join(wrong[:3]) or 'no measured dataset differs'}): this is an incident for the platform's owner")


def deploy(commit=None, request_arguments=None, allow_unaudited=None):
    config = settings()
    earlier = []
    if STATE.is_file():
        last = read_json(STATE)
        if last.get("state") in MAY_HOLD_A_CHANGE:
            raise SystemExit(f"release {last['release']} of this deployment is recorded as {last['state']} and may have left production changed: "
                             "run the deployment's stop first (it rolls that release back and checks production)")
        if last.get("state") == "published":
            raise SystemExit(f"release {last['release']} of this deployment is published: its stop rolls it back, and a newer commit is "
                             "released by a new deployment")
        earlier = last.pop("earlier", []) + [last]
    commit = commit or os.environ.get("KAPSO_OPTION_COMMIT")
    if request_arguments is None:
        request_arguments = shlex.split(os.environ.get("KAPSO_OPTION_REQUEST_ARGUMENTS") or "")
    if allow_unaudited is None:
        allow_unaudited = os.environ.get("KAPSO_OPTION_ALLOW_UNAUDITED") == "yes"
    before = production_stands_at()
    if before is None:
        raise SystemExit("the release command did not say which commit production stands at: nothing was requested")
    record, code, printed = platform(["request"] + (["--commit", commit] if commit else []) + request_arguments)
    if record is None or not record.get("release") or not record.get("commit"):
        raise SystemExit(f"the platform refused the release: {(record or {}).get('error') or printed}")
    if commit and not record["commit"].startswith(commit):
        # Before anything is recorded: a rehearsal of another commit never becomes what a restart asks for.
        raise SystemExit(f"the platform rehearsed {record['commit'][:12]}, not the commit asked for ({commit[:12]}): nothing was published, "
                         "and this deployment's record is unchanged")
    staged = record.get("staged") or []
    state = {"release": record["release"], "commit": record["commit"], "state": "requested", "requested_at": stamp(),
             "request_arguments": request_arguments, "allow_unaudited": allow_unaudited, "production_before": before, "staged": staged,
             "tickets": {}, "left_out": [], "earlier": earlier}
    write_json(STATE, state)
    not_built = [entry for entry in staged if entry.get("status") != "success"]
    if code != 0 or record.get("state") != "staged" or not_built or not staged:
        for entry in not_built:
            print(f"  not built in the rehearsal: {entry.get('project')} {entry.get('resource')}: " + " | ".join(entry.get("errors", []))[:300])
        raise SystemExit(f"release {record['release']} is {record.get('state')}: nothing was published")
    if any(not isinstance(entry.get("project"), str) or not isinstance(entry.get("resource"), str) for entry in staged):
        raise SystemExit("the platform's record names a built resource without its `project` and `resource`: the release cannot be audited")
    built = {(entry["project"], entry["resource"]) for entry in staged}
    print(f"release {record['release']} at {record['commit'][:12]}: {len(built)} resource(s) built in the rehearsal")
    with tempfile.TemporaryDirectory(prefix="released-") as tree:
        export_released(record["commit"], tree)
        released, merged_and_not_built = {}, []
        for ticket, unit in units().items():
            different = files_at(record["commit"], unit)
            missing = [name for name in unit["datasets"] if (unit["project"], name) not in built]
            if not unit["files"]:
                state["left_out"].append({"ticket": ticket, "why": "its proof names no changed file"})
            elif different:
                state["left_out"].append({"ticket": ticket, "why": f"the released commit holds {len(unit['files']) - len(different)} of its "
                                          f"{len(unit['files'])} file(s) as delivered (not merged, or merged differently): " + ", ".join(different[:4])})
            elif missing:
                merged_and_not_built.append(f"{ticket} ({', '.join(missing)})")
            else:
                released[ticket] = unit
        for item in state["left_out"]:
            print(f"  left out  {item['ticket']}: {item['why']}")
        covered = {(unit["project"], name) for unit in released.values() for name in unit["datasets"]}
        state["without_a_ticket"] = sorted(f"{project} {resource}" for project, resource in built - covered)
        write_json(STATE, state)
        if merged_and_not_built:
            raise SystemExit("merged, and the platform did not build what the ticket changes: " + "; ".join(merged_and_not_built) + ". Production "
                             "would carry the merge without the fix, so nothing was published. Ask the platform to build those datasets with "
                             "the release: the option request_arguments passes what the repository's documentation gives for that")
        if not released:
            raise SystemExit("no ticket of this campaign is in the release; nothing to audit, nothing was published")
        if state["without_a_ticket"] and not allow_unaudited:
            raise SystemExit("the release would also publish what no released ticket's checks cover: " + ", ".join(state["without_a_ticket"])
                             + ". Nothing was published. Release it with the option allow_unaudited=yes when that is intended")
        if state["without_a_ticket"]:
            print(f"  published on the merge alone, with no ticket's checks behind it: {', '.join(state['without_a_ticket'])}")
        failed_audit = []
        for ticket, unit in released.items():
            checks = run_checks(ticket, unit, config["rehearsal_target"], tree)
            state["tickets"][ticket] = {"datasets": unit["datasets"], "rehearsal": checks}
            held = all_pass(checks, unit)
            print(f"  rehearsal {'pass' if held else 'FAIL'}  {ticket}: " + "; ".join(f"{name} {result['passed']} pass" + (f", not passing: {', '.join(result['not_passing'][:3])}" if result["not_passing"] else "")
                                                                          for name, result in checks.items()))
            if not held:
                failed_audit.append(ticket)
        state["state"] = "rehearsed"
        write_json(STATE, state)
        if failed_audit:
            raise SystemExit(f"the rehearsal does not pass the checks of {', '.join(failed_audit)}: release {record['release']} was not published")
        state["before"] = fingerprints(state, "before", tree)
        unmeasured = not_measured(state["before"], staged)
        write_json(STATE, state)
    if unmeasured:
        raise SystemExit("production as it stands before the release could not be recorded for " + "; ".join(unmeasured[:6])
                         + ". A rollback could not be checked against it, so nothing was published")
    if production_stands_at() != before:
        raise SystemExit(f"production moved from {before[:12]} while this release was rehearsed: its record of production is of another "
                         "state, so nothing was published. Request the release again")
    state["state"] = "publishing"
    write_json(STATE, state)
    # The publication and its checks run as a process of their own: whatever way that process ends, this one
    # reads the record it left and settles the release unless the record says it holds.
    ended = subprocess.run([sys.executable, str(Path(__file__).resolve()), "publish-and-check"]).returncode
    state = read_json(STATE)
    if ended != 0 or state.get("state") != "published":
        why = state.get("did_not_hold") or f"the publication of release {record['release']} ended without a result (exit {ended})"
        raise SystemExit(f"{why}. " + roll_back(state, why)[1])
    print(f"released {len(released)} ticket(s) into production at {record['commit'][:12]}; left out: {len(state['left_out'])}")
    print(f"KAPSO_ENDPOINT={STATE}")
    print(f"KAPSO_VALUE_RELEASE={record['release']}")
    print(f"KAPSO_VALUE_COMMIT={record['commit']}")
    print(f"KAPSO_VALUE_TICKETS={','.join(sorted(released))}")


def publish_and_check():
    """Publish the rehearsed release and run every released ticket's checks on production. Started by deploy
    only, which settles any end of this process that leaves the record short of `published`."""
    state = recorded()
    if state.get("state") != "publishing":
        raise SystemExit(f"release {state['release']} is {state.get('state')}: only deploy starts a publication")
    config, all_units = settings(), units()
    with tempfile.TemporaryDirectory(prefix="released-") as tree:
        export_released(state["commit"], tree)   # before anything is published: an audit that cannot run publishes nothing
        published, code, printed = platform(["publish", state["release"]])
        state.update(published=(published or {}).get("published", []), backed_up=(published or {}).get("backed_up", []))
        if code != 0 or published is None or published.get("state") != "published":
            state["did_not_hold"] = (f"release {state['release']} did not complete in production "
                                     f"({(published or {}).get('state') or 'no record'}: {(published or {}).get('error') or printed})")
            write_json(STATE, state)
            sys.exit(1)
        failed_after = []
        for ticket in state["tickets"]:
            checks = run_checks(ticket, all_units[ticket], config["reference_target"], tree)
            state["tickets"][ticket]["production"] = checks
            held = all_pass(checks, all_units[ticket])
            print(f"  production {'pass' if held else 'FAIL'}  {ticket}")
            if not held:
                failed_after.append(ticket)
    stands_at = production_stands_at()
    if failed_after or stands_at != state["commit"]:
        state["did_not_hold"] = (f"production does not pass the checks of {', '.join(failed_after)} after release {state['release']}" if failed_after
                                 else f"the platform says release {state['release']} is published and production stands at {str(stands_at)[:12]}")
        write_json(STATE, state)
        sys.exit(1)
    state.update(state="published", published_at=stamp())
    write_json(STATE, state)


def recorded():
    if not STATE.is_file():
        raise SystemExit(f"no {STATE}: this deployment released nothing")
    return read_json(STATE)


def restart():
    state = recorded()
    if state.get("state") == "published" and production_stands_at() == state["commit"]:
        print(f"release {state['release']} is published and production stands at it")
        return
    deploy(commit=state["commit"], request_arguments=state.get("request_arguments", []), allow_unaudited=state.get("allow_unaudited", False))


def run():
    json.load(sys.stdin)
    state = recorded()
    if state.get("state") != "published":
        raise SystemExit(f"release {state['release']} is {state.get('state')}, not published: there is nothing of it on production to prove")
    config, all_units = settings(), units()
    tickets, failing = {}, []
    with tempfile.TemporaryDirectory(prefix="released-") as tree:
        export_released(state["commit"], tree)
        for ticket in state["tickets"]:
            checks = run_checks(ticket, all_units[ticket], config["reference_target"], tree)
            tickets[ticket] = {"checks_passing": sum(result["passed"] for result in checks.values()),
                               "not_passing": [f"{name}: {check}" for name, result in checks.items() for check in result["not_passing"]]}
            if not all_pass(checks, all_units[ticket]):
                failing.append(ticket)
    stands_at = production_stands_at()
    print(json.dumps({"release": state["release"], "commit": state["commit"], "production_stands_at": stands_at,
                      "tickets": tickets, "left_out": [item["ticket"] for item in state["left_out"]],
                      "without_a_ticket": state.get("without_a_ticket", [])}))
    if stands_at != state["commit"]:
        raise SystemExit("production does not stand at the released commit")
    if failing:
        raise SystemExit(f"not every check of {', '.join(failing)} passes on production")


def rollback():
    state = recorded()
    if state.get("state") in ("rolled_back", "not_published"):
        print(f"release {state['release']} is {state['state']}: nothing of it is in production")
        return
    if state.get("state") not in MAY_HOLD_A_CHANGE + ("published",):
        print(f"release {state['release']} is {state.get('state')}: it was never published, and nothing of it is in production")
        return
    restored, what_happened = roll_back(state, "the deployment was stopped")
    print(what_happened)
    sys.exit(0 if restored else 1)


def health():
    state = recorded()
    stands_at = production_stands_at()
    if state.get("state") != "published" or stands_at != state["commit"]:
        print(f"the release is {state.get('state')} and production stands at {str(stands_at)[:12]}; the released commit is {state['commit'][:12]}", file=sys.stderr)
        sys.exit(1)


def logs():
    record, code, printed = platform(["status", recorded()["release"]])
    print(json.dumps(record, indent=1) if record else printed)
    sys.exit(code)


VERBS = {"deploy": deploy, "restart": restart, "run": run, "rollback": rollback, "health": health, "logs": logs,
         "publish-and-check": publish_and_check}
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VERBS:
        raise SystemExit(__doc__)
    sys.stdout.reconfigure(line_buffering=True)   # the publication's own lines land between this process's, in order
    VERBS[sys.argv[1]]()
