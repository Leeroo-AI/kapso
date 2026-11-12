"""Measure a change the way the validation report needs it, and compute the label those measurements earn.

Usage:
  evidence.py environment [--ticket T]
  evidence.py baseline --ticket T --kind fix|new|refactor|performance|migration --select MODEL [MODEL ...]
                       [--snapshot NAME ...] [--project-dir DIR] [--no-code-build] [--again REASON]
  evidence.py prove    --ticket T [--with-downstream]
  evidence.py label    (--ticket T | --all) [--write] [--json] [--evidence FILE]
  evidence.py verify   --all --out FILE [--only TICKET ...]
  evidence.py lower    --in FILE --ticket T --reason TEXT

`environment` checks prepared/environment.json, the file the setup session writes: the keys the scripts read,
their types, that each command's program exists and each pattern compiles. With --ticket it also runs the
acceptance command for that ticket and says how its output was read.

`baseline` is run before any file of the ticket changes: it records the commit and every file that already
differs from it (a file the proof later names must have changed after this point), puts the build namespace
back to the reference state (the reset command), records the SHA-256 of each dataset's invariants spec
(results/<ticket>/spec_<dataset>.json, one per dataset, written first), and for an existing dataset builds the
code as it stands and runs the spec on it, so the defect is measured before it is removed. A dataset that does
not build is recorded with dbt's error when the error is its own. A second baseline needs --again with the
reason, and when the commit has moved since the first one the before-state is read from the reference alone.

`prove` is run after the change: it puts the namespace back to the reference state, builds the datasets (and,
with --with-downstream, everything that reads them), requires dbt's own record to say each one succeeded, runs
each spec on the build and on the reference target (what production holds), fingerprints both, compares the
rows by `except` when the kind promises an unchanged output, runs the task's acceptance check when the setup
pinned one, and writes results/<ticket>/evidence.json: what was run, the SHA-256 of every report it wrote, and
of every file that differs from the baseline's commit. A dataset downstream whose own build failed is built
once more from the code as it stood at the baseline's commit, exported beside the workspace, on the reset
namespace; the label sets it aside only when dbt records the same error for it there.

`label` computes what the proof and those reports earn, obligation by obligation. It reads the reports named
by hash, never a summary, and requires the code to be as it was measured. "safe to merge" needs every
obligation met; one missing gives "unsure" with what is missing as the reason. `--write` puts the label and
its reasons into the proof. `--evidence FILE` computes one ticket's label from another measurement of it than
the attempt's own: verification's.

`verify` is the verification session's: for every closed ticket it resets, rebuilds from the attempt's code,
measures both sides again in a scratch directory beside the report, runs the acceptance check again, and
reports the label claimed and the label the fresh measurement earns. That measurement, taken on the code as it
is delivered, is the ticket's after-state from then on: a pull request is published on it. What the attempt's
own measurement said differently is recorded, and lowers the label only when the code is the same in both. A
ticket it did not measure again is reported as unsure. `lower` records a reviewer's finding against a ticket
in that report; it can only lower.

What these commands cannot do: the files they write sit in the workspace, where the session that wrote the
change can write too. The label is a consistency check of its inputs, and it is believed because verification
takes the measurements again from the code, not because the files are protected.

Other skills' scripts are found through KAPSO_HARNESS_ROOT, which the engine sets in every session; a run by
hand finds them beside this skill. Exit status: 0 done; 1 a build failed, an artifact is missing or a
measurement could not be taken (the reason is printed); `label` exits 0 whatever the label is. Standard library
only; dbt and git on PATH.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
HARNESS = Path(os.environ["KAPSO_HARNESS_ROOT"]) if os.environ.get("KAPSO_HARNESS_ROOT") else SCRIPTS.parents[2]
INVARIANTS = HARNESS / "skills" / "invariants" / "scripts" / "invariants.py"
FINGERPRINT = HARNESS / "skills" / "data-diff" / "scripts" / "fingerprint.py"
ENVIRONMENT = Path("prepared/environment.json")
TEXT_KEYS = ("build_target", "reference_target", "rehearsal_target", "reset_command", "acceptance_command",
             "acceptance_pass_pattern", "acceptance_fail_pattern", "release_command")
COMMAND_KEYS = ("reset_command", "acceptance_command", "release_command")
PARALLEL_KEY = "acceptance_at_a_time"
KINDS = ("fix", "new", "refactor", "performance", "migration")
EQUAL_OUTPUT_KINDS = ("refactor", "performance", "migration")
SAFE, UNSURE, WONT_DO = "safe to merge", "unsure", "won't do"
ORDER = (WONT_DO, UNSURE, SAFE)
INDEPENDENT_CHECKS = ("reconcile", "completeness", "custom")
READS = re.compile(r"\b(?:from|join)\s+(\{\{.*?\}\}|[A-Za-z_][\w.$\"]*)", re.IGNORECASE | re.DOTALL)
SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
NAMED_SUBQUERY = re.compile(r"(?:\bwith|,)\s+([A-Za-z_]\w*)\s*(?:\([^()]*\))?\s*as\s*\(", re.IGNORECASE)
FAMILY = re.compile(r"^[A-Za-z0-9_]{3,}\*$")
RUNBOOK_KEYS = ("one_time_actions", "order", "verification", "recovery")
# What the harness and the engine write in the workspace: their records are never a ticket's change, and a
# session's own record grows while it runs. A repository's own deploy/ directory is code.
OUTSIDE_THE_CODE = ("results/", "prepared/", "deploy/pull_requests/", "deploy/release/", ".kapso/")
SINGLE_TICKET = "change"
TICKET_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
INVENTORY = Path("prepared/backlog/inventory.json")
ERROR_LINES = 8
NO_DBT = ("dbt is not on PATH: these scripts measure through dbt and cannot run on another stack. Say so in the "
          "proof (validated.skipped), and leave the label to `evidence.py label`, which reads unsure")


def run(command, cwd=None):
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.returncode, completed.stdout


def git(*arguments):
    """(exit code, stdout) of one git command in the workspace, its stderr kept apart."""
    completed = subprocess.run(["git", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.returncode, completed.stdout


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def environment():
    """What setup pinned about the task's environment; every key is optional."""
    return read_json(ENVIRONMENT) if ENVIRONMENT.is_file() else {}


def dbt_errors(output):
    """dbt's own error lines, without warnings and colour codes."""
    lines = [line.strip() for line in output.replace("\x1b[0m", "").splitlines()]
    wanted = [line for line in lines if line and any(word in line for word in ("Error", "error", "ERROR", "not found", "does not exist", "invalid", "undefined"))
              and "deprecat" not in line.lower() and "WARNING" not in line]
    return (wanted or [line for line in lines if line])[-ERROR_LINES:]


def dbt(project_dir, arguments, target):
    if shutil.which("dbt") is None:
        raise SystemExit(NO_DBT)
    return run(["dbt", "--no-use-colors", *arguments] + (["--target", target] if target else []), cwd=project_dir)


def spec_path(results, ticket, dataset):
    return results / ticket / f"spec_{dataset}.json"


def invariants(spec, project_dir, target, out):
    """Run one spec on one target; the report, or None with the reason when no report was written."""
    out = Path(out)
    if out.exists():
        out.unlink()
    command = [sys.executable, str(INVARIANTS), str(spec), "--project-dir", str(project_dir), "--out", str(out)]
    code, output = run(command + (["--target", target] if target else []))
    if not out.is_file():
        return None, " | ".join(dbt_errors(output))[:600]
    return read_json(out), None


def fingerprints(selector, project_dir, target, out):
    out = Path(out)
    if out.exists():
        out.unlink()
    command = [sys.executable, str(FINGERPRINT), "--select", selector, "--project-dir", str(project_dir), "--save", str(out)]
    code, output = run(command + (["--target", target] if target else []))
    if not out.is_file():
        return None, " | ".join(dbt_errors(output))[:600]
    return read_json(out), None


def listed(selector, project_dir, target):
    """(rows, error): the models and snapshots a selector lists, each with its name, type, relation and config.
    dbt prints its notices on the same stream; a listed resource is a line that is one JSON object."""
    code, output = dbt(project_dir, ["ls", "--select", *selector, "--resource-type", "model", "snapshot", "--output", "json",
                                     "--output-keys", "name", "resource_type", "database", "schema", "config"], target)
    if code != 0:
        return None, " | ".join(dbt_errors(output))[:400]
    return [json.loads(line) for line in (raw.strip() for raw in output.splitlines()) if line.startswith("{") and line.endswith("}")], None


def downstream_of(datasets, project_dir, target):
    """(names of everything that reads the datasets, error), the datasets themselves left out. An ephemeral
    model has no relation to measure: what reads it is listed in its own right."""
    rows, error = listed([f"{name}+" for name in datasets], project_dir, target)
    if rows is None:
        return None, error
    return sorted({row["name"] for row in rows if (row.get("config") or {}).get("materialized") != "ephemeral"} - set(datasets)), None


def resource_name(unique_id):
    """The name of a model or snapshot from dbt's id for it: `model.<package>.<name>`, with `.v<n>` after the
    name for a versioned model."""
    parts = unique_id.split(".")
    return parts[2] if len(parts) >= 3 else unique_id


def run_dbt_recorded(project_dir, arguments, target):
    """One dbt command and its own record of what it ran: (exit status, output, {resource name: status},
    {resource name: dbt's message for one that did not succeed}), with None in place of the two mappings when
    dbt left no record. The record of the previous command is removed first, so what is read is this
    command's or nothing; a model with several versions is as good as its worst. The record is read where
    dbt writes it by default, <project>/target/run_results.json."""
    record = Path(project_dir) / "target" / "run_results.json"
    if record.is_file():
        record.unlink()
    code, output = dbt(project_dir, arguments, target)
    if not record.is_file():
        return code, output, None, None
    statuses, messages = {}, {}
    for entry in read_json(record).get("results", []):
        name = resource_name(entry["unique_id"])
        if statuses.get(name, "success") == "success":
            statuses[name] = entry["status"]
            if entry["status"] != "success":
                messages[name] = str(entry.get("message") or "")
    return code, output, statuses, messages


def same_failure(message, other):
    """Whether two of dbt's error messages name the same failure: the same first line, once the spacing is
    evened out. An empty message is the same as nothing."""
    first = [" ".join((text or "").strip().splitlines()[0].split()) if (text or "").strip() else "" for text in (message, other)]
    return bool(first[0]) and first[0] == first[1]


def build_command(selection):
    """The models and snapshots of a selection in the order the project's graph gives them, and no test."""
    return ["build", "--resource-type", "model", "snapshot", "--select", *selection]


def build(project_dir, models, snapshots, with_downstream, target):
    """Build the datasets (and what reads them, when asked): one command, so a snapshot runs where the graph
    puts it. Returns the steps, each with its command, exit status, error lines, whether dbt left its record
    and the resources that record calls failed, then {resource: status} and {resource: dbt's message} for
    the ones that did not succeed."""
    arguments = build_command([f"{name}+" for name in models + snapshots] if with_downstream else models + snapshots)
    code, output, recorded, messages = run_dbt_recorded(project_dir, arguments, target)
    step = {"command": "dbt " + " ".join(arguments), "exit": code, "errors": dbt_errors(output) if code else [], "recorded": recorded is not None,
            "failed": sorted(name for name, status in (recorded or {}).items() if status in ("error", "fail"))}
    return [step], recorded or {}, messages or {}


def built_as_it_stood(commit, project_dir, selection, target, reset_command):
    """({resource: status}, {resource: dbt's message}, why not): what dbt records when a selection is built
    from the code as it stood at one commit, on the reset namespace. The commit's files are exported beside
    the workspace, which is not touched; the project's installed packages and an untracked profile are
    copied over, since no commit holds them. None for both mappings, with the reason, when that build could
    not be run or left no record."""
    if run(shlex.split(reset_command))[0] != 0:
        return None, None, "the namespace could not be reset"
    with tempfile.TemporaryDirectory(prefix="code-as-it-stood-") as root:
        archive = subprocess.run(["git", "archive", "--format=tar", commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if archive.returncode != 0 or subprocess.run(["tar", "-x", "-C", root], input=archive.stdout, stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE).returncode != 0:
            return None, None, f"the code at {commit[:12]} could not be exported"
        project = Path(root) / project_dir
        if not project.is_dir():
            return None, None, f"{project_dir} does not exist at {commit[:12]}"
        for name in ("dbt_packages", "profiles.yml"):
            source = Path(project_dir) / name
            if source.is_dir() and not (project / name).exists():
                shutil.copytree(source, project / name)
            elif source.is_file() and not (project / name).exists():
                shutil.copy(source, project / name)
        code, output, statuses, messages = run_dbt_recorded(project, build_command(selection), target)
    if statuses is None:
        return None, None, "dbt built nothing from it: " + (" | ".join(dbt_errors(output)[-2:])[:240] or f"exit {code}")
    return statuses, messages, None


def control_for_broken_consumers(evidence, first_head, settings):
    """The control for every consumer whose own build failed after the change: the same selection built from
    the code as it stood at the ticket's first baseline, on the reset namespace, and what dbt recorded for
    each of them there (status and message), kept beside what it recorded after the change. The label reads
    the two. Run after everything that reads the candidate's build: the namespace is left holding the
    control's. Nothing to do without a failed consumer or a reset command."""
    broken = [item for item in evidence["downstream"]["unmeasured"] if item.get("status") == "error"]
    if not broken or not settings.get("reset_command"):
        return
    datasets = evidence["select"] + evidence["snapshots"]
    statuses, messages, why_not = built_as_it_stood(first_head, evidence["project_dir"], [f"{name}+" for name in datasets],
                                                    settings.get("build_target"), settings["reset_command"])
    for item in broken:
        item["on_the_code_as_it_stood"] = ({"status": statuses.get(item["model"], "not run"), "error": (messages.get(item["model"]) or "")[:400]}
                                           if statuses is not None else {"status": f"not taken ({why_not})", "error": ""})
    evidence["namespace_holds_the_control"] = True


def report_statuses(report):
    return {result["name"]: result["status"] for result in (report or {}).get("results", [])}


def changed_since(commit):
    """{path: sha256, or "deleted"} for every file of the code that differs from the commit: tracked changes,
    committed or not, and files git does not track yet. None when the commit is not in this repository."""
    code, tracked = git("diff", "--name-only", "-z", commit)
    if code != 0:
        return None
    code, untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted({path for path in (tracked + untracked).split("\0") if path and not path.startswith(OUTSIDE_THE_CODE)})
    return {path: sha256(path) if Path(path).is_file() else "deleted" for path in paths}


def acceptance(ticket, settings):
    """What the task's acceptance check says about one ticket, run now on the working tree; None when the setup
    pinned no acceptance command. `{ticket}` in the command and in the patterns is the ticket's key. With a
    pass pattern it passes when the command exits 0 and its output matches the pattern (and not the fail
    pattern, when one is pinned), and `readable` is false when the output matched neither pattern, or both;
    without one the exit status alone is the verdict."""
    command, pass_pattern = settings.get("acceptance_command"), settings.get("acceptance_pass_pattern")
    if not command:
        return None
    filled = command.replace("{ticket}", ticket)
    code, output = run(shlex.split(filled))
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not pass_pattern:
        return {"command": filled, "verdict": "pass" if code == 0 else "fail", "exit": code, "readable": True,
                "said": (lines[-1] if lines else "no output")[:200]}

    def matches(pattern):
        return re.search(pattern.replace("{ticket}", re.escape(ticket)), output, re.MULTILINE) is not None

    passed = matches(pass_pattern)
    fail_pattern = settings.get("acceptance_fail_pattern")
    failed = matches(fail_pattern) if fail_pattern else not passed
    return {"command": filled, "verdict": "pass" if code == 0 and passed and not failed else "fail", "exit": code,
            "readable": passed != failed, "said": (lines[-1] if lines else "no output")[:200]}


# ---- environment ------------------------------------------------------------------------------------------------

def check_environment(args):
    if not ENVIRONMENT.is_file():
        print(f"{ENVIRONMENT} does not exist: the setup session writes it from the task statement and the repository's documentation")
        return 1
    settings = read_json(ENVIRONMENT)
    if not isinstance(settings, dict):
        print(f"{ENVIRONMENT} must hold one JSON object")
        return 1
    problems = []
    for key, value in settings.items():
        if key == PARALLEL_KEY:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"`{key}` is a whole number of checks that may run at once, 1 or more")
        elif key not in TEXT_KEYS:
            problems.append(f"`{key}` is not a key the scripts read (they read {', '.join(TEXT_KEYS + (PARALLEL_KEY,))})")
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"`{key}` must be a non-empty string, or left out")
    text = {key: value for key, value in settings.items() if key in TEXT_KEYS and isinstance(value, str) and value.strip()}
    for key in COMMAND_KEYS:
        if key in text and shutil.which(shlex.split(text[key])[0]) is None:
            problems.append(f"`{key}`: {shlex.split(text[key])[0]} is not a program on PATH")
    for key in ("acceptance_pass_pattern", "acceptance_fail_pattern"):
        if key in text and "acceptance_command" not in text:
            problems.append(f"`{key}` needs `acceptance_command`")
    if "acceptance_fail_pattern" in text and "acceptance_pass_pattern" not in text:
        problems.append("`acceptance_fail_pattern` needs `acceptance_pass_pattern`: without a pass pattern the exit status is the verdict")
    if ("release_command" in text) != ("rehearsal_target" in text):
        problems.append("`release_command` and `rehearsal_target` are pinned together")
    inventory = read_json(INVENTORY) if INVENTORY.is_file() else []
    rows = inventory.get("tickets") if isinstance(inventory, dict) else inventory
    tickets = len(rows) if isinstance(rows, list) else 0
    if "acceptance_command" in text and "{ticket}" not in text["acceptance_command"] and tickets > 1:
        problems.append(f"`acceptance_command` has no `{{ticket}}` and the backlog holds {tickets} tickets: one command "
                        "cannot tell them apart")
    if problems:
        print(f"{ENVIRONMENT} cannot be used as it is:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    for key in ("acceptance_pass_pattern", "acceptance_fail_pattern"):
        if key in text:
            re.compile(text[key].replace("{ticket}", "ticket"))   # a pattern that does not compile stops here with its position
    for key in TEXT_KEYS + (PARALLEL_KEY,):
        print(f"  {key:<26} {settings[key] if key in settings else 'not pinned'}")
    if args.ticket and "acceptance_command" in text:
        result = acceptance(args.ticket, settings)
        print(f"acceptance check of {args.ticket}: read as {result['verdict']} (exit {result['exit']}; last line: {result['said']})")
        if not result["readable"]:
            print("its output matched neither pattern, or both: a pattern is wrong, or the command did not run the check")
            return 1
    return 0


# ---- baseline ---------------------------------------------------------------------------------------------------

def baseline(args):
    results = Path(args.results_dir)
    settings = environment()
    folder = results / args.ticket
    # The project is recorded by its path inside the repository: the same record is read in another checkout
    # of it (the control's export, the deployment's copy), where an absolute path would point back here.
    args.project_dir = os.path.relpath(os.path.realpath(args.project_dir), os.path.realpath("."))
    if args.project_dir.startswith(".."):
        print("--project-dir names a dbt project inside the repository this workspace holds")
        return 1
    datasets = args.select + args.snapshot
    specs = {name: spec_path(results, args.ticket, name) for name in datasets}
    missing = [str(path) for path in specs.values() if not path.is_file()]
    if missing:
        print(f"every dataset of the ticket has its invariants spec before the baseline; missing: {', '.join(missing)} "
              "(the invariants skill says how; the spec is the oracle and is written before the change)")
        return 1
    misnamed = [str(path) for name, path in specs.items() if read_json(path).get("model") != name]
    if misnamed:
        print(f"a spec names the dataset its file is for (`model`): {', '.join(misnamed)}")
        return 1
    previous = read_json(folder / "baseline.json") if (folder / "baseline.json").is_file() else None
    if previous is not None and not args.again:
        print(f"{args.ticket} has a baseline, taken {previous['context']['taken_at']}. Another one replaces the measurement of the code as it "
              "stood, so it is taken only when the spec was wrong, with the reason: --again \"<what was wrong>\"")
        return 1
    head_code, head = git("rev-parse", "HEAD")
    already_changed = changed_since("HEAD") if head_code == 0 else None
    if already_changed is None:
        print("the workspace is not a git repository with a commit: a baseline is tied to the commit it measures")
        return 1
    # What already differs from the commit is recorded: a file edited before its baseline was measured as
    # edited, so a proof may not name it as this ticket's change.
    record = {"tool": "evidence.py baseline", "ticket": args.ticket, "kind": args.kind, "project_dir": args.project_dir,
              "select": args.select, "snapshots": args.snapshot,
              "specs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in specs.items()},
              "context": {"taken_at": stamp(), "git_head": head.strip(), "changed_already": already_changed,
                          "reference_target": settings.get("reference_target"), "build_target": settings.get("build_target")}}
    if previous is not None:
        record["retaken"] = previous.get("retaken", []) + [{
            "taken_at": previous["context"]["taken_at"], "git_head": previous["context"]["git_head"],
            "specs": {name: entry["sha256"] for name, entry in previous["specs"].items()}, "reason": args.again}]
    reset_command = settings.get("reset_command")
    if reset_command:
        code, output = run(shlex.split(reset_command))
        record["reset"] = {"command": reset_command, "exit": code, "said": output.strip().splitlines()[-1][:300] if output.strip() else ""}
        print(f"reset: {record['reset']['said']}")
        if code != 0:
            print("the build namespace could not be put back to the reference state; nothing was measured")
            return 1
    else:
        record["reset"] = None
        print("no reset command in prepared/environment.json: the build namespace is measured as it stands")
    build_target = settings.get("build_target")
    code_state = {"status": "not_attempted"}
    if args.kind != "new" and not args.no_code_build:
        steps, statuses, _messages = build(args.project_dir, args.select, args.snapshot, False, build_target)
        failed = [step for step in steps if step["exit"] != 0]
        if not failed and not steps[0]["recorded"]:
            print(f"dbt left no record of `{steps[0]['command']}` at {Path(args.project_dir) / 'target' / 'run_results.json'}: these scripts read "
                  "dbt's own record where dbt writes it by default, and nothing was measured")
            return 1
        if failed:
            errors = [line for step in failed for line in step["errors"]]
            if not any(name in " ".join(errors) for name in datasets):
                print("the project does not build as it stands, for a reason dbt does not put on this ticket's datasets: "
                      + " | ".join(errors)[:500])
                print("Fix that first, as a change of its own, and take this baseline on the commit that builds. When the failing "
                      "code is this ticket's own and dbt does not name it, take the baseline with --no-code-build: the before-state is then "
                      "read from the reference alone.")
                return 1
            code_state = {"status": "does_not_build", "errors": errors}
            print("the code as it stands does not build: " + " | ".join(errors)[:400])
        else:
            code_state = {"status": "built", "resources": {name: statuses.get(name, "not built: no such resource yet") for name in datasets},
                          "invariants": {}}
            for name, path in specs.items():
                if statuses.get(name) != "success":
                    print(f"code as it stands, {name}: {code_state['resources'][name]}")
                    continue
                out = folder / f"invariants_before_code_{name}.json"
                report, error = invariants(path, args.project_dir, build_target, out)
                code_state["invariants"][name] = {"path": str(out), "sha256": sha256(out)} if report else {"error": error}
                print(f"code as it stands, {name}: " + (f"{report['passed']} pass, {report['failed']} fail, {report['errored']} error" if report else error))
    record["code_as_it_stands"] = code_state
    stale = folder / "evidence.json"
    if stale.is_file():
        stale.unlink()   # measured against the baseline this one replaces
    write_json(folder / "baseline.json", record)
    print(f"written {folder / 'baseline.json'}; now make the change, then: evidence.py prove --ticket {args.ticket}")
    return 0


# ---- prove ------------------------------------------------------------------------------------------------------

def reference_relation(name, project_dir, target):
    """(the fully named relation a dataset has on a target, error)."""
    rows, error = listed([name], project_dir, target)
    if rows is None:
        return None, error
    match = [row for row in rows if row["name"] == name]
    if len(match) != 1:
        return None, f"{name} resolves to {len(match)} resources on {target}"
    row = match[0]
    return ".".join(part for part in (row.get("database"), row.get("schema"), (row.get("config") or {}).get("alias") or name) if part), None


def measure_after(ticket, taken, results, with_downstream, settings):
    """Put the namespace back to the reference state, build the selection and measure it and the reference:
    the specs on both, the fingerprints of both, the rows by `except` when the kind promises an unchanged
    output. Every measurement of a candidate starts from that same state, whatever was built by hand since
    the baseline. Returns (evidence, error); every report it writes goes into the evidence's `artifacts`
    with its SHA-256."""
    folder = results / ticket
    folder.mkdir(parents=True, exist_ok=True)
    project_dir, models, snapshots = taken["project_dir"], taken["select"], taken["snapshots"]
    datasets = models + snapshots
    build_target, reference_target = settings.get("build_target"), settings.get("reference_target")
    for name in datasets:
        entry = taken["specs"].get(name)
        if entry is None or not Path(entry["path"]).is_file() or sha256(entry["path"]) != entry["sha256"]:
            return None, (f"the spec of {name} is missing or changed since the baseline: the oracle is fixed before the change. "
                          "When the spec was wrong, take the baseline again and say why (--again)")
    changed = changed_since(taken["context"]["git_head"])
    if changed is None:
        return None, f"the baseline's commit {taken['context']['git_head'][:12]} is not in this repository: the measurement cannot be tied to the code"
    if settings.get("reset_command"):
        code, output = run(shlex.split(settings["reset_command"]))
        if code != 0:
            return None, f"the namespace could not be put back to the reference state before measuring: {output.strip()[-300:]}"
    steps, statuses, messages = build(project_dir, models, snapshots, with_downstream, build_target)
    unrecorded = [step["command"] for step in steps if not step["recorded"]]
    if unrecorded:
        return None, (f"dbt left no record of `{unrecorded[0]}` at {Path(project_dir) / 'target' / 'run_results.json'} (exit {steps[0]['exit']}): "
                      + (" | ".join(steps[0]["errors"])[:500] or "these scripts read dbt's own record where dbt writes it by default"))
    not_built = [name for name in datasets if statuses.get(name) != "success"]
    unexplained = [step["command"] for step in steps if step["exit"] != 0 and not step["failed"]]
    if not_built or unexplained:
        return None, ((f"the build of {', '.join(not_built)} did not succeed: " if not_built
                       else f"`{unexplained[0]}` exited with an error and dbt recorded no failed resource: ")
                      + " | ".join(line for step in steps for line in step["errors"])[:600])
    evidence = {"tool": "evidence.py prove", "ticket": ticket, "kind": taken["kind"], "project_dir": project_dir,
                "select": models, "snapshots": snapshots, "build": steps, "with_downstream": with_downstream,
                "context": {"taken_at": stamp(), "build_target": build_target, "reference_target": reference_target,
                            "baseline_taken_at": taken["context"]["taken_at"], "baseline_retaken": len(taken.get("retaken", []))},
                "code": {"since": taken["context"]["git_head"], "files": changed, "changed_before_the_baseline": taken["context"]["changed_already"]},
                "specs": taken["specs"], "datasets": {}, "artifacts": {}}

    def keep(path):
        evidence["artifacts"][str(path)] = sha256(path)

    selector = " ".join(f"{name}+" if with_downstream else name for name in datasets)
    after_prints, error = fingerprints(selector, project_dir, build_target, folder / "fingerprints_after.json")
    if after_prints is None:
        return None, f"the build could not be fingerprinted: {error}"
    keep(folder / "fingerprints_after.json")
    reference_prints, reference_error = (None, "no reference target is pinned")
    if reference_target:
        reference_prints, reference_error = fingerprints(selector, project_dir, reference_target, folder / "fingerprints_reference.json")
        if reference_prints is not None:
            keep(folder / "fingerprints_reference.json")
    rows, error = listed(datasets, project_dir, build_target)
    if rows is None:
        return None, f"the datasets could not be listed: {error}"
    # A load is what keeps state between runs, however it was selected: a snapshot, an incremental model.
    evidence["loads"] = sorted(set(snapshots) | {row["name"] for row in rows if row.get("resource_type") == "snapshot"
                                                 or (row.get("config") or {}).get("materialized") == "incremental"})
    # A baseline retaken at another commit measured code that already held the change: its before-state is not used.
    first_head = (taken.get("retaken") or [{"git_head": taken["context"]["git_head"]}])[0]["git_head"]
    evidence["code"]["first_baseline"] = first_head
    stood = taken["code_as_it_stands"]
    evidence["code_as_it_stood"] = {"status": stood["status"], "errors": stood.get("errors", []),
                                    "usable": first_head == taken["context"]["git_head"]}
    for name in datasets:
        built_print = after_prints["models"].get(name, {})
        if built_print.get("rows") is None:
            return None, f"{name} was built and could not be measured: {built_print.get('error', 'no fingerprint')}"
        held = (reference_prints or {"models": {}})["models"].get(name)
        entry = {"built_rows": built_print["rows"]}
        if held is None:
            entry["reference"] = f"unmeasured: {reference_error or 'the reference fingerprint does not list it'}"
        elif held.get("absent"):
            entry["reference"] = "absent"
        elif held.get("rows") is None:
            entry["reference"] = f"unmeasured: {held.get('error', 'no fingerprint')}"
        else:
            entry["reference"] = "measured"
            differ = sorted(key for key in set(built_print["columns"]) | set(held["columns"]) if built_print["columns"].get(key) != held["columns"].get(key))
            entry["against_reference"] = {"rows_reference": held["rows"], "rows_built": built_print["rows"], "columns_that_differ": differ,
                                          "columns_on_one_side_only": sorted(set(built_print["columns"]) ^ set(held["columns"])),
                                          "changed": bool(differ) or built_print["rows"] != held["rows"]}
        spec = taken["specs"][name]["path"]
        after, error = invariants(spec, project_dir, build_target, folder / f"invariants_after_{name}.json")
        if after is None:
            return None, f"the spec of {name} could not be run on the build: {error}"
        keep(folder / f"invariants_after_{name}.json")
        if entry["reference"] == "measured":
            reference, error = invariants(spec, project_dir, reference_target, folder / f"invariants_reference_{name}.json")
            if reference is None:
                entry["reference"] = f"unmeasured: the spec could not be run on the reference: {error}"
            else:
                keep(folder / f"invariants_reference_{name}.json")
        code_run = stood.get("invariants", {}).get(name, {})
        if evidence["code_as_it_stood"]["usable"] and code_run.get("path") and Path(code_run["path"]).is_file() and sha256(code_run["path"]) == code_run["sha256"]:
            keep(code_run["path"])
        if taken["kind"] in EQUAL_OUTPUT_KINDS and entry["reference"] == "measured":
            relation, error = reference_relation(name, project_dir, reference_target)
            if relation is not None:
                out = folder / f"equivalence_{name}.json"
                if out.exists():
                    out.unlink()
                run([sys.executable, str(FINGERPRINT), "--except", "{{ ref('%s') }}" % name, relation, "--project-dir", str(project_dir),
                     "--save", str(out)] + (["--target", build_target] if build_target else []))
                if out.is_file():
                    keep(out)
        evidence["datasets"][name] = entry
    downstream, error = downstream_of(datasets, project_dir, build_target)
    if downstream is None:
        return None, f"the datasets downstream could not be listed: {error}"
    # A downstream dataset is screened by its fingerprint, not compared row by row: the same fingerprint is
    # recorded as that, never as "unchanged".
    compared = {"names": downstream, "rebuilt": with_downstream, "changed": [], "same_fingerprint": [], "unmeasured": []}
    if with_downstream:
        for name in downstream:
            built_print, held = after_prints["models"].get(name, {}), (reference_prints or {"models": {}})["models"].get(name, {})
            if statuses.get(name) != "success":
                compared["unmeasured"].append({"model": name, "status": statuses.get(name, "not run"), "error": (messages.get(name) or "")[:400],
                                               "why": f"dbt did not build it ({statuses.get(name, 'not run')})"})
            elif built_print.get("rows") is None or held.get("rows") is None:
                compared["unmeasured"].append({"model": name, "why": (built_print.get("error") or held.get("error") or reference_error or "not measured")[:200]})
            elif built_print["rows"] != held["rows"] or built_print["columns"] != held["columns"]:
                compared["changed"].append({"model": name, "rows_reference": held["rows"], "rows_built": built_print["rows"],
                                            "columns_that_differ": sorted(k for k in set(built_print["columns"]) | set(held["columns"])
                                                                          if built_print["columns"].get(k) != held["columns"].get(k))})
            else:
                compared["same_fingerprint"].append(name)
    evidence["downstream"] = compared
    return evidence, None


def prove(args):
    results = Path(args.results_dir)
    folder = results / args.ticket
    if not (folder / "baseline.json").is_file():
        print(f"no baseline for {args.ticket}: run `evidence.py baseline` before changing anything (it resets the namespace and fixes the spec)")
        return 1
    current = folder / "evidence.json"
    if current.is_file():
        current.unlink()   # a run that fails must not leave the last run's evidence standing for this code
    settings = environment()
    # A check the setup says builds a state of its own, away from this checkout (it pinned how many run at
    # once), runs beside the measurement; any other may read the build namespace, so it runs once the
    # measurement has built it and before the control for a broken consumer replaces that build.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        accepting = executor.submit(acceptance, args.ticket, settings) if PARALLEL_KEY in settings else None
        evidence, error = measure_after(args.ticket, read_json(folder / "baseline.json"), results, args.with_downstream, settings)
        accepted = accepting.result() if accepting is not None else (acceptance(args.ticket, settings) if evidence is not None else None)
    if evidence is None:
        print(error)
        return 1
    control_for_broken_consumers(evidence, evidence["code"]["first_baseline"], settings)
    evidence["artifacts"][str(folder / "baseline.json")] = sha256(folder / "baseline.json")
    evidence["acceptance"] = accepted
    write_json(current, evidence)
    measured = measured_checks(evidence)
    for name, entry in evidence["datasets"].items():
        line = f"{name}: built {entry['built_rows']} rows"
        if "against_reference" in entry:
            compared = entry["against_reference"]
            line += f"; reference {compared['rows_reference']} rows, " + ("differs in " + ", ".join(compared["columns_that_differ"][:6]) if compared["changed"] else "identical fingerprint")
        else:
            line += f"; reference {entry['reference']}"
        print(line)
        checks = measured.get(name)
        if checks:
            print(f"  checks after: {len(checks['after']['pass'])} pass, {len(checks['after']['fail'])} fail, {len(checks['after']['error'])} error; "
                  f"moved {len(checks['moved'])}, preserved {len(checks['preserved'])}, regressed {len(checks['regressed'])}")
    down = evidence["downstream"]
    print(f"downstream: {len(down['names'])} dataset(s)" + (f", {len(down['changed'])} changed, {len(down['same_fingerprint'])} with the same fingerprint, {len(down['unmeasured'])} unmeasured"
                                                              if down["rebuilt"] else (" (not rebuilt: --with-downstream rebuilds and compares them)" if down["names"] else "")))
    for item in down["unmeasured"]:
        print(f"  not measured: {item['model']}: {item['why']}"
              + (f"; built from the code as it stood at the baseline, dbt records: {item['on_the_code_as_it_stood']['status']}" if "on_the_code_as_it_stood" in item else ""))
    if evidence.get("namespace_holds_the_control"):
        print("  the build namespace was reset and built from the code as it stood for that comparison: build the ticket's datasets again to look at them")
    if accepted:
        print(f"acceptance check: {accepted['verdict']} ({accepted['said']})")
    print(f"written {current}")
    label, reasons, _ = earned(args.ticket, results)
    print(f"label earned as the proof stands: {label}")
    for reason in reasons:
        print(f"  - {reason}")
    return 0


# ---- the label --------------------------------------------------------------------------------------------------

def ledger_rows(results):
    ledger = results / "backlog.json"
    if not ledger.is_file():
        return {}
    document = read_json(ledger)
    rows = document.get("tickets") if isinstance(document, dict) else document
    return {row.get("ticket"): row for row in rows if isinstance(row, dict)} if isinstance(rows, list) else {}


def proof_path(ticket, results):
    """A backlog keeps one proof per ticket under results/proofs/; a single ticket keeps results/proof.json."""
    return results / "proof.json" if ticket == SINGLE_TICKET and not (results / "proofs" / f"{ticket}.json").is_file() else results / "proofs" / f"{ticket}.json"


def tickets_with_proofs(results):
    """Every ticket that has a proof: the backlog's files, or the single ticket."""
    found = sorted(path.stem for path in (results / "proofs").glob("*.json"))
    return found or ([SINGLE_TICKET] if (results / "proof.json").is_file() else [])


def independent_names(spec, dataset):
    """The names of the checks that compare the dataset with something other than itself: a reconciliation, a
    completeness or a custom check whose SQL reads, in a FROM or a JOIN, at least one relation that is not the
    dataset (comments and the query's own named subqueries left out). A check that reads nothing, or the
    dataset alone, tests the dataset against its own output. This reads the SQL's shape, not its meaning:
    whether the other relation is a fair oracle is a reviewer's judgement."""
    own = re.compile(r"\{\{\s*ref\(\s*['\"]%s['\"]\s*\)\s*\}\}" % re.escape(dataset))
    names = set()
    for kind in INDEPENDENT_CHECKS:
        for item in spec.get(kind) or []:
            sql = " ".join(str(item.get(key) or "") for key in ("left", "right", "violations"))
            sql += " from " + str(item["source"]) if item.get("source") else ""
            sql = own.sub("the_dataset_itself", SQL_COMMENT.sub(" ", sql))
            inner = {name.lower() for name in NAMED_SUBQUERY.findall(sql)} | {"the_dataset_itself"}
            others = [read for read in READS.findall(sql) if read.lower() not in inner]
            # A completeness check without a name runs under the one the invariants runner gives it.
            name = item.get("name") or (f"completeness {item['model_key']}" if kind == "completeness" and item.get("model_key") else None)
            if name and others:
                names.add(name)
    return names


def artifact(evidence, filename):
    """The path of the evidence's report of this name, when its hash is recorded and the file is there."""
    for path in evidence.get("artifacts", {}):
        if Path(path).name == filename and Path(path).is_file():
            return Path(path)
    return None


def measured_checks(evidence):
    """Per dataset, from the reports the evidence names by hash: the checks after the change, and which moved
    from failing to passing, which were preserved, which regressed. The reports are the source; a report that
    was not run on the recorded spec counts for nothing, and a check a before-report has and the after-report
    lacks counts as regressed."""
    out = {}
    for name in evidence.get("select", []) + evidence.get("snapshots", []):
        after_file = artifact(evidence, f"invariants_after_{name}.json")
        if after_file is None:
            continue
        after_report = read_json(after_file)
        if after_report.get("spec_sha256") != evidence.get("specs", {}).get(name, {}).get("sha256"):
            continue
        now = report_statuses(after_report)
        before = {}
        reference_file = artifact(evidence, f"invariants_reference_{name}.json")
        if reference_file is not None:
            before["reference"] = report_statuses(read_json(reference_file))
        code_file = artifact(evidence, f"invariants_before_code_{name}.json")
        if code_file is not None and evidence.get("code_as_it_stood", {}).get("usable"):
            before["code_as_it_stood"] = report_statuses(read_json(code_file))
        bad_before = {check for statuses in before.values() for check, status in statuses.items() if status != "pass"}
        seen_before = {check for statuses in before.values() for check in statuses}
        good_before = {check for check in seen_before if all(statuses.get(check, "pass") == "pass" for statuses in before.values())}
        spec_file = Path(evidence["specs"][name]["path"]) if name in evidence.get("specs", {}) else None
        spec = read_json(spec_file) if spec_file is not None and spec_file.is_file() else {}
        out[name] = {
            "after": {status: sorted(check for check, value in now.items() if value == status) for status in ("pass", "fail", "error")},
            "measured_before_on": sorted(before),
            "moved": sorted(check for check in bad_before if now.get(check) == "pass"),
            "preserved": sorted(check for check in good_before if now.get(check) == "pass"),
            "regressed": sorted(check for check in good_before if now.get(check) != "pass"),
            "independent": sorted(check for check in now if check in independent_names(spec, name)),
            "has_grain": bool(spec.get("grain")),
        }
    return out


def earned(ticket, results, evidence_file=None):
    """(label, reasons, obligations): what the proof and the evidence of one ticket earn.

    `reasons` says why the label is not higher (or, for a safe label, what carries it); `obligations` is the
    table, one row per obligation with met true or false and the artifact or the gap."""
    proof_file = proof_path(ticket, results)
    proof = read_json(proof_file) if proof_file.is_file() else {}
    rows = []

    def obligation(text, met, by):
        rows.append({"obligation": text, "met": bool(met), "by": by})

    def verdict():
        unmet = [f"{row['obligation']}: {row['by']}" for row in rows if not row["met"]]
        return (UNSURE, unmet, rows) if unmet else (SAFE, [f"{row['obligation']}: {row['by']}" for row in rows], rows)

    if proof.get("change_kind") == "no_change":
        files = proof.get("decision_evidence") or []
        held = bool(files) and all(isinstance(item, str) and Path(item).is_file() for item in files)
        obligation("decision evidence names existing files", held, ", ".join(files) or "none")
        obligation("no file changed", proof.get("changed_files") == [], str(proof.get("changed_files")))
        return (WONT_DO if all(row["met"] for row in rows) else UNSURE), [row["obligation"] for row in rows if not row["met"]], rows
    evidence_file = Path(evidence_file) if evidence_file else results / ticket / "evidence.json"
    if not evidence_file.is_file():
        obligation("the change was measured (evidence.py baseline, then prove)", False, f"{evidence_file} does not exist")
        return UNSURE, ["nothing was measured: no baseline and no proof run"], rows
    evidence = read_json(evidence_file)
    kind = evidence.get("kind")
    datasets = evidence.get("select", []) + evidence.get("snapshots", [])
    artifacts = evidence.get("artifacts", {})
    whole = (evidence.get("tool") == "evidence.py prove" and kind in KINDS and bool(datasets) and bool(artifacts)
             and any(Path(path).name == "baseline.json" for path in artifacts) and isinstance(evidence.get("datasets"), dict)
             and isinstance(evidence.get("downstream"), dict) and isinstance(evidence.get("code"), dict)
             and all(key in evidence["downstream"] for key in ("names", "rebuilt", "changed", "same_fingerprint", "unmeasured")))
    obligation("the evidence is a whole record of a prove run", whole, "as prove wrote it" if whole else "parts of the record are missing")
    if not whole:
        return verdict()

    cause = proof.get("root_cause") if isinstance(proof.get("root_cause"), dict) else {}
    if not proof:
        obligation("the proof is written", False, f"{proof_file} does not exist")
    else:
        obligation("the proof's kind is the kind the baseline was taken for", proof.get("change_kind") == kind,
                   f"{kind}" if proof.get("change_kind") == kind else f"the proof says {proof.get('change_kind')!r}, the baseline {kind!r}")
        if kind == "fix":
            obligation("the fix names its cause with the file and the reproducing query", bool(cause.get("file") and cause.get("reproducing_query")),
                       str(cause.get("file") or "no root_cause.file and reproducing_query in the proof"))
    stale = [path for path, digest in artifacts.items() if not Path(path).is_file() or sha256(path) != digest]
    obligation("the reports are the ones the measurement wrote", not stale, "all hashes match" if not stale else f"changed or missing since: {', '.join(stale[:4])}")
    drifted = [name for name in datasets if name not in evidence.get("specs", {}) or not Path(evidence["specs"][name]["path"]).is_file()
               or sha256(evidence["specs"][name]["path"]) != evidence["specs"][name]["sha256"]]
    obligation("every dataset has its spec, unchanged since the baseline", not drifted, "unchanged" if not drifted else f"missing or changed: {', '.join(drifted)}")

    named = [str(path) for path in (proof.get("changed_files") or [])]
    files = evidence["code"].get("files") or {}
    before = evidence["code"].get("changed_before_the_baseline") or {}
    # What the measurement saw changed in the ticket's project since its baseline. Each of those files, and
    # each file the proof names, must still be as it was measured: a macro edited afterwards changes what
    # the measurement would say, whether the proof names it or not.
    project = os.path.normpath(evidence.get("project_dir") or ".")
    since = {path for path in files if (project == "." or path.startswith(project + "/")) and files[path] != before.get(path)}
    moved_on = sorted(path for path in set(named) | since if files.get(path) != (sha256(path) if Path(path).is_file() else "deleted"))
    obligation("the code is as it was measured, and the files the proof names are part of it", bool(named) and not moved_on,
               f"{len(named)} file(s) named, {len(since)} measured" if named and not moved_on
               else ("the proof names no changed file" if not named else f"changed since the measurement, or not part of it: {', '.join(moved_on[:4])}"))
    # Every one of those files is some delivered ticket's change: this proof names it, or the proof of
    # another closed ticket does (a ticket worked later, measured here because this ticket was measured
    # again). A change no closed ticket's proof names was measured with this ticket and would ship with none.
    ledger = ledger_rows(results)
    elsewhere = {str(path) for other in tickets_with_proofs(results)
                 if other != ticket and (not ledger or ledger.get(other, {}).get("status") == "done")
                 for path in (read_json(proof_path(other, results)).get("changed_files") or [])}
    unowned = sorted(since - set(named) - elsewhere)
    obligation("every file changed in the project since the baseline is named by a closed ticket's proof", not unowned,
               "all named" if not unowned else f"changed since the baseline and in no closed ticket's `changed_files`: {', '.join(unowned[:4])}")
    if kind != "new":
        # The same content before the baseline and now means the baseline measured the change, not what stood before it.
        early = [path for path in named if path in before and before[path] == files.get(path)]
        obligation("the baseline was taken before the ticket's files changed", not early,
                   "they changed after it" if not early else f"already as delivered when the baseline was taken: {', '.join(early[:4])}")

    checks = measured_checks(evidence)
    unbuilt = [name for name in datasets if not isinstance(evidence["datasets"].get(name, {}).get("built_rows"), int)]
    failed_steps = [step["command"] for step in evidence.get("build", []) if step.get("exit") != 0 and not evidence.get("with_downstream")]
    unchecked = [name for name in datasets if name not in checks or not any(checks[name]["after"].values())]
    obligation("every dataset was built and has its checks run", not unbuilt and not failed_steps and not unchecked,
               ", ".join(datasets) if not (unbuilt or failed_steps or unchecked) else f"not built or not checked: {', '.join(sorted(set(unbuilt + unchecked))) or failed_steps[0]}")
    not_passing = [f"{name}: {check}" for name, entry in checks.items() for check in entry["after"]["fail"] + entry["after"]["error"]]
    obligation("every check passes after the change", not unchecked and not not_passing, "all pass" if not not_passing else "; ".join(not_passing[:4]))
    regressed = [f"{name}: {check}" for name, entry in checks.items() for check in entry["regressed"]]
    obligation("no check that passed before fails after", not regressed, "none" if not regressed else "; ".join(regressed[:4]))
    unmeasured_reference = [f"{name} ({entry.get('reference')})" for name, entry in evidence["datasets"].items()
                            if str(entry.get("reference", "unmeasured")).startswith("unmeasured")]
    obligation("what the reference holds was measured, or shown absent", not unmeasured_reference,
               "measured" if not unmeasured_reference else "; ".join(unmeasured_reference)[:300])
    if kind == "fix":
        moved = [f"{name}: {check}" for name, entry in checks.items() for check in entry["moved"]]
        stood = evidence["code_as_it_stood"]
        own_failure = stood.get("usable") and stood.get("status") == "does_not_build" and any(name in " ".join(stood.get("errors", [])) for name in datasets)
        obligation("a check failed before the change and passes after it, or the ticket's own code did not build and now does", bool(moved) or own_failure,
                   "; ".join(moved[:4]) if moved else ("the code as it stood did not build: " + " | ".join(stood.get("errors", [])[:2])[:240]
                                                       if own_failure else "no check moved: the spec does not see the defect"))
    # A dataset with nothing before it to compare with (the kind is `new`, or production does not hold it) is
    # grounded by its declared grain and a passing check that reads another relation: every such dataset.
    fresh = [name for name in datasets if kind == "new" or evidence["datasets"].get(name, {}).get("reference") == "absent"]
    ungrounded = [name for name in fresh if not (checks.get(name, {}).get("has_grain") and set(checks[name]["independent"]) & set(checks[name]["after"]["pass"]))]
    if fresh:
        obligation("every new dataset has its grain declared and passes a check against another relation", not ungrounded,
                   ", ".join(fresh) if not ungrounded
                   else f"no grain, or no passing reconcile, completeness or custom check that reads another relation: {', '.join(ungrounded)}")
    if kind in EQUAL_OUTPUT_KINDS:
        # A row comparison goes by position: a renamed column, or one that turned from a number into text,
        # compares equal. The fingerprints carry every column's name and kind, so the structure is read there.
        reshaped = [f"{name}: {', '.join(entry['against_reference']['columns_on_one_side_only'][:4])}" for name, entry in evidence["datasets"].items()
                    if (entry.get("against_reference") or {}).get("columns_on_one_side_only")]
        unshaped = [name for name, entry in evidence["datasets"].items() if "columns_on_one_side_only" not in (entry.get("against_reference") or {})]
        obligation("the output has the reference's columns, by name and kind", not reshaped and not unshaped,
                   "the same columns" if not reshaped and not unshaped
                   else "; ".join(reshaped[:3]) or f"not compared with the reference: {', '.join(unshaped[:4])}")
        unequal = []
        for name in datasets:
            report_file = artifact(evidence, f"equivalence_{name}.json")
            report = read_json(report_file) if report_file is not None else None
            if report is None:
                unequal.append(f"{name}: no row comparison with the reference")
            elif report["only_in_a"] or report["only_in_b"] or report["rows_a"] != report["rows_b"]:
                unequal.append(f"{name}: {report['only_in_a']} row(s) only in the build, {report['only_in_b']} only in the reference")
            elif not (report["rows_a"] == report.get("distinct_a") and report["rows_b"] == report.get("distinct_b")):
                unequal.append(f"{name}: duplicate rows on one side, which a set comparison does not count")
        obligation("the output equals the reference's, row by row", not unequal, "no row differs, and no row is duplicated" if not unequal else "; ".join(unequal[:3]))

    down = evidence["downstream"]
    reasons_given = {str(item.get("model")): str(item.get("reason") or "").strip()
                     for item in ((proof.get("blast_radius") or {}).get("changed_models") or []) if isinstance(item, dict)} \
        if isinstance(proof.get("blast_radius"), dict) else {}
    accounted = {item["model"] for item in down["changed"]} | set(down["same_fingerprint"]) | {item["model"] for item in down["unmeasured"]}
    if not down["names"]:
        obligation("every changed downstream dataset is listed with a reason", True, "nothing reads these datasets")
    elif not down["rebuilt"]:
        obligation("every changed downstream dataset is listed with a reason", False,
                   f"{len(down['names'])} dataset(s) read these and were not rebuilt and compared (prove --with-downstream)")
    elif accounted != set(down["names"]):
        obligation("every changed downstream dataset is listed with a reason", False,
                   f"{len(set(down['names']) - accounted)} of {len(down['names'])} downstream dataset(s) have no comparison at all")
    else:
        # A reason covers one dataset by its name, or a family by a name prefix of three characters or more
        # followed by `*`; no other pattern is read.
        exact = {name.lower(): reason for name, reason in reasons_given.items() if reason and "*" not in name}
        prefixes = [name[:-1].lower() for name, reason in reasons_given.items() if reason and FAMILY.match(name)]
        unexplained = [item["model"] for item in down["changed"]
                       if item["model"].lower() not in exact and not any(item["model"].lower().startswith(prefix) for prefix in prefixes)]
        # A consumer dbt could not build after the change was not compared. It is set aside only when its own
        # build failed and the same selection, built from the code as it stood at the baseline, failed on it
        # with the same error: it cannot be built with or without this change. A consumer that was skipped,
        # a control that was not taken, or another error leaves it unmeasured.
        broken_before = [item["model"] for item in down["unmeasured"]
                         if item.get("status") == "error" and isinstance(item.get("on_the_code_as_it_stood"), dict)
                         and item["on_the_code_as_it_stood"].get("status") == "error"
                         and same_failure(item.get("error"), item["on_the_code_as_it_stood"].get("error"))]
        unmeasured = [item["model"] + (f" (from the code as it stood: {item['on_the_code_as_it_stood']['status']})"
                                       if isinstance(item.get("on_the_code_as_it_stood"), dict) else "")
                      for item in down["unmeasured"] if item["model"] not in broken_before]
        obligation("every changed downstream dataset is listed with a reason", not unexplained and not unmeasured,
                   f"{len(down['changed'])} changed, all explained; {len(down['same_fingerprint'])} with the same fingerprint"
                   + (f"; failing with the same error on the code as it stood: {', '.join(broken_before[:6])}" if broken_before else "")
                   if not unexplained and not unmeasured
                   else f"changed without a reason in the proof: {', '.join(unexplained[:6]) or 'none'}; could not be rebuilt and compared: {', '.join(unmeasured[:6]) or 'none'}")

    validated = proof.get("validated") if isinstance(proof.get("validated"), dict) else {}
    assumptions = [str(item) for item in (validated.get("assumptions") or []) if str(item).strip()]
    obligation("no assumption is left open", not assumptions, "none" if not assumptions else "; ".join(assumptions[:3]))
    unknowns = [str(item) for item in (proof.get("unknowns") or []) if str(item).strip()]
    obligation("no question is left for whoever asked for the change", not unknowns, "none" if not unknowns else "; ".join(unknowns[:3]))

    if evidence.get("loads") or kind == "migration":
        what = ", ".join(evidence.get("loads") or []) or "a migration"
        rerun = proof.get("rerun")
        replay = read_json(rerun) if isinstance(rerun, str) and Path(rerun).is_file() else None
        # No row only on one side, the same number of rows, no duplicate row on either side (a set comparison
        # does not see duplicates), and taken for this ticket: after its baseline.
        replay_holds = (isinstance(replay, dict) and replay.get("tool") == "fingerprint.py except" and replay.get("only_in_a") == 0
                        and replay.get("only_in_b") == 0 and isinstance(replay.get("rows_a"), int)
                        and replay["rows_a"] == replay.get("rows_b") == replay.get("distinct_a") == replay.get("distinct_b")
                        and str((replay.get("context") or {}).get("taken_at", "")) >= str(evidence["context"].get("baseline_taken_at")))
        if evidence.get("loads"):
            obligation("a changed load has its replay compared row by row", replay_holds,
                       f"{rerun}: no row differs" if replay_holds
                       else f"{what}: the proof's `rerun` names no `fingerprint.py --except --save` report, taken after the baseline, with no difference and no duplicate row")
        runbook = proof.get("runbook")
        written = isinstance(runbook, dict) and all(isinstance(runbook.get(key), list) for key in RUNBOOK_KEYS) and any(runbook.get(key) for key in RUNBOOK_KEYS)
        obligation("a changed load or a migration ships its runbook", written,
                   "in the proof" if written else f"{what}: the proof's `runbook` needs {', '.join(RUNBOOK_KEYS)} (lists), at least one filled")

    if environment().get("acceptance_command"):
        accepted = evidence.get("acceptance") or {}
        obligation("the task's acceptance check passes", accepted.get("verdict") == "pass",
                   f"{accepted['verdict']}: {accepted.get('said')}" if accepted else "not run: prove runs it and records the verdict")
    return verdict()


def label(args):
    results = Path(args.results_dir)
    tickets = [args.ticket] if args.ticket else tickets_with_proofs(results)
    report = {}
    for ticket in tickets:
        # --evidence reads one ticket's label from another measurement of it than the attempt's own:
        # verification's, which is the one a pull request is published on.
        evidence_file = Path(args.evidence) if args.evidence else results / ticket / "evidence.json"
        earned_label, reasons, rows = earned(ticket, results, evidence_file=evidence_file)
        report[ticket] = {"label": earned_label, "reasons": reasons, "obligations": rows,
                          "checks": measured_checks(read_json(evidence_file)) if evidence_file.is_file() else {}}
        proof_file = proof_path(ticket, results)
        if args.write and proof_file.is_file():
            proof = read_json(proof_file)
            proof["label"], proof["label_reasons"], proof["label_computed_by"] = earned_label, reasons, "evidence.py label"
            write_json(proof_file, proof)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    if not report:
        print(f"no proof under {results}: nothing to label")
    for ticket, entry in report.items():
        print(f"{ticket}: {entry['label']}")
        for row in entry["obligations"]:
            print(f"  {'met    ' if row['met'] else 'MISSING'} {row['obligation']}: {row['by'][:200]}")
    return 0


# ---- verification ----------------------------------------------------------------------------------------------

def verify(args):
    results = Path(args.results_dir)
    # The report and its measurements are read again at publication, from another working directory: every
    # path they hold of their own is absolute.
    args.out = str(Path(args.out).resolve())
    scratch = Path(args.out).parent / "verification_measurements"
    settings = environment()
    report = {"tool": "evidence.py verify", "taken_at": stamp(), "tickets": {}}
    # The closed tickets: the ledger's done and no_change rows, or without a ledger every proof by its kind.
    ledger = ledger_rows(results)
    kinds = ({ticket: {"done": "changed", "no_change": "no_change"}.get(row.get("status")) for ticket, row in ledger.items()} if ledger
             else {ticket: "no_change" if read_json(proof_path(ticket, results)).get("change_kind") == "no_change" else "changed"
                   for ticket in tickets_with_proofs(results)})
    done = [ticket for ticket, kind in kinds.items() if kind == "changed"]
    declined = [ticket for ticket, kind in kinds.items() if kind == "no_change"]
    again = [ticket for ticket in done if not args.only or ticket in args.only]
    report["not_measured_again"] = [ticket for ticket in done if ticket not in again]
    # A reviewer's findings recorded in an earlier report at this path survive a second run.
    earlier = read_json(args.out).get("tickets", {}) if Path(args.out).is_file() else {}
    # Checks the setup says build a state of their own, away from this checkout, run beside the
    # re-measurements, as many at once as it pinned; any other runs after its ticket's re-measurement, on the
    # namespace that measurement built and before a control replaces it.
    own_state, at_a_time = PARALLEL_KEY in settings, settings.get(PARALLEL_KEY, 1)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=at_a_time if isinstance(at_a_time, int) and at_a_time > 0 else 1)
    accepting = {ticket: executor.submit(acceptance, ticket, settings) for ticket in again} if own_state else {}
    for ticket in done + declined:
        proof_file = proof_path(ticket, results)
        claimed = read_json(proof_file).get("label") if proof_file.is_file() else None
        from_attempt, attempt_reasons, _ = earned(ticket, results)
        entry = {"claimed": claimed, "earned_from_the_attempt": from_attempt, "measured_again": False, "did_not_reproduce": []}
        report["tickets"][ticket] = entry
        if ticket in declined:
            # A no-change outcome has nothing to build: its decision evidence is the reviewers' to read.
            entry["earned"], entry["reasons"] = from_attempt, attempt_reasons
            continue
        if ticket not in again:
            entry["earned"] = UNSURE if from_attempt == SAFE else from_attempt
            entry["reasons"] = ["verification did not measure this ticket again"] + (attempt_reasons if from_attempt != SAFE else [])
            continue
        baseline_file, evidence_file = results / ticket / "baseline.json", results / ticket / "evidence.json"
        if not baseline_file.is_file() or not evidence_file.is_file():
            entry["earned"], entry["reasons"] = from_attempt, attempt_reasons
            print(f"{ticket}: claimed {claimed}, earned {from_attempt} (the attempt left nothing to measure again)")
            continue
        taken, attempt = read_json(baseline_file), read_json(evidence_file)
        fresh_folder = scratch / ticket
        if fresh_folder.exists():
            shutil.rmtree(fresh_folder)
        fresh, error = measure_after(ticket, taken, scratch, attempt.get("with_downstream", False), settings)
        if fresh is None:
            entry["earned"], entry["reasons"] = UNSURE, [f"the after-state could not be measured again: {error}"]
            entry["did_not_reproduce"].append(error)
            print(f"{ticket}: claimed {claimed}, earned {UNSURE}: {error[:160]}")
            continue
        fresh["artifacts"][str(baseline_file)] = sha256(baseline_file)
        fresh["acceptance"] = accepting[ticket].result() if own_state else acceptance(ticket, settings)
        control_for_broken_consumers(fresh, fresh["code"]["first_baseline"], settings)
        write_json(fresh_folder / "evidence.json", fresh)
        told, measured = measured_checks(attempt), measured_checks(fresh)
        for name, now in fresh["datasets"].items():
            then = attempt.get("datasets", {}).get(name, {})
            if now.get("built_rows") != then.get("built_rows"):
                entry["did_not_reproduce"].append(f"{name}: {now.get('built_rows')} rows now, {then.get('built_rows')} in the attempt's evidence")
            for status in ("pass", "fail", "error"):
                here, there = set(measured.get(name, {}).get("after", {}).get(status, [])), set(told.get(name, {}).get("after", {}).get(status, []))
                if here != there:
                    entry["did_not_reproduce"].append(f"{name}: checks that {status} differ: now {sorted(here ^ there)[:4]}")
        if (attempt.get("acceptance") or {}).get("verdict") == "pass" and (fresh["acceptance"] or {}).get("verdict") != "pass":
            entry["did_not_reproduce"].append(f"the acceptance check passed in the attempt and now says: {(fresh['acceptance'] or {}).get('said', 'nothing')}")
        # This measurement, taken on the code as it is delivered, is the ticket's after-state from here on: the
        # label is what it earns, and a pull request is published on it. The attempt's own measurement is
        # history. Where the two disagree on the same code, something is not stable (an input moved, a
        # build is not deterministic, a number was made up) and the label says so; where the code changed
        # in between (a later ticket's commit), a difference is expected and is only recorded.
        fresh_label, fresh_reasons, _ = earned(ticket, results, evidence_file=fresh_folder / "evidence.json")
        entry["same_code_as_the_attempt_measured"] = attempt.get("code", {}).get("files") == fresh["code"]["files"]
        if entry["did_not_reproduce"] and entry["same_code_as_the_attempt_measured"]:
            fresh_label, fresh_reasons = UNSURE, ["the attempt's measurement did not reproduce on the same code: "
                                                  + "; ".join(entry["did_not_reproduce"][:3])] + fresh_reasons
        entry["measured_again"] = True
        entry["evidence"] = str(fresh_folder / "evidence.json")
        entry["earned"] = fresh_label
        entry["reasons"] = fresh_reasons if fresh_label != SAFE else []
        if from_attempt != SAFE:
            entry["attempt_reasons"] = attempt_reasons
        print(f"{ticket}: claimed {claimed}, earned {entry['earned']}" + (f"; did not reproduce: {entry['did_not_reproduce'][0][:140]}" if entry["did_not_reproduce"] else ""))
    executor.shutdown()
    for ticket, entry in report["tickets"].items():
        findings = (earlier.get(ticket) or {}).get("lowered_by_review") or []
        if findings:
            entry["lowered_by_review"] = findings
            entry["earned"] = UNSURE if entry.get("earned") in (SAFE, WONT_DO) else entry.get("earned")
            entry["reasons"] = [f"the review found: {finding}" for finding in findings] + list(entry.get("reasons") or [])
    write_json(args.out, report)
    overstated = [ticket for ticket, entry in report["tickets"].items() if entry["claimed"] == SAFE and entry["earned"] != SAFE]
    print(f"written {args.out}: {len(report['tickets'])} ticket(s), {len(again)} measured again, {len(overstated)} claim a label they do not earn")
    return 0


def lower(args):
    """A reviewer's finding against one ticket, recorded in the verification report: a safe label, and a
    won't-do decision the finding disputes, go to unsure with the finding as the first reason. Nothing here
    raises a label."""
    report = read_json(args.infile)
    entry = report.get("tickets", {}).get(args.ticket)
    if entry is None:
        print(f"{args.infile} has no ticket {args.ticket}")
        return 1
    entry["lowered_by_review"] = entry.get("lowered_by_review", []) + [args.reason]
    if entry.get("earned") in (SAFE, WONT_DO):
        entry["earned"] = UNSURE
    entry["reasons"] = [f"the review found: {args.reason}"] + list(entry.get("reasons") or [])
    write_json(args.infile, report)
    print(f"{args.ticket}: {entry['earned']} (the review found: {args.reason})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("environment", "baseline", "prove", "label", "verify", "lower"))
    parser.add_argument("--ticket", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kind", choices=KINDS, default=None)
    parser.add_argument("--select", nargs="*", default=[])
    parser.add_argument("--snapshot", nargs="*", default=[])
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-code-build", action="store_true")
    parser.add_argument("--again", default=None, metavar="REASON", help="baseline: replace an earlier baseline, with why the spec was wrong")
    parser.add_argument("--with-downstream", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--evidence", default=None, metavar="FILE",
                        help="label --ticket: compute the label from this measurement of the ticket instead of the attempt's own (verification's)")
    parser.add_argument("--in", dest="infile", default=None, help="lower: the verification report to record the finding in")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--only", nargs="*", default=[], help="verify: measure only these tickets again")
    args = parser.parse_args()
    for key in [args.ticket, *args.only]:
        if key is not None and not TICKET_KEY.fullmatch(key):
            parser.error(f"a ticket's key names a file, a directory and a branch: letters, digits, `.`, `_` and `-` only "
                         f"(got {key!r}); give the ticket a short key in the inventory and keep its title there")
    if args.command == "environment":
        sys.exit(check_environment(args))
    if args.command == "baseline":
        if not args.ticket or not args.kind or not (args.select or args.snapshot):
            parser.error("baseline needs --ticket, --kind and --select (or --snapshot)")
        sys.exit(baseline(args))
    if args.command == "prove":
        if not args.ticket:
            parser.error("prove needs --ticket")
        sys.exit(prove(args))
    if args.command == "label":
        if not args.ticket and not args.all:
            parser.error("label needs --ticket or --all")
        if args.evidence and (not args.ticket or args.write):
            parser.error("--evidence reads one ticket's label from another measurement: with --ticket, without --write")
        sys.exit(label(args))
    if args.command == "lower":
        if not args.infile or not args.ticket or not args.reason:
            parser.error("lower needs --in, --ticket and --reason")
        sys.exit(lower(args))
    if not args.out:
        parser.error("verify needs --out")
    sys.exit(verify(args))


if __name__ == "__main__":
    main()
