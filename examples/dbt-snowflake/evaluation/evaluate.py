#!/usr/bin/env python3
"""The Brightmere backlog's evaluation: the team's ticket checks, run as Kapso's evaluator.

    python kapso_evaluation/evaluate.py                   # the backlog; last line `tickets_passing: <share>`
    python kapso_evaluation/evaluate.py --fast            # the four-ticket smoke subset: `fast_tickets_passing`,
                                                          # a spot check, never compared with a full score
    python kapso_evaluation/evaluate.py --ticket <key>    # one ticket (repeatable), also a spot check

This runs `run_checks.py` beside this file: a copy of the data platform's client (`checks/run_checks.py` in
the repository) with its own `tickets.json` (the seventeen tickets and the smoke subset) and `service.json`
(the check service's address). The client sends `warehouse/` and `marketing/` as they stand in the workspace
to the service, which builds each ticket's models in a check database of its own and runs the ticket's
acceptance checks there. Those checks live in the service and nowhere in the workspace. Nothing here is
seeded: the checks are deterministic on a fixed clone of the warehouse.

The client's output is relayed line by line, and `result.json` is written beside this file with
`evaluation_script_path`, `evaluation_output`, `score` (the share, 0 to 1), `metric` and `comparable` (false
for a --fast or --ticket run). The exit code is the client's: 0 when the checks ran, 2 when the service could
not be reached or refused the submission, 3 when no ticket's build succeeded; 1 when no score line was
printed. A non-zero exit is a failed evaluation, not a score.

The workspace is the directory holding both `warehouse/` and `marketing/`: the current directory or one of
its parents, a parent of this file, or one directory below the current one.
"""
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECTS = ("warehouse", "marketing")
SCORE_LINE = re.compile(r"^(?P<metric>(?:fast_)?tickets_passing): (?P<value>[0-9]*\.?[0-9]+)\s*$")


def holds_projects(directory):
    return all((directory / project / "dbt_project.yml").is_file() for project in PROJECTS)


def workspace():
    cwd = pathlib.Path.cwd()
    candidates = [cwd, *cwd.parents, HERE, *HERE.parents]
    try:
        candidates += sorted(path for path in cwd.iterdir() if path.is_dir())
    except OSError:
        pass
    for directory in candidates:
        if holds_projects(directory):
            return directory
    sys.exit("evaluate: no directory holding both warehouse/ and marketing/ was found from the current "
             "directory or from this file")


def main():
    root = workspace()
    arguments = sys.argv[1:]
    command = [sys.executable, str(HERE / "run_checks.py"), *arguments]
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
    exit_code = process.wait()

    metric, score = None, None
    for line in reversed(lines):
        found = SCORE_LINE.match(line)
        if found:
            metric, score = found["metric"], float(found["value"])
            break

    record = {
        "evaluation_script_path": str(pathlib.Path(__file__).resolve()),
        "evaluation_output": "\n".join(lines),
        "score": score,
        "metric": metric,
        "comparable": metric == "tickets_passing",
        "arguments": arguments,
        "workspace": str(root),
        "exit_code": exit_code,
    }
    (HERE / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if exit_code != 0:
        print(f"evaluate: the ticket checks did not produce a score (exit {exit_code}); see the lines above",
              flush=True)
        sys.exit(exit_code)
    if score is None:
        print("evaluate: the ticket checks exited 0 but printed no tickets_passing line", flush=True)
        sys.exit(1)
    if metric != "tickets_passing":
        print(f"evaluate: a spot check ({metric}), not comparable with a full tickets_passing score", flush=True)
        print(f"{metric}: {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
