"""`kapso deploy`: deploy the solution a build wrote as running software,
and stop, start, read and list what was deployed (docs/plans/redesign/
wp5-deploy.md, "The verb")."""

import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from kapso.cli.common import AVAILABLE_AGENTS
from kapso.deployment import DeployError, Software
from kapso.deployment.record import RUNNING, DeploymentRecord, open_record, records
from kapso.deployment.targets import FUNCTION
from kapso.handles import Solution
from kapso.kapso import Kapso

NAME = "deploy"
LIFECYCLE = ("status", "stop", "start", "logs")
# The flags of a deploy, refused beside a lifecycle flag.
DEPLOY_ONLY = ("target", "env", "option", "coding_agent", "harness")


def pairs(values: Optional[List[str]], flag: str) -> Dict[str, str]:
    """`KEY=VALUE` arguments; a value without `=` is refused by name."""
    parsed: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise DeployError(f"{flag} takes KEY=VALUE, got {value!r}")
        key, text = value.split("=", 1)
        parsed[key] = text
    return parsed


def workspace_size(workspace: Path) -> str:
    total = sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file())
    return f"{total / 1_000_000:.1f} MB"


def health_word(software: Software) -> str:
    """` (healthy)` or ` (not answering)`; a function deployment has no
    process of its own, so the word says what was proved: its module
    imports in a worker of this call's."""
    if software.run_kind == FUNCTION:
        return " (healthy: its module imports in a worker)" if software.is_healthy() else " (not answering: its module does not import)"
    return " (healthy)" if software.is_healthy() else " (not answering)"


def summary(software: Software, campaign: str, config: Optional[str] = None) -> List[str]:
    """What a person reads after a deployment: what runs where, the call
    it proved and what the engine's own call answered, and the next
    actions pinned to this deployment's number, each copy-pastable."""
    record = json.loads(software.record.read_text(encoding="utf-8"))
    run = record["result"]["run"]
    kind = run["kind"]
    how = {
        "function": f"function — {run.get('module')}.{run.get('callable')} in a worker process",
        "http": f"http — POST {run.get('url')}{run.get('path')}",
        "script": "script — the target's scripts/run.py",
    }[kind]
    proof = record["result"]["proof"]
    observed = record["observed"]
    proof_line = f"  Proof:     {json.dumps(proof['inputs'])} -> {json.dumps(observed)}"
    if observed != proof["output"]:
        proof_line += f"   (the engine's own call; the session's answered {json.dumps(proof['output'])})"
    interpreter = f"  Interpreter: {record['interpreter']}"
    if kind == FUNCTION:
        interpreter += " (runs the function; a kapso upgrade replaces it, so reinstall the solution's requirements there after one)"
    verb = f"kapso deploy {shlex.quote(campaign)}"
    suffix = f" --number {software.number}" + (f" --config {shlex.quote(config)}" if config else "")
    lines = [
        "",
        "=" * 60,
        f"DEPLOYED: deployment {software.number} of {campaign}",
        "=" * 60,
        f"  Target:    {software.target}" + (f" (chosen: {record['reasoning']})" if record["target_requested"] == "auto" else ""),
        f"  State:     {software.state}" + health_word(software),
        f"  Run:       {how}",
        proof_line,
        f"  Endpoint:  {software.endpoint or 'none'}",
        f"  Summary:   {record['result']['summary']}",
        interpreter,
        f"  Workspace: {software.workspace} (the solution with its data, {workspace_size(software.workspace)})",
        f"  Record:    {software.record}",
        "Next:",
        f"  from kapso import Software; Software.open({campaign!r}, {software.number}{', ' + repr(config) if config else ''}).run({proof['inputs']!r})   # from Python",
        f"  {verb} --logs{suffix}",
        f"  {verb} --stop{suffix}",
        f"  {verb} --start{suffix}",
    ]
    return lines


def status_line(record: DeploymentRecord, config: Optional[str]) -> str:
    """One deployment's row: number, state (with health for a running
    one, the error for a failed one), target (the request until the
    session chose), endpoint, when the state was written."""
    note = ""
    if record.state == RUNNING:
        if record.workspace.is_dir() and (record.data["target"] is None or record.target_dir.is_dir()):
            software = Software.open(record.campaign, record.number, config=config)
            note = health_word(software)
            software.close()
        else:
            note = " (its files are gone)"
    elif record.data["error"]:
        note = f" ({record.data['error']})"
    target = record.data["target"] or f"{record.data['target_requested']} (requested)"
    return f"  {record.number}  {record.state}{note}  {target}  {record.data['endpoint'] or 'no endpoint'}  {record.data['state_at']}"


def lifecycle(args) -> None:
    campaign = args.campaign
    if args.status:
        found = records(Path(campaign)) if args.number is None else [open_record(Path(campaign), args.number)]
        # A campaign not yet deployed is a plain answer to a status question,
        # not an error: the release gate's cold engineer asked before the
        # first deploy and read the exit code as a wrong call.
        if not found:
            print(f"{campaign} has no deployment yet")
            return
        for record in found:
            print(status_line(record, args.config))
        return
    software = Software.open(campaign, args.number, config=args.config)
    if args.stop:
        noted = software.stop()
        print(f"stopped deployment {software.number} of {campaign}")
        for line in noted[:-1]:
            print(f"  {line}")
    elif args.start:
        noted = software.start()
        print(f"started deployment {software.number} of {campaign}{health_word(software)}")
        for line in noted[:-1]:
            print(f"  {line}")
        software.close()
    else:
        print(software.logs())


def cmd_deploy(args) -> None:
    chosen = [flag for flag in LIFECYCLE if getattr(args, flag)]
    if chosen:
        for flag in DEPLOY_ONLY:
            if getattr(args, flag):
                raise DeployError(f"--{flag.replace('_', '-')} goes with a deploy, not with --{chosen[0]}")
        lifecycle(args)
        return
    if args.number is not None:
        raise DeployError("--number goes with --status, --stop, --start or --logs; a deploy takes the next number")
    # The solution a build wrote in that directory (its .kapso/solution.json)
    solution = Solution.open(args.campaign)
    env = pairs(args.env, "--env")
    options = pairs(args.option, "--option")
    kapso = Kapso(config=args.config)
    if args.harness:
        kapso.load_harness(args.harness)
    software = kapso.deploy(
        solution, target=args.target, env=env or None, options=options or None, coding_agent=args.coding_agent,
    )
    print("\n".join(summary(software, args.campaign, args.config)))
    software.close()


def add_parser(subparsers):
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy the solution a build wrote as running software; stop, start, read or list what was deployed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  kapso deploy ../churn-build
  kapso deploy ../churn-build --target docker --env API_KEY=xxx
  kapso deploy ../churn-build --target paper-trading --option schedule="*/5 * * * *"
  kapso deploy ../churn-build --status
  kapso deploy ../churn-build --stop --number 2
"""
    )
    deploy_parser.add_argument(
        "campaign", type=str,
        help="The campaign directory a build wrote (holds .kapso/solution.json)",
    )
    deploy_parser.add_argument(
        "--target", type=str, default=None,
        help="Where to deploy: auto, a bundled target (local, docker, modal, bentoml, langgraph) or a harness's "
             "(default: config deployment.target)",
    )
    deploy_parser.add_argument("--env", type=str, action="append", help="A variable the software needs at runtime, KEY=VALUE (repeatable)")
    deploy_parser.add_argument("--option", type=str, action="append", help="An option for the target's workflow, KEY=VALUE (repeatable)")
    deploy_parser.add_argument(
        "--coding-agent", type=str, choices=AVAILABLE_AGENTS, default=None,
        help="The host the deployment session runs on (default: config deployment.cli, else the build's host)",
    )
    deploy_parser.add_argument(
        "--config", type=str, default=None,
        help="Config path (default: packaged config.yaml); pass the one the build used",
    )
    deploy_parser.add_argument(
        "--harness", type=str, default=None,
        help="The harness whose deployment module the session carries: the build's "
             "(from the campaign record, the default) or another version of it",
    )
    action = deploy_parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="List the campaign's deployments with their state instead of deploying (one with --number)")
    action.add_argument("--stop", action="store_true", help="Stop a deployment (the highest number, or --number)")
    action.add_argument("--start", action="store_true", help="Start a stopped deployment again")
    action.add_argument("--logs", action="store_true", help="Print a deployment's logs")
    deploy_parser.add_argument("--number", type=int, default=None, help="Which deployment --status, --stop, --start and --logs act on (default: every one for --status, the highest otherwise)")
    return cmd_deploy
