"""`kapso build`: build software that meets a brief, by experiment."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from kapso.core.config import load_config
from kapso.core.env_file import load_env_file
from kapso.kapso import Kapso
from kapso.handles import Brief, Budget
from kapso.execution.inbox import launch_receipt_path, read_launch_record
from kapso.execution.observability import OperationStatusView, TERMINAL_STATES, mark_failed, pid_alive
from kapso.experience import recorded_reference
from kapso.harness import BUILD_MODULES, unmet_message
from kapso.cli.common import (
    AVAILABLE_AGENTS,
    _default_log,
    _detach,
    build_handoff,
    _inside,
    _print_build_summary,
    _without_detach_flags,
)

NAME = "build"


def cmd_build(args) -> None:
    """Handle the build command."""
    if args.stop:
        beside = [flag for flag, value in (("--resume", args.resume), ("--brief", args.brief), ("--goal", args.goal), ("--goal-file", args.goal_file)) if value]
        if beside:
            print(f"Error: --stop stops a running campaign and takes no other flag; drop {', '.join(beside)}", file=sys.stderr)
            sys.exit(1)
        _stop(args)
        return
    if args.brief:
        brief_path = Path(args.brief).expanduser()
        if not brief_path.is_file():
            print(f"Error: brief not found: {brief_path}", file=sys.stderr)
            sys.exit(1)
        brief = Brief.read(brief_path)
    elif args.goal_file:
        brief = Path(args.goal_file).expanduser().read_text(encoding="utf-8")
    elif args.goal:
        brief = args.goal
    elif args.resume:
        brief = None   # a resume takes the brief from the checkpoint
    else:
        print(
            "Error: --goal, --goal-file or --brief is required "
            "(a resume takes the brief from the checkpoint)",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.detach:
        if not args.output:
            print(
                "Error: --detach needs --output: the campaign directory is "
                "where its status file lives and its log goes beside",
                file=sys.stderr,
            )
            sys.exit(1)
        campaign = Path(args.output).expanduser().resolve()
        if args.log and _inside(Path(args.log).expanduser().resolve(), campaign):
            print(
                "Error: --log must be outside --output: the campaign directory "
                "must be empty for seeding, and the log would be in it",
                file=sys.stderr,
            )
            sys.exit(1)
        # The harness is loaded and its prerequisites checked before the
        # fork, so the handoff below never claims a launch that would die
        # on a missing command or variable in the child's log; the
        # experience the same, so a reference that names nothing, or a
        # source that is gone, refuses here with its line. A bare resume
        # takes both from the launch record, as the build will.
        kapso = Kapso(config=args.config)
        record = read_launch_record(campaign) if args.resume else None
        ref = _harness_reference(args, brief)
        if ref is None and record is not None and record.get("harness"):
            ref = record["harness"]["source"]
        if record is not None:
            # The check counts the campaign's .env, as the child's will.
            load_env_file(record.get("dotenv_path", ""))
        if ref is not None:
            _load_harness_or_exit(kapso, args, brief, ref=ref)
        references = _experience_references(args, brief)
        if not references and record is not None:
            references = [recorded_reference(entry) for entry in record.get("experience", [])]
        for reference in references:
            kapso.load_experience(reference)
        if args.resume:
            Kapso._refuse_a_live_campaign(str(campaign), "stop before resuming")
        _detach(
            "kapso build",
            _without_detach_flags(sys.argv[1:]),
            log_path=(
                Path(args.log).expanduser().resolve() if args.log
                else _default_log(campaign)
            ),
            receipt_path=launch_receipt_path(campaign),
            handoff=build_handoff(campaign),
        )
        return

    kapso = Kapso(config=args.config)
    _load_harness_or_exit(kapso, args, brief)
    for reference in _experience_references(args, brief):
        kapso.load_experience(reference)
    solution = kapso.build(
        brief,
        evaluation=args.evaluation,
        data=args.data,
        repo=args.repo,
        output=args.output,
        budget=Budget(
            minutes=args.budget_minutes,
            experiments=args.budget_experiments,
            usd=args.budget_usd,
            reserve_minutes=args.budget_reserve_minutes,
        ),
        resume=args.resume,
        coding_agent=args.coding_agent,
    )
    _print_build_summary(solution)


def _stop(args) -> None:
    """End the campaign running in --output: its process group (the
    status file's pid, a detached launch being its own session) gets
    SIGTERM, then SIGKILL once the grace the config gives has passed; the
    status is marked failed with the reason and the receipt goes, so
    `kapso watch` says what happened and `--resume` continues from the
    checkpoint. A campaign whose status is terminal, or whose writer is
    gone already, is refused by state: there is nothing to end."""
    if not args.output:
        print("Error: --stop needs --output: the campaign directory whose run ends", file=sys.stderr)
        sys.exit(1)
    campaign = Path(args.output).expanduser().resolve()
    status_path = OperationStatusView.locate(campaign)
    if status_path is None:
        print(f"Error: no status under {campaign}: nothing is running there", file=sys.stderr)
        sys.exit(1)
    view = OperationStatusView(status_path)
    if view.state in TERMINAL_STATES:
        print(f"Error: the campaign at {campaign} is {view.state}; nothing to stop", file=sys.stderr)
        sys.exit(1)
    pid = view.data.get("pid")
    # The writer's own stamp, not the bare pid: a pid handed to a later
    # process would take that stranger and everything under it.
    if view.process_alive is None:
        print(f"Error: this machine cannot tell whether pid {pid} is the campaign's writer; nothing was signalled", file=sys.stderr)
        sys.exit(1)
    if not view.process_alive:
        print(
            f"Error: the campaign at {campaign} has no live writer (pid {pid}): it is dead, not running; "
            f"`kapso build --output {campaign} --resume --detach` continues it",
            file=sys.stderr,
        )
        sys.exit(1)
    budget = load_config(args.config)["budget"]
    grace, poll = float(budget["stop_grace_seconds"]), float(budget["stop_poll_seconds"])
    # The campaign's processes: the writer and everything under it (the
    # coding-agent CLIs run in sessions of their own, outside the writer's
    # group). A group leader takes its group with it; a process sharing
    # its parent's group (a foreground launch from a script) is signalled
    # alone, so the stop never reaches what launched the campaign.
    targets = [pid, *_descendants(pid)]
    _signal(targets, signal.SIGTERM)
    _wait_gone(targets, grace, poll)
    if any(pid_alive(target) for target in targets):
        _signal(targets, signal.SIGKILL)
        # Bounded again: a process in uninterruptible sleep can outlast it.
        _wait_gone(targets, grace, poll)
    mark_failed(status_path, "stopped by the person", stopped_reason="stopped")
    receipt = launch_receipt_path(campaign)
    if receipt.is_file():
        receipt.unlink()
    print(f"stopped the campaign at {campaign} (pid {pid})")
    print(f"  status:  {status_path}")
    print(f"  resume:  kapso build --output {campaign} --resume --detach")


def _descendants(pid: int) -> List[int]:
    """Every process under `pid`, read from the process table."""
    table = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True).stdout.split()
    children: Dict[int, List[int]] = {}
    for child, parent in zip(table[0::2], table[1::2]):
        children.setdefault(int(parent), []).append(int(child))
    found: List[int] = []
    queue = [pid]
    while queue:
        for child in children.get(queue.pop(), []):
            found.append(child)
            queue.append(child)
    return found


def _signal(pids: List[int], signum: int) -> None:
    """Each live process: its whole group when it leads one, itself
    otherwise."""
    for target in pids:
        if pid_alive(target) is not True:
            continue
        group = os.getpgid(target)
        if group == target:
            os.killpg(group, signum)
        else:
            os.kill(target, signum)


def _wait_gone(pids: List[int], seconds: float, poll: float) -> None:
    deadline = time.monotonic() + seconds
    while any(pid_alive(target) for target in pids) and time.monotonic() < deadline:
        time.sleep(poll)


def _harness_reference(args, brief):
    """`--harness` wins over the brief's Harness line; a sentence or a
    plain-text goal names none."""
    if args.harness:
        return args.harness
    return brief.harness if isinstance(brief, Brief) else None


def _experience_references(args, brief) -> List[str]:
    """`--experience` (comma-separated) wins over the brief's Experience
    line; a sentence or a plain-text goal names none. A bare resume names
    none here: the build takes the record's list."""
    if args.experience:
        return [item.strip() for item in args.experience.split(",") if item.strip()]
    return list(brief.experience) if isinstance(brief, Brief) else []


def _load_harness_or_exit(kapso, args, brief, ref=None):
    """Load the harness a build names (or the one given) and refuse, before
    anything starts, while a requirement of the build's own modules is
    unmet."""
    ref = ref or _harness_reference(args, brief)
    if ref is None:
        return None
    harness = kapso.load_harness(ref)
    unmet = harness.missing_for(BUILD_MODULES)
    if unmet:
        print(unmet_message(harness, unmet, "build"))
        sys.exit(1)
    return harness


def add_parser(subparsers):
    build_parser = subparsers.add_parser(
        "build",
        help="Build software that meets a brief, by experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # No abbreviations: a `--det` the child kept would detach again, forever.
        allow_abbrev=False,
        epilog="""
Examples:
  kapso build --brief kapso-brief.md --evaluation ./evaluation --data ./data --repo . --output ../churn-build --detach
  kapso build --goal "Build a web scraper for news articles" --output ./scraper
  kapso build --goal-file problem.txt --budget-experiments 20 --output ./campaign
  kapso build --output ./campaign --resume       # brief and flags from the checkpoint and the launch record
  kapso build --output ./campaign --stop         # end a running campaign; --resume continues it later
  kapso watch ./campaign --follow                # follow a detached build
"""
    )

    goal_group = build_parser.add_mutually_exclusive_group()
    goal_group.add_argument(
        "-g", "--goal", type=str,
        help="The objective in a sentence (a resume takes the brief from the checkpoint)",
    )
    goal_group.add_argument(
        "-f", "--goal-file", type=str,
        help="A file holding the objective as plain text",
    )
    goal_group.add_argument(
        "--brief", type=str,
        help="A brief file (kapso-brief.md): objective, how it is measured, "
             "baseline, target, rules, budget",
    )

    build_parser.add_argument(
        "--evaluation", type=str,
        help="Directory holding the evaluator (copied to kapso_evaluation/, integrity-protected)",
    )
    build_parser.add_argument(
        "--data", type=str,
        help="Directory holding the data (copied to kapso_datasets/)",
    )
    build_parser.add_argument(
        "--repo", type=str,
        help="Starting repository: a local path (copied) or a GitHub URL (cloned)",
    )
    build_parser.add_argument(
        "-o", "--output", type=str,
        help="The campaign directory (empty for a fresh build): the branches, the record, the solution",
    )

    build_parser.add_argument(
        "--budget-minutes", type=float, default=None,
        help="Wall-clock budget for the campaign (durable across resumes)",
    )
    build_parser.add_argument(
        "--budget-experiments", type=int, default=None,
        help=f"Max experiments this run (default: {load_config()['budget']['experiments']}; a resume keeps the launch's)",
    )
    build_parser.add_argument(
        "--budget-usd", type=float, default=None,
        help="Best-effort model-spend budget in USD",
    )
    build_parser.add_argument(
        "--budget-reserve-minutes", type=float, default=None,
        help="Wall clock escrowed for the final checkout and evaluation",
    )
    build_parser.add_argument(
        "--harness", type=str, default=None,
        help="The harness to build with: a bundled name, a directory, or a git "
             "reference (wins over the brief's Harness line)",
    )
    build_parser.add_argument(
        "--experience", type=str, default=None,
        help="Comma-separated sources to serve: mine, mine:all, leeroopedia, a bank "
             "path or git reference, an index name (wins over the brief's Experience line)",
    )
    build_parser.add_argument(
        "--resume", action="store_true",
        help="Continue the campaign in --output from its checkpoint",
    )
    build_parser.add_argument(
        "--detach", action="store_true",
        help="Start the campaign in a session of its own and return at once "
             "with its pid, log, status path and the watch command (needs --output)",
    )
    build_parser.add_argument(
        "--log", type=str, default=None,
        help="Where a detached run's output goes (default: <output>.log beside the campaign)",
    )
    build_parser.add_argument(
        "--stop", action="store_true",
        help="End the campaign running in --output (its process group), mark its status, and keep "
             "its checkpoint for --resume",
    )

    build_parser.add_argument(
        "-a", "--coding-agent", type=str, choices=AVAILABLE_AGENTS,
        help="The host every session runs on, with the model the config's hosts table "
             "gives it (default: search_strategy.params.host)",
    )
    build_parser.add_argument(
        "--config", type=str, default=None,
        help="Your override file, layered over the packaged config (default: the packaged defaults)",
    )
    return cmd_build
