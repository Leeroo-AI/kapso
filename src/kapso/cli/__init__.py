#!/usr/bin/env python3
# Kapso CLI
#
# One module per verb under kapso/cli/. Each exposes NAME and
# add_parser(subparsers), which registers the verb's parser and returns
# its handler; main() wires them in the order below. Shared pieces (the
# agent list, detached runs, the campaign summary) live in common.py.
#
# Usage:
#     kapso build --brief kapso-brief.md --output ../campaign --detach
#     kapso experience add --repo https://github.com/org/x --name org-x
#     kapso learn import --subset docs/plans/learning/d1-subset.yaml
#     kapso deploy ../campaign

import argparse
import sys

from dotenv import find_dotenv, load_dotenv
# Resolve .env from the caller's CWD upward. The no-arg default anchors at
# THIS file's directory — site-packages in a pip install — so the .env the
# README tells users to create next to their project was never read.
load_dotenv(find_dotenv(usecwd=True))

from kapso import __version__
from kapso.cli import build, deploy, doctor, experience, harness, inbox, learn, report, trace, watch
from kapso.cli.common import list_agents
from kapso.cli.errors import install_error_hook
from kapso.execution.coding_agents.factory import CodingAgentFactory

# The verbs, in the order `kapso --help` lists them.
VERB_MODULES = (build, harness, experience, learn, deploy, watch, trace, report, inbox, doctor)


def build_parser() -> argparse.ArgumentParser:
    """The CLI's parser, every verb registered: what `main` parses with,
    and what a test parses a documented line against."""
    parser = argparse.ArgumentParser(
        description="Kapso: turn coding-agent output into verified outcomes, not review debt. Plug in the harness for your domain, set the goal, every step traced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  build      Build software that meets a brief, by experiment
  harness    Author, check and fetch harnesses: the domain layer a build loads
  experience What a build is served that was measured before: your bank, the hosted gate, ingested corpora
  learn      Trajectory learning (store, mining, updating, grading)
  deploy     Deploy solutions as running software
  watch      Watch a running build / learn / experience operation
  trace      Read a campaign's record: the attempts, the lessons, what was voided, what was delivered
  report     Export a campaign as a shareable Markdown summary
  inbox      What a build is waiting on you for, and how to answer it
  doctor     Check this machine is ready to run Kapso

Examples:
  # Build a solution
  kapso build --brief kapso-brief.md --output ../campaign --detach

  # Ingest a repository into an index a build can load
  kapso experience add --repo https://github.com/org/x --name org-x

  # Learn from a finished campaign
  kapso learn ../campaign

  # Deploy a solution
  kapso deploy ../campaign --target local
"""
    )

    # Global options
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available coding agents"
    )
    parser.add_argument("--version", action="version", version=f"kapso {__version__}")

    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser.set_defaults(handlers={module.NAME: module.add_parser(subparsers) for module in VERB_MODULES})
    _refuse_abbreviations(parser)
    return parser


def _refuse_abbreviations(parser: argparse.ArgumentParser) -> None:
    """Every parser, at every depth, refuses a shortened flag: `--det` is
    not `--detach` (a detached child keeps its parent's command line, and
    a shortened flag the parent did not strip would detach it again)."""
    parser.allow_abbrev = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                _refuse_abbreviations(child)


def main():
    # The engine's own errors print their message alone (cli/errors.py).
    install_error_hook()
    CodingAgentFactory.register_quiet_exit()
    parser = build_parser()

    # =========================================================================
    # Parse and execute
    # =========================================================================
    args = parser.parse_args(learn.normalize_argv(sys.argv[1:]))
    handlers = args.handlers

    # Handle global options
    if args.list_agents:
        list_agents()
        return

    # Route to command handler
    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()
        print("\nError: Please specify a command (build, harness, experience, learn, deploy, watch, trace, inbox, doctor)", file=sys.stderr)
        sys.exit(1)
