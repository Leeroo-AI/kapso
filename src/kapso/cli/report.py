"""`kapso report`: a campaign as a shareable Markdown summary. A pure
reader over what the campaign already wrote: it never writes into the
campaign and makes no model calls."""

import argparse
from pathlib import Path

from kapso.core.config import load_config
from kapso.execution.campaign_report import render_campaign_report

NAME = "report"


def cmd_report(args) -> None:
    top = args.top if args.top is not None else load_config(args.config)["report"]["leaderboard_size"]
    report = render_campaign_report(args.campaign, top=top)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
        return
    print(report, end="")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "report",
        help="Export a campaign as a shareable Markdown summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Render a finished or running campaign as Markdown: summary, deliverable, leaderboard, lineage, "
            "what each leader changed, human input and outcomes. Reads only the campaign's own files and git "
            "history; writes nothing into it and makes no model calls."
        ),
        epilog="""
Examples:
  kapso report ./campaign                  # Markdown to stdout
  kapso report ./campaign -o report.md     # write to a file
  kapso report ./campaign --top 10         # leaderboard size
""",
    )
    parser.add_argument("campaign", help="Campaign directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Write the report to this file instead of stdout")
    parser.add_argument("--top", type=int, default=None, help="Leaderboard size (default: report.leaderboard_size in the config)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config whose report.leaderboard_size sets the default (default: packaged config.yaml)")
    return cmd_report
