"""`kapso compare`: two campaigns set side by side. A pure reader over what
each campaign already wrote: it never writes into a campaign and makes no
model calls."""

import argparse
import json
from pathlib import Path

from kapso.execution.campaign_compare import compare_campaigns, render_comparison

NAME = "compare"


def cmd_compare(args) -> None:
    comparison = compare_campaigns(args.campaign_a, args.campaign_b)
    text = json.dumps(comparison, indent=1) + "\n" if args.json else render_comparison(comparison)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Comparison written to {args.output}")
        return
    print(text, end="")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "compare",
        help="Set two campaigns side by side: the delivered scores, the spend, how each search progressed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare two finished or running campaigns: state, iterations, time, spend, the delivered "
            "score and the best score so far after each candidate. A winner is named only when the two "
            "scores mean the same thing: the same scoring direction and the same registered evaluator. "
            "Reads only the campaigns' own checkpoint and status files; writes nothing into them and "
            "makes no model calls."
        ),
        epilog="""
Examples:
  kapso compare ./baseline ./with-harness                # Markdown to stdout
  kapso compare ./baseline ./with-harness -o compare.md  # write to a file
  kapso compare ./baseline ./with-harness --json         # the same facts as data
""",
    )
    parser.add_argument("campaign_a", help="The first campaign directory (A)")
    parser.add_argument("campaign_b", help="The second campaign directory (B)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Write the comparison to this file instead of stdout")
    parser.add_argument("--json", action="store_true", help="Print the comparison as JSON")
    return cmd_compare
