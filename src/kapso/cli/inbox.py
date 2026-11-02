"""`kapso inbox`: what a campaign is waiting on you for, and how to answer it
(docs/research/evolve-hub-design.md v4 §4.6)."""

import argparse
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kapso.kapso import Kapso, DEFAULT_CONFIG_PATH
from kapso.core.config import load_config
from kapso.execution.inbox import (
    launch_receipt_path,
    list_registered_campaigns,
    looks_like_a_secret,
    render_awaiting_resume,
    render_requests,
)
from kapso.execution.observability import OperationStatusView
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.cli.common import _default_log, _detach, _inside, _print_build_summary, build_handoff

NAME = "inbox"


def _campaign_here(start: Path) -> Optional[Path]:
    """The nearest ancestor of `start` that is a campaign directory."""
    for candidate in (start, *start.parents):
        if (candidate / ".kapso" / "run_state.json").is_file():
            return candidate
    return None


def _inbox_campaign(explicit: Optional[str]) -> Optional[Path]:
    """The campaign a `kapso inbox` invocation means: the named directory,
    else the campaign the cwd is inside, else None."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{explicit} is not a directory")
        return path.resolve()
    return _campaign_here(Path.cwd().resolve())


def _campaign_goal(campaign: Path) -> str:
    """The goal's first line. A campaign launched through build() holds
    the rendered brief, whose first line starts with the field name; one
    launched by a benchmark runner holds its plain goal."""
    store = RunCheckpointStore(str(campaign))
    if not store.exists():
        return ""
    goal = store.load().goal.strip()
    if not goal:
        return ""
    first = goal.splitlines()[0]
    objective = re.match(r"^objective[ \t]+(.*)$", first, re.IGNORECASE)
    return objective.group(1).strip() if objective else first


def _waiting_since(requests) -> str:
    newest = max(request.requested_at for request in requests)
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(newest)).total_seconds()
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _render_campaign_inbox(campaign: Path) -> str:
    requests = Kapso.inbox(str(campaign))
    awaiting = Kapso.awaiting_resume(str(campaign))
    if not requests and not awaiting:
        return f"Nothing waiting on you in {campaign}."
    listed = requests or awaiting
    label = f"  waiting {_waiting_since(listed)}"
    if len(listed) > 1:
        label += f" · {len(listed)} requests"
    goal = _campaign_goal(campaign)
    header = f"{campaign}  {goal}{label}" if goal else f"{campaign}{label}"
    target = f" {requests[0].id}" if len(requests) > 1 else ""
    # The short form resolves the campaign from the current directory, so
    # anywhere else the suggestion has to name it.
    if _campaign_here(Path.cwd().resolve()) != campaign.resolve():
        target = f" {shlex.quote(str(campaign))}{target}"
    ideas = Kapso.inbox_ideas(str(campaign))
    if not requests:
        return "\n".join([
            header,
            "",
            render_awaiting_resume(awaiting, ideas),
            "",
            f"  retry with   kapso inbox reply{target}        # keeps your reply; a note replaces it",
        ])
    return "\n".join([
        header,
        "",
        render_requests(requests, ideas),
        "",
        f'  reply with   kapso inbox reply{target} "…"',
    ])


def cmd_inbox(args) -> None:
    """`kapso inbox [CAMPAIGN]` — what is waiting on you; `kapso inbox
    reply [CAMPAIGN] [ID] [NOTE]` — answer it, and the campaign resumes."""
    words = list(args.words)
    if words and words[0] == "reply":
        cmd_inbox_reply(words[1:], yes=args.yes, detach=args.detach, log=args.log)
        return
    if len(words) > 1:
        print("Error: kapso inbox takes at most a campaign directory", file=sys.stderr)
        sys.exit(1)
    campaign = _inbox_campaign(words[0] if words else None)
    if campaign is not None:
        print(_render_campaign_inbox(campaign))
        return
    registry = load_config(args.config)["inbox"]["registry"]
    waiting = [
        Path(entry["path"])
        for entry in list_registered_campaigns(registry)
        if Kapso.inbox(entry["path"]) or Kapso.awaiting_resume(entry["path"])
    ]
    if not waiting:
        print("Nothing waiting on you.")
        return
    print("\n\n".join(_render_campaign_inbox(campaign) for campaign in waiting))


def cmd_inbox_reply(
    words, *, yes: bool, detach: bool = False, log: Optional[str] = None
) -> None:
    """Answer one request; the campaign resumes here when it can, or in a
    session of its own with --detach (the note is checked here first, so
    a refusal is immediate; the child records it and resumes). The id may
    be left out when one request is open, or when a failed resume left
    one to retry. An empty note means done, or on a retry keeps the reply."""
    words = list(words)
    explicit = None
    if words and not words[0].isdigit() and Path(words[0]).expanduser().is_dir():
        explicit = words.pop(0)
    campaign = _inbox_campaign(explicit)
    if campaign is None:
        print("Error: not inside a campaign — name it: kapso inbox reply <campaign> [id] [note]", file=sys.stderr)
        sys.exit(1)
    explicit_id = int(words.pop(0)) if words and words[0].isdigit() else None
    note = " ".join(words)
    if explicit_id is not None:
        request_id = explicit_id
    else:
        open_requests = Kapso.inbox(str(campaign))
        # A retry of a node that asked for several things needs no choice of
        # id unless a note has to go on one of them.
        awaiting = [] if open_requests else Kapso.awaiting_resume(str(campaign))
        if len(open_requests) == 1:
            request_id = open_requests[0].id
        elif awaiting and (len(awaiting) == 1 or not note):
            request_id = awaiting[0].id
        else:
            print("Error: which request? kapso inbox reply <id> [note]", file=sys.stderr)
            sys.exit(1)
    if looks_like_a_secret(note) and not yes:
        print(
            "This reply looks like a credential. Replies are stored in plain "
            "text and handed to the coder as text; put the value where the "
            "fix says (usually the .env the run loaded) and reply with a note "
            "instead. To store it anyway, add --yes at the end."
        )
        sys.exit(1)
    if detach:
        if log and _inside(Path(log).expanduser().resolve(), campaign):
            print("Error: --log must be outside the campaign directory", file=sys.stderr)
            sys.exit(1)
        located = OperationStatusView.locate(campaign)
        if located is not None and OperationStatusView(located).alive:
            pid = OperationStatusView(located).data.get("pid")
            print(f"{campaign} is running (pid {pid}); wait for it to pause before replying")
            sys.exit(1)
        argv = ["inbox", "reply", str(campaign), str(request_id)]
        if note:
            argv.append(note)
        if yes:
            argv.append("--yes")
        _detach(
            "kapso inbox reply",
            argv,
            log_path=(
                Path(log).expanduser().resolve() if log else _default_log(campaign)
            ),
            receipt_path=launch_receipt_path(campaign),
            handoff=build_handoff(campaign),
        )
        return
    solution = Kapso.reply(str(campaign), request_id, note, allow_secret=yes)
    if solution is not None:
        _print_build_summary(solution)


def add_parser(subparsers):
    inbox_parser = subparsers.add_parser(
        "inbox",
        help="What a build is waiting on you for, and how to answer it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        description=(
            "A session that needs something only a person can provide asks "
            "through the inbox and the campaign pauses. `kapso inbox` shows "
            "what is waiting; `kapso inbox reply` answers one request and "
            "resumes the campaign here, continuing the session that asked."
        ),
        epilog="""
Examples:
  kapso inbox                              # every campaign waiting on you (this one, inside a campaign)
  kapso inbox ./campaign                   # this campaign's requests
  kapso inbox reply 1 "added the key"      # answer request 1 here; the campaign resumes
  kapso inbox reply "added the key"        # the id may be left out when one request is open
  kapso inbox reply ./campaign 1           # answer from elsewhere; an empty note means done
  kapso inbox reply 1 "added the key" --detach   # resume in a session of its own; follow with kapso watch
  kapso inbox reply                        # retry a resume that failed; your reply is kept, a note replaces it
""",
    )
    inbox_parser.add_argument(
        "words", nargs="*",
        help="[CAMPAIGN]  or  reply [CAMPAIGN] [ID] [NOTE]",
    )
    inbox_parser.add_argument(
        "--yes", action="store_true",
        help="Store a reply that looks like a credential anyway",
    )
    inbox_parser.add_argument(
        "--detach", action="store_true",
        help="Resume the campaign in a session of its own and return at once",
    )
    inbox_parser.add_argument(
        "--log", type=str, default=None,
        help="Where a detached resume's output goes (default: <campaign>.log beside it)",
    )
    inbox_parser.add_argument(
        "--config", type=str, default=None,
        help="Config whose inbox.registry lists the campaigns (default: packaged config.yaml)",
    )
    return cmd_inbox
