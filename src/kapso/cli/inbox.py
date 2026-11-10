"""`kapso inbox`: what a campaign is waiting on you for, and how to answer it
(docs/research/evolve-hub-design.md v4 §4.6); `kapso inbox note`: what you
tell a campaign that it did not ask for."""

import argparse
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kapso.kapso import Kapso, DEFAULT_CONFIG_PATH
from kapso.core.config import load_config
from kapso.execution.inbox import (
    has_undecodable_bytes,
    launch_receipt_path,
    leave_note,
    list_registered_campaigns,
    looks_like_a_secret,
    notes_path,
    read_notes,
    render_awaiting_resume,
    render_notes,
    render_requests,
)
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


def _is_campaign_root(directory: Path) -> bool:
    """Whether a build runs, ran or will resume in this directory: its
    status file is there from the workspace's setup on, its checkpoint
    from the first save. A bare `.kapso/` does not say so: every session
    clone has one (the repository memory) and so does the home directory
    (the registry), and nothing reads a note filed there."""
    state = directory / ".kapso"
    return (state / "status.json").is_file() or (state / "run_state.json").is_file()


def _campaign_root_here(start: Path) -> Optional[Path]:
    """The nearest ancestor of `start` that is a campaign root: where a
    note left from inside a campaign goes. The same rule as for a named
    directory, so a campaign in its setup minutes (a status file, no
    checkpoint yet) is found, and never passed over for one around it."""
    for candidate in (start, *start.parents):
        if _is_campaign_root(candidate):
            return candidate
    return None


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
    """What `kapso inbox` shows for one campaign: what waits on the
    person, then the notes they left for it."""
    waiting = _render_waiting(campaign)
    notes = read_notes(notes_path(campaign))
    if not notes:
        return waiting
    return "\n".join([
        waiting,
        "",
        "  your notes   every ideation, implementation and judging session that starts reads them",
        "",
        render_notes(notes),
    ])


def _render_waiting(campaign: Path) -> str:
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
    """`kapso inbox [CAMPAIGN]` — what is waiting on you, and the notes
    you left; `kapso inbox reply [CAMPAIGN] [ID] [NOTE]` — answer it, and
    the campaign resumes; `kapso inbox note [CAMPAIGN] TEXT` — tell the
    campaign something it did not ask for."""
    words = list(args.words)
    if words and words[0] == "reply":
        cmd_inbox_reply(words[1:], yes=args.yes, detach=args.detach, log=args.log)
        return
    if words and words[0] == "note":
        cmd_inbox_note(words[1:], yes=args.yes)
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
        Kapso._refuse_a_live_campaign(str(campaign), "pause before replying")
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


def cmd_inbox_note(words, *, yes: bool) -> None:
    """Leave a note for a campaign: something no session asked for. The
    campaign is the first word when that is a directory, else the one the
    current directory is inside; the note is the one word after it, kept
    exactly as given. Nothing is paused, resumed or interrupted: a session
    reads the notes when it starts. A note cannot be taken back, so
    anything that leaves the campaign or the text in doubt is refused, and
    a text shaped like a credential needs --yes, as a reply does."""
    words = list(words)
    # The first word names the campaign when it is a directory. `os.path`
    # asks without raising: the word may be the note itself, and pathlib
    # raises on a long one (the file-name limit) and on one that starts
    # with a tilde (no such user).
    named = bool(words) and os.path.isdir(os.path.expanduser(words[0]))
    campaign = (
        Path(os.path.expanduser(words.pop(0))).resolve() if named
        else _campaign_root_here(Path.cwd().resolve())
    )
    if len(words) > 1:
        problem = "the note is more than one argument" if named else f"{words[0]} is not a directory"
        print(
            f'Error: {problem} — the form is kapso inbox note [campaign] "<text>", the text as one quoted argument',
            file=sys.stderr,
        )
        sys.exit(1)
    if campaign is None:
        print('Error: not inside a campaign — name it: kapso inbox note <campaign> "<text>"', file=sys.stderr)
        sys.exit(1)
    if not _is_campaign_root(campaign):
        print(
            f"Error: {campaign} is not a campaign directory (no status file or checkpoint under .kapso/)",
            file=sys.stderr,
        )
        sys.exit(1)
    text = words[0] if words else ""
    if not text.strip():
        print('Error: a note needs text: kapso inbox note [campaign] "<text>"', file=sys.stderr)
        sys.exit(1)
    if has_undecodable_bytes(text):
        print(
            "Error: the note carries bytes that are not UTF-8 text, which no session's "
            "prompt can hold; type it again as plain text",
            file=sys.stderr,
        )
        sys.exit(1)
    if looks_like_a_secret(text) and not yes:
        print(
            "This note looks like a credential. Notes are stored in plain "
            "text and handed to every later session as text, and a note "
            "cannot be removed; put the value in the .env the run loaded and "
            "leave a note that says where it is. To store it anyway, add "
            "--yes at the end."
        )
        sys.exit(1)
    note = leave_note(notes_path(campaign), text, allow_secret=yes)
    print(f"Note {note.id} left for {campaign}.")
    print("A session that is running right now has already started and will not see it.")
    print("Every ideation, implementation and judging session that starts from here on reads it.")


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
            "resumes the campaign here, continuing the session that asked. "
            "`kapso inbox note` tells a campaign something it did not ask "
            "for: every ideation, implementation and judging session that "
            "starts afterwards reads it."
        ),
        epilog="""
Examples:
  kapso inbox                              # every campaign waiting on you (this one, inside a campaign)
  kapso inbox ./campaign                   # this campaign's requests, and the notes you left for it
  kapso inbox reply 1 "added the key"      # answer request 1 here; the campaign resumes
  kapso inbox reply "added the key"        # the id may be left out when one request is open
  kapso inbox reply ./campaign 1           # answer from elsewhere; an empty note means done
  kapso inbox reply 1 "added the key" --detach   # resume in a session of its own; follow with kapso watch
  kapso inbox reply                        # retry a resume that failed; your reply is kept, a note replaces it
  kapso inbox note "use DEV_DB"            # tell the campaign something it did not ask; sessions that start later read it
  kapso inbox note ./campaign "skip ticket 14"   # the same from elsewhere; the text is one quoted argument
""",
    )
    inbox_parser.add_argument(
        "words", nargs="*",
        help="[CAMPAIGN]  or  reply [CAMPAIGN] [ID] [NOTE]  or  note [CAMPAIGN] TEXT",
    )
    inbox_parser.add_argument(
        "--yes", action="store_true",
        help="Store a reply or a note that looks like a credential anyway",
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
