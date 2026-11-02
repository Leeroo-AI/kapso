"""The campaign inbox: what a session asked a person for, and what the
person replied.

Design: docs/research/evolve-hub-design.md (v4). One append-only JSONL
file per campaign, ``.kapso/inbox.jsonl``, holding events about
requests: ``requested`` (written by the session through the
``request_from_user`` tool), ``replied`` (written by ``kapso inbox
reply``) and ``continued`` (written by the orchestrator when the session
was resumed). A request's state is the fold of its events; nothing is
ever rewritten, and no request carries a value. A later ``replied`` before
``continued`` replaces the answer, since no session has read it yet.

Also the two small records that make a reply self-sufficient: the launch
record (``.kapso/launch.json`` — the build arguments the checkpoint does
not hold) and the campaign registry (one line per launch, so ``kapso
inbox`` with no campaign can list every campaign waiting on a person).

Every read of the inbox file raises on a malformed line (Rule 2); a
missing file is the documented "no requests" default. Only a line ending
in a newline is an event: see ``_committed``.
"""

import fcntl
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from contextlib import contextmanager

class InboxOpenError(RuntimeError):
    """A suspended node still waits for a reply: the campaign must pause,
    never run an iteration."""


INBOX_RELATIVE_PATH = Path(".kapso") / "inbox.jsonl"
LAUNCH_RELATIVE_PATH = Path(".kapso") / "launch.json"
RESUME_LOCK_RELATIVE_PATH = Path(".kapso") / "resume.lock"
LAUNCH_SCHEMA_VERSION = 2

# The fields a session supplies for one request (Appendix A.2). All are
# required, all are non-empty strings.
REQUEST_FIELDS = ("key", "hit", "tried", "fix", "next_steps")
EVENT_TYPES = ("requested", "replied", "continued")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inbox_path(workspace_dir: str | Path) -> Path:
    """Absolute: the path crosses into the gate server, whose cwd is the
    session folder (live L1 run 2: a relative campaign dir put the
    request under the session's tree and the campaign never paused)."""
    return Path(workspace_dir).resolve() / INBOX_RELATIVE_PATH


@dataclass
class Request:
    """One request, as folded from the inbox events."""

    id: int
    node: int
    session: str
    key: str
    hit: str
    tried: str
    fix: str
    next_steps: str
    requested_at: str
    previous_reply: Optional[str] = None
    reply: Optional[str] = None
    replied_at: Optional[str] = None
    continued: bool = False

    @property
    def open(self) -> bool:
        return self.reply is None

    @property
    def state(self) -> str:
        if self.continued:
            return "continued"
        if self.reply is not None:
            return "answered"
        return "open"


# =============================================================================
# THE FILE
# =============================================================================

def _committed(text: str) -> str:
    """The complete lines of the file. A writer acknowledges an event only
    once its whole line is on disk, so a tail with no newline is a write
    that never finished (a full disk, a killed process): nobody was told
    it happened, and the request it was for is unchanged. That is not
    corruption, and it must not make the campaign unreadable; a complete
    line that does not parse still raises."""
    return text[: text.rfind("\n") + 1]


@contextmanager
def _locked(path: Path) -> Iterator[Any]:
    """The inbox file opened for append-plus-read under an exclusive
    lock, so a read-fold-append sequence sees and writes a consistent
    file even with the gate, the CLI and the orchestrator all writing.
    Holding the lock proves no write is in flight, so an unfinished tail
    is removed here, before an append could join it to the next event.
    What the caller wrote is on disk when this returns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read()
        committed = _committed(text)
        if committed != text:
            handle.truncate(len(committed.encode("utf-8")))
        handle.seek(0)
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def resume_lock(workspace_dir: str | Path) -> Iterator[None]:
    """Held for the whole of a resume started from a reply, so two replies
    cannot both resume the campaign. It blocks rather than refuses: the
    second reply decides what to do after the first has finished, knowing
    how it ended. The kernel releases it when the holder dies, so a killed
    resume leaves nothing to clean up, and the sessions a resume starts do
    not inherit it."""
    with _locked(Path(workspace_dir).resolve() / RESUME_LOCK_RELATIVE_PATH):
        yield


def _parse_events(text: str, source: str) -> List[Dict[str, Any]]:
    events = []
    for number, line in enumerate(text.splitlines(), start=1):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError(f"{source}:{number}: inbox event must be an object")
        if event.get("event") not in EVENT_TYPES:
            raise ValueError(
                f"{source}:{number}: unknown inbox event {event.get('event')!r}"
            )
        if isinstance(event.get("id"), bool) or not isinstance(event.get("id"), int):
            raise ValueError(f"{source}:{number}: inbox event id must be an integer")
        events.append(event)
    return events


def read_events(path: str | Path) -> List[Dict[str, Any]]:
    """Every event in the file, in order. Missing file → no events."""
    path = Path(path)
    if not path.is_file():
        return []
    return _parse_events(_committed(path.read_text(encoding="utf-8")), str(path))


def fold(events: Sequence[Dict[str, Any]]) -> Dict[int, Request]:
    """Requests by id, with their current state."""
    requests: Dict[int, Request] = {}
    for event in events:
        kind = event["event"]
        request_id = event["id"]
        if kind == "requested":
            if request_id in requests:
                raise ValueError(f"inbox request #{request_id} requested twice")
            requests[request_id] = Request(
                id=request_id,
                node=int(event["node"]),
                session=str(event["session"]),
                key=str(event["key"]),
                hit=str(event["hit"]),
                tried=str(event["tried"]),
                fix=str(event["fix"]),
                next_steps=str(event["next_steps"]),
                requested_at=str(event["ts"]),
                previous_reply=event.get("previous_reply"),
            )
            continue
        if request_id not in requests:
            raise ValueError(
                f"inbox event {kind!r} names unknown request #{request_id}"
            )
        request = requests[request_id]
        if kind == "replied":
            if request.continued:
                raise ValueError(
                    f"inbox request #{request_id} replied after it was continued"
                )
            request.reply = str(event["note"])
            request.replied_at = str(event["ts"])
        else:
            if request.reply is None:
                raise ValueError(
                    f"inbox request #{request_id} continued before a reply"
                )
            request.continued = True
    return requests


def load_requests(path: str | Path) -> Dict[int, Request]:
    return fold(read_events(path))


def open_requests(requests: Dict[int, Request]) -> List[Request]:
    return [request for request in requests.values() if request.open]


def requests_for_ids(
    requests: Dict[int, Request], ids: Sequence[int]
) -> List[Request]:
    missing = [request_id for request_id in ids if request_id not in requests]
    if missing:
        raise ValueError(f"inbox has no request(s) {missing}")
    return [requests[request_id] for request_id in ids]


def all_answered(requests: Dict[int, Request], ids: Sequence[int]) -> bool:
    """True when the ids exist, there is at least one, and none is open."""
    if not ids:
        return False
    return all(not request.open for request in requests_for_ids(requests, ids))


def awaiting_resume(
    requests: Dict[int, Request], node_history: Sequence[Dict[str, Any]]
) -> List[Request]:
    """The requests whose node the checkpoint still holds suspended with
    every reply in: the person has done their part and the campaign owes a
    resume, because the last one failed or was killed. The checkpoint
    decides, not ``continued``, which is written as the session starts and
    says nothing about whether it finished."""
    awaiting: List[Request] = []
    for node in node_history:
        ids = node.get("request_ids") or []
        if node.get("suspended") and all_answered(requests, ids):
            awaiting.extend(requests_for_ids(requests, ids))
    return awaiting


def _validate_entries(entries: Any) -> List[Dict[str, str]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("requests must be a non-empty list")
    validated = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"request {position} must be an object")
        missing = [
            name
            for name in REQUEST_FIELDS
            if not isinstance(entry.get(name), str) or not entry[name].strip()
        ]
        if missing:
            raise ValueError(
                f"request {position} is missing {', '.join(missing)} — every "
                f"request needs {', '.join(REQUEST_FIELDS)}, each a non-empty "
                "string"
            )
        validated.append({name: entry[name].strip() for name in REQUEST_FIELDS})
    return validated


def file_requests(
    path: str | Path,
    *,
    node: int,
    session: str,
    entries: Any,
) -> List[Tuple[int, Optional[str]]]:
    """The tool's write: one ``requested`` event per entry, under one lock
    with the fold that decides ids and dedupe.

    A key that is currently open joins that request (its id comes back,
    nothing is written). A key whose latest request was answered gets a
    fresh request carrying ``previous_reply``, so the person sees the
    loop. Returns ``(id, previous_reply)`` per entry, in order.
    """
    validated = _validate_entries(entries)
    path = Path(path)
    results: List[Tuple[int, Optional[str]]] = []
    with _locked(path) as handle:
        requests = fold(_parse_events(handle.read(), str(path)))
        next_id = max(requests, default=0) + 1
        for entry in validated:
            same_key = [r for r in requests.values() if r.key == entry["key"]]
            still_open = [r for r in same_key if r.open]
            if still_open:
                results.append((still_open[-1].id, None))
                continue
            previous_reply = same_key[-1].reply if same_key else None
            event = {
                "ts": _utcnow(),
                "event": "requested",
                "id": next_id,
                "node": node,
                "session": session,
                **entry,
            }
            if previous_reply is not None:
                event["previous_reply"] = previous_reply
            handle.write(json.dumps(event) + "\n")
            requests[next_id] = Request(
                id=next_id,
                node=node,
                session=session,
                requested_at=event["ts"],
                previous_reply=previous_reply,
                **entry,
            )
            results.append((next_id, previous_reply))
            next_id += 1
    return results


# A reply that looks like a credential. Replies are stored in plain text
# and handed to the coder as text, so the value belongs in the file the
# fix names, never here. Shapes of common tokens — structural, not knobs.
SECRET_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[abp]-[A-Za-z0-9-]{20,}"),
)


def looks_like_a_secret(note: str) -> bool:
    return any(shape.search(note) for shape in SECRET_SHAPES)


def record_reply(
    path: str | Path, request_id: int, note: str, *, allow_secret: bool = False
) -> Request:
    """The person's answer. The request must exist, and a session must not
    have been given an earlier answer yet: until then the latest one wins.

    A note shaped like a credential is refused unless the caller says it
    knows (`allow_secret=True`, the CLI's --yes): the guard holds for every
    caller — the CLI, a script, a host plugin — not only the command line.
    """
    if looks_like_a_secret(note) and not allow_secret:
        raise ValueError(
            "reply looks like a credential; replies are stored in plain text "
            "and handed to the coder as text — put the value where the "
            "request's fix says (usually the .env the run loaded) and reply "
            "with a note, or pass allow_secret=True to store it anyway"
        )
    path = Path(path)
    with _locked(path) as handle:
        requests = fold(_parse_events(handle.read(), str(path)))
        if request_id not in requests:
            raise ValueError(f"inbox has no request #{request_id}")
        request = requests[request_id]
        if request.continued:
            raise ValueError(
                f"inbox request #{request_id} was already answered and handed "
                f"to its session: {request.reply!r}"
            )
        event = {"ts": _utcnow(), "event": "replied", "id": request_id, "note": note}
        handle.write(json.dumps(event) + "\n")
        request.reply = note
        request.replied_at = event["ts"]
        return request


def record_continued(
    path: str | Path, request_ids: Sequence[int], *, node: int, session: str
) -> None:
    """The orchestrator resumed the session these requests belong to."""
    path = Path(path)
    with _locked(path) as handle:
        requests = fold(_parse_events(handle.read(), str(path)))
        for request_id in request_ids:
            if request_id not in requests or requests[request_id].open:
                raise ValueError(
                    f"inbox request #{request_id} cannot be continued: "
                    "missing or still open"
                )
            handle.write(json.dumps({
                "ts": _utcnow(),
                "event": "continued",
                "id": request_id,
                "node": node,
                "session": session,
            }) + "\n")


def continue_answered(path: str | Path, *, node: int, session: str) -> List[int]:
    """The answered requests of ``node`` not yet continued, recorded as
    continued by ``session`` — a restarted session that carries the
    replies (the harness setup session, which the orchestrator runs again
    rather than resuming). Their ids, in request order."""
    pending = [
        request.id for request in load_requests(path).values()
        if request.node == node and not request.open and not request.continued
    ]
    if pending:
        record_continued(path, pending, node=node, session=session)
    return pending


def new_requests_for_session(
    path: str | Path, session_id: str, seen_bytes: int
) -> Tuple[int, List[int]]:
    """The adapters' tail: request ids filed for ``session_id`` in the
    bytes appended since ``seen_bytes``. Only complete lines are read; a
    line still being written is left for the next poll. Returns the new
    high-water mark and the ids."""
    path = Path(path)
    if not path.is_file():
        return seen_bytes, []
    size = path.stat().st_size
    if size <= seen_bytes:
        return seen_bytes, []
    with open(path, "rb") as handle:
        handle.seek(seen_bytes)
        chunk = handle.read()
    last_newline = chunk.rfind(b"\n")
    if last_newline < 0:
        return seen_bytes, []
    complete = chunk[: last_newline + 1]
    ids = []
    for event in _parse_events(complete.decode("utf-8"), str(path)):
        if event["event"] == "requested" and event.get("session") == session_id:
            ids.append(event["id"])
    return seen_bytes + len(complete), ids


def render_stop_text(results: Sequence[Tuple[int, Optional[str]]], keys: Sequence[str]) -> str:
    """The tool result the session reads (Appendix A.2)."""
    ids = [request_id for request_id, _ in results]
    label = "request" if len(ids) == 1 else "requests"
    lines = [
        f"Recorded as {label} {', '.join(f'#{i}' for i in ids)}. Your session "
        "is being stopped now — do nothing further. You will be resumed in "
        "this conversation with the person's reply."
    ]
    for (request_id, previous_reply), key in zip(results, keys):
        if previous_reply is not None:
            lines.append(
                f"Note: {key} was requested before in this campaign and "
                f"answered — {previous_reply!r}. The person will see that "
                f"reply next to request #{request_id}."
            )
    return "\n".join(lines)


# =============================================================================
# THE LAUNCH RECORD AND THE REGISTRY
# =============================================================================

def launch_record_path(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir) / LAUNCH_RELATIVE_PATH


def write_launch_record(workspace_dir: str | Path, record: Dict[str, Any]) -> Path:
    """Written once, on a fresh campaign; a resume never rewrites it.
    Paths are stored resolved: `kapso inbox reply` resumes from inside
    the campaign, not from the directory the launch ran in. Replaced
    atomically: every resume reads this file, and a torn one would strand
    the campaign."""
    path = launch_record_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": LAUNCH_SCHEMA_VERSION, **record}
    for key in _RECORDED_PATH_KEYS:
        if payload.get(key):
            payload[key] = _resolved_path(payload[key])
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_launch_record(workspace_dir: str | Path) -> Optional[Dict[str, Any]]:
    """The record, or None when the campaign predates it (documented
    default: such a campaign is not resumable from the inbox)."""
    path = launch_record_path(workspace_dir)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != LAUNCH_SCHEMA_VERSION:
        raise ValueError(f"{path}: not a launch record of schema {LAUNCH_SCHEMA_VERSION}")
    return record


# The launch arguments a resume takes from the record when the caller
# leaves them unset. Budgets are among them: a resume that names none keeps
# the launch's, and one that names a bigger one is allowed (the checkpoint
# does not fingerprint budgets).
RESUME_FROM_RECORD = (
    "host", "eval_dir", "data_dir", "experiments", "minutes", "usd", "reserve_minutes",
)
# The launch settings the checkpoint fingerprints. Changing one is refused
# by the checkpoint with a fingerprint mismatch that names nothing; naming
# the setting here is what the person needs.
FINGERPRINTED_LAUNCH_KEYS = ("host", "eval_dir", "config_path", "experience")
_PATH_LAUNCH_KEYS = ("eval_dir", "config_path")
# The loaded experience is campaign identity by its references and scopes;
# the heads are re-pinned from the record, so a bank that moved since the
# launch is not a change (docs/plans/redesign/wp3-experience.md).
_EXPERIENCE_LAUNCH_KEY = "experience"
_RECORDED_PATH_KEYS = (*_PATH_LAUNCH_KEYS, "data_dir")


def _resolved_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def resume_arguments(record: Dict[str, Any], requested: Dict[str, Any]) -> Dict[str, Any]:
    """What a resume runs with: every argument the caller passed, and the
    launch record's value for each one the caller left unset."""
    merged = dict(requested)
    for key in RESUME_FROM_RECORD:
        if merged.get(key) in (None, "") and key in record:
            merged[key] = record[key]
    return merged


def launch_mismatches(record: Dict[str, Any], requested: Dict[str, Any]) -> Dict[str, tuple]:
    """The fingerprinted launch settings a resume changed, as
    ``{setting: (recorded, requested)}``. Paths compare resolved, so a
    respelled directory is not a change."""

    def normal(key: str, value: Any) -> Any:
        if key == _EXPERIENCE_LAUNCH_KEY:
            return [(source["reference"], source["scope"]) for source in (value or [])]
        if value is None or key not in _PATH_LAUNCH_KEYS:
            return value
        return _resolved_path(value)

    return {
        key: (record.get(key), requested.get(key))
        for key in FINGERPRINTED_LAUNCH_KEYS
        if key in record and normal(key, record.get(key)) != normal(key, requested.get(key))
    }


def register_campaign(registry_path: str | Path, workspace_dir: str | Path, goal: str) -> None:
    path = Path(registry_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": _utcnow(),
        "path": str(Path(workspace_dir).resolve()),
        "goal": goal.strip().splitlines()[0] if goal.strip() else "",
    }
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(line) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def beside(campaign: Path, suffix: str) -> Path:
    """The file that lies beside a campaign directory, named after it: the
    log, the launch receipt, the learn log. The path is resolved first, so
    a campaign named `.` or `..` (an empty name, which `with_name`
    refuses) is the directory it stands for."""
    resolved = Path(campaign).resolve()
    return resolved.with_name(resolved.name + suffix)


def launch_receipt_path(campaign: Path) -> Path:
    """The detached launch's receipt, beside the log: the child's pid, the
    log's path, the argv and the time, written before the child starts, so
    `kapso watch` can tell a campaign that is starting from one that died
    before writing its status (wp6-observability.md, change 7)."""
    return beside(campaign, ".launch.json")


def read_launch_receipt(campaign: Path) -> Optional[Dict[str, Any]]:
    """The receipt of the last detached launch of `campaign`, or None
    when it was never launched detached."""
    path = launch_receipt_path(Path(campaign))
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or not isinstance(receipt.get("pid"), int) or "log" not in receipt:
        raise ValueError(f"{path}: a launch receipt carries a pid and a log")
    return receipt


def list_registered_campaigns(registry_path: str | Path) -> List[Dict[str, Any]]:
    """Registered campaigns that still exist, newest first, each path once
    under its latest launch (a campaign deleted and launched again at the
    same path has two lines). A campaign exists when its directory does,
    or when a launch receipt lies beside the path: a detached launch that
    died at a launch check made no directory, and the receipt is what
    `kapso watch` tells it by. Missing registry → none; a malformed line
    raises."""
    path = Path(registry_path).expanduser()
    if not path.is_file():
        return []
    campaigns: Dict[str, Dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"{path}:{number}: registry line must carry a path")
        campaign = Path(entry["path"])
        if campaign.is_dir() or launch_receipt_path(campaign).is_file():
            campaigns.pop(entry["path"], None)
            campaigns[entry["path"]] = entry
    return list(reversed(campaigns.values()))


# =============================================================================
# WHAT THE PERSON READS
# =============================================================================

def idea_line(solution: str) -> str:
    """The idea in one line, for the `for` row: the first content line
    after a `# Core Idea` heading when there is one, else the first
    non-empty line. Display only — never model input."""
    lines = [line.strip() for line in solution.splitlines()]
    content = [line for line in lines if line]
    for index, line in enumerate(content):
        if line.lower().lstrip("# ").startswith("core idea") and index + 1 < len(content):
            return content[index + 1].lstrip("-* ")
    return content[0].lstrip("#-* ").strip() if content else ""


def request_payload(request: Request, idea: str) -> Dict[str, Any]:
    """One request as the status file, the result metadata and the
    `on_status` hook carry it."""
    return {
        "id": request.id,
        "node": request.node,
        "key": request.key,
        "for": idea,
        "hit": request.hit,
        "tried": request.tried,
        "fix": request.fix,
        "next_steps": request.next_steps,
        "previous_reply": request.previous_reply,
        "requested_at": request.requested_at,
    }


def render_requests(requests: Sequence[Request], ideas: Dict[int, str]) -> str:
    """The request rows of the pause message and `kapso inbox`."""
    blocks = []
    for request in requests:
        again = (
            f"   again — your previous reply was: {request.previous_reply!r}"
            if request.previous_reply is not None else ""
        )
        rows = [f"  #{request.id}  {request.key}{again}"]
        idea = ideas.get(request.node, "")
        if idea:
            rows.append(f"      for    node {request.node} · {idea}")
        else:
            rows.append(f"      for    node {request.node}")
        rows.extend([
            f"      hit    {request.hit}",
            f"      tried  {request.tried}",
            f"      fix    {request.fix}",
            f"      next   {request.next_steps}",
        ])
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def render_awaiting_resume(requests: Sequence[Request], ideas: Dict[int, str]) -> str:
    """The rows of `kapso inbox` for replies a resume has not delivered."""
    blocks = []
    for request in requests:
        idea = ideas.get(request.node, "")
        blocks.append("\n".join([
            f"  #{request.id}  {request.key}   answered, campaign not resumed",
            f"      for    node {request.node} · {idea}" if idea else f"      for    node {request.node}",
            f"      reply  {request.reply!r}",
        ]))
    return "\n\n".join(blocks)


def render_pause_block(
    requests: Sequence[Request], ideas: Dict[int, str], campaign_dir: str | Path
) -> str:
    """What the terminal shows when the campaign pauses (design v4 §4.6)."""
    first = requests[0].id if requests else 1
    reply_target = f" {first}" if len(requests) > 1 else ""
    return "\n".join([
        "kapso build — waiting on you",
        "",
        render_requests(requests, ideas),
        "",
        f'  reply with   kapso inbox reply{reply_target} "…"        # after the fix, or with what to do instead',
        f"  any time     kapso inbox {campaign_dir}",
    ])
