"""The campaign inbox: what a session asked a person for, and what the
person replied.

Design: docs/research/evolve-hub-design.md (v4). One append-only JSONL
file per campaign, ``.kapso/inbox.jsonl``, holding events about
requests: ``requested`` (written by the session through the
``request_from_user`` tool), ``replied`` (written by ``kapso inbox
reply``) and ``continued`` (written by the orchestrator when the session
was resumed). A request's state is the fold of its events; nothing is
ever rewritten, and no request carries a value.

Also the two small records that make a reply self-sufficient: the launch
record (``.kapso/launch.json`` — the evolve arguments the checkpoint does
not hold) and the campaign registry (one line per launch, so ``kapso
inbox`` with no campaign can list every campaign waiting on a person).

Every read of the inbox file raises on a malformed line (Rule 2); a
missing file is the documented "no requests" default.
"""

import fcntl
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from contextlib import contextmanager

INBOX_RELATIVE_PATH = Path(".kapso") / "inbox.jsonl"
LAUNCH_RELATIVE_PATH = Path(".kapso") / "launch.json"
LAUNCH_SCHEMA_VERSION = 1

# The fields a session supplies for one request (Appendix A.2). All are
# required, all are non-empty strings.
REQUEST_FIELDS = ("key", "hit", "tried", "fix", "next_steps")
EVENT_TYPES = ("requested", "replied", "continued")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inbox_path(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir) / INBOX_RELATIVE_PATH


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

@contextmanager
def _locked(path: Path) -> Iterator[Any]:
    """The inbox file opened for append-plus-read under an exclusive
    lock, so a read-fold-append sequence sees and writes a consistent
    file even with the gate, the CLI and the orchestrator all writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        yield handle
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


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
    return _parse_events(path.read_text(encoding="utf-8"), str(path))


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
            if request.reply is not None:
                raise ValueError(f"inbox request #{request_id} replied twice")
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


def record_reply(path: str | Path, request_id: int, note: str) -> Request:
    """The person's answer. The request must exist and be open."""
    path = Path(path)
    with _locked(path) as handle:
        requests = fold(_parse_events(handle.read(), str(path)))
        if request_id not in requests:
            raise ValueError(f"inbox has no request #{request_id}")
        request = requests[request_id]
        if not request.open:
            raise ValueError(
                f"inbox request #{request_id} was already answered: "
                f"{request.reply!r}"
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
    """Written once, on a fresh campaign; a resume never rewrites it."""
    path = launch_record_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": LAUNCH_SCHEMA_VERSION, **record}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def list_registered_campaigns(registry_path: str | Path) -> List[Dict[str, Any]]:
    """Registered campaigns whose directory still exists, newest first.
    Missing registry → none; a malformed line raises."""
    path = Path(registry_path).expanduser()
    if not path.is_file():
        return []
    campaigns = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"{path}:{number}: registry line must carry a path")
        if Path(entry["path"]).is_dir():
            campaigns.append(entry)
    campaigns.reverse()
    return campaigns
