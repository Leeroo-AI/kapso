# Kapso observability — one status layer for every long-running operation.
#
# Design: docs/research/evolve-observability-design.md (v3, approved
# 2026-08-25). One abstract class owns the mechanics — atomic status-file
# writes, the state machine, phase timing, heartbeat, the recent ring —
# and each operation contributes ONLY its phase list and payload fields
# (subclasses may add fields and phases, never mechanics). The surface is
# deliberately TIME-ONLY: codex sessions report $0.00, so any displayed
# dollar figure would lie for the platform's default implementor.

import json
import os
import tempfile
import threading
from abc import ABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Structural constant, not a knob: how many consecutive heartbeats may be
# missed before a reader calls the operation stalled.
STALL_MISSED_HEARTBEATS = 3

# The recent ring's size — a human glance, not a log (design §1).
RECENT_RING_SIZE = 10

_TERMINAL_STATES = ("done", "failed")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


class OperationStatus(ABC):
    """Mechanics shared by every long-running operation.

    - atomic status-file write (tmp + fsync + replace, the checkpoint
      pattern) so a reader never sees a torn file;
    - state machine: starting -> running -> done | failed, terminal
      states are final (a later update RAISES — that is a wiring bug);
    - phase tracking: ``phase`` + ``phase_started_at`` (reset on change),
      legal values come from the subclass's PHASES;
    - ``heartbeat_at`` refreshed on every write, plus an OPTIONAL daemon
      thread for operations with no natural per-minute update site
      (evolve reuses its checkpoint-heartbeat daemon instead);
    - ``recent``: a 10-line human ring inside the file;
    - an optional ``on_status`` hook, invoked synchronously with the SAME
      dict the file carries after every write and heartbeat. Hook
      exceptions PROPAGATE (Rule 2): it is caller-owned code.
    """

    OPERATION: str = ""
    PHASES: Tuple[str, ...] = ()

    def __init__(
        self,
        path: str | Path,
        *,
        heartbeat_seconds: Optional[float] = None,
        daemon: bool = False,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        **payload: Any,
    ):
        if not self.OPERATION or not self.PHASES:
            raise ValueError(
                "OperationStatus subclasses must define OPERATION and PHASES"
            )
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._on_status = on_status
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        now = _utcnow()
        self._data: Dict[str, Any] = {
            "operation": self.OPERATION,
            "phases": list(self.PHASES),
            "state": "starting",
            "pid": os.getpid(),
            "started_at": now,
            "heartbeat_at": now,
            "phase": None,
            "phase_started_at": None,
            "recent": [],
            **payload,
        }
        if heartbeat_seconds is not None:
            self._data["heartbeat_seconds"] = float(heartbeat_seconds)
        self._write()
        if daemon:
            if not heartbeat_seconds:
                raise ValueError(
                    "daemon=True requires heartbeat_seconds (source it from "
                    "budget.checkpoint_heartbeat_seconds)"
                )
            self._thread = threading.Thread(
                target=self._run_daemon,
                args=(float(heartbeat_seconds),),
                name=f"{self.OPERATION}-status-heartbeat",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------ writes

    def _write(self) -> None:
        """Atomic write + hook. Callers hold the lock (or are __init__)."""
        self._data["heartbeat_at"] = _utcnow()
        descriptor, temp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".status.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=1, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.path)
        if self._on_status is not None:
            self._on_status(dict(self._data))

    def _assert_live(self) -> None:
        if self._data["state"] in _TERMINAL_STATES:
            raise RuntimeError(
                f"{self.OPERATION} status is already "
                f"{self._data['state']!r} — updates after a terminal state "
                "are a wiring bug"
            )

    def update(self, **fields: Any) -> None:
        """Merge payload fields; flips starting -> running."""
        with self._lock:
            self._assert_live()
            self._data["state"] = "running"
            self._data.update(fields)
            self._write()

    def phase(self, name: str, **fields: Any) -> None:
        """Enter a phase (resets the phase timer on change)."""
        if name not in self.PHASES:
            raise ValueError(
                f"unknown {self.OPERATION} phase {name!r}: "
                f"expected one of {list(self.PHASES)}"
            )
        with self._lock:
            self._assert_live()
            self._data["state"] = "running"
            if self._data["phase"] != name:
                self._data["phase"] = name
                self._data["phase_started_at"] = _utcnow()
            self._data.update(fields)
            self._write()

    def note(self, line: str, **fields: Any) -> None:
        """Append one human line to the recent ring (oldest drops)."""
        with self._lock:
            self._assert_live()
            self._data["state"] = "running"
            stamp = datetime.now(timezone.utc).strftime("%H:%M")
            ring = list(self._data["recent"]) + [f"{stamp} {line}"]
            self._data["recent"] = ring[-RECENT_RING_SIZE:]
            self._data.update(fields)
            self._write()

    def heartbeat(self, **fields: Any) -> None:
        """Refresh liveness (and optionally payload). No-op once terminal
        — the daemon may race the final write, and a heartbeat after done
        is noise, not a wiring bug."""
        with self._lock:
            if self._data["state"] in _TERMINAL_STATES:
                return
            self._data.update(fields)
            self._write()

    def done(self, **fields: Any) -> None:
        with self._lock:
            self._assert_live()
            self._data["state"] = "done"
            self._data["finished_at"] = _utcnow()
            self._data.update(fields)
            self._write()
        self._stop_daemon()

    def failed(self, error: BaseException | str, **fields: Any) -> None:
        with self._lock:
            self._assert_live()
            self._data["state"] = "failed"
            self._data["finished_at"] = _utcnow()
            self._data["error"] = (
                f"{type(error).__name__}: {error}"
                if isinstance(error, BaseException) else str(error)
            )
            self._data.update(fields)
            self._write()
        self._stop_daemon()

    # ------------------------------------------------------------ daemon

    def _run_daemon(self, interval_seconds: float) -> None:
        while not self._stop.wait(interval_seconds):
            self.heartbeat()

    def _stop_daemon(self) -> None:
        self._stop.set()


class EvolveStatus(OperationStatus):
    OPERATION = "evolve"
    PHASES = (
        "lens_planning", "ideation", "implementation", "evaluation",
        "feedback",
    )
    # payload: budget{elapsed_min,total_min}, best{score,node},
    #          last{score,node}, iteration, active_stream


class LessonStatus(OperationStatus):
    OPERATION = "learn"
    PHASES = ("harvest", "mine", "exam", "lesson", "push")
    # payload: trajectory_id, bank_head_before, repair_round,
    #          cards{created,updated} (filled at lesson end)


class KnowledgeStatus(OperationStatus):
    OPERATION = "learn_knowledge"
    PHASES = ("ingest", "merge")
    # payload: sources{done,total}, current_source, pages_extracted


# =========================================================================
# The reader — one truth for `kapso watch` and Kapso.status()
# =========================================================================

class OperationStatusView:
    """Typed read-only view over a status file.

    Corrupt JSON raises (Rule 2); a missing file raises FileNotFoundError
    — "no status yet" is the caller's state to handle, not a default.
    """

    def __init__(self, path: str | Path):
        resolved = self.resolve(path)
        self.path = resolved
        self.data: Dict[str, Any] = json.loads(
            resolved.read_text(encoding="utf-8")
        )

    @staticmethod
    def resolve(path: str | Path) -> Path:
        """A workspace (-> .kapso/status.json), a status file, or a
        directory of status files (-> newest by mtime)."""
        p = Path(path).expanduser()
        if p.is_file():
            return p
        workspace_status = p / ".kapso" / "status.json"
        if workspace_status.is_file():
            return workspace_status
        if p.is_dir():
            candidates = sorted(
                p.glob("*.json"), key=lambda f: f.stat().st_mtime
            )
            if candidates:
                return candidates[-1]
        raise FileNotFoundError(
            f"no status file at {p} (looked for the file itself, "
            f"{workspace_status}, and *.json inside it)"
        )

    # ------------------------------------------------------------ fields

    @property
    def operation(self) -> str:
        return self.data["operation"]

    @property
    def state(self) -> str:
        return self.data["state"]

    @property
    def phase(self) -> Optional[str]:
        return self.data.get("phase")

    @property
    def recent(self) -> list:
        return list(self.data.get("recent", []))

    @property
    def budget(self) -> Optional[Dict[str, Any]]:
        return self.data.get("budget")

    @property
    def best(self) -> Optional[Dict[str, Any]]:
        return self.data.get("best")

    @property
    def heartbeat_age_seconds(self) -> float:
        beat = _parse_at(self.data["heartbeat_at"])
        return (datetime.now(timezone.utc) - beat).total_seconds()

    @property
    def phase_elapsed_min(self) -> Optional[float]:
        started = self.data.get("phase_started_at")
        if not started:
            return None
        delta = datetime.now(timezone.utc) - _parse_at(started)
        return round(delta.total_seconds() / 60, 1)

    @property
    def alive(self) -> Optional[bool]:
        """True when running with a fresh heartbeat; False when running
        stale (STALLED) or terminal; None when the file records no
        heartbeat cadence to judge staleness against."""
        if self.state in _TERMINAL_STATES:
            return False
        interval = self.data.get("heartbeat_seconds")
        if not interval:
            return None
        return self.heartbeat_age_seconds < (
            STALL_MISSED_HEARTBEATS * float(interval)
        )

    @property
    def stalled(self) -> bool:
        return self.alive is False and self.state not in _TERMINAL_STATES

    # ---------------------------------------------------------- renderer

    def explain(self) -> str:
        """The watch screen as a string — the CLI shows this verbatim."""
        d = self.data
        age = self.heartbeat_age_seconds
        head = f"{d['state'].upper()} ♥ {age:.0f}s ago"
        if self.stalled:
            head += "   ⚠ STALLED?"
        lines = [head + f"      pid {d.get('pid')}"]
        if self.phase:
            elapsed = self.phase_elapsed_min
            lines.append(
                f"phase: {self.phase}"
                + (f" — {elapsed}m elapsed" if elapsed is not None else "")
            )
        lines.extend(self._operation_block())
        if d.get("error"):
            lines.append(f"error: {d['error']}")
        if self.recent:
            lines.append("recent:")
            lines.extend(f"  {line}" for line in self.recent)
        width = max(len(line) for line in lines) + 2
        title = f" {self.operation} {self.path} "
        bordered = [f"┌─{title}".ljust(width + 1, "─") + "┐"]
        bordered.extend(f"│ {line}".ljust(width + 1) + "│" for line in lines)
        bordered.append("└" + "─" * width + "┘")
        return "\n".join(bordered)

    def _operation_block(self) -> list:
        d = self.data
        lines = []
        if self.operation == "evolve":
            budget = d.get("budget") or {}
            if budget.get("total_min"):
                elapsed = float(budget.get("elapsed_min") or 0.0)
                total = float(budget["total_min"])
                filled = int(round(20 * min(1.0, elapsed / total)))
                lines.append(
                    f"budget: {'▓' * filled}{'░' * (20 - filled)} "
                    f"{elapsed:.0f}/{total:.0f} min"
                )
            if d.get("iteration") is not None:
                lines.append(f"iteration {d['iteration']}")
            best, last = d.get("best"), d.get("last")
            if best or last:
                parts = []
                if best:
                    parts.append(f"best: {best.get('score')}  node {best.get('node')}")
                if last:
                    parts.append(f"last: {last.get('score')}  node {last.get('node')}")
                lines.append("      ".join(parts))
        elif self.operation == "learn":
            chain = []
            reached = True
            for phase_name in d.get("phases", []):
                if phase_name == self.phase:
                    elapsed = self.phase_elapsed_min
                    chain.append(
                        f"{phase_name} ({elapsed}m) …" if elapsed is not None
                        else f"{phase_name} …"
                    )
                    reached = False
                elif reached and self.state != "starting":
                    chain.append(f"{phase_name} ✓")
                else:
                    chain.append(f"{phase_name} ·")
            if self.state == "done":
                chain = [f"{p} ✓" for p in d.get("phases", [])]
            lines.append("  ".join(chain))
            if d.get("trajectory_id"):
                lines.append(f"trajectory {d['trajectory_id']}")
            if d.get("bank_head_before"):
                head_line = f"bank head {d['bank_head_before'][:8]} (pinned pre-lesson)"
                if d.get("repair_round"):
                    head_line += f"      repair round {d['repair_round']}"
                lines.append(head_line)
        elif self.operation == "learn_knowledge":
            sources = d.get("sources") or {}
            if sources:
                lines.append(
                    f"sources {sources.get('done', 0)}/{sources.get('total', '?')}"
                )
            if d.get("current_source"):
                lines.append(f"current: {d['current_source']}")
            if d.get("pages_extracted") is not None:
                lines.append(f"pages extracted so far: {d['pages_extracted']}")
        return lines
