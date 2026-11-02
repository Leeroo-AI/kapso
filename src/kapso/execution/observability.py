# Kapso observability — one status layer for every long-running operation.
#
# Design: docs/research/evolve-observability-design.md (v3, approved
# 2026-08-25). One abstract class owns the mechanics — atomic status-file
# writes, the state machine, phase timing, heartbeat, the recent ring —
# and each operation contributes ONLY its phase list and payload fields
# (subclasses may add fields and phases, never mechanics). The surface is
# deliberately TIME-ONLY: codex sessions report $0.00, so any displayed
# dollar figure would lie for the platform's default implementor. The
# platform-cost bar is in the campaign's own unit (cost.unit, e.g.
# credits), never dollars, so that stance holds.

import json
import os
import tempfile
import threading
from abc import ABC
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from kapso.core.processes import process_start
from kapso.execution.inbox import list_registered_campaigns, read_launch_receipt
from kapso.execution.iteration_evaluator import comparison_summary

# Structural constant, not a knob: how many consecutive heartbeats may be
# missed before a reader calls the operation stalled.
STALL_MISSED_HEARTBEATS = 3

# The recent ring's size — a human glance, not a log (design §1).
RECENT_RING_SIZE = 10

TERMINAL_STATES = ("done", "failed")


def knowledge_summary(phase_telemetry: Mapping[str, Mapping[str, float]]) -> str:
    """What the gates were asked, from the per-gate call keys the phases
    record (`<gate>_calls`, `<gate>.<tool>`), summed across the phases
    given: "leeroopedia 2 (build_plan, diagnose_failure); repo_memory 3".
    Empty when no gate was consulted. Any mapping of label -> phase
    values works, so a campaign total is the nodes' phases merged."""
    totals: Dict[str, float] = {}
    tools: Dict[str, Dict[str, float]] = {}
    for phase_values in phase_telemetry.values():
        for key, count in phase_values.items():
            if key.endswith("_calls"):
                gate = key[: -len("_calls")]
                totals[gate] = totals.get(gate, 0) + count
            elif "." in key:
                gate, tool = key.split(".", 1)
                per_gate = tools.setdefault(gate, {})
                per_gate[tool] = per_gate.get(tool, 0) + count
    parts = []
    for gate, total in sorted(totals.items(), key=lambda item: -item[1]):
        detail = ", ".join(
            f"{tool}×{int(n)}" if n > 1 else tool
            for tool, n in sorted(tools.get(gate, {}).items(), key=lambda item: -item[1])
        )
        parts.append(f"{gate} {int(total)}" + (f" ({detail})" if detail else ""))
    return "; ".join(parts)


def credits_summary(phase_telemetry: Mapping[str, Mapping[str, float]]) -> str:
    """What the metered gates reported, from the `<gate>_credits_first` /
    `<gate>_credits_last` keys: "leeroopedia 761 → 747 remaining" — the
    balance after the earliest answer to the balance after the latest,
    over all the phases given (lanes run in parallel, so earliest is the
    highest balance and latest the lowest). Empty when no metered gate
    answered."""
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    for phase_values in phase_telemetry.values():
        for key, balance in phase_values.items():
            if key.endswith("_credits_first"):
                gate = key[: -len("_credits_first")]
                first[gate] = max(first.get(gate, balance), balance)
            elif key.endswith("_credits_last"):
                gate = key[: -len("_credits_last")]
                last[gate] = min(last.get(gate, balance), balance)
    return "; ".join(
        f"{gate} {int(first[gate])} → {int(last[gate])} remaining"
        for gate in sorted(first) if gate in last
    )


def platform_cost_summary(platform_budget: Mapping[str, Any]) -> str:
    """The status payload's platform_budget {used, budget, unit, refused}
    as one clause: "2.0/3.0 credits, 1 candidate refused". No budget
    renders "2.0 credits"; the refusal clause appears only when a
    candidate was refused."""
    used = float(platform_budget["used"])
    budget = platform_budget["budget"]
    text = (
        f"{used}/{float(budget)} {platform_budget['unit']}" if budget is not None
        else f"{used} {platform_budget['unit']}"
    )
    refused = platform_budget["refused"]
    if refused == 1:
        text += ", 1 candidate refused"
    elif refused > 1:
        text += f", {refused} candidates refused"
    return text


def pid_alive(pid: Optional[int], started: Optional[str] = None) -> Optional[bool]:
    """Whether a process runs at `pid`, and is the one that recorded `started`
    (its start stamp) when one was recorded: a later process handed the same
    pid does not count. None with no pid, or where the kernel will not say."""
    if pid is None:
        return None
    current = process_start(pid)
    if current is None:
        return None
    return bool(current) and started in (None, current)


def launch_alive(receipt: Mapping[str, Any]) -> Optional[bool]:
    """Whether the child a launch receipt names still runs."""
    return pid_alive(receipt["pid"], receipt.get("pid_started"))


def source_line(source: Mapping[str, Any]) -> str:
    """One source as the status file records it (reference, kind, head,
    provenance, scope), the way the launch banner names it."""
    reference, kind, provenance = source["reference"], source["kind"], source["provenance"]
    head = (source.get("head") or "")[:12]
    if kind == "bank":
        every = ", every scope" if source.get("scope") == "all" else ""
        return f"{reference} at {head}, {provenance}{every}"
    if kind == "index":
        return f"{reference} (index {head}), {provenance}"
    return f"{reference} (hosted), {provenance}"


def campaign_listing(registry_path, status_dir, *, everything: bool = False) -> list:
    """What `kapso watch` lists with no path: the campaigns the registry
    knows and the learn and experience operations under the status
    directory, one entry each, the registry's newest first; those whose
    writer runs (the view's own verdict, `starting` included, `stalled`
    marking one past its beats, and a kernel that cannot say judged by the
    heartbeat) or waiting on the person, unless
    `everything`. A registered campaign with no status yet is `starting`
    on its launch receipt while the pid lives, `stopped` once it is gone."""
    entries = []
    for registered in list_registered_campaigns(registry_path):
        campaign = Path(registered["path"])
        located = OperationStatusView.locate(campaign)
        if located is not None:
            entries.append(_listing_entry(OperationStatusView(located), str(campaign)))
            continue
        receipt = read_launch_receipt(campaign)
        if receipt is None:
            continue
        alive = launch_alive(receipt)
        entries.append({
            "path": str(campaign), "operation": "build",
            "state": "starting" if alive is not False else "stopped",
            "phase": None, "heartbeat_age_seconds": None, "best": None,
            "stopped_reason": None, "alive": alive, "stalled": False,
        })
    root = Path(status_dir).expanduser()
    if root.is_dir():
        for file in sorted(root.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            entries.append(_listing_entry(OperationStatusView(file), str(file)))
    if everything:
        return entries
    return [
        entry for entry in entries
        if entry["alive"] is not False
        or (entry["state"] == "done" and entry["stopped_reason"] == "waiting_for_user")
    ]


def _listing_entry(view: "OperationStatusView", path: str) -> Dict[str, Any]:
    return {
        "path": path,
        "operation": view.operation,
        "state": view.state,
        "phase": view.phase,
        "heartbeat_age_seconds": round(view.heartbeat_age_seconds, 1),
        "best": view.best,
        "stopped_reason": view.data.get("stopped_reason"),
        "alive": view.running,
        "stalled": view.stalled and view.process_alive is True,
    }


def render_listing(entries: list) -> str:
    """One line per entry: the state (WAITING for a campaign waiting on the
    person), the operation, the phase, the heartbeat's age, the best score
    and the path."""
    lines = []
    for entry in entries:
        waiting = entry["state"] == "done" and entry["stopped_reason"] == "waiting_for_user"
        # A writer gone is dead, whatever the file's own word; where the
        # kernel cannot say, so is one silent past its beats. The
        # single-campaign screen says the same.
        dead = entry["alive"] is False and entry["state"] in ("running", "starting")
        state = "WAITING" if waiting else "DEAD" if dead else "STALLED" if entry["stalled"] else entry["state"].upper()
        age = f"{entry['heartbeat_age_seconds']:.0f}s ago" if entry["heartbeat_age_seconds"] is not None else "-"
        best = entry["best"].get("score") if isinstance(entry["best"], dict) else None
        lines.append(
            f"{state:<9} {entry['operation']:<10} {entry['phase'] or '-':<15} {age:>10}  "
            f"best {best if best is not None else '-':<8} {entry['path']}"
        )
    return "\n".join(lines)


def _progress_bar(fraction: float) -> str:
    filled = int(round(20 * min(1.0, fraction)))
    return "▓" * filled + "░" * (20 - filled)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def mark_failed(path, error: BaseException | str, **fields: Any) -> bool:
    """Publish a failure on a status file whose writer is gone: the
    strategy constructor or the orchestrator's setup raised after the
    status was born, and the object that owned the file went with the
    exception; or a person ended the run (`kapso build --stop`, a string
    with `stopped_reason`). The file is rewritten as the writer would
    have written it (`failed`, the error, `finished_at`, the fields);
    nothing is written when there is no file or it is terminal already.
    Returns whether it wrote."""
    file = Path(path)
    if not file.is_file():
        return False
    data = json.loads(file.read_text(encoding="utf-8"))
    if data["state"] in TERMINAL_STATES:
        return False
    data["state"] = "failed"
    data["finished_at"] = _utcnow()
    data["error"] = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    data.update(fields)
    temporary = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, file)
    return True


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
      (a build runs it through setup and, once the loop starts, its
      checkpoint-heartbeat daemon beats the file as well);
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
        phase: Optional[str] = None,
        **payload: Any,
    ):
        if not self.OPERATION or not self.PHASES:
            raise ValueError(
                "OperationStatus subclasses must define OPERATION and PHASES"
            )
        if phase is not None and phase not in self.PHASES:
            raise ValueError(
                f"unknown {self.OPERATION} phase {phase!r}: "
                f"expected one of {list(self.PHASES)}"
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
            "pid_started": process_start(os.getpid()),
            "started_at": now,
            "heartbeat_at": now,
            # A phase at birth keeps the state `starting`: the operation
            # is preparing (a build seeds its workspace), not yet running.
            "phase": phase,
            "phase_started_at": now if phase is not None else None,
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

    @property
    def terminal(self) -> bool:
        """Done or failed: this status accepts no further update. A new
        slice of the same operation opens a fresh one."""
        return self._data["state"] in TERMINAL_STATES

    def observe(
        self, on_status: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        """Attach (or replace) the hook after construction. A build's status
        is born at workspace setup, before solve() knows its caller's
        hook; the hook then sees every write from the first iteration on."""
        with self._lock:
            self._on_status = on_status

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
        if self._data["state"] in TERMINAL_STATES:
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
            if self._data["state"] in TERMINAL_STATES:
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


class BuildStatus(OperationStatus):
    OPERATION = "build"
    # `setup` is the strategy constructor's phase: the workspace is seeded
    # and repo memory bootstrapped (one inference session, minutes) before
    # the first iteration. The file exists from that moment, so a watcher
    # sees the campaign from its first seconds.
    # `harness_setup` is the harness's own setup session (a domain statement
    # and prepared resources), once per campaign before the first
    # iteration; `verification` is the harness's per-attempt check between
    # the evaluation and the judge. Both exist only with a harness.
    PHASES = (
        "setup", "harness_setup", "lens_planning", "ideation", "implementation",
        "evaluation", "verification", "feedback",
    )
    # payload: budget{elapsed_min,total_min}, best{score,node},
    #          last{score,node}, iteration, stopped_reason, stop_detail,
    #          requests (waiting_for_user), platform_budget{used,budget,
    #          unit,refused} (only when cost is enabled), harness{name,
    #          version,hash,source} (null without one), harness_setup
    #          {prepared} once the harness setup session ran. Documented in
    #          docs/reference/cli.mdx under `kapso watch` — keep in step.


class LessonStatus(OperationStatus):
    OPERATION = "learn"
    PHASES = ("harvest", "mine", "exam", "lesson", "push")
    # payload: trajectory_id, bank_head_before, repair_round,
    #          cards{created,updated} (filled at lesson end)


class KnowledgeStatus(OperationStatus):
    OPERATION = "experience"
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
    def locate(path: str | Path) -> Optional[Path]:
        """Where `path`'s status file is, or None when there is none yet:
        a workspace (-> .kapso/status.json), a status file, or a directory
        of status files (-> newest by mtime).

        A workspace whose status file is not there yet is None rather than
        some other JSON beside it: a seeded project carries files like
        result.json, and the status file appears only once the strategy
        has seeded the workspace."""
        p = Path(path).expanduser()
        if p.is_file():
            return p
        workspace_status = p / ".kapso" / "status.json"
        if workspace_status.is_file():
            return workspace_status
        if (p / ".kapso").is_dir():
            return None
        if p.is_dir():
            candidates = sorted(
                p.glob("*.json"), key=lambda f: f.stat().st_mtime
            )
            if candidates:
                return candidates[-1]
        return None

    @staticmethod
    def resolve(path: str | Path) -> Path:
        """locate(), raising when there is no status file yet."""
        located = OperationStatusView.locate(path)
        if located is not None:
            return located
        p = Path(path).expanduser()
        if (p / ".kapso").is_dir():
            raise FileNotFoundError(
                f"{p} is a campaign workspace with no status file yet: it is "
                "still seeding its workspace, or it died before its first "
                "heartbeat"
            )
        raise FileNotFoundError(
            f"no status file at {p} (looked for the file itself, "
            f"{p / '.kapso' / 'status.json'}, and *.json inside it)"
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

    @cached_property
    def process_alive(self) -> Optional[bool]:
        """Whether the process that wrote the file still runs. The status file
        cannot record its own writer's death, so a fresh heartbeat proves
        nothing once the process is gone. A pid handed to a later process does
        not count: its start stamp differs. None where the kernel will not say;
        the heartbeat is then the only signal."""
        return pid_alive(self.data.get("pid"), self.data.get("pid_started"))

    @property
    def dead(self) -> bool:
        return self.state not in TERMINAL_STATES and self.process_alive is False

    @property
    def alive(self) -> Optional[bool]:
        """True when running with a live writer and a fresh heartbeat; False
        when the writer is gone (DEAD), the heartbeat is stale (STALLED) or
        the state is terminal; None when the file records no heartbeat
        cadence to judge staleness against."""
        if self.state in TERMINAL_STATES or self.dead:
            return False
        interval = self.data.get("heartbeat_seconds")
        if not interval:
            return None
        return self.heartbeat_age_seconds < (
            STALL_MISSED_HEARTBEATS * float(interval)
        )

    @property
    def stalled(self) -> bool:
        return self.alive is False and self.state not in TERMINAL_STATES

    @property
    def running(self) -> Optional[bool]:
        """Whether something is still working on this operation, which is what
        a second session must not race. A process the kernel says is running
        holds it however old its heartbeat is, because a heartbeat also stops
        for a suspended process, a sleeping laptop or a heartbeat thread that
        died. Where the kernel cannot say, this is `alive`, the heartbeat's
        verdict, None included."""
        if self.state in TERMINAL_STATES:
            return False
        if self.process_alive is None:
            return self.alive
        return self.process_alive

    # ---------------------------------------------------------- renderer

    def explain(self) -> str:
        """The watch screen as a string — the CLI shows this verbatim."""
        d = self.data
        age = self.heartbeat_age_seconds
        head = f"{d['state'].upper()} ♥ {age:.0f}s ago"
        if self.dead:
            head += f"   ⚠ DEAD (pid {d.get('pid')} is gone)"
        elif self.stalled:
            head += "   ⚠ STALLED?"
        lines = [head + f"      pid {d.get('pid')}"]
        # The campaign's identity from the first second: the harness and
        # the sources it is served (wp6-observability.md, change 7).
        if d.get("harness"):
            harness = d["harness"]
            lines.append(f"harness: {harness['name']} {harness['version']} ({harness['hash'][:12]})")
        for source in d.get("experience") or []:
            lines.append(f"served: {source_line(source)}")
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
        title = f" {self.operation} {self.path} "
        width = max(
            [len(line) for line in lines] + [len(title)]
        ) + 2
        bordered = [f"┌─{title}".ljust(width + 1, "─") + "┐"]
        bordered.extend(f"│ {line}".ljust(width + 1) + "│" for line in lines)
        bordered.append("└" + "─" * width + "┘")
        return "\n".join(bordered)

    def _operation_block(self) -> list:
        d = self.data
        lines = []
        if self.operation == "build":
            budget = d.get("budget") or {}
            if budget.get("total_min"):
                elapsed = float(budget.get("elapsed_min") or 0.0)
                total = float(budget["total_min"])
                lines.append(
                    f"budget: {_progress_bar(elapsed / total)} "
                    f"{elapsed:.0f}/{total:.0f} min"
                )
            platform = d.get("platform_budget")
            if platform:
                summary = platform_cost_summary(platform)
                if platform["budget"]:
                    bar = _progress_bar(platform["used"] / platform["budget"])
                    lines.append(f"cost:   {bar} {summary}")
                else:
                    lines.append(f"cost:   {summary}")
            if d.get("iteration") is not None:
                lines.append(f"iteration {d['iteration']}")
            if d.get("stopped_reason") == "waiting_for_user":
                requests = d.get("requests") or []
                label = "request" if len(requests) == 1 else "requests"
                lines.append(f"WAITING ON YOU · {len(requests)} {label}")
                for request in requests:
                    lines.append(f"  #{request['id']}  {request['key']}")
                    lines.append(f"      fix  {request['fix']}")
                lines.append("  reply with: kapso inbox reply <id> \"…\"")
            best, last = d.get("best"), d.get("last")
            if best or last:
                parts = []
                if best:
                    parts.append(f"best: {best.get('score')}  node {best.get('node')}")
                if last:
                    parts.append(f"last: {last.get('score')}  node {last.get('node')}")
                lines.append("      ".join(parts))
            external = d.get("external_evaluation") or {}
            comparison = comparison_summary(external.get("metadata") or {})
            if comparison:
                lines.append(
                    f"external node {external.get('node')}: {comparison}"
                )
            if external.get("error"):
                lines.append(f"external evaluation error: {external['error']}")
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
        elif self.operation == "experience":
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
