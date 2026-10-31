"""The timeline of a campaign (docs/plans/redesign/wp9-semantic-suite.md,
change 8): the status file sampled at the engine config's heartbeat
cadence into `timeline.jsonl`, each row the stored state beside the
observed liveness (the writer's process in the process table, the
heartbeat's age), so S5 can tell within one heartbeat whether the stored
state was truthful and S6 can read both builds on one elapsed grid.

    python3 status_timeline.py <campaign> <timeline.jsonl> --every <seconds> [--until-terminal]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

TERMINAL = ("done", "failed")


def installed_python() -> Path:
    """The interpreter of the installed package (a uv tool's, a venv's, a
    conda env's): the `kapso` script's shebang, so the config read below is
    the engine's own."""
    kapso = shutil.which("kapso")
    if not kapso:
        raise RuntimeError("kapso is not on PATH")
    first = Path(kapso).resolve().read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if not first.startswith("#!"):
        raise RuntimeError(f"{kapso} has no shebang to read the interpreter from")
    return Path(first[2:].split()[0])


def heartbeat_seconds() -> float:
    """The engine config's `budget.checkpoint_heartbeat_seconds`, read from
    the installed package: the one cadence the sampler and the grids use."""
    run = subprocess.run([str(installed_python()), "-c", "from kapso.core.config import load_config; print(load_config()['budget']['checkpoint_heartbeat_seconds'])"], capture_output=True, text=True, check=True)
    return float(run.stdout.strip())


SHOWN_STATES = ("DEAD", "STALLED", "WAITING", "FAILED", "DONE", "RUNNING", "STARTING")
_STATE_WORD = re.compile(r"\b(" + "|".join(SHOWN_STATES) + r")\b")


def shown_state(watch_text: str) -> Optional[str]:
    """The state `kapso watch` prints for the campaign — the verdict the
    user reads (DEAD when the writer is gone, STALLED on a stale heartbeat),
    not a re-derivation. The single-campaign screen keeps the stored word
    and appends the verdict as a marker ("RUNNING ♥ 289s ago   ⚠ DEAD (pid
    5554 is gone)"), so the marker wins over the word (s5-claude-5 read
    RUNNING through a writer the screen called dead). A detached resume
    whose child has not yet written its status prints "starting again
    (pid …); the status below is the earlier run's" above that earlier
    run's screen, DEAD marker and all: the product's verdict is the top
    line, and the sample reads it STARTING (rel-s5-codex read DEAD off
    the earlier run's marker a minute after a resume, and the run's
    truthfulness bar fell on the sampler's own misreading). None when
    nothing was printed."""
    if watch_text.lstrip().startswith("starting"):
        return "STARTING"
    if "⚠ DEAD" in watch_text:
        return "DEAD"
    if "⚠ STALLED" in watch_text:
        return "STALLED"
    match = _STATE_WORD.search(watch_text)
    return match.group(1) if match else None


def sample(campaign: Path) -> Dict[str, Any]:
    """One timeline row: the state `kapso watch` SHOWS (the product's own
    liveness verdict), beside the stored status it serialises. `watch` is
    the user's window; the stored fields are kept for provenance, the raw
    pid only as a secondary observation."""
    now = datetime.now(timezone.utc)
    row: Dict[str, Any] = {"ts": now.isoformat(timespec="seconds"), "shown": None, "stored": None, "pid_alive": None, "heartbeat_age_s": None}
    human = subprocess.run(["kapso", "watch", str(campaign)], capture_output=True, text=True)
    row["shown"] = shown_state(human.stdout)
    data = subprocess.run(["kapso", "watch", str(campaign), "--json"], capture_output=True, text=True)
    if data.returncode != 0 or not data.stdout.strip():
        return row
    status = json.loads(data.stdout)
    if not isinstance(status, dict) or "state" not in status:
        return row
    row["stored"] = {key: status.get(key) for key in ("state", "phase", "iteration", "best", "stopped_reason", "heartbeat_at", "pid", "error")}
    pid = status.get("pid")
    row["pid_alive"] = isinstance(pid, int) and Path(f"/proc/{pid}").exists()
    heartbeat = status.get("heartbeat_at")
    if heartbeat:
        row["heartbeat_age_s"] = round((now - datetime.fromisoformat(heartbeat)).total_seconds(), 1)
    return row


def run(campaign: Path, timeline: Path, every: float, until_terminal: bool) -> None:
    while True:
        began = time.monotonic()
        row = sample(campaign)
        with open(timeline, "a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
        if until_terminal and row["stored"] and row["stored"]["state"] in TERMINAL:
            return
        # The cadence counts from one sample's start to the next: two
        # `kapso watch` calls take seconds, and a sleep of the whole cadence
        # on top left no sample inside S5's one-heartbeat window.
        time.sleep(max(0.0, every - (time.monotonic() - began)))


def rows(timeline: Path):
    return [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign")
    parser.add_argument("timeline")
    parser.add_argument("--every", type=float, required=True, help="the cadence in seconds: the engine config's budget.checkpoint_heartbeat_seconds, read by the runner")
    parser.add_argument("--until-terminal", action="store_true")
    args = parser.parse_args(argv)
    run(Path(args.campaign), Path(args.timeline), args.every, args.until_terminal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
