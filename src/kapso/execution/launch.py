"""A detached launch: this interpreter's `kapso.cli` run again in a session
of its own, its output in a log file, the child's pid returned at once.

A session of its own (`start_new_session`) is what outlives the shell call
that launched it: a coding agent's tool call ends a plain background child
when it returns, and the campaign with it (Codex, verified 2026-09-07);
every host and the skill used to reinvent the nohup/setsid dance for this
one flag, and every engineer of the release gate's S1 reinvented it in
Python because the API offered none. The CLI's `--detach` and the API's
`detach=True` both come here. A build writes a receipt beside its log,
which `kapso watch` reads before the status exists; a learn or an ingest
writes none (their status is born within seconds under the learning status
directory, and nothing reads a receipt there).
"""

import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


def detach(argv: Sequence[str], *, log_path: Path, receipt_path: Optional[Path]) -> int:
    """Start `python -m kapso.cli <argv>` in a session of its own, appending
    its output to `log_path`, write the receipt when a path is given, and
    return the child's pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The receipt lies beside the campaign: its directory exists before the
    # child does, so a launch under a parent not yet made still gets one.
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # The child is meant to outlive this process, so the Popen object is let
    # go while it runs; Python would otherwise print "subprocess N is still
    # running" at the end of a successful launch.
    with warnings.catch_warnings(), open(log_path, "a", encoding="utf-8") as log:
        warnings.simplefilter("ignore", ResourceWarning)
        child = subprocess.Popen(
            [sys.executable, "-m", "kapso.cli", *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
        )
        pid = child.pid
        del child
    if receipt_path is not None:
        receipt_path.write_text(json.dumps({
            "pid": pid,
            "log": str(log_path),
            "argv": list(argv),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n", encoding="utf-8")
    return pid
