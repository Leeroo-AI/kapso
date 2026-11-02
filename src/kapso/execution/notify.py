"""Tell the person when a build stops, through the `notify:` block's command
or webhook. A failed delivery is reported, never raised: the build's own
outcome is already published."""

import concurrent.futures
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from kapso.execution.observability import OperationStatusView

STOP_EVENTS = {
    "goal_achieved": "completed",
    "waiting_for_user": "waiting_for_user",
    "budget_exhausted": "budget_stop",
    "cost_budget_exhausted": "budget_stop",
    "max_iterations": "budget_stop",
}


def build_event(status: Mapping[str, Any]) -> Optional[str]:
    if status["state"] == "failed":
        return "failed"
    if status["state"] != "done":
        return None
    if status["stopped_reason"] not in STOP_EVENTS:
        raise ValueError(f"no notify event for stopped_reason {status['stopped_reason']!r}")
    return STOP_EVENTS[status["stopped_reason"]]


def run_command(command: Sequence[str], body: str, timeout: float) -> Optional[str]:
    program = shutil.which(os.path.expanduser(command[0]))
    if program is None:
        return f"{command[0]} is not an executable on PATH"
    # Output to a file, not a pipe, so a child left running cannot hold the wait open.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [program, *command[1:]], stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        thread = threading.Thread(target=process.communicate, args=(body,), daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            os.killpg(process.pid, signal.SIGKILL)
            thread.join()
            return f"{command[0]} did not finish within {timeout:g} s"
        if process.returncode != 0:
            output.seek(0)
            return f"{command[0]} exited {process.returncode}: {output.read().strip()}"
    return None


def post_webhook(url: str, body: str, timeout: float) -> Optional[str]:
    request = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: urllib.request.urlopen(request, timeout=timeout).close())
    failure = future.exception()
    if failure is None:
        return None
    # The host alone: a webhook's path often carries its token.
    parts = urlsplit(url)
    return f"POST {parts.scheme}://{parts.netloc}: {type(failure).__name__}: {failure}"


def notify_build_end(block: Mapping[str, Any], campaign: Path, objective: str, since: datetime) -> None:
    """Send the event the build's terminal status announces, from the finally
    that published it. Ctrl-C is the person's own stop, and a status older
    than `since` is an earlier slice's, announced already."""
    if not block["events"] or isinstance(sys.exc_info()[1], KeyboardInterrupt):
        return
    campaign = Path(campaign).resolve()
    path = campaign / ".kapso" / "status.json"
    if not path.is_file():
        return
    status = OperationStatusView(path).data
    if datetime.fromisoformat(status["started_at"]) < since:
        return
    event = build_event(status)
    if event not in block["events"]:
        return
    body = json.dumps({
        "event": event,
        "campaign": str(campaign),
        "objective": objective,
        **{key: status.get(key) for key in ("stopped_reason", "stop_detail", "error", "best", "budget")},
        "started_at": status["started_at"],
        "finished_at": status["finished_at"],
        "requests": [
            {**request, "reply": f'kapso inbox reply {shlex.quote(str(campaign))} {request["id"]} "…"'}
            for request in status.get("requests", [])
        ],
    }, indent=2)
    for target, send in (("command", run_command), ("webhook", post_webhook)):
        if block[target] is None:
            continue
        failure = send(block[target], body, float(block["timeout_seconds"]))
        if failure is None:
            print(f"notify: {event} sent to the {target}")
        else:
            print(f"notify: {event} not delivered to the {target}: {failure}", file=sys.stderr)
