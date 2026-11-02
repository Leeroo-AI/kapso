"""The trace fixtures, rebuilt from archived campaigns (docs/plans/
redesign/wp6-observability.md, change 11).

    python tests/fixtures/trace/build_fixture.py build <archive.tgz> <fixture-dir>
    python tests/fixtures/trace/build_fixture.py render <fixture-dir>

`build` keeps the campaign's records (the status, the launch record, the
checkpoint, the experiment history, the solution record, the serving
record, the pull log, each verification's verdict and evaluation output)
and every session record with its run entries, and drops the streams, the
diffs, the bank clones and the caches. The WP2 and WP3 archives predate
three things the engine writes now, which the build reconstructs from the
record itself, deterministically: the run entries (one per invocation, a
Claude Code stream split at its `result` events, a trailing slice without
one being a killed invocation; the times from the node's start and its
phases' durations), the pull log's `session` key (the event's time
against the same windows) and the current schemas (the solution record's
`session_failures`, the checkpoint's `experience`, a node's fields the
dataclass has gained, the launch record in the handle's words). A record
that carries its runs (the WP4 proof runs on) keeps them as written. The
checkpoint's `agent_output` is blanked (the trace does not read it; the
diff it does, for the delivered files).
Paths under the campaign's own directory are rebased to `/campaign`.

`render` writes `<fixture-dir>/trace.txt`, the golden `kapso trace`
output, with the fixture's absolute path written as `<campaign>`.
Byte-for-byte: the same archive gives the same fixture.
"""
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from kapso.execution.coding_agents.session_record import EXIT_ERROR, EXIT_SUCCESS, EXIT_TIMEOUT, append_run, relative_files, run_entry
from kapso.execution.coding_agents.transcripts import CLAUDE_HOST, digest_claude, stream_lines, truncated
from kapso.execution.inbox import LAUNCH_SCHEMA_VERSION
from kapso.execution.run_checkpoint import RunCheckpoint
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.trace import FEEDBACK, IDEATION, IMPLEMENTATION, VERIFICATION, failed_runs, read_trace
from kapso.handles import SOLUTION_RECORD, SOLUTION_SCHEMA, Solution
from kapso.harness.staging import SESSION_RECORD_FILE, SESSIONS_DIR, STREAM_FILE
from kapso.learning.serving_launch import PULL_LOG_FILE, SERVING_DIR, SERVING_RECORD_FILE

REBASED_ROOT = "/campaign"
GOLDEN = "trace.txt"
CAMPAIGN_PLACEHOLDER = "<campaign>"
# The record WP2 and WP3 wrote per session, before it was named for the
# session (wp6-observability.md, change 1).
OLD_RECORD_FILE = "harness.json"
EVENT_HEAD = "{"
RESULT_EVENT = "result"
# The events a host emits before it announces itself: its session hooks.
HOOK_EVENTS = ("hook_started", "hook_response")
PHASES = (IDEATION, IMPLEMENTATION, VERIFICATION, FEEDBACK)
SETUP_LABEL = "harness_setup"
SETUP_MODULE = "setup"
IMPLEMENTATION_FAILED = "Implementation failed: "
DEADLINE_ERROR = "Claude Code CLI exceeded its deadline and was terminated"
ENDED_WITHOUT_RESULT = "the invocation ended without a result event and the session was resumed (the archive predates the run entries)"
KEPT_RECORDS = ("status.json", "experiment_history.json")
VERIFICATION_KEPT = ("verdict.json", "evaluation_output.txt")


def build(archive: Path, fixture: Path) -> None:
    if fixture.exists():
        shutil.rmtree(fixture)
    with tempfile.TemporaryDirectory() as scratch:
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
        # The campaign directory, whatever the run called it: the one
        # holding the launch record.
        campaign = next(path.parent.parent for path in sorted(Path(scratch).glob("*/.kapso/launch.json")))
        _build_from(campaign, fixture)


def _build_from(campaign: Path, fixture: Path) -> None:
    kapso = campaign / ".kapso"
    out = fixture / ".kapso"
    out.mkdir(parents=True)
    launch = json.loads((kapso / "launch.json").read_text(encoding="utf-8"))
    root = launch["output_path"]
    rebase = lambda text: text.replace(root, REBASED_ROOT)
    for name in KEPT_RECORDS:
        (out / name).write_text(rebase((kapso / name).read_text(encoding="utf-8")), encoding="utf-8")
    # The launch record at the current schema (wp4-config.md): the
    # archives predate the record in the handle's words, without the
    # mode and the context arguments.
    renamed = {
        "max_iterations": "experiments", "time_budget_minutes": "minutes", "cost_budget": "usd",
        "finalization_reserve_minutes": "reserve_minutes", "coding_agent": "host",
    }
    launch_record = {
        renamed.get(key, key): value for key, value in launch.items()
        if key not in ("mode", "context", "additional_context")
    }
    launch_record["schema_version"] = LAUNCH_SCHEMA_VERSION
    (out / "launch.json").write_text(rebase(json.dumps(launch_record, indent=2, sort_keys=True) + "\n"), encoding="utf-8")
    status = json.loads((kapso / "status.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((kapso / "run_state.json").read_text(encoding="utf-8"))
    nodes = checkpoint["strategy_state"]["node_history"]
    windows = _windows(status, checkpoint, nodes)

    # The sessions: the record (or one made for a session the archive
    # recorded only as a stream) with its run entries, kept when the record
    # carries them and rebuilt from the stream when it predates them.
    sessions_in = kapso / SESSIONS_DIR.name
    session_dirs = {path.parent for name in (SESSION_RECORD_FILE, OLD_RECORD_FILE, STREAM_FILE) for path in sessions_in.rglob(name)}
    for session_in in sorted(session_dirs):
        label = session_in.relative_to(sessions_in).as_posix()
        record_file = next((session_in / name for name in (SESSION_RECORD_FILE, OLD_RECORD_FILE) if (session_in / name).is_file()), None)
        record = json.loads(record_file.read_text(encoding="utf-8")) if record_file else _record_for(label, root)
        written = record.get("runs") or []
        record = {key: value for key, value in record.items() if key != "runs"}
        # WP2's records named no stream; the path is where the staging puts it.
        record["stream"] = rebase(record.get("stream") or f"{root}/{SESSIONS_DIR.as_posix()}/{label}/{STREAM_FILE}")
        session_dir = record["session_dir"]
        record["session_dir"] = rebase(session_dir)
        record["runs"] = json.loads(rebase(json.dumps(written)))
        record_path = fixture / SESSIONS_DIR / label / SESSION_RECORD_FILE
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        stream = session_in / STREAM_FILE
        if written or not stream.is_file():
            continue
        node = next((n for n in nodes if n["branch_name"] == label), None)
        for entry in _run_entries(stream, label, windows[label], node, session_dir):
            append_run(record_path, entry)

    # The serving record and the pull log, each event stamped with the
    # session whose window holds its time.
    serving_in = kapso / SERVING_DIR.name
    if (serving_in / SERVING_RECORD_FILE).is_file():
        (out / SERVING_DIR.name).mkdir()
        (out / SERVING_DIR.name / SERVING_RECORD_FILE).write_text(
            rebase((serving_in / SERVING_RECORD_FILE).read_text(encoding="utf-8")), encoding="utf-8",
        )
        if (serving_in / PULL_LOG_FILE).is_file():
            stamped = []
            for line in (serving_in / PULL_LOG_FILE).read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                session = _session_at(datetime.fromisoformat(event["ts"]), windows)
                stamped.append(json.dumps({"ts": event["ts"], "tool": event["tool"], "session": session, **event}))
            (out / SERVING_DIR.name / PULL_LOG_FILE).write_text("\n".join(stamped) + "\n", encoding="utf-8")

    # Each verification's verdict and evaluation output.
    for directory in sorted((kapso / "verification").glob("*")) if (kapso / "verification").is_dir() else []:
        for name in VERIFICATION_KEPT:
            if (directory / name).is_file():
                (out / "verification" / directory.name).mkdir(parents=True, exist_ok=True)
                (out / "verification" / directory.name / name).write_text(
                    rebase((directory / name).read_text(encoding="utf-8")), encoding="utf-8",
                )

    # The checkpoint at the current schema, the bulk fields blanked.
    defaults = SearchNode(node_id=0).to_dict()
    checkpoint["schema_version"] = RunCheckpoint.SCHEMA_VERSION
    checkpoint.setdefault("experience", [])
    checkpoint = {"change_requests_filed": 0, **checkpoint}
    # The archives predate the node's `iteration`: one node per iteration
    # in every archived campaign, so the n-th node is iteration n's.
    checkpoint["strategy_state"]["node_history"] = [
        {**defaults, **node, "iteration": number, "agent_output": ""} for number, node in enumerate(nodes, start=1)
    ]
    RunCheckpoint.from_dict(checkpoint)
    (out / "run_state.json").write_text(rebase(json.dumps(checkpoint, indent=2) + "\n"), encoding="utf-8")

    # The solution record at the current schema, its failed sessions from
    # the records just rebuilt.
    solution = json.loads((kapso / "solution.json").read_text(encoding="utf-8"))
    solution = {key: value for key, value in solution.items() if key not in ("knowledge_index", "experience_head", "session_failures")}
    solution["schema_version"] = SOLUTION_SCHEMA
    solution["session_failures"] = failed_runs(fixture)
    (fixture / SOLUTION_RECORD).write_text(rebase(json.dumps(solution, indent=2) + "\n"), encoding="utf-8")
    Solution.open(fixture)


def _record_for(label: str, root: str) -> dict:
    """A record for a session the archive kept only as a stream (a build
    without a harness wrote none): nothing staged, the module from the
    label's grammar."""
    head = label.partition("/")[0]
    module = {IDEATION: IDEATION, VERIFICATION: VERIFICATION, FEEDBACK: FEEDBACK, SETUP_LABEL: SETUP_MODULE}.get(head, IMPLEMENTATION)
    # An implementation session runs in its own directory under the
    # campaign's `sessions/`; the others at the campaign root.
    session_dir = f"{root}/sessions/{label}" if module == IMPLEMENTATION else root
    return {
        "module": module, "host": CLAUDE_HOST, "harness": None, "session_dir": session_dir,
        "pages": [], "skills": [], "mcp_servers": [], "agents": [], "hooks": [], "prepared": [],
        "stream": f"{root}/.kapso/sessions/{label}/{STREAM_FILE}",
    }


def _windows(status: dict, checkpoint: dict, nodes: list) -> dict:
    """When each session ran: the node's start and its phases' durations,
    in the order the phases run; the harness setup from the campaign's
    start."""
    windows = {}
    setup = checkpoint.get("harness_setup") or {}
    if setup.get("duration_seconds") is not None:
        start = datetime.fromisoformat(status["started_at"])
        windows[SETUP_LABEL] = (start, start + timedelta(seconds=setup["duration_seconds"]))
    for number, node in enumerate(nodes, start=1):
        clock = datetime.fromisoformat(node["started_at"])
        for phase in PHASES:
            telemetry = node["phase_telemetry"].get(phase)
            if telemetry is None:
                continue
            end = clock + timedelta(seconds=telemetry["duration_seconds"])
            label = {
                IDEATION: f"{IDEATION}/iter{number}",
                IMPLEMENTATION: node["branch_name"],
                VERIFICATION: f"{VERIFICATION}/{node['branch_name']}",
                FEEDBACK: f"{FEEDBACK}/{node['branch_name']}",
            }[phase]
            windows[label] = (clock, end)
            clock = end
    return _Windows(windows)


class _Windows(dict):
    """Lookup by a session's label: an ideation member's label
    (`ideation/iter1/claude_code`) falls under its iteration's window."""

    def __missing__(self, label: str):
        head, _, rest = label.partition("/")
        if head == IDEATION:
            iteration = rest.partition("/")[0]
            return self[f"{IDEATION}/{iteration}"]
        raise KeyError(label)


def _session_at(moment: datetime, windows: dict) -> str:
    for label, (start, end) in windows.items():
        if start <= moment <= end:
            return label
    raise ValueError(f"no session window holds {moment.isoformat()}")


def _run_entries(stream: Path, label: str, window: tuple, node: dict | None, session_dir: str) -> list:
    """One entry per invocation in a Claude Code stream: an invocation
    begins at its `system/init` event (the hook events just before it
    belong to it), and holds every turn until the next init; one without
    a `result` event was killed at its deadline. The files changed are
    the edit tools' inputs, relative to the session directory as the
    archive recorded it."""
    lines = stream_lines(stream.read_text(encoding="utf-8"))
    parsed = []
    for line in lines:
        stripped = line.rstrip("\n")
        event = json.loads(stripped) if stripped.startswith(EVENT_HEAD) and not truncated(line) else {}
        parsed.append((line, event))
    groups: list = []          # one list of (line, event) per invocation
    current: list = []
    for line, event in parsed:
        if event.get("type") == "system" and event.get("subtype") == "init":
            carried = []
            while current and current[-1][1].get("subtype") in HOOK_EVENTS:
                carried.insert(0, current.pop())
            if current:
                groups.append(current)
            current = carried + [(line, event)]
            continue
        current.append((line, event))
    if any(line.strip() for line, _ in current):
        groups.append(current)
    slices = []
    for group in groups:
        results = [event for _, event in group if event.get("type") == RESULT_EVENT]
        slices.append(([line for line, _ in group], results[-1] if results else None))
    entries = []
    clock = window[0]
    session_id = model = None
    for number, (slice_lines, result) in enumerate(slices, start=1):
        digest = digest_claude(slice_lines)
        # A resumed turn has no init event; the result event names the
        # session, and a killed slice continues the one before it.
        session_id = digest["cli_session_id"] or (result or {}).get("session_id") or session_id
        model = digest["model"] or model
        if result is None and number < len(slices):
            # The process ended without a result event and the adapter
            # resumed the session (a turn ending with a delegate still
            # running); how the adapter judged it was not recorded.
            exit, error = EXIT_ERROR, ENDED_WITHOUT_RESULT
        elif result is None:
            exit, error = EXIT_TIMEOUT, _deadline_error(node)
        elif result.get("is_error"):
            exit, error = EXIT_ERROR, (digest["errors"][-1] if digest["errors"] else str(result.get("result")))
        else:
            exit, error = EXIT_SUCCESS, None
        # The host's own figure; a lone invocation killed at its deadline
        # ran its phase's window; one resumed after ending without a
        # result has no figure of its own.
        duration = (
            digest["duration_ms"] / 1000.0 if digest["duration_ms"] is not None
            else (window[1] - window[0]).total_seconds() if len(slices) == 1 else None
        )
        ended = clock + timedelta(seconds=duration or 0.0)
        entries.append(run_entry(
            digest, model=digest["model"] or model, cli_session_id=session_id, exit=exit, error=error,
            files_changed=relative_files(digest["files"], session_dir), started=clock.isoformat(timespec="seconds"),
            ended=ended.isoformat(timespec="seconds") if duration is not None else None,
            duration_seconds=round(duration, 3) if duration is not None else None,
        ))
        clock = ended
    return entries


def _deadline_error(node: dict | None) -> str:
    """The adapter's words when the deadline killed the invocation: the
    node's evaluation output starts with them for an implementation."""
    if node is not None and node["evaluation_output"].startswith(IMPLEMENTATION_FAILED):
        return node["evaluation_output"].splitlines()[0][len(IMPLEMENTATION_FAILED):]
    return DEADLINE_ERROR


def render(fixture: Path) -> None:
    text = read_trace(fixture).explain().replace(str(fixture.resolve()), CAMPAIGN_PLACEHOLDER)
    (fixture / GOLDEN).write_text(text + "\n", encoding="utf-8")


def main(argv: list) -> None:
    if argv[:1] == ["build"] and len(argv) == 3:
        build(Path(argv[1]), Path(argv[2]))
    elif argv[:1] == ["render"] and len(argv) == 2:
        render(Path(argv[1]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
