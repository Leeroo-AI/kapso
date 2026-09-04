"""Live mechanics of the inbox (plan §3.2, L1–L10): real sessions on the
pinned CLIs against the fixture in inbox_live_support. Skipped without
--run-live; each test spends real subscription quota and minutes.

Assertions are on files and transcripts, never on model prose: the
request record, the checkpoint, the CLI's own transcript (the follow-up
must land after the tool result, in the same session), the branch, and
the sum the continued session computes.
"""

from pathlib import Path

import git
import os
import signal
import subprocess
import time

import pytest

from inbox_live_support import (
    EXTRA_FILE,
    EXTRA_NUMBERS,
    FOLLOW_UP_MARK,
    KEY_VAR,
    TOTAL,
    build_fixture,
    checkpoint,
    claude_transcript,
    codex_rollout,
    drop_extra_file,
    init_events,
    nodes,
    result_events,
    run_evolve,
    run_reply,
    run_resume,
    run_sum,
    set_key,
    start_reply,
    status,
    stream_events,
    transcript_positions,
)
from kapso.execution.inbox import inbox_path, load_requests

STOP_MARK = "Your session is being stopped now"
pytestmark = pytest.mark.live


def _transcript(cli: str, session_id: str) -> Path:
    path = claude_transcript(session_id) if cli == "claude" else codex_rollout(session_id)
    assert path is not None, f"no {cli} transcript for session {session_id}"
    return path


def _paused(root: Path, cli: str):
    """Launch the campaign and check the pause. Returns (campaign, node, request)."""
    campaign = root / "campaign"
    evolve = run_evolve(root, log=root / "evolve.log")
    assert evolve.returncode == 0, evolve.stdout[-3000:]
    assert "WAITING ON YOU" in evolve.stdout or "stopped_reason=waiting_for_user" in evolve.stdout

    requests = list(load_requests(inbox_path(campaign)).values())
    assert len(requests) == 1, [r.key for r in requests]
    request = requests[0]
    assert request.open and KEY_VAR in request.key and KEY_VAR in request.tried
    assert request.fix and request.next_steps

    saved = checkpoint(campaign)
    assert saved["last_stop"] == "waiting_for_user" and saved["completed_iterations"] == 0
    node = nodes(campaign)[0]
    assert node["suspended"] is True and node["request_ids"] == [request.id]
    assert node["cli_session_id"]
    assert status(campaign)["stopped_reason"] == "waiting_for_user"

    repo = git.Repo(campaign)
    assert node["branch_name"] in {head.name for head in repo.heads}
    assert repo.commit(node["branch_name"]).hexsha != repo.commit("main").hexsha, "the stopped session's tree was not committed"
    return campaign, node, request


def _no_secret_on_disk(root: Path, key: str) -> None:
    """§3.4: the key value is in nothing Kapso wrote, although the .env
    holds it and the continued session used it. (Any 64-hex token would be
    the wrong test: the checkpoint carries sha256 fingerprints. The
    session's own streamed transcript is the CLI's record and out of
    scope: a coder that cats its .env puts the value there.)"""
    campaign = root / "campaign"
    for name in ("inbox.jsonl", "status.json", "run_state.json", "launch.json"):
        assert key not in (campaign / ".kapso" / name).read_text(), name


def _continued_cleanly(campaign: Path, node: dict, cli: str) -> None:
    """L4 and L5 on the branch's stream: the first turn ended cleanly
    before the continuation began, and the continuation mounted the
    same servers and tools (Claude) or the same thread (Codex)."""
    events = stream_events(campaign, node["branch_name"])
    inits = init_events(events)
    assert len(inits) == 2, f"expected the first session and one continuation, saw {len(inits)} starts"
    first_end = next(i for i, e in enumerate(events) if e in result_events(events))
    second_start = next(i for i, e in enumerate(events) if e is inits[1])
    assert first_end < second_start, "the first turn did not end cleanly before the continuation"
    if cli == "claude":
        servers = [sorted(s["name"] for s in e.get("mcp_servers", [])) for e in inits]
        assert servers[0] == servers[1] and servers[0], servers
        assert sorted(inits[0].get("tools", [])) == sorted(inits[1].get("tools", []))
        assert inits[0].get("model") == inits[1].get("model")
    else:
        assert inits[0]["thread_id"] == inits[1]["thread_id"] == node["cli_session_id"]


@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_stop_and_resume_continues_the_same_session(tmp_path, cli):
    """L1 / L2: one request, the transcript exists, the reply continues the
    very session (same id, follow-up after the tool result), the branch
    carries the pre-stop commit, and the continued session prints the sum."""
    root = tmp_path / cli
    key = build_fixture(root, cli)["key"]
    campaign, node, request = _paused(root, cli)
    transcript = _transcript(cli, node["cli_session_id"])
    before = transcript_positions(transcript, [STOP_MARK, FOLLOW_UP_MARK])
    assert before[STOP_MARK] is not None and before[FOLLOW_UP_MARK] is None

    set_key(root, key)
    reply = run_reply(root, request.id, "added to .env", log=root / "reply.log")
    assert reply.returncode == 0, reply.stdout[-3000:]
    assert f"#{request.id} answered" in reply.stdout and "COMPLETED" in reply.stdout

    after = nodes(campaign)[0]
    assert after["cli_session_id"] == node["cli_session_id"]
    assert after["suspended"] is False and after["request_ids"] == []
    assert checkpoint(campaign)["completed_iterations"] == 1
    assert load_requests(inbox_path(campaign))[request.id].state == "continued"

    positions = transcript_positions(transcript, [STOP_MARK, FOLLOW_UP_MARK])
    assert positions[FOLLOW_UP_MARK] is not None, "the follow-up never reached the session"
    assert positions[STOP_MARK] < positions[FOLLOW_UP_MARK]
    _continued_cleanly(campaign, node, cli)
    _no_secret_on_disk(root, key)
    assert run_sum(campaign, node["branch_name"], key) == str(TOTAL)


def test_grace_then_sigterm_still_continues_the_session(tmp_path):
    """L3: the goal orders the coder to keep working after the call; the
    adapter ends the session after the shortened grace, and the reply
    still continues the interrupted turn with the follow-up after the
    tool result."""
    root = tmp_path / "keep"
    key = build_fixture(root, "claude", variant="keep_working")["key"]
    campaign, node, request = _paused(root, "claude")
    events = stream_events(campaign, node["branch_name"])
    assert not result_events(events), "the session ended on its own; the grace kill was not exercised"
    transcript = _transcript("claude", node["cli_session_id"])

    set_key(root, key)
    reply = run_reply(root, request.id, "added to .env", log=root / "reply.log")
    assert reply.returncode == 0 and "COMPLETED" in reply.stdout, reply.stdout[-3000:]
    positions = transcript_positions(transcript, [STOP_MARK, FOLLOW_UP_MARK])
    assert positions[STOP_MARK] is not None and positions[FOLLOW_UP_MARK] is not None
    assert positions[STOP_MARK] < positions[FOLLOW_UP_MARK]
    after = nodes(campaign)[0]
    assert after["cli_session_id"] == node["cli_session_id"] and after["suspended"] is False
    assert run_sum(campaign, node["branch_name"], key) == str(TOTAL)


def test_two_needs_two_replies(tmp_path):
    """L6: one call carries both needs; the first reply waits, the second
    resumes, and the continued session uses both."""
    root = tmp_path / "two"
    key = build_fixture(root, "claude", variant="two_needs")["key"]
    campaign = root / "campaign"
    evolve = run_evolve(root, log=root / "evolve.log")
    assert evolve.returncode == 0 and "WAITING ON YOU" in evolve.stdout, evolve.stdout[-3000:]
    requests = sorted(load_requests(inbox_path(campaign)).values(), key=lambda r: r.id)
    assert len(requests) == 2 and all(r.open for r in requests), [r.key for r in requests]
    assert len({r.session for r in requests}) == 1, "the two needs were not one call"
    key_request = next(r for r in requests if KEY_VAR in r.key)
    file_request = next(r for r in requests if r.id != key_request.id)
    assert "extra" in file_request.key
    node = nodes(campaign)[0]
    assert node["suspended"] is True and sorted(node["request_ids"]) == [r.id for r in requests]

    set_key(root, key)
    first = run_reply(root, key_request.id, "added to .env", log=root / "reply1.log")
    assert first.returncode == 0 and "still open, so node" in first.stdout, first.stdout[-2000:]
    assert nodes(campaign)[0]["suspended"] is True
    drop_extra_file(root)
    second = run_reply(root, file_request.id, f"dropped at {EXTRA_FILE}", log=root / "reply2.log")
    assert second.returncode == 0 and "COMPLETED" in second.stdout, second.stdout[-3000:]
    after = nodes(campaign)[0]
    assert after["suspended"] is False and after["cli_session_id"] == node["cli_session_id"]
    assert run_sum(campaign, node["branch_name"], key, extra=True) == str(TOTAL + sum(EXTRA_NUMBERS))


def test_wrong_value_makes_the_coder_ask_again(tmp_path):
    """L7: a reply with a wrong key; the continued session verifies, asks
    again with the previous reply attached, and the right key succeeds."""
    root = tmp_path / "wrong"
    key = build_fixture(root, "claude")["key"]
    campaign, node, request = _paused(root, "claude")

    set_key(root, "00" * 32)
    reply = run_reply(root, request.id, "added to .env", log=root / "reply1.log")
    assert reply.returncode == 0 and "WAITING ON YOU" in reply.stdout, reply.stdout[-3000:]
    requests = sorted(load_requests(inbox_path(campaign)).values(), key=lambda r: r.id)
    assert len(requests) == 2, [r.key for r in requests]
    again = requests[1]
    assert again.open and KEY_VAR in again.key and again.previous_reply == "added to .env"
    assert "your previous reply was" in reply.stdout
    waiting = nodes(campaign)[0]
    assert waiting["suspended"] is True and waiting["request_ids"] == [again.id]
    assert waiting["cli_session_id"] == node["cli_session_id"]

    set_key(root, key)
    final = run_reply(root, again.id, "wrong paste before; the right key is in .env now", log=root / "reply2.log")
    assert final.returncode == 0 and "COMPLETED" in final.stdout, final.stdout[-3000:]
    assert nodes(campaign)[0]["suspended"] is False
    assert run_sum(campaign, node["branch_name"], key) == str(TOTAL)


def test_transcript_gone_fails_loud_and_keeps_the_node(tmp_path):
    """L8: the CLI cannot resume a deleted transcript; the reply fails
    loud with the session named, the node stays suspended, and nothing
    else runs (the request is recorded as continued; a later resume
    retries the continuation)."""
    root = tmp_path / "gone"
    key = build_fixture(root, "claude")["key"]
    campaign, node, request = _paused(root, "claude")
    _transcript("claude", node["cli_session_id"]).unlink()

    set_key(root, key)
    reply = run_reply(root, request.id, "added to .env", log=root / "reply.log")
    assert reply.returncode != 0
    assert "could not resume CLI session" in reply.stdout and node["cli_session_id"] in reply.stdout
    assert nodes(campaign)[0]["suspended"] is True
    assert checkpoint(campaign)["completed_iterations"] == 0
    assert load_requests(inbox_path(campaign))[request.id].state == "continued"


def test_killed_mid_continuation_resumes_the_same_session(tmp_path):
    """L9: Kapso dies while the continued session runs; the checkpoint
    still marks the node suspended, and `kapso evolve --resume` continues
    the same session id to completion."""
    root = tmp_path / "killed"
    key = build_fixture(root, "claude")["key"]
    campaign, node, request = _paused(root, "claude")
    transcript = _transcript("claude", node["cli_session_id"])

    set_key(root, key)
    reply = start_reply(root, request.id, "added to .env", root / "reply.log")
    deadline = time.monotonic() + 600
    while transcript_positions(transcript, [FOLLOW_UP_MARK])[FOLLOW_UP_MARK] is None:
        assert reply.poll() is None, "the continuation ended before it could be killed"
        assert time.monotonic() < deadline, "the follow-up never reached the session"
        time.sleep(2)
    time.sleep(15)
    children = subprocess.run(["pgrep", "-P", str(reply.pid)], text=True, capture_output=True).stdout.split()
    for child in children:
        os.killpg(int(child), signal.SIGKILL)
    os.killpg(reply.pid, signal.SIGKILL)
    reply.wait(timeout=30)
    assert nodes(campaign)[0]["suspended"] is True
    assert load_requests(inbox_path(campaign))[request.id].state == "continued"

    resumed = run_resume(root, log=root / "resume.log")
    assert resumed.returncode == 0 and "COMPLETED" in resumed.stdout, resumed.stdout[-3000:]
    after = nodes(campaign)[0]
    assert after["cli_session_id"] == node["cli_session_id"] and after["suspended"] is False
    assert checkpoint(campaign)["completed_iterations"] == 1
    assert run_sum(campaign, node["branch_name"], key) == str(TOTAL)
