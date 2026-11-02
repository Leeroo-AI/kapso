"""The campaign inbox record (docs/research/evolve-hub-design.md v4 §4.1).

What must hold: a request's state is the fold of its events (open →
answered → continued); ids are campaign-local and contiguous; a post on
an open key joins it; a post on an answered key opens a fresh request
carrying the previous reply; a reply must name an open request; a
continue must follow a reply; concurrent writers never tear the file; a
malformed line raises; a missing file is "no requests"; the launch record
round-trips; the registry skips deleted campaigns and raises on a
malformed line. A write that never finished is not corruption: the event
it was for did not happen, the file stays readable, and the next writer
removes the fragment.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kapso.execution.inbox import (
    LAUNCH_SCHEMA_VERSION,
    all_answered,
    awaiting_resume,
    continue_answered,
    file_requests,
    inbox_path,
    launch_mismatches,
    list_registered_campaigns,
    load_requests,
    open_requests,
    read_launch_record,
    record_continued,
    record_reply,
    register_campaign,
    render_stop_text,
    resume_arguments,
    write_launch_record,
)

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError at the embedding step",
    "tried": "OPENAI_KEY unset too; no .env in the repo; README says export it",
    "fix": "add OPENAI_API_KEY=sk-... to /home/me/churn/.env",
    "next_steps": "embed the candidate texts, re-rank, run the evaluation",
}
DATA_ENTRY = {**ENTRY, "key": "data/transactions-2019.csv", "hit": "FileNotFoundError"}


def test_request_lifecycle_open_answered_continued(tmp_path):
    path = inbox_path(tmp_path)
    ids = file_requests(path, node=3, session="s-1", entries=[ENTRY])
    assert ids == [(1, None)]
    request = load_requests(path)[1]
    assert (request.state, request.open, request.node, request.session) == ("open", True, 3, "s-1")
    assert request.key == ENTRY["key"] and request.next_steps == ENTRY["next_steps"]

    answered = record_reply(path, 1, "added the key")
    assert answered.state == "answered" and answered.reply == "added the key"
    assert open_requests(load_requests(path)) == []
    assert all_answered(load_requests(path), [1])

    record_continued(path, [1], node=3, session="s-1")
    assert load_requests(path)[1].state == "continued"


def test_ids_are_contiguous_and_one_call_files_several(tmp_path):
    path = inbox_path(tmp_path)
    assert file_requests(path, node=3, session="s", entries=[ENTRY, DATA_ENTRY]) == [(1, None), (2, None)]
    requests = load_requests(path)
    assert sorted(requests) == [1, 2]
    assert not all_answered(requests, [1, 2])
    record_reply(path, 1, "")
    assert not all_answered(load_requests(path), [1, 2])
    record_reply(path, 2, "dropped the file")
    assert all_answered(load_requests(path), [1, 2])
    assert not all_answered(load_requests(path), [])


def test_open_key_is_joined_and_answered_key_carries_previous_reply(tmp_path):
    path = inbox_path(tmp_path)
    file_requests(path, node=3, session="s", entries=[ENTRY])
    # The same key while #1 is open: joined, nothing written.
    assert file_requests(path, node=3, session="s", entries=[ENTRY]) == [(1, None)]
    assert len(load_requests(path)) == 1
    record_reply(path, 1, "added the key")
    # After a reply the same key opens a fresh request with the loop visible.
    assert file_requests(path, node=3, session="s", entries=[ENTRY]) == [(2, "added the key")]
    assert load_requests(path)[2].previous_reply == "added the key"


def test_reply_and_continue_are_validated(tmp_path):
    path = inbox_path(tmp_path)
    with pytest.raises(ValueError, match="no request #7"):
        record_reply(path, 7, "x")
    file_requests(path, node=0, session="s", entries=[ENTRY])
    with pytest.raises(ValueError, match="still open"):
        record_continued(path, [1], node=0, session="s")
    record_reply(path, 1, "done")
    record_continued(path, [1], node=0, session="s")
    with pytest.raises(ValueError, match="already answered"):
        record_reply(path, 1, "again")


def test_a_later_reply_replaces_the_answer_until_a_session_reads_it(tmp_path):
    path = inbox_path(tmp_path)
    file_requests(path, node=0, session="s", entries=[ENTRY])
    record_reply(path, 1, "added the key")
    assert record_reply(path, 1, "use the staging key").reply == "use the staging key"
    assert load_requests(path)[1].reply == "use the staging key"
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "requested", "replied", "replied",
    ]


def test_awaiting_resume_follows_the_checkpoint_not_the_events(tmp_path):
    path = inbox_path(tmp_path)
    file_requests(path, node=3, session="s", entries=[ENTRY, DATA_ENTRY])
    suspended = {"suspended": True, "request_ids": [1, 2]}

    assert awaiting_resume(load_requests(path), [suspended]) == []
    record_reply(path, 1, "added the key")
    assert awaiting_resume(load_requests(path), [suspended]) == []
    record_reply(path, 2, "dropped the file")
    assert [r.id for r in awaiting_resume(load_requests(path), [suspended])] == [1, 2]
    # A kill after the session started: `continued` is on file, the node is
    # still suspended, and the resume is still owed.
    record_continued(path, [1, 2], node=3, session="s")
    assert [r.id for r in awaiting_resume(load_requests(path), [suspended])] == [1, 2]
    # The node moved on, or a new request replaced these.
    assert awaiting_resume(load_requests(path), [{"suspended": False, "request_ids": []}]) == []


def test_entries_are_validated_before_anything_is_written(tmp_path):
    path = inbox_path(tmp_path)
    with pytest.raises(ValueError, match="non-empty list"):
        file_requests(path, node=0, session="s", entries=[])
    with pytest.raises(ValueError, match="missing tried, fix"):
        file_requests(path, node=0, session="s", entries=[{"key": "k", "hit": "h", "next_steps": "n", "tried": " "}])
    assert not path.exists()


def test_concurrent_writers_never_tear_the_file(tmp_path):
    path = inbox_path(tmp_path)
    file_requests(path, node=0, session="s", entries=[ENTRY])

    def reply_and_ask(index: int) -> None:
        file_requests(path, node=index, session=f"s-{index}", entries=[{**ENTRY, "key": f"k-{index}"}])

    threads = [threading.Thread(target=reply_and_ask, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    requests = load_requests(path)
    assert sorted(requests) == list(range(1, 42))
    assert len(path.read_text().splitlines()) == 41


def test_an_unfinished_write_leaves_the_request_answerable(tmp_path):
    """A real short write, the way a full disk produces one: the writer is
    allowed the first bytes of the event and then fails."""
    path = inbox_path(tmp_path)
    file_requests(path, node=3, session="s", entries=[ENTRY])
    size = path.stat().st_size
    writer = (
        "import resource, sys\n"
        "from kapso.execution.inbox import record_reply\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({size + 40}, resource.RLIM_INFINITY))\n"
        f"record_reply({str(path)!r}, 1, 'added the key')\n"
    )
    assert subprocess.run([sys.executable, "-c", writer], capture_output=True).returncode != 0
    assert path.stat().st_size == size + 40

    assert load_requests(path)[1].open
    assert record_reply(path, 1, "added the key").state == "answered"
    assert load_requests(path)[1].reply == "added the key"
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_the_session_can_ask_after_an_unfinished_write(tmp_path):
    path = inbox_path(tmp_path)
    file_requests(path, node=3, session="s", entries=[ENTRY])
    with open(path, "a") as handle:
        handle.write('{"ts": "now", "event": "repl')
    assert file_requests(path, node=3, session="s", entries=[DATA_ENTRY]) == [(2, None)]
    assert sorted(load_requests(path)) == [1, 2]


def test_a_failed_launch_record_write_leaves_the_old_record(tmp_path, monkeypatch):
    write_launch_record(tmp_path, {"mode": "GENERIC"})

    def no_replace(source, target):
        raise OSError("killed before the replace")

    monkeypatch.setattr(os, "replace", no_replace)
    with pytest.raises(OSError):
        write_launch_record(tmp_path, {"mode": "MINIMAL"})
    assert read_launch_record(tmp_path)["mode"] == "GENERIC"


def test_malformed_line_raises_and_missing_file_is_empty(tmp_path):
    path = inbox_path(tmp_path)
    assert load_requests(path) == {}
    path.parent.mkdir(parents=True)
    path.write_text('{"event": "requested", "id": 1, "node": 0, "session": "s", "key": "k", "hit": "h", "tried": "t", "fix": "f", "next_steps": "n", "ts": "now"}\nnot json\n')
    with pytest.raises(json.JSONDecodeError):
        load_requests(path)
    path.write_text('{"event": "replied", "id": 1, "note": "x", "ts": "now"}\n')
    with pytest.raises(ValueError, match="unknown request #1"):
        load_requests(path)
    path.write_text('{"event": "bogus", "id": 1, "ts": "now"}\n')
    with pytest.raises(ValueError, match="unknown inbox event"):
        load_requests(path)


def test_stop_text_names_ids_and_the_previous_reply():
    text = render_stop_text([(1, None), (2, "added the key")], ["env:A", "env:B"])
    assert text.startswith("Recorded as requests #1, #2.")
    assert "do nothing further" in text
    assert "env:B was requested before" in text and "'added the key'" in text
    assert "env:A was requested" not in text


def test_inbox_path_is_absolute_whatever_the_workspace_argument(tmp_path, monkeypatch):
    """The path crosses into the gate server, whose cwd is the session
    folder; a relative campaign dir lost the request (live L1 run 2)."""
    monkeypatch.chdir(tmp_path)
    assert inbox_path("campaign") == tmp_path.resolve() / "campaign" / ".kapso" / "inbox.jsonl"
    assert inbox_path("campaign").is_absolute()


def test_a_credential_shaped_reply_is_refused_by_the_record_itself(tmp_path):
    """The guard holds for every caller, not only the CLI's --yes check."""
    path = tmp_path / "inbox.jsonl"
    file_requests(path, node=1, session="s", entries=[ENTRY])
    token = "sk-abcdefghijklmnopqrstuvwxyz0123"
    with pytest.raises(ValueError, match="looks like a credential"):
        record_reply(path, 1, f"added {token}")
    assert load_requests(path)[1].open
    assert record_reply(path, 1, token, allow_secret=True).reply == token


def test_launch_record_round_trip_and_missing_default(tmp_path):
    assert read_launch_record(tmp_path) is None
    write_launch_record(tmp_path, {"output_path": str(tmp_path), "experiments": 4})
    record = read_launch_record(tmp_path)
    assert record["schema_version"] == LAUNCH_SCHEMA_VERSION
    assert record["experiments"] == 4
    Path(tmp_path, ".kapso", "launch.json").write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="launch record"):
        read_launch_record(tmp_path)


def test_registry_lists_existing_campaigns_newest_first_and_raises_on_junk(tmp_path):
    registry = tmp_path / "registry.jsonl"
    assert list_registered_campaigns(registry) == []
    alive = tmp_path / "alive"
    alive.mkdir()
    gone = tmp_path / "gone"
    gone.mkdir()
    register_campaign(registry, gone, "first goal\nsecond line")
    register_campaign(registry, alive, "second goal")
    gone.rmdir()
    listed = list_registered_campaigns(registry)
    assert [entry["goal"] for entry in listed] == ["second goal"]
    assert listed[0]["path"] == str(alive.resolve())
    registry.write_text(registry.read_text() + '{"no": "path"}\n')
    with pytest.raises(ValueError, match="registry line"):
        list_registered_campaigns(registry)


def test_registry_lists_a_relaunched_campaign_once(tmp_path):
    """Deleting a campaign and launching again at the same path adds a
    second registry line; `kapso inbox` must still show the campaign once,
    under its newest launch."""
    registry = tmp_path / "registry.jsonl"
    campaign = tmp_path / "campaign"
    other = tmp_path / "other"
    campaign.mkdir()
    other.mkdir()
    register_campaign(registry, campaign, "first try")
    register_campaign(registry, other, "other goal")
    register_campaign(registry, campaign, "second try")
    listed = list_registered_campaigns(registry)
    assert [entry["goal"] for entry in listed] == ["second try", "other goal"]


def test_resume_arguments_fill_only_what_the_caller_left_unset():
    """`kapso build --output <dir> --resume` with nothing else must run
    with the launch's arguments; an explicit value (a bigger budget) wins."""
    record = {"host": "codex", "eval_dir": "eval", "experiments": 3, "minutes": 60.0, "usd": None}
    merged = resume_arguments(record, {
        "host": None, "eval_dir": None, "experiments": None, "minutes": 90.0, "usd": None,
    })
    assert merged["host"] == "codex" and merged["eval_dir"] == "eval"
    assert merged["experiments"] == 3
    assert merged["minutes"] == 90.0
    assert merged["usd"] is None


def test_launch_mismatches_name_fingerprinted_changes_and_ignore_respelled_paths(tmp_path):
    """The checkpoint refuses a changed eval-dir or host with a bare
    fingerprint mismatch; the record can say which setting changed. A
    respelled path to the same directory is not a change."""
    (tmp_path / "eval").mkdir()
    record = {
        "host": None, "eval_dir": str(tmp_path / "eval"),
        "config_path": "/x/config.yaml", "minutes": 60.0,
    }
    respelled = {**record, "eval_dir": str(tmp_path / "sub" / ".." / "eval"), "minutes": 120.0}
    assert launch_mismatches(record, respelled) == {}

    changed = {**record, "eval_dir": None, "host": "codex"}
    assert launch_mismatches(record, changed) == {
        "host": (None, "codex"),
        "eval_dir": (record["eval_dir"], None),
    }
    # The loaded experience is identity by reference and scope: a bank
    # that moved since the launch (another head) is not a change, a
    # different list or scope is, by name.
    mine = {"reference": "mine", "kind": "bank", "resolved": "/b", "head": "aaa", "provenance": "measured by you", "scope": "harness"}
    served = {**record, "experience": [mine]}
    assert launch_mismatches(served, {**served, "experience": [{**mine, "head": "bbb"}]}) == {}
    lifted = [{**mine, "scope": "all"}]
    assert launch_mismatches(served, {**served, "experience": lifted}) == {"experience": ([mine], lifted)}
    assert launch_mismatches(served, {**served, "experience": []}) == {"experience": ([mine], [])}


def test_a_restarted_session_continues_the_answered_requests_of_its_node(tmp_path):
    # The harness setup session (node -1) is run again with the replies rather than
    # resumed: its answered requests are recorded continued by the session that
    # carries them, once; an open one stays open, another node's are not its own.
    path = inbox_path(tmp_path)
    file_requests(path, node=-1, session="setup-1", entries=[ENTRY, DATA_ENTRY])
    file_requests(path, node=0, session="s-0", entries=[{**ENTRY, "key": "env:OTHER"}])
    record_reply(path, 1, "the key is in .env now")
    record_reply(path, 3, "the other key is in .env now")
    assert continue_answered(path, node=-1, session="setup-2") == [1]
    requests = load_requests(path)
    assert (requests[1].state, requests[2].state, requests[3].state) == ("continued", "open", "answered")
    last = json.loads(path.read_text().splitlines()[-1])
    assert {key: last[key] for key in ("event", "id", "node", "session")} == {"event": "continued", "id": 1, "node": -1, "session": "setup-2"}
    assert continue_answered(path, node=-1, session="setup-3") == [], "recorded once; a third run has nothing to continue"
