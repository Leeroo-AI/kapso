"""The campaign inbox record (docs/research/evolve-hub-design.md v4 §4.1).

What must hold: a request's state is the fold of its events (open →
answered → continued); ids are campaign-local and contiguous; a post on
an open key joins it; a post on an answered key opens a fresh request
carrying the previous reply; a reply must name an open request; a
continue must follow a reply; concurrent writers never tear the file; a
malformed line raises; a missing file is "no requests"; the launch record
round-trips; the registry skips deleted campaigns and raises on a
malformed line.
"""

import json
import threading
from pathlib import Path

import pytest

from kapso.execution.inbox import (
    LAUNCH_SCHEMA_VERSION,
    all_answered,
    file_requests,
    inbox_path,
    list_registered_campaigns,
    load_requests,
    open_requests,
    read_launch_record,
    record_continued,
    record_reply,
    register_campaign,
    render_stop_text,
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
    with pytest.raises(ValueError, match="already answered"):
        record_reply(path, 1, "again")


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


def test_launch_record_round_trip_and_missing_default(tmp_path):
    assert read_launch_record(tmp_path) is None
    write_launch_record(tmp_path, {"output_path": str(tmp_path), "max_iterations": 4})
    record = read_launch_record(tmp_path)
    assert record["schema_version"] == LAUNCH_SCHEMA_VERSION
    assert record["max_iterations"] == 4
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
