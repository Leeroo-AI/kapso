"""`kapso inbox` and `kapso inbox reply` (design v4 §4.6, §10 scenarios 21–26).

What must hold: inside a campaign `kapso inbox` lists its open requests
with the idea line and the reply hint (the id left out when one request
is open); from elsewhere a named directory or the registry is used;
`kapso inbox reply` resolves the campaign and the id the same way,
refuses a credential unless --yes, and prints the campaign's summary —
COMPLETED, or WAITING ON YOU when it asked again. `kapso inbox note`
records the text exactly as given for the campaign named or the one the
current directory is inside, says which sessions will read it, and
writes nothing when the text is empty, shaped like a credential without
--yes or not one argument, or when no campaign is meant; `kapso inbox`
lists the notes below whatever waits, and still says when nothing does.
"""

import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from kapso.cli import main
from kapso.cli.common import _print_build_summary
from kapso.cli.inbox import _campaign_goal, cmd_inbox
from kapso.execution.inbox import (
    file_requests,
    inbox_path,
    notes_path,
    read_notes,
    record_reply,
    register_campaign,
)
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore, config_fingerprint
from kapso.execution.search_strategies.base import SearchNode
from kapso.handles import Brief
from tests.handles_support import make_solution
from kapso.kapso import Kapso

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError",
    "tried": "OPENAI_KEY unset too",
    "fix": "add OPENAI_API_KEY=sk-... to .env",
    "next_steps": "embed, re-rank, evaluate",
}


def _paused_campaign(root: Path, name: str = "campaign", *, requests: int = 1) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True)
    entries = [ENTRY] + [{**ENTRY, "key": "data/x.csv"}] * (requests - 1)
    file_requests(inbox_path(workspace), node=3, session="s", entries=entries)
    node = SearchNode(
        node_id=3, branch_name="generic_exp_3", solution="# Core Idea\nre-rank with embeddings\n",
        suspended=True, request_ids=list(range(1, requests + 1)), cli_session_id="cli-1",
    )
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal=Brief(objective="Improve support\nmore detail").render(),
        config_fingerprint=config_fingerprint({"m": 1}), status="running", completed_iterations=2,
        cumulative_cost=1.0, current_feedback=None, strategy_state={"node_history": [node.to_dict()]},
        last_stop="waiting_for_user",
    ))
    return workspace


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["kapso", *argv])
    main()


def test_the_goal_line_reads_a_brief_or_a_plain_goal(tmp_path):
    """A campaign launched through build() holds the rendered brief; one a
    benchmark runner launched holds its plain goal. Both list."""
    built = _paused_campaign(tmp_path, "built")
    assert _campaign_goal(built) == "Improve support"
    plain = tmp_path / "plain"
    plain.mkdir()
    RunCheckpointStore(str(plain)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Predict churn on rel-f1\nwith the official split",
        config_fingerprint=config_fingerprint({"m": 1}), status="running", completed_iterations=0,
        cumulative_cost=0.0, current_feedback=None, strategy_state={"node_history": []},
    ))
    assert _campaign_goal(plain) == "Predict churn on rel-f1"
    assert _campaign_goal(tmp_path / "nowhere") == ""


def _capture_reply(monkeypatch, result=None) -> List[Any]:
    calls: List[Any] = []

    def reply(cls, campaign, request_id, note="", *, allow_secret=False):
        calls.append((campaign, request_id, note, allow_secret))
        return result

    monkeypatch.setattr(Kapso, "reply", classmethod(reply))
    return calls


def test_inbox_inside_a_campaign_lists_its_requests(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox")
    out = capsys.readouterr().out
    assert out.startswith(f"{workspace.resolve()}  Improve support  waiting 1m")
    assert "#1  env:OPENAI_API_KEY" in out
    assert "for    node 3 · re-rank with embeddings" in out
    assert 'reply with   kapso inbox reply "…"' in out


def test_inbox_names_a_campaign_from_elsewhere(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path, requests=2)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", str(workspace))
    out = capsys.readouterr().out
    assert "· 2 requests" in out and "#2  data/x.csv" in out
    assert f'reply with   kapso inbox reply {shlex.quote(str(workspace.resolve()))} 1 "…"' in out

    monkeypatch.setattr(Kapso, "inbox", staticmethod(lambda campaign: []))
    _run(monkeypatch, "inbox", str(workspace))
    assert capsys.readouterr().out.strip() == f"Nothing waiting on you in {workspace.resolve()}."


def test_inbox_from_elsewhere_reads_the_registry(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(f"inbox:\n  registry: {registry}\n")
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "--config", str(config))
    assert capsys.readouterr().out.strip() == "Nothing waiting on you."

    quiet = tmp_path / "quiet"
    quiet.mkdir()
    register_campaign(registry, quiet, "Nothing asked")
    waiting = _paused_campaign(tmp_path, "waiting")
    register_campaign(registry, waiting, "Improve support")
    _run(monkeypatch, "inbox", "--config", str(config))
    out = capsys.readouterr().out
    assert str(waiting) in out and str(quiet) not in out
    assert "#1  env:OPENAI_API_KEY" in out
    assert f'reply with   kapso inbox reply {shlex.quote(str(waiting.resolve()))} "…"' in out


def test_suggested_reply_works_where_it_was_printed(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path, "my campaign")
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", str(workspace))
    hint = next(
        line for line in capsys.readouterr().out.splitlines() if "reply with" in line
    )
    command = shlex.split(hint.split("reply with", 1)[1].replace("…", "added the key"))
    assert command[:3] == ["kapso", "inbox", "reply"]
    _run(monkeypatch, *command[1:])
    assert calls == [(str(workspace.resolve()), 1, "added the key", False)]


def test_reply_resolves_the_campaign_and_the_id(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    calls = _capture_reply(monkeypatch)

    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "reply", str(workspace), "1", "added the key")
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "reply", "added the key")
    _run(monkeypatch, "inbox", "reply")

    resolved = str(workspace.resolve())
    assert calls == [
        (resolved, 1, "added the key", False),
        (resolved, 1, "added the key", False),
        (resolved, 1, "", False),
    ]
    assert capsys.readouterr().out == ""


def test_reply_needs_the_id_when_several_are_open(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path, requests=2)
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "done")
    assert "which request?" in capsys.readouterr().err and calls == []

    _run(monkeypatch, "inbox", "reply", "2", "done")
    assert calls == [(str(workspace.resolve()), 2, "done", False)]


def _failed_resume(root: Path, name: str = "campaign", *, requests: int = 1) -> Path:
    workspace = _paused_campaign(root, name, requests=requests)
    for request_id in range(1, requests + 1):
        record_reply(inbox_path(workspace), request_id, "added the key")
    return workspace


def test_inbox_shows_a_reply_a_failed_resume_did_not_deliver(tmp_path, monkeypatch, capsys):
    workspace = _failed_resume(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox")
    out = capsys.readouterr().out
    assert "#1  env:OPENAI_API_KEY   answered, campaign not resumed" in out
    assert "reply  'added the key'" in out
    assert "retry with   kapso inbox reply " in out
    assert "Nothing waiting" not in out


def test_the_global_inbox_lists_a_campaign_awaiting_a_resume(tmp_path, monkeypatch, capsys):
    workspace = _failed_resume(tmp_path)
    registry = tmp_path / "registry.jsonl"
    register_campaign(registry, workspace, "Improve support")
    config = tmp_path / "config.yaml"
    config.write_text(f"inbox:\n  registry: {registry}\n")
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "--config", str(config))
    assert f"retry with   kapso inbox reply {shlex.quote(str(workspace.resolve()))}" in capsys.readouterr().out


def test_a_retry_needs_no_id(tmp_path, monkeypatch, capsys):
    workspace = _failed_resume(tmp_path, requests=2)
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "reply")
    assert calls[-1][1:3] == (1, "")
    # A note has to go on one request of the two.
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "use staging")
    assert "which request?" in capsys.readouterr().err
    _run(monkeypatch, "inbox", "reply", "2", "use staging")
    assert calls[-1][1:3] == (2, "use staging")


def test_reply_outside_a_campaign_must_name_it(tmp_path, monkeypatch, capsys):
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "1", "done")
    assert "not inside a campaign" in capsys.readouterr().err and calls == []


def test_reply_refuses_a_credential_without_yes(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(workspace)
    token = "sk-abcdefghijklmnopqrstuvwxyz0123"
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "1", token)
    assert "looks like a credential" in capsys.readouterr().out and calls == []

    _run(monkeypatch, "inbox", "reply", "1", token, "--yes")
    assert calls == [(str(workspace.resolve()), 1, token, True)]


def test_reply_prints_the_campaign_summary(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _capture_reply(monkeypatch, make_solution(workspace, "Improve support", stopped_reason="max_iterations", cost_usd=2.5))
    _run(monkeypatch, "inbox", "reply", "1", "added the key")
    out = capsys.readouterr().out
    assert "COMPLETED" in out and "Stopped:         max_iterations" in out and "Cost:            $2.500" in out


def test_the_summary_says_waiting_on_you_when_the_campaign_asked_again(tmp_path, capsys):
    workspace = tmp_path / "campaign"
    request = {"id": 2, "key": "env:OPENAI_API_KEY"}
    _print_build_summary(make_solution(
        workspace, "Improve support",
        stopped_reason="waiting_for_user", requests=[request, {**request, "id": 3}],
    ))
    out = capsys.readouterr().out
    assert "WAITING ON YOU" in out and "COMPLETED" not in out
    assert f'Reply with:  kapso inbox reply 2 "…"   (inside {workspace})' in out
    assert f"Any time:    kapso inbox {workspace}" in out


def test_inbox_help_shows_every_form(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "inbox", "--help")
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "reply [CAMPAIGN] [ID] [NOTE]" in out and "kapso inbox reply 1" in out
    # argparse wraps the forms' line at the terminal's width.
    assert "note [CAMPAIGN] TEXT" in " ".join(out.split()) and 'kapso inbox note ./campaign "skip ticket 14"' in out
    assert cmd_inbox.__doc__


TOKEN = "sk-abcdefghijklmnopqrstuvwxyz0123"
NOTE = "  The staging warehouse is read-only.\n\nWrite to DEV_DB;  keep  the double spaces.\n"


def test_a_note_is_recorded_as_given_and_listed_below_what_waits(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "note", NOTE)
    assert capsys.readouterr().out == (
        f"Note 1 left for {workspace.resolve()}.\n"
        "A session that is running right now has already started and will not see it.\n"
        "Every ideation, implementation and judging session that starts from here on reads it.\n"
    )
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "note", str(workspace), "skip ticket 14")
    assert capsys.readouterr().out.startswith(f"Note 2 left for {workspace.resolve()}.\n")
    first, second = read_notes(notes_path(workspace))
    assert (first.id, first.text, second.id, second.text) == (1, NOTE, 2, "skip ticket 14")

    _run(monkeypatch, "inbox", str(workspace))
    out = capsys.readouterr().out
    assert "#1  env:OPENAI_API_KEY" in out
    assert out.index("reply with") < out.index("your notes") < out.index("note 1") < out.index("note 2")
    assert (
        f"  note 1  {first.written_at}\n"
        "        The staging warehouse is read-only.\n"
        "\n"
        "      Write to DEV_DB;  keep  the double spaces.\n"
        "\n"
        f"  note 2  {second.written_at}\n"
        "      skip ticket 14\n"
    ) in out


def _quiet_campaign(root: Path, name: str = "campaign") -> Path:
    """A campaign that never asked for anything (its inbox may be off)."""
    workspace = root / name
    workspace.mkdir(parents=True)
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Predict churn", config_fingerprint=config_fingerprint({"m": 1}),
        status="running", completed_iterations=0, cumulative_cost=0.0, current_feedback=None,
        strategy_state={"node_history": []},
    ))
    return workspace


def test_notes_are_listed_when_nothing_is_waiting(tmp_path, monkeypatch, capsys):
    """A campaign that never asked for anything still takes notes and
    shows them."""
    workspace = _quiet_campaign(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "note", str(workspace), "skip ticket 14")
    capsys.readouterr()

    _run(monkeypatch, "inbox", str(workspace))
    out = capsys.readouterr().out
    assert out.startswith(f"Nothing waiting on you in {workspace.resolve()}.\n\n  your notes   ")
    assert out.rstrip().endswith("      skip ticket 14")


@pytest.mark.parametrize("words, stream, message", [
    ((), "err", "a note needs text"),
    (("  \n",), "err", "a note needs text"),
    (("{campaign}",), "err", "a note needs text"),          # the campaign named, the text forgotten
    (("the key is {token}",), "out", "looks like a credential"),
    (("ticket {undecodable} is fixed",), "err", "bytes that are not UTF-8 text"),
    (("ticket {undecodable} is fixed", "--yes"), "err", "bytes that are not UTF-8 text"),
    (("skip", "ticket", "14"), "err", "skip is not a directory — the form is kapso inbox note [campaign]"),
    (("{campaign}", "skip", "ticket", "14"), "err", "the note is more than one argument"),
])
def test_a_note_in_doubt_is_refused_and_nothing_is_written(tmp_path, monkeypatch, capsys, words, stream, message):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    # A byte that was not UTF-8 on the command line: Python holds it as a lone surrogate.
    words = [word.format(campaign=workspace, token=TOKEN, undecodable=chr(0xDCE9)) for word in words]
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "inbox", "note", *words)
    assert stop.value.code == 1
    assert message in getattr(capsys.readouterr(), stream)
    assert not notes_path(workspace).exists()


@pytest.mark.parametrize("text", [
    # Longer than a file name may be: asking the file system whether the
    # note is a directory must not raise.
    "The staging warehouse was rebuilt this morning and every nightly model must stay under its row budget. " * 12,
    # A leading tilde opens a user's home directory, and there is no such user.
    "~10% of orders rows have a null customer_id",
], ids=["a long paragraph", "a leading tilde"])
def test_the_short_form_takes_any_text(tmp_path, monkeypatch, capsys, text):
    """Inside the campaign the note is the only word, so the question "is
    the first word a directory" is put to the note itself."""
    workspace = _quiet_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "note", text)
    assert capsys.readouterr().out.startswith(f"Note 1 left for {workspace.resolve()}.\n")
    assert [note.text for note in read_notes(notes_path(workspace))] == [text]


def test_a_credential_shaped_note_is_stored_with_yes(tmp_path, monkeypatch, capsys):
    """The shapes also match hyphenated names, so the refusal names the
    reply's way out and --yes takes it."""
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    text = "start from mask-rcnn-resnet50-fpn-coco"
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "note", text)
    assert "To store it anyway, add --yes at the end." in capsys.readouterr().out
    assert not notes_path(workspace).exists()

    _run(monkeypatch, "inbox", "note", text, "--yes")
    assert capsys.readouterr().out.startswith("Note 1 left for ")
    assert [note.text for note in read_notes(notes_path(workspace))] == [text]


def test_a_note_needs_a_campaign(tmp_path, monkeypatch, capsys):
    """A directory is a campaign by its status file or its checkpoint. A
    bare `.kapso/` is not one: a session's clone has it (the repository
    memory) and so does the home directory (the registry), and a note
    filed there would be promised to sessions that never read it."""
    plain = tmp_path / "plain"
    plain.mkdir()
    clone = tmp_path / "campaign" / "sessions" / "generic_exp_3"
    (clone / ".kapso").mkdir(parents=True)
    (clone / ".kapso" / "repo_memory.json").write_text("{}")
    monkeypatch.chdir(clone)
    for words, message in [
        (("skip ticket 14",), "not inside a campaign"),
        ((str(tmp_path / "nowhere"), "skip ticket 14"), "nowhere is not a directory"),
        ((str(plain), "skip ticket 14"), "plain is not a campaign directory"),
        ((".", "skip ticket 14"), "generic_exp_3 is not a campaign directory"),
    ]:
        with pytest.raises(SystemExit) as stop:
            _run(monkeypatch, "inbox", "note", *words)
        assert stop.value.code == 1
        assert message in capsys.readouterr().err
    assert not any(plain.iterdir()) and sorted(path.name for path in (clone / ".kapso").iterdir()) == ["repo_memory.json"]


def test_a_campaign_in_its_setup_minutes_takes_a_note_by_either_form(tmp_path, monkeypatch, capsys):
    """The status file is there from the workspace's setup on, the
    checkpoint only after the setup sessions: both forms go by either, so
    a note left from inside a campaign that is still setting up is filed
    there, never in a campaign around it."""
    outer = _quiet_campaign(tmp_path, "outer")
    inner = outer / "inner"
    (inner / ".kapso").mkdir(parents=True)
    (inner / ".kapso" / "status.json").write_text('{"state": "starting", "phase": "setup"}')
    monkeypatch.chdir(inner)
    _run(monkeypatch, "inbox", "note", "left by the short form")
    _run(monkeypatch, "inbox", "note", ".", "left by naming it")
    assert capsys.readouterr().out.count(f"left for {inner.resolve()}.") == 2
    assert [note.text for note in read_notes(notes_path(inner))] == ["left by the short form", "left by naming it"]
    assert not notes_path(outer).exists()


def test_a_campaign_named_with_a_tilde_is_found(tmp_path, monkeypatch, capsys):
    """A quoted `~/campaign` reaches the command unexpanded."""
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = _quiet_campaign(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "note", "~/campaign", "skip ticket 14")
    assert capsys.readouterr().out.startswith(f"Note 1 left for {workspace.resolve()}.\n")
    assert [note.text for note in read_notes(notes_path(workspace))] == ["skip ticket 14"]


def test_a_note_from_inside_a_session_clone_reaches_its_campaign(tmp_path, monkeypatch, capsys):
    """With no directory named, the campaign is the one the current
    directory is inside: the nearest campaign root, not the clone's own
    `.kapso/`."""
    workspace = _quiet_campaign(tmp_path)
    clone = workspace / "sessions" / "generic_exp_3"
    (clone / ".kapso").mkdir(parents=True)
    monkeypatch.chdir(clone)
    _run(monkeypatch, "inbox", "note", "skip ticket 14")
    assert capsys.readouterr().out.startswith(f"Note 1 left for {workspace.resolve()}.\n")
    assert [note.text for note in read_notes(notes_path(workspace))] == ["skip ticket 14"]
    assert not notes_path(clone).exists()


def test_a_wrong_directory_never_sends_the_note_to_the_campaign_you_stand_in(tmp_path, monkeypatch, capsys):
    """Inside one campaign, naming another that does not exist must not
    file the path and the text as a note here: a note cannot be removed."""
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "inbox", "note", "../campaign-b", "skip ticket 14")
    assert stop.value.code == 1
    assert "../campaign-b is not a directory" in capsys.readouterr().err
    assert not notes_path(workspace).exists()
