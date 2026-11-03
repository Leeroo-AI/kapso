"""`kapso experience`: every subcommand's flags are accounted for; `add
mine` creates the bank once and says so the second time; `add
leeroopedia` is the key's presence; `list` has an empty state and the
rows; `share` attaches a remote to a bare bank and pushes it."""

import argparse
import sys
from pathlib import Path

import pytest

import kapso.cli.experience as experience_cli
from kapso.cli import main
from kapso.learning.bank_remote import bank_origin
from tests.handles_support import git
from tests.test_experience_reference import bank_home, settings_for


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["kapso", "experience", *argv])
    main()


@pytest.fixture
def settings(monkeypatch, tmp_path):
    values = settings_for(tmp_path)
    monkeypatch.setattr(experience_cli, "experience_settings", lambda config: dict(values))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEEROOPEDIA_API_KEY", raising=False)
    return values


def _subparsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    experience_cli.add_parser(subparsers)
    verb = subparsers.choices["experience"]
    sub = next(action for action in verb._actions if isinstance(action, argparse._SubParsersAction))
    return {name: {a.dest for a in p._actions if a.dest != "help"} for name, p in sub.choices.items()}


def test_every_subcommand_and_flag_is_accounted_for():
    assert _subparsers() == {
        "add": {"ref", "repo", "research", "paper", "name", "config", "detach"},
        "list": {"config"},
        "share": {"url", "config"},
        "update": {"ref", "config"},
        "remove": {"ref", "config"},
    }


def test_update_and_remove_go_through_the_facade(settings, monkeypatch):
    calls = []

    class FakeKapso:
        def __init__(self, config=None):
            calls.append(("init", config))

        def update_experience(self, ref=None):
            calls.append(("update", ref))

        def remove_experience(self, ref):
            calls.append(("remove", ref))

    monkeypatch.setattr(experience_cli, "Kapso", FakeKapso)
    _run(monkeypatch, "update")
    _run(monkeypatch, "update", "acme/kapso-bank", "--config", "c.yaml")
    _run(monkeypatch, "remove", "org-x")
    assert calls == [
        ("init", None), ("update", None),
        ("init", "c.yaml"), ("update", "acme/kapso-bank"),
        ("init", None), ("remove", "org-x"),
    ]


def test_add_mine_creates_the_bank_once(settings, monkeypatch, capsys):
    _run(monkeypatch, "add", "mine")
    home = Path(settings["bank_home"])
    assert (home / "HEAD").is_file()
    assert capsys.readouterr().out.startswith(f"created your bank at {home} (empty;")
    _run(monkeypatch, "add", "mine")
    assert capsys.readouterr().out.startswith(f"your bank exists at {home}: head ")


def test_add_leeroopedia_is_the_keys_presence(settings, monkeypatch, capsys):
    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, "add", "leeroopedia")
    assert caught.value.code == 1 and "put LEEROOPEDIA_API_KEY=kpsk_... in .env" in capsys.readouterr().err
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "")
    with pytest.raises(SystemExit):
        _run(monkeypatch, "add", "leeroopedia")   # an empty key is no key
    capsys.readouterr()
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    monkeypatch.setattr(experience_cli, "probe_hosted_gate", lambda settings: "Leeroopedia")
    _run(monkeypatch, "add", "leeroopedia")
    assert "LEEROOPEDIA_API_KEY present and the gate answers as Leeroopedia" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _run(monkeypatch, "add", "org-x")
    assert "an index is ingested with --repo <url> --name <n>" in capsys.readouterr().err


def test_add_ingests_a_repository_or_a_question_through_the_facade(settings, monkeypatch, capsys):
    calls = []

    class FakeKapso:
        def __init__(self, config=None):
            calls.append(("init", config))

        def add_experience(self, **kwargs):
            calls.append(("add", kwargs))
            return kwargs["name"]

    monkeypatch.setattr(experience_cli, "Kapso", FakeKapso)
    _run(monkeypatch, "add", "--repo", "https://github.com/org/x", "--name", "org-x")
    assert calls == [("init", None), ("add", {"repo": "https://github.com/org/x", "research": None, "paper": None, "name": "org-x"})]
    calls.clear()
    _run(monkeypatch, "add", "--research", "QLoRA memory", "--name", "qlora", "--config", "c.yaml")
    assert calls == [("init", "c.yaml"), ("add", {"repo": None, "research": "QLoRA memory", "paper": None, "name": "qlora"})]
    calls.clear()
    _run(monkeypatch, "add", "--paper", "paper.pdf", "--name", "paper")
    assert calls == [("init", None), ("add", {"repo": None, "research": None, "paper": "paper.pdf", "name": "paper"})]
    # exactly one of the four, and an index needs its name
    for argv in (["add"], ["add", "mine", "--repo", "u"], ["add", "--repo", "u", "--research", "q"]):
        with pytest.raises(SystemExit):
            _run(monkeypatch, *argv)
        assert "exactly one of" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _run(monkeypatch, "add", "--repo", "https://github.com/org/x")
    assert "an index needs a name: --name <n>" in capsys.readouterr().err


def test_list_has_an_empty_state_and_the_rows(settings, monkeypatch, capsys, tmp_path):
    _run(monkeypatch, "list")
    out = capsys.readouterr().out
    assert out.startswith("no experience yet: `kapso experience add mine` creates your bank")

    home = bank_home(tmp_path, cards=2)
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    _run(monkeypatch, "list")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("mine") and str(home) in lines[0] and "2 cards" in lines[0] and "no remote (kapso experience share <url>)" in lines[0]
    assert lines[1].startswith("leeroopedia") and lines[1].rstrip().endswith("key present")
    assert len(lines) == 2


def test_share_attaches_a_remote_and_pushes(settings, monkeypatch, capsys, tmp_path):
    remote = tmp_path / "team.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    _run(monkeypatch, "share", str(remote))
    home = Path(settings["bank_home"])
    out = capsys.readouterr().out
    assert f"created your bank at {home}" in out and f"shares to {remote}" in out
    assert "measured by you and your team" in out
    assert bank_origin(home) == str(remote)
    assert git(remote, "rev-parse", "main") == git(home, "rev-parse", "main")
    _run(monkeypatch, "list")
    assert f"remote {remote}" in capsys.readouterr().out.splitlines()[0]
