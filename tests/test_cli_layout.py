"""The CLI is a package with one module per verb. Each module exposes NAME
and add_parser(subparsers) returning its handler; main() wires them in a
fixed order. This pins the contract a new verb module must meet and the
verb list `kapso --help` shows."""

import argparse
import sys

import pytest

import kapso.cli as cli
from kapso.cli import main

VERBS_IN_ORDER = [
    "build", "harness", "experience", "learn", "deploy", "watch", "trace", "report", "inbox", "doctor",
]


def test_every_verb_module_meets_the_contract():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    names = []
    for module in cli.VERB_MODULES:
        assert isinstance(module.NAME, str) and module.NAME
        handler = module.add_parser(subparsers)
        assert callable(handler), module.NAME
        assert handler.__name__ == f"cmd_{module.NAME}", module.NAME
        names.append(module.NAME)
    assert names == VERBS_IN_ORDER
    assert list(subparsers.choices) == VERBS_IN_ORDER


def test_help_lists_every_verb(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["kapso", "--help"])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for verb in VERBS_IN_ORDER:
        assert verb in out


def test_no_command_prints_help_and_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["kapso"])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 1
    assert "Please specify a command" in capsys.readouterr().err
