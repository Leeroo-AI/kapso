"""The public commands pass --config through to the Kapso facade."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kapso.cli as cli


@pytest.mark.parametrize(
    "verb,argument",
    [
        ("evolve", "--goal"),
        ("research", "--objective"),
    ],
)
@pytest.mark.parametrize("config_path", [None, "/tmp/kapso-config.yaml"])
def test_command_passes_config_to_facade(
    monkeypatch, verb, argument, config_path
):
    facade = Mock()
    facade.research.return_value = SimpleNamespace()
    constructor = Mock(return_value=facade)
    monkeypatch.setattr(cli, "Kapso", constructor)
    monkeypatch.setattr(cli, "_print_evolve_summary", lambda _: None)
    argv = ["kapso", verb, argument, "test objective"]
    if config_path:
        argv.extend(["--config", config_path])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert constructor.call_args.kwargs["config_path"] == config_path
    getattr(facade, verb).assert_called_once()
