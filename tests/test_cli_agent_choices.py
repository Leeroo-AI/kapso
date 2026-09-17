"""Regression tests for coding-agent choices exposed by the main CLI."""

import sys

import pytest

import kapso.cli as cli
from kapso.cli import AVAILABLE_AGENTS
from kapso.execution.coding_agents.factory import CodingAgentFactory


def test_cli_agent_choices_match_factory_registry():
    """Inference-only adapters stay registered but cannot code or deploy."""
    assert "openai_compatible" in CodingAgentFactory.list_available()
    assert "openai_compatible" not in AVAILABLE_AGENTS
    assert AVAILABLE_AGENTS == [
        name
        for name in CodingAgentFactory.list_available()
        if not CodingAgentFactory.get_agent_info(name).get(
            "inference_only", False
        )
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["evolve", "--goal", "test", "-a", "openai_compatible"],
        ["deploy", "--coding-agent", "openai_compatible"],
    ],
)
def test_cli_rejects_inference_adapter(monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["kapso", *args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "invalid choice: 'openai_compatible'" in capsys.readouterr().err
