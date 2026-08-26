"""Regression tests for coding-agent choices exposed by the main CLI."""

from kapso.cli import AVAILABLE_AGENTS
from kapso.execution.coding_agents.factory import CodingAgentFactory


def test_codex_is_available_to_cli_and_factory():
    """The CLI must accept every shipped, registered Codex adapter."""
    assert "codex" in AVAILABLE_AGENTS
    assert CodingAgentFactory.is_available("codex")
