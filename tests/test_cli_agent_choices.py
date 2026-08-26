"""Regression tests for coding-agent choices exposed by the main CLI."""

from kapso.cli import AVAILABLE_AGENTS
from kapso.execution.coding_agents.factory import CodingAgentFactory


def test_cli_agent_choices_match_factory_registry():
    """The CLI must expose every coding agent registered by the factory."""
    assert AVAILABLE_AGENTS == CodingAgentFactory.list_available()
