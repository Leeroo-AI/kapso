# Pluggable Coding Agents Module
#
# This module provides a pluggable architecture for different coding agents
# (Aider, Gemini CLI, Claude Code, OpenHands) that can be swapped in the
# orchestrator's experiment loop.
#
# Usage:
#   from kapso.execution.coding_agents import CodingAgentFactory, CodingAgentConfig
#   
#   config = CodingAgentConfig(agent_type="aider", model="o3", ...)
#   agent = CodingAgentFactory.create(config)

from kapso.execution.coding_agents.base import (
    CodingAgentInterface,
    CodingAgentConfig,
    CodingResult,
)
from kapso.execution.coding_agents.factory import CodingAgentFactory

# CommitMessageGenerator is NOT re-exported here: it imports
# kapso.core.cli_inference, which imports this package's factory — a
# package-level re-export would close that loop into an import cycle.
# Import it from its module directly.

__all__ = [
    "CodingAgentInterface",
    "CodingAgentConfig",
    "CodingResult",
    "CodingAgentFactory",
]
