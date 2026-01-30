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
from kapso.execution.coding_agents.commit_message_generator import CommitMessageGenerator

__all__ = [
    "CodingAgentInterface",
    "CodingAgentConfig", 
    "CodingResult",
    "CodingAgentFactory",
    "CommitMessageGenerator",
]
