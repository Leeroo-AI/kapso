"""
Gated MCP Server - Selective tool exposure for Claude Code agents.

This module provides a configurable MCP server that exposes different
tool sets based on gate selection.

Usage:
    from src.knowledge.gated_mcp import get_allowed_tools_for_gates
    
    # Get allowed tools for Claude Code config
    tools = get_allowed_tools_for_gates(["idea", "research"], "gated-knowledge")
    
    # Configure MCP server
    mcp_servers = {
        "gated-knowledge": {
            "command": "python",
            "args": ["-m", "src.knowledge.gated_mcp.server"],
            "env": {"MCP_ENABLED_GATES": "idea,research"},
        }
    }
"""

from src.knowledge.gated_mcp.presets import (
    GATES,
    GateDefinition,
    get_allowed_tools_for_gates,
    list_gates,
    get_gate_config,
)
from src.knowledge.gated_mcp.server import create_gated_mcp_server

__all__ = [
    # Gates
    "GATES",
    "GateDefinition",
    "get_allowed_tools_for_gates",
    "list_gates",
    "get_gate_config",
    # Server
    "create_gated_mcp_server",
]
