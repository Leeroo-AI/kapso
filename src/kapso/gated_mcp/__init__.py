"""
Gated MCP Server - Selective tool exposure for Claude Code agents.

This module provides a configurable MCP server that exposes different
tool sets based on gate selection.

Usage:
    from kapso.gated_mcp import get_mcp_config
    
    # Get MCP config and allowed tools in one call
    mcp_servers, allowed_tools = get_mcp_config(["idea", "research"])
    
    # Use in Claude Code config
    config = CodingAgentConfig(agent_specific={
        "mcp_servers": mcp_servers,
        "allowed_tools": allowed_tools,
    })
"""

from kapso.gated_mcp.presets import (
    GATES,
    GateDefinition,
    get_allowed_tools_for_gates,
    get_mcp_config,
    list_gates,
    get_gate_config,
)
from kapso.gated_mcp.server import create_gated_mcp_server

__all__ = [
    # Gates
    "GATES",
    "GateDefinition",
    "get_allowed_tools_for_gates",
    "get_mcp_config",
    "list_gates",
    "get_gate_config",
    # Server
    "create_gated_mcp_server",
]
