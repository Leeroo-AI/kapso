"""
Gated MCP Server - Selective tool exposure for Claude Code agents.

This module provides a configurable MCP server that exposes different
tool sets based on presets. Each preset enables specific gates with
custom parameters.

Usage:
    from src.knowledge.gated_mcp import get_allowed_tools_for_preset
    
    # Get allowed tools for Claude Code config
    tools = get_allowed_tools_for_preset("ideation", "gated-knowledge")
    
    # Configure MCP server
    mcp_servers = {
        "gated-knowledge": {
            "command": "python",
            "args": ["-m", "src.knowledge.gated_mcp.server"],
            "env": {"MCP_PRESET": "ideation"},
        }
    }
"""

from src.knowledge.gated_mcp.presets import (
    PRESETS,
    Preset,
    GateConfig,
    get_preset,
    get_allowed_tools_for_preset,
    list_presets,
    GATE_TOOL_NAMES,
)
from src.knowledge.gated_mcp.server import create_gated_mcp_server

__all__ = [
    # Presets
    "PRESETS",
    "Preset",
    "GateConfig",
    "get_preset",
    "get_allowed_tools_for_preset",
    "list_presets",
    "GATE_TOOL_NAMES",
    # Server
    "create_gated_mcp_server",
]
