#!/usr/bin/env python3
"""
Gated MCP Server

A configurable MCP server that exposes different tool sets based on gate selection.

Usage:
    # With gate list
    MCP_ENABLED_GATES=idea,research python -m kapso.gated_mcp.server

Environment Variables:
    MCP_ENABLED_GATES: Comma-separated gate names (e.g., "idea,research")
    MCP_GATE_FAILURE_POLICY: Missing-capability behavior (skip, warn, or error)
    KG_INDEX_PATH: Path to .index file for KG configuration
"""

import logging
import os
from typing import Any, Dict, List, Type

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None

from kapso.gated_mcp.presets import GATES, resolve_gates
from kapso.gated_mcp.gates.base import GateConfig
from kapso.gated_mcp.gates import (
    ToolGate,
    KGGate,
    IdeaGate,
    CodeGate,
    ResearchGate,
    ExperimentHistoryGate,
    RepoMemoryGate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gate class registry
GATE_CLASSES: Dict[str, Type[ToolGate]] = {
    "kg": KGGate,
    "idea": IdeaGate,
    "code": CodeGate,
    "research": ResearchGate,
    "experiment_history": ExperimentHistoryGate,
    "repo_memory": RepoMemoryGate,
}


def _resolve_configuration() -> Dict[str, GateConfig]:
    """
    Resolve which gates to enable and their configurations.
    
    Reads MCP_ENABLED_GATES env var for comma-separated gate names.
    Falls back to all gates if not specified.
    
    Returns:
        Dict mapping gate names to their configurations
    """
    enabled_gates = os.getenv("MCP_ENABLED_GATES", "").strip()

    if enabled_gates:
        requested_gates = [
            gate.strip() for gate in enabled_gates.split(",") if gate.strip()
        ]
        logger.info(f"Requested gates: {requested_gates}")
    else:
        requested_gates = list(GATE_CLASSES)
        logger.info("No gates specified, checking all bundled gates")

    unsupported = [
        gate for gate in requested_gates if gate in GATES and gate not in GATE_CLASSES
    ]
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"External gate(s) cannot run in the bundled MCP server: {names}"
        )

    resolution = resolve_gates(
        requested_gates,
        policy=os.getenv("MCP_GATE_FAILURE_POLICY", "warn"),
        env=os.environ,
    )

    configs = {}
    for name in resolution.enabled_gates:
        default_params = GATES[name].default_params
        configs[name] = GateConfig(enabled=True, params=default_params.copy())
    return configs


def create_gated_mcp_server() -> "Server":
    """
    Create and configure the gated MCP server.
    
    Returns:
        Configured MCP Server instance
        
    Raises:
        ImportError: If mcp package not installed
        ValueError: If tool name collision detected
    """
    if not HAS_MCP:
        raise ImportError("MCP package not installed. Install with: pip install mcp")
    
    # Resolve configuration
    gate_configs = _resolve_configuration()
    
    # Initialize enabled gates
    active_gates: Dict[str, ToolGate] = {}
    for gate_name, config in gate_configs.items():
        if not config.enabled:
            continue
        if gate_name not in GATE_CLASSES:
            logger.warning(f"Unknown gate: {gate_name}")
            continue
        
        gate_class = GATE_CLASSES[gate_name]
        active_gates[gate_name] = gate_class(config)
        logger.info(f"Enabled gate: {gate_name}")
    
    # Build tool registry with collision detection
    tool_to_gate: Dict[str, ToolGate] = {}
    all_tools: List[Tool] = []
    
    for gate_name, gate in active_gates.items():
        for tool in gate.get_tools():
            if tool.name in tool_to_gate:
                existing_gate = tool_to_gate[tool.name]
                raise ValueError(
                    f"Tool name collision: '{tool.name}' in both "
                    f"'{existing_gate.name}' and '{gate_name}' gates"
                )
            tool_to_gate[tool.name] = gate
            all_tools.append(tool)
    
    logger.info(f"Registered {len(all_tools)} tools from {len(active_gates)} gates")
    
    # Create MCP server
    mcp = Server("gated-knowledge")
    
    @mcp.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return all_tools
    
    @mcp.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls by dispatching to appropriate gate."""
        if name not in tool_to_gate:
            available = ", ".join(tool_to_gate.keys())
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}. Available: {available}",
            )]
        
        gate = tool_to_gate[name]
        try:
            result = await gate.handle_call(name, arguments)
            if result is None:
                return [TextContent(type="text", text=f"Tool '{name}' returned no result")]
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error in {name}: {str(e)}")]
    
    return mcp


async def run_server():
    """Run the MCP server with stdio transport."""
    if not HAS_MCP:
        raise ImportError("MCP package not installed. Install with: pip install mcp")
    
    logger.info("Starting Gated MCP Server...")
    
    mcp = create_gated_mcp_server()
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio transport")
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


def main():
    """CLI entry point."""
    import asyncio
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
