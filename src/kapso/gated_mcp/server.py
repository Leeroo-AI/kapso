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

import argparse
import asyncio
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
from kapso.gated_mcp.gates.base import GateConfig, ToolGate
from kapso.gated_mcp.gates.code_gate import CodeGate
from kapso.gated_mcp.gates.experiment_history_gate import ExperimentHistoryGate
from kapso.gated_mcp.gates.idea_gate import IdeaGate
from kapso.gated_mcp.gates.kg_gate import KGGate
from kapso.gated_mcp.gates.prior_knowledge_gate import PriorKnowledgeGate
from kapso.gated_mcp.gates.repo_memory_gate import RepoMemoryGate
from kapso.gated_mcp.gates.research_gate import ResearchGate

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
    "prior_knowledge": PriorKnowledgeGate,
}


def _resolve_configuration(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
) -> Dict[str, GateConfig]:
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
        requested_gates = [
            gate_name
            for gate_name in GATE_CLASSES
            if gate_name != "prior_knowledge" or prior_knowledge_path is not None
        ]
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
    if "prior_knowledge" in configs:
        if prior_knowledge_path is None:
            raise ValueError(
                "prior_knowledge gate requires an explicit materialization path"
            )
        if (
            isinstance(prior_knowledge_maximum_bytes, bool)
            or not isinstance(prior_knowledge_maximum_bytes, int)
            or prior_knowledge_maximum_bytes <= 0
        ):
            raise ValueError(
                "prior_knowledge gate requires a positive materialization byte budget"
            )
        configs["prior_knowledge"].params["materialization_path"] = prior_knowledge_path
        configs["prior_knowledge"].params[
            "maximum_bytes"
        ] = prior_knowledge_maximum_bytes
    return configs


def create_gated_mcp_server(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
) -> "Server":
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
    gate_configs = _resolve_configuration(
        prior_knowledge_path,
        prior_knowledge_maximum_bytes,
    )

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
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}. Available: {available}",
                )
            ]

        gate = tool_to_gate[name]
        try:
            result = await gate.handle_call(name, arguments)
            if result is None:
                return [
                    TextContent(type="text", text=f"Tool '{name}' returned no result")
                ]
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error in {name}: {str(e)}")]

    return mcp


async def run_server(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
):
    """Run the MCP server with stdio transport."""
    if not HAS_MCP:
        raise ImportError("MCP package not installed. Install with: pip install mcp")

    logger.info("Starting Gated MCP Server...")

    mcp = create_gated_mcp_server(
        prior_knowledge_path,
        prior_knowledge_maximum_bytes,
    )

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio transport")
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-knowledge-path")
    parser.add_argument("--prior-knowledge-maximum-bytes", type=int)
    arguments = parser.parse_args()

    try:
        asyncio.run(
            run_server(
                arguments.prior_knowledge_path,
                arguments.prior_knowledge_maximum_bytes,
            )
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
