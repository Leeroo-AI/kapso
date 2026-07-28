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

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from kapso.gated_mcp.presets import GATES, resolve_gates
from kapso.gated_mcp.gates.base import GateConfig, ToolGate
from kapso.gated_mcp.gates.code_gate import CodeGate
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
    "repo_memory": RepoMemoryGate,
    "prior_knowledge": PriorKnowledgeGate,
}


def _resolve_configuration(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
    prior_knowledge_audit_path: str | None = None,
    operation_id: str | None = None,
    enabled_gate_names: tuple[str, ...] | None = None,
    gate_failure_policy: str | None = None,
) -> Dict[str, GateConfig]:
    """
    Resolve which gates to enable and their configurations.

    Explicit call arguments are authoritative for isolated subprocesses. The
    existing ambient preset path remains available to the standalone server.

    Returns:
        Dict mapping gate names to their configurations
    """
    enabled_gates = os.getenv("MCP_ENABLED_GATES", "").strip()

    if enabled_gate_names is not None:
        requested_gates = list(enabled_gate_names)
        resolution_environment = {}
        selected_failure_policy = gate_failure_policy
        if selected_failure_policy is None:
            raise ValueError("explicit gates require an explicit failure policy")
    elif enabled_gates:
        requested_gates = [
            gate.strip() for gate in enabled_gates.split(",") if gate.strip()
        ]
        resolution_environment = os.environ
        selected_failure_policy = os.getenv("MCP_GATE_FAILURE_POLICY", "warn")
        logger.info(f"Requested gates: {requested_gates}")
    else:
        requested_gates = [
            gate_name
            for gate_name in GATE_CLASSES
            if gate_name != "prior_knowledge" or prior_knowledge_path is not None
        ]
        resolution_environment = os.environ
        selected_failure_policy = os.getenv("MCP_GATE_FAILURE_POLICY", "warn")
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
        policy=selected_failure_policy,
        env=resolution_environment,
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
        if (prior_knowledge_audit_path is None) != (operation_id is None):
            raise ValueError(
                "prior knowledge audit path and operation id must appear together"
            )
        if prior_knowledge_audit_path is not None:
            configs["prior_knowledge"].params["audit_path"] = prior_knowledge_audit_path
            configs["prior_knowledge"].params["operation_id"] = operation_id
    return configs


def create_gated_mcp_server(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
    prior_knowledge_audit_path: str | None = None,
    operation_id: str | None = None,
    enabled_gate_names: tuple[str, ...] | None = None,
    gate_failure_policy: str | None = None,
) -> "Server":
    """
    Create and configure the gated MCP server.

    Returns:
        Configured MCP Server instance

    Raises:
        ImportError: If mcp package not installed
        ValueError: If tool name collision detected
    """
    # Resolve configuration
    gate_configs = _resolve_configuration(
        prior_knowledge_path,
        prior_knowledge_maximum_bytes,
        prior_knowledge_audit_path,
        operation_id,
        enabled_gate_names,
        gate_failure_policy,
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
            raise ValueError(f"Unknown tool: {name}. Available: {available}")

        gate = tool_to_gate[name]
        result = await gate.handle_call(name, arguments)
        if result is None:
            raise ValueError(f"Tool '{name}' returned no result")
        return result

    return mcp


async def run_server(
    prior_knowledge_path: str | None = None,
    prior_knowledge_maximum_bytes: int | None = None,
    prior_knowledge_audit_path: str | None = None,
    operation_id: str | None = None,
    enabled_gate_names: tuple[str, ...] | None = None,
    gate_failure_policy: str | None = None,
):
    """Run the MCP server with stdio transport."""
    logger.info("Starting Gated MCP Server...")

    mcp = create_gated_mcp_server(
        prior_knowledge_path,
        prior_knowledge_maximum_bytes,
        prior_knowledge_audit_path,
        operation_id,
        enabled_gate_names,
        gate_failure_policy,
    )

    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio transport")
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-knowledge-path")
    parser.add_argument("--prior-knowledge-maximum-bytes", type=int)
    parser.add_argument("--prior-knowledge-audit-path")
    parser.add_argument("--operation-id")
    parser.add_argument("--enabled-gates", nargs="+")
    parser.add_argument("--gate-failure-policy")
    arguments = parser.parse_args()

    asyncio.run(
        run_server(
            arguments.prior_knowledge_path,
            arguments.prior_knowledge_maximum_bytes,
            arguments.prior_knowledge_audit_path,
            arguments.operation_id,
            (
                None
                if arguments.enabled_gates is None
                else tuple(arguments.enabled_gates)
            ),
            arguments.gate_failure_policy,
        )
    )


if __name__ == "__main__":
    main()
