"""Prior-knowledge-only stdio MCP executable for native coding-agent actions."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from kapso.gated_mcp.gates.base import GateConfig
from kapso.gated_mcp.gates.prior_knowledge_gate import PriorKnowledgeGate

_GATE_NAME = "prior_knowledge"
_FAILURE_POLICY = "error"


def _create_prior_knowledge_server(
    *,
    gate: PriorKnowledgeGate,
) -> Server:
    tools = gate.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    if len(tools_by_name) != len(tools):
        raise ValueError("prior-knowledge MCP tools are not uniquely named")
    server = Server("prior-knowledge")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: Dict[str, Any],
    ) -> List[TextContent]:
        if name not in tools_by_name:
            raise ValueError(f"unknown prior-knowledge tool: {name}")
        result = await gate.handle_call(name, arguments)
        if result is None:
            raise ValueError(f"prior-knowledge tool returned no result: {name}")
        return result

    return server


async def _run_prior_knowledge_server(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Run exactly one explicitly bounded prior-knowledge gate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-knowledge-path", required=True)
    parser.add_argument(
        "--prior-knowledge-maximum-bytes",
        required=True,
        type=int,
    )
    parser.add_argument("--prior-knowledge-audit-path", required=True)
    parser.add_argument(
        "--prior-knowledge-audit-maximum-bytes",
        required=True,
        type=int,
    )
    parser.add_argument("--operation-id", required=True)
    parser.add_argument(
        "--enabled-gates",
        choices=(_GATE_NAME,),
        required=True,
    )
    parser.add_argument(
        "--gate-failure-policy",
        choices=(_FAILURE_POLICY,),
        required=True,
    )
    arguments = parser.parse_args()
    gate = PriorKnowledgeGate(
        GateConfig(
            enabled=True,
            params={
                "audit_maximum_bytes": arguments.prior_knowledge_audit_maximum_bytes,
                "audit_path": arguments.prior_knowledge_audit_path,
                "materialization_path": arguments.prior_knowledge_path,
                "maximum_bytes": arguments.prior_knowledge_maximum_bytes,
                "operation_id": arguments.operation_id,
            },
        )
    )
    server = _create_prior_knowledge_server(gate=gate)
    asyncio.run(_run_prior_knowledge_server(server))


__all__ = ["_create_prior_knowledge_server", "_run_prior_knowledge_server", "main"]


if __name__ == "__main__":
    main()
