# Wiki MCP Server
#
# MCP (Model Context Protocol) server for exposing the knowledge search
# functionality to AI coding agents like Claude Code, Cursor, and others.
#
# This module wraps the knowledge search backend as an MCP server,
# enabling seamless integration with AI-powered development tools.

from src.knowledge.wiki_mcps.mcp_server import (
    create_mcp_server,
    run_mcp_server,
)

__all__ = [
    "create_mcp_server",
    "run_mcp_server",
]

