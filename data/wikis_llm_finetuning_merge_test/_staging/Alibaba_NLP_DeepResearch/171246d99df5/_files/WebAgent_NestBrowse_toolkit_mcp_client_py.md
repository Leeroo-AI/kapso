# File: `WebAgent/NestBrowse/toolkit/mcp_client.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 49 |
| Functions | `mcp_client` |
| Imports | anyio, asyncio, contextlib, logging, mcp, traceback, typing, uuid |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides an async context manager for establishing and maintaining Model Context Protocol (MCP) client connections to browser servers.

**Mechanism:** The `mcp_client` async context manager connects to a browser MCP server via Server-Sent Events (SSE), initializes a `ClientSession`, and sets up a background ping loop (every 20 seconds) to keep the connection alive. Each connection is identified by a unique UUID route key. Upon entering the context, it lists available tools and yields both the session and an asyncio lock for thread-safe operations.

**Significance:** Infrastructure component that enables NestBrowse to communicate with browser automation servers using the MCP protocol. Handles connection lifecycle, keepalive, and provides session management for browser tool invocations.
