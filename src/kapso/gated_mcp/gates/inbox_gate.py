"""
Inbox Gate
==========

One tool, ``request_from_user``: the session asks the person running the
campaign for something only a person can provide, and is stopped
(docs/research/evolve-hub-design.md v4, Appendix A.2). The gate appends
the requests to the campaign's inbox file through the env-injection
channel the sibling gates use (set by the launching runner, never read
by core kapso code):

- KAPSO_INBOX_PATH   the campaign's .kapso/inbox.jsonl
- KAPSO_SESSION_ID   the CLI session id kapso minted for this session
- KAPSO_NODE_ID      the experiment node this session implements

The adapter that launched the session tails the same file and ends the
session once a request for its session id appears. The feedback judge
never gets this gate.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Tool = None
    TextContent = None

from kapso.execution.inbox import REQUEST_FIELDS, file_requests, render_stop_text
from kapso.gated_mcp.gates.base import ToolGate

REQUEST_FROM_USER_TOOL = "request_from_user"

_FIELD_DESCRIPTIONS = {
    "key": (
        "What is needed, short and stable: env:OPENAI_API_KEY, "
        "access:hf:meta-llama/Llama-3.1-8B-Instruct, "
        "data/transactions-2019.csv, tool:docker"
    ),
    "hit": "The exact error from your smallest reproduction of the failure",
    "tried": (
        "What you ruled out and tried before asking: other variable names, "
        "config files, paths, packages, retries, alternative routes"
    ),
    "fix": (
        "What the person should do, copy-pasteable: the line to add to .env, "
        "the URL to accept terms at and the login command, the path to drop "
        "a file at"
    ),
    "next_steps": (
        "What you will do once this is met, in your own words — you will be "
        "resumed with this"
    ),
}


class InboxGate(ToolGate):
    """Gate for asking the person running the campaign."""

    name = "inbox"
    description = "Ask the person running the campaign for what only a person can provide"

    def get_tools(self) -> List["Tool"]:
        if not HAS_MCP:
            return []
        return [
            Tool(
                name=REQUEST_FROM_USER_TOOL,
                description=(
                    "Ask the person running this campaign for something only a "
                    "person can provide — a credential, a licence acceptance, "
                    "an access grant, a file, credits — when the solution "
                    "cannot be implemented without it. Calling this STOPS your "
                    "session: the campaign pauses until the person replies, "
                    "then this same session is resumed with their reply. Call "
                    "it only after you have reproduced the failure with the "
                    "smallest command that shows it and ruled out every cause "
                    "you could fix yourself; say what you tried. Put every "
                    "blocker you can see into one call. Never use it for "
                    "things you can do yourself (installs, downloads, retries, "
                    "design choices), and never fake the resource instead of "
                    "asking."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "requests": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    name: {
                                        "type": "string",
                                        "description": _FIELD_DESCRIPTIONS[name],
                                    }
                                    for name in REQUEST_FIELDS
                                },
                                "required": list(REQUEST_FIELDS),
                            },
                        }
                    },
                    "required": ["requests"],
                },
            )
        ]

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List["TextContent"]]:
        if not HAS_MCP or tool_name != REQUEST_FROM_USER_TOOL:
            return None
        entries = arguments.get("requests")
        results = file_requests(
            os.environ["KAPSO_INBOX_PATH"],
            node=int(os.environ["KAPSO_NODE_ID"]),
            session=os.environ["KAPSO_SESSION_ID"],
            entries=entries,
        )
        keys = [entry["key"] for entry in entries]
        return [TextContent(type="text", text=render_stop_text(results, keys))]
