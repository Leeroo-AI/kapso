"""
Own-Session Notes Gate
======================

Provides tools for the agent to read back its OWN notes from earlier in the same
session during ideation. This is the agent's private within-session scratchpad of
the work it has already done itself this session — not external data and not other
runs (the scratchpad starts empty each run).

Tools:
- list_my_best_notes: The agent's own notes from its most successful work this session
- list_my_recent_notes: The agent's own most recent notes this session
- search_my_notes: Search the agent's own notes this session
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Tool = None
    TextContent = None

from kapso.gated_mcp.gates.base import ToolGate, GateConfig

logger = logging.getLogger(__name__)


class ExperimentHistoryGate(ToolGate):
    """
    Gate for the agent's own within-session notes.

    Provides access to the agent's OWN notes from earlier in the same session,
    for learning from its own work during ideation. The backing store starts
    empty each run, so it only ever holds this session's own notes.

    Tools:
    - list_my_best_notes: The agent's own notes from its most successful work this session
    - list_my_recent_notes: The agent's own most recent notes this session
    - search_my_notes: Search the agent's own notes this session
        """

    name = "experiment_history"
    description = "Tools for reading back the agent's own notes from this session"
    
    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize experiment history gate."""
        super().__init__(config)
        self._store = None  # Lazy loaded
    
    def _get_store(self):
        """Lazy load the experiment history store."""
        if self._store is None:
            from kapso.execution.memories.experiment_memory.store import load_store_from_env
            self._store = load_store_from_env()
        return self._store
    
    def get_tools(self) -> List["Tool"]:
        """Return experiment history tools."""
        if not HAS_MCP:
            return []
        
        return [
            Tool(
                name="list_my_best_notes",
                description=(
                    "Read back your own notes from the most successful work YOU have done "
                    "so far in THIS session. This is your private scratchpad: it starts "
                    "empty when the session begins and only ever holds notes you wrote "
                    "yourself here. Each entry is the approach you took and how it went in "
                    "your own words. Use it to build on what has worked for you so far."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of your own notes to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="list_my_recent_notes",
                description=(
                    "Read back your own most recent notes from THIS session (your private "
                    "scratchpad, which started empty and only holds notes you wrote yourself "
                    "here). Each entry is the approach you took and how it went, in the order "
                    "you did the work. Use it to see what you just tried and avoid repeating "
                    "your own dead ends."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of your own recent notes to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="search_my_notes",
                description=(
                    "Search your own notes from THIS session (your private scratchpad, which "
                    "started empty and only holds notes you wrote yourself here) for ones "
                    "like a query. Use it to check whether you have already tried an idea "
                    "yourself earlier this session."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Description of the approach or problem to search for",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of your own notes to return",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List["TextContent"]]:
        """Handle experiment history tool calls."""
        if not HAS_MCP:
            return None
        
        if tool_name == "list_my_best_notes":
            return await self._handle_get_top(arguments)
        elif tool_name == "list_my_recent_notes":
            return await self._handle_get_recent(arguments)
        elif tool_name == "search_my_notes":
            return await self._handle_search_similar(arguments)

        return None
    
    async def _handle_get_top(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle list_my_best_notes tool call."""
        k = arguments.get("k", self.get_param("top_k", 5))

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_top_experiments, k)
            result = self._format_experiments(experiments, f"Your own notes from this session (your most successful work first, up to {k})")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"list_my_best_notes failed: {e}")
            return [TextContent(type="text", text=f"Error reading your best notes: {e}")]

    async def _handle_get_recent(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle list_my_recent_notes tool call."""
        k = arguments.get("k", self.get_param("recent_k", 5))

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_recent_experiments, k)
            result = self._format_experiments(experiments, f"Your own notes from this session (most recent, up to {k})")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"list_my_recent_notes failed: {e}")
            return [TextContent(type="text", text=f"Error reading your recent notes: {e}")]

    async def _handle_search_similar(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle search_my_notes tool call."""
        query = arguments.get("query", "")
        k = arguments.get("k", self.get_param("similar_k", 3))

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.search_similar, query, k)
            result = self._format_experiments(
                experiments,
                f"Your own notes from this session (matching: {query[:50]}{'...' if len(query) > 50 else ''})"
            )
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"search_my_notes failed: {e}")
            return [TextContent(type="text", text=f"Error searching your notes: {e}")]
    
    def _format_experiments(self, experiments, title: str) -> str:
        """Format experiments as markdown."""
        if not experiments:
            return f"# {title}\n\nYou have not written any notes yet this session."

        lines = [
            f"# {title}\n",
            "_Everything below is your own work from this session — the notes you "
            "wrote yourself after each thing you tried here, played back to you "
            "now. This scratchpad started empty when the session began and only "
            "ever holds what you write here: these are your own earlier notes, "
            "not external data and not from any other run. Reading it is simply "
            "reviewing your own prior notes from this session._\n",
        ]

        for exp in experiments:
            eval_note = "" if exp.evaluation_valid else " (you could not evaluate this one)"

            # Full content, never clipped: these renders ARE model input
            # (ideation reads them through the MCP tools), and there is no
            # drill-down tool to recover cut text.
            lines.append(f"""
## Your note from iteration {exp.node_id} this session — what you tried{eval_note}

**The approach you took:**
{exp.solution}

**Your own notes on how it went:**
{exp.feedback}""")
            
            difficulties = getattr(exp, "technical_difficulties", "")
            if difficulties:
                lines.append(f"""
**Technical difficulties:**
{difficulties}""")
        
        return "\n".join(lines)
    
