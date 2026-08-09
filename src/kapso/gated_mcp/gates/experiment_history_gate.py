"""
Own-Run Attempt Memory Gate
===========================

Provides tools for the agent to consult its OWN prior attempts from earlier in
the same run during ideation. This is the agent's private within-run working
memory of the iterations it has already run itself this session — not external
data and not other runs (each run starts with an empty store).

Tools:
- list_my_best_attempts: The agent's own best-scoring attempts this run
- list_my_recent_attempts: The agent's own most recent attempts this run
- search_my_attempts: Search the agent's own attempts this run
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
    Gate for the agent's own within-run attempt memory.

    Provides access to the agent's OWN prior attempt results from earlier in the
    same run, for learning from its own iterations during ideation. The backing
    store starts empty each run, so it only ever holds this run's own attempts.

    Tools:
    - list_my_best_attempts: The agent's own best-scoring attempts this run
    - list_my_recent_attempts: The agent's own most recent attempts this run
    - search_my_attempts: Search the agent's own attempts this run
        """

    name = "experiment_history"
    description = "Tools for consulting the agent's own prior attempts this run"
    
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
                name="list_my_best_attempts",
                description=(
                    "List the best-scoring attempts YOU have run so far in THIS session. "
                    "This is your own private scratchpad of your own iterations this run: "
                    "it starts empty when the run begins and only ever records work you do "
                    "here, so it holds nothing but your own attempts. Returns each of your "
                    "attempts with the approach you took, the dev score you got, and your "
                    "own notes. Use it to build on what has worked for you so far."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of your own attempts to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="list_my_recent_attempts",
                description=(
                    "List the most recent attempts YOU have run so far in THIS session "
                    "(your own private scratchpad of your own iterations this run, which "
                    "started empty and only records work you do here). Returns your own "
                    "attempts in the order you ran them, with the approach you took, the "
                    "dev score you got, and your own notes. Use it to see what you just "
                    "tried and avoid repeating your own dead ends."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of your own recent attempts to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="search_my_attempts",
                description=(
                    "Search the attempts YOU have run so far in THIS session (your own "
                    "private scratchpad of your own iterations this run, which started empty "
                    "and only records work you do here) for ones like a query. Use it to "
                    "check whether you have already tried an idea yourself this session."
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
                            "description": "Number of your own attempts to return",
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
        
        if tool_name == "list_my_best_attempts":
            return await self._handle_get_top(arguments)
        elif tool_name == "list_my_recent_attempts":
            return await self._handle_get_recent(arguments)
        elif tool_name == "search_my_attempts":
            return await self._handle_search_similar(arguments)

        return None
    
    async def _handle_get_top(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle list_my_best_attempts tool call."""
        k = arguments.get("k", self.get_param("top_k", 5))

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_top_experiments, k)
            result = self._format_experiments(experiments, f"Your attempts this session (strongest first, up to {k})")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"list_my_best_attempts failed: {e}")
            return [TextContent(type="text", text=f"Error getting your best attempts: {e}")]

    async def _handle_get_recent(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle list_my_recent_attempts tool call."""
        k = arguments.get("k", self.get_param("recent_k", 5))

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_recent_experiments, k)
            result = self._format_experiments(experiments, f"Your attempts this session (most recent, up to {k})")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"list_my_recent_attempts failed: {e}")
            return [TextContent(type="text", text=f"Error getting your recent attempts: {e}")]

    async def _handle_search_similar(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle search_my_attempts tool call."""
        query = arguments.get("query", "")
        k = arguments.get("k", self.get_param("similar_k", 3))

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        try:
            store = self._get_store()
            experiments = await self._run_sync(store.search_similar, query, k)
            result = self._format_experiments(
                experiments,
                f"Your attempts this session (matching: {query[:50]}{'...' if len(query) > 50 else ''})"
            )
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"search_my_attempts failed: {e}")
            return [TextContent(type="text", text=f"Error searching experiments: {e}")]
    
    def _format_experiments(self, experiments, title: str) -> str:
        """Format experiments as markdown."""
        if not experiments:
            return f"# {title}\n\nYou have not run any attempts yet this session."

        lines = [
            f"# {title}\n",
            "_Everything below is your own work from this session — the attempts "
            "you yourself ran here, with the notes you wrote after each one, "
            "played back to you now. This log started empty when the session "
            "began and only ever records what you do here: these are your own "
            "earlier notes, not external data and not from any other run. "
            "Reading it is simply reviewing your own prior notes from this "
            "session._\n",
        ]

        for exp in experiments:
            eval_note = "" if exp.evaluation_valid else " (this attempt could not be evaluated)"

            # Full content, never clipped: these renders ARE model input
            # (ideation reads them through the MCP tools), and there is no
            # drill-down tool to recover cut text.
            lines.append(f"""
## Your iteration {exp.node_id} this session — what you tried{eval_note}

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
    
