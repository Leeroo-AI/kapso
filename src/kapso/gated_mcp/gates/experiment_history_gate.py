"""
Experiment History Gate
=======================

Provides tools for querying experiment history during ideation.

Tools:
- get_top_experiments: Get best experiments by score
- get_recent_experiments: Get most recent experiments
- search_similar_experiments: Semantic search for similar experiments
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
from kapso.execution.memories.experiment_memory.store import (
    format_experiments,
    load_store_from_env,
)

logger = logging.getLogger(__name__)


class ExperimentHistoryGate(ToolGate):
    """
    Gate for experiment history tools.
    
    Provides access to past experiment results for learning from
    previous attempts during ideation.
    
    Tools:
    - get_top_experiments: Get best experiments by score
    - get_recent_experiments: Get most recent experiments
    - search_similar_experiments: Semantic search for similar experiments
        """
    
    name = "experiment_history"
    description = "Tools for querying experiment history"
    
    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize experiment history gate."""
        super().__init__(config)
        self._store = None  # Lazy loaded
    
    def _get_store(self):
        """Lazy load the experiment history store."""
        if self._store is None:
            self._store = load_store_from_env()
        return self._store
    
    def get_tools(self) -> List["Tool"]:
        """Return experiment history tools."""
        if not HAS_MCP:
            return []
        
        return [
            Tool(
                name="get_top_experiments",
                description=(
                    "Get the top k experiments by score. "
                    "Returns experiments sorted by score (best first) with solution, score, feedback, and technical difficulties. "
                    "Use this to understand what approaches have worked best so far."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of top experiments to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="get_recent_experiments",
                description=(
                    "Get the most recent k experiments. "
                    "Returns experiments in chronological order with solution, score, feedback, and technical difficulties. "
                    "Use this to see what was tried recently and avoid repeating failures."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Number of recent experiments to return",
                            "default": 5,
                        },
                    },
                },
            ),
            Tool(
                name="search_similar_experiments",
                description=(
                    "Search for experiments similar to the given query. "
                    "Uses semantic search to find experiments with similar solutions, error patterns, or feedback. "
                    "Use this when you have a specific idea and want to see if something similar was tried before."
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
                            "description": "Number of similar experiments to return",
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
        
        if tool_name == "get_top_experiments":
            return await self._handle_get_top(arguments)
        elif tool_name == "get_recent_experiments":
            return await self._handle_get_recent(arguments)
        elif tool_name == "search_similar_experiments":
            return await self._handle_search_similar(arguments)
        
        return None
    
    async def _handle_get_top(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_top_experiments tool call."""
        k = arguments.get("k", self.get_param("top_k", 5))
        
        store = self._get_store()
        experiments = await self._run_sync(store.get_top_experiments, k)
        result = self._format_experiments(
            experiments,
            f"Top {k} Experiments by Objective-Normalized Utility",
        )
        return [TextContent(type="text", text=result)]
    
    async def _handle_get_recent(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_recent_experiments tool call."""
        k = arguments.get("k", self.get_param("recent_k", 5))
        
        store = self._get_store()
        experiments = await self._run_sync(store.get_recent_experiments, k)
        result = self._format_experiments(
            experiments,
            f"Most Recent {k} Experiments",
        )
        return [TextContent(type="text", text=result)]
    
    async def _handle_search_similar(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle search_similar_experiments tool call."""
        query = arguments.get("query", "")
        k = arguments.get("k", self.get_param("similar_k", 3))
        
        if not query:
            return [TextContent(type="text", text="Error: query is required")]
        
        store = self._get_store()
        experiments = await self._run_sync(store.search_similar, query, k)
        result = self._format_experiments(
            experiments,
            f"Experiments Similar to: {query}",
        )
        return [TextContent(type="text", text=result)]
    
    def _format_experiments(self, experiments, title: str) -> str:
        """Format experiments as markdown."""
        return f"# {title}\n\n{format_experiments(experiments)}"
    
