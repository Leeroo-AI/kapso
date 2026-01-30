"""
Experiment History Gate
=======================

Provides tools for querying experiment history during ideation.

Tools:
- get_top_experiments: Get best experiments by score
- get_recent_experiments: Get most recent experiments
- search_similar_experiments: Semantic search for similar experiments
- get_insights: Get experiments with extracted insights
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
    Gate for experiment history tools.
    
    Provides access to past experiment results for learning from
    previous attempts during ideation.
    
    Tools:
    - get_top_experiments: Get best experiments by score
    - get_recent_experiments: Get most recent experiments
    - search_similar_experiments: Semantic search for similar experiments
    - get_insights: Get experiments with extracted insights
    """
    
    name = "experiment_history"
    description = "Tools for querying experiment history and insights"
    
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
                name="get_top_experiments",
                description=(
                    "Get the top k experiments by score. "
                    "Returns experiments sorted by score (best first) with solution, score, feedback, and insights. "
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
                    "Returns experiments in chronological order with solution, score, feedback, and insights. "
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
            Tool(
                name="get_insights",
                description=(
                    "Get experiments with extracted insights (generalized lessons). "
                    "Insights are LLM-extracted lessons from errors (critical_error) or successes (best_practice). "
                    "Use this to learn from past mistakes and successful patterns."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "k": {
                            "type": "integer",
                            "description": "Maximum number of insights to return",
                            "default": 10,
                        },
                        "insight_type": {
                            "type": "string",
                            "description": "Filter by insight type: 'critical_error' or 'best_practice'",
                            "enum": ["critical_error", "best_practice"],
                        },
                    },
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
        elif tool_name == "get_insights":
            return await self._handle_get_insights(arguments)
        
        return None
    
    async def _handle_get_top(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_top_experiments tool call."""
        k = arguments.get("k", self.get_param("top_k", 5))
        
        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_top_experiments, k)
            result = self._format_experiments(experiments, f"Top {k} Experiments by Score")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"get_top_experiments failed: {e}")
            return [TextContent(type="text", text=f"Error getting top experiments: {e}")]
    
    async def _handle_get_recent(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_recent_experiments tool call."""
        k = arguments.get("k", self.get_param("recent_k", 5))
        
        try:
            store = self._get_store()
            experiments = await self._run_sync(store.get_recent_experiments, k)
            result = self._format_experiments(experiments, f"Most Recent {k} Experiments")
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"get_recent_experiments failed: {e}")
            return [TextContent(type="text", text=f"Error getting recent experiments: {e}")]
    
    async def _handle_search_similar(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle search_similar_experiments tool call."""
        query = arguments.get("query", "")
        k = arguments.get("k", self.get_param("similar_k", 3))
        
        if not query:
            return [TextContent(type="text", text="Error: query is required")]
        
        try:
            store = self._get_store()
            experiments = await self._run_sync(store.search_similar, query, k)
            result = self._format_experiments(
                experiments, 
                f"Experiments Similar to: {query[:50]}{'...' if len(query) > 50 else ''}"
            )
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"search_similar_experiments failed: {e}")
            return [TextContent(type="text", text=f"Error searching experiments: {e}")]
    
    async def _handle_get_insights(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_insights tool call."""
        k = arguments.get("k", 10)
        insight_type = arguments.get("insight_type")
        
        try:
            store = self._get_store()
            experiments = await self._run_sync(
                store.get_experiments_with_insights, 
                k, 
                insight_type
            )
            result = self._format_insights(experiments, insight_type)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"get_insights failed: {e}")
            return [TextContent(type="text", text=f"Error getting insights: {e}")]
    
    def _format_experiments(self, experiments, title: str) -> str:
        """Format experiments as markdown."""
        if not experiments:
            return f"# {title}\n\nNo experiments found."
        
        lines = [f"# {title}\n"]
        
        for exp in experiments:
            if exp.had_error:
                status = f"FAILED: {exp.error_message[:100]}"
            else:
                status = f"score={exp.score}"
            
            lines.append(f"""
## Experiment {exp.node_id} ({status})

**Solution:**
{exp.solution[:500]}{'...' if len(exp.solution) > 500 else ''}

**Feedback:**
{exp.feedback[:300]}{'...' if len(exp.feedback) > 300 else ''}""")
            
            # Include insight if available
            if exp.insight:
                conf = exp.insight_confidence or 0
                lines.append(f"""
**Insight ({exp.insight_type}, confidence={conf:.2f}):**
{exp.insight}""")
        
        return "\n".join(lines)
    
    def _format_insights(self, experiments, insight_type: Optional[str]) -> str:
        """Format insights as markdown."""
        type_filter = f" ({insight_type})" if insight_type else ""
        title = f"Extracted Insights{type_filter}"
        
        if not experiments:
            return f"# {title}\n\nNo insights found."
        
        lines = [f"# {title}\n"]
        lines.append(f"Found {len(experiments)} experiments with insights.\n")
        
        for exp in experiments:
            if not exp.insight:
                continue
            
            conf = exp.insight_confidence or 0
            exp_status = f"score={exp.score}" if not exp.had_error else "FAILED"
            
            lines.append(f"""
## Insight from Experiment {exp.node_id} ({exp_status})

**Type:** {exp.insight_type} | **Confidence:** {conf:.2f}

{exp.insight}

**Tags:** {', '.join(exp.insight_tags) if exp.insight_tags else 'none'}
""")
        
        return "\n".join(lines)
