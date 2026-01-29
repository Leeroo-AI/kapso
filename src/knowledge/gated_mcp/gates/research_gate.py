"""
Research Gate - Deep web research using OpenAI's web_search.

Provides tools for research_idea, research_implementation, and research_study.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from src.knowledge.gated_mcp.gates.base import ToolGate
from src.knowledge.gated_mcp.backends import get_researcher_backend

logger = logging.getLogger(__name__)


class ResearchGate(ToolGate):
    """Research gate for deep web research using OpenAI's web_search."""
    
    name = "research"
    description = "Deep web research using OpenAI's web_search"
    
    def get_tools(self) -> List[Tool]:
        """Return research tools."""
        default_top_k = self.get_param("default_top_k", 5)
        default_depth = self.get_param("default_depth", "deep")
        
        return [
            Tool(
                name="research_idea",
                description=f"Research conceptual ideas from web. Returns up to {default_top_k} ideas.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to research"},
                        "top_k": {"type": "integer", "default": default_top_k},
                        "depth": {"type": "string", "enum": ["light", "deep"], "default": default_depth},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="research_implementation",
                description=f"Research code implementations from web. Returns up to {default_top_k} examples.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What implementation to research"},
                        "top_k": {"type": "integer", "default": default_top_k},
                        "depth": {"type": "string", "enum": ["light", "deep"], "default": default_depth},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="research_study",
                description="Generate comprehensive research report on a topic.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Research topic"},
                        "depth": {"type": "string", "enum": ["light", "deep"], "default": "deep"},
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[List[TextContent]]:
        """Handle research tool calls."""
        if tool_name == "research_idea":
            return await self._handle_idea(arguments)
        elif tool_name == "research_implementation":
            return await self._handle_implementation(arguments)
        elif tool_name == "research_study":
            return await self._handle_study(arguments)
        return None
    
    async def _handle_idea(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle research_idea."""
        try:
            researcher = get_researcher_backend()
            query = arguments["query"]
            top_k = arguments.get("top_k", self.get_param("default_top_k", 5))
            depth = arguments.get("depth", self.get_param("default_depth", "deep"))
            
            logger.info(f"Research idea: '{query}' (top_k={top_k}, depth={depth})")
            ideas = await self._run_sync(
                researcher.research,
                query,
                mode="idea",
                top_k=top_k,
                depth=depth,
            )
            
            if not ideas:
                return [TextContent(type="text", text=f'# Research Ideas: "{query}"\n\nNo results found.')]
            
            # Source.Idea has: query, source (url), content
            parts = [f'# Research Ideas: "{query}"\n\nFound **{len(ideas)}** ideas:\n']
            for i, idea in enumerate(ideas, 1):
                parts.append(f"\n---\n## [{i}] Idea from: {idea.source}\n\n")
                parts.append(f"{idea.content}\n")
            return [TextContent(type="text", text="".join(parts))]
        except Exception as e:
            logger.error(f"Research idea failed: {e}", exc_info=True)
            import traceback
            error_details = traceback.format_exc()
            return [TextContent(type="text", text=f"Research error: {str(e)}\n\nDetails:\n```\n{error_details}\n```")]
    
    async def _handle_implementation(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle research_implementation."""
        try:
            researcher = get_researcher_backend()
            query = arguments["query"]
            top_k = arguments.get("top_k", self.get_param("default_top_k", 5))
            depth = arguments.get("depth", self.get_param("default_depth", "deep"))
            
            logger.info(f"Research implementation: '{query}'")
            impls = await self._run_sync(
                researcher.research,
                query,
                mode="implementation",
                top_k=top_k,
                depth=depth,
            )
            
            if not impls:
                return [TextContent(type="text", text=f'# Research Implementations: "{query}"\n\nNo results found.')]
            
            # Source.Implementation has: query, source (url), content
            parts = [f'# Research Implementations: "{query}"\n\nFound **{len(impls)}** implementations:\n']
            for i, impl in enumerate(impls, 1):
                parts.append(f"\n---\n## [{i}] Implementation from: {impl.source}\n\n")
                parts.append(f"{impl.content}\n")
            return [TextContent(type="text", text="".join(parts))]
        except Exception as e:
            logger.error(f"Research implementation failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Research error: {str(e)}")]
    
    async def _handle_study(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle research_study."""
        try:
            researcher = get_researcher_backend()
            query = arguments["query"]
            depth = arguments.get("depth", "deep")
            
            logger.info(f"Research study: '{query}'")
            report = await self._run_sync(researcher.research, query, mode="study", depth=depth)
            
            text = report.to_string() if report else f"No report generated for: {query}"
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.error(f"Research study failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Research error: {str(e)}")]
