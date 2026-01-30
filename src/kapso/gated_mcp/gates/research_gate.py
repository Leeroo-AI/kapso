"""
Research Gate - Deep web research using OpenAI's web_search.

Provides tools for research_idea, research_implementation, and research_study.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from kapso.gated_mcp.gates.base import ToolGate
from kapso.gated_mcp.backends import get_researcher_backend

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
                description=f"""Search the web for conceptual ideas, principles, and best practices.

Use this tool when you need to find:
- Theoretical concepts and principles about ML/AI topics
- Best practices and heuristics from the community
- Design patterns and architectural approaches
- Research findings and academic insights

Returns up to {default_top_k} ideas by default, each with source URL and content.

Example queries:
- "LoRA fine-tuning best practices"
- "transformer attention mechanisms explained"
- "how to prevent overfitting in deep learning"

Depth options:
- "light": Quick search, fewer sources (faster)
- "deep": Thorough search, more sources (slower but more comprehensive)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about concepts, principles, or best practices",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of ideas to return (default: {default_top_k})",
                            "default": default_top_k,
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["light", "deep"],
                            "description": f"Search depth - 'light' for quick results, 'deep' for thorough research (default: {default_depth})",
                            "default": default_depth,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="research_implementation",
                description=f"""Search the web for code implementations, examples, and technical guides.

Use this tool when you need to find:
- Code examples and implementation patterns
- API usage and library documentation
- Step-by-step technical tutorials
- Configuration and setup guides

Returns up to {default_top_k} implementations by default, each with source URL and code/content.

Example queries:
- "PyTorch LoRA implementation example"
- "how to use HuggingFace transformers for text classification"
- "CUDA memory optimization techniques code"

Depth options:
- "light": Quick search, fewer sources (faster)
- "deep": Thorough search, more sources (slower but more comprehensive)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about code, implementations, or technical how-tos",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of implementations to return (default: {default_top_k})",
                            "default": default_top_k,
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["light", "deep"],
                            "description": f"Search depth - 'light' for quick results, 'deep' for thorough research (default: {default_depth})",
                            "default": default_depth,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="research_study",
                description="""Generate a comprehensive research report on a topic.

Use this tool when you need:
- A thorough overview of a complex topic
- Multiple perspectives and sources synthesized together
- Background research before starting a project
- Understanding of the state-of-the-art in a field

Returns a structured report with sections covering different aspects of the topic.
This is more comprehensive than research_idea or research_implementation but takes longer.

Example queries:
- "state of the art in efficient fine-tuning methods"
- "comparison of vector databases for ML applications"
- "best practices for training large language models"

Depth options:
- "light": Faster report with fewer sources
- "deep": Comprehensive report with extensive research (recommended)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Research topic to investigate comprehensively",
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["light", "deep"],
                            "description": "Research depth - 'deep' recommended for thorough reports (default: deep)",
                            "default": "deep",
                        },
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
