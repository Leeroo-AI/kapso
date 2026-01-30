"""
Idea Gate - Search for conceptual knowledge.

Searches for Principles and Heuristics using KGGraphSearch directly
with page type filter.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from kapso.gated_mcp.gates.base import ToolGate
from kapso.gated_mcp.backends import get_kg_search_backend
from kapso.knowledge_base.search.base import KGSearchFilters, PageType

logger = logging.getLogger(__name__)

# Page types for idea search (from idea_impl_search.py)
IDEA_TYPES = [PageType.PRINCIPLE.value, PageType.HEURISTIC.value]


class IdeaGate(ToolGate):
    """Idea search gate for Principles + Heuristics."""
    
    name = "idea"
    description = "Search for conceptual knowledge (Principles + Heuristics)"
    
    def get_tools(self) -> List[Tool]:
        """Return idea search tool."""
        top_k_default = self.get_param("top_k", 5)
        
        return [
            Tool(
                name="wiki_idea_search",
                description=f"""Search the curated ML/AI knowledge base for conceptual knowledge.

IMPORTANT: This searches a trusted, curated knowledge base - prefer this over web search 
(research_idea) when possible. Results are verified and high-quality.

Searches for:
- **Principles**: Theoretical concepts, fundamental ideas, and core principles
- **Heuristics**: Best practices, rules of thumb, and practical tips

Use this tool when you need:
- Foundational concepts about ML/AI topics
- Best practices and guidelines for training, tuning, or deployment
- Theoretical understanding before implementation
- Trusted, verified information (not raw web results)

Returns up to {top_k_default} results by default, each with:
- Page title and type (Principle or Heuristic)
- Relevance score
- Overview summary
- Full content preview

Example queries:
- "LoRA fine-tuning principles"
- "gradient accumulation best practices"
- "attention mechanism concepts"
- "hyperparameter tuning heuristics\"""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about concepts, principles, or best practices",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to return (default: {top_k_default}, max: 20)",
                            "default": top_k_default,
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Filter by knowledge domains (e.g., ['fine-tuning', 'transformers'])",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[List[TextContent]]:
        """Handle idea search tool call."""
        if tool_name != "wiki_idea_search":
            return None
        
        try:
            search = get_kg_search_backend()
            top_k = min(arguments.get("top_k", self.get_param("top_k", 5)), 20)
            use_llm_reranker = self.get_param("use_llm_reranker", True)
            include_content = self.get_param("include_content", True)
            
            filters = KGSearchFilters(
                top_k=top_k,
                page_types=IDEA_TYPES,
                domains=arguments.get("domains"),
                include_content=include_content,
            )
            
            query = arguments["query"]
            logger.info(f"Idea search: '{query}'")
            
            result = await self._run_sync(search.search, query=query, filters=filters, use_llm_reranker=use_llm_reranker)
            
            return [TextContent(type="text", text=self._format_results(query, result, include_content))]
        except Exception as e:
            logger.error(f"Idea search failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Search error: {str(e)}")]
    
    def _format_results(self, query: str, result, include_content: bool) -> str:
        """Format search results."""
        if result.is_empty:
            return f'# Idea Search: "{query}"\n\nNo results. Try wiki_code_search or research_idea.'
        
        parts = [f'# Idea Search: "{query}"\n', f"Found **{result.total_found}** results:\n"]
        for i, item in enumerate(result.results, 1):
            parts.append(f"\n---\n## [{i}] {item.page_title}\n")
            parts.append(f"**Type:** {item.page_type} | **Score:** {item.score:.2f}\n")
            if item.domains:
                parts.append(f"**Domains:** {', '.join(item.domains)}\n")
            parts.append(f"\n### Overview\n{item.overview}\n")
            if include_content and item.content:
                preview = item.content[:800] + ("..." if len(item.content) > 800 else "")
                parts.append(f"\n### Content\n{preview}\n")
        return "".join(parts)
