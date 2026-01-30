"""
Code Gate - Search for code knowledge.

Searches for Implementations and Environments using KGGraphSearch directly
with page type filter.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from kapso.gated_mcp.gates.base import ToolGate
from kapso.gated_mcp.backends import get_kg_search_backend
from kapso.knowledge_base.search.base import KGSearchFilters, PageType

logger = logging.getLogger(__name__)

# Page types for code search (from idea_impl_search.py)
CODE_TYPES = [PageType.IMPLEMENTATION.value, PageType.ENVIRONMENT.value]


class CodeGate(ToolGate):
    """Code search gate for Implementations + Environments."""
    
    name = "code"
    description = "Search for code knowledge (Implementations + Environments)"
    
    def get_tools(self) -> List[Tool]:
        """Return code search tool."""
        top_k_default = self.get_param("top_k", 5)
        
        return [
            Tool(
                name="wiki_code_search",
                description=f"""Search the curated ML/AI knowledge base for code and implementation knowledge.

IMPORTANT: This searches a trusted, curated knowledge base - prefer this over web search 
(research_implementation) when possible. Results are verified and high-quality.

Searches for:
- **Implementations**: Code patterns, API usage, algorithms, and working examples
- **Environments**: Setup guides, configuration, dependencies, and infrastructure

Use this tool when you need:
- Working code examples and patterns
- API documentation and usage guides
- Environment setup and configuration instructions
- Trusted, verified implementations (not raw web results)

Returns up to {top_k_default} results by default, each with:
- Page title and type (Implementation or Environment)
- Relevance score
- Overview summary
- Full content preview with code

Example queries:
- "PyTorch LoRA implementation"
- "HuggingFace trainer configuration"
- "CUDA environment setup"
- "distributed training code example\"""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about code, implementations, APIs, or setup",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to return (default: {top_k_default}, max: 20)",
                            "default": top_k_default,
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Filter by knowledge domains (e.g., ['pytorch', 'huggingface'])",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[List[TextContent]]:
        """Handle code search tool call."""
        if tool_name != "wiki_code_search":
            return None
        
        try:
            search = get_kg_search_backend()
            top_k = min(arguments.get("top_k", self.get_param("top_k", 5)), 20)
            use_llm_reranker = self.get_param("use_llm_reranker", True)
            include_content = self.get_param("include_content", True)
            
            filters = KGSearchFilters(
                top_k=top_k,
                page_types=CODE_TYPES,
                domains=arguments.get("domains"),
                include_content=include_content,
            )
            
            query = arguments["query"]
            logger.info(f"Code search: '{query}'")
            
            result = await self._run_sync(search.search, query=query, filters=filters, use_llm_reranker=use_llm_reranker)
            
            return [TextContent(type="text", text=self._format_results(query, result, include_content))]
        except Exception as e:
            logger.error(f"Code search failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Search error: {str(e)}")]
    
    def _format_results(self, query: str, result, include_content: bool) -> str:
        """Format search results."""
        if result.is_empty:
            return f'# Code Search: "{query}"\n\nNo results. Try wiki_idea_search or research_implementation.'
        
        parts = [f'# Code Search: "{query}"\n', f"Found **{result.total_found}** results:\n"]
        for i, item in enumerate(result.results, 1):
            parts.append(f"\n---\n## [{i}] {item.page_title}\n")
            parts.append(f"**Type:** {item.page_type} | **Score:** {item.score:.2f}\n")
            if item.domains:
                parts.append(f"**Domains:** {', '.join(item.domains)}\n")
            parts.append(f"\n### Overview\n{item.overview}\n")
            if include_content and item.content:
                preview = item.content[:1000] + ("..." if len(item.content) > 1000 else "")
                parts.append(f"\n### Content\n{preview}\n")
        return "".join(parts)
