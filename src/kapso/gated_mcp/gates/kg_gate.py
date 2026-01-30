"""
KG Gate - Full Knowledge Graph operations.

Provides tools for:
- search_knowledge: Semantic search with filters
- get_wiki_page: Retrieve page by title
- kg_index: Index pages into KG
- kg_edit: Edit existing pages
- get_page_structure: Get section definitions
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from kapso.gated_mcp.gates.base import ToolGate
from kapso.gated_mcp.backends import get_kg_search_backend, get_index_data_source
from kapso.knowledge_base.search.base import (
    KGSearchFilters,
    KGIndexInput,
    KGIndexMetadata,
    KGEditInput,
    WikiPage,
)

logger = logging.getLogger(__name__)


class KGGate(ToolGate):
    """
    Full Knowledge Graph operations gate.
    
    Provides complete access to KG search, indexing, and editing.
    Used by KnowledgeMerger and admin tools.
    """
    
    name = "kg"
    description = "Full Knowledge Graph operations"
    
    def get_tools(self) -> List[Tool]:
        """Return KG tools."""
        return [
            Tool(
                name="search_knowledge",
                description="""Search the ML/AI knowledge base for relevant wiki pages.

Use this tool when you need to find:
- How-to guides and workflows for ML tasks
- Best practices and heuristics for training models
- Implementation details and code patterns
- Theoretical concepts and principles
- Environment setup and configuration guides

The search uses semantic embeddings + LLM reranking for high accuracy.
Results include page title, type, relevance score, overview, and content preview.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5, max: 20)",
                            "default": 5,
                        },
                        "page_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by page types: Workflow, Principle, Implementation, Environment, Heuristic",
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by knowledge domains",
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum relevance score threshold (0.0 to 1.0)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_wiki_page",
                description="""Retrieve a specific wiki page by its exact title.

Use this when you already know the page title (from a previous search)
and want to get the complete content.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "Exact title of the wiki page",
                        },
                    },
                    "required": ["page_title"],
                },
            ),
            Tool(
                name="kg_index",
                description="""Index wiki pages into the knowledge graph.

Supports two modes:
1. Directory mode: Index all .md files from a wiki directory
2. Single page mode: Add or update a single page""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wiki_dir": {
                            "type": "string",
                            "description": "Path to wiki directory",
                        },
                        "page_data": {
                            "type": "object",
                            "description": "Single page to add/update",
                            "properties": {
                                "page_title": {"type": "string"},
                                "page_type": {
                                    "type": "string",
                                    "enum": ["Workflow", "Principle", "Implementation", "Environment", "Heuristic"],
                                },
                                "overview": {"type": "string"},
                                "content": {"type": "string"},
                                "domains": {"type": "array", "items": {"type": "string"}},
                                "sources": {"type": "array"},
                                "outgoing_links": {"type": "array"},
                            },
                            "required": ["page_title", "page_type", "overview", "content"],
                        },
                        "persist_path": {
                            "type": "string",
                            "description": "Path to JSON cache file",
                        },
                        "clear_existing": {
                            "type": "boolean",
                            "description": "Clear existing data before indexing",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="kg_edit",
                description="""Edit an existing wiki page in the knowledge graph.

Updates the page across all storage layers. Only include fields you want to update.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID (format: 'Type/Title')",
                        },
                        "updates": {
                            "type": "object",
                            "description": "Fields to update",
                            "properties": {
                                "overview": {"type": "string"},
                                "content": {"type": "string"},
                                "domains": {"type": "array", "items": {"type": "string"}},
                                "sources": {"type": "array"},
                                "outgoing_links": {"type": "array"},
                            },
                        },
                        "wiki_dir": {
                            "type": "string",
                            "description": "Wiki directory path (for source file update)",
                        },
                        "auto_timestamp": {
                            "type": "boolean",
                            "description": "Auto-update last_updated field",
                            "default": True,
                        },
                    },
                    "required": ["page_id", "updates"],
                },
            ),
            Tool(
                name="get_page_structure",
                description="""Get the section structure definition for a specific page type.

IMPORTANT: Use this BEFORE creating or editing a page to ensure you follow
the correct structure.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page_type": {
                            "type": "string",
                            "description": "Page type: principle, implementation, environment, heuristic, or workflow",
                            "enum": ["principle", "implementation", "environment", "heuristic", "workflow"],
                        },
                    },
                    "required": ["page_type"],
                },
            ),
        ]
    
    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List[TextContent]]:
        """Handle KG tool calls."""
        
        if tool_name == "search_knowledge":
            return await self._handle_search(arguments)
        elif tool_name == "get_wiki_page":
            return await self._handle_get_page(arguments)
        elif tool_name == "kg_index":
            return await self._handle_index(arguments)
        elif tool_name == "kg_edit":
            return await self._handle_edit(arguments)
        elif tool_name == "get_page_structure":
            return await self._handle_get_structure(arguments)
        
        return None
    
    async def _handle_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle search_knowledge tool call."""
        try:
            search = get_kg_search_backend()
            
            top_k = min(arguments.get("top_k", 5), 20)
            include_content = self.get_param("include_content", True)
            
            filters = KGSearchFilters(
                top_k=top_k,
                min_score=arguments.get("min_score"),
                page_types=arguments.get("page_types"),
                domains=arguments.get("domains"),
                include_content=include_content,
            )
            
            query = arguments["query"]
            logger.info(f"KG search: '{query}' with filters: {filters}")
            
            result = await self._run_sync(search.search, query=query, filters=filters)
            
            return [TextContent(
                type="text",
                text=self._format_search_results(query, result, include_content),
            )]
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Search error: {str(e)}")]
    
    async def _handle_get_page(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle get_wiki_page tool call."""
        try:
            search = get_kg_search_backend()
            page_title = arguments["page_title"]
            
            logger.info(f"Getting page: '{page_title}'")
            page = await self._run_sync(search.get_page, page_title)
            
            if page is None:
                return [TextContent(
                    type="text",
                    text=f"Page not found: '{page_title}'\n\nTip: Use search_knowledge to find pages by topic.",
                )]
            
            return [TextContent(
                type="text",
                text=self._format_page(page),
            )]
            
        except Exception as e:
            logger.error(f"Get page failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error retrieving page: {str(e)}")]
    
    async def _handle_index(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle kg_index tool call."""
        try:
            search = get_kg_search_backend()
            
            wiki_dir = arguments.get("wiki_dir")
            page_data = arguments.get("page_data")
            persist_path = arguments.get("persist_path")
            clear_existing = arguments.get("clear_existing", False)
            
            if clear_existing:
                logger.info("Clearing existing index data...")
                await self._run_sync(search.clear)
            
            # Directory mode
            if wiki_dir and not page_data:
                wiki_path = Path(wiki_dir)
                if not wiki_path.exists():
                    return [TextContent(
                        type="text",
                        text=f"Error: Wiki directory not found: {wiki_dir}",
                    )]
                
                logger.info(f"Indexing pages from directory: {wiki_dir}")
                
                index_input = KGIndexInput(
                    wiki_dir=wiki_path,
                    persist_path=Path(persist_path) if persist_path else None,
                )
                await self._run_sync(search.index, index_input)
                
                return [TextContent(
                    type="text",
                    text=f"Successfully indexed pages from: {wiki_dir}",
                )]
            
            # Single page mode
            elif page_data:
                page_title = page_data.get("page_title", "")
                page_type = page_data.get("page_type", "")
                overview = page_data.get("overview", "")
                content = page_data.get("content", "")
                domains = page_data.get("domains", [])
                sources = page_data.get("sources", [])
                outgoing_links = page_data.get("outgoing_links", [])
                
                if not all([page_title, page_type, overview, content]):
                    return [TextContent(
                        type="text",
                        text="Error: page_data requires page_title, page_type, overview, and content",
                    )]
                
                # Determine wiki_dir
                if not wiki_dir:
                    wiki_dir = get_index_data_source() or "data/wikis"
                
                wiki_path = Path(wiki_dir)
                page_id = f"{page_type}/{page_title}"
                
                page = WikiPage(
                    id=page_id,
                    page_title=page_title,
                    page_type=page_type,
                    overview=overview,
                    content=content,
                    domains=domains,
                    sources=sources,
                    outgoing_links=outgoing_links,
                )
                
                # Check if page exists
                page_exists = False
                if hasattr(search, 'page_exists'):
                    page_exists = await self._run_sync(search.page_exists, page_id)
                
                if page_exists:
                    # Update existing
                    edit_input = KGEditInput(
                        page_id=page_id,
                        overview=overview,
                        content=content,
                        domains=domains,
                        sources=sources if sources else None,
                        outgoing_links=outgoing_links if outgoing_links else None,
                        wiki_dir=wiki_path,
                        persist_path=Path(persist_path) if persist_path else None,
                    )
                    success = await self._run_sync(search.edit, edit_input)
                    
                    if success:
                        return [TextContent(
                            type="text",
                            text=f"Successfully updated existing page: {page_id}",
                        )]
                    else:
                        return [TextContent(type="text", text=f"Failed to update page: {page_id}")]
                else:
                    # Add new
                    persist = Path(persist_path) if persist_path else None
                    
                    if hasattr(search, 'add_page'):
                        success = await self._run_sync(search.add_page, page, wiki_path, persist)
                    else:
                        index_input = KGIndexInput(pages=[page])
                        await self._run_sync(search.index, index_input)
                        success = True
                    
                    if success:
                        return [TextContent(
                            type="text",
                            text=f"Successfully added new page: {page_id}",
                        )]
                    else:
                        return [TextContent(type="text", text=f"Failed to add page: {page_id}")]
            
            else:
                return [TextContent(
                    type="text",
                    text="Error: Must provide either 'wiki_dir' or 'page_data'",
                )]
            
        except Exception as e:
            logger.error(f"Index failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Index error: {str(e)}")]
    
    async def _handle_edit(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle kg_edit tool call."""
        try:
            search = get_kg_search_backend()
            
            page_id = arguments.get("page_id")
            updates = arguments.get("updates", {})
            wiki_dir = arguments.get("wiki_dir")
            auto_timestamp = arguments.get("auto_timestamp", True)
            
            if not page_id:
                return [TextContent(type="text", text="Error: 'page_id' is required")]
            
            if not updates:
                return [TextContent(
                    type="text",
                    text="Error: 'updates' must contain at least one field to update",
                )]
            
            logger.info(f"Editing page: {page_id}, fields: {list(updates.keys())}")
            
            edit_input = KGEditInput(
                page_id=page_id,
                wiki_dir=Path(wiki_dir) if wiki_dir else None,
                auto_timestamp=auto_timestamp,
                overview=updates.get("overview"),
                content=updates.get("content"),
                domains=updates.get("domains"),
                sources=updates.get("sources"),
                outgoing_links=updates.get("outgoing_links"),
            )
            
            success = await self._run_sync(search.edit, edit_input)
            
            if success:
                fields_updated = list(updates.keys())
                return [TextContent(
                    type="text",
                    text=f"Successfully edited page: {page_id}\n\nFields updated: {', '.join(fields_updated)}",
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Edit failed: Page '{page_id}' not found or update failed",
                )]
            
        except Exception as e:
            logger.error(f"Edit failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Edit error: {str(e)}")]
    
    async def _handle_get_structure(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle get_page_structure tool call."""
        try:
            page_type = arguments.get("page_type", "").lower()
            valid_types = ["principle", "implementation", "environment", "heuristic", "workflow"]
            
            if page_type not in valid_types:
                return [TextContent(
                    type="text",
                    text=f"Invalid page type: '{page_type}'. Must be one of: {', '.join(valid_types)}",
                )]
            
            # Path to wiki_structure directory
            wiki_structure_dir = Path(__file__).parent.parent.parent / "knowledge_base" / "wiki_structure"
            sections_file = wiki_structure_dir / f"{page_type}_page" / "sections_definition.md"
            
            if not sections_file.exists():
                return [TextContent(
                    type="text",
                    text=f"Sections definition not found for page type: {page_type}",
                )]
            
            content = sections_file.read_text(encoding="utf-8")
            
            return [TextContent(
                type="text",
                text=f"# Page Structure Definition: {page_type.title()}\n\n{content}",
            )]
            
        except Exception as e:
            logger.error(f"Get page structure failed: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _format_search_results(self, query: str, result, include_content: bool) -> str:
        """Format search results as markdown."""
        if result.is_empty:
            return f"""# Search Results for: "{query}"

No relevant knowledge found.

**Suggestions:**
- Try broader search terms
- Remove filters to get more results
- Check spelling of technical terms
"""
        
        parts = [
            f"# Search Results for: \"{query}\"\n",
            f"Found **{result.total_found}** relevant pages:\n",
        ]
        
        for i, item in enumerate(result.results, 1):
            parts.append(f"\n---\n")
            parts.append(f"## [{i}] {item.page_title}\n")
            parts.append(f"**Type:** {item.page_type} | **Score:** {item.score:.2f}\n")
            
            if item.domains:
                parts.append(f"**Domains:** {', '.join(item.domains)}\n")
            
            parts.append(f"\n### Overview\n{item.overview}\n")
            
            if include_content and item.content:
                content_preview = item.content[:800].strip()
                if len(item.content) > 800:
                    content_preview += "\n\n... [truncated - use `get_wiki_page` for full content]"
                parts.append(f"\n### Content Preview\n{content_preview}\n")
        
        return "".join(parts)
    
    def _format_page(self, page: WikiPage) -> str:
        """Format a wiki page as markdown."""
        parts = [
            f"# {page.page_title}\n",
            f"**Type:** {page.page_type}\n",
        ]
        
        if page.domains:
            parts.append(f"**Domains:** {', '.join(page.domains)}\n")
        
        if page.last_updated:
            parts.append(f"**Last Updated:** {page.last_updated}\n")
        
        parts.append(f"\n---\n")
        parts.append(f"\n## Overview\n{page.overview}\n")
        parts.append(f"\n## Content\n{page.content}\n")
        
        if page.sources:
            parts.append("\n## Sources\n")
            for src in page.sources:
                src_type = src.get('type', 'Link')
                src_title = src.get('title', 'Reference')
                src_url = src.get('url', '')
                if src_url:
                    parts.append(f"- **{src_type}:** [{src_title}]({src_url})\n")
                else:
                    parts.append(f"- **{src_type}:** {src_title}\n")
        
        if page.outgoing_links:
            parts.append("\n## Related Pages\n")
            for link in page.outgoing_links[:10]:
                edge_type = link.get('edge_type', 'related')
                target = link.get('target_id', '')
                target_type = link.get('target_type', '')
                parts.append(f"- {edge_type} → {target_type}: {target}\n")
        
        return "".join(parts)
