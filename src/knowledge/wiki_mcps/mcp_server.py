#!/usr/bin/env python3
"""
Knowledge Search MCP Server

Exposes the knowledge search functionality as an MCP (Model Context Protocol)
server for integration with Claude Code, Cursor, and other MCP-compatible
AI agents.

MCP is an open standard by Anthropic that enables AI models to interact with
external tools and data sources using JSON-RPC 2.0 over stdio or HTTP.

Usage:
    # Run directly with stdio transport (for local integration)
    python -m src.knowledge.wiki_mcps.mcp_server
    
    # Or with uv
    uv run src/knowledge/wiki_mcps/mcp_server.py
    
    # Use a different search backend
    KG_SEARCH_BACKEND=kg_llm_navigation python -m src.knowledge.wiki_mcps.mcp_server

Environment Variables:
    KG_INDEX_PATH: Optional path to a Kapso `.index` file.
        If set, the MCP server initializes the search backend from that file
        (backend type + backend_refs like Weaviate collection).
        This overrides `KG_SEARCH_BACKEND`.
    KG_SEARCH_BACKEND: Search backend to use (default: kg_graph_search)
        - kg_graph_search: Hybrid vector + graph search (Weaviate + Neo4j)
        - kg_llm_navigation: LLM-guided multi-hop navigation
    OPENAI_API_KEY: Required for embeddings
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password
    WEAVIATE_URL: Weaviate server URL (default: http://localhost:8081)
"""

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
    )
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None

from src.knowledge.search.factory import KnowledgeSearchFactory
from src.knowledge.search.base import (
    KGSearchFilters,
    KGIndexInput,
    KGIndexMetadata,
    KGEditInput,
    PageType,
    WikiPage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Search Backend Singleton
# =============================================================================

_search_backend = None
_backend_type = None
_index_path: Optional[str] = None
_index_metadata: Optional[KGIndexMetadata] = None
_index_data_source: Optional[str] = None


def get_search_backend():
    """
    Get or create the search backend singleton.
    
    The backend type is configured via KG_SEARCH_BACKEND environment variable:
    - kg_graph_search (default): Hybrid vector + graph search
    - kg_llm_navigation: LLM-guided multi-hop navigation
    
    The backend is initialized lazily on first access.
    
    Returns:
        KnowledgeSearch instance
    """
    global _search_backend, _backend_type, _index_path, _index_metadata, _index_data_source
    if _search_backend is None:
        # Option A: index-aware initialization.
        #
        # If KG_INDEX_PATH is set, we treat the `.index` file as the source of truth
        # for which backend to use AND which backend refs to use (e.g. Weaviate
        # collection name). This prevents drift between:
        # - a Kapso instance that loaded a specific `.index`, and
        # - the MCP server that Claude Code spawns for kg_index/kg_edit/search.
        #
        # If the index cannot be loaded, we fall back to env/default behavior.
        _index_path = os.getenv("KG_INDEX_PATH")
        if _index_path:
            try:
                index_path = Path(_index_path).expanduser().resolve()
                if not index_path.exists():
                    raise FileNotFoundError(f"Index file not found: {index_path}")

                index_data = json.loads(index_path.read_text(encoding="utf-8"))
                _index_metadata = KGIndexMetadata.from_dict(index_data)
                _index_data_source = _index_metadata.data_source

                # Backend type comes from the .index file.
                # If missing, default to kg_graph_search.
                _backend_type = (_index_metadata.search_backend or "").strip() or "kg_graph_search"
                backend_refs = _index_metadata.backend_refs or {}

                logger.info(
                    f"Initializing search backend from KG_INDEX_PATH={str(index_path)!r}: {_backend_type}"
                )
                _search_backend = KnowledgeSearchFactory.create(_backend_type, params=backend_refs)
                logger.info(
                    f"Knowledge search backend '{_backend_type}' initialized from index "
                    f"(data_source={_index_metadata.data_source!r})"
                )
                return _search_backend

            except Exception as e:
                # Best-effort fallback: keep the MCP server usable even if the
                # index file is missing/corrupt or the backend can't initialize.
                logger.warning(f"Failed to initialize from KG_INDEX_PATH={_index_path!r}: {e}")

        # Fallback: Read backend type from environment variable.
        _backend_type = os.getenv("KG_SEARCH_BACKEND", "kg_graph_search")
        logger.info(f"Initializing search backend: {_backend_type}")
        _search_backend = KnowledgeSearchFactory.create(_backend_type)
        logger.info(f"Knowledge search backend '{_backend_type}' initialized")
    return _search_backend


def get_backend_type() -> str:
    """Get the current backend type name."""
    global _backend_type
    if _backend_type is None:
        _backend_type = os.getenv("KG_SEARCH_BACKEND", "kg_graph_search")
    return _backend_type


def reset_search_backend():
    """Reset the search backend singleton (useful for testing)."""
    global _search_backend, _backend_type, _index_path, _index_metadata, _index_data_source
    if _search_backend is not None:
        _search_backend.close()
        _search_backend = None
    _backend_type = None
    _index_path = None
    _index_metadata = None
    _index_data_source = None


# =============================================================================
# MCP Server Factory
# =============================================================================

def create_mcp_server() -> "Server":
    """
    Create and configure the MCP server with all tools and resources.
    
    Returns:
        Configured MCP Server instance
        
    Raises:
        ImportError: If mcp package is not installed
    """
    if not HAS_MCP:
        raise ImportError(
            "MCP package not installed. Install with: pip install mcp"
        )
    
    # Create the MCP server instance
    mcp = Server("knowledge-search")
    
    # Register tool handlers
    _register_tools(mcp)
    
    # Register resource handlers
    _register_resources(mcp)
    
    logger.info("MCP server configured with tools and resources")
    return mcp


# =============================================================================
# Tool Registration
# =============================================================================

def _register_tools(mcp: "Server") -> None:
    """Register all MCP tools on the server."""
    
    @mcp.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools for the MCP client."""
        return [
            # Primary search tool
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
                            "description": "Natural language search query (e.g., 'How to fine-tune LLM with limited GPU memory?')",
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
                            "description": "Filter by knowledge domains: LLMs, Deep_Learning, NLP, Fine_Tuning, PEFT, etc.",
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum relevance score threshold (0.0 to 1.0)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            
            # Direct page retrieval
            Tool(
                name="get_wiki_page",
                description="""Retrieve a specific wiki page by its exact title.

Use this when you already know the page title (from a previous search)
and want to get the complete content. Returns full page with all sections.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page_title": {
                            "type": "string",
                            "description": "Exact title of the wiki page (e.g., 'QLoRA_Finetuning', 'Model_Training')",
                        },
                    },
                    "required": ["page_title"],
                },
            ),
            
            # Page types reference
            Tool(
                name="list_page_types",
                description="""List all available page types in the knowledge base with descriptions.

Use this to understand what types of knowledge are available and how to
filter search results effectively.""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            
            # Search with context
            Tool(
                name="search_with_context",
                description="""Search the knowledge base with additional context from previous work.

Similar to search_knowledge but allows providing context about the current
experiment or problem to get more relevant results.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context (e.g., current experiment setup, error messages, previous attempts)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                        "page_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by page types",
                        },
                    },
                    "required": ["query"],
                },
            ),
            
            # Index pages into the knowledge graph
            Tool(
                name="kg_index",
                description="""Index wiki pages into the knowledge graph.

Use this to add new knowledge pages to the search backend. Supports two modes:
1. Directory mode: Index all .md files from a wiki directory
2. Single page mode: Index a single page with provided data

The indexing updates both Weaviate (embeddings) and Neo4j (graph structure).""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wiki_dir": {
                            "type": "string",
                            "description": "Path to directory containing wiki pages (.md files)",
                        },
                        "page_data": {
                            "type": "object",
                            "description": "Single page to index (alternative to wiki_dir)",
                            "properties": {
                                "page_title": {
                                    "type": "string",
                                    "description": "Title of the page (e.g., 'QLoRA_Finetuning')",
                                },
                                "page_type": {
                                    "type": "string",
                                    "enum": ["Workflow", "Principle", "Implementation", "Environment", "Heuristic"],
                                    "description": "Type of knowledge page",
                                },
                                "overview": {
                                    "type": "string",
                                    "description": "Brief overview/summary (used for embeddings)",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Full page content in MediaWiki format",
                                },
                                "domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Domain tags (e.g., ['LLMs', 'Fine_Tuning'])",
                                },
                            },
                            "required": ["page_title", "page_type", "overview", "content"],
                        },
                        "persist_path": {
                            "type": "string",
                            "description": "Path to save parsed pages JSON (for caching)",
                        },
                        "clear_existing": {
                            "type": "boolean",
                            "description": "Clear existing data before indexing (default: false)",
                            "default": False,
                        },
                    },
                },
            ),
            
            # Edit an existing wiki page
            Tool(
                name="kg_edit",
                description="""Edit an existing wiki page in the knowledge graph.

Updates the page across all storage layers:
1. Source file (.md) - if wiki_dir provided
2. Weaviate (embeddings + properties)
3. Neo4j (graph nodes/edges)

Only include fields you want to update - others remain unchanged.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page_id": {
                            "type": "string",
                            "description": "Page ID to edit (format: 'Type/Title', e.g., 'Workflow/QLoRA_Finetuning')",
                        },
                        "updates": {
                            "type": "object",
                            "description": "Fields to update",
                            "properties": {
                                "overview": {
                                    "type": "string",
                                    "description": "New overview text (triggers re-embedding)",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Full new content",
                                },
                                "domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "New domain tags",
                                },
                                "sources": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "title": {"type": "string"},
                                            "url": {"type": "string"},
                                        },
                                    },
                                    "description": "Updated sources/references",
                                },
                                "outgoing_links": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "edge_type": {"type": "string"},
                                            "target_type": {"type": "string"},
                                            "target_id": {"type": "string"},
                                        },
                                    },
                                    "description": "Updated links (triggers edge rebuild)",
                                },
                            },
                        },
                        "wiki_dir": {
                            "type": "string",
                            "description": "Wiki directory path (for source file update)",
                        },
                        "auto_timestamp": {
                            "type": "boolean",
                            "description": "Auto-update last_updated field (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["page_id", "updates"],
                },
            ),
        ]
    
    @mcp.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls from the MCP client."""
        
        if name == "search_knowledge":
            return await _handle_search(arguments)
        
        elif name == "get_wiki_page":
            return await _handle_get_page(arguments)
        
        elif name == "list_page_types":
            return await _handle_list_types(arguments)
        
        elif name == "search_with_context":
            return await _handle_search_with_context(arguments)
        
        elif name == "kg_index":
            return await _handle_index(arguments)
        
        elif name == "kg_edit":
            return await _handle_edit(arguments)
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}. Available tools: search_knowledge, get_wiki_page, list_page_types, search_with_context, kg_index, kg_edit",
            )]


# =============================================================================
# Tool Handlers
# =============================================================================

async def _handle_search(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handle search_knowledge tool call.
    
    Performs semantic search with optional filters and returns
    formatted results with relevance scores.
    """
    try:
        search = get_search_backend()
        
        # Validate and cap top_k
        top_k = min(arguments.get("top_k", 5), 20)
        
        # Build search filters
        filters = KGSearchFilters(
            top_k=top_k,
            min_score=arguments.get("min_score"),
            page_types=arguments.get("page_types"),
            domains=arguments.get("domains"),
            include_content=True,
        )
        
        # Execute search
        query = arguments["query"]
        logger.info(f"Searching: '{query}' with filters: {filters}")
        result = search.search(query=query, filters=filters)
        
        # Format results
        return [TextContent(
            type="text",
            text=_format_search_results(query, result),
        )]
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Search error: {str(e)}",
        )]


async def _handle_search_with_context(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handle search_with_context tool call.
    
    Similar to search but includes additional context for better relevance.
    """
    try:
        search = get_search_backend()
        
        top_k = min(arguments.get("top_k", 5), 20)
        
        filters = KGSearchFilters(
            top_k=top_k,
            page_types=arguments.get("page_types"),
            include_content=True,
        )
        
        query = arguments["query"]
        context = arguments.get("context", "")
        
        logger.info(f"Searching with context: '{query}'")
        result = search.search(query=query, filters=filters, context=context)
        
        return [TextContent(
            type="text",
            text=_format_search_results(query, result),
        )]
        
    except Exception as e:
        logger.error(f"Search with context failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Search error: {str(e)}",
        )]


async def _handle_get_page(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handle get_wiki_page tool call.
    
    Retrieves a specific page by exact title match.
    """
    try:
        search = get_search_backend()
        page_title = arguments["page_title"]
        
        logger.info(f"Getting page: '{page_title}'")
        page = search.get_page(page_title)
        
        if page is None:
            return [TextContent(
                type="text",
                text=f"Page not found: '{page_title}'\n\nTip: Use search_knowledge to find pages by topic.",
            )]
        
        return [TextContent(
            type="text",
            text=_format_wiki_page(page),
        )]
        
    except Exception as e:
        logger.error(f"Get page failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error retrieving page: {str(e)}",
        )]


async def _handle_list_types(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle list_page_types tool call."""
    types_info = """# Available Page Types in Knowledge Base

| Type | Description | Best For |
|------|-------------|----------|
| **Workflow** | Step-by-step guides and recipes | "How do I..." questions |
| **Principle** | Theoretical concepts and foundations | "What is..." or "Why does..." questions |
| **Implementation** | Code patterns, APIs, and syntax | Technical implementation details |
| **Environment** | Setup, dependencies, and configuration | Installation and setup questions |
| **Heuristic** | Best practices, tips, and optimizations | Debugging and optimization advice |

## Usage Examples

**Find workflows for a task:**
```
search_knowledge(
    query="fine-tune transformer model",
    page_types=["Workflow"]
)
```

**Get best practices:**
```
search_knowledge(
    query="LoRA configuration",
    page_types=["Heuristic", "Principle"]
)
```

**Find implementation details:**
```
search_knowledge(
    query="HuggingFace Trainer API",
    page_types=["Implementation"]
)
```
"""
    return [TextContent(
        type="text",
        text=types_info,
    )]


async def _handle_index(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handle kg_index tool call.
    
    Index wiki pages into the knowledge graph from directory or single page.
    """
    try:
        search = get_search_backend()
        
        wiki_dir = arguments.get("wiki_dir")
        page_data = arguments.get("page_data")
        persist_path = arguments.get("persist_path")
        clear_existing = arguments.get("clear_existing", False)
        
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing index data...")
            search.clear()
        
        # Mode 1: Index from directory
        if wiki_dir:
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
            search.index(index_input)
            
            return [TextContent(
                type="text",
                text=f"Successfully indexed pages from: {wiki_dir}\n\nBackend: {get_backend_type()}",
            )]
        
        # Mode 2: Index single page
        elif page_data:
            page_title = page_data.get("page_title", "")
            page_type = page_data.get("page_type", "")
            overview = page_data.get("overview", "")
            content = page_data.get("content", "")
            domains = page_data.get("domains", [])
            
            if not all([page_title, page_type, overview, content]):
                return [TextContent(
                    type="text",
                    text="Error: page_data requires page_title, page_type, overview, and content",
                )]
            
            # Create WikiPage object
            page = WikiPage(
                id=f"{page_type}/{page_title}",
                page_title=page_title,
                page_type=page_type,
                overview=overview,
                content=content,
                domains=domains,
                sources=[],
                outgoing_links=[],
            )
            
            logger.info(f"Indexing single page: {page.id}")
            
            index_input = KGIndexInput(pages=[page])
            search.index(index_input)
            
            return [TextContent(
                type="text",
                text=f"Successfully indexed page: {page.id}\n\nType: {page_type}\nDomains: {', '.join(domains) if domains else 'None'}\nBackend: {get_backend_type()}",
            )]
        
        else:
            return [TextContent(
                type="text",
                text="Error: Must provide either 'wiki_dir' or 'page_data'",
            )]
        
    except Exception as e:
        logger.error(f"Index failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Index error: {str(e)}",
        )]


async def _handle_edit(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Handle kg_edit tool call.
    
    Edit an existing wiki page across all storage layers.
    """
    try:
        search = get_search_backend()
        
        page_id = arguments.get("page_id")
        updates = arguments.get("updates", {})
        wiki_dir = arguments.get("wiki_dir")
        auto_timestamp = arguments.get("auto_timestamp", True)
        
        if not page_id:
            return [TextContent(
                type="text",
                text="Error: 'page_id' is required",
            )]
        
        if not updates:
            return [TextContent(
                type="text",
                text="Error: 'updates' must contain at least one field to update",
            )]
        
        logger.info(f"Editing page: {page_id}, fields: {list(updates.keys())}")
        
        # Build KGEditInput
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
        
        # Execute edit
        success = search.edit(edit_input)
        
        if success:
            fields_updated = [k for k in updates.keys()]
            return [TextContent(
                type="text",
                text=f"Successfully edited page: {page_id}\n\nFields updated: {', '.join(fields_updated)}\nAuto-timestamp: {auto_timestamp}\nBackend: {get_backend_type()}",
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Edit failed: Page '{page_id}' not found or update failed",
            )]
        
    except Exception as e:
        logger.error(f"Edit failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Edit error: {str(e)}",
        )]


# =============================================================================
# Result Formatting
# =============================================================================

def _format_search_results(query: str, result) -> str:
    """Format search results as markdown."""
    if result.is_empty:
        return f"""# Search Results for: "{query}"

No relevant knowledge found for your query.

**Suggestions:**
- Try broader search terms
- Remove filters to get more results
- Check spelling of technical terms
"""
    
    # Build response with rich formatting
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
        
        # Include content preview (first 800 chars)
        if item.content:
            content_preview = item.content[:800].strip()
            if len(item.content) > 800:
                content_preview += "\n\n... [truncated - use `get_wiki_page` for full content]"
            parts.append(f"\n### Content Preview\n{content_preview}\n")
        
        # Show connected pages if available
        connected = item.metadata.get("connected_pages", [])
        if connected:
            conn_summary = ", ".join(
                f"{c.get('title', c.get('id', 'Unknown'))}" 
                for c in connected[:3]
            )
            if len(connected) > 3:
                conn_summary += f", ... (+{len(connected) - 3} more)"
            parts.append(f"\n**Connected:** {conn_summary}\n")
    
    return "".join(parts)


def _format_wiki_page(page: WikiPage) -> str:
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
        for link in page.outgoing_links[:10]:  # Limit to 10
            edge_type = link.get('edge_type', 'related')
            target = link.get('target_id', '')
            target_type = link.get('target_type', '')
            parts.append(f"- {edge_type} → {target_type}: {target}\n")
    
    return "".join(parts)


# =============================================================================
# Resource Registration
# =============================================================================

def _register_resources(mcp: "Server") -> None:
    """Register MCP resources for browsing the knowledge base."""
    
    @mcp.list_resources()
    async def list_resources() -> List[Resource]:
        """List available resources."""
        return [
            Resource(
                uri="knowledge://overview",
                name="Knowledge Base Overview",
                description="Overview of the ML/AI knowledge base structure and how to use it",
                mimeType="text/markdown",
            ),
            Resource(
                uri="knowledge://page-types",
                name="Page Types Reference",
                description="Detailed reference of all page types and their purposes",
                mimeType="text/markdown",
            ),
        ]
    
    @mcp.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        
        if uri == "knowledge://overview":
            return _get_overview_resource()
        
        elif uri == "knowledge://page-types":
            return _get_page_types_resource()
        
        return f"Resource not found: {uri}"


def _get_overview_resource() -> str:
    """Get the knowledge base overview resource content."""
    return """# ML/AI Knowledge Base

This knowledge base contains curated wiki pages about machine learning and AI topics,
organized in a graph structure for easy navigation and discovery.

## Architecture

The knowledge base uses a **hybrid search** approach:
- **Weaviate**: Vector database for semantic search using embeddings
- **Neo4j**: Graph database for relationship-based navigation
- **LLM Reranking**: GPT-based reranking for improved relevance

## Page Types

Pages are organized by type following a Top-Down DAG structure:
- **Workflow** → **Principle** → **Implementation** → **Environment/Heuristic**

| Type | Role | Example |
|------|------|---------|
| Workflow | The Recipe | "Fine-tune LLM with LoRA" |
| Principle | The Theory | "Low-Rank Adaptation Theory" |
| Implementation | The Code | "HuggingFace PEFT API" |
| Environment | The Context | "GPU Memory Requirements" |
| Heuristic | The Wisdom | "Optimal LoRA Rank Selection" |

## How to Search

### Basic Search
```
search_knowledge(query="How to fine-tune BERT for classification")
```

### Filtered Search
```
search_knowledge(
    query="fine-tune LLM",
    page_types=["Workflow", "Heuristic"],
    domains=["LLMs", "PEFT"],
    top_k=5
)
```

### Get Specific Page
```
get_wiki_page(page_title="QLoRA_Finetuning")
```

## Available Domains

Common domain tags include:
- LLMs, Deep_Learning, NLP, Fine_Tuning
- PEFT, LoRA, QLoRA, Quantization
- Training, Optimization, Evaluation
- PyTorch, HuggingFace, Transformers

## Tips for Effective Search

1. **Use natural language**: "How do I reduce GPU memory when fine-tuning?"
2. **Filter by type**: Use page_types for targeted results
3. **Check connected pages**: Related pages often have useful context
4. **Iterate**: Start broad, then narrow with filters
"""


def _get_page_types_resource() -> str:
    """Get the page types reference resource content."""
    return """# Page Types Reference

## Workflow (The Recipe)

**Purpose**: Step-by-step guides for completing ML tasks.

**Structure**:
- Prerequisites and requirements
- Ordered sequence of steps
- Code examples and commands
- Expected outputs and validation

**When to search**: "How do I...", "Steps to...", "Guide for..."

**Example titles**: 
- `Fine_Tune_LLM_With_LoRA`
- `Train_Classification_Model`
- `Deploy_Model_To_Production`

---

## Principle (The Theory)

**Purpose**: Library-agnostic theoretical concepts and foundations.

**Structure**:
- Definition and explanation
- Mathematical foundations (if applicable)
- Key concepts and terminology
- Trade-offs and considerations

**When to search**: "What is...", "Why does...", "Explain..."

**Example titles**:
- `Low_Rank_Adaptation`
- `Attention_Mechanism`
- `Gradient_Accumulation`

---

## Implementation (The Code)

**Purpose**: Concrete syntax, APIs, and code patterns.

**Structure**:
- API reference
- Code examples
- Configuration options
- Common patterns

**When to search**: "API for...", "Code to...", "How to use..."

**Example titles**:
- `HuggingFace_Trainer_API`
- `PyTorch_DataLoader`
- `PEFT_LoraConfig`

---

## Environment (The Context)

**Purpose**: Hardware, OS, dependencies, and setup requirements.

**Structure**:
- System requirements
- Installation instructions
- Configuration steps
- Troubleshooting

**When to search**: "Requirements for...", "Setup...", "Install..."

**Example titles**:
- `GPU_Memory_Requirements`
- `CUDA_Setup`
- `Docker_Environment`

---

## Heuristic (The Wisdom)

**Purpose**: Best practices, tips, optimizations, and debugging advice.

**Structure**:
- Rule of thumb
- When to apply
- Examples and benchmarks
- Caveats and exceptions

**When to search**: "Best practice for...", "Tips...", "Optimize..."

**Example titles**:
- `LoRA_Rank_Selection`
- `Learning_Rate_Scheduling`
- `Batch_Size_Tuning`
"""


# =============================================================================
# Server Runner
# =============================================================================

async def run_mcp_server():
    """
    Run the MCP server with stdio transport.
    
    This is the main entry point for running the server.
    The server communicates via stdin/stdout using JSON-RPC 2.0.
    """
    if not HAS_MCP:
        raise ImportError(
            "MCP package not installed. Install with: pip install mcp"
        )
    
    logger.info("Starting Knowledge Search MCP Server...")
    
    mcp = create_mcp_server()
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio transport")
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options(),
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """CLI entry point for the MCP server."""
    import asyncio
    
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

