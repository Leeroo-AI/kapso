"""
KG Search implementation for Leeroopedia API.

Provides wiki_idea_search and wiki_code_search functionality
using the KGGraphSearch backend.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formatters import format_idea_results, format_code_results

logger = logging.getLogger(__name__)

# Page types for each search type
IDEA_TYPES = ["Principle", "Heuristic"]
CODE_TYPES = ["Implementation", "Environment"]

# Backend singleton
_kg_search_backend = None
_kg_search_lock = threading.Lock()
_circuit_open = False
_consecutive_failures = 0
FAILURE_THRESHOLD = 5


class BackendUnavailableError(Exception):
    """Raised when the KG backend is unavailable."""
    pass


@dataclass
class SearchResult:
    """Result from a search operation."""
    success: bool
    content: str
    results_count: int
    latency_ms: int
    error: Optional[str] = None


def get_kg_search_backend() -> Any:
    """
    Get or create the KGGraphSearch backend singleton.

    Thread-safe lazy initialization with circuit breaker pattern.

    Returns:
        KGGraphSearch instance

    Raises:
        BackendUnavailableError: If backend is unavailable or circuit is open
    """
    global _kg_search_backend, _circuit_open, _consecutive_failures

    if _circuit_open:
        raise BackendUnavailableError("KG search backend circuit breaker is open")

    if _kg_search_backend is not None:
        return _kg_search_backend

    with _kg_search_lock:
        # Double-check after acquiring lock
        if _kg_search_backend is not None:
            return _kg_search_backend

        try:
            from kapso.knowledge_base.search.factory import KnowledgeSearchFactory
            from kapso.knowledge_base.search.base import KGIndexMetadata

            # Get index path from environment
            index_path_str = os.getenv("KG_INDEX_PATH")
            backend_type = "kg_graph_search"
            backend_refs = {}

            if index_path_str:
                try:
                    index_path = Path(index_path_str).expanduser().resolve()
                    if index_path.exists():
                        index_data = json.loads(index_path.read_text(encoding="utf-8"))
                        metadata = KGIndexMetadata.from_dict(index_data)

                        backend_type = (metadata.search_backend or "").strip() or "kg_graph_search"
                        backend_refs = metadata.backend_refs or {}

                        logger.info(
                            f"Initializing KGGraphSearch from index: {index_path}, "
                            f"backend={backend_type}"
                        )
                    else:
                        logger.warning(f"KG_INDEX_PATH not found: {index_path}")
                except Exception as e:
                    logger.warning(f"Failed to read KG_INDEX_PATH: {e}")

            logger.info(f"Creating KGGraphSearch backend: {backend_type}")
            _kg_search_backend = KnowledgeSearchFactory.create(backend_type, params=backend_refs)
            _consecutive_failures = 0
            logger.info("KGGraphSearch backend initialized successfully")

            return _kg_search_backend

        except Exception as e:
            _consecutive_failures += 1
            logger.error(
                f"KGGraphSearch initialization failed: {e}, "
                f"consecutive_failures={_consecutive_failures}"
            )

            if _consecutive_failures >= FAILURE_THRESHOLD:
                _circuit_open = True
                logger.error("Circuit breaker opened due to repeated failures")

            raise BackendUnavailableError(f"KGGraphSearch initialization failed: {e}") from e


def reset_backends() -> None:
    """Reset all backend singletons."""
    global _kg_search_backend, _circuit_open, _consecutive_failures

    with _kg_search_lock:
        if _kg_search_backend is not None:
            try:
                _kg_search_backend.close()
            except Exception as e:
                logger.warning(f"Error closing KGGraphSearch: {e}")
            _kg_search_backend = None

        _circuit_open = False
        _consecutive_failures = 0

    logger.info("All backends reset")


async def _run_sync(func, *args, **kwargs) -> Any:
    """Run a synchronous function in an executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def execute_search(
    query: str,
    tool: str,
    top_k: int = 5,
    domains: Optional[List[str]] = None,
) -> SearchResult:
    """
    Execute a wiki search.

    Args:
        query: Search query
        tool: Tool name (wiki_idea_search or wiki_code_search)
        top_k: Number of results to return (max 20)
        domains: Optional domain filters

    Returns:
        SearchResult with content and metadata
    """
    start_time = time.time()

    try:
        from kapso.knowledge_base.search.base import KGSearchFilters

        search = get_kg_search_backend()

        if not query:
            return SearchResult(
                success=False,
                content="Error: query is required",
                results_count=0,
                latency_ms=0,
                error="missing_query",
            )

        top_k = min(top_k, 20)

        # Determine page types based on tool
        if tool == "wiki_idea_search":
            page_types = IDEA_TYPES
            search_type = "idea"
        elif tool == "wiki_code_search":
            page_types = CODE_TYPES
            search_type = "code"
        else:
            return SearchResult(
                success=False,
                content=f"Error: Unknown tool {tool}",
                results_count=0,
                latency_ms=0,
                error="unknown_tool",
            )

        filters = KGSearchFilters(
            top_k=top_k,
            page_types=page_types,
            domains=domains,
            include_content=True,
        )

        logger.info(
            f"{search_type.capitalize()} search: query={query}, top_k={top_k}, domains={domains}"
        )

        # Execute search
        result = await _run_sync(
            search.search,
            query=query,
            filters=filters,
            use_llm_reranker=True,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Format results
        if search_type == "idea":
            content = format_idea_results(query, result)
        else:
            content = format_code_results(query, result)

        return SearchResult(
            success=True,
            content=content,
            results_count=result.total_found,
            latency_ms=latency_ms,
        )

    except BackendUnavailableError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Backend unavailable during search: {e}")
        return SearchResult(
            success=False,
            content="Search service is temporarily unavailable. Please retry later.",
            results_count=0,
            latency_ms=latency_ms,
            error="backend_unavailable",
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Search failed: {e}", exc_info=True)
        return SearchResult(
            success=False,
            content=f"Search error: {str(e)}",
            results_count=0,
            latency_ms=latency_ms,
            error="search_error",
        )
