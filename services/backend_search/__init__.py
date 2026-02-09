"""
Backend Search Service

Provides KG search logic for the Leeroopedia API.
Can run as HTTP service for distributed deployment.

Usage:
    uvicorn backend_search.app:app --host 0.0.0.0 --port 3003
"""

from .search import execute_search, SearchResult
from .formatters import format_idea_results, format_code_results

__all__ = [
    "execute_search",
    "SearchResult",
    "format_idea_results",
    "format_code_results",
]
