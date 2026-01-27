# Implementation Ingestor
#
# Converts `Implementation` (from researcher) into `WikiPage` objects.

import logging
import re
from datetime import datetime, timezone
from typing import Any, List

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug for WikiPage IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
    return (cleaned or "impl")[:max_len]


@register_ingestor("implementation")
class ImplementationIngestor(Ingestor):
    """
    Ingest researcher.Implementation into Implementation WikiPage.
    
    Extracts:
    - Main implementation as a WikiPage (type: Implementation)
    """

    @property
    def source_type(self) -> str:
        return "implementation"

    def ingest(self, source: Any) -> List[WikiPage]:
        # Extract attributes
        if isinstance(source, dict):
            query = source.get("query", "")
            src_url = source.get("source", "")
            content = source.get("content", "")
        else:
            query = getattr(source, "query", "")
            src_url = getattr(source, "source", "")
            content = getattr(source, "content", "")

        if not query:
            raise ValueError("ImplementationIngestor expected a non-empty 'query'")

        slug = _slugify(query)
        now = datetime.now(timezone.utc).isoformat()

        page_id = f"Implementation/Research_Impl_{slug}"
        page_title = f"Research_Impl_{slug}"
        overview = f"Implementation for: {query}"
        
        page_content = (
            "== Overview ==\n"
            f"Implementation extracted from web research.\n\n"
            "== Query ==\n"
            f"{query}\n\n"
            "== Source ==\n"
            f"{src_url}\n\n"
            "== Content ==\n"
            f"{content}\n\n"
            "== Metadata ==\n"
            f"* Generated at (UTC): {now}\n"
        )

        return [WikiPage(
            id=page_id,
            page_title=page_title,
            page_type="Implementation",
            overview=overview,
            content=page_content,
            domains=["Research", "Implementation"],
            sources=[src_url] if src_url else [],
            outgoing_links=[],
        )]
