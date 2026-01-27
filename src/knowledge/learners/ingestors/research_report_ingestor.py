# Research Report Ingestor
#
# Converts `ResearchReport` (from researcher) into `WikiPage` objects.

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
    return (cleaned or "report")[:max_len]


@register_ingestor("researchreport")
class ResearchReportIngestor(Ingestor):
    """
    Ingest researcher.ResearchReport into Principle WikiPage.
    
    Extracts:
    - Full report as a WikiPage (type: Principle)
    """

    @property
    def source_type(self) -> str:
        return "researchreport"

    def ingest(self, source: Any) -> List[WikiPage]:
        # Extract attributes
        if isinstance(source, dict):
            query = source.get("query", "")
            content = source.get("content", "")
        else:
            query = getattr(source, "query", "")
            content = getattr(source, "content", "")

        if not query:
            raise ValueError("ResearchReportIngestor expected a non-empty 'query'")

        slug = _slugify(query)
        now = datetime.now(timezone.utc).isoformat()

        page_id = f"Principle/Research_Report_{slug}"
        page_title = f"Research_Report_{slug}"
        overview = f"Research report for: {query}"
        
        page_content = (
            "== Overview ==\n"
            f"Comprehensive research report from web research.\n\n"
            "== Query ==\n"
            f"{query}\n\n"
            "== Report ==\n"
            f"{content}\n\n"
            "== Metadata ==\n"
            f"* Generated at (UTC): {now}\n"
        )

        return [WikiPage(
            id=page_id,
            page_title=page_title,
            page_type="Principle",
            overview=overview,
            content=page_content,
            domains=["Research", "Report"],
            sources=[],
            outgoing_links=[],
        )]
