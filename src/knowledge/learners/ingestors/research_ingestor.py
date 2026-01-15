# Research Ingestor
#
# Converts `Source.Research` (public web research) into `WikiPage` objects.
# This is Stage 1 of the knowledge learning pipeline (ingestion).
#
# IMPORTANT:
# - This module performs NO network calls. It is a pure transformation.
# - The research report is embedded into the resulting page content.

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.knowledge.learners.ingestors.base import Ingestor
from src.knowledge.learners.ingestors.factory import register_ingestor
from src.knowledge.search.base import WikiPage

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug for WikiPage IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_")
    return (cleaned or "research")[:max_len]


@register_ingestor("research")
class ResearchIngestor(Ingestor):
    """Ingest Source.Research into Principle/Implementation WikiPages."""

    _ALLOWED_MODES = {"idea", "implementation", "both"}

    @property
    def source_type(self) -> str:
        return "research"

    def ingest(self, source: Any) -> List[WikiPage]:
        if isinstance(source, dict):
            objective = source.get("objective")
            mode = source.get("mode")
            report_markdown = source.get("report_markdown")
        else:
            objective = getattr(source, "objective", None)
            mode = getattr(source, "mode", None)
            report_markdown = getattr(source, "report_markdown", None)

        if not objective or not isinstance(objective, str):
            raise ValueError("ResearchIngestor expected a non-empty string 'objective'")
        if mode not in self._ALLOWED_MODES:
            raise ValueError(f"ResearchIngestor expected mode in {sorted(self._ALLOWED_MODES)}; got: {mode!r}")

        report_text = (report_markdown or "").strip()
        if not report_text:
            logger.warning("[ResearchIngestor] Empty report_markdown; using placeholder content")

        slug = _slugify(objective)
        now = datetime.now(timezone.utc).isoformat()

        def make_page(page_type: str) -> WikiPage:
            page_id = f"{page_type}/Web_Research_{slug}"
            page_title = f"Web_Research_{slug}"
            overview = f"Auto-generated web research ({mode}) for: {objective}"
            content = (
                "== Overview ==\n"
                "This page was generated from public web research.\n"
                "Review it for correctness.\n\n"
                "== Objective ==\n"
                f"{objective}\n\n"
                "== Mode ==\n"
                f"{mode}\n\n"
                "== Report ==\n"
                f"{report_text if report_text else '(empty report)'}\n\n"
                "== Metadata ==\n"
                f"* Generated at (UTC): {now}\n"
            )
            return WikiPage(
                id=page_id,
                page_title=page_title,
                page_type=page_type,
                overview=overview,
                content=content,
                domains=["Research", "Web"],
                sources=[],
                outgoing_links=[],
            )

        pages: List[WikiPage] = []
        if mode in {"idea", "both"}:
            pages.append(make_page("Principle"))
        if mode in {"implementation", "both"}:
            pages.append(make_page("Implementation"))

        return pages

