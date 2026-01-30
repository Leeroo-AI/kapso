# Research Report Ingestor
#
# Agentic ingestor for `Source.ResearchReport` objects.
# Uses Claude Code to analyze content and create appropriate wiki pages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import ResearchReportIngestor
#     from kapso.knowledge_base.types import Source
#     
#     ingestor = ResearchReportIngestor()
#     pages = ingestor.ingest(report)  # report is Source.ResearchReport

from kapso.knowledge_base.learners.ingestors.factory import register_ingestor
from kapso.knowledge_base.learners.ingestors.research_ingestor.base import ResearchIngestorBase


@register_ingestor("researchreport")
class ResearchReportIngestor(ResearchIngestorBase):
    """
    Agentic ingestor for researcher.ResearchReport objects.
    
    Uses a three-phase pipeline (planning, writing, auditing) to convert
    comprehensive research reports into properly structured wiki pages.
    
    The agent analyzes the report content and extracts multiple pages:
    - Multiple Principle pages (key concepts from the report)
    - Implementation pages (code examples, if any)
    - Heuristic pages (best practices, if any)
    - The agent decides based on report content
    
    Research reports are typically comprehensive and may produce more pages
    than Idea or Implementation inputs.
    
    Example:
        ingestor = ResearchReportIngestor()
        pages = ingestor.ingest(report)
        
        # With custom settings
        ingestor = ResearchReportIngestor(params={
            "use_bedrock": False,
            "model": "claude-sonnet-4-20250514",
            "timeout": 600,  # Longer timeout for comprehensive reports
        })
    """
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "researchreport"
