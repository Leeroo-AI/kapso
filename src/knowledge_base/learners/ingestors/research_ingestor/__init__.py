# Research Ingestors Package
#
# Agentic ingestors for research outputs (Idea, Implementation, ResearchReport).
# Uses Claude Code to analyze content and create properly structured wiki pages.
#
# Usage:
#     from src.knowledge_base.learners.ingestors.research_ingestor import (
#         IdeaIngestor,
#         ImplementationIngestor,
#         ResearchReportIngestor,
#     )
#     
#     ingestor = IdeaIngestor()
#     pages = ingestor.ingest(idea)

from src.knowledge_base.learners.ingestors.research_ingestor.base import ResearchIngestorBase
from src.knowledge_base.learners.ingestors.research_ingestor.idea_ingestor import IdeaIngestor
from src.knowledge_base.learners.ingestors.research_ingestor.implementation_ingestor import ImplementationIngestor
from src.knowledge_base.learners.ingestors.research_ingestor.research_report_ingestor import ResearchReportIngestor

__all__ = [
    "ResearchIngestorBase",
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
