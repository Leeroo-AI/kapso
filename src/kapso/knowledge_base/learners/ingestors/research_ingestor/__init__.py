# Research Ingestors Package
#
# Agentic ingestors for research outputs (Idea, Implementation, ResearchReport).
# Uses Claude Code to analyze content and create properly structured wiki pages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors.research_ingestor import (
#         IdeaIngestor,
#         ImplementationIngestor,
#         ResearchReportIngestor,
#     )
#     
#     ingestor = IdeaIngestor()
#     pages = ingestor.ingest(idea)

from kapso.knowledge_base.learners.ingestors.research_ingestor.base import ResearchIngestorBase
from kapso.knowledge_base.learners.ingestors.research_ingestor.idea_ingestor import IdeaIngestor
from kapso.knowledge_base.learners.ingestors.research_ingestor.implementation_ingestor import ImplementationIngestor
from kapso.knowledge_base.learners.ingestors.research_ingestor.research_report_ingestor import ResearchReportIngestor

__all__ = [
    "ResearchIngestorBase",
    "IdeaIngestor",
    "ImplementationIngestor",
    "ResearchReportIngestor",
]
