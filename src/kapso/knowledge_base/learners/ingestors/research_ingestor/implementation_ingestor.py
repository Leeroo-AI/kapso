# Implementation Ingestor
#
# Agentic ingestor for `Source.Implementation` objects.
# Uses Claude Code to analyze content and create appropriate wiki pages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import ImplementationIngestor
#     from kapso.knowledge_base.types import Source
#     
#     ingestor = ImplementationIngestor()
#     pages = ingestor.ingest(impl)  # impl is Source.Implementation

from kapso.knowledge_base.learners.ingestors.factory import register_ingestor
from kapso.knowledge_base.learners.ingestors.research_ingestor.base import ResearchIngestorBase


@register_ingestor("implementation")
class ImplementationIngestor(ResearchIngestorBase):
    """
    Agentic ingestor for researcher.Implementation objects.
    
    Uses a three-phase pipeline (planning, writing, auditing) to convert
    research implementations into properly structured wiki pages.
    
    The agent analyzes the implementation content and decides what page types to create:
    - Implementation pages (if content has code/API docs)
    - Principle pages (if content explains underlying concepts)
    - Environment pages (if dependencies are significant)
    - Heuristic pages (if content has tips/pitfalls)
    - Any combination based on content
    
    Example:
        ingestor = ImplementationIngestor()
        pages = ingestor.ingest(impl)
        
        # With custom settings
        ingestor = ImplementationIngestor(params={
            "use_bedrock": False,
            "model": "claude-sonnet-4-20250514",
            "timeout": 300,
        })
    """
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "implementation"
