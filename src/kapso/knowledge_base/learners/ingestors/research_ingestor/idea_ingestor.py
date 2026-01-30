# Idea Ingestor
#
# Agentic ingestor for `Source.Idea` objects.
# Uses Claude Code to analyze content and create appropriate wiki pages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import IdeaIngestor
#     from kapso.knowledge_base.types import Source
#     
#     ingestor = IdeaIngestor()
#     pages = ingestor.ingest(idea)  # idea is Source.Idea

from kapso.knowledge_base.learners.ingestors.factory import register_ingestor
from kapso.knowledge_base.learners.ingestors.research_ingestor.base import ResearchIngestorBase


@register_ingestor("idea")
class IdeaIngestor(ResearchIngestorBase):
    """
    Agentic ingestor for researcher.Idea objects.
    
    Uses a three-phase pipeline (planning, writing, auditing) to convert
    research ideas into properly structured wiki pages.
    
    The agent analyzes the idea content and decides what page types to create:
    - Principle pages (if content is theoretical)
    - Implementation pages (if content has code examples)
    - Heuristic pages (if content has tips/trade-offs)
    - Any combination based on content
    
    Example:
        ingestor = IdeaIngestor()
        pages = ingestor.ingest(idea)
        
        # With custom settings
        ingestor = IdeaIngestor(params={
            "use_bedrock": False,
            "model": "claude-sonnet-4-20250514",
            "timeout": 300,
        })
    """
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "idea"
