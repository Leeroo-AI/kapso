# Knowledge Learner Pipeline
#
# Main orchestrator for the knowledge learning pipeline.
# Coordinates ingestors (Stage 1) and merger (Stage 2) to:
#   Source → Ingestor → WikiPages → Merger → Updated KG
#
# Usage:
#     from kapso.knowledge_base.learners import KnowledgePipeline, Source
#     
#     pipeline = KnowledgePipeline()
#     result = pipeline.run(Source.Repo("https://github.com/user/repo"))

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from kapso.knowledge_base.learners.ingestors.factory import IngestorFactory
from kapso.knowledge_base.learners.merger import (
    KnowledgeMerger,
    MergeResult,
)
from kapso.knowledge_base.search.base import WikiPage, DEFAULT_WIKI_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PipelineResult:
    """
    Result from a knowledge pipeline run.
    
    Attributes:
        sources_processed: Number of sources that were processed
        total_pages_extracted: Total WikiPages extracted from all sources
        merge_result: Result from the merger (if merging was performed)
        extracted_pages: List of all extracted WikiPages
        errors: Errors the merge reported (extraction failures raise)
    """
    sources_processed: int = 0
    total_pages_extracted: int = 0
    merge_result: Optional[MergeResult] = None
    extracted_pages: List[WikiPage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def created(self) -> int:
        """Number of new pages created in KG."""
        return len(self.merge_result.created) if self.merge_result else 0
    
    @property
    def edited(self) -> int:
        """Number of pages edited/merged with existing."""
        return len(self.merge_result.edited) if self.merge_result else 0
    
    @property
    def success(self) -> bool:
        """Whether the merge reported no errors. Extraction failures raise
        out of run() rather than landing here, so this cannot be True for a
        run that lost a source or a phase."""
        return not self.errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sources_processed": self.sources_processed,
            "total_pages_extracted": self.total_pages_extracted,
            "created": self.created,
            "edited": self.edited,
            "merge_result": self.merge_result.to_dict() if self.merge_result else None,
            "errors": self.errors,
        }
    
    def __repr__(self) -> str:
        return (
            f"PipelineResult(sources={self.sources_processed}, "
            f"extracted={self.total_pages_extracted}, "
            f"created={self.created}, edited={self.edited})"
        )


# =============================================================================
# Knowledge Pipeline
# =============================================================================

class KnowledgePipeline:
    """
    Complete knowledge learning pipeline.
    
    Orchestrates the two-stage process:
    1. Ingestion: Source → Ingestor → WikiPages
    2. Merging: WikiPages → Merger → Updated KG
    
    The KG is stored in:
    - Neo4j: Graph structure (nodes + edges) - THE INDEX
    - Weaviate: Embeddings for semantic search
    - Source files: Ground truth .md files
    
    Usage:
        from kapso.knowledge_base.learners import KnowledgePipeline, Source
        
        pipeline = KnowledgePipeline()
        
        # Single source - full pipeline
        result = pipeline.run(Source.Repo("https://github.com/user/repo"))
        print(f"Created: {result.created}, Edited: {result.edited}")
        
        # Multiple sources
        result = pipeline.run(
            Source.Repo("https://github.com/user/repo"),
        )
        
        # Extract only (skip merge step)
        result = pipeline.run(
            Source.Repo("https://github.com/user/repo"),
            skip_merge=True,
        )
        
        # Ingest only (get pages without merging)
        pages = pipeline.ingest_only(Source.Repo("https://github.com/user/repo"))
    """
    
    def __init__(
        self,
        wiki_dir: Optional[Union[str, Path]] = None,
        ingestor_params: Optional[Dict[str, Any]] = None,
        merger_params: Optional[Dict[str, Any]] = None,
        index_builder: Optional[Callable[..., str]] = None,
    ):
        """
        Initialize the knowledge pipeline.
        
        Args:
            wiki_dir: Path to wiki directory (default: data/wikis)
            ingestor_params: Default parameters for all ingestors
            merger_params: Parameters for the knowledge merger
            index_builder: Builds the search index of a wiki that has none
                (`Kapso.index_kg`; see KnowledgeMerger). The backends and the
                collection the index names come from the config it reads.
        """
        # Normalize wiki_dir and make it absolute
        self.wiki_dir = (Path(wiki_dir) if wiki_dir else DEFAULT_WIKI_DIR).expanduser().resolve()
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        
        # Ingestor params - share wiki_dir
        self.ingestor_params = ingestor_params or {}
        self.ingestor_params.setdefault("wiki_dir", self.wiki_dir)
        
        # Merger params
        self.merger_params = merger_params or {}
        
        # Initialize merger
        self._merger = KnowledgeMerger(agent_config=self.merger_params, index_builder=index_builder)
    
    def run(
        self,
        *sources,
        skip_merge: bool = False,
        status=None,
    ) -> PipelineResult:
        """
        Run the complete knowledge pipeline.

        Args:
            *sources: One or more Source objects (Source.Repo, Source.Solution)
            skip_merge: If True, only extract (same as ingest_only but returns PipelineResult)
            status: Optional KnowledgeStatus (observability §2) — receives
                per-source ingest progress and the merge phase.

        Returns:
            PipelineResult with statistics and any errors
        """
        result = PipelineResult()

        if not sources:
            raise ValueError("No sources provided")

        # Stage 1: Ingest all sources
        all_pages = []
        source_urls = []
        last_staging_dir = None  # Track the most recent staging directory

        for source_index, source in enumerate(sources):
            if status is not None:
                status.phase(
                    "ingest",
                    sources={"done": source_index, "total": len(sources)},
                    current_source=str(source),
                    pages_extracted=len(all_pages),
                )
            logger.info(f"Ingesting source: {source}")
            
            # Get the appropriate ingestor
            ingestor = IngestorFactory.for_source(source, **self.ingestor_params)
            
            # Run ingestion (a failure raises: a source that could not be
            # ingested must not be reported as a partial success)
            pages = ingestor.ingest(source)
            
            all_pages.extend(pages)
            result.sources_processed += 1
            
            # Track staging directory for the merger prompt
            staging = ingestor.get_staging_dir()
            if staging:
                last_staging_dir = staging
            
            # Track source URL for context
            if hasattr(source, 'url'):
                source_urls.append(source.url)
            elif hasattr(source, 'path'):
                source_urls.append(source.path)
            
            logger.info(f"Extracted {len(pages)} pages from source")
        
        result.total_pages_extracted = len(all_pages)
        result.extracted_pages = all_pages
        if status is not None:
            status.note(
                f"ingest done: {len(all_pages)} pages from "
                f"{result.sources_processed}/{len(sources)} sources",
                sources={
                    "done": result.sources_processed,
                    "total": len(sources),
                },
                pages_extracted=len(all_pages),
            )

        # If no pages extracted or skip_merge, return early
        if not all_pages or skip_merge:
            if not all_pages:
                logger.warning("No pages extracted from any source")
            return result

        # Stage 2: Merge into KG
        if status is not None:
            status.phase("merge")
        # Run merge (Stage 2) — pass staging_dir so the agent can read
        # candidate page files from disk on demand. A failed merge raises.
        merge_result = self._merger.merge(
            all_pages,
            wiki_dir=self.wiki_dir,
            staging_dir=last_staging_dir,
        )
        result.merge_result = merge_result
        
        # Add merge errors to result
        if merge_result.errors:
            result.errors.extend(merge_result.errors)
        
        logger.info(
            f"Pipeline complete: {result.created} created, "
            f"{result.edited} edited"
        )
        
        return result
    
    def ingest_only(self, *sources) -> List[WikiPage]:
        """
        Run only Stage 1: ingest sources and return WikiPages.
        
        This is useful for previewing what would be extracted before
        committing to a merge.
        
        Args:
            *sources: One or more Source objects
            
        Returns:
            List of all extracted WikiPage objects
        """
        result = self.run(*sources, skip_merge=True)
        return result.extracted_pages
    
    def merge_pages(
        self,
        pages: List[WikiPage],
        staging_dir: Optional[Path] = None,
    ) -> MergeResult:
        """
        Run only Stage 2: merge existing WikiPages into KG.
        
        This is useful when you have pre-extracted pages and want
        to merge them separately.
        
        Args:
            pages: List of WikiPage objects to merge
            staging_dir: Optional staging directory where candidate .md files
                         live on disk (for the agentic merge prompt)
            
        Returns:
            MergeResult with statistics
        """
        return self._merger.merge(pages, wiki_dir=self.wiki_dir, staging_dir=staging_dir)
    
    def close(self) -> None:
        """Clean up resources."""
        if self._merger:
            self._merger.close()

