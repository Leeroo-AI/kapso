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
from typing import Any, Dict, List, Optional, Union

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
        errors: List of errors encountered during processing
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
        """Whether the pipeline completed without critical errors."""
        return len(self.errors) == 0 or self.total_pages_extracted > 0
    
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
        weaviate_collection: str = "KGWikiPages",
        ingestor_params: Optional[Dict[str, Any]] = None,
        merger_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the knowledge pipeline.
        
        Args:
            wiki_dir: Path to wiki directory (default: data/wikis)
            weaviate_collection: Weaviate collection for embeddings
            ingestor_params: Default parameters for all ingestors
            merger_params: Parameters for the knowledge merger
        """
        # Normalize wiki_dir and make it absolute
        self.wiki_dir = (Path(wiki_dir) if wiki_dir else DEFAULT_WIKI_DIR).expanduser().resolve()
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        
        # Store collection name
        self.weaviate_collection = weaviate_collection
        
        # Ingestor params - share wiki_dir
        self.ingestor_params = ingestor_params or {}
        self.ingestor_params.setdefault("wiki_dir", self.wiki_dir)
        
        # Merger params
        self.merger_params = merger_params or {}
        
        # Initialize merger
        self._merger = KnowledgeMerger(agent_config=self.merger_params)
    
    def run(
        self,
        *sources,
        skip_merge: bool = False,
    ) -> PipelineResult:
        """
        Run the complete knowledge pipeline.
        
        Args:
            *sources: One or more Source objects (Source.Repo, Source.Solution)
            skip_merge: If True, only extract (same as ingest_only but returns PipelineResult)
            
        Returns:
            PipelineResult with statistics and any errors
        """
        result = PipelineResult()
        
        if not sources:
            result.errors.append("No sources provided")
            return result
        
        # Stage 1: Ingest all sources
        all_pages = []
        source_urls = []
        
        for source in sources:
            try:
                logger.info(f"Ingesting source: {source}")
                
                # Get the appropriate ingestor
                ingestor = IngestorFactory.for_source(source, **self.ingestor_params)
                
                # Run ingestion
                pages = ingestor.ingest(source)
                
                all_pages.extend(pages)
                result.sources_processed += 1
                
                # Track source URL for context
                if hasattr(source, 'url'):
                    source_urls.append(source.url)
                elif hasattr(source, 'path'):
                    source_urls.append(source.path)
                
                logger.info(f"Extracted {len(pages)} pages from source")
                
            except Exception as e:
                error_msg = f"Failed to ingest source {source}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        result.total_pages_extracted = len(all_pages)
        result.extracted_pages = all_pages
        
        # If no pages extracted or skip_merge, return early
        if not all_pages or skip_merge:
            if not all_pages:
                logger.warning("No pages extracted from any source")
            return result
        
        # Stage 2: Merge into KG
        try:
            # Run merge (Stage 2)
            merge_result = self._merger.merge(all_pages, wiki_dir=self.wiki_dir)
            result.merge_result = merge_result
            
            # Add merge errors to result
            if merge_result.errors:
                result.errors.extend(merge_result.errors)
            
            logger.info(
                f"Pipeline complete: {result.created} created, "
                f"{result.edited} edited"
            )
            
        except Exception as e:
            error_msg = f"Merge failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
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
    ) -> MergeResult:
        """
        Run only Stage 2: merge existing WikiPages into KG.
        
        This is useful when you have pre-extracted pages and want
        to merge them separately.
        
        Args:
            pages: List of WikiPage objects to merge
            
        Returns:
            MergeResult with statistics
        """
        return self._merger.merge(pages, wiki_dir=self.wiki_dir)
    
    def close(self) -> None:
        """Clean up resources."""
        if self._merger:
            self._merger.close()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for the knowledge pipeline."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Knowledge Learning Pipeline - Extract and merge knowledge into KG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from a GitHub repository
  python -m kapso.knowledge_base.learners https://github.com/user/repo

  # Dry run (analyze without modifying KG)
  python -m kapso.knowledge_base.learners https://github.com/user/repo --dry-run

  # Extract only (don't merge)
  python -m kapso.knowledge_base.learners https://github.com/user/repo --extract-only

  # Learn from a paper (when implemented)
  python -m kapso.knowledge_base.learners ./paper.pdf --type paper
        """
    )
    
    parser.add_argument(
        "source",
        help="Source URL or path"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["repo", "paper", "solution"],
        default="repo",
        help="Source type (default: repo)"
    )
    
    parser.add_argument(
        "--branch", "-b",
        default="main",
        help="Git branch for repo sources (default: main)"
    )
    
    parser.add_argument(
        "--extract-only", "-e",
        action="store_true",
        help="Only extract, don't merge into KG"
    )
    
    parser.add_argument(
        "--wiki-dir", "-w",
        type=Path,
        default=None,
        help=f"Wiki directory path (default: {DEFAULT_WIKI_DIR})"
    )
    
    parser.add_argument(
        "--collection", "-c",
        default="KGWikiPages",
        help="Weaviate collection name (default: KGWikiPages)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import Source after parsing to avoid slow imports on --help
    from kapso.knowledge_base.learners.sources import Source
    
    # Create source object based on type
    if args.type == "repo":
        source = Source.Repo(args.source, branch=args.branch)
    else:
        print(f"Unsupported source type: {args.type}")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("  Knowledge Learning Pipeline")
    print("=" * 70)
    print(f"\nSource: {args.source}")
    print(f"Type:   {args.type}")
    print(f"Mode:   {'Extract Only' if args.extract_only else 'Full Pipeline'}")
    
    # Run pipeline
    pipeline = KnowledgePipeline(
        wiki_dir=args.wiki_dir,
        weaviate_collection=args.collection,
    )
    
    result = pipeline.run(
        source,
        skip_merge=args.extract_only,
    )
    
    # Print results
    print("\n" + "-" * 70)
    print("Results:")
    print(f"  Sources processed:    {result.sources_processed}")
    print(f"  Pages extracted:      {result.total_pages_extracted}")
    
    if result.merge_result:
        print(f"  Pages created:        {result.created}")
        print(f"  Pages edited:         {result.edited}")
    
    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"    - {error}")
    
    if result.extracted_pages and args.extract_only:
        print(f"\n  Extracted Pages:")
        for page in result.extracted_pages[:10]:
            print(f"    - {page.page_title} ({page.page_type})")
        if len(result.extracted_pages) > 10:
            print(f"    ... and {len(result.extracted_pages) - 10} more")
    
    print("\n" + "=" * 70)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
