# Knowledge Base Module
#
# Handles knowledge storage, learning (ingestion), and search (retrieval).
#
# Submodules:
#   - types: Unified source types (Source.Repo, Source.Idea, etc.)
#   - search: Unified search backends (KG Graph Search, etc.)
#   - learners: Modular knowledge ingestion pipeline
#   - wiki_structure: Wiki page templates and definitions

from kapso.knowledge_base.types import Source, ResearchFindings

from kapso.knowledge_base.search import (
    KnowledgeSearch,
    WikiPage,
    KGIndexInput,
    KGOutput,
    KGResultItem,
    KGSearchFilters,
    PageType,
    KnowledgeSearchFactory,
    register_knowledge_search,
    parse_wiki_directory,
)

from kapso.knowledge_base.learners import (
    # Main pipeline
    KnowledgePipeline,
    PipelineResult,
    # Merger
    KnowledgeMerger,
    MergeResult,
    # Ingestors
    Ingestor,
    IngestorFactory,
    register_ingestor,
)

__all__ = [
    # Types
    "Source",
    "ResearchFindings",
    # Search
    "KnowledgeSearch",
    "WikiPage",
    "KGIndexInput",
    "KGOutput",
    "KGResultItem",
    "KGSearchFilters",
    "PageType",
    "KnowledgeSearchFactory",
    "register_knowledge_search",
    "parse_wiki_directory",
    # Learners - Pipeline
    "KnowledgePipeline",
    "PipelineResult",
    # Learners - Merger
    "KnowledgeMerger",
    "MergeResult",
    # Learners - Ingestors
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
]
