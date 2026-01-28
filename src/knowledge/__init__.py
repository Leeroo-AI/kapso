# Knowledge Module - Knowledge Management
#
# Handles knowledge learning (ingestion) and search (retrieval).
#
# Submodules:
#   - types: Unified source types (Source.Repo, Source.Idea, etc.)
#   - search: Unified search backends (KG LLM Navigation, RAG, etc.)
#   - learners: Modular knowledge ingestion pipeline
#   - researcher: Web research utilities

from src.knowledge.types import Source, ResearchFindings

from src.knowledge.search import (
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

from src.knowledge.learners import (
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
