# Knowledge Module - Knowledge Management
#
# Handles knowledge learning (ingestion) and search (retrieval).
#
# Submodules:
#   - search: Unified search backends (KG LLM Navigation, RAG, etc.)
#   - learners: Modular knowledge ingestion pipeline (Repo, Paper, Experiment)

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
    Source,
    # Merger
    KnowledgeMerger,
    MergeResult,
    # Ingestors
    Ingestor,
    IngestorFactory,
    register_ingestor,
)

__all__ = [
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
    "Source",
    # Learners - Merger
    "KnowledgeMerger",
    "MergeResult",
    # Learners - Ingestors
    "Ingestor",
    "IngestorFactory",
    "register_ingestor",
]
