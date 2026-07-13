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
)

# Heavy symbols loaded lazily.
#
# Why: `learners` (KnowledgeMerger, KnowledgePipeline, etc.) and
# `parse_wiki_directory` (from kg_graph_search) import CodingAgentFactory
# which loads aider/litellm — adding ~30 seconds to the MCP server cold
# start.  The MCP server only needs base search types (above), so we defer
# all heavy loading until explicitly requested.
_LAZY = {
    # learners
    "KnowledgePipeline":    ("kapso.knowledge_base.learners", "KnowledgePipeline"),
    "PipelineResult":       ("kapso.knowledge_base.learners", "PipelineResult"),
    "KnowledgeMerger":      ("kapso.knowledge_base.learners", "KnowledgeMerger"),
    "MergeResult":          ("kapso.knowledge_base.learners", "MergeResult"),
    "Ingestor":             ("kapso.knowledge_base.learners", "Ingestor"),
    "IngestorFactory":      ("kapso.knowledge_base.learners", "IngestorFactory"),
    "register_ingestor":    ("kapso.knowledge_base.learners", "register_ingestor"),
    # kg_graph_search helpers
    "parse_wiki_directory": ("kapso.knowledge_base.search.kg_graph_search", "parse_wiki_directory"),
}

def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
