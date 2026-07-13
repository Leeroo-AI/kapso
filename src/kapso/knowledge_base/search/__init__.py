# Knowledge Search Module
#
# Unified interface for knowledge indexing and searching.
# Each implementation handles both indexing and querying.
#
# To add a new search backend:
# 1. Create a new file in this directory (e.g., rag_search.py)
# 2. Subclass KnowledgeSearch from base.py
# 3. Use @register_knowledge_search("rag") decorator
# 4. Add configuration presets in knowledge_search.yaml
#
# Example usage:
#   from kapso.knowledge_base.search import (
#       KnowledgeSearchFactory,
#       KGIndexInput,
#       KGSearchFilters,
#       PageType,
#   )
#   
#   # Create search instance
#   search = KnowledgeSearchFactory.create("wiki_search", enabled=True)
#   
#   # Index wiki pages from directory
#   search.index(KGIndexInput(
#       wiki_dir="data/wikis/allenai_allennlp",
#   ))
#   
#   # Search for knowledge
#   result = search.search(
#       query="How to train a model?",
#       filters=KGSearchFilters(top_k=5, page_types=[PageType.WORKFLOW]),
#   )
#   print(result.to_context_string())

from kapso.knowledge_base.search.base import (
    # Core classes
    KnowledgeSearch,
    # Index input
    WikiPage,
    KGIndexInput,
    # Edit input
    KGEditInput,
    # Search filters
    KGSearchFilters,
    PageType,
    # Search output
    KGOutput,
    KGResultItem,
)
from kapso.knowledge_base.search.factory import (
    KnowledgeSearchFactory,
    register_knowledge_search,
)

# KGGraphSearch and wiki parsers are loaded lazily.
#
# Why: kg_graph_search.py imports kapso.core.llm which triggers the full
# execution stack (CodingAgentFactory, aider/litellm, etc.) — adding ~4-5
# seconds to the MCP server cold start. The MCP server only needs the base
# types above, not KGGraphSearch itself (backends.py handles that lazily).
_LAZY = {
    "KGGraphSearch":       ("kapso.knowledge_base.search.kg_graph_search", "KGGraphSearch"),
    "parse_wiki_directory":("kapso.knowledge_base.search.kg_graph_search", "parse_wiki_directory"),
    "parse_wiki_file":     ("kapso.knowledge_base.search.kg_graph_search", "parse_wiki_file"),
}

def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core classes
    "KnowledgeSearch",
    # Index input
    "WikiPage",
    "KGIndexInput",
    # Edit input
    "KGEditInput",
    # Search filters
    "KGSearchFilters",
    "PageType",
    # Search output
    "KGOutput",
    "KGResultItem",
    # Factory
    "KnowledgeSearchFactory",
    "register_knowledge_search",
    # Wiki parser
    "parse_wiki_directory",
    "parse_wiki_file",
    # Search implementations
    "KGGraphSearch",
]
