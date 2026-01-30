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
#       persist_path="data/indexes/allenai_allennlp.json",
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
from kapso.knowledge_base.search.kg_graph_search import (
    KGGraphSearch,
    parse_wiki_directory,
    parse_wiki_file,
)

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
