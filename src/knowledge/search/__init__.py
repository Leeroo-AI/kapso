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
#   from src.knowledge.search import (
#       KnowledgeSearchFactory,
#       register_knowledge_search,
#   )
#   
#   # Create search instance
#   search = KnowledgeSearchFactory.create("kg_llm_navigation", enabled=True, params={...})
#   
#   # Index data
#   search.index(kg_data)
#   
#   # Search for knowledge
#   result = search.search("How to solve classification problems?")
#   print(result.text_results)

from src.knowledge.search.base import (
    KnowledgeSearch,
    KGOutput,
    KGResultItem,
    KGSearchFilters,
    PageType,
)
from src.knowledge.search.factory import (
    KnowledgeSearchFactory,
    register_knowledge_search,
)

__all__ = [
    # Base classes
    "KnowledgeSearch",
    "KGOutput",
    "KGResultItem",
    "KGSearchFilters",
    "PageType",
    # Factory
    "KnowledgeSearchFactory",
    "register_knowledge_search",
]

