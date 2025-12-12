# Knowledge Module - Knowledge Management
#
# Handles knowledge learning (ingestion) and search (retrieval).
#
# Submodules:
#   - search: Unified search backends (KG LLM Navigation, RAG, etc.)
#   - learners: Modular knowledge ingestion (Repo, Paper, File, Experiment)

from src.knowledge.search import (
    KnowledgeSearch,
    KGOutput,
    KGResultItem,
    KGSearchFilters,
    PageType,
    KnowledgeSearchFactory,
    register_knowledge_search,
)

from src.knowledge.learners import (
    Learner,
    KnowledgeChunk,
    LearnerFactory,
    register_learner,
)

__all__ = [
    # Search
    "KnowledgeSearch",
    "KGOutput",
    "KGResultItem",
    "KGSearchFilters",
    "PageType",
    "KnowledgeSearchFactory",
    "register_knowledge_search",
    # Learners
    "Learner",
    "KnowledgeChunk",
    "LearnerFactory",
    "register_learner",
]
