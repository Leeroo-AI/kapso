# Idea and Implementation Search
#
# Wiki-based retrieval tools that wrap KGGraphSearch with page type filters.
# - WikiIdeaSearch: Searches for Principles and Heuristics (conceptual knowledge)
# - WikiCodeSearch: Searches for Implementations and Environments (code knowledge)
#
# Usage:
#   from kapso.knowledge_base.search.idea_impl_search import WikiIdeaSearch, WikiCodeSearch
#   
#   idea_search = WikiIdeaSearch()
#   results = idea_search.search("how to fine-tune LLMs efficiently")
#   
#   code_search = WikiCodeSearch()
#   results = code_search.search("QLoRA configuration for Llama")

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kapso.knowledge_base.search.base import KGSearchFilters, PageType, KGOutput


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class RetrievalItem:
    """
    Single item from retrieval.
    
    Attributes:
        title: Item title
        content: Main content text  
        score: Relevance score (0.0 to 1.0)
        item_type: Type of content (Principle, Implementation, etc.)
    """
    title: str
    content: str
    score: float = 0.0
    item_type: str = ""
    
    def to_string(self) -> str:
        """Format item for LLM context."""
        type_str = f" ({self.item_type})" if self.item_type else ""
        return f"### {self.title}{type_str}\n{self.content}"


@dataclass
class RetrievalResult:
    """
    Result from a retrieval tool search.
    
    Attributes:
        query: Original search query
        items: List of retrieved items
        source: Tool name that produced this result
    """
    query: str
    items: List[RetrievalItem] = field(default_factory=list)
    source: str = ""
    
    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def to_context_string(self, max_items: int = 5) -> str:
        """Format results as context for LLM prompts."""
        if self.is_empty:
            return f"No results found for: {self.query}"
        parts = [item.to_string() for item in self.items[:max_items]]
        return f"## {self.source} Results\n\n" + "\n\n---\n\n".join(parts)


# =============================================================================
# Wiki Search Tools
# =============================================================================

class WikiIdeaSearch:
    """
    Search wiki for ideas: Principles and Heuristics.
    
    Wraps KGGraphSearch with page_type filter for conceptual knowledge.
    Use in expand/select phase to find relevant principles and best practices.
    """
    
    # Page types representing "ideas"
    IDEA_TYPES = [PageType.PRINCIPLE.value, PageType.HEURISTIC.value]
    
    def __init__(
        self,
        kg_search: Optional["KGGraphSearch"] = None,
        use_llm_reranker: bool = True,
    ):
        """
        Initialize WikiIdeaSearch.
        
        Args:
            kg_search: Existing KGGraphSearch instance (creates via factory if None)
            use_llm_reranker: Whether to use LLM for reranking results
        """
        # Use factory to get params from config (weaviate_collection, etc.)
        if kg_search is None:
            from kapso.knowledge_base.search import KnowledgeSearchFactory
            kg_search = KnowledgeSearchFactory.create("kg_graph_search")
        self._kg_search = kg_search
        self._use_llm_reranker = use_llm_reranker
    
    @property
    def name(self) -> str:
        return "wiki_idea"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        domains: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Search for principles and heuristics in the wiki.
        
        Args:
            query: Search query (problem description, concept name, etc.)
            top_k: Maximum number of results
            min_score: Minimum relevance score threshold
            domains: Filter by domain tags
            context: Additional context for search
            
        Returns:
            RetrievalResult with matching principles and heuristics
        """
        # Build filters for idea types
        filters = KGSearchFilters(
            top_k=top_k,
            page_types=self.IDEA_TYPES,
            min_score=min_score,
            domains=domains,
            include_content=True,
        )
        
        # Delegate to KGGraphSearch
        kg_output = self._kg_search.search(
            query=query,
            filters=filters,
            context=context,
            use_llm_reranker=self._use_llm_reranker,
        )
        
        # Convert to RetrievalResult
        items = [
            RetrievalItem(
                title=r.page_title,
                content=r.content or r.overview,
                score=r.score,
                item_type=r.page_type,
            )
            for r in kg_output.results
        ]
        
        return RetrievalResult(query=query, items=items, source=self.name)


class WikiCodeSearch:
    """
    Search wiki for code: Implementations and Environments.
    
    Wraps KGGraphSearch with page_type filter for concrete code knowledge.
    Use in implement/debug phase to find code examples and environment configs.
    """
    
    # Page types representing "code"
    CODE_TYPES = [PageType.IMPLEMENTATION.value, PageType.ENVIRONMENT.value]
    
    def __init__(
        self,
        kg_search: Optional["KGGraphSearch"] = None,
        use_llm_reranker: bool = True,
    ):
        """
        Initialize WikiCodeSearch.
        
        Args:
            kg_search: Existing KGGraphSearch instance (creates via factory if None)
            use_llm_reranker: Whether to use LLM for reranking results
        """
        # Use factory to get params from config (weaviate_collection, etc.)
        if kg_search is None:
            from kapso.knowledge_base.search import KnowledgeSearchFactory
            kg_search = KnowledgeSearchFactory.create("kg_graph_search")
        self._kg_search = kg_search
        self._use_llm_reranker = use_llm_reranker
    
    @property
    def name(self) -> str:
        return "wiki_code"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        domains: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Search for implementations and environments in the wiki.
        
        Args:
            query: Search query (code task, library name, error message, etc.)
            top_k: Maximum number of results
            min_score: Minimum relevance score threshold
            domains: Filter by domain tags
            context: Additional context for search
            
        Returns:
            RetrievalResult with matching implementations and environments
        """
        # Build filters for code types
        filters = KGSearchFilters(
            top_k=top_k,
            page_types=self.CODE_TYPES,
            min_score=min_score,
            domains=domains,
            include_content=True,
        )
        
        # Delegate to KGGraphSearch
        kg_output = self._kg_search.search(
            query=query,
            filters=filters,
            context=context,
            use_llm_reranker=self._use_llm_reranker,
        )
        
        # Convert to RetrievalResult
        items = [
            RetrievalItem(
                title=r.page_title,
                content=r.content or r.overview,
                score=r.score,
                item_type=r.page_type,
            )
            for r in kg_output.results
        ]
        
        return RetrievalResult(query=query, items=items, source=self.name)
