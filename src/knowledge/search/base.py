# Knowledge Search Base
#
# Data structures and abstract interface for knowledge search backends.
# Each implementation handles both indexing and searching:
# - Knowledge Graph (Neo4j) with LLM navigation
# - RAG (Vector embeddings) - future
# - External APIs - future

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums and Constants
# =============================================================================

class PageType(str, Enum):
    """
    Wiki page types in the Knowledge Graph.
    
    Follows the Top-Down DAG structure:
    Workflow -> Principle -> Implementation -> Environment/Heuristic
    """
    WORKFLOW = "Workflow"           # The Recipe - ordered sequence of steps
    PRINCIPLE = "Principle"         # The Theory - library-agnostic concepts
    IMPLEMENTATION = "Implementation"  # The Code - concrete syntax/API
    ENVIRONMENT = "Environment"     # The Context - hardware/OS/dependencies
    HEURISTIC = "Heuristic"         # The Wisdom - tips, optimizations, tricks
    
    @classmethod
    def values(cls) -> List[str]:
        """Return all page type values."""
        return [e.value for e in cls]


# =============================================================================
# Search Input Data Structures
# =============================================================================

@dataclass
class KGSearchFilters:
    """
    Filters for Knowledge Graph search.
    
    Used to narrow down search results by various criteria.
    All filters are optional - None means no filtering on that field.
    
    Attributes:
        top_k: Maximum number of results to return (default: 10)
        min_score: Minimum relevance score threshold (0.0 to 1.0)
        page_types: Filter by page types (e.g., ["Workflow", "Principle"])
        domains: Filter by domain tags (e.g., ["Deep_Learning", "NLP"])
        include_content: Whether to include full page content (default: True)
    
    Example:
        # Get top 5 Workflow and Principle pages in NLP domain
        filters = KGSearchFilters(
            top_k=5,
            min_score=0.5,
            page_types=[PageType.WORKFLOW, PageType.PRINCIPLE],
            domains=["NLP", "Deep_Learning"],
        )
    """
    top_k: int = 10
    min_score: Optional[float] = None
    page_types: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    include_content: bool = True
    
    def __post_init__(self):
        """Validate and normalize filter values."""
        # Normalize page_types to string values
        if self.page_types:
            self.page_types = [
                pt.value if isinstance(pt, PageType) else pt 
                for pt in self.page_types
            ]
        
        # Validate min_score range
        if self.min_score is not None:
            if not 0.0 <= self.min_score <= 1.0:
                raise ValueError(f"min_score must be between 0.0 and 1.0, got {self.min_score}")
        
        # Validate top_k
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


# =============================================================================
# Search Result Data Structures
# =============================================================================

@dataclass
class KGResultItem:
    """
    Single result item from a Knowledge Graph search.
    
    Follows common patterns from Qdrant, Weaviate, Pinecone.
    
    Attributes:
        id: Unique identifier for the page/node
        score: Relevance score (higher = more relevant, 0.0 to 1.0)
        page_title: Title of the wiki page
        page_type: Node type (Workflow, Principle, Implementation, Environment, Heuristic)
        overview: Brief summary/description (the "card" content)
        content: Full page content (may be empty if include_content=False)
        metadata: Additional structured data (domains, sources, last_updated, etc.)
    """
    id: str
    score: float
    page_title: str
    page_type: str
    overview: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def domains(self) -> List[str]:
        """Get domain tags from metadata."""
        return self.metadata.get("domains", [])
    
    @property
    def sources(self) -> List[Dict[str, str]]:
        """Get knowledge sources from metadata."""
        return self.metadata.get("sources", [])
    
    @property
    def last_updated(self) -> Optional[str]:
        """Get last updated timestamp from metadata."""
        return self.metadata.get("last_updated")
    
    def __repr__(self) -> str:
        return f"KGResultItem(id={self.id!r}, score={self.score:.3f}, title={self.page_title!r}, type={self.page_type!r})"


@dataclass
class KGOutput:
    """
    Output from a Knowledge Graph search.
    
    Contains the original query, filters used, and a ranked list of result items.
    
    Attributes:
        query: Original search query
        filters: Filters applied to the search (for reference)
        results: List of KGResultItem, ordered by score (descending)
        total_found: Total number of matching results (before filters/limit)
        search_metadata: Information about the search itself (time, params, etc.)
    """
    query: str
    filters: Optional[KGSearchFilters] = None
    results: List[KGResultItem] = field(default_factory=list)
    total_found: int = 0
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_empty(self) -> bool:
        """Check if no results were found."""
        return len(self.results) == 0
    
    @property
    def top_result(self) -> Optional[KGResultItem]:
        """Get the highest-scoring result, or None if empty."""
        return self.results[0] if self.results else None
    
    def get_by_type(self, page_type: str) -> List[KGResultItem]:
        """Filter results by page type."""
        type_val = page_type.value if isinstance(page_type, PageType) else page_type
        return [r for r in self.results if r.page_type == type_val]
    
    def get_by_domain(self, domain: str) -> List[KGResultItem]:
        """Filter results by domain tag."""
        return [r for r in self.results if domain in r.domains]
    
    def get_above_score(self, min_score: float) -> List[KGResultItem]:
        """Filter results above a minimum score threshold."""
        return [r for r in self.results if r.score >= min_score]
    
    def to_context_string(self, max_results: int = 5, include_content: bool = True) -> str:
        """
        Format results as a context string for LLM prompts.
        
        Args:
            max_results: Maximum number of results to include
            include_content: Whether to include full content or just overview
        
        Returns:
            Formatted text with titles, overviews, and optionally content.
        """
        if self.is_empty:
            return "No relevant knowledge found."
        
        parts = []
        for item in self.results[:max_results]:
            if include_content and item.content:
                parts.append(
                    f"## {item.page_title} ({item.page_type})\n"
                    f"**Overview:** {item.overview}\n\n"
                    f"{item.content}"
                )
            else:
                parts.append(
                    f"## {item.page_title} ({item.page_type})\n"
                    f"{item.overview}"
                )
        return "\n\n---\n\n".join(parts)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)


# =============================================================================
# Abstract Base Class
# =============================================================================

class KnowledgeSearch(ABC):
    """
    Abstract base class for knowledge search backends.
    
    Each implementation handles both indexing and searching,
    keeping related functionality together.
    
    Subclasses must implement:
    - index(): Load knowledge into the backend
    - search(): Query for relevant knowledge
    
    To create a new search backend:
    1. Subclass KnowledgeSearch
    2. Implement index() and search()
    3. Register with @register_knowledge_search("your_name") decorator
    4. Add configuration presets in knowledge_search.yaml
    """
    
    def __init__(
        self,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize knowledge search.
        
        Args:
            enabled: Whether the search backend is active
            params: Implementation-specific parameters
        """
        self.enabled = enabled
        self.params = params or {}
    
    @abstractmethod
    def index(self, data: Dict[str, Any]) -> None:
        """
        Index knowledge into the backend.
        
        Args:
            data: Knowledge data to index (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
    ) -> KGOutput:
        """
        Search for relevant knowledge.
        
        Args:
            query: The search query (typically problem description)
            filters: Optional filters for results (top_k, min_score, page_types, domains)
            context: Optional additional context (e.g., last experiment)
            
        Returns:
            KGOutput with ranked and filtered results
        
        Example:
            # Search with filters
            result = search.search(
                query="How to fine-tune transformers?",
                filters=KGSearchFilters(
                    top_k=5,
                    min_score=0.6,
                    page_types=["Workflow", "Principle"],
                    domains=["NLP", "Deep_Learning"],
                ),
                context="Previous experiment used BERT",
            )
        """
        pass
    
    def clear(self) -> None:
        """
        Clear all indexed data.
        
        Optional - subclasses may override if supported.
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if search backend is enabled."""
        return self.enabled
    
    def close(self) -> None:
        """
        Clean up resources.
        
        Optional - subclasses may override to close connections.
        """
        pass


class NullKnowledgeSearch(KnowledgeSearch):
    """
    Null implementation that returns empty results.
    
    Used when knowledge search is disabled.
    """
    
    def __init__(self):
        """Initialize null search (always disabled)."""
        super().__init__(enabled=False)
    
    def index(self, data: Dict[str, Any]) -> None:
        """No-op index."""
        pass
    
    def search(
        self, 
        query: str, 
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
    ) -> KGOutput:
        """Return empty results."""
        return KGOutput(query=query, filters=filters)
    
    def is_enabled(self) -> bool:
        """Always disabled."""
        return False
