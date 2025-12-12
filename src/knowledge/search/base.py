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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
# Index Input Data Structures
# =============================================================================

@dataclass
class WikiPage:
    """
    Parsed wiki page ready for indexing.
    
    Represents a single wiki page with all metadata extracted.
    Maps to the wiki structure defined in src/knowledge/wiki_structure/.
    
    Attributes:
        id: Unique identifier (e.g., "allenai_allennlp/Model_Training")
        page_title: Human-readable title
        page_type: PageType value (Workflow, Principle, Implementation, etc.)
        overview: Brief summary/description (the "card" content)
        content: Full page content
        domains: Domain tags (e.g., ["Deep_Learning", "NLP"])
        sources: Knowledge sources (repo URLs, papers, etc.)
        last_updated: Last update timestamp
        outgoing_links: Graph connections parsed from [[edge::Type:Target]] syntax
    """
    id: str
    page_title: str
    page_type: str
    overview: str
    content: str
    domains: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    last_updated: Optional[str] = None
    outgoing_links: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "page_title": self.page_title,
            "page_type": self.page_type,
            "overview": self.overview,
            "content": self.content,
            "domains": self.domains,
            "sources": self.sources,
            "last_updated": self.last_updated,
            "outgoing_links": self.outgoing_links,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WikiPage":
        """Create WikiPage from dictionary."""
        return cls(
            id=data["id"],
            page_title=data["page_title"],
            page_type=data["page_type"],
            overview=data["overview"],
            content=data["content"],
            domains=data.get("domains", []),
            sources=data.get("sources", []),
            last_updated=data.get("last_updated"),
            outgoing_links=data.get("outgoing_links", []),
        )
    
    def __repr__(self) -> str:
        return f"WikiPage(id={self.id!r}, type={self.page_type!r}, title={self.page_title!r})"


@dataclass
class KGIndexInput:
    """
    Input for Knowledge Graph indexing.
    
    Supports two input modes:
    - wiki_dir: Path to directory of .mediawiki files (will be parsed)
    - pages: Pre-parsed WikiPage objects
    
    Attributes:
        wiki_dir: Path to directory containing .mediawiki files
        pages: Pre-parsed WikiPage objects
        persist_path: Where to save indexed data for later loading
    
    Example:
        # Index from directory
        input_data = KGIndexInput(
            wiki_dir="data/wikis/allenai_allennlp",
            persist_path="data/indexes/allenai_allennlp.json",
        )
        search.index(input_data)
        
        # Index pre-parsed pages
        input_data = KGIndexInput(pages=[page1, page2, ...])
        search.index(input_data)
    """
    # Input mode 1: Directory of wiki files
    wiki_dir: Optional[Union[str, Path]] = None
    
    # Input mode 2: Pre-parsed pages
    pages: Optional[List[WikiPage]] = None
    
    # Persistence option
    persist_path: Optional[Union[str, Path]] = None
    
    def __post_init__(self):
        """Validate input."""
        if not self.wiki_dir and not self.pages:
            raise ValueError("Must provide either wiki_dir or pages")
        
        # Convert paths to Path objects
        if self.wiki_dir:
            self.wiki_dir = Path(self.wiki_dir)
        if self.persist_path:
            self.persist_path = Path(self.persist_path)


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
    - index(): Load wiki pages into the backend
    - search(): Query for relevant knowledge
    
    To create a new search backend:
    1. Subclass KnowledgeSearch
    2. Implement index() and search()
    3. Register with @register_knowledge_search("your_name") decorator
    4. Add configuration presets in knowledge_search.yaml
    """
    
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize knowledge search.
        
        Args:
            params: Implementation-specific parameters
        """
        self.params = params or {}
    
    @abstractmethod
    def index(self, data: KGIndexInput) -> None:
        """
        Index wiki pages into the backend.
        
        Args:
            data: KGIndexInput with wiki_dir or pages
            
        If data.wiki_dir is provided, parses .mediawiki files from the directory.
        If data.pages is provided, indexes the pre-parsed WikiPage objects.
        If data.persist_path is set, saves the index for later loading.
        
        Example:
            # Index from directory
            search.index(KGIndexInput(
                wiki_dir="data/wikis/allenai_allennlp",
                persist_path="data/indexes/allenai_allennlp.json",
            ))
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> KGOutput:
        """
        Search for relevant knowledge.
        
        Args:
            query: The search query (typically problem description)
            filters: Optional filters for results (top_k, min_score, page_types, domains)
            context: Optional additional context (e.g., last experiment)
            **kwargs: Implementation-specific options (e.g., use_llm_reranker)
            
        Returns:
            KGOutput with ranked and filtered results
        
        Example:
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
    
    @abstractmethod
    def get_page(self, page_title: str) -> Optional[WikiPage]:
        """
        Retrieve a wiki page by its title.
        
        This is a direct lookup, not a search. Given an exact page title,
        returns the complete WikiPage with all content.
        
        Args:
            page_title: Exact title of the page to retrieve
            
        Returns:
            WikiPage if found, None otherwise
            
        Example:
            page = search.get_page("Model_Training")
            if page:
                print(page.content)
        """
        pass
    
    def clear(self) -> None:
        """
        Clear all indexed data.
        
        Optional - subclasses may override if supported.
        """
        pass
    
    def close(self) -> None:
        """
        Clean up resources.
        
        Optional - subclasses may override to close connections.
        """
        pass


class NullKnowledgeSearch(KnowledgeSearch):
    """
    Null implementation that returns empty results.
    
    Used when you explicitly want a no-op search backend.
    """
    
    def __init__(self):
        """Initialize null search."""
        super().__init__()
    
    def index(self, data: KGIndexInput) -> None:
        """No-op index."""
        pass
    
    def search(
        self, 
        query: str, 
        filters: Optional[KGSearchFilters] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> KGOutput:
        """Return empty results."""
        return KGOutput(query=query, filters=filters)
    
    def get_page(self, page_title: str) -> Optional[WikiPage]:
        """Return None (no pages in null search)."""
        return None
