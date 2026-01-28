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
# Default Paths
# =============================================================================

# Default wiki directory and persist path for indexing/editing
DEFAULT_WIKI_DIR = Path("data/wikis")
DEFAULT_PERSIST_PATH = Path("data/indexes/wikis.json")


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
        wiki_dir: Path to directory containing wiki files (default: data/wikis)
        pages: Pre-parsed WikiPage objects
        persist_path: Where to save indexed data (default: data/indexes/wikis.json)
    
    Example:
        # Index with defaults (data/wikis -> data/indexes/wikis.json)
        search.index(KGIndexInput())
        
        # Index from custom directory
        input_data = KGIndexInput(
            wiki_dir="data/wikis/custom",
            persist_path="data/indexes/custom.json",
        )
        search.index(input_data)
        
        # Index pre-parsed pages (no wiki_dir needed)
        input_data = KGIndexInput(pages=[page1, page2, ...])
        search.index(input_data)
    """
    # Input mode 1: Directory of wiki files (default: data/wikis)
    wiki_dir: Optional[Union[str, Path]] = field(default=None)
    
    # Input mode 2: Pre-parsed pages
    pages: Optional[List[WikiPage]] = None
    
    # Persistence option (default: data/indexes/wikis.json)
    persist_path: Optional[Union[str, Path]] = field(default=None)
    
    # Internal flag to track if defaults should be used
    _use_defaults: bool = field(default=True, repr=False)
    
    def __post_init__(self):
        """Validate input and apply defaults."""
        # Apply defaults if neither wiki_dir nor pages provided
        if self.wiki_dir is None and self.pages is None and self._use_defaults:
            self.wiki_dir = DEFAULT_WIKI_DIR
        
        # Apply default persist_path if wiki_dir is used
        if self.persist_path is None and self.wiki_dir is not None:
            self.persist_path = DEFAULT_PERSIST_PATH
        
        # Validate: must have at least one input source
        if not self.wiki_dir and not self.pages:
            raise ValueError("Must provide either wiki_dir or pages")
        
        # Convert paths to Path objects
        if self.wiki_dir:
            self.wiki_dir = Path(self.wiki_dir)
        if self.persist_path:
            self.persist_path = Path(self.persist_path)


@dataclass
class KGEditInput:
    """
    Input for editing an existing wiki page.
    
    Updates all layers in sync: raw source files, JSON cache, Weaviate, and Neo4j.
    Only the fields provided (not None) will be updated.
    
    Attributes:
        page_id: Unique identifier of the page to edit (required)
        page_title: New title (optional)
        page_type: New page type (optional - may move file to different subdir)
        overview: New overview text (optional - triggers re-embedding)
        content: New full page content (optional - replaces entire file content)
        domains: New domain tags (optional, replaces existing)
        sources: New sources (optional, replaces existing)
        outgoing_links: New graph links (optional - triggers edge rebuild)
        auto_timestamp: Whether to auto-update last_updated (default: True)
        wiki_dir: Root wiki directory for source file updates (default: data/wikis)
        persist_path: Path to JSON cache file (default: data/indexes/wikis.json)
        update_source_files: Whether to update raw .md/.mediawiki files (default: True)
        update_persist_cache: Whether to update JSON cache (default: True)
    
    Example:
        # Edit with defaults (uses data/wikis and data/indexes/wikis.json)
        edit_input = KGEditInput(
            page_id="Workflow/QLoRA_Finetuning",
            overview="Updated overview text",
            domains=["LLMs", "Fine_Tuning", "PEFT"],
        )
        search.edit(edit_input)
        
        # Edit full content (replaces entire file)
        edit_input = KGEditInput(
            page_id="Workflow/QLoRA_Finetuning",
            content="... entire new file content ...",
        )
        search.edit(edit_input)
    """
    # Required: identifies the page
    page_id: str
    
    # Optional fields to update (None = no change)
    page_title: Optional[str] = None
    page_type: Optional[str] = None
    overview: Optional[str] = None
    content: Optional[str] = None  # Full file content replacement
    domains: Optional[List[str]] = None
    sources: Optional[List[Dict[str, str]]] = None
    outgoing_links: Optional[List[Dict[str, str]]] = None
    
    # Auto-update timestamp on edit
    auto_timestamp: bool = True
    
    # Source file tracking (with defaults)
    wiki_dir: Optional[Union[str, Path]] = field(default=None)
    persist_path: Optional[Union[str, Path]] = field(default=None)
    
    # Control which layers to update
    update_source_files: bool = True
    update_persist_cache: bool = True
    
    def __post_init__(self):
        """Validate input and apply defaults."""
        if not self.page_id:
            raise ValueError("page_id is required for editing")
        
        # Validate page_type if provided
        if self.page_type is not None:
            valid_types = PageType.values()
            if self.page_type not in valid_types:
                raise ValueError(
                    f"Invalid page_type '{self.page_type}'. "
                    f"Valid types: {valid_types}"
                )
        
        # Apply default paths
        if self.wiki_dir is None:
            self.wiki_dir = DEFAULT_WIKI_DIR
        if self.persist_path is None:
            self.persist_path = DEFAULT_PERSIST_PATH
        
        # Convert paths to Path objects
        self.wiki_dir = Path(self.wiki_dir)
        self.persist_path = Path(self.persist_path)
    
    @property
    def requires_reembedding(self) -> bool:
        """Check if edit requires regenerating embeddings."""
        # Overview is what gets embedded in Weaviate
        return self.overview is not None
    
    @property
    def requires_edge_rebuild(self) -> bool:
        """Check if edit requires rebuilding Neo4j edges."""
        return self.outgoing_links is not None
    
    @property
    def requires_file_rewrite(self) -> bool:
        """Check if edit requires rewriting the source file."""
        # Full content replacement always requires file rewrite
        if self.content is not None:
            return True
        # Metadata changes also require file update (they're embedded in file)
        return any([
            self.overview is not None,
            self.domains is not None,
            self.sources is not None,
            self.outgoing_links is not None,
            self.auto_timestamp,  # Timestamp is in file metadata
        ])
    
    def get_updates(self) -> Dict[str, Any]:
        """Get dict of non-None fields to update."""
        updates = {}
        for field_name in ['page_title', 'page_type', 'overview', 'content', 
                           'domains', 'sources', 'outgoing_links']:
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        return updates


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
# Index Metadata (for .index files)
# =============================================================================

@dataclass
class KGIndexMetadata:
    """
    Metadata stored in .index files.
    
    This dataclass represents the contents of an index file that tracks
    what data was indexed and where it's stored. Users create this once
    via index_kg() and load it on subsequent runs to skip re-indexing.
    
    Attributes:
        version: Schema version for forward compatibility
        created_at: ISO timestamp when index was created
        data_source: Path to source data (wiki_dir or data_path)
        search_backend: Name of the search backend used (e.g., "kg_graph_search")
        backend_refs: Backend-specific references (collection names, URIs, etc.)
        page_count: Number of pages/nodes indexed
    """
    version: str = "1.0"
    created_at: str = ""
    data_source: str = ""  # wiki_dir for kg_graph_search, data_path for kg_llm_navigation
    search_backend: str = ""
    backend_refs: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "data_source": self.data_source,
            "search_backend": self.search_backend,
            "backend_refs": self.backend_refs,
            "page_count": self.page_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGIndexMetadata":
        """Create KGIndexMetadata from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            data_source=data.get("data_source", ""),
            search_backend=data.get("search_backend", ""),
            backend_refs=data.get("backend_refs", {}),
            page_count=data.get("page_count", 0),
        )


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
    
    @property
    def text_results(self) -> str:
        """
        Get formatted text results for context managers.
        
        This is an alias for to_context_string() to maintain compatibility
        with context managers that expect this property.
        """
        return self.to_context_string()
    
    @property
    def code_results(self) -> str:
        """
        Get code/implementation results for context managers.
        
        Extracts Implementation-type results and formats them as code context.
        """
        code_items = self.get_by_type("Implementation")
        if not code_items:
            # Also check for "code" type (used by kg_llm_navigation)
            code_items = [r for r in self.results if r.page_type.lower() == "code"]
        
        if not code_items:
            return ""
        
        return "\n\n".join(
            f"## {item.page_title}\n{item.content}" for item in code_items
        )
    
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
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize knowledge search.
        
        Args:
            params: Implementation-specific parameters
        """
        self.enabled = enabled
        self.params = params or {}
        self._enabled = self.params.get("enabled", True)

    def is_enabled(self) -> bool:
        """Check if this knowledge search backend is enabled."""
        return self._enabled
    
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
    
    @abstractmethod
    def edit(self, data: "KGEditInput") -> bool:
        """
        Edit an existing wiki page and update all storage layers.
        
        Updates in order:
        1. Raw source file (.md or .mediawiki) - if update_source_files=True
        2. Persist JSON cache - if update_persist_cache=True
        3. Weaviate (embeddings + properties)
        4. Neo4j (node properties + edges)
        5. Internal memory cache
        
        Args:
            data: KGEditInput with page_id and fields to update
            
        Returns:
            True if edit was successful, False if page not found
            
        Notes:
            - Only non-None fields in data are updated
            - If overview is changed, embeddings are regenerated in Weaviate
            - If outgoing_links is changed, edges are rebuilt in Neo4j
            - If auto_timestamp=True, last_updated is set to current time
        
        Example:
            success = search.edit(KGEditInput(
                page_id="Workflow/QLoRA_Finetuning",
                content="Updated tutorial content...",
                domains=["LLMs", "Fine_Tuning", "Memory_Efficient"],
            ))
        """
        pass
    
    def add_page(
        self,
        page: "WikiPage",
        wiki_dir: Path,
        persist_path: Optional[Path] = None,
    ) -> bool:
        """
        Add a new page to all storage layers.
        
        Creates (in order):
        1. Raw source file (.md) in wiki_dir
        2. Persist JSON cache entry (if persist_path provided)
        3. Weaviate (embeddings + properties)
        4. Neo4j (node + edges)
        5. Internal memory cache
        
        Args:
            page: WikiPage object to add
            wiki_dir: Directory to write the source .md file
            persist_path: Optional path to JSON cache file
            
        Returns:
            True if successful, False on failure
            
        Notes:
            - Use this for NEW pages only
            - For existing pages, use edit() instead
            - Subclasses should override this method
        
        Example:
            page = WikiPage(
                id="Principle/My_New_Concept",
                page_title="My New Concept",
                page_type="Principle",
                overview="A brief overview...",
                content="Full content...",
            )
            success = search.add_page(page, Path("data/wikis"))
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("add_page not implemented for this backend")
    
    def page_exists(self, page_id: str) -> bool:
        """
        Check if a page exists in the knowledge graph.
        
        Args:
            page_id: Page ID to check (e.g., "Principle/My_Concept")
            
        Returns:
            True if page exists, False otherwise
        """
        # Default implementation - subclasses should override
        return False
    
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
    
    def get_backend_refs(self) -> Dict[str, Any]:
        """
        Return backend-specific references for index file.
        
        These references identify WHERE the indexed data is stored
        (e.g., Weaviate collection name, Neo4j database).
        Used when saving/loading .index files.
        
        Returns:
            Dictionary of backend-specific identifiers
            
        Example (kg_graph_search):
            {"weaviate_collection": "KapsoKG", "embedding_model": "text-embedding-3-large"}
            
        Example (kg_llm_navigation):
            {"neo4j_uri": "bolt://localhost:7687", "node_label": "Node"}
        """
        return {}
    
    def validate_backend_data(self) -> bool:
        """
        Check if the backend has indexed data.
        
        Used when loading from an .index file to verify that the
        backend actually contains the expected data.
        
        Returns:
            True if backend has indexed data, False otherwise
        """
        return True
    
    def get_indexed_count(self) -> int:
        """
        Get the number of indexed pages/nodes.
        
        Returns:
            Count of indexed items, or 0 if unknown
        """
        return 0


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
    
    def edit(self, data: KGEditInput) -> bool:
        """No-op edit - always returns False."""
        return False
    
    def add_page(
        self,
        page: WikiPage,
        wiki_dir: Path,
        persist_path: Optional[Path] = None,
    ) -> bool:
        """No-op add_page - always returns False."""
        return False
    
    def page_exists(self, page_id: str) -> bool:
        """No pages exist in null search."""
        return False
