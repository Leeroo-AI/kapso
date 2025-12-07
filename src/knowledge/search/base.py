# Knowledge Search Base
#
# Abstract interface for knowledge search backends.
# Each implementation handles both indexing and searching:
# - Knowledge Graph (Neo4j) with LLM navigation
# - RAG (Vector embeddings) - future
# - External APIs - future

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KnowledgeResult:
    """
    Result from knowledge search.
    
    Attributes:
        text_results: General knowledge/information text
        code_results: Code snippets and examples
        metadata: Optional metadata about the search
    """
    text_results: str = ""
    code_results: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_empty(self) -> bool:
        """Check if results are empty."""
        return not self.text_results and not self.code_results


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
    def search(self, query: str, context: Optional[str] = None) -> KnowledgeResult:
        """
        Search for relevant knowledge.
        
        Args:
            query: The search query (typically problem description)
            context: Optional additional context (e.g., last experiment)
            
        Returns:
            KnowledgeResult with text and code results
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
    
    def search(self, query: str, context: Optional[str] = None) -> KnowledgeResult:
        """Return empty results."""
        return KnowledgeResult()
    
    def is_enabled(self) -> bool:
        """Always disabled."""
        return False

