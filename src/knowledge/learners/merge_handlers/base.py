# Merge Handler Base
#
# Abstract base class for type-specific merge handlers.
# Each page type has different merge semantics.

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.knowledge.search.base import WikiPage


class MergeHandler(ABC):
    """
    Base class for type-specific merge handlers.
    
    Each page type has different merge semantics:
    - Workflows: Match by goal/process, merge steps
    - Principles: Match by theoretical concept, merge explanations
    - Implementations: Match by API/function, merge examples
    - Environments: Match by tech stack, merge dependencies
    - Heuristics: Match by problem domain, avoid contradictions
    
    The handler provides:
    1. Search query builder - how to find related pages
    2. Merge instructions - guidance for the agent
    3. Search filters - default filters for this type
    """
    
    @property
    @abstractmethod
    def page_type(self) -> str:
        """Page type this handler processes (e.g., 'Workflow')."""
        pass
    
    @property
    @abstractmethod
    def merge_instructions(self) -> str:
        """
        Type-specific instructions for the agent.
        
        Should include:
        - Search strategy
        - Merge decision criteria
        - Quality checks
        """
        pass
    
    @abstractmethod
    def build_search_query(self, page: WikiPage) -> str:
        """
        Build search query to find related pages in KG.
        
        Args:
            page: The proposed page to find matches for
            
        Returns:
            Search query string
        """
        pass
    
    def get_search_filters(self) -> Dict[str, Any]:
        """
        Get default search filters for this type.
        
        Returns:
            Dict with page_types, top_k, min_score
        """
        return {
            "page_types": [self.page_type],
            "top_k": 5,
            "min_score": 0.5,
        }

