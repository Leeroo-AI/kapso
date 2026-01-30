# Context Manager Base
#
# Abstract base class for context managers.
# Each implementation defines its own params structure.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.knowledge.search.base import (
    KnowledgeSearch, 
    NullKnowledgeSearch,
)
from src.environment.handlers.base import ProblemHandler

# Re-export ContextData for convenience
__all__ = ["ContextManager", "ContextData"]


class ContextManager(ABC):
    """
    Abstract base class for context managers.
    
    Each implementation:
    - Accepts dependencies via constructor
    - Defines its own params parsing
    - Implements get_context()
    
    To create a new context manager:
    1. Subclass ContextManager
    2. Implement get_context()
    3. Register with @register_context_manager("your_name") decorator
    4. Add configuration presets in context_manager.yaml
    """
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize context manager.
        
        Args:
            problem_handler: Handler providing problem context
            search_strategy: Provider of experiment history
            knowledge_search: Optional search backend for knowledge enrichment
            params: Implementation-specific parameters (from YAML)
        """
        self.problem_handler = problem_handler
        self.search_strategy = search_strategy
        self.knowledge_search = knowledge_search or NullKnowledgeSearch()
        self.params = params or {}
    
    @abstractmethod
    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Gather context for solution generation.
        
        Args:
            budget_progress: Current budget progress (0-100)
            
        Returns:
            ContextData with problem, history, and knowledge results
        """
        pass

    def should_stop(self) -> bool:
        """
        Check if the context manager wants to stop the experiment loop.
        
        Override in subclasses that support decision-based stopping.
        Default returns False (never stop based on context manager).
        
        Returns:
            True if experiments should stop (e.g., LLM decided COMPLETE)
        """
        return False