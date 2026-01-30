# Basic Context Manager
#
# Simple context manager without any KG retrieval functionality.
# Provides problem description and experiment history only.
# Registered as "basic" via the factory decorator.

from typing import Any, Dict, Optional

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler


@register_context_manager("basic")
class BasicContextManager(ContextManager):
    """
    Basic context manager without KG retrieval.
    
    Gathers context from problem handler and experiment history only.
    No knowledge graph queries or enrichment.
    
    Sources:
    - Problem handler (problem description with budget awareness)
    - Search strategy (experiment history - best and recent)
    
    Use this when:
    - KG is not available or not needed
    - You want minimal context overhead
    - Testing without KG dependencies
    
    Params (defined in context_manager.yaml):
        - max_experiment_history_count: Max top experiments to include
        - max_recent_experiment_count: Max recent experiments to include
    """
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize basic context manager."""
        # Note: knowledge_search is accepted but ignored (for interface compatibility)
        super().__init__(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=params,
        )
        
        # Extract params with defaults
        self.max_experiment_history_count = self.params.get("max_experiment_history_count", 5)
        self.max_recent_experiment_count = self.params.get("max_recent_experiment_count", 5)

    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Gather context for solution generation (no KG enrichment).
        
        Args:
            budget_progress: Current budget progress (0-100)
            
        Returns:
            ContextData with problem and experiment history only
        """
        # Get problem description (budget-aware)
        problem = self.problem_handler.get_problem_context(budget_progress=budget_progress)
        
        # Get top experiments (sorted by score)
        experiment_history = self.search_strategy.get_experiment_history(best_last=True)
        experiment_history = experiment_history[-self.max_experiment_history_count:]
        
        # Get recent experiments (chronological)
        recent_experiment_history = self.search_strategy.get_experiment_history(best_last=False)
        recent_experiment_history = recent_experiment_history[-self.max_recent_experiment_count:]
        
        # Exclude duplicates (experiments that appear in both lists)
        recent_experiment_history = [
            exp for exp in recent_experiment_history 
            if exp not in experiment_history
        ]
        
        # Format history as additional context
        additional_info = ""
        
        if recent_experiment_history:
            additional_info += "## Previous Recent Experiments:\n"
            additional_info += "\n".join(str(exp) for exp in recent_experiment_history)
            additional_info += "\n\n"
        
        if experiment_history:
            additional_info += "## Previous Top Experiments:\n"
            additional_info += "\n".join(str(exp) for exp in experiment_history)
        
        # Include additional context from problem handler if available
        if self.problem_handler.additional_context:
            if additional_info:
                additional_info = self.problem_handler.additional_context + "\n\n" + additional_info
            else:
                additional_info = self.problem_handler.additional_context

        # Return context without KG results
        return ContextData(
            problem=problem,
            kg_results="",  # No KG retrieval
            kg_code_results="",  # No KG code retrieval
            additional_info=additional_info.strip(),
        )
