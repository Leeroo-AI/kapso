# Knowledge-Enriched Context Manager
#
# Context manager that enriches context with knowledge search.
# Registered as "kg_enriched" via the factory decorator.

from typing import Any, Dict, Optional

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler


@register_context_manager("kg_enriched")
class KGEnrichedContextManager(ContextManager):
    """
    Knowledge-enriched context manager.
    
    Gathers context from multiple sources and enriches it with
    knowledge from an injected search backend (KG, RAG, etc.).
    
    Sources:
    - Problem handler (problem description with budget awareness)
    - Search strategy (experiment history - best and recent)
    - Knowledge search (injected - can be KG LLM Navigation, RAG, or null)
    
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
        """Initialize knowledge-enriched context manager."""
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
        Gather and enrich context for solution generation.
        
        Args:
            budget_progress: Current budget progress (0-100)
            
        Returns:
            ContextData with problem, history, and knowledge results
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
        additional_info = (
            "## Previous Recent Experiments:\n" 
            + "\n".join(str(exp) for exp in recent_experiment_history) 
            + "\n## Previous Top Experiments:\n" 
            + "\n".join(str(exp) for exp in experiment_history) 
        )

        # Enrich with knowledge from injected search backend
        kg_results = ""
        kg_code_results = ""
        
        # Include additional context if available
        if len(self.problem_handler.additional_context) > 0:
            kg_results += self.problem_handler.additional_context + "\n\n"
        
        # Query knowledge search (any implementation: KG LLM Navigation, RAG, etc.)
        if self.knowledge_search.is_enabled():
            last_exp_context = str(experiment_history[-1]) if experiment_history else None
            
            knowledge_result = self.knowledge_search.search(
                query=problem,
                context=last_exp_context,
            )
            
            if not knowledge_result.is_empty:
                # Format results as context strings
                kg_results += knowledge_result.to_context_string()
                # Extract Implementation-type results for code
                code_items = knowledge_result.get_by_type("Implementation")
                if code_items:
                    kg_code_results = "\n\n".join(
                        f"## {item.page_title}\n{item.content}" for item in code_items
                    )

        return ContextData(
            problem=problem,
            kg_results=kg_results,
            kg_code_results=kg_code_results,
            additional_info=additional_info,
        )
