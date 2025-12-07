# Template Context Manager
#
# Copy this file to create a new context manager implementation.
#
# Steps:
#   1. Copy this file: cp _template.py my_context_manager.py
#   2. Rename the class and update the @register_context_manager name
#   3. Implement get_context() with your custom logic
#   4. Add presets in context_manager.yaml
#   5. That's it! Auto-discovered on import.

from typing import Any, Dict, Optional

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler


# Uncomment and rename to activate this context manager
# @register_context_manager("template")
class TemplateContextManager(ContextManager):
    """
    Template context manager - copy and customize.
    
    This template shows the minimal structure needed for a context manager.
    Copy this file and implement your custom logic.
    
    Params (add to context_manager.yaml):
        - example_param: Example parameter description
        - max_items: Maximum items to include
    """
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize template context manager.
        
        Args:
            problem_handler: Handler providing problem context
            search_strategy: Provider of experiment history
            knowledge_search: Optional knowledge search backend (KG LLM Navigation, RAG, etc.)
            params: Custom parameters from YAML config
        """
        super().__init__(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=params,
        )
        
        # =====================================================================
        # Extract your custom params from self.params
        # Always provide sensible defaults
        # =====================================================================
        self.example_param = self.params.get("example_param", "default")
        self.max_items = self.params.get("max_items", 5)

    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Gather context for solution generation.
        
        This is the main method to implement. It should:
        1. Get problem description from problem_handler
        2. Get experiment history from search_strategy
        3. Optionally enrich with knowledge_retriever
        4. Return ContextData with all gathered information
        
        Args:
            budget_progress: Current budget progress (0-100)
            
        Returns:
            ContextData with problem, history, and knowledge results
        """
        
        # =====================================================================
        # 1. Get problem description (budget-aware)
        # =====================================================================
        problem = self.problem_handler.get_problem_context(
            budget_progress=budget_progress
        )
        
        # =====================================================================
        # 2. Get experiment history from search strategy
        # =====================================================================
        # best_last=True: sorted by score (best at end)
        # best_last=False: chronological order (recent at end)
        experiment_history = self.search_strategy.get_experiment_history(best_last=True)
        experiment_history = experiment_history[-self.max_items:]
        
        # Format as string
        additional_info = "\n".join(str(exp) for exp in experiment_history)
        
        # =====================================================================
        # 3. Optional: Enrich with knowledge search
        # =====================================================================
        kg_results = ""
        kg_code_results = ""
        
        # Include additional context if available
        if len(self.problem_handler.additional_context) > 0:
            kg_results += self.problem_handler.additional_context + "\n\n"
        
        # Query knowledge search backend (if enabled)
        if self.knowledge_search.is_enabled():
            # Optionally provide context (e.g., last experiment)
            context = str(experiment_history[-1]) if experiment_history else None
            
            result = self.knowledge_search.search(
                query=problem,
                context=context,
            )
            
            if not result.is_empty:
                kg_results += result.text_results
                kg_code_results = result.code_results
        
        # =====================================================================
        # 4. Return ContextData
        # =====================================================================
        return ContextData(
            problem=problem,
            additional_info=additional_info,
            kg_results=kg_results,
            kg_code_results=kg_code_results,
        )

