from typing import Any, Dict, Optional, List
import logging

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler
from src.memory.controller import CognitiveController
from src.memory.types import WorkingMemory

logger = logging.getLogger(__name__)

@register_context_manager("cognitive")
class CognitiveContextManager(ContextManager):
    """
    Cognitive Context Manager 3.0.
    
    Uses the CognitiveController to synthesizing 'Briefings' instead of raw logs.
    Manages Working Memory (whiteboard) and Episodic Memory (insights).
    """
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=params,
        )
        
        # Initialize Controller
        self.controller = CognitiveController(
            knowledge_search=knowledge_search,
            episodic_store_path=self.params.get("episodic_store_path", ".memory_store.json")
        )
        
        # Initialize Working Memory (Whiteboard)
        # In a tree search, this might be overwritten by the Node's state,
        # but here we keep a default one.
        self.working_memory = WorkingMemory(
            current_goal="Solve the problem",
            active_plan=[]
        )
        
        # Track last error for meta-cognition
        self.last_error = None

    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Synthesize a Briefing Packet for the agent.
        """
        # 1. Update Goal from Problem Handler (it might have changed)
        problem_desc = self.problem_handler.get_problem_context(budget_progress=budget_progress)
        self.working_memory.current_goal = problem_desc[:500] + "..." # Truncate for sanity
        
        # 2. Get recent history to find last error
        # (The Controller handles the synthesis, we just need to provide the raw signal)
        recent_history = self.search_strategy.get_experiment_history(best_last=False)
        last_error = None
        if recent_history:
            last_exp = recent_history[-1]
            if last_exp.run_had_error:
                last_error = str(last_exp.error_details)
        
        # 3. Process the LAST result if we haven't yet
        # (This is a bit tricky: Orchestrator loop doesn't explicitly call 'process_result' on context manager.
        # Ideally, we should hook into the result processing. For now, we rely on the fact that
        # search strategy updates history.)
        if recent_history:
             self.working_memory, _ = self.controller.process_result(recent_history[-1], self.working_memory)

        # 4. Generate Briefing
        briefing = self.controller.prepare_briefing(self.working_memory, last_error=last_error)
        
        # 5. Return as ContextData
        # We put the Briefing in 'additional_info' which agents usually prepend to the prompt.
        return ContextData(
            problem=problem_desc, # Still pass full problem
            additional_info=briefing.to_string(),
            kg_results="", # Cleared because it's inside the briefing
            kg_code_results="" # Cleared because it's inside the briefing (or we can pass specific snippets)
        )

