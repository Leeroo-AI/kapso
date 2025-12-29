# =============================================================================
# Cognitive Context Manager - Workflow-Aware Context Generation
# =============================================================================
#
# Uses CognitiveController to provide workflow-aware context to the agent.
# 
# For legacy behavior (no workflow tracking), use `kg_enriched` context manager
# in config:
#   context_manager:
#     type: kg_enriched
#
# For workflow-aware behavior:
#   context_manager:
#     type: cognitive
# =============================================================================

from typing import Any, Dict, Optional
import logging

from src.execution.context_manager.types import ContextData, ExperimentHistoryProvider
from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.factory import register_context_manager
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler
from src.memory.cognitive_controller import CognitiveController

logger = logging.getLogger(__name__)


@register_context_manager("cognitive")
class CognitiveContextManager(ContextManager):
    """
    Cognitive Context Manager with full workflow tracking.
    
    Uses CognitiveController to:
    - Retrieve/synthesize workflows from KG
    - Track step-by-step progress  
    - Generate briefings with step-specific heuristics
    
    Config example:
        context_manager:
          type: cognitive
          params:
            max_step_retries: 3
            allow_skip: true
            allow_pivot: true
            
    For legacy (no workflow), use kg_enriched type instead.
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
        
        # Initialize controller (only pass valid params)
        self.controller = CognitiveController(
            knowledge_search=self.knowledge_search,
            episodic_store_path=self.params.get("episodic_store_path", ".memory_store.json"),
            decision_model=self.params.get("decision_model"),
        )
        
        self._goal_initialized = False
        self._last_action: str = ""  # Track last decision action
        # Guardrail: `get_context()` may be called multiple times per iteration by
        # some orchestrators/strategies. Without this, we'd re-process the same
        # last experiment repeatedly (duplicating insights and inflating failure
        # counters).
        self._last_processed_experiment_branch: Optional[str] = None
        logger.info("CognitiveContextManager initialized")
    
    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Get context for the coding agent.
        
        Uses TWO-STAGE RETRIEVAL:
        - All knowledge (workflow, principles, code, heuristics) → additional_info
        - kg_code_results is empty (code already in additional_info, no duplication)
        
        On first call: initializes goal and retrieves workflow from KG
        On subsequent calls: updates based on experiment results
        """
        # 1. Get problem context
        problem_desc = self.problem_handler.get_problem_context(budget_progress=budget_progress)
        
        # 2. Initialize goal on first call (triggers Stage 1: Planning retrieval)
        if not self._goal_initialized:
            goal = self._extract_goal(problem_desc)
            self.controller.initialize_goal(goal)
            self._goal_initialized = True
            logger.info(f"Initialized goal: {goal}")
        
        # 3. Process last experiment result
        self._process_last_experiment()
        
        # 4. Generate briefing (contains full KGKnowledge.render() output)
        briefing = self.controller.prepare_briefing()
        
        # 5. Add workflow progress info
        progress = self.controller.get_workflow_progress()
        additional_info = briefing.to_string()
        
        if progress.get("has_workflow"):
            workflow_status = f"""
## Workflow Progress
- Workflow: {progress.get('title', 'Unknown')}
- Source: {progress.get('source', 'Unknown')}
- Steps: {progress.get('total_steps', '?')} total
"""
            additional_info = workflow_status + "\n" + additional_info
        
        # 6. Code is already in additional_info via KGKnowledge.render()
        # Don't duplicate it in kg_code_results
        kg_code_results = ""
        
        context_data = ContextData(
            problem=problem_desc,
            additional_info=additional_info,
            kg_results="",  # Included in briefing
            kg_code_results=kg_code_results
        )
        
        # Log what's being sent to the agent
        logger.info(f"  📤 Context prepared for agent:")
        logger.info(f"     Problem: {len(problem_desc)} chars")
        logger.info(f"     KG guidance (includes code): {len(additional_info)} chars")
        
        return context_data
    
    
    def _process_last_experiment(self):
        """
        Process the last experiment result.
        
        Passes ALL available data to the controller:
        - success/error status
        - score from evaluator
        - feedback from evaluator (LLM judge explanation, etc.)
        - execution output
        - solution code that was tried
        
        Stores the decision action for orchestrator to check.
        """
        recent_history = self.search_strategy.get_experiment_history(best_last=False)
        if not recent_history:
            return
        
        last_exp = recent_history[-1]
        
        # Do not process the same experiment twice.
        if self._last_processed_experiment_branch == last_exp.branch_name:
            return
        self._last_processed_experiment_branch = last_exp.branch_name
        
        # Extract all available data from ExperimentResult
        success = not last_exp.had_error
        error_msg = last_exp.error_message if last_exp.had_error else None
        
        action, details = self.controller.process_result(
            success=success,
            error_message=error_msg,
            experiment_id=last_exp.branch_name,
            score=last_exp.score,
            feedback=last_exp.feedbacks,      # Evaluator feedback
            output=last_exp.output,            # Execution output
            solution=last_exp.solution,        # Code that was tried
        )
        
        # Store action so orchestrator can check for COMPLETE
        self._last_action = action
        
        logger.info(f"Decision: {action}, score={last_exp.score}")
        logger.debug(f"Decision details: {details}")
    
    def _extract_goal(self, problem_desc: str) -> str:
        """
        Extract goal from problem description.
        
        Handles various formats:
        - "# Goal\n\nActual goal text..."
        - "# Problem Description\n# Goal\n\nActual goal text..."
        - Plain text (uses first substantial paragraph)
        
        Returns the actual goal content, not just headers.
        """
        import re
        
        # Try to find content after "# Goal" header
        goal_match = re.search(r'#\s*Goal\s*\n+(.+?)(?:\n#|\n\n#|$)', problem_desc, re.DOTALL)
        if goal_match:
            goal_text = goal_match.group(1).strip()
            # If we got actual content, use it
            if len(goal_text) > 20:
                return goal_text
        
        # Fallback: Find first substantial paragraph (non-header, >20 chars)
        paragraphs = problem_desc.split("\n\n")
        for para in paragraphs:
            # Skip headers and short lines
            clean = para.strip()
            if clean and not clean.startswith('#') and len(clean) > 20:
                return clean
        
        # Last resort: use everything (no truncation)
        return problem_desc
    
    # =========================================================================
    # Status Methods
    # =========================================================================
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        return self.controller.get_workflow_progress()
    
    def is_workflow_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.controller.is_complete()
    
    def should_stop(self) -> bool:
        """
        Check if LLM decided to stop (COMPLETE action).
        
        Returns True if the last decision was COMPLETE.
        Orchestrator should check this after get_context() to stop early.
        """
        return self._last_action == "complete"
    
    def get_last_action(self) -> str:
        """Get the last decision action (retry/pivot/complete)."""
        return self._last_action
    
    def close(self):
        """Cleanup resources."""
        if self.controller:
            self.controller.close()
