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
        logger.info("CognitiveContextManager initialized")
    
    def get_context(self, budget_progress: float = 0) -> ContextData:
        """
        Get context for the coding agent.
        
        Uses TWO-STAGE RETRIEVAL:
        - Stage 1 (Planning): Workflow, Principles, Heuristics → additional_info
        - Stage 2 (Implementation): Code snippets, Implementation pages → kg_code_results
        
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
            logger.info(f"Initialized goal: {goal[:50]}...")
        
        # 3. Process last experiment result
        self._process_last_experiment()
        
        # 4. Generate briefing (Stage 1 output: workflow, principles, heuristics)
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
        
        # 6. Stage 2: Get implementation context (code snippets, implementation pages)
        kg_code_results = self._get_implementation_context()
        
        # Log context summary for debugging
        context_data = ContextData(
            problem=problem_desc,
            additional_info=additional_info,
            kg_results="",  # Included in briefing
            kg_code_results=kg_code_results
        )
        
        # Log what's being sent to the agent
        logger.info(f"  📤 Context prepared for agent:")
        logger.info(f"     Problem: {len(problem_desc)} chars")
        logger.info(f"     Workflow guidance: {len(additional_info)} chars")
        logger.info(f"     Implementation code: {len(kg_code_results)} chars")
        
        return context_data
    
    def _get_implementation_context(self) -> str:
        """
        Get implementation-level context.
        
        UNIFIED: Uses graph-traversed implementations from workflow if available.
        Falls back to semantic search only when no workflow.
        """
        goal = self.controller._goal or ""
        
        # Get workflow from controller's context (has graph-traversed implementations)
        workflow = None
        if self.controller._context and self.controller._context.workflow:
            workflow = self.controller._context.workflow
        
        # Pass workflow to get graph-based implementations
        impl_context = self.controller.retriever.get_implementation_context(
            goal, 
            workflow=workflow
        )
        
        if not impl_context:
            return ""
        
        lines = []
        
        # Add implementation summaries
        impls = impl_context.get("implementations", [])
        if impls:
            lines.append("## Implementation Reference")
            for impl in impls[:3]:
                title = impl.get("title", "Unknown")
                overview = impl.get("overview", "")
                lines.append(f"### {title}")
                if overview:
                    lines.append(overview)
                lines.append("")
        
        # Add code snippets
        snippets = impl_context.get("code_snippets", [])
        if snippets:
            lines.append("## Code Patterns")
            for i, snippet in enumerate(snippets[:4], 1):
                lines.append(f"**Pattern {i}:**")
                lines.append(f"```python\n{snippet.strip()}\n```")
                lines.append("")
        
        # Add environment requirements
        envs = impl_context.get("environment", [])
        if envs:
            lines.append("## Environment Requirements")
            for env in envs[:2]:
                title = env.get("title", "")
                reqs = env.get("requirements", "")
                if title:
                    lines.append(f"**{title}:** {reqs}")
        
        return "\n".join(lines)
    
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
                if len(goal_text) > 500:
                    return goal_text[:500] + "..."
                return goal_text
        
        # Fallback: Find first substantial paragraph (non-header, >20 chars)
        paragraphs = problem_desc.split("\n\n")
        for para in paragraphs:
            # Skip headers and short lines
            clean = para.strip()
            if clean and not clean.startswith('#') and len(clean) > 20:
                if len(clean) > 500:
                    return clean[:500] + "..."
                return clean
        
        # Last resort: use everything (truncated)
        if len(problem_desc) > 500:
            return problem_desc[:500] + "..."
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
