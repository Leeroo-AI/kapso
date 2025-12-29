# =============================================================================
# Cognitive Controller - Workflow-Aware Context Management
# =============================================================================
#
# Key principles:
# 1. ALL decisions are made by the LLM, not by rules
# 2. SINGLE unified context (CognitiveContext.render()) goes to both agent and decision maker
# 3. LLM extracts GENERALIZED insights from errors (not raw error text)
# 4. LLM governs episodic retrieval (goal/step-aware queries)
#
# The controller:
# 1. Retrieves/synthesizes workflows from KG (with heuristics pre-loaded)
# 2. Tracks step-by-step progress
# 3. Calls LLM to decide what action to take after each experiment
# 4. Uses LLM to extract reusable insights from errors/successes
# 5. Uses LLM to retrieve relevant episodic memories
# =============================================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from src.memory.context import (
    CognitiveContext,
    ExperimentState, KGRetrievalState, EpisodicState, InsightSummary, MetaState
)
from src.memory.decisions import DecisionMaker, WorkflowAction, ActionDecision
from src.memory.knowledge_retriever import KnowledgeRetriever
from src.memory.kg_types import KGKnowledge, KGTier  # NEW: KGKnowledge types
from src.memory.episodic import EpisodicStore
from src.memory.insight_extractor import InsightExtractor, ExtractedInsight
from src.memory.episodic_retriever import EpisodicRetriever, RankedInsight
from src.memory.types import Insight, InsightType, Briefing, Goal, GoalType
from src.memory.objective import Objective

if TYPE_CHECKING:
    from src.knowledge.search.base import KnowledgeSearch
    from src.memory.config import CognitiveMemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class SuccessRecord:
    """Record of successful goal completion for learning."""
    experiment_id: str
    goal: str
    workflow_id: Optional[str]
    workflow_source: str
    steps_completed: List[str]
    heuristics_used: List[str]
    total_iterations: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CognitiveController:
    """
    Workflow-aware cognitive controller with LLM-based decisions.
    
    Key design principles:
    - Heuristics are loaded WITH the workflow from KG, not added later
    - ALL decisions (advance/retry/skip/pivot) are made by the LLM
    - The LLM receives full context (goal, workflow, experiment result, history)
    
    Usage:
        controller = CognitiveController(knowledge_search=kg)
        
        # Initialize - retrieves workflow WITH heuristics from KG
        controller.initialize_goal("Fine-tune LLaMA with LoRA")
        
        # Each iteration
        briefing = controller.prepare_briefing()
        # ... agent executes ...
        action, details = controller.process_result(success=True/False, error="...")
    """
    
    def __init__(
        self,
        knowledge_search: Optional["KnowledgeSearch"] = None,
        episodic_store_path: Optional[str] = None,
        config: Optional["CognitiveMemoryConfig"] = None,
        decision_model: Optional[str] = None,
    ):
        """
        Initialize cognitive controller.
        
        Args:
            knowledge_search: KG search backend (may be None)
            episodic_store_path: Path for episodic memory persistence
            config: Configuration (loads from YAML if None)
            decision_model: LLM model for decisions (defaults to gpt-4o-mini)
        """
        self.kg = knowledge_search
        
        # Load config
        if config is None:
            from src.memory.config import CognitiveMemoryConfig
            config = CognitiveMemoryConfig.load()
        self._config = config
        
        # Initialize core components
        self.retriever = KnowledgeRetriever(knowledge_search=knowledge_search)
        self.decision_maker = DecisionMaker(model=decision_model)
        self.episodic = EpisodicStore(persist_path=episodic_store_path)
        
        # Initialize LLM-governed components
        self.insight_extractor = InsightExtractor(model=decision_model)
        self.episodic_retriever = EpisodicRetriever(
            episodic_store=self.episodic,
            model=decision_model,
        )
        
        # State
        self._context: Optional[CognitiveContext] = None
        self._knowledge: Optional[KGKnowledge] = None  # NEW: Full KG knowledge
        self._goal: Optional[str] = None
        self._iteration: int = 0
        self._success_records: List[SuccessRecord] = []
        self._decision_history: List[ActionDecision] = []
        
        logger.info("CognitiveController initialized (LLM-governed insights & retrieval)")
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def initialize_goal(self, goal: "str | Goal | Objective") -> Optional[KGKnowledge]:
        """
        Initialize controller with a goal.
        
        Retrieves workflow from KG (or synthesizes one).
        Each step comes WITH heuristics already loaded - not added later.
        
        Args:
            goal: The high-level goal (string, Goal, or Objective object)
            
        Returns:
            The retrieved KGKnowledge (Tier 1/2). This is the single source of truth.
        """
        # Convert to Goal if needed - Objective is the preferred type
        if isinstance(goal, Objective):
            self._objective = goal
            goal_obj = Goal(
                description=goal.description,
                goal_type=GoalType(goal.objective_type.value) if hasattr(goal.objective_type, 'value') else GoalType.GENERIC,
                constraints=goal.constraints,
                success_criteria=goal.success_criteria,
                source=goal.source,
                metadata=goal.metadata,
            )
        elif isinstance(goal, str):
            goal_obj = Goal.from_string(goal)
            self._objective = None
        else:
            goal_obj = goal
            self._objective = None
        
        self._goal = goal_obj.description
        self._goal_obj = goal_obj  # Keep structured Goal for richer context
        self._iteration = 1
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │                    GOAL INITIALIZATION                            │
        # └─────────────────────────────────────────────────────────────────┘
        logger.info("┌" + "─"*65 + "┐")
        logger.info("│  🎯 GOAL INITIALIZATION" + " "*40 + "│")
        logger.info("└" + "─"*65 + "┘")
        
        goal_type_str = goal_obj.goal_type.value if hasattr(goal_obj.goal_type, 'value') else str(goal_obj.goal_type)
        logger.info(f"  Goal: {goal_obj.description}")
        logger.info(f"  Type: {goal_type_str} | Source: {goal_obj.source}")
        if goal_obj.constraints:
            logger.info(f"  Constraints: {goal_obj.constraints}")
        
        # Log Objective-specific context (data files, etc.)
        if self._objective and self._objective.data_files:
            logger.info(f"  Data files: {len(self._objective.data_files)}")
            for df in self._objective.data_files:
                logger.info(f"    • {df.path} ({'INPUT' if df.is_input else 'OUTPUT'})")
        
        # Retrieve knowledge using NEW KGKnowledge structure
        # This returns properly nested wiki structure (no truncation)
        self._knowledge = self.retriever.retrieve_knowledge(goal_obj.to_kg_query())
        
        # Log retrieval result
        logger.info("")
        logger.info(f"  📚 KG Retrieval: {self._knowledge.tier.value.upper()}")
        if self._knowledge.workflow:
            wf = self._knowledge.workflow
            logger.info(f"     Workflow: {wf.title}")
            logger.info(f"     Confidence: {self._knowledge.confidence:.0%}")
            logger.info(f"     Steps ({len(wf.steps)}):")
            for step in wf.steps:
                p = step.principle
                impl_count = len(p.implementations)
                heur_count = len(p.heuristics)
                impl_name = p.implementations[0].title if p.implementations else 'N/A'
                logger.info(f"       {step.number}. {p.title}")
                logger.info(f"          → impl: {impl_name} | heuristics: {heur_count}")
        elif self._knowledge.principles:
            logger.info(f"     No workflow, found {len(self._knowledge.principles)} relevant Principles")
        else:
            logger.info(f"     No knowledge found")
        logger.info("")
        
        # Build initial context with Goal object
        # Knowledge rendering comes from KGKnowledge (via _render_full_context).
        self._context = CognitiveContext(
            goal=goal_obj,
            iteration=1,
            kg_retrieval=KGRetrievalState(
                consulted_at_iteration=1,
                reason="initialization",
                query_used=self._knowledge.query_used,
                heuristics=[],
            ),
            meta=MetaState(
                steps_since_kg_consult=0,
                total_kg_consults=1,
            ),
        )
        # Cache the fully rendered context on the context object so the
        # DecisionMaker sees exactly what the agent sees.
        self._context.rendered_context = self._render_full_context()
        
        return self._knowledge
    
    def prepare_briefing(self) -> Briefing:
        """
        Prepare briefing for the coding agent.
        
        UNIFIED CONTEXT: The same CognitiveContext.render() output goes to:
        1. The coding agent (as the briefing)
        2. The decision maker (when deciding what action to take)
        
        This ensures transparency - both see the exact same information.
        
        The briefing includes:
        - Current goal with type/constraints
        - Current step (highlighted) with pre-loaded heuristics
        - Progress through workflow
        - Relevant past insights from episodic memory (LLM-retrieved)
        - Last experiment result with score/feedback
        """
        logger.info("┌" + "─"*65 + "┐")
        logger.info("│  📋 PREPARING BRIEFING" + " "*42 + "│")
        logger.info("└" + "─"*65 + "┘")
        if not self._context:
            return Briefing(
                goal="Unknown",
                plan="",
                insights=[],
                relevant_knowledge="",
                recent_history_summary="Controller not initialized. Call initialize_goal() first."
            )
        
        # =====================================================================
        # LLM-GOVERNED EPISODIC RETRIEVAL (before rendering context)
        # =====================================================================
        # Use LLM to retrieve relevant insights based on current context
        current_step_title = None  # We do not track step state; knowledge is advisory.
        
        last_error = None
        last_feedback = None
        if self._context.last_experiment:
            if not self._context.last_experiment.success:
                last_error = self._context.last_experiment.error_message
            last_feedback = self._context.last_experiment.feedback
        
        # Retrieve relevant episodic insights using LLM
        try:
            ranked_insights = self.episodic_retriever.retrieve_relevant_insights(
                goal=self._context.goal_str,
                current_step=current_step_title,
                last_error=last_error,
                last_feedback=last_feedback,
                max_insights=5,
            )
            
            # Update context with retrieved insights
            if ranked_insights:
                useful_insights = [r for r in ranked_insights if r.should_use]
                if useful_insights:
                    self._context.episodic_memory = EpisodicState(
                        relevant_insights=[
                            InsightSummary(
                                content=f"{r.content}\n  → Applies: {r.applicability}",
                                insight_type=r.insight_type,
                                confidence=r.relevance_score,
                            )
                            for r in useful_insights
                        ],
                        similar_errors=[],
                    )
                    logger.info(f"  💭 Episodic Memory: {len(useful_insights)} relevant insights retrieved")
                else:
                    logger.info(f"  💭 Episodic Memory: No applicable insights (checked {len(ranked_insights)})")
            else:
                logger.info(f"  💭 Episodic Memory: Empty (no past experiments)")
        except Exception as e:
            logger.warning(f"Episodic retrieval failed: {e}")
        
        # =====================================================================
        # UNIFIED CONTEXT RENDERING (using KGKnowledge)
        # =====================================================================
        # Knowledge comes from KGKnowledge.render()
        # Experiment state comes from CognitiveContext
        unified_context = self._render_full_context()
        # Ensure DecisionMaker always sees the same blob the agent sees.
        self._context.rendered_context = unified_context
        
        # Create Briefing wrapper (for backward compatibility)
        current_step_num = None
        current_step_title = None
        total_steps = None
        if self._knowledge and self._knowledge.workflow:
            total_steps = len(self._knowledge.workflow.steps)
            if total_steps > 0:
                current_step_num = 1  # Always show as step 1 (we don't track step-by-step)
                current_step_title = self._knowledge.workflow.steps[0].principle.title
        
        # Build insights list from episodic memory - no truncation
        insights = []
        if self._context.episodic_memory:
            if self._context.episodic_memory.relevant_insights:
                insights = [i.content for i in self._context.episodic_memory.relevant_insights]
            elif self._context.episodic_memory.similar_errors:
                insights = [i.content for i in self._context.episodic_memory.similar_errors]
        
        briefing = Briefing(
            goal=self._context.goal_str,
            plan=unified_context,  # Use FULL unified context as the plan
            insights=insights,
            relevant_knowledge="",  # Already in unified context
            recent_history_summary="",  # Already in unified context
            current_step_number=current_step_num,
            current_step_title=current_step_title,
            total_steps=total_steps,
        )
        
        # Log detailed briefing summary
        logger.info(f"  ✓ Briefing ready for iteration {self._iteration}")
        if self._knowledge and self._knowledge.workflow:
            wf = self._knowledge.workflow
            logger.info(f"    Workflow: {wf.title} ({len(wf.steps)} steps)")
            for step in wf.steps:
                p = step.principle
                impl_count = len(p.implementations)
                logger.info(f"      Step {step.number}: {p.title} ({impl_count} impls)")
        elif self._knowledge and self._knowledge.principles:
            logger.info(f"    Relevant Principles: {len(self._knowledge.principles)}")
        
        if insights:
            logger.info(f"    Episodic insights: {len(insights)}")
        logger.info("")
        
        return briefing
    
    def _render_full_context(self) -> str:
        """
        Render full context combining KGKnowledge + experiment state.
        
        This is the UNIFIED context that goes to both agent and decision maker.
        """
        lines = []
        
        # Goal section
        lines.append("## Goal")
        if isinstance(self._context.goal, Goal):
            goal = self._context.goal
            lines.append(f"**{goal.description}**")
            lines.append(f"- Type: {goal.goal_type.value}")
            lines.append(f"- Source: {goal.source}")
            if goal.constraints:
                lines.append(f"- Constraints: {goal.constraints}")
        else:
            lines.append(str(self._context.goal))
        lines.append("")
        
        # Status section
        lines.append("## Status")
        lines.append(f"- Iteration: {self._iteration}")
        lines.append(f"- Consecutive failures: {self._context.meta.consecutive_failures}")
        lines.append("")
        
        # Knowledge from KGKnowledge (TIER 1, 2, or 3)
        if self._knowledge:
            lines.append(self._knowledge.render())
        else:
            lines.append("## Implementation Guide")
            lines.append("*No knowledge available.*")
        lines.append("")
        
        # Last experiment result
        if self._context.last_experiment:
            exp = self._context.last_experiment
            lines.append("## Last Experiment")
            if exp.success:
                lines.append("**Result: ✓ SUCCESS**")
            else:
                lines.append("**Result: ✗ FAILED**")
            
            if exp.score is not None:
                lines.append(f"**Score: {exp.score:.2f}**")
            
            if exp.feedback:
                lines.append("")
                lines.append("**Evaluator Feedback:**")
                lines.append(f"> {exp.feedback}")
            
            if exp.error_message and not exp.success:
                lines.append("")
                lines.append("**Error to fix:**")
                lines.append(f"```\n{exp.error_message}\n```")
            lines.append("")
        
        # Episodic insights
        if self._context.episodic_memory:
            has_insights = (
                self._context.episodic_memory.similar_errors or
                self._context.episodic_memory.relevant_insights
            )
            if has_insights:
                lines.append("## Lessons from Past Experiments")
                for insight in self._context.episodic_memory.similar_errors:
                    lines.append(f"- ⚠️ {insight.content}")
                for insight in self._context.episodic_memory.relevant_insights:
                    lines.append(f"- 💡 {insight.content}")
                lines.append("")
        
        return "\n".join(lines)
    
    # NOTE: WorkflowState/StepState conversion removed. The sole workflow/knowledge
    # structure is `KGKnowledge`, and both agent + DecisionMaker consume the same
    # rendered text blob stored in `CognitiveContext.rendered_context`.
    
    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get workflow progress info for context manager."""
        if self._knowledge and self._knowledge.workflow:
            wf = self._knowledge.workflow
            return {
                "has_workflow": True,
                "title": wf.title,
                "source": wf.source,
                "total_steps": len(wf.steps),
                "tier": self._knowledge.tier.value,
            }
        elif self._knowledge and self._knowledge.principles:
            return {
                "has_workflow": False,
                "tier": self._knowledge.tier.value,
                "principles_count": len(self._knowledge.principles),
            }
        return {"has_workflow": False}
    
    def process_result(
        self,
        success: bool,
        error_message: Optional[str] = None,
        experiment_id: Optional[str] = None,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
        output: Optional[str] = None,
        solution: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process experiment result and decide next action.
        
        The LLM decides what to do based on full context:
        - ADVANCE: Move to next step
        - RETRY: Try again
        - SKIP: Skip current step
        - PIVOT: Try different workflow
        - COMPLETE: All done
        
        Args:
            success: Whether experiment succeeded
            error_message: Error message if failed
            experiment_id: Unique experiment ID
            score: Optional score from evaluation
            feedback: Evaluator feedback (e.g., LLM judge explanation)
            output: Execution output (stdout)
            solution: The code/solution that was tried
            
        Returns:
            Tuple of (action_name, details_dict)
        """
        if not self._context:
            return "error", {"message": "Controller not initialized"}
        
        exp_id = experiment_id or f"exp_{self._iteration}"
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │                    PROCESSING EXPERIMENT RESULT                   │
        # └─────────────────────────────────────────────────────────────────┘
        logger.info("┌" + "─"*65 + "┐")
        logger.info("│  ⚙️  PROCESSING EXPERIMENT RESULT" + " "*30 + "│")
        logger.info("└" + "─"*65 + "┘")
        
        status_icon = "✅" if success else "❌"
        score_str = f"{score:.2f}" if score is not None else "N/A"
        logger.info(f"  {status_icon} Iteration {self._iteration} | Score: {score_str}")
        if error_message:
            logger.info(f"  Error: {error_message}")
        if feedback:
            logger.info(f"  Feedback: {feedback}")
        
        # Update context with experiment result.
        #
        # IMPORTANT:
        # - This is the *only* place where we persist evaluator feedback into the
        #   cognitive state. The DecisionMaker and episodic retriever both rely
        #   on `context.last_experiment.feedback`.
        # - Historically this was accidentally dropped, which made decisions and
        #   episodic retrieval much lower quality (the LLM had less signal).
        self._context.last_experiment = ExperimentState(
            experiment_id=exp_id,
            # Use the real experiment/branch identifier if provided. This makes
            # logs and debugging align with the workspace branches created by
            # the search strategy.
            branch_name=exp_id,
            success=success,
            error_message=error_message,
            score=score,
            feedback=feedback,
        )
        # No WorkflowState step tracking. The agent implements full solution in one go.
        
        # Update meta state and store insights
        if success:
            self._context.meta.consecutive_failures = 0
            # Store success insight from evaluator feedback
            if feedback:
                self._store_success_insight(feedback, score, exp_id)
        else:
            self._context.meta.consecutive_failures += 1
            # Store error insight
            if error_message:
                self._store_error_insight(error_message, exp_id)
        
        # =====================================================================
        # LLM DECIDES WHAT ACTION TO TAKE
        # =====================================================================
        # Update rendered_context before decisions so DecisionMaker is grounded in
        # the same unified text we send to the agent.
        self._context.rendered_context = self._render_full_context()
        decision = self.decision_maker.decide_action(self._context)
        self._decision_history.append(decision)
        
        # Log LLM decision
        logger.info("")
        logger.info(f"  🧠 LLM Decision: {decision.action.value}")
        logger.info(f"     Confidence: {decision.confidence:.0%}")
        logger.info(f"     Reasoning: {decision.reasoning}")
        
        # Execute the decided action
        result = self._execute_action(decision)
        action_icon = {"complete": "🏁", "retry": "🔄", "pivot": "↩️"}.get(result[0], "•")
        logger.info(f"  {action_icon} Action executed: {result[0].upper()}")
        logger.info("")
        self._iteration += 1
        self._context.iteration = self._iteration  # Keep context in sync
        return result
    
    def _execute_action(self, decision: ActionDecision) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the action decided by the LLM.
        
        SIMPLIFIED: Only iteration-level actions (no step-level ADVANCE/SKIP)
        since agent implements full solution in one go.
        """
        action = decision.action
        
        if action == WorkflowAction.COMPLETE:
            self._record_success()
            return "complete", {"reasoning": decision.reasoning}
        
        elif action == WorkflowAction.PIVOT:
            return self._do_pivot()
        
        elif action == WorkflowAction.RETRY:
            return "retry", {
                "reasoning": decision.reasoning,
                "iteration": self._iteration,
            }
        
        else:
            # Default to retry for unknown actions
            return "retry", {"action": str(action), "fallback": True}
    
    def _do_pivot(self) -> Tuple[str, Dict[str, Any]]:
        """Pivot to a different workflow."""
        if not self._goal:
            return "pivot", {"success": False}
        
        old_workflow = self._knowledge.workflow.title if (self._knowledge and self._knowledge.workflow) else None
        
        # Re-retrieve, excluding current workflow (single retrieval implementation).
        # This returns nested KGKnowledge (workflow + principles/heuristics/implementations).
        knowledge = self.retriever.retrieve_knowledge(
            goal=self._goal,
            exclude_workflow=old_workflow,
        )
        
        if knowledge.workflow and self._context:
            # The authoritative knowledge passed to the agent is `self._knowledge`.
            self._knowledge = knowledge
            logger.info(f"Pivoted from '{old_workflow}' to '{knowledge.workflow.title}'")
            return "pivot", {
                "old_workflow": old_workflow,
                "new_workflow": knowledge.workflow.title
            }
        
        return "pivot", {"success": False, "reason": "No alternative found"}
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _store_error_insight(self, error_message: str, experiment_id: str, code_snippet: Optional[str] = None):
        """
        Store GENERALIZED insight from error using LLM extraction.
        
        Instead of storing raw error text, the LLM extracts:
        - A generalized lesson (reusable principle)
        - Trigger conditions (when this applies)
        - Suggested fix (actionable steps)
        
        Also triggers TIER 3 retrieval to find heuristics related to the error.
        """
        # =====================================================================
        # 1. LLM-BASED INSIGHT EXTRACTION (generalize the error)
        # =====================================================================
        current_step = None
        
        try:
            extracted = self.insight_extractor.extract_from_error(
                error_message=error_message,
                goal=self._goal or "Unknown",
                current_step=current_step,
                code_snippet=code_snippet,
            )
            
            # Store the GENERALIZED insight (not raw error)
            insight_content = f"{extracted.lesson}\n→ When: {extracted.trigger_conditions}\n→ Fix: {extracted.suggested_fix}"
            insight = Insight(
                content=insight_content,
                insight_type=InsightType.CRITICAL_ERROR,
                confidence=extracted.confidence,
                source_experiment_id=experiment_id,
                tags=extracted.tags,
            )
            self.episodic.add_insight(insight)
            logger.info(f"Stored generalized error insight: {extracted.lesson}")
            
        except Exception as e:
            logger.warning(f"LLM insight extraction failed, storing raw error: {e}")
            # Fallback: store raw error (only if LLM extraction failed)
            insight = Insight(
                content=f"Error: {error_message}",  # No truncation - store full error
                insight_type=InsightType.CRITICAL_ERROR,
                confidence=0.5,
                source_experiment_id=experiment_id,
            )
            self.episodic.add_insight(insight)
        
        # =====================================================================
        # 2. TIER 3: Consult KG for error-specific heuristics
        # =====================================================================
        if self._goal and self._knowledge and self._context.meta.consecutive_failures >= 1:
            # Consult KG after 1+ failures to get error-specific help
            self._knowledge = self.retriever.retrieve_knowledge(
                goal=self._goal,
                existing_knowledge=self._knowledge,
                last_error=error_message,
            )
            
            # Update KG retrieval state for tracking
            self._context.kg_retrieval = KGRetrievalState(
                consulted_at_iteration=self._iteration,
                # Keep the reason short and stable. The full error is already
                # available in `last_experiment.error_message`.
                reason="error_recovery",
                # Record the *actual* queries used by the retriever (Tier 1/2/3).
                # This is critical for log auditing and PR review.
                query_used=self._knowledge.query_used,
                heuristics=[h.title for h in self._knowledge.error_heuristics],
                code_patterns=[],
            )
            self._context.meta.total_kg_consults += 1
            logger.info(f"TIER 3: Added {len(self._knowledge.error_heuristics)} error heuristics, {len(self._knowledge.alternative_implementations)} alternatives")
    
    def _store_success_insight(self, feedback: str, score: Optional[float], experiment_id: str, solution_summary: Optional[str] = None):
        """
        Store GENERALIZED best practice from success using LLM extraction.
        
        Instead of storing raw feedback, the LLM extracts:
        - A reusable pattern/principle
        - When to apply this pattern
        - How to implement it
        
        Args:
            feedback: Evaluator feedback text
            score: Score from evaluation (0.0-1.0)
            experiment_id: Which experiment this came from
            solution_summary: Summary of the solution that worked
        """
        if not feedback or len(feedback.strip()) < 10:
            return  # Skip empty or trivial feedback
        
        # =====================================================================
        # LLM-BASED INSIGHT EXTRACTION (generalize the success)
        # =====================================================================
        current_step = None
        
        try:
            extracted = self.insight_extractor.extract_from_success(
                feedback=feedback,
                goal=self._goal or "Unknown",
                score=score or 0.5,
                current_step=current_step,
                solution_summary=solution_summary,
            )
            
            # Store the GENERALIZED best practice
            insight_content = f"✓ {extracted.lesson}\n→ When: {extracted.trigger_conditions}\n→ How: {extracted.suggested_fix}"
            insight = Insight(
                content=insight_content,
                insight_type=InsightType.BEST_PRACTICE,
                confidence=extracted.confidence,
                source_experiment_id=experiment_id,
                tags=extracted.tags + ["success"],
            )
            self.episodic.add_insight(insight)
            logger.info(f"Stored generalized success insight: {extracted.lesson}")
            
        except Exception as e:
            logger.warning(f"LLM insight extraction failed, storing raw feedback: {e}")
            # Fallback: store raw feedback (only if LLM extraction failed)
            confidence = min(score, 1.0) if score is not None else 0.6
            score_str = f"[Score: {score:.2f}] " if score is not None else ""
            content = f"{score_str}{feedback}"  # No truncation
            
            insight = Insight(
                content=content,
                insight_type=InsightType.BEST_PRACTICE,
                confidence=confidence,
                source_experiment_id=experiment_id,
                tags=["success", "evaluator_feedback"],
            )
            self.episodic.add_insight(insight)
            logger.debug(f"Stored success insight: {content}")


    def _record_success(self):
        """Record successful completion for learning."""
        if not self._context:
            return
        
        steps_completed: List[str] = []
        heuristics_used: List[str] = []
        
        record = SuccessRecord(
            experiment_id=f"goal_{self._iteration}",
            goal=self._goal or "",
            workflow_id=self._knowledge.workflow.id if (self._knowledge and self._knowledge.workflow) else None,
            workflow_source=self._knowledge.workflow.source if (self._knowledge and self._knowledge.workflow) else "none",
            steps_completed=steps_completed,
            heuristics_used=list(set(heuristics_used)),
            total_iterations=self._iteration,
        )
        self._success_records.append(record)
        logger.info(f"Recorded success: {len(steps_completed)} steps in {self._iteration} iterations")
    
    # =========================================================================
    # State Access
    # =========================================================================
    
    def get_context(self) -> Optional[CognitiveContext]:
        """Get current context for inspection."""
        return self._context
    
    def get_knowledge(self) -> Optional[KGKnowledge]:
        """Get current KG knowledge for inspection."""
        return self._knowledge
    
    # NOTE: get_workflow_progress() is defined earlier in the class
    
    def get_decision_history(self) -> List[ActionDecision]:
        """Get history of all LLM decisions made."""
        return self._decision_history
    
    def is_complete(self) -> bool:
        """Check if the run is complete (DecisionMaker signaled COMPLETE)."""
        # The orchestrator primarily uses ContextManager.should_stop().
        # This method is retained for compatibility but no longer uses WorkflowState.
        return bool(self._decision_history and self._decision_history[-1].action == WorkflowAction.COMPLETE)
    
    def close(self):
        """Clean up resources."""
        if self.episodic:
            self.episodic.close()
