# =============================================================================
# Cognitive Context - Unified context for decision-making and agent briefing
# =============================================================================
#
# DESIGN PRINCIPLE: The SAME context goes to both the decision-maker and the
# coding agent. This ensures transparency - what the decision-maker sees is
# exactly what the agent sees.
#
# =============================================================================

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union

from src.memory.types import Goal


@dataclass
class ExperimentState:
    """State of the last experiment."""
    experiment_id: str
    branch_name: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None  # ← ADDED: Evaluator feedback
    code_summary: Optional[str] = None


@dataclass
class KGRetrievalState:
    """State of the last KG retrieval."""
    consulted_at_iteration: int
    reason: str
    query_used: str
    heuristics: List[str] = field(default_factory=list)
    code_patterns: List[str] = field(default_factory=list)
    suggested_workflow: Optional[str] = None


@dataclass
class InsightSummary:
    """Summary of an episodic insight."""
    content: str
    insight_type: str
    confidence: float
    source_experiment: Optional[str] = None


@dataclass
class EpisodicState:
    """State of episodic memory retrieval."""
    relevant_insights: List[InsightSummary] = field(default_factory=list)
    similar_errors: List[InsightSummary] = field(default_factory=list)


@dataclass
class MetaState:
    """Meta-information about the cognitive system."""
    steps_since_kg_consult: int = 0
    total_kg_consults: int = 0
    consecutive_failures: int = 0
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# No hardcoded limits - pages are already size-limited


@dataclass
class CognitiveContext:
    """
    Unified context object - SINGLE SOURCE OF TRUTH.
    
    The SAME render() output goes to:
    1. DecisionMaker - for deciding what action to take
    2. Coding Agent - as part of the briefing
    
    This ensures transparency and consistency.
    """
    goal: Union[str, Goal]  # Accepts string (backward compat) or Goal object
    iteration: int
    last_experiment: Optional[ExperimentState] = None
    kg_retrieval: Optional[KGRetrievalState] = None
    episodic_memory: Optional[EpisodicState] = None
    meta: MetaState = field(default_factory=MetaState)
    # The cognitive system passes a single rendered context blob to BOTH:
    # - the coding agent (via ContextManager.additional_info)
    # - the DecisionMaker (for RETRY/PIVOT/COMPLETE)
    #
    # Keeping this string avoids maintaining a separate "workflow state machine"
    # which is not used by the search strategy (it only consumes text).
    rendered_context: Optional[str] = None
    
    @property
    def goal_str(self) -> str:
        """Get goal as string (for backward compatibility)."""
        if isinstance(self.goal, Goal):
            return self.goal.description
        return self.goal
    
    @property
    def goal_obj(self) -> Goal:
        """Get goal as Goal object (creates one from string if needed)."""
        if isinstance(self.goal, Goal):
            return self.goal
        return Goal.from_string(self.goal)
    
    def render(self) -> str:
        """
        Render context to text.
        
        THIS IS THE ONLY RENDER METHOD.
        The same output goes to both decision-maker and coding agent.
        
        DESIGN: "Workflow as Guidance"
        - Shows ALL steps with ALL heuristics at once
        - Agent implements full solution in one go
        - Workflow is advisory, not step-by-step execution
        """
        # If the controller already produced a unified context string,
        # use it directly. This is the intended execution path.
        if self.rendered_context:
            return self.rendered_context
        
        lines = []
        
        # =================================================================
        # GOAL
        # =================================================================
        lines.append("## Goal")
        if isinstance(self.goal, Goal):
            lines.append(f"**{self.goal.description}**")
            lines.append(f"- Type: {self.goal.goal_type.value}")
            lines.append(f"- Source: {self.goal.source}")
            if self.goal.constraints:
                lines.append(f"- Constraints: {self.goal.constraints}")
            if self.goal.success_criteria:
                lines.append(f"- Success criteria: {self.goal.success_criteria}")
        else:
            lines.append(self.goal)
        lines.append("")
        
        # =================================================================
        # STATUS
        # =================================================================
        lines.append(f"## Status")
        lines.append(f"- Iteration: {self.iteration}")
        lines.append(f"- Consecutive failures: {self.meta.consecutive_failures}")
        lines.append("")
        
        # NOTE: Knowledge/workflow guidance is owned by KGKnowledge.render()
        # and is injected by the controller into rendered_context. If we reach
        # this fallback path, we only provide minimal scaffolding.
        lines.append("## Implementation Guide")
        lines.append("See rendered knowledge in controller output.")
        lines.append("")
        
        # =================================================================
        # LAST EXPERIMENT RESULT
        # =================================================================
        if self.last_experiment:
            exp = self.last_experiment
            lines.append("## Last Experiment")
            
            if exp.success:
                lines.append(f"**Result: ✓ SUCCESS**")
            else:
                lines.append(f"**Result: ✗ FAILED**")
            
            if exp.score is not None:
                score_desc = "excellent" if exp.score >= 0.8 else "good" if exp.score >= 0.6 else "needs improvement" if exp.score >= 0.4 else "low"
                lines.append(f"**Score: {exp.score:.2f}** ({score_desc})")
            
            if exp.feedback:
                lines.append("")
                lines.append("**Evaluator Feedback:**")
                lines.append(f"> {exp.feedback}")
            
            if exp.error_message and not exp.success:
                lines.append("")
                lines.append("**Error to fix:**")
                lines.append(f"```\n{exp.error_message}\n```")
            lines.append("")
        
        # =================================================================
        # PAST LEARNINGS (from episodic memory)
        # =================================================================
        if self.episodic_memory:
            has_insights = (
                self.episodic_memory.similar_errors or 
                self.episodic_memory.relevant_insights
            )
            if has_insights:
                lines.append("## Lessons from Past Experiments")
                
            if self.episodic_memory.similar_errors:
                for insight in self.episodic_memory.similar_errors:
                    lines.append(f"- ⚠️ {insight.content}")
            
            if self.episodic_memory.relevant_insights:
                for insight in self.episodic_memory.relevant_insights:
                    lines.append(f"- 💡 {insight.content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        return convert(self)
