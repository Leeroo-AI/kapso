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
class StepState:
    """State of a single workflow step."""
    number: int
    title: str
    status: str = "pending"  # pending, in_progress, completed, skipped
    description: str = ""
    heuristics: List[str] = field(default_factory=list)
    code_patterns: List[str] = field(default_factory=list)
    attempts: int = 0
    last_error: Optional[str] = None
    # Implementation info (from graph traversal)
    implementation: Optional[Dict[str, str]] = None  # {title, overview, code_snippets}
    principle_id: Optional[str] = None  # For tracking graph links


@dataclass
class WorkflowState:
    """Current workflow state."""
    id: str
    title: str
    source: str  # "kg_exact", "kg_synthesized", "agent_created"
    confidence: float
    steps: List[StepState]
    current_step_index: int = 0
    revision_count: int = 0
    
    @property
    def current_step(self) -> Optional[StepState]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def progress(self) -> str:
        return f"{self.current_step_index + 1}/{len(self.steps)}"
    
    @property
    def completed_steps(self) -> List[StepState]:
        return [s for s in self.steps if s.status == "completed"]
    
    @property
    def is_complete(self) -> bool:
        return all(s.status in ["completed", "skipped"] for s in self.steps) if self.steps else True


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


MAX_ERROR_LENGTH = 1500


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
    workflow: Optional[WorkflowState] = None
    last_experiment: Optional[ExperimentState] = None
    kg_retrieval: Optional[KGRetrievalState] = None
    episodic_memory: Optional[EpisodicState] = None
    meta: MetaState = field(default_factory=MetaState)
    
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
        
        # =================================================================
        # WORKFLOW GUIDANCE - Show ALL steps with ALL heuristics
        # =================================================================
        if self.workflow:
            wf = self.workflow
            lines.append("## Implementation Guide")
            lines.append(f"**{wf.title}** (from {wf.source}, confidence: {wf.confidence:.0%})")
            lines.append("")
            lines.append("Follow these steps to implement the solution:")
            lines.append("")
            
            # Show ALL steps with their heuristics, implementations, and code patterns
            for step in wf.steps:
                lines.append(f"### Step {step.number}: {step.title}")
                
                if step.description:
                    lines.append(f"{step.description}")
                    lines.append("")
                
                # Show implementation for THIS step (from graph traversal)
                if step.implementation:
                    impl = step.implementation
                    lines.append(f"**Implementation:** `{impl.get('title', 'Unknown')}`")
                    if impl.get('overview'):
                        lines.append(f"> {impl['overview'][:200]}")
                    # Show implementation code snippets
                    if impl.get('code_snippets'):
                        for snippet in impl['code_snippets'][:2]:
                            lines.append(f"```python\n{snippet.strip()}\n```")
                    lines.append("")
                
                # Show heuristics for THIS step
                if step.heuristics:
                    lines.append("**Tips:**")
                    for h in step.heuristics:
                        lines.append(f"- {h}")
                    lines.append("")
                
                # Show additional code patterns for THIS step (from TIER 3 error recovery)
                if step.code_patterns:
                    lines.append("**Additional patterns:**")
                    for p in step.code_patterns[:2]:  # Limit to 2 patterns per step
                        lines.append(f"```python\n{p}\n```")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        else:
            lines.append("## Implementation Guide")
            lines.append("No specific workflow guidance available. Implement based on the goal.")
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
                lines.append(f"> {exp.feedback[:500]}")
            
            if exp.error_message and not exp.success:
                lines.append("")
                lines.append("**Error to fix:**")
                lines.append(f"```\n{exp.error_message[:MAX_ERROR_LENGTH]}\n```")
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
                for insight in self.episodic_memory.similar_errors[:3]:
                    lines.append(f"- ⚠️ {insight.content}")
            
            if self.episodic_memory.relevant_insights:
                for insight in self.episodic_memory.relevant_insights[:3]:
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
