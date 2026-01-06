# =============================================================================
# Memory Data Types for Cognitive Memory Architecture
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from enum import Enum


# =============================================================================
# Goal - Structured representation of what we're trying to achieve
# =============================================================================

class GoalType(Enum):
    """Categories of goals for better KG retrieval and decision-making."""
    ML_TRAINING = "ml_training"          # Fine-tuning, training models
    ML_INFERENCE = "ml_inference"        # Running inference, predictions
    DATA_PROCESSING = "data_processing"  # ETL, data cleaning, transformation
    CODE_GENERATION = "code_generation"  # Writing new code
    BUG_FIX = "bug_fix"                  # Fixing existing code
    OPTIMIZATION = "optimization"        # Performance tuning
    RESEARCH = "research"                # Exploring, experimenting
    GENERIC = "generic"                  # Default/unknown


@dataclass
class Goal:
    """
    Structured goal representation.
    
    Provides richer context than a plain string for:
    - Better KG retrieval (search by goal type)
    - Better decision-making (constraints inform choices)
    - Better logging (all context captured)
    
    Attributes:
        description: Natural language description of what to achieve
        goal_type: Category of goal for KG retrieval
        constraints: Resource/time constraints
        expected_outputs: What files/artifacts should be produced
        success_criteria: How to know if we succeeded
        source: Where this goal came from (user, automated, benchmark)
        metadata: Additional context (benchmark name, problem ID, etc.)
    """
    description: str
    goal_type: GoalType = GoalType.GENERIC
    constraints: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    success_criteria: Optional[str] = None
    source: str = "user"  # "user", "benchmark", "automated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self) -> str:
        """Return description for backward compatibility with string goals."""
        return self.description
    
    @classmethod
    def from_string(cls, description: str) -> "Goal":
        """Create a Goal from a plain string (backward compatibility)."""
        # Try to infer goal type from keywords
        desc_lower = description.lower()
        
        goal_type = GoalType.GENERIC
        if any(kw in desc_lower for kw in ["train", "fine-tune", "finetune", "lora"]):
            goal_type = GoalType.ML_TRAINING
        elif any(kw in desc_lower for kw in ["predict", "inference", "classify"]):
            goal_type = GoalType.ML_INFERENCE
        elif any(kw in desc_lower for kw in ["fix", "bug", "error", "issue"]):
            goal_type = GoalType.BUG_FIX
        elif any(kw in desc_lower for kw in ["optimize", "performance", "speed"]):
            goal_type = GoalType.OPTIMIZATION
        elif any(kw in desc_lower for kw in ["data", "process", "etl", "clean"]):
            goal_type = GoalType.DATA_PROCESSING
        
        return cls(description=description, goal_type=goal_type)
    
    def to_kg_query(self) -> str:
        """Format goal for KG search query."""
        type_context = f"[{self.goal_type.value}] " if self.goal_type != GoalType.GENERIC else ""
        return f"{type_context}{self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "goal_type": self.goal_type.value,
            "constraints": self.constraints,
            "expected_outputs": self.expected_outputs,
            "success_criteria": self.success_criteria,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class InsightType(Enum):
    """Categories for learned insights."""
    CRITICAL_ERROR = "critical_error"      # Mistakes to avoid
    BEST_PRACTICE = "best_practice"        # Patterns that work
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Facts about the problem domain


@runtime_checkable
class ExperimentResultProtocol(Protocol):
    """
    Protocol for experiment results that can be processed by CognitiveController.
    
    Any object with these attributes can be passed to process_result().
    """
    run_had_error: bool
    error_details: Optional[str]


@dataclass
class Insight:
    """
    A learned insight from past experiments.
    
    Attributes:
        content: The insight text (rule or recommendation)
        insight_type: Category of insight
        confidence: Confidence score (0.0-1.0)
        source_experiment_id: Which experiment this came from
        tags: Optional tags for filtering
        created_at: When the insight was created
    """
    content: str
    insight_type: InsightType
    confidence: float
    source_experiment_id: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def age_hours(self) -> float:
        """Return age of insight in hours."""
        now = datetime.now(timezone.utc)
        # Handle naive datetime for backwards compatibility
        if self.created_at.tzinfo is None:
            created = self.created_at.replace(tzinfo=timezone.utc)
        else:
            created = self.created_at
        return (now - created).total_seconds() / 3600


@dataclass
class Briefing:
    """
    Synthesized context packet for the coding agent.
    
    This is what gets passed to the agent before each task attempt.
    Contains goal, plan, relevant insights, KG knowledge, and history.
    
    The current_step field explicitly tells the agent which step they're on.
    """
    goal: str
    plan: str
    insights: List[str]
    relevant_knowledge: str
    recent_history_summary: str
    # New fields for explicit step tracking
    current_step_number: Optional[int] = None
    current_step_title: Optional[str] = None
    total_steps: Optional[int] = None
    
    def to_string(self) -> str:
        """
        Format briefing as a string for LLM context.
        
        UNIFIED CONTEXT: If self.plan contains the full rendered context
        (detected by presence of "## Goal" and knowledge sections), use it directly.
        Otherwise, we format the individual fields.
        """
        # Check if plan contains unified context (from CognitiveController._render_full_context())
        # Detect by "## Goal" + any knowledge section (workflow, relevant knowledge, or implementation guide)
        has_unified = (
            self.plan and 
            "## Goal" in self.plan and 
            ("## Implementation Guide" in self.plan or "## Relevant Knowledge" in self.plan)
        )
        if has_unified:
            # UNIFIED CONTEXT MODE: plan is the full rendered context
            # Just add episodic insights if not already included
            unified = self.plan
            
            # Add episodic insights section if we have any and they're not in unified
            if self.insights and "## Episodic Insights" not in unified:
                insights_section = "\n## Episodic Insights (Past Learnings)\n"
                for insight in self.insights:
                    insights_section += f"- {insight}\n"
                unified += "\n" + insights_section
            
            return unified
        
        # LEGACY MODE: format individual fields
        insights_text = "\n".join(f"- {i}" for i in self.insights) if self.insights else "None"
        
        # Build current step header if we have step info
        step_header = ""
        if self.current_step_number and self.current_step_title:
            step_header = f"""
# 🎯 CURRENT STEP: {self.current_step_number}/{self.total_steps or '?'} - {self.current_step_title}
Focus on completing this step before moving to the next.
"""
        
        return f"""
# CURRENT MISSION
{self.goal}
{step_header}
# WORKFLOW PLAN
{self.plan}

# ⚠️ CRITICAL INSIGHTS FROM PAST ATTEMPTS
{insights_text}

# 📚 KNOWLEDGE FOR CURRENT STEP
{self.relevant_knowledge}

# RECENT HISTORY
{self.recent_history_summary}
"""


# =============================================================================
# Success Record - For Learning from Successes
# =============================================================================

@dataclass
class SuccessRecord:
    """
    Record of a successful goal completion.
    
    Used to learn from successes and potentially harvest patterns
    back to the Knowledge Graph.
    
    Attributes:
        experiment_id: Unique identifier
        goal: The goal that was achieved
        workflow_id: ID of workflow used (if any)
        workflow_source: Where workflow came from (kg_exact, synthesized, etc.)
        steps_completed: List of step titles completed
        heuristics_used: Heuristics that were applied
        total_iterations: How many iterations it took
        final_score: Final performance score (if available)
        created_at: When success was recorded
    """
    experiment_id: str
    goal: str
    workflow_id: Optional[str] = None
    workflow_source: str = "none"
    steps_completed: List[str] = field(default_factory=list)
    heuristics_used: List[str] = field(default_factory=list)
    total_iterations: int = 0
    final_score: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "goal": self.goal,
            "workflow_id": self.workflow_id,
            "workflow_source": self.workflow_source,
            "steps_completed": self.steps_completed,
            "heuristics_used": self.heuristics_used,
            "total_iterations": self.total_iterations,
            "final_score": self.final_score,
            "created_at": self.created_at.isoformat(),
        }
