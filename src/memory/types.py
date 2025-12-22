# =============================================================================
# Memory Data Types for Cognitive Memory Architecture
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from enum import Enum


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
class WorkingMemory:
    """
    Active state of the current task.
    
    Attributes:
        current_goal: What we're trying to achieve
        active_plan: Current plan steps
        facts: Known facts and context
    """
    current_goal: str
    active_plan: List[str]
    facts: Dict[str, Any] = field(default_factory=dict)
    
    def update_plan(self, new_plan: List[str]):
        """Update the active plan."""
        self.active_plan = new_plan


@dataclass
class Briefing:
    """
    Synthesized context packet for the coding agent.
    
    This is what gets passed to the agent before each task attempt.
    Contains goal, plan, relevant insights, KG knowledge, and history.
    """
    goal: str
    plan: str
    insights: List[str]
    relevant_knowledge: str
    recent_history_summary: str
    
    def to_string(self) -> str:
        """Format briefing as a string for LLM context."""
        insights_text = "\n".join(f"- {i}" for i in self.insights) if self.insights else "None"
        return f"""
# CURRENT MISSION
{self.goal}

# ACTIVE PLAN
{self.plan}

# ⚠️ CRITICAL INSIGHTS
{insights_text}

# RELEVANT KNOWLEDGE
{self.relevant_knowledge}

# RECENT HISTORY
{self.recent_history_summary}
"""
