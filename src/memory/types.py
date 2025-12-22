# Memory data types for Cognitive Memory Architecture
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class InsightType(Enum):
    """Categories for learned insights."""
    CRITICAL_ERROR = "critical_error"      # Mistakes to avoid
    BEST_PRACTICE = "best_practice"        # Patterns that work
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Facts about the problem domain


@dataclass
class Insight:
    """A learned insight from past experiments."""
    content: str                           # The insight text
    insight_type: InsightType              # Category
    confidence: float                      # 0.0-1.0 confidence score
    source_experiment_id: str              # Which experiment this came from
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkingMemory:
    """Active state of the current task."""
    current_goal: str                      # What we're trying to achieve
    active_plan: List[str]                 # Current plan steps
    facts: Dict[str, Any] = field(default_factory=dict)  # Known facts
    
    def update_plan(self, new_plan: List[str]):
        self.active_plan = new_plan


@dataclass
class Briefing:
    """Synthesized context packet for the coding agent."""
    goal: str
    plan: str
    insights: List[str]
    relevant_knowledge: str
    recent_history_summary: str
    
    def to_string(self) -> str:
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

