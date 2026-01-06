# =============================================================================
# Episodic Learner - Learn from Experiment History
# =============================================================================
#
# This learner takes episodic memories (insights from experiments) and:
# 1. Extracts generalizable patterns
# 2. Proposes new heuristics for the KG
# 3. Harvests successful workflows back to KG
#
# Follows the same pattern as KnowledgePipeline:
#   Source (EpisodicStore) → Learner → WikiPages → Merger → Updated KG
#
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.episodic import EpisodicStore
    from src.memory.types import Insight
    from src.knowledge.search.base import WikiPage


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LearnedPattern:
    """A pattern learned from episodic memory."""
    pattern_type: str  # "error_solution", "best_practice", "workflow_improvement"
    description: str
    confidence: float
    supporting_experiments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LearnedHeuristic:
    """A heuristic extracted from patterns, ready for KG."""
    title: str
    content: str
    applies_to: List[str]  # Related page types/titles
    confidence: float
    source_patterns: List[str] = field(default_factory=list)


@dataclass
class HarvestedWorkflow:
    """A successful workflow to add to KG."""
    title: str
    steps: List[str]
    heuristics: List[str]
    success_rate: float
    experiment_count: int


@dataclass
class EpisodicLearnerResult:
    """Result from episodic learning."""
    patterns: List[LearnedPattern] = field(default_factory=list)
    heuristics: List[LearnedHeuristic] = field(default_factory=list)
    workflows: List[HarvestedWorkflow] = field(default_factory=list)
    insights_processed: int = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Base Learner Interface
# =============================================================================

class EpisodicLearner(ABC):
    """
    Abstract base class for episodic learners.
    
    An episodic learner receives experiment insights and extracts
    generalizable knowledge to improve the KG.
    
    Subclasses must implement:
    - learn(): Process insights and extract patterns/heuristics
    - to_wiki_pages(): Convert learned knowledge to WikiPages for KG
    """
    
    @abstractmethod
    def learn(
        self,
        insights: List["Insight"],
        success_records: Optional[List[Dict[str, Any]]] = None,
    ) -> EpisodicLearnerResult:
        """
        Learn from episodic memories.
        
        Args:
            insights: List of Insight objects from EpisodicStore
            success_records: Optional list of successful experiment records
            
        Returns:
            EpisodicLearnerResult with patterns, heuristics, workflows
        """
        pass
    
    @abstractmethod
    def to_wiki_pages(self, result: EpisodicLearnerResult) -> List["WikiPage"]:
        """
        Convert learning result to WikiPages for KG ingestion.
        
        Args:
            result: EpisodicLearnerResult from learn()
            
        Returns:
            List of WikiPage objects ready for KnowledgeMerger
        """
        pass


# =============================================================================
# LLM-Based Episodic Learner (Interface Only)
# =============================================================================

class LLMEpisodicLearner(EpisodicLearner):
    """
    LLM-based episodic learner.
    
    Uses LLM to:
    1. Analyze insights for patterns
    2. Extract generalizable heuristics
    3. Identify successful workflows to harvest
    
    This is the interface - implementation TBD.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        min_insights_for_pattern: int = 3,
        min_success_rate_for_harvest: float = 0.7,
    ):
        """
        Initialize LLM-based learner.
        
        Args:
            model: LLM model to use for analysis
            min_insights_for_pattern: Minimum insights needed to identify a pattern
            min_success_rate_for_harvest: Minimum success rate to harvest a workflow
        """
        self.model = model
        self.min_insights = min_insights_for_pattern
        self.min_success_rate = min_success_rate_for_harvest
    
    def learn(
        self,
        insights: List["Insight"],
        success_records: Optional[List[Dict[str, Any]]] = None,
    ) -> EpisodicLearnerResult:
        """
        Learn from insights using LLM analysis.
        
        TODO: Implement with:
        1. Group insights by type/topic
        2. LLM call to identify patterns
        3. LLM call to extract heuristics from patterns
        4. Analyze success_records for harvestable workflows
        """
        # Placeholder implementation
        return EpisodicLearnerResult(
            insights_processed=len(insights),
            errors=["Not yet implemented"],
        )
    
    def to_wiki_pages(self, result: EpisodicLearnerResult) -> List["WikiPage"]:
        """
        Convert learning result to WikiPages.
        
        TODO: Implement with:
        1. LearnedHeuristic → WikiPage(type="Heuristic")
        2. HarvestedWorkflow → WikiPage(type="Workflow")
        """
        # Placeholder implementation
        return []


# =============================================================================
# Factory
# =============================================================================

_LEARNER_REGISTRY: Dict[str, type] = {}


def register_episodic_learner(name: str):
    """Decorator to register an episodic learner."""
    def decorator(cls):
        _LEARNER_REGISTRY[name] = cls
        return cls
    return decorator


class EpisodicLearnerFactory:
    """Factory for creating episodic learners."""
    
    @staticmethod
    def create(learner_type: str = "llm", **kwargs) -> EpisodicLearner:
        """
        Create an episodic learner.
        
        Args:
            learner_type: Type of learner ("llm", etc.)
            **kwargs: Learner-specific parameters
            
        Returns:
            EpisodicLearner instance
        """
        if learner_type == "llm":
            return LLMEpisodicLearner(**kwargs)
        
        if learner_type in _LEARNER_REGISTRY:
            return _LEARNER_REGISTRY[learner_type](**kwargs)
        
        raise ValueError(f"Unknown episodic learner type: {learner_type}")


# Register built-in learners
register_episodic_learner("llm")(LLMEpisodicLearner)
