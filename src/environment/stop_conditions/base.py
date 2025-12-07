# Stop Condition Base Classes
#
# Abstract base class and result types for stop conditions.
# Stop conditions determine when to end the experiment loop early.
#
# To create a custom stop condition:
# 1. Subclass StopCondition
# 2. Implement check() method
# 3. Register with @register_stop_condition("name") decorator in factory.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StopDecision:
    """
    Result of a stop condition check.
    
    Attributes:
        should_stop: Whether to stop the experiment loop
        reason: Human-readable reason for the decision
        details: Additional details about the decision
    """
    should_stop: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow direct use as boolean."""
        return self.should_stop
    
    def __repr__(self) -> str:
        status = "STOP" if self.should_stop else "CONTINUE"
        return f"StopDecision({status}: {self.reason})"


class StopCondition(ABC):
    """
    Abstract base class for stop conditions.
    
    Stop conditions determine when to end the experiment loop early,
    before the budget is exhausted.
    
    To create a custom stop condition:
    1. Subclass StopCondition
    2. Implement check() method
    3. Register with @register_stop_condition("name") decorator
    
    The check() method receives:
    - best_score: Best score achieved so far
    - current_score: Score from most recent run
    - iteration: Current iteration number
    - **context: Additional context (history, timing, cost, etc.)
    """
    
    # Metadata (override in subclasses)
    name: str = "base"
    description: str = "Base stop condition"
    
    def __init__(self, **params):
        """
        Initialize stop condition with parameters.
        
        Args:
            **params: Condition-specific parameters
        """
        self.params = params
    
    @abstractmethod
    def check(
        self,
        best_score: float,
        current_score: float,
        iteration: int,
        **context
    ) -> StopDecision:
        """
        Check if execution should stop.
        
        Args:
            best_score: Best score achieved so far
            current_score: Score from most recent evaluation
            iteration: Current iteration number (1-indexed)
            **context: Additional context:
                - history: List[float] - All previous scores
                - elapsed_time: float - Time since start (seconds)
                - cost: float - Total cost so far
                - had_error: bool - Whether last run had error
                
        Returns:
            StopDecision indicating whether to stop and why
        """
        pass
    
    def __call__(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int,
        **context
    ) -> bool:
        """
        Callable interface - returns just the boolean decision.
        
        This allows stop conditions to be used as simple functions.
        
        Returns:
            True if should stop, False otherwise
        """
        return self.check(best_score, current_score, iteration, **context).should_stop

