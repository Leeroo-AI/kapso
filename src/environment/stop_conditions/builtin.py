# Built-in Stop Conditions
#
# Ready-to-use stop conditions for common stopping patterns.
# All conditions are automatically registered via @register_stop_condition decorator.
#
# Available conditions:
# - never: Never stop early
# - threshold: Stop when score reaches threshold
# - max_iterations: Stop after N iterations
# - plateau: Stop if no improvement for N iterations
# - cost_limit: Stop when cost limit reached
# - time_limit: Stop when time limit reached
# - consecutive_errors: Stop after N consecutive errors
# - composite: Combine multiple conditions

from typing import Any, Dict, List, Optional

from src.environment.stop_conditions.base import StopCondition, StopDecision
from src.environment.stop_conditions.factory import register_stop_condition


# =============================================================================
# Basic Stop Conditions
# =============================================================================

@register_stop_condition("never")
class NeverStopCondition(StopCondition):
    """
    Never stop early - run until budget exhausted.
    
    This is the default stop condition.
    """
    
    description = "Never stop early"
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        return StopDecision(
            should_stop=False, 
            reason="Never stop condition - continue until budget exhausted"
        )


@register_stop_condition("threshold")
class ThresholdStopCondition(StopCondition):
    """
    Stop when score reaches a threshold.
    
    Params:
        threshold: Score to reach
        maximize: If True, stop when score >= threshold; else when <= threshold
        
    Example:
        condition = StopConditionFactory.create("threshold", threshold=0.95)
    """
    
    description = "Stop when score reaches threshold"
    
    def __init__(self, threshold: float, maximize: bool = True, **params):
        super().__init__(**params)
        self.threshold = threshold
        self.maximize = maximize
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        if self.maximize:
            reached = best_score >= self.threshold
            comparison = ">="
        else:
            reached = best_score <= self.threshold
            comparison = "<="
        
        return StopDecision(
            should_stop=reached,
            reason=f"Threshold {comparison} {self.threshold}: best_score={best_score:.4f}",
            details={
                "threshold": self.threshold, 
                "best_score": best_score,
                "maximize": self.maximize,
            },
        )


@register_stop_condition("max_iterations")
class MaxIterationsStopCondition(StopCondition):
    """
    Stop after maximum iterations.
    
    Params:
        max_iter: Maximum number of iterations
        
    Example:
        condition = StopConditionFactory.create("max_iterations", max_iter=50)
    """
    
    description = "Stop after N iterations"
    
    def __init__(self, max_iter: int, **params):
        super().__init__(**params)
        self.max_iter = max_iter
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        reached = iteration >= self.max_iter
        return StopDecision(
            should_stop=reached,
            reason=f"Iteration {iteration}/{self.max_iter}",
            details={"iteration": iteration, "max_iter": self.max_iter},
        )


@register_stop_condition("plateau")
class PlateauStopCondition(StopCondition):
    """
    Stop if no improvement for N iterations.
    
    Params:
        patience: Number of iterations without improvement before stopping
        min_delta: Minimum improvement to count as progress
        
    Example:
        condition = StopConditionFactory.create("plateau", patience=10, min_delta=0.001)
    """
    
    description = "Stop if no improvement for N iterations"
    
    def __init__(
        self, 
        patience: int = 5, 
        min_delta: float = 0.001, 
        **params
    ):
        super().__init__(**params)
        self.patience = patience
        self.min_delta = min_delta
        self._best_seen: Optional[float] = None
        self._no_improve_count = 0
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        # Check for improvement
        if self._best_seen is None or best_score > self._best_seen + self.min_delta:
            self._best_seen = best_score
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        
        plateau = self._no_improve_count >= self.patience
        
        return StopDecision(
            should_stop=plateau,
            reason=f"No improvement for {self._no_improve_count}/{self.patience} iterations",
            details={
                "no_improve_count": self._no_improve_count,
                "patience": self.patience,
                "best_seen": self._best_seen,
                "min_delta": self.min_delta,
            },
        )


@register_stop_condition("cost_limit")
class CostLimitStopCondition(StopCondition):
    """
    Stop when cost exceeds limit.
    
    Params:
        max_cost: Maximum cost in dollars
        
    Note: Requires 'cost' in context when calling check()
        
    Example:
        condition = StopConditionFactory.create("cost_limit", max_cost=50.0)
    """
    
    description = "Stop when cost limit reached"
    
    def __init__(self, max_cost: float, **params):
        super().__init__(**params)
        self.max_cost = max_cost
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        cost = context.get("cost", 0.0)
        exceeded = cost >= self.max_cost
        
        return StopDecision(
            should_stop=exceeded,
            reason=f"Cost: ${cost:.2f} / ${self.max_cost:.2f}",
            details={"cost": cost, "max_cost": self.max_cost},
        )


@register_stop_condition("time_limit")
class TimeLimitStopCondition(StopCondition):
    """
    Stop when time limit exceeded.
    
    Params:
        max_seconds: Maximum time in seconds
        
    Note: Requires 'elapsed_time' in context when calling check()
        
    Example:
        condition = StopConditionFactory.create("time_limit", max_seconds=3600)
    """
    
    description = "Stop when time limit reached"
    
    def __init__(self, max_seconds: float, **params):
        super().__init__(**params)
        self.max_seconds = max_seconds
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        elapsed = context.get("elapsed_time", 0.0)
        exceeded = elapsed >= self.max_seconds
        
        minutes = elapsed / 60
        max_minutes = self.max_seconds / 60
        
        return StopDecision(
            should_stop=exceeded,
            reason=f"Time: {minutes:.1f}min / {max_minutes:.1f}min",
            details={"elapsed_seconds": elapsed, "max_seconds": self.max_seconds},
        )


@register_stop_condition("consecutive_errors")
class ConsecutiveErrorsStopCondition(StopCondition):
    """
    Stop after N consecutive errors.
    
    Params:
        max_errors: Maximum consecutive errors before stopping
        
    Note: Requires 'had_error' in context when calling check()
        
    Example:
        condition = StopConditionFactory.create("consecutive_errors", max_errors=5)
    """
    
    description = "Stop after N consecutive errors"
    
    def __init__(self, max_errors: int = 5, **params):
        super().__init__(**params)
        self.max_errors = max_errors
        self._error_count = 0
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        had_error = context.get("had_error", False)
        
        if had_error:
            self._error_count += 1
        else:
            self._error_count = 0
        
        stop = self._error_count >= self.max_errors
        
        return StopDecision(
            should_stop=stop,
            reason=f"Consecutive errors: {self._error_count}/{self.max_errors}",
            details={
                "error_count": self._error_count, 
                "max_errors": self.max_errors
            },
        )


# =============================================================================
# Composite Stop Condition
# =============================================================================

@register_stop_condition("composite")
class CompositeStopCondition(StopCondition):
    """
    Combine multiple stop conditions.
    
    Params:
        conditions: List of (type, params) tuples
        mode: "any" (stop if ANY condition met) or "all" (stop if ALL conditions met)
        
    Example:
        # Stop if score >= 0.95 OR after 50 iterations
        condition = StopConditionFactory.create(
            "composite",
            conditions=[
                ("threshold", {"threshold": 0.95}),
                ("max_iterations", {"max_iter": 50}),
            ],
            mode="any",
        )
        
        # Stop only if both conditions are met
        condition = StopConditionFactory.create(
            "composite",
            conditions=[
                ("threshold", {"threshold": 0.9}),
                ("plateau", {"patience": 5}),
            ],
            mode="all",
        )
    """
    
    description = "Combine multiple stop conditions"
    
    def __init__(
        self,
        conditions: List[tuple],  # [(type, params), ...]
        mode: str = "any",
        **params
    ):
        super().__init__(**params)
        
        # Import factory here to avoid circular import
        from src.environment.stop_conditions.factory import StopConditionFactory
        
        self.sub_conditions = [
            StopConditionFactory.create(cond_type, **cond_params)
            for cond_type, cond_params in conditions
        ]
        self.mode = mode.lower()
    
    def check(
        self, 
        best_score: float, 
        current_score: float, 
        iteration: int, 
        **context
    ) -> StopDecision:
        results = [
            cond.check(best_score, current_score, iteration, **context)
            for cond in self.sub_conditions
        ]
        
        if self.mode == "any":
            should_stop = any(r.should_stop for r in results)
            triggered = [r.reason for r in results if r.should_stop]
        else:  # "all"
            should_stop = all(r.should_stop for r in results)
            triggered = [r.reason for r in results]
        
        reason_str = ", ".join(triggered) if triggered else "none triggered"
        
        return StopDecision(
            should_stop=should_stop,
            reason=f"Composite ({self.mode}): {reason_str}",
            details={
                "mode": self.mode,
                "sub_results": [
                    {"should_stop": r.should_stop, "reason": r.reason}
                    for r in results
                ]
            },
        )

