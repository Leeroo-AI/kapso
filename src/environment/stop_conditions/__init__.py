# Stop Conditions Module
#
# Pluggable stop condition functions for controlling experiment termination.
#
# Quick Start:
#     from src.environment.stop_conditions import StopConditionFactory
#     
#     # Create condition by name
#     condition = StopConditionFactory.create("threshold", threshold=0.95)
#     
#     # Check if should stop
#     decision = condition.check(best_score=0.9, current_score=0.85, iteration=10)
#     if decision.should_stop:
#         print(f"Stopping: {decision.reason}")
#     
#     # List available conditions
#     StopConditionFactory.list_conditions()
#
# Available Conditions:
#     - never: Never stop early
#     - threshold: Stop when score reaches threshold
#     - max_iterations: Stop after N iterations
#     - plateau: Stop if no improvement for N iterations
#     - cost_limit: Stop when cost limit reached
#     - time_limit: Stop when time limit reached
#     - consecutive_errors: Stop after N consecutive errors
#     - composite: Combine multiple conditions

# Base classes
from src.environment.stop_conditions.base import (
    StopCondition,
    StopDecision,
)

# Factory and decorator
from src.environment.stop_conditions.factory import (
    StopConditionFactory,
    register_stop_condition,
)

# Built-in conditions (import to register them)
from src.environment.stop_conditions.builtin import (
    NeverStopCondition,
    ThresholdStopCondition,
    MaxIterationsStopCondition,
    PlateauStopCondition,
    CostLimitStopCondition,
    TimeLimitStopCondition,
    ConsecutiveErrorsStopCondition,
    CompositeStopCondition,
)

__all__ = [
    # Base
    "StopCondition",
    "StopDecision",
    # Factory
    "StopConditionFactory",
    "register_stop_condition",
    # Built-in conditions
    "NeverStopCondition",
    "ThresholdStopCondition",
    "MaxIterationsStopCondition",
    "PlateauStopCondition",
    "CostLimitStopCondition",
    "TimeLimitStopCondition",
    "ConsecutiveErrorsStopCondition",
    "CompositeStopCondition",
]

