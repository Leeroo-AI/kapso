"""Mechanical projection from one finalized linked node to its idea outcome."""

from typing import Mapping, Optional

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation.types import (
    EvaluationStatus,
    IdeaOutcome,
    IdeaRecord,
    ImplementationStatus,
    ObjectiveDirection,
)


def validation_tier(node: SearchNode) -> str:
    if node.eval_fidelity == "full" and node.build_fidelity == "full":
        return "full"
    if node.eval_fidelity == "full":
        return "validated"
    return "probe"


def build_idea_outcome(
    *,
    node: SearchNode,
    idea: IdeaRecord,
    nodes_by_id: Mapping[int, SearchNode],
    objective_direction: ObjectiveDirection,
) -> Optional[IdeaOutcome]:
    """Return no terminal outcome while a same-node recovery remains admissible."""
    if node.idea_id != idea.idea_id:
        raise ValueError("finalized node and idea identity do not match")
    if node.selection_batch_id != idea.selected_in_batch_id:
        raise ValueError("finalized node and idea selection batch do not match")
    if node.node_id != idea.experiment_node_id:
        raise ValueError("finalized node and idea experiment link do not match")
    if node.recoverable_error:
        if not node.had_error:
            raise ValueError("recoverable outcome requires a technical failure")
        return None
    if node.had_error:
        return IdeaOutcome(
            evaluation_status=EvaluationStatus.NOT_RUN,
            implementation_status=ImplementationStatus.FAILED_TECHNICAL,
            normalized_delta=None,
            validation_tier=None,
            actual_cost=node.cost_usd,
            actual_duration=node.duration_seconds,
        )
    if not node.evaluation_valid:
        evaluation_status = EvaluationStatus.INVALID
        normalized_delta = None
    elif node.score is None:
        evaluation_status = EvaluationStatus.INCONCLUSIVE
        normalized_delta = None
    else:
        evaluation_status = EvaluationStatus.VALID
        comparison_score = 0.0
        parent_id = idea.resolved_parent.node_id
        if parent_id is not None and parent_id != node.node_id:
            if parent_id not in nodes_by_id:
                raise ValueError("idea comparison parent is missing")
            parent = nodes_by_id[parent_id]
            if (
                parent.had_error
                or not parent.evaluation_valid
                or parent.score is None
            ):
                raise ValueError("idea comparison parent has no valid score")
            comparison_score = parent.score
        sign = (
            1.0
            if objective_direction == ObjectiveDirection.MAXIMIZE
            else -1.0
        )
        normalized_delta = sign * (node.score - comparison_score)
    return IdeaOutcome(
        evaluation_status=evaluation_status,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=normalized_delta,
        validation_tier=validation_tier(node),
        actual_cost=node.cost_usd,
        actual_duration=node.duration_seconds,
    )
