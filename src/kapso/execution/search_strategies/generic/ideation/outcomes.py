"""Mechanical projection from one finalized linked node to its idea outcome."""

from typing import Mapping, Optional

from kapso.execution.fidelity import project_score
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


def _comparable_node_score(node: SearchNode) -> Optional[float]:
    """Return the node score under its latest measurement identity."""
    if not node.evaluation_attempts:
        return None
    comparability = node.evaluation_attempts[-1].comparability_class
    if comparability.fidelity != node.eval_fidelity:
        return None
    score = project_score(node, comparability)
    if score != node.score:
        return None
    return score


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
        normalized_score = _comparable_node_score(node)
        comparison_score = 0.0
        parent_id = idea.resolved_parent.node_id
        if parent_id is not None and parent_id != node.node_id:
            if parent_id not in nodes_by_id:
                raise ValueError("idea comparison parent is missing")
            parent = nodes_by_id[parent_id]
            if parent.had_error or not parent.evaluation_valid:
                raise ValueError("idea comparison parent has no valid score")
            if not node.evaluation_attempts:
                normalized_score = None
            else:
                comparison_score = project_score(
                    parent,
                    node.evaluation_attempts[-1].comparability_class,
                )
        if normalized_score is None or comparison_score is None:
            evaluation_status = EvaluationStatus.INCONCLUSIVE
            normalized_delta = None
        else:
            evaluation_status = EvaluationStatus.VALID
            sign = 1.0 if objective_direction == ObjectiveDirection.MAXIMIZE else -1.0
            normalized_delta = sign * (normalized_score - comparison_score)
    return IdeaOutcome(
        evaluation_status=evaluation_status,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=normalized_delta,
        validation_tier=validation_tier(node),
        actual_cost=node.cost_usd,
        actual_duration=node.duration_seconds,
    )
