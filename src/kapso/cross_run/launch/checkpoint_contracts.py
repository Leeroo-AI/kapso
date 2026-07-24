"""Exact content-addressed checkpoint contracts for cross-run evolution."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.contracts import BootstrapPin
from kapso.cross_run.launch.derived_state_contracts import (
    RunDerivedStateGeneration,
    RunStateAuthority,
    RunStateLayout,
)
from kapso.cross_run.launch.resume_contracts import (
    RunEligibilityDisposition,
    RunSafetyState,
)
from kapso.execution.evaluation_integrity import (
    AGENT_GENERATED,
    PROVIDED,
    VALID_PROVENANCE,
    manifest_fingerprint,
)
from kapso.execution.search_strategies.node import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
    archive_is_compatible_descendant,
)
from kapso.execution.search_strategies.generic.ideation.types import BatchStatus

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GENERIC_FIELDS = {
    "evaluation_integrity",
    "evaluator_transition",
    "idea_archive_snapshot",
    "iteration_count",
    "node_history",
    "previous_errors",
    "scores_evaluator_id",
}
_TREE_FIELDS = {
    "evaluation_integrity",
    "experimentation_count",
    "node_history_ids",
    "nodes",
    "previous_errors",
}
_TREE_NODE_FIELDS = {
    "children_ids",
    "ideation_repo_memory_sections_consulted",
    "is_root",
    "is_terminated",
    "node_event_history",
    "parent_id",
}
_TREE_EVENTS = {"create", "expand", "experiment", "terminate"}
_EVALUATION_INTEGRITY_FIELDS = {"fingerprint", "manifest", "provenance"}
_EVALUATOR_TRANSITION_FIELDS = {
    "new_evaluator_id",
    "old_evaluator_id",
    "priority_node_id",
    "status",
}


class RunCheckpointContractError(ValueError):
    """A run checkpoint or its strategy state is invalid."""


class RunStrategyKind(str, Enum):
    """Strategies supported by the exact cross-run checkpoint."""

    GENERIC = "generic"
    BENCHMARK_TREE_SEARCH = "benchmark_tree_search"


class RunCheckpointStatus(str, Enum):
    """Durable campaign lifecycle state."""

    ACTIVE = "active"
    COMPLETED = "completed"


class RunCheckpointStop(str, Enum):
    """A resumable reason an active campaign yielded."""

    TIME_BUDGET = "time_budget"
    COST_BUDGET = "cost_budget"
    FINALIZATION_RESERVE = "finalization_reserve"


@dataclass(frozen=True)
class RunCheckpointHead(StrictContract):
    """Content-addressed monotonic witness for the checkpoint pathname."""

    run_checkpoint_head_id: str
    bootstrap_pin_id: str
    predecessor_head_id: str | None
    checkpoint: RunCheckpoint | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-checkpoint-head"
    IDENTITY_FIELD: ClassVar[str] = "run_checkpoint_head_id"

    def _validate(self) -> None:
        require_content_id(self.bootstrap_pin_id, "run checkpoint head bootstrap pin")
        if (
            self.bootstrap_pin_id.split(":sha256:", 1)[0]
            != BootstrapPin.CONTENT_NAMESPACE
        ):
            raise RunCheckpointContractError(
                "run checkpoint head bootstrap pin uses the wrong namespace"
            )
        empty = self.checkpoint is None
        if empty != (self.predecessor_head_id is None):
            raise RunCheckpointContractError(
                "run checkpoint head frontier fields differ"
            )
        if empty:
            return
        require_content_id(
            self.predecessor_head_id,
            "run checkpoint head predecessor",
        )
        if self.predecessor_head_id.split(":sha256:", 1)[0] != self.CONTENT_NAMESPACE:
            raise RunCheckpointContractError(
                "run checkpoint head dependency uses the wrong namespace"
            )
        if (
            type(self.checkpoint) is not RunCheckpoint
            or self.checkpoint.safety_state.bootstrap_pin.bootstrap_pin_id
            != self.bootstrap_pin_id
        ):
            raise RunCheckpointContractError(
                "run checkpoint head carries another launch frontier"
            )
        if (
            self.checkpoint.derived_state_generation.predecessor_checkpoint_head_id
            != self.predecessor_head_id
        ):
            raise RunCheckpointContractError(
                "run checkpoint derived generation names another predecessor head"
            )

    @classmethod
    def initial(cls, bootstrap_pin: BootstrapPin) -> "RunCheckpointHead":
        if type(bootstrap_pin) is not BootstrapPin:
            raise RunCheckpointContractError(
                "run checkpoint head requires one exact bootstrap pin"
            )
        return cls.mint(
            bootstrap_pin_id=bootstrap_pin.bootstrap_pin_id,
            predecessor_head_id=None,
            checkpoint=None,
        )

    def advance(self, checkpoint: "RunCheckpoint") -> "RunCheckpointHead":
        if type(checkpoint) is not RunCheckpoint:
            raise RunCheckpointContractError(
                "run checkpoint head requires one exact checkpoint"
            )
        if (
            checkpoint.safety_state.bootstrap_pin.bootstrap_pin_id
            != self.bootstrap_pin_id
        ):
            raise RunCheckpointContractError(
                "run checkpoint does not exactly advance its durable head"
            )
        checkpoint.require_predecessor(self.checkpoint)
        return type(self).mint(
            bootstrap_pin_id=self.bootstrap_pin_id,
            predecessor_head_id=self.run_checkpoint_head_id,
            checkpoint=checkpoint,
        )

    def require_checkpoint(self, checkpoint: "RunCheckpoint | None") -> None:
        if checkpoint is None:
            if self.checkpoint is not None:
                raise RunCheckpointContractError(
                    "durable checkpoint head names an absent checkpoint"
                )
            return
        if type(checkpoint) is not RunCheckpoint or checkpoint != self.checkpoint:
            raise RunCheckpointContractError(
                "durable checkpoint differs from its monotonic head"
            )


def _require_exact_fields(
    value: object,
    expected: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RunCheckpointContractError(f"{name} fields are incompatible")
    return value


def _require_non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RunCheckpointContractError(f"{name} must be non-negative")
    return value


def _require_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RunCheckpointContractError(f"{name} must be an array of strings")
    return tuple(value)


def _validate_evaluation_integrity(value: object) -> None:
    state = _require_exact_fields(
        value,
        _EVALUATION_INTEGRITY_FIELDS,
        "evaluation integrity",
    )
    provenance = state["provenance"]
    manifest = state["manifest"]
    fingerprint = state["fingerprint"]
    if provenance not in VALID_PROVENANCE:
        raise RunCheckpointContractError("evaluation integrity provenance is invalid")
    if not isinstance(manifest, dict) or any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or _DIGEST_PATTERN.fullmatch(digest) is None
        for path, digest in manifest.items()
    ):
        raise RunCheckpointContractError("evaluation integrity manifest is invalid")
    expected_fingerprint = manifest_fingerprint(manifest) if manifest else None
    if fingerprint != expected_fingerprint:
        raise RunCheckpointContractError("evaluation integrity fingerprint is invalid")


def _decode_exact_search_node(value: object) -> SearchNode:
    node_fields = {item.name for item in fields(SearchNode)}
    if not isinstance(value, dict) or set(value) != node_fields:
        raise RunCheckpointContractError("search node fields are incompatible")
    node = SearchNode.from_dict(value)
    if node.to_dict() != value:
        raise RunCheckpointContractError("search node is not canonically normalized")
    return node


def _validate_generic_state(
    state: dict[str, Any],
    campaign_id: str,
) -> tuple[IdeaArchiveState, tuple[SearchNode, ...]]:
    archive = IdeaArchiveState.from_dict(state["idea_archive_snapshot"])
    if archive.to_dict() != state["idea_archive_snapshot"]:
        raise RunCheckpointContractError(
            "generic strategy archive is not canonically normalized"
        )
    if archive.campaign_id != campaign_id:
        raise RunCheckpointContractError(
            "generic strategy archive belongs to another campaign"
        )
    active_batches = tuple(
        batch
        for batch in archive.batches
        if batch.status not in {BatchStatus.COMPLETED, BatchStatus.ABANDONED}
    )
    if len(active_batches) > 1 or (
        active_batches
        and active_batches[0].iteration_index
        != max(batch.iteration_index for batch in archive.batches)
    ):
        raise RunCheckpointContractError(
            "generic strategy archive has an invalid active batch"
        )
    raw_nodes = state["node_history"]
    if not isinstance(raw_nodes, list):
        raise RunCheckpointContractError(
            "generic strategy node history must be an array"
        )
    nodes = tuple(_decode_exact_search_node(value) for value in raw_nodes)
    node_ids = tuple(node.node_id for node in nodes)
    if node_ids != tuple(range(len(nodes))):
        raise RunCheckpointContractError(
            "generic strategy node IDs must be contiguous from zero"
        )
    iteration_count = _require_non_negative_integer(
        state["iteration_count"],
        "generic strategy iteration count",
    )
    if iteration_count < len(nodes):
        raise RunCheckpointContractError(
            "generic strategy iteration count is behind its nodes"
        )
    _require_string_tuple(
        state["previous_errors"],
        "generic strategy previous errors",
    )
    _validate_evaluation_integrity(state["evaluation_integrity"])
    if not isinstance(state["scores_evaluator_id"], str):
        raise RunCheckpointContractError("generic strategy evaluator ID must be text")
    transition = state["evaluator_transition"]
    if transition is not None:
        transition = _require_exact_fields(
            transition,
            _EVALUATOR_TRANSITION_FIELDS,
            "generic evaluator transition",
        )
        if (
            transition["status"] not in {"pending", "anchored"}
            or not isinstance(transition["old_evaluator_id"], str)
            or not transition["old_evaluator_id"]
            or not isinstance(transition["new_evaluator_id"], str)
            or not transition["new_evaluator_id"]
        ):
            raise RunCheckpointContractError("generic evaluator transition is invalid")
        priority_node_id = transition["priority_node_id"]
        if priority_node_id is not None:
            _require_non_negative_integer(
                priority_node_id,
                "generic evaluator transition priority node",
            )
            if priority_node_id not in node_ids:
                raise RunCheckpointContractError(
                    "generic evaluator transition references an unknown node"
                )
    nodes_by_id = {node.node_id: node for node in nodes}
    ideas_by_id = {idea.idea_id: idea for idea in archive.ideas}
    batches_by_id = {batch.batch_id: batch for batch in archive.batches}
    for node in nodes:
        if node.idea_id is None or node.selection_batch_id is None:
            raise RunCheckpointContractError(
                "generic strategy node lacks idea provenance"
            )
        idea = ideas_by_id.get(node.idea_id)
        batch = batches_by_id.get(node.selection_batch_id)
        if (
            idea is None
            or batch is None
            or idea.selected_in_batch_id != node.selection_batch_id
            or idea.experiment_node_id != node.node_id
            or batch.selection is None
            or batch.selection.selected_idea_id != node.idea_id
            or node.solution != idea.proposal
            or node.parent_node_id != idea.resolved_parent.node_id
        ):
            raise RunCheckpointContractError("generic strategy idea linkage is corrupt")
        if node.parent_node_id is None:
            if node.parent_branch_name not in {"", "main"}:
                raise RunCheckpointContractError(
                    "generic baseline parent branch is invalid"
                )
        else:
            parent = nodes_by_id.get(node.parent_node_id)
            if parent is None or parent.node_id >= node.node_id:
                raise RunCheckpointContractError(
                    "generic strategy parent must be an earlier node"
                )
            if (
                node.parent_branch_name
                and node.parent_branch_name != parent.branch_name
            ):
                raise RunCheckpointContractError(
                    "generic strategy parent node and branch differ"
                )
    linked_node_ids = tuple(
        sorted(
            idea.experiment_node_id
            for idea in archive.ideas
            if idea.experiment_node_id is not None
        )
    )
    if linked_node_ids != tuple(range(len(linked_node_ids))):
        raise RunCheckpointContractError(
            "generic archive experiment links are not contiguous"
        )
    if node_ids != linked_node_ids[: len(node_ids)]:
        raise RunCheckpointContractError(
            "generic strategy nodes are not an archive-linked prefix"
        )
    identities = tuple(batch.cross_run_identity for batch in archive.batches)
    if identities and (
        any(identity is None for identity in identities)
        or any(identity != identities[0] for identity in identities[1:])
    ):
        raise RunCheckpointContractError(
            "generic archive mixes cross-run launch identities"
        )
    return archive, nodes


def _validate_tree_state(
    state: dict[str, Any],
) -> tuple[SearchNode, ...]:
    raw_nodes = state["nodes"]
    if not isinstance(raw_nodes, list):
        raise RunCheckpointContractError("tree strategy nodes must be an array")
    base_fields = {item.name for item in fields(SearchNode)}
    expected_fields = base_fields | _TREE_NODE_FIELDS
    nodes: list[SearchNode] = []
    relationships: dict[int, dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or set(raw_node) != expected_fields:
            raise RunCheckpointContractError(
                "tree strategy node fields are incompatible"
            )
        node = _decode_exact_search_node({name: raw_node[name] for name in base_fields})
        if node.execution_revision != 0:
            raise RunCheckpointContractError(
                "tree strategy node execution revision must remain zero"
            )
        if node.node_id in relationships:
            raise RunCheckpointContractError(
                "tree strategy contains duplicate node IDs"
            )
        if type(raw_node["is_terminated"]) is not bool:
            raise RunCheckpointContractError(
                "tree strategy termination state must be boolean"
            )
        parent_id = raw_node["parent_id"]
        if parent_id is not None:
            _require_non_negative_integer(parent_id, "tree strategy parent ID")
        if node.parent_node_id != parent_id:
            raise RunCheckpointContractError("tree strategy parent identities differ")
        if raw_node["is_root"] is not (parent_id is None):
            raise RunCheckpointContractError("tree strategy root flag is invalid")
        children_ids = raw_node["children_ids"]
        if (
            not isinstance(children_ids, list)
            or any(
                type(child_id) is not int or child_id < 0 for child_id in children_ids
            )
            or len(children_ids) != len(set(children_ids))
        ):
            raise RunCheckpointContractError(
                "tree strategy children must be unique node IDs"
            )
        events = raw_node["node_event_history"]
        if not isinstance(events, list):
            raise RunCheckpointContractError(
                "tree strategy event history must be an array"
            )
        previous_event_iteration = -1
        for event in events:
            if (
                not isinstance(event, list)
                or len(event) != 2
                or type(event[0]) is not int
                or event[0] < 0
                or event[0] < previous_event_iteration
                or event[1] not in _TREE_EVENTS
            ):
                raise RunCheckpointContractError(
                    "tree strategy event history is invalid"
                )
            previous_event_iteration = event[0]
        sections = raw_node["ideation_repo_memory_sections_consulted"]
        if (
            not isinstance(sections, list)
            or any(not isinstance(section, str) or not section for section in sections)
            or len(sections) != len(set(sections))
        ):
            raise RunCheckpointContractError(
                "tree strategy consulted sections must be unique strings"
            )
        nodes.append(node)
        relationships[node.node_id] = raw_node
    if tuple(node.node_id for node in nodes) != tuple(range(len(nodes))):
        raise RunCheckpointContractError(
            "tree strategy node IDs must be ordered and contiguous from zero"
        )
    node_ids = set(relationships)
    for node_id, relationship in relationships.items():
        parent_id = relationship["parent_id"]
        if parent_id is not None and parent_id not in node_ids:
            raise RunCheckpointContractError(
                "tree strategy references an unknown parent"
            )
        for child_id in relationship["children_ids"]:
            child = relationships.get(child_id)
            if child is None or child["parent_id"] != node_id:
                raise RunCheckpointContractError(
                    "tree strategy parent and child links are not reciprocal"
                )
        if (
            parent_id is not None
            and node_id not in relationships[parent_id]["children_ids"]
        ):
            raise RunCheckpointContractError(
                "tree strategy child and parent links are not reciprocal"
            )
    visited: set[int] = set()
    active: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in active:
            raise RunCheckpointContractError("tree strategy contains a cycle")
        if node_id in visited:
            return
        active.add(node_id)
        for child_id in relationships[node_id]["children_ids"]:
            visit(child_id)
        active.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(node_ids):
        visit(node_id)
    history_ids = state["node_history_ids"]
    if (
        not isinstance(history_ids, list)
        or any(type(node_id) is not int or node_id < 0 for node_id in history_ids)
        or len(history_ids) != len(set(history_ids))
        or not set(history_ids).issubset(node_ids)
    ):
        raise RunCheckpointContractError("tree strategy history IDs are invalid")
    experimentation_count = _require_non_negative_integer(
        state["experimentation_count"],
        "tree strategy experimentation count",
    )
    if any(
        event[0] > experimentation_count
        for relationship in relationships.values()
        for event in relationship["node_event_history"]
    ):
        raise RunCheckpointContractError(
            "tree strategy event occurs after its experimentation frontier"
        )
    _require_string_tuple(
        state["previous_errors"],
        "tree strategy previous errors",
    )
    _validate_evaluation_integrity(state["evaluation_integrity"])
    if any(
        nodes[node_id].evaluation_provenance
        != state["evaluation_integrity"]["provenance"]
        for node_id in history_ids
    ):
        raise RunCheckpointContractError(
            "tree strategy executed-node evaluation provenance differs"
        )
    return tuple(nodes)


def _node_is_compatible_descendant(
    original: SearchNode,
    current: SearchNode,
    *,
    mutable_populated_identity_fields: frozenset[str] = frozenset(),
) -> bool:
    immutable_fields = (
        "node_id",
        "parent_node_id",
        "idea_id",
        "selection_batch_id",
        "solution",
    )
    populated_identity_fields = (
        "branch_name",
        "parent_branch_name",
        "implementation_base_ref",
        "diff_base_ref",
        "feedback_base_ref",
    )
    if any(
        getattr(current, name) != getattr(original, name) for name in immutable_fields
    ):
        return False
    if any(
        getattr(original, name) and getattr(current, name) != getattr(original, name)
        for name in populated_identity_fields
        if name not in mutable_populated_identity_fields
    ):
        return False
    return current.evaluation_attempts[: len(original.evaluation_attempts)] == (
        original.evaluation_attempts
    )


def _generic_node_is_revision_descendant(
    original: SearchNode,
    current: SearchNode,
) -> bool:
    if (
        current.execution_revision != original.execution_revision + 1
        or not _node_is_compatible_descendant(
            original,
            current,
            mutable_populated_identity_fields=(
                frozenset({"implementation_base_ref"})
                if original.recoverable_error
                else frozenset()
            ),
        )
    ):
        return False
    appended_attempt_count = len(current.evaluation_attempts) - len(
        original.evaluation_attempts
    )
    if original.recoverable_error:
        if (
            not current.implementation_base_ref
            or appended_attempt_count not in {0, 1}
            or not _node_cumulative_resources_are_descendant(original, current)
        ):
            return False
        if current.score is not None and (
            appended_attempt_count != 1
            or current.score != current.evaluation_attempts[-1].score
        ):
            return False
        return True
    permitted_changes = {
        "evaluation_attempts",
        "evaluation_valid",
        "execution_revision",
        "score",
    }
    if any(
        getattr(current, item.name) != getattr(original, item.name)
        for item in fields(SearchNode)
        if item.name not in permitted_changes
    ):
        return False
    if appended_attempt_count == 1:
        appended_score = current.evaluation_attempts[-1].score
        return current.score in {None, original.score, appended_score} and (
            current.evaluation_valid == original.evaluation_valid
            or (not original.evaluation_valid and current.evaluation_valid)
        )
    return (
        appended_attempt_count == 0
        and original.score is not None
        and current.score is None
        and current.evaluation_valid == original.evaluation_valid
    )


def _node_cumulative_resources_are_descendant(
    original: SearchNode,
    current: SearchNode,
) -> bool:
    for name in ("duration_seconds", "cost_usd"):
        previous = getattr(original, name)
        candidate = getattr(current, name)
        if previous is not None and (candidate is None or candidate < previous):
            return False
    for phase, previous_measurements in original.phase_telemetry.items():
        candidate_measurements = current.phase_telemetry.get(phase)
        if candidate_measurements is None or any(
            measurement not in candidate_measurements
            or candidate_measurements[measurement] < previous_value
            for measurement, previous_value in previous_measurements.items()
        ):
            return False
    return True


def _tree_node_is_phase_descendant(
    original: SearchNode,
    current: SearchNode,
) -> bool:
    if (
        original.execution_revision != current.execution_revision
        or not _node_is_compatible_descendant(original, current)
    ):
        return False
    immutable_fields = {
        "execution_revision",
        "idea_id",
        "node_id",
        "parent_node_id",
        "selection_batch_id",
        "solution",
    }
    false_to_true_fields = {
        "had_error",
        "recoverable_error",
        "should_stop",
    }
    for item in fields(SearchNode):
        name = item.name
        original_value = getattr(original, name)
        current_value = getattr(current, name)
        if (
            name in immutable_fields
            or original_value == current_value
            or (
                name == "evaluation_attempts"
                and current_value[: len(original_value)] == original_value
            )
            or (
                name in false_to_true_fields
                and original_value is False
                and current_value is True
            )
            or (
                name == "evaluation_valid"
                and original_value is True
                and current_value is False
            )
            or (
                name == "evaluation_provenance"
                and original_value == AGENT_GENERATED
                and current_value == PROVIDED
            )
            or original_value is None
            or original_value == ""
            or (isinstance(original_value, (dict, list)) and not original_value)
        ):
            continue
        return False
    return True


def _require_evaluator_successor(
    original: dict[str, Any],
    current: dict[str, Any],
) -> None:
    original_evaluator = original["scores_evaluator_id"]
    current_evaluator = current["scores_evaluator_id"]
    original_transition = original["evaluator_transition"]
    current_transition = current["evaluator_transition"]
    if (
        not original["node_history"]
        and not original_evaluator
        and original_transition is None
    ):
        if current_transition is None or (
            current_transition["old_evaluator_id"] == current_evaluator
            and current_transition["status"] == "pending"
        ):
            return
    elif original_transition is None:
        if current_transition is None and current_evaluator == original_evaluator:
            return
        if (
            current_transition is not None
            and current_transition["old_evaluator_id"] == original_evaluator
            and (
                (
                    current_transition["status"] == "pending"
                    and current_evaluator == original_evaluator
                )
                or (
                    current_transition["status"] == "anchored"
                    and current_evaluator == current_transition["new_evaluator_id"]
                )
            )
        ):
            return
    elif original_transition["status"] == "pending":
        if (
            current_transition is not None
            and {
                "old_evaluator_id": current_transition["old_evaluator_id"],
                "new_evaluator_id": current_transition["new_evaluator_id"],
                "priority_node_id": current_transition["priority_node_id"],
            }
            == {
                "old_evaluator_id": original_transition["old_evaluator_id"],
                "new_evaluator_id": original_transition["new_evaluator_id"],
                "priority_node_id": original_transition["priority_node_id"],
            }
            and (
                (
                    current_transition["status"] == "pending"
                    and current_evaluator == original_evaluator
                )
                or (
                    current_transition["status"] == "anchored"
                    and current_evaluator == current_transition["new_evaluator_id"]
                )
            )
        ):
            return
    else:
        anchored_evaluator = original_transition["new_evaluator_id"]
        if current_evaluator == anchored_evaluator and (
            current_transition == original_transition
            or (
                current_transition is not None
                and current_transition["old_evaluator_id"] == anchored_evaluator
                and current_transition["status"] == "pending"
            )
        ):
            return
    raise RunCheckpointContractError(
        "generic strategy evaluator authority changed or rolled back"
    )


@dataclass(frozen=True)
class RunStrategyState(StrictContract):
    """One fully validated, canonical strategy-state payload."""

    strategy_state_id: str
    strategy_kind: RunStrategyKind
    campaign_id: str
    canonical_state_json: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-strategy-state"
    IDENTITY_FIELD: ClassVar[str] = "strategy_state_id"

    def _validate(self) -> None:
        require_identifier(self.campaign_id, "run strategy campaign")
        parsed = parse_json_bytes(self.canonical_state_json)
        if canonical_json_bytes(parsed).decode("utf-8") != self.canonical_state_json:
            raise RunCheckpointContractError("run strategy state is not canonical JSON")
        if self.strategy_kind is RunStrategyKind.GENERIC:
            state = _require_exact_fields(
                parsed,
                _GENERIC_FIELDS,
                "generic strategy state",
            )
            _validate_generic_state(state, self.campaign_id)
        else:
            state = _require_exact_fields(
                parsed,
                _TREE_FIELDS,
                "tree strategy state",
            )
            _validate_tree_state(state)

    @classmethod
    def build(
        cls,
        *,
        strategy_kind: RunStrategyKind,
        campaign_id: str,
        state: Mapping[str, Any],
    ) -> "RunStrategyState":
        return cls.mint(
            strategy_kind=strategy_kind,
            campaign_id=campaign_id,
            canonical_state_json=canonical_json_bytes(state).decode("utf-8"),
        )

    def parsed_state(self) -> dict[str, Any]:
        parsed = parse_json_bytes(self.canonical_state_json)
        if not isinstance(parsed, dict):
            raise RunCheckpointContractError("run strategy state must be an object")
        return parsed

    @property
    def iteration_count(self) -> int:
        state = self.parsed_state()
        field = (
            "iteration_count"
            if self.strategy_kind is RunStrategyKind.GENERIC
            else "experimentation_count"
        )
        return state[field]

    def nodes(self) -> tuple[SearchNode, ...]:
        state = self.parsed_state()
        if self.strategy_kind is RunStrategyKind.GENERIC:
            _, nodes = _validate_generic_state(state, self.campaign_id)
            return nodes
        return _validate_tree_state(state)

    @property
    def durable_revision_count(self) -> int:
        """Return the exact number of executed node revisions in this state."""

        state = self.parsed_state()
        if self.strategy_kind is RunStrategyKind.GENERIC:
            return sum(node.execution_revision + 1 for node in self.nodes())
        return len(state["node_history_ids"])

    def describes_empty_durable_frontier(self) -> bool:
        state = self.parsed_state()
        if state["previous_errors"]:
            return False
        if self.strategy_kind is RunStrategyKind.BENCHMARK_TREE_SEARCH:
            return not state["nodes"] and not state["node_history_ids"]
        archive, nodes = _validate_generic_state(state, self.campaign_id)
        return (
            archive.revision == 0
            and archive.created_at == archive.updated_at
            and not archive.batches
            and not archive.ideas
            and not archive.claims
            and not archive.gaps
            and not nodes
            and state["evaluator_transition"] is None
        )

    def archive_state(self) -> IdeaArchiveState | None:
        if self.strategy_kind is not RunStrategyKind.GENERIC:
            return None
        state = self.parsed_state()
        archive, _ = _validate_generic_state(state, self.campaign_id)
        return archive

    def require_bootstrap_pin(self, bootstrap_pin: BootstrapPin) -> None:
        if type(bootstrap_pin) is not BootstrapPin:
            raise RunCheckpointContractError(
                "run strategy state requires one bootstrap pin"
            )
        manifest = bootstrap_pin.launch_manifest
        installation = bootstrap_pin.installation_receipt
        if (
            self.campaign_id != installation.campaign_id
            or self.strategy_kind.value != manifest.launch_request.search_mode
        ):
            raise RunCheckpointContractError(
                "run strategy state belongs to another launch"
            )
        archive = self.archive_state()
        if archive is None or not archive.batches:
            return
        identity = archive.batches[0].cross_run_identity
        if (
            identity is None
            or identity.launch_manifest_id != manifest.launch_manifest_id
            or identity.scope_contract_id != manifest.scope_contract.scope_contract_id
            or identity.knowledge_snapshot_id != manifest.knowledge_manifest.snapshot_id
            or identity.expert_base_release_id != manifest.expert_manifest.release_id
            or identity.embedding_space_id
            != manifest.knowledge_embedding_space.embedding_space_id
            or identity.task_context_binding != manifest.task_context_binding
        ):
            raise RunCheckpointContractError(
                "generic strategy state has another cross-run identity"
            )

    def require_predecessor(self, predecessor: "RunStrategyState") -> None:
        if (
            type(predecessor) is not RunStrategyState
            or self.strategy_kind is not predecessor.strategy_kind
            or self.campaign_id != predecessor.campaign_id
            or self.iteration_count < predecessor.iteration_count
        ):
            raise RunCheckpointContractError(
                "run strategy state changed identity or rolled back"
            )
        current_nodes = {node.node_id: node for node in self.nodes()}
        for previous_node in predecessor.nodes():
            current_node = current_nodes.get(previous_node.node_id)
            if current_node is None:
                raise RunCheckpointContractError(
                    "run strategy node history changed or rolled back"
                )
            if self.strategy_kind is RunStrategyKind.BENCHMARK_TREE_SEARCH:
                valid_successor = _tree_node_is_phase_descendant(
                    previous_node,
                    current_node,
                )
            elif current_node.execution_revision == previous_node.execution_revision:
                valid_successor = current_node == previous_node
            else:
                valid_successor = _generic_node_is_revision_descendant(
                    previous_node,
                    current_node,
                )
            if not valid_successor:
                raise RunCheckpointContractError(
                    "run strategy node history changed or rolled back"
                )
        previous_archive = predecessor.archive_state()
        current_archive = self.archive_state()
        if previous_archive is not None and current_archive is not None:
            if not archive_is_compatible_descendant(
                previous_archive,
                current_archive,
            ):
                raise RunCheckpointContractError(
                    "run strategy archive changed or rolled back"
                )
        previous_state = predecessor.parsed_state()
        current_state = self.parsed_state()
        if (
            current_state["previous_errors"][: len(previous_state["previous_errors"])]
            != previous_state["previous_errors"]
            or current_state["evaluation_integrity"]
            != previous_state["evaluation_integrity"]
        ):
            raise RunCheckpointContractError(
                "run strategy prompt or evaluation history changed"
            )
        if self.strategy_kind is RunStrategyKind.BENCHMARK_TREE_SEARCH:
            previous_relationships = {
                node["node_id"]: node for node in previous_state["nodes"]
            }
            current_relationships = {
                node["node_id"]: node for node in current_state["nodes"]
            }
            for node_id, previous_relationship in previous_relationships.items():
                current_relationship = current_relationships[node_id]
                if (
                    current_relationship["parent_id"]
                    != previous_relationship["parent_id"]
                    or current_relationship["children_ids"][
                        : len(previous_relationship["children_ids"])
                    ]
                    != previous_relationship["children_ids"]
                    or current_relationship["node_event_history"][
                        : len(previous_relationship["node_event_history"])
                    ]
                    != previous_relationship["node_event_history"]
                    or current_relationship["ideation_repo_memory_sections_consulted"][
                        : len(
                            previous_relationship[
                                "ideation_repo_memory_sections_consulted"
                            ]
                        )
                    ]
                    != previous_relationship["ideation_repo_memory_sections_consulted"]
                    or (
                        previous_relationship["is_terminated"]
                        and not current_relationship["is_terminated"]
                    )
                ):
                    raise RunCheckpointContractError(
                        "tree strategy lineage changed or rolled back"
                    )
            if (
                current_state["node_history_ids"][
                    : len(previous_state["node_history_ids"])
                ]
                != previous_state["node_history_ids"]
            ):
                raise RunCheckpointContractError(
                    "tree strategy experiment history changed or rolled back"
                )
        else:
            _require_evaluator_successor(previous_state, current_state)


@dataclass(frozen=True)
class RunFeedbackSource(StrictContract):
    """Exact node revision supplying the next-iteration feedback."""

    node_id: int
    execution_revision: int

    def _validate(self) -> None:
        _require_non_negative_integer(self.node_id, "run feedback node ID")
        _require_non_negative_integer(
            self.execution_revision,
            "run feedback execution revision",
        )


@dataclass(frozen=True)
class RunTerminationDecision(StrictContract):
    """Terminal policy decision kept separate from measured node evidence."""

    delivery_source: RunFeedbackSource
    reasons: tuple[str, ...]

    def _validate(self) -> None:
        if type(self.delivery_source) is not RunFeedbackSource:
            raise RunCheckpointContractError(
                "run termination requires an exact delivery source"
            )
        if not self.reasons or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.reasons
        ):
            raise RunCheckpointContractError(
                "run termination requires non-empty policy reasons"
            )


@dataclass(frozen=True)
class RunCheckpoint(StrictContract):
    """Complete, immutable authority for one durable campaign frontier."""

    run_checkpoint_id: str
    predecessor_checkpoint_id: str | None
    checkpoint_sequence: int
    status: RunCheckpointStatus
    last_stop: RunCheckpointStop | None
    completed_iterations: int
    cumulative_cost: float
    elapsed_seconds: float
    cost_by_component: Mapping[str, float]
    feedback_source: RunFeedbackSource | None
    current_feedback: str | None
    termination_decision: RunTerminationDecision | None
    strategy_state: RunStrategyState
    safety_state: RunSafetyState
    derived_state_generation: RunDerivedStateGeneration
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-checkpoint"
    IDENTITY_FIELD: ClassVar[str] = "run_checkpoint_id"

    def _validate(self) -> None:
        if (self.predecessor_checkpoint_id is None) != (self.checkpoint_sequence == 0):
            raise RunCheckpointContractError(
                "run checkpoint predecessor and sequence differ"
            )
        if self.predecessor_checkpoint_id is not None:
            require_content_id(
                self.predecessor_checkpoint_id,
                "run checkpoint predecessor",
            )
            if (
                self.predecessor_checkpoint_id.split(":sha256:", 1)[0]
                != self.CONTENT_NAMESPACE
            ):
                raise RunCheckpointContractError(
                    "run checkpoint predecessor uses the wrong namespace"
                )
        _require_non_negative_integer(
            self.checkpoint_sequence,
            "run checkpoint sequence",
        )
        _require_non_negative_integer(
            self.completed_iterations,
            "run checkpoint completed iterations",
        )
        for value, name in (
            (self.cumulative_cost, "cumulative cost"),
            (self.elapsed_seconds, "elapsed seconds"),
        ):
            if not math.isfinite(value) or value < 0:
                raise RunCheckpointContractError(
                    f"run checkpoint {name} must be finite and non-negative"
                )
        for component, cost in self.cost_by_component.items():
            require_identifier(component, "run checkpoint cost component")
            if not math.isfinite(cost) or cost < 0:
                raise RunCheckpointContractError(
                    "run checkpoint component cost must be finite and non-negative"
                )
        if (
            type(self.strategy_state) is not RunStrategyState
            or type(self.safety_state) is not RunSafetyState
            or type(self.derived_state_generation) is not RunDerivedStateGeneration
        ):
            raise RunCheckpointContractError(
                "run checkpoint requires typed strategy, safety, and derived states"
            )
        if (self.feedback_source is None) != (self.current_feedback is None):
            raise RunCheckpointContractError(
                "run checkpoint feedback and source must be present together"
            )
        if self.feedback_source is not None:
            nodes = {node.node_id: node for node in self.strategy_state.nodes()}
            source = nodes.get(self.feedback_source.node_id)
            if (
                source is None
                or source.execution_revision != self.feedback_source.execution_revision
                or source.feedback != self.current_feedback
            ):
                raise RunCheckpointContractError(
                    "run checkpoint feedback differs from its node revision"
                )
        if self.status is RunCheckpointStatus.COMPLETED:
            if (
                type(self.termination_decision) is not RunTerminationDecision
                or self.feedback_source is not None
            ):
                raise RunCheckpointContractError(
                    "completed run checkpoint requires one terminal decision"
                )
            nodes = {node.node_id: node for node in self.strategy_state.nodes()}
            source = nodes.get(self.termination_decision.delivery_source.node_id)
            if (
                source is None
                or source.execution_revision
                != self.termination_decision.delivery_source.execution_revision
                or source.score is None
                or source.had_error
                or source.recoverable_error
                or not source.evaluation_valid
            ):
                raise RunCheckpointContractError(
                    "run termination delivery source is not eligible"
                )
        elif self.termination_decision is not None:
            raise RunCheckpointContractError(
                "active run checkpoint cannot carry a terminal decision"
            )
        self.strategy_state.require_bootstrap_pin(self.safety_state.bootstrap_pin)
        generation = self.derived_state_generation
        bootstrap_pin = self.safety_state.bootstrap_pin
        evidence = self.safety_state.derivative_frontier.evidence
        if (
            generation.bootstrap_pin_id != bootstrap_pin.bootstrap_pin_id
            or generation.predecessor_checkpoint_id != self.predecessor_checkpoint_id
            or generation.target_evidence_id != evidence.evidence_id
        ):
            raise RunCheckpointContractError(
                "run checkpoint derived generation belongs to another frontier"
            )
        receipt_layout = bootstrap_pin.installation_receipt.layout
        authority_paths = {
            RunStateAuthority.ACTION_LEDGER: (
                receipt_layout.run_action_ledger_relative_path
            ),
            RunStateAuthority.EXPERIMENT_HISTORY: (
                receipt_layout.run_experiment_history_relative_path
            ),
            RunStateAuthority.EXECUTION_JOURNAL: (
                receipt_layout.run_execution_journal_relative_path
            ),
        }
        if self.strategy_state.strategy_kind is RunStrategyKind.GENERIC:
            authority_paths[RunStateAuthority.IDEA_ARCHIVE] = (
                receipt_layout.run_idea_archive_relative_path
            )
        expected_layout = RunStateLayout.build(
            strategy_kind=self.strategy_state.strategy_kind.value,
            authority_paths=authority_paths,
        )
        if generation.run_state_layout != expected_layout:
            raise RunCheckpointContractError(
                "run checkpoint derived-state layout differs from its launch"
            )
        transitions_by_binding = {
            transition.authority_binding_id: transition
            for transition in generation.payload_transitions
        }
        if set(evidence.state_authority_digests) != {
            authority.value for authority in authority_paths
        } or set(evidence.state_authority_revisions) != {
            authority.value for authority in authority_paths
        }:
            raise RunCheckpointContractError(
                "run checkpoint state-authority set differs from its strategy"
            )
        transitions_by_authority = {
            binding.authority: transitions_by_binding[binding.authority_binding_id]
            for binding in generation.run_state_layout.bindings
        }
        if any(
            transition.target_digest
            != evidence.state_authority_digests[authority.value]
            or transition.target_revision
            != evidence.state_authority_revisions[authority.value]
            for authority, transition in transitions_by_authority.items()
        ):
            raise RunCheckpointContractError(
                "run checkpoint derived generation differs from safety evidence"
            )
        archive = self.strategy_state.archive_state()
        if archive is not None:
            archive_payload = canonical_json_bytes(archive.to_dict())
            archive_transition = transitions_by_authority[
                RunStateAuthority.IDEA_ARCHIVE
            ]
            if (
                archive.revision != archive_transition.target_revision
                or tree_or_blob_digest(archive_payload)
                != archive_transition.target_digest
                or len(archive_payload) != archive_transition.target_size_bytes
            ):
                raise RunCheckpointContractError(
                    "run checkpoint archive and derived generation differ"
                )
        experiment_revision = evidence.state_authority_revisions[
            RunStateAuthority.EXPERIMENT_HISTORY.value
        ]
        journal_revision = evidence.state_authority_revisions[
            RunStateAuthority.EXECUTION_JOURNAL.value
        ]
        if (
            experiment_revision != journal_revision
            or journal_revision != self.strategy_state.durable_revision_count
        ):
            raise RunCheckpointContractError(
                "run checkpoint executed-revision frontiers differ"
            )
        if self.completed_iterations != self.strategy_state.iteration_count:
            raise RunCheckpointContractError(
                "run checkpoint iterations differ from its strategy frontier"
            )
        if self.status is RunCheckpointStatus.COMPLETED and self.last_stop is not None:
            raise RunCheckpointContractError(
                "completed run checkpoint cannot have a resumable stop"
            )
        expected_dependencies = {
            self.strategy_state.strategy_state_id,
            self.safety_state.safety_state_id,
            self.derived_state_generation.generation_id,
        }
        if self.predecessor_checkpoint_id is not None:
            expected_dependencies.add(self.predecessor_checkpoint_id)
        if (
            self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids)))
            or set(self.exact_dependency_ids) != expected_dependencies
        ):
            raise RunCheckpointContractError(
                "run checkpoint dependency closure is not exact"
            )
        if self.predecessor_checkpoint_id is None and (
            self.status is not RunCheckpointStatus.ACTIVE
            or self.last_stop is not None
            or self.completed_iterations != 0
            or self.cumulative_cost != 0.0
            or self.elapsed_seconds != 0.0
            or self.cost_by_component
            or self.feedback_source is not None
            or self.strategy_state.iteration_count != 0
            or not self.strategy_state.describes_empty_durable_frontier()
            or self.safety_state.predecessor_safety_state_id is not None
            or self.derived_state_generation.predecessor_checkpoint_head_id
            != RunCheckpointHead.initial(
                self.safety_state.bootstrap_pin
            ).run_checkpoint_head_id
        ):
            raise RunCheckpointContractError(
                "initial run checkpoint must describe the empty durable frontier"
            )

    @classmethod
    def build(
        cls,
        *,
        predecessor: "RunCheckpoint | None",
        status: RunCheckpointStatus,
        last_stop: RunCheckpointStop | None,
        completed_iterations: int,
        cumulative_cost: float,
        elapsed_seconds: float,
        cost_by_component: Mapping[str, float],
        feedback_source: RunFeedbackSource | None,
        current_feedback: str | None,
        termination_decision: RunTerminationDecision | None,
        strategy_state: RunStrategyState,
        safety_state: RunSafetyState,
        derived_state_generation: RunDerivedStateGeneration,
    ) -> "RunCheckpoint":
        predecessor_id = None if predecessor is None else predecessor.run_checkpoint_id
        dependencies = {
            strategy_state.strategy_state_id,
            safety_state.safety_state_id,
            derived_state_generation.generation_id,
        }
        if predecessor_id is not None:
            dependencies.add(predecessor_id)
        checkpoint = cls.mint(
            predecessor_checkpoint_id=predecessor_id,
            checkpoint_sequence=(
                0 if predecessor is None else predecessor.checkpoint_sequence + 1
            ),
            status=status,
            last_stop=last_stop,
            completed_iterations=completed_iterations,
            cumulative_cost=float(cumulative_cost),
            elapsed_seconds=float(elapsed_seconds),
            cost_by_component={
                component: float(cost) for component, cost in cost_by_component.items()
            },
            feedback_source=feedback_source,
            current_feedback=current_feedback,
            termination_decision=termination_decision,
            strategy_state=strategy_state,
            safety_state=safety_state,
            derived_state_generation=derived_state_generation,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        checkpoint.require_predecessor(predecessor)
        return checkpoint

    def require_bootstrap_pin(self, bootstrap_pin: BootstrapPin) -> None:
        self.safety_state.require_bootstrap_pin(bootstrap_pin)
        self.strategy_state.require_bootstrap_pin(bootstrap_pin)

    def require_predecessor(self, predecessor: "RunCheckpoint | None") -> None:
        if predecessor is None:
            if self.predecessor_checkpoint_id is not None:
                raise RunCheckpointContractError(
                    "initial run checkpoint has a predecessor"
                )
            return
        if type(predecessor) is not RunCheckpoint:
            raise RunCheckpointContractError(
                "run checkpoint predecessor has the wrong type"
            )
        if (
            predecessor.status is RunCheckpointStatus.COMPLETED
            or predecessor.safety_state.disposition
            is RunEligibilityDisposition.SECURITY_BLOCKED
        ):
            raise RunCheckpointContractError(
                "terminal run checkpoint cannot have a successor"
            )
        if (
            self.predecessor_checkpoint_id != predecessor.run_checkpoint_id
            or self.checkpoint_sequence != predecessor.checkpoint_sequence + 1
            or self.completed_iterations < predecessor.completed_iterations
            or self.cumulative_cost < predecessor.cumulative_cost
            or self.elapsed_seconds < predecessor.elapsed_seconds
            or (
                self.completed_iterations == predecessor.completed_iterations
                and self.status is RunCheckpointStatus.ACTIVE
                and (
                    self.feedback_source != predecessor.feedback_source
                    or self.current_feedback != predecessor.current_feedback
                )
            )
            or any(
                component not in self.cost_by_component
                or self.cost_by_component[component] < cost
                for component, cost in predecessor.cost_by_component.items()
            )
        ):
            raise RunCheckpointContractError(
                "run checkpoint changed or rolled back its predecessor"
            )
        self.strategy_state.require_predecessor(predecessor.strategy_state)
        self.safety_state.require_predecessor(predecessor.safety_state)
        predecessor_generation = predecessor.derived_state_generation
        generation = self.derived_state_generation
        if (
            generation.run_state_layout != predecessor_generation.run_state_layout
            or generation.predecessor_evidence_id
            != predecessor.safety_state.derivative_frontier.evidence.evidence_id
        ):
            raise RunCheckpointContractError(
                "run checkpoint derived generation changed its predecessor"
            )
        predecessor_targets = {
            transition.authority_binding_id: transition
            for transition in predecessor_generation.payload_transitions
        }
        if any(
            transition.predecessor_digest
            != predecessor_targets[transition.authority_binding_id].target_digest
            or transition.predecessor_revision
            != predecessor_targets[transition.authority_binding_id].target_revision
            or transition.predecessor_size_bytes
            != predecessor_targets[transition.authority_binding_id].target_size_bytes
            for transition in generation.payload_transitions
        ):
            raise RunCheckpointContractError(
                "run checkpoint derived payloads changed or skipped their predecessor"
            )


__all__ = [
    "RunCheckpoint",
    "RunCheckpointContractError",
    "RunCheckpointHead",
    "RunCheckpointStatus",
    "RunCheckpointStop",
    "RunFeedbackSource",
    "RunStrategyKind",
    "RunStrategyState",
    "RunTerminationDecision",
]
