"""Domain-neutral execution contracts for adapter-owned release-matrix cases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode


class TaskEvaluationContractError(ValueError):
    """A task-evaluation authority is structurally invalid or incomplete."""


TASK_EVALUATION_REQUEST_CONTRACT_VERSION = "kapso.task_evaluation_request.v1"


class TaskEvaluationLegKind(str, Enum):
    PARENT_CONTROL = "parent_control"
    CANDIDATE = "candidate"


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise TaskEvaluationContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise TaskEvaluationContractError(f"{name} is invalid")


def _require_sorted_ids(
    values: tuple[str, ...],
    namespace: str,
    name: str,
    *,
    required: bool = True,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise TaskEvaluationContractError(
            f"{name} must be sorted, unique, and complete"
        )
    for value in values:
        _require_namespaced_id(value, namespace, name)


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise TaskEvaluationContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class TaskEvaluationExpertLeg(StrictContract):
    leg_id: str
    kind: TaskEvaluationLegKind
    expert_artifact_id: str
    expert_source_receipt_id: str
    expert_tree_hash: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-leg"
    IDENTITY_FIELD: ClassVar[str] = "leg_id"

    def _validate(self) -> None:
        if self.kind is TaskEvaluationLegKind.PARENT_CONTROL:
            artifact_namespace = "expert-base-release"
            receipt_namespace = "expert-parent-tree-receipt"
        else:
            artifact_namespace = "expert-candidate"
            receipt_namespace = "expert-candidate-commit"
        _require_namespaced_id(
            self.expert_artifact_id,
            artifact_namespace,
            "task evaluation leg expert artifact",
        )
        _require_namespaced_id(
            self.expert_source_receipt_id,
            receipt_namespace,
            "task evaluation leg expert source receipt",
        )
        _require_digest(self.expert_tree_hash, "task evaluation leg expert tree")
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "task evaluation leg dependencies",
        )
        if set(self.exact_dependency_ids) != {
            self.expert_artifact_id,
            self.expert_source_receipt_id,
        }:
            raise TaskEvaluationContractError(
                "task evaluation leg dependency closure is not exact"
            )


@dataclass(frozen=True)
class TaskEvaluationComputeBinding(StrictContract):
    compute_binding_id: str
    execution_protocol_version: str
    execution_provider_id: str
    execution_provider_version: str
    execution_provider_settings_digest: str
    sandbox_policy_version: str
    leg_wall_time_limit_seconds: int
    termination_grace_seconds: int
    cpu_millicore_limit: int
    memory_byte_limit: int
    shared_memory_byte_limit: int
    process_limit: int
    open_file_limit: int
    writable_inode_limit: int
    writable_storage_byte_limit: int
    output_entry_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    accelerator_class_id: str | None
    accelerator_count: int
    leg_order: tuple[TaskEvaluationLegKind, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-compute-binding"
    IDENTITY_FIELD: ClassVar[str] = "compute_binding_id"

    def _validate(self) -> None:
        for value, name in (
            (self.execution_protocol_version, "execution protocol version"),
            (self.execution_provider_id, "execution provider ID"),
            (self.execution_provider_version, "execution provider version"),
            (self.sandbox_policy_version, "sandbox policy version"),
        ):
            require_identifier(value, f"task evaluation {name}")
        _require_digest(
            self.execution_provider_settings_digest,
            "task evaluation provider settings",
        )
        for value, name in (
            (self.leg_wall_time_limit_seconds, "leg wall time limit"),
            (self.termination_grace_seconds, "termination grace"),
            (self.cpu_millicore_limit, "CPU limit"),
            (self.memory_byte_limit, "memory limit"),
            (self.shared_memory_byte_limit, "shared memory limit"),
            (self.process_limit, "process limit"),
            (self.open_file_limit, "open file limit"),
            (self.writable_inode_limit, "writable inode limit"),
            (self.writable_storage_byte_limit, "writable storage limit"),
            (self.output_entry_limit, "output entry limit"),
            (self.output_byte_limit, "output byte limit"),
            (self.stdout_byte_limit, "stdout byte limit"),
            (self.stderr_byte_limit, "stderr byte limit"),
        ):
            if type(value) is not int or value <= 0:
                raise TaskEvaluationContractError(
                    f"task evaluation {name} must be a positive integer"
                )
        if (
            self.termination_grace_seconds > self.leg_wall_time_limit_seconds
            or self.shared_memory_byte_limit > self.memory_byte_limit
            or self.output_entry_limit >= self.writable_inode_limit
            or self.output_byte_limit > self.writable_storage_byte_limit
        ):
            raise TaskEvaluationContractError(
                "task evaluation compute limits are internally inconsistent"
            )
        if type(self.accelerator_count) is not int or self.accelerator_count < 0:
            raise TaskEvaluationContractError(
                "task evaluation accelerator count must be non-negative"
            )
        if (self.accelerator_class_id is None) != (self.accelerator_count == 0):
            raise TaskEvaluationContractError(
                "task evaluation accelerator class and count must be present together"
            )
        if self.accelerator_class_id is not None:
            require_identifier(
                self.accelerator_class_id,
                "task evaluation accelerator class",
            )
        if (
            not self.leg_order
            or len(self.leg_order) > len(TaskEvaluationLegKind)
            or len(self.leg_order) != len(set(self.leg_order))
            or TaskEvaluationLegKind.CANDIDATE not in self.leg_order
        ):
            raise TaskEvaluationContractError(
                "task evaluation leg order must contain candidate and no duplicates"
            )


@dataclass(frozen=True)
class TaskEvaluationCase(StrictContract):
    evaluation_case_id: str
    adapter_authority_id: str
    provenance_binding_id: str
    release_matrix_case_id: str
    task_context_binding_id: str
    independence_group_id: str
    evaluation_cell_ids: tuple[str, ...]
    evaluation_fingerprint_ids: tuple[str, ...]
    starting_artifact_ids: tuple[str, ...]
    compute_binding: TaskEvaluationComputeBinding
    legs: tuple[TaskEvaluationExpertLeg, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-case"
    IDENTITY_FIELD: ClassVar[str] = "evaluation_case_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.adapter_authority_id,
                "expert-release-matrix-adapter-authority",
                "task evaluation adapter authority",
            ),
            (
                self.provenance_binding_id,
                "expert-release-matrix-provenance-binding",
                "task evaluation provenance binding",
            ),
            (
                self.release_matrix_case_id,
                "task-adapter-release-matrix-case",
                "task evaluation signed case",
            ),
            (
                self.task_context_binding_id,
                "task-context-binding",
                "task evaluation context binding",
            ),
            (
                self.independence_group_id,
                "task-adapter-release-matrix-independence-group",
                "task evaluation independence group",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_sorted_ids(
            self.evaluation_cell_ids,
            "expert-release-matrix-evaluation-cell",
            "task evaluation cells",
        )
        _require_sorted_ids(
            self.evaluation_fingerprint_ids,
            "evaluation-fingerprint",
            "task evaluation fingerprints",
        )
        if len(self.evaluation_cell_ids) != len(self.evaluation_fingerprint_ids):
            raise TaskEvaluationContractError(
                "task evaluation cells must cover fingerprints exactly once"
            )
        _require_sorted_ids(
            self.starting_artifact_ids,
            "task-adapter-release-matrix-starting-artifact",
            "task evaluation starting artifacts",
            required=False,
        )
        leg_ids = tuple(leg.leg_id for leg in self.legs)
        leg_kinds = tuple(leg.kind for leg in self.legs)
        if (
            not leg_ids
            or leg_ids != tuple(sorted(set(leg_ids)))
            or set(leg_kinds) != set(self.compute_binding.leg_order)
            or len(leg_kinds) != len(set(leg_kinds))
        ):
            raise TaskEvaluationContractError(
                "task evaluation legs differ from their canonical compute schedule"
            )
        expected_dependencies = {
            self.adapter_authority_id,
            self.provenance_binding_id,
            self.release_matrix_case_id,
            self.task_context_binding_id,
            self.independence_group_id,
            *self.evaluation_cell_ids,
            *self.evaluation_fingerprint_ids,
            *self.starting_artifact_ids,
            self.compute_binding.compute_binding_id,
            *leg_ids,
            *(
                dependency_id
                for leg in self.legs
                for dependency_id in leg.exact_dependency_ids
            ),
        }
        if self.exact_dependency_ids != tuple(sorted(expected_dependencies)):
            raise TaskEvaluationContractError(
                "task evaluation case dependency closure is not exact"
            )

    @property
    def canonical_key(self) -> tuple[str, str]:
        return self.adapter_authority_id, self.provenance_binding_id


@dataclass(frozen=True)
class TaskEvaluationRequest(StrictContract):
    request_id: str
    request_contract_version: str
    plan_reservation_operation_id: str
    evaluation_plan_id: str
    mode: ExpertReleaseMatrixMode
    authorization_transition_id: str
    authorization_state_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    scope_id: str
    parent_release_id: str | None
    parent_tree_hash: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    release_matrix_evaluator_id: str
    release_matrix_evaluator_role: str
    release_matrix_evaluator_version: str
    plan_dependency_ids: tuple[str, ...]
    cases: tuple[TaskEvaluationCase, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-request"
    IDENTITY_FIELD: ClassVar[str] = "request_id"

    def _validate(self) -> None:
        if self.request_contract_version != TASK_EVALUATION_REQUEST_CONTRACT_VERSION:
            raise TaskEvaluationContractError(
                "task evaluation request contract version is unsupported"
            )
        require_identifier(
            self.release_matrix_evaluator_id,
            "task evaluation release matrix evaluator ID",
        )
        require_identifier(
            self.release_matrix_evaluator_role,
            "task evaluation release matrix evaluator role",
        )
        require_identifier(
            self.release_matrix_evaluator_version,
            "task evaluation release matrix evaluator version",
        )
        require_identifier(self.scope_id, "task evaluation scope ID")
        for value, namespace, name in (
            (
                self.plan_reservation_operation_id,
                "expert-validation-operation",
                "task evaluation plan reservation operation",
            ),
            (
                self.evaluation_plan_id,
                "expert-release-matrix-evaluation-plan",
                "task evaluation plan",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "task evaluation authorization transition",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "task evaluation authorization state",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "task evaluation validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "task evaluation candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "task evaluation candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "task evaluation scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "task evaluation validation policy",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "task evaluation candidate tree")
        _require_digest(
            self.configuration_fingerprint,
            "task evaluation configuration fingerprint",
        )
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            if self.parent_release_id is not None or self.parent_tree_hash is not None:
                raise TaskEvaluationContractError(
                    "bootstrap task evaluation cannot name a parent"
                )
            expected_leg_kinds = {TaskEvaluationLegKind.CANDIDATE}
        else:
            if self.parent_release_id is None or self.parent_tree_hash is None:
                raise TaskEvaluationContractError(
                    "parent task evaluation requires a parent"
                )
            _require_namespaced_id(
                self.parent_release_id,
                "expert-base-release",
                "task evaluation parent release",
            )
            _require_digest(self.parent_tree_hash, "task evaluation parent tree")
            expected_leg_kinds = set(TaskEvaluationLegKind)
        case_keys = tuple(case.canonical_key for case in self.cases)
        case_ids = tuple(case.evaluation_case_id for case in self.cases)
        provenance_ids = tuple(case.provenance_binding_id for case in self.cases)
        if (
            not case_keys
            or case_keys != tuple(sorted(set(case_keys)))
            or len(case_ids) != len(set(case_ids))
            or any(
                {leg.kind for leg in case.legs} != expected_leg_kinds
                for case in self.cases
            )
        ):
            raise TaskEvaluationContractError(
                "task evaluation cases or mode-specific legs are noncanonical"
            )
        if len(provenance_ids) != len(set(provenance_ids)):
            raise TaskEvaluationContractError(
                "task evaluation cases must name unique provenances"
            )
        parent_source_receipt_ids = set()
        for case in self.cases:
            for leg in case.legs:
                if leg.kind is TaskEvaluationLegKind.CANDIDATE:
                    matches_expert_authority = (
                        leg.expert_artifact_id == self.candidate_id
                        and leg.expert_source_receipt_id
                        == self.candidate_commit_record_id
                        and leg.expert_tree_hash == self.candidate_tree_hash
                    )
                else:
                    matches_expert_authority = (
                        leg.expert_artifact_id == self.parent_release_id
                        and leg.expert_tree_hash == self.parent_tree_hash
                    )
                    parent_source_receipt_ids.add(leg.expert_source_receipt_id)
                if not matches_expert_authority:
                    raise TaskEvaluationContractError(
                        "task evaluation leg differs from request expert authority"
                    )
        if len(parent_source_receipt_ids) > 1:
            raise TaskEvaluationContractError(
                "task evaluation cases disagree on parent source authority"
            )
        cell_ids = tuple(
            cell_id for case in self.cases for cell_id in case.evaluation_cell_ids
        )
        if len(cell_ids) != len(set(cell_ids)):
            raise TaskEvaluationContractError(
                "task evaluation cases reuse a planned cell"
            )
        _require_sorted_content_ids(
            self.plan_dependency_ids,
            "task evaluation plan dependencies",
        )
        expected_dependencies = {
            self.plan_reservation_operation_id,
            self.evaluation_plan_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            *self.plan_dependency_ids,
            *case_ids,
            *(
                dependency_id
                for case in self.cases
                for dependency_id in case.exact_dependency_ids
            ),
        }
        if self.parent_release_id is not None:
            expected_dependencies.add(self.parent_release_id)
        if self.exact_dependency_ids != tuple(sorted(expected_dependencies)):
            raise TaskEvaluationContractError(
                "task evaluation request dependency closure is not exact"
            )


@dataclass(frozen=True)
class TaskEvaluationReservation(StrictContract):
    reservation_id: str
    request_id: str
    plan_reservation_operation_id: str
    evaluation_plan_id: str
    mode: ExpertReleaseMatrixMode
    authorization_transition_id: str
    authorization_state_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    scope_id: str
    current_release_observation_id: str
    observed_current_release_id: str | None
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-reservation"
    IDENTITY_FIELD: ClassVar[str] = "reservation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.request_id, "task-evaluation-request", "task evaluation request"),
            (
                self.plan_reservation_operation_id,
                "expert-validation-operation",
                "task evaluation plan reservation operation",
            ),
            (
                self.evaluation_plan_id,
                "expert-release-matrix-evaluation-plan",
                "task evaluation plan",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "task evaluation authorization transition",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "task evaluation authorization state",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "task evaluation validation attempt",
            ),
            (self.candidate_id, "expert-candidate", "task evaluation candidate"),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "task evaluation scope contract",
            ),
            (
                self.current_release_observation_id,
                "task-evaluation-current-release-observation",
                "task evaluation current release observation",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(
            self.candidate_tree_hash,
            "task evaluation reservation candidate tree",
        )
        require_identifier(self.scope_id, "task evaluation reservation scope ID")
        if self.observed_current_release_id is not None:
            _require_namespaced_id(
                self.observed_current_release_id,
                "expert-base-release",
                "task evaluation observed current release",
            )
        if (self.mode is ExpertReleaseMatrixMode.BOOTSTRAP) != (
            self.observed_current_release_id is None
        ):
            raise TaskEvaluationContractError(
                "task evaluation reservation current release differs from its mode"
            )
        expected_dependencies = {
            self.request_id,
            self.plan_reservation_operation_id,
            self.evaluation_plan_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.scope_contract_id,
            self.current_release_observation_id,
        }
        if self.observed_current_release_id is not None:
            expected_dependencies.add(self.observed_current_release_id)
        if self.exact_dependency_ids != tuple(sorted(expected_dependencies)):
            raise TaskEvaluationContractError(
                "task evaluation reservation dependency closure is not exact"
            )
