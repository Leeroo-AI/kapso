"""Accepted release-matrix stage evidence bound to validation authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixReport


class ExpertReleaseMatrixStageContractError(ValueError):
    """A release-matrix stage result is structurally or relationally invalid."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseMatrixStageContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ExpertReleaseMatrixStageContractError(f"{name} is invalid")


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertReleaseMatrixStageContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertReleaseMatrixStageResultRecord(StrictContract):
    """One factual matrix report sealed to the exact validation-stage authority."""

    stage_result_record_id: str
    validation_attempt_id: str
    authorization_transition_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    source_base_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    plan_reservation_operation_id: str
    task_evaluation_reservation_id: str
    release_matrix_report: ExpertReleaseMatrixReport
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-stage-result"
    IDENTITY_FIELD: ClassVar[str] = "stage_result_record_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release matrix stage validation attempt",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "release matrix stage authorization transition",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "release matrix stage authorization state",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "release matrix stage candidate",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "release matrix stage scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "release matrix stage validation policy",
            ),
            (
                self.plan_reservation_operation_id,
                "expert-validation-operation",
                "release matrix stage plan reservation operation",
            ),
            (
                self.task_evaluation_reservation_id,
                "task-evaluation-reservation",
                "release matrix stage task reservation",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.source_base_release_id is not None:
            _require_namespaced_id(
                self.source_base_release_id,
                "expert-base-release",
                "release matrix stage source-base release",
            )
        _require_digest(
            self.candidate_tree_hash,
            "release matrix stage candidate tree",
        )
        _require_digest(
            self.configuration_fingerprint,
            "release matrix stage configuration fingerprint",
        )
        report = self.release_matrix_report
        if (
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_tree_hash,
            self.scope_contract_id,
            self.source_base_release_id,
            self.validation_policy_id,
            self.configuration_fingerprint,
            self.plan_reservation_operation_id,
        ) != (
            report.validation_attempt_id,
            report.candidate_id,
            report.candidate_tree_hash,
            report.scope_contract_id,
            report.source_base_release_id,
            report.validation_policy_id,
            report.configuration_fingerprint,
            report.plan_reservation_operation_id,
        ):
            raise ExpertReleaseMatrixStageContractError(
                "release matrix stage subjects differ from the embedded report"
            )
        task_evidence = report.task_execution_evidence
        if task_evidence is None:
            raise ExpertReleaseMatrixStageContractError(
                "release matrix stage report requires task execution evidence"
            )
        if task_evidence.reservation_id != self.task_evaluation_reservation_id:
            raise ExpertReleaseMatrixStageContractError(
                "release matrix stage task reservation differs from the embedded report"
            )
        required_task_authority = {
            self.authorization_transition_id,
            self.authorization_state_id,
            self.plan_reservation_operation_id,
            report.evaluation_plan.evaluation_plan_id,
        }
        if not required_task_authority.issubset(
            task_evidence.reservation_dependency_ids
        ) or not required_task_authority.issubset(task_evidence.request_dependency_ids):
            raise ExpertReleaseMatrixStageContractError(
                "release matrix task evidence omits stage authority"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "release matrix stage exact dependencies",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.scope_contract_id,
            self.validation_policy_id,
            self.plan_reservation_operation_id,
            self.task_evaluation_reservation_id,
            report.release_matrix_report_id,
            *report.exact_dependency_ids,
        }
        if self.source_base_release_id is not None:
            expected_dependencies.add(self.source_base_release_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertReleaseMatrixStageContractError(
                "release matrix stage dependency closure is not exact"
            )
