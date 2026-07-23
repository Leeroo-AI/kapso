"""Content-addressed evidence and summaries for immutable expert releases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    StrictContract,
)
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExpertReleaseContractError(ValueError):
    """An expert release evidence contract is incomplete or contradictory."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ExpertReleaseContractError(f"{name} must be a sha256 digest")


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertReleaseContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertReleaseMatrixSummary(StrictContract):
    """Public aggregate of the exact accepted release-matrix decision."""

    summary_id: str
    release_matrix_stage_result_id: str
    release_matrix_report_id: str
    promotion_decision_id: str
    mode: ExpertReleaseMatrixMode
    outcome: ExpertReleaseMatrixDecisionOutcome
    evaluation_cell_count: int
    provenance_count: int
    task_adapter_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-summary"
    IDENTITY_FIELD: ClassVar[str] = "summary_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "matrix summary stage result",
            ),
            (
                self.release_matrix_report_id,
                "expert-release-matrix-report",
                "matrix summary report",
            ),
            (
                self.promotion_decision_id,
                "expert-release-matrix-promotion-decision",
                "matrix summary decision",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED:
            raise ExpertReleaseContractError(
                "release matrix summary must represent an approved decision"
            )
        for value, name in (
            (self.evaluation_cell_count, "evaluation_cell_count"),
            (self.provenance_count, "provenance_count"),
            (self.task_adapter_count, "task_adapter_count"),
        ):
            if type(value) is not int or value <= 0:
                raise ExpertReleaseContractError(f"{name} must be positive")


@dataclass(frozen=True)
class ExpertReleaseEvidenceManifest(StrictContract):
    """Safe canonical record projection backing one scientific release identity."""

    evidence_manifest_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    validation_attempt_id: str
    approval_transition_id: str
    approval_state_id: str
    publication_eligibility_result_id: str
    release_matrix_summary_id: str
    record_ids: tuple[str, ...]
    record_checksums: Mapping[str, str]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-evidence-manifest"
    IDENTITY_FIELD: ClassVar[str] = "evidence_manifest_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "evidence candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "evidence candidate commit",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "evidence validation attempt",
            ),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "evidence approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "evidence approval state",
            ),
            (
                self.publication_eligibility_result_id,
                "expert-publication-eligibility-stage-result",
                "evidence publication result",
            ),
            (
                self.release_matrix_summary_id,
                "expert-release-matrix-summary",
                "evidence release matrix summary",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "evidence candidate tree")
        _require_sorted_content_ids(self.record_ids, "evidence record IDs")
        if not self.record_checksums:
            raise ExpertReleaseContractError(
                "evidence manifest must checksum its projected records"
            )
        for relative_path, digest in self.record_checksums.items():
            path = PurePosixPath(relative_path)
            if (
                not relative_path
                or path.is_absolute()
                or path == PurePosixPath(".")
                or ".." in path.parts
                or path.as_posix() != relative_path
            ):
                raise ExpertReleaseContractError(
                    "evidence record checksum path is invalid"
                )
            _require_digest(digest, "evidence record checksum")
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "evidence exact dependencies",
        )
        required = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.validation_attempt_id,
            self.approval_transition_id,
            self.approval_state_id,
            self.publication_eligibility_result_id,
            self.release_matrix_summary_id,
            *self.record_ids,
        }
        if set(self.exact_dependency_ids) != required:
            raise ExpertReleaseContractError(
                "evidence manifest dependency closure is not exact"
            )


__all__ = [
    "ExpertReleaseContractError",
    "ExpertReleaseEvidenceManifest",
    "ExpertReleaseMatrixSummary",
]
