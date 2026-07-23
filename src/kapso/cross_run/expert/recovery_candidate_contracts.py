"""Durable admission authority for clean-forward recovery candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    StrictContract,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.recovery_contracts import (
    ExpertCleanForwardRecoveryPlan,
)


class ExpertRecoveryCandidateContractError(ValueError):
    """Recovery candidate admission is incomplete or contradictory."""


@dataclass(frozen=True)
class ExpertRecoveryCandidateAdmission(StrictContract):
    """Exact recovery plan authorized for one immutable candidate package."""

    admission_id: str
    recovery_plan: ExpertCleanForwardRecoveryPlan
    candidate_id: str
    candidate_commit_record_id: str
    replay_basis_packet_id: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-recovery-candidate-admission"
    IDENTITY_FIELD: ClassVar[str] = "admission_id"

    def _validate(self) -> None:
        if type(self.recovery_plan) is not ExpertCleanForwardRecoveryPlan:
            raise ExpertRecoveryCandidateContractError(
                "recovery admission requires one exact plan"
            )
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "recovery candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "recovery candidate commit",
            ),
            (
                self.replay_basis_packet_id,
                "expert-trigger-evidence-packet",
                "recovery replay basis",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertRecoveryCandidateContractError(
                    f"{name} uses the wrong namespace"
                )
        if self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids))):
            raise ExpertRecoveryCandidateContractError(
                "recovery admission dependencies must be canonical"
            )
        for dependency_id in self.exact_dependency_ids:
            require_content_id(
                dependency_id,
                "recovery admission dependency",
            )
        expected = {
            self.recovery_plan.recovery_plan_id,
            *self.recovery_plan.exact_dependency_ids,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.replay_basis_packet_id,
        }
        if set(self.exact_dependency_ids) != expected:
            raise ExpertRecoveryCandidateContractError(
                "recovery admission dependency closure is not exact"
            )

    @property
    def control_dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.recovery_plan.recovery_plan_id,
                    *self.recovery_plan.control_dependency_ids,
                }
            )
        )


def validate_recovery_candidate_admission(
    *,
    admission: ExpertRecoveryCandidateAdmission,
    closure: ExpertCandidateClosure,
    commit_record: ExpertCandidateCommitRecord,
) -> None:
    """Join a recovery admission to the exact source, barrier, and candidate."""

    if (
        type(admission) is not ExpertRecoveryCandidateAdmission
        or type(closure) is not ExpertCandidateClosure
        or type(commit_record) is not ExpertCandidateCommitRecord
    ):
        raise ExpertRecoveryCandidateContractError(
            "recovery admission validation requires exact typed inputs"
        )
    plan = admission.recovery_plan
    manifest = closure.manifest
    context = closure.validation_context
    derivation = closure.derivation
    if plan.source_base_release_id is None:
        raise ExpertRecoveryCandidateContractError(
            "historical recovery admission cannot authorize canonical empty"
        )
    selected = plan.assessments[-1].manifest
    barrier = plan.assessments[0].manifest
    if (
        admission.candidate_id != manifest.candidate_id
        or admission.candidate_commit_record_id != commit_record.commit_record_id
        or commit_record.candidate_id != manifest.candidate_id
        or manifest.derivation_kind
        is not ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE
        or admission.replay_basis_packet_id
        != derivation.replay_basis_packet.evidence_packet_id
        or derivation.replay_basis_packet.source_base_release != barrier
        or derivation.replay_basis_packet.scope_contract != plan.scope_contract
        or context.scope_contract != plan.scope_contract
        or context.source_base_release != selected
        or manifest.source_base_release_id != plan.source_base_release_id
        or manifest.source_base_tree_hash != plan.source_base_tree_hash
        or manifest.candidate_tree_hash != plan.source_base_tree_hash
        or manifest.source_base_repository_map_ref
        != plan.source_base_repository_map_ref
        or manifest.module_contract_refs != plan.source_base_module_contract_refs
    ):
        raise ExpertRecoveryCandidateContractError(
            "recovery admission does not join its source, barrier, and candidate"
        )


__all__ = [
    "ExpertRecoveryCandidateAdmission",
    "ExpertRecoveryCandidateContractError",
    "validate_recovery_candidate_admission",
]
