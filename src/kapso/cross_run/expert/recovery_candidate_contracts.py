"""Durable admission authority for clean-forward recovery candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    ExpertCandidateOperationKind,
    StrictContract,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertAgentProposalDerivation,
    ExpertDeterministicRecoveryRestoreDerivation,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.recovery_candidate import (
    project_canonical_empty_recovery_packet,
)
from kapso.cross_run.expert.recovery_contracts import (
    ExpertCleanForwardRecoveryPlan,
)
from kapso.cross_run.expert.triggers import ExpertTriggerEvidencePacket


class ExpertRecoveryCandidateContractError(ValueError):
    """Recovery candidate admission is incomplete or contradictory."""


@dataclass(frozen=True)
class ExpertRecoveryCandidateAdmission(StrictContract):
    """Exact recovery plan authorized for one immutable candidate package."""

    admission_id: str
    recovery_plan: ExpertCleanForwardRecoveryPlan
    candidate_id: str
    candidate_commit_record_id: str
    barrier_replay_basis: ExpertTriggerEvidencePacket
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-recovery-candidate-admission"
    IDENTITY_FIELD: ClassVar[str] = "admission_id"

    def _validate(self) -> None:
        if type(self.recovery_plan) is not ExpertCleanForwardRecoveryPlan:
            raise ExpertRecoveryCandidateContractError(
                "recovery admission requires one exact plan"
            )
        if type(self.barrier_replay_basis) is not ExpertTriggerEvidencePacket:
            raise ExpertRecoveryCandidateContractError(
                "recovery admission requires one exact barrier replay basis"
            )
        barrier = self.recovery_plan.assessments[0].manifest
        if (
            self.barrier_replay_basis.scope_contract
            != self.recovery_plan.scope_contract
            or self.barrier_replay_basis.source_base_release != barrier
        ):
            raise ExpertRecoveryCandidateContractError(
                "recovery admission barrier replay basis differs from its plan"
            )
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "recovery candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "recovery candidate commit",
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
            self.barrier_replay_basis.evidence_packet_id,
        }
        if set(self.exact_dependency_ids) != expected:
            raise ExpertRecoveryCandidateContractError(
                "recovery admission dependency closure is not exact"
            )
        dependency_ids = set(self.dependency_ids)
        control_dependency_ids = set(self.control_dependency_ids)
        scientific_dependency_ids = set(self.scientific_dependency_ids)
        if (
            control_dependency_ids & scientific_dependency_ids
            or control_dependency_ids | scientific_dependency_ids != dependency_ids
        ):
            raise ExpertRecoveryCandidateContractError(
                "recovery admission dependency partition is not exact"
            )
        required_scientific_dependencies = {
            self.candidate_id,
            self.candidate_commit_record_id,
        }
        if self.recovery_plan.source_base_release_id is None:
            if (
                self.barrier_replay_basis.evidence_packet_id
                not in control_dependency_ids
            ):
                raise ExpertRecoveryCandidateContractError(
                    "empty recovery barrier replay basis must be control authority"
                )
        else:
            selected_assessment = self.recovery_plan.assessments[-1]
            selected_source = selected_assessment.manifest
            required_scientific_dependencies.update(
                {
                    selected_assessment.assessment_id,
                    *selected_assessment.exact_dependency_ids,
                    selected_source.release_id,
                    selected_source.candidate_id,
                    selected_source.candidate_commit_record_id,
                    selected_source.repository_map_ref,
                    self.barrier_replay_basis.evidence_packet_id,
                    *selected_source.module_contract_refs,
                    *selected_source.consumed_dependency_ids,
                }
            )
        if not required_scientific_dependencies.issubset(scientific_dependency_ids):
            raise ExpertRecoveryCandidateContractError(
                "recovery admission classifies scientific evidence as control"
            )

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        """Return the exact dependency universe, including this admission."""

        return tuple(sorted({self.admission_id, *self.exact_dependency_ids}))

    @property
    def control_dependency_ids(self) -> tuple[str, ...]:
        dependencies = {
            self.admission_id,
            self.recovery_plan.recovery_plan_id,
            *self.recovery_plan.control_dependency_ids,
        }
        if self.recovery_plan.source_base_release_id is not None:
            selected_assessment = self.recovery_plan.assessments[-1]
            selected_source = selected_assessment.manifest
            dependencies.difference_update(
                {
                    selected_assessment.assessment_id,
                    *selected_assessment.exact_dependency_ids,
                    selected_source.release_id,
                    selected_source.candidate_id,
                    selected_source.candidate_commit_record_id,
                    selected_source.repository_map_ref,
                    *selected_source.module_contract_refs,
                    *selected_source.consumed_dependency_ids,
                }
            )
        if self.recovery_plan.source_base_release_id is None:
            dependencies.add(self.barrier_replay_basis.evidence_packet_id)
        return tuple(sorted(dependencies))

    @property
    def scientific_dependency_ids(self) -> tuple[str, ...]:
        """Return exact scientific inputs without temporal recovery authority."""

        control_dependency_ids = set(self.control_dependency_ids)
        return tuple(
            dependency_id
            for dependency_id in self.dependency_ids
            if dependency_id not in control_dependency_ids
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
    barrier = plan.assessments[0].manifest
    barrier_basis = admission.barrier_replay_basis
    if (
        admission.candidate_id != manifest.candidate_id
        or admission.candidate_commit_record_id != commit_record.commit_record_id
        or commit_record.candidate_id != manifest.candidate_id
        or barrier_basis.source_base_release != barrier
        or barrier_basis.scope_contract != plan.scope_contract
        or context.scope_contract != plan.scope_contract
    ):
        raise ExpertRecoveryCandidateContractError(
            "recovery admission does not join its source, barrier, and candidate"
        )
    if plan.source_base_release_id is not None:
        selected = plan.assessments[-1].manifest
        if (
            type(derivation) is not ExpertDeterministicRecoveryRestoreDerivation
            or manifest.derivation_kind
            is not ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE
            or derivation.replay_basis_packet != barrier_basis
            or context.source_base_release != selected
            or manifest.source_base_release_id != plan.source_base_release_id
            or manifest.source_base_tree_hash != plan.source_base_tree_hash
            or manifest.candidate_tree_hash != plan.source_base_tree_hash
            or manifest.source_base_repository_map_ref
            != plan.source_base_repository_map_ref
            or manifest.module_contract_refs != plan.source_base_module_contract_refs
        ):
            raise ExpertRecoveryCandidateContractError(
                "historical recovery admission differs from its selected source"
            )
        return
    empty_packet = project_canonical_empty_recovery_packet(barrier_basis)
    if (
        type(derivation) is not ExpertAgentProposalDerivation
        or derivation.operation.operation_kind
        is not ExpertCandidateOperationKind.RECOVERY_BOOTSTRAP
        or manifest.derivation_kind
        is not ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP
        or derivation.trigger_packet != empty_packet
        or derivation.trigger_packet.recovery_barrier_basis_packet_id
        != barrier_basis.evidence_packet_id
        or context.source_base_release is not None
        or context.source_base_scope_contract is not None
        or context.source_base_tree_receipt is not None
        or context.source_base_repository_map is not None
        or context.source_base_module_contracts
        or manifest.source_base_release_id is not None
        or manifest.source_base_repository_map_ref is not None
        or manifest.source_base_tree_hash != EMPTY_EXPERT_TREE_DIGEST
    ):
        raise ExpertRecoveryCandidateContractError(
            "empty recovery admission differs from canonical agent bootstrap"
        )


__all__ = [
    "ExpertRecoveryCandidateAdmission",
    "ExpertRecoveryCandidateContractError",
    "validate_recovery_candidate_admission",
]
