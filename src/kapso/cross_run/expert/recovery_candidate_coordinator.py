"""Production-owned construction and admission of recovery candidates."""

from __future__ import annotations

from kapso.cross_run.contracts import (
    ExpertCandidateCommitRecord,
    ExpertScopeContract,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.composition_base_provider import (
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.recovery_base import (
    ExpertRecoveryBaseSelection,
    ExpertRecoveryBaseSelector,
)
from kapso.cross_run.expert.recovery_candidate import (
    project_historical_recovery_candidate,
)
from kapso.cross_run.expert.recovery_candidate_authority import (
    _seal_expert_recovery_candidate_authority,
)
from kapso.cross_run.expert.recovery_candidate_contracts import (
    ExpertRecoveryCandidateAdmission,
    validate_recovery_candidate_admission,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    StoredExpertCandidate,
)
from kapso.cross_run.expert.triggers import ExpertTriggerEvidencePacket


class ExpertRecoveryCandidateCoordinatorError(ValueError):
    """Recovery candidate authority cannot produce one admissible closure."""


class ExpertCleanForwardRecoveryCandidateCoordinator:
    """Sole runtime path from authenticated history to an admitted candidate."""

    __slots__ = (
        "_authority",
        "base_provider",
        "candidate_store",
        "selector",
    )

    def __init__(
        self,
        *,
        selector: ExpertRecoveryBaseSelector,
        base_provider: GitHubExpertCompositionBaseProvider,
        candidate_store: ExpertCandidateStore,
    ) -> None:
        if (
            type(selector) is not ExpertRecoveryBaseSelector
            or type(base_provider) is not GitHubExpertCompositionBaseProvider
            or type(candidate_store) is not ExpertCandidateStore
        ):
            raise ExpertRecoveryCandidateCoordinatorError(
                "recovery candidate coordinator requires exact production components"
            )
        self.selector = selector
        self.base_provider = base_provider
        self.candidate_store = candidate_store
        self._authority = _seal_expert_recovery_candidate_authority(
            coordinator=self,
            candidate_store=candidate_store,
        )
        candidate_store._bind_recovery_candidate_authority(self._authority)

    def restore_historical(
        self,
        *,
        scope_contract: ExpertScopeContract,
        replay_basis_packet: ExpertTriggerEvidencePacket,
    ) -> StoredExpertCandidate:
        """Select, restore, recheck, and atomically admit one clean checkpoint."""

        if (
            type(scope_contract) is not ExpertScopeContract
            or type(replay_basis_packet) is not ExpertTriggerEvidencePacket
        ):
            raise ExpertRecoveryCandidateCoordinatorError(
                "historical recovery requires exact scope and replay basis"
            )
        selection = self.selector.select(scope_contract)
        activation = selection.selected_activation
        if activation is None:
            raise ExpertRecoveryCandidateCoordinatorError(
                "historical recovery cannot use canonical empty selection"
            )
        plan = selection.plan
        if (
            replay_basis_packet.scope_contract != scope_contract
            or replay_basis_packet.source_base_release != plan.assessments[0].manifest
        ):
            raise ExpertRecoveryCandidateCoordinatorError(
                "recovery replay basis does not describe the blocked CURRENT"
            )
        base = self.base_provider.resolve_historical(
            scope_contract,
            activation,
        )
        if (
            base.release_manifest.release_id != plan.source_base_release_id
            or base.reference.source_tree_hash != plan.source_base_tree_hash
        ):
            raise ExpertRecoveryCandidateCoordinatorError(
                "materialized recovery base differs from selected source"
            )
        closure = project_historical_recovery_candidate(
            base=base,
            replay_basis_packet=replay_basis_packet,
            sanitizer=self.candidate_store.validator.sanitizer,
        )
        return self.candidate_store._commit_recovery_candidate(
            authority=self._authority,
            selection=selection,
            closure=closure,
        )

    def _finalize_recovery_admission_under_store_lock(
        self,
        *,
        selection: ExpertRecoveryBaseSelection,
        closure: ExpertCandidateClosure,
        commit_record: ExpertCandidateCommitRecord,
    ) -> ExpertRecoveryCandidateAdmission:
        fresh = self.selector.require_fresh(selection)
        plan = fresh.plan
        dependencies = tuple(
            sorted(
                {
                    plan.recovery_plan_id,
                    *plan.exact_dependency_ids,
                    closure.manifest.candidate_id,
                    commit_record.commit_record_id,
                    closure.derivation.replay_basis_packet.evidence_packet_id,
                }
            )
        )
        admission = ExpertRecoveryCandidateAdmission.mint(
            recovery_plan=plan,
            candidate_id=closure.manifest.candidate_id,
            candidate_commit_record_id=commit_record.commit_record_id,
            replay_basis_packet_id=(
                closure.derivation.replay_basis_packet.evidence_packet_id
            ),
            exact_dependency_ids=dependencies,
        )
        validate_recovery_candidate_admission(
            admission=admission,
            closure=closure,
            commit_record=commit_record,
        )
        return admission


__all__ = [
    "ExpertCleanForwardRecoveryCandidateCoordinator",
    "ExpertRecoveryCandidateCoordinatorError",
]
