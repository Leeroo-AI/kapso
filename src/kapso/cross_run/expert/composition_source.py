"""Trusted runtime resolution of terminally approved composition sources."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import MappingProxyType
from typing import Iterator

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertCandidateDerivationKind,
    ExpertCandidateValidationState,
    ExpertPromotionState,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionSourceReference,
)
from kapso.cross_run.expert.composition import (
    ExpertCompositionReductionSource,
)
from kapso.cross_run.expert.composition_admission_authority import (
    ExpertCompositionApprovalLease,
    _seal_expert_composition_approval_lease,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.validation import ExpertCandidateReader
from kapso.cross_run.expert.validation_snapshots import (
    ExpertValidationSnapshot,
    ExpertValidationTransition,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore

_APPROVED_EXPERT_COMPOSITION_SOURCE_SEAL = object()


class ExpertCompositionSourceError(ValueError):
    """A composition source lacks complete, current approval authority."""


def _immutable_stored_candidate_view(
    stored_candidate: StoredExpertCandidate,
) -> StoredExpertCandidate:
    closure = stored_candidate.closure
    return replace(
        stored_candidate,
        closure=replace(
            closure,
            candidate_contents=MappingProxyType(dict(closure.candidate_contents)),
            derivation=replace(
                closure.derivation,
                operation_artifacts=MappingProxyType(
                    dict(closure.derivation.operation_artifacts)
                ),
            ),
        ),
    )


def project_expert_composition_source_reference(
    stored_candidate: StoredExpertCandidate,
) -> ExpertCompositionSourceReference:
    """Project stable scientific identity from one reopened candidate package."""

    if type(stored_candidate) is not StoredExpertCandidate:
        raise ExpertCompositionSourceError(
            "composition source projection requires one stored candidate"
        )
    manifest = stored_candidate.closure.manifest
    if manifest.derivation_kind is not ExpertCandidateDerivationKind.AGENT_PROPOSAL:
        raise ExpertCompositionSourceError(
            "composition sources must be direct agent proposals"
        )
    if (
        manifest.source_base_release_id is None
        or manifest.source_base_repository_map_ref is None
    ):
        raise ExpertCompositionSourceError(
            "bootstrap candidate cannot be a composition source"
        )
    module_contract_ids = tuple(
        sorted(
            module.module_contract_id
            for module in stored_candidate.closure.module_contracts
        )
    )
    patch = stored_candidate.closure.patch
    candidate_tree = stored_candidate.closure.candidate_tree
    repository_map = stored_candidate.closure.repository_map
    if (
        stored_candidate.commit_record.candidate_id != manifest.candidate_id
        or manifest.candidate_tree_ref != candidate_tree.source_tree_manifest_id
        or manifest.candidate_tree_hash != candidate_tree.tree_hash
        or manifest.patch_ref != patch.patch_id
        or manifest.patch_digest != tree_or_blob_digest(patch.to_json_bytes())
        or patch.source_base_tree_hash != manifest.source_base_tree_hash
        or patch.candidate_tree_hash != manifest.candidate_tree_hash
        or manifest.proposed_repository_map_ref != repository_map.repository_map_id
        or manifest.module_contract_refs != module_contract_ids
    ):
        raise ExpertCompositionSourceError(
            "composition source projection has inconsistent candidate closure"
        )
    stable_authority_ids = tuple(
        sorted(
            {
                manifest.candidate_id,
                stored_candidate.commit_record.commit_record_id,
                manifest.scope_contract_id,
                manifest.derivation_ref,
                manifest.validation_context_ref,
                manifest.source_base_release_id,
                manifest.source_base_repository_map_ref,
                patch.patch_id,
                repository_map.repository_map_id,
                *module_contract_ids,
            }
        )
    )
    return ExpertCompositionSourceReference.mint(
        candidate_id=manifest.candidate_id,
        candidate_commit_record_id=(stored_candidate.commit_record.commit_record_id),
        scope_contract_id=manifest.scope_contract_id,
        change_kind=manifest.change_kind,
        derivation_kind=manifest.derivation_kind,
        derivation_ref=manifest.derivation_ref,
        validation_context_ref=manifest.validation_context_ref,
        origin_principal_ids=stored_candidate.closure.origin_principal_ids,
        source_base_release_id=manifest.source_base_release_id,
        source_base_repository_map_id=manifest.source_base_repository_map_ref,
        source_base_tree_hash=manifest.source_base_tree_hash,
        candidate_tree_hash=manifest.candidate_tree_hash,
        patch_id=patch.patch_id,
        patch_digest=manifest.patch_digest,
        proposed_repository_map_id=repository_map.repository_map_id,
        module_contract_ids=module_contract_ids,
        candidate_configuration_fingerprint=manifest.configuration_fingerprint,
        stable_authority_ids=stable_authority_ids,
    )


class ApprovedExpertCompositionSource:
    """Immutable process-local proof of one current terminal approval."""

    __slots__ = (
        "_approval_snapshot",
        "_owner_process_id",
        "_publication_eligibility_result",
        "_resolver",
        "_security_subject_ids",
        "_source_reference",
        "_stored_candidate",
    )

    def __init__(
        self,
        seal: object,
        resolver: ExpertCompositionSourceResolver,
        *,
        stored_candidate: StoredExpertCandidate,
        approval_snapshot: ExpertValidationSnapshot,
        publication_eligibility_result: ExpertPublicationEligibilityStageResultRecord,
        source_reference: ExpertCompositionSourceReference,
    ) -> None:
        if seal is not _APPROVED_EXPERT_COMPOSITION_SOURCE_SEAL:
            raise ExpertCompositionSourceError(
                "approved composition source capability is not resolver sealed"
            )
        fence = publication_eligibility_result.publication_authority_fence
        if fence is None:
            raise ExpertCompositionSourceError(
                "approved composition source lacks publication authority"
            )
        security_subject_ids = tuple(
            sorted(
                {
                    source_reference.source_reference_id,
                    *source_reference.stable_authority_ids,
                    approval_snapshot.transition.transition_id,
                    approval_snapshot.state.validation_state_id,
                    publication_eligibility_result.stage_result_record_id,
                    *publication_eligibility_result.exact_dependency_ids,
                    fence.fence_id,
                    *fence.security_subject_ids,
                    *fence.exact_dependency_ids,
                }
            )
        )
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(
            self,
            "_stored_candidate",
            _immutable_stored_candidate_view(stored_candidate),
        )
        object.__setattr__(self, "_approval_snapshot", approval_snapshot)
        object.__setattr__(
            self,
            "_publication_eligibility_result",
            publication_eligibility_result,
        )
        object.__setattr__(self, "_source_reference", source_reference)
        object.__setattr__(self, "_security_subject_ids", security_subject_ids)

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionSourceError(
            "approved composition source capability is immutable"
        )

    def __reduce__(self):
        raise ExpertCompositionSourceError(
            "approved composition source capability cannot be serialized"
        )

    def __reduce_ex__(self, protocol):
        raise ExpertCompositionSourceError(
            "approved composition source capability cannot be serialized"
        )

    @property
    def stored_candidate(self) -> StoredExpertCandidate:
        self._require_owner_process()
        return self._stored_candidate

    @property
    def approval_snapshot(self) -> ExpertValidationSnapshot:
        self._require_owner_process()
        return self._approval_snapshot

    @property
    def publication_eligibility_result(
        self,
    ) -> ExpertPublicationEligibilityStageResultRecord:
        self._require_owner_process()
        return self._publication_eligibility_result

    @property
    def source_reference(self) -> ExpertCompositionSourceReference:
        self._require_owner_process()
        return self._source_reference

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        self._require_owner_process()
        return self._security_subject_ids

    @property
    def reduction_source(self) -> ExpertCompositionReductionSource:
        self._require_owner_process()
        closure = self._stored_candidate.closure
        return ExpertCompositionReductionSource(
            source_reference=self._source_reference,
            validation_context=closure.validation_context,
            patch=closure.patch,
            candidate_tree=closure.candidate_tree,
            repository_map=closure.repository_map,
            module_contracts=closure.module_contracts,
            candidate_contents=closure.candidate_contents,
        )

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise ExpertCompositionSourceError(
                "approved composition source capability is foreign"
            )

    def _require_bound(self, resolver: ExpertCompositionSourceResolver) -> None:
        self._require_owner_process()
        if self._resolver is not resolver:
            raise ExpertCompositionSourceError(
                "approved composition source capability is foreign"
            )


class ExpertCompositionSourceResolver:
    """Resolve serialized candidate identity into live approved-source authority."""

    __slots__ = ("_candidate_store", "_validation_store")

    def __init__(self, validation_store: ExpertValidationStore) -> None:
        if type(validation_store) is not ExpertValidationStore:
            raise ExpertCompositionSourceError(
                "composition source resolver requires an expert validation store"
            )
        object.__setattr__(self, "_validation_store", validation_store)
        object.__setattr__(
            self,
            "_candidate_store",
            validation_store.reducer.candidate_store,
        )

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionSourceError(
            "composition source resolver authority is immutable"
        )

    @property
    def validation_store(self) -> ExpertValidationStore:
        return self._validation_store

    @property
    def candidate_store(self) -> ExpertCandidateReader:
        return self._candidate_store

    def resolve(self, candidate_id: str) -> ApprovedExpertCompositionSource:
        require_content_id(candidate_id, "composition source candidate")
        if candidate_id.split(":sha256:", 1)[0] != "expert-candidate":
            raise ExpertCompositionSourceError(
                "composition source candidate uses the wrong namespace"
            )
        approval_snapshot = self.validation_store.snapshot(candidate_id)
        publication_result = self._require_approved_snapshot(approval_snapshot)
        stored_candidate = self.candidate_store.read(candidate_id)
        source_reference = project_expert_composition_source_reference(stored_candidate)
        self._require_candidate_approval_join(
            stored_candidate=stored_candidate,
            approval_snapshot=approval_snapshot,
            publication_result=publication_result,
            source_reference=source_reference,
        )
        reopened_snapshot = self.validation_store.snapshot(candidate_id)
        if reopened_snapshot != approval_snapshot:
            raise ExpertCompositionSourceError(
                "composition source validation head changed during resolution"
            )
        return ApprovedExpertCompositionSource(
            _APPROVED_EXPERT_COMPOSITION_SOURCE_SEAL,
            self,
            stored_candidate=stored_candidate,
            approval_snapshot=approval_snapshot,
            publication_eligibility_result=publication_result,
            source_reference=source_reference,
        )

    def require_current(
        self,
        capability: ApprovedExpertCompositionSource,
    ) -> None:
        if type(capability) is not ApprovedExpertCompositionSource:
            raise ExpertCompositionSourceError(
                "composition source freshness requires its live capability"
            )
        capability._require_bound(self)
        candidate_id = capability.source_reference.candidate_id
        current_snapshot = self.validation_store.snapshot(candidate_id)
        self._require_approved_snapshot(current_snapshot)
        if current_snapshot != capability.approval_snapshot:
            raise ExpertCompositionSourceError(
                "approved composition source validation head is no longer current"
            )
        current_candidate = self.candidate_store.read(candidate_id)
        if current_candidate != capability.stored_candidate:
            raise ExpertCompositionSourceError(
                "approved composition source candidate package changed"
            )
        reopened_snapshot = self.validation_store.snapshot(candidate_id)
        if reopened_snapshot != capability.approval_snapshot:
            raise ExpertCompositionSourceError(
                "approved composition source validation head changed during freshness check"
            )

    @contextmanager
    def lease_current_approvals(
        self,
        capabilities: tuple[ApprovedExpertCompositionSource, ...],
    ) -> Iterator[ExpertCompositionApprovalLease]:
        """Hold current terminal approval heads across composition persistence."""

        with ExitStack() as stack:
            stack.enter_context(self.validation_store._lock(exclusive=False))
            self._require_approvals_current_unlocked(capabilities)
            lease = _seal_expert_composition_approval_lease(
                resolver=self,
                approved_sources=capabilities,
            )
            stack.callback(lease._deactivate)
            yield lease

    def _require_approvals_current_unlocked(
        self,
        capabilities: tuple[ApprovedExpertCompositionSource, ...],
    ) -> None:
        if (
            type(capabilities) is not tuple
            or not capabilities
            or any(
                type(capability) is not ApprovedExpertCompositionSource
                for capability in capabilities
            )
        ):
            raise ExpertCompositionSourceError(
                "composition source approval lease requires exact capabilities"
            )
        for capability in capabilities:
            capability._require_bound(self)
            candidate_id = capability.source_reference.candidate_id
            current_snapshot = self.validation_store._snapshot_unlocked(candidate_id)
            self._require_approved_snapshot(current_snapshot)
            if current_snapshot != capability.approval_snapshot:
                raise ExpertCompositionSourceError(
                    "approved composition source validation head is no longer current"
                )

    @staticmethod
    def _require_approved_snapshot(
        snapshot: ExpertValidationSnapshot | None,
    ) -> ExpertPublicationEligibilityStageResultRecord:
        if (
            type(snapshot) is not ExpertValidationSnapshot
            or type(snapshot.transition) is not ExpertValidationTransition
            or type(snapshot.state) is not ExpertCandidateValidationState
            or type(snapshot.latest_attempt) is not ExpertValidationAttempt
            or not snapshot.accepted_stage_results
            or not snapshot.state.accepted_stage_results
            or not snapshot.transition.accepted_stage_result_record_ids
            or type(snapshot.accepted_stage_results[-1])
            is not ExpertPublicationEligibilityStageResultRecord
        ):
            raise ExpertCompositionSourceError(
                "composition source lacks complete typed approval history"
            )
        transition = snapshot.transition
        state = snapshot.state
        attempt = snapshot.latest_attempt
        result = snapshot.accepted_stage_results[-1]
        final_state_reference = state.accepted_stage_results[-1]
        fence = result.publication_authority_fence
        if (
            state.promotion_state is not ExpertPromotionState.APPROVED
            or state.next_stage is not None
            or result.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED
            or fence is None
            or transition.target_state_id != state.validation_state_id
            or transition.latest_attempt_id != attempt.validation_attempt_id
            or state.validation_attempt_id != attempt.validation_attempt_id
            or transition.transition_stage_result_record_id
            != result.stage_result_record_id
            or transition.accepted_stage_result_record_ids
            != tuple(
                reference.stage_result_record_id
                for reference in state.accepted_stage_results
            )
            or len(snapshot.accepted_stage_results) != len(state.accepted_stage_results)
            or transition.accepted_stage_result_record_ids[-1]
            != result.stage_result_record_id
            or final_state_reference.stage
            is not ExpertValidationStage.PUBLICATION_ELIGIBILITY
            or final_state_reference.stage_result_record_id
            != result.stage_result_record_id
            or result.accepted_stage_results != state.accepted_stage_results[:-1]
            or state.transition_evidence_id != result.stage_result_record_id
            or state.terminal_evidence_ids
            != tuple(
                sorted(
                    {
                        result.promotion_decision.promotion_decision_id,
                        result.release_use_decision.release_use_decision_id,
                        result.release_use_decision.policy_observation.observation_id,
                    }
                )
            )
            or result.release_matrix_acceptance_transition_id
            != transition.predecessor_transition_id
            or result.release_matrix_acceptance_state_id
            != transition.predecessor_state_id
        ):
            raise ExpertCompositionSourceError(
                "composition source validation head is not terminally approved"
            )
        return result

    @staticmethod
    def _require_candidate_approval_join(
        *,
        stored_candidate: StoredExpertCandidate,
        approval_snapshot: ExpertValidationSnapshot,
        publication_result: ExpertPublicationEligibilityStageResultRecord,
        source_reference: ExpertCompositionSourceReference,
    ) -> None:
        attempt = approval_snapshot.latest_attempt
        if attempt is None:
            raise ExpertCompositionSourceError(
                "composition source approval has no validation attempt"
            )
        manifest = stored_candidate.closure.manifest
        transition = approval_snapshot.transition
        state = approval_snapshot.state
        commit_record_id = stored_candidate.commit_record.commit_record_id
        if (
            manifest.candidate_id != source_reference.candidate_id
            or manifest.candidate_id != transition.candidate_id
            or manifest.candidate_id != state.candidate_id
            or manifest.candidate_id != attempt.candidate_id
            or manifest.candidate_id != publication_result.candidate_id
            or manifest.candidate_tree_hash != source_reference.candidate_tree_hash
            or manifest.candidate_tree_hash != transition.candidate_tree_hash
            or manifest.candidate_tree_hash != state.candidate_tree_hash
            or manifest.candidate_tree_hash != attempt.candidate_tree_hash
            or manifest.candidate_tree_hash != publication_result.candidate_tree_hash
            or commit_record_id != source_reference.candidate_commit_record_id
            or commit_record_id != attempt.candidate_commit_record_id
            or commit_record_id != publication_result.candidate_commit_record_id
            or manifest.scope_contract_id != source_reference.scope_contract_id
            or manifest.scope_contract_id != attempt.scope_contract_id
            or manifest.scope_contract_id != publication_result.scope_contract_id
            or manifest.source_base_release_id
            != source_reference.source_base_release_id
            or manifest.source_base_release_id != attempt.source_base_release_id
            or manifest.source_base_release_id
            != publication_result.expected_current_release_id
            or manifest.source_base_repository_map_ref
            != source_reference.source_base_repository_map_id
        ):
            raise ExpertCompositionSourceError(
                "composition source candidate differs from approval authority"
            )
