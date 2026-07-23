"""Fresh authority coordination for atomic composition candidate admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    ExpertCandidateCommitRecord,
    ExpertScopeContract,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.composition import (
    ExpertCompositionReducer,
    ExpertCompositionReduction,
)
from kapso.cross_run.expert.composition_admission_contracts import (
    ExpertCompositionAdmissionFence,
    ExpertCompositionSourceAdmissionAuthority,
    composition_admission_security_subject_ids,
)
from kapso.cross_run.expert.composition_admission_authority import (
    _seal_expert_composition_admission_authority,
)
from kapso.cross_run.expert.composition_base import (
    expert_composition_base_security_subject_ids,
)
from kapso.cross_run.expert.composition_base_provider import (
    CurrentExpertCompositionBase,
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.composition_candidate import (
    project_deterministic_composition_candidate,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionDisposition,
    ExpertCompositionPlan,
    expert_composition_configuration_fingerprint,
)
from kapso.cross_run.expert.composition_source import (
    ApprovedExpertCompositionSource,
    ExpertCompositionSourceResolver,
)
from kapso.cross_run.expert.store import ExpertCandidateStore, StoredExpertCandidate
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)


class ExpertCompositionAdmissionError(ValueError):
    """A deterministic composition lacks exact fresh persistence authority."""


class ExpertCompositionAdmissionDenylistAuthority(Protocol):
    """Authenticate current non-rollback denylist state for exact subjects."""

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


@dataclass(frozen=True)
class _ExpertCompositionAdmissionFreshness:
    current_base: CurrentExpertCompositionBase
    approved_sources: tuple[ApprovedExpertCompositionSource, ...]
    source_authorities: tuple[ExpertCompositionSourceAdmissionAuthority, ...]

    def __post_init__(self) -> None:
        if (
            type(self.current_base) is not CurrentExpertCompositionBase
            or type(self.approved_sources) is not tuple
            or not self.approved_sources
            or any(
                type(source) is not ApprovedExpertCompositionSource
                for source in self.approved_sources
            )
            or type(self.source_authorities) is not tuple
            or any(
                type(authority) is not ExpertCompositionSourceAdmissionAuthority
                for authority in self.source_authorities
            )
        ):
            raise ExpertCompositionAdmissionError(
                "composition admission freshness context is invalid"
            )


def compose_expert_active_task_bindings(
    binding_sets: tuple[tuple[CrossRunTaskBindingSettings, ...], ...],
) -> tuple[CrossRunTaskBindingSettings, ...]:
    """Return the canonical union; no source may be hidden by source ordering."""

    if (
        type(binding_sets) is not tuple
        or not binding_sets
        or any(
            type(bindings) is not tuple
            or not bindings
            or any(
                type(binding) is not CrossRunTaskBindingSettings for binding in bindings
            )
            for bindings in binding_sets
        )
    ):
        raise ExpertCompositionAdmissionError(
            "composition active task binding inputs must be exact and non-empty"
        )
    bindings_by_key: dict[tuple[str, str], CrossRunTaskBindingSettings] = {}
    for bindings in binding_sets:
        for binding in bindings:
            key = (binding.task_family_id, binding.task_adapter_id)
            existing = bindings_by_key.get(key)
            if existing is not None and existing != binding:
                raise ExpertCompositionAdmissionError(
                    "composition sources disagree on one active task binding"
                )
            bindings_by_key[key] = binding
    return tuple(binding for _, binding in sorted(bindings_by_key.items()))


def build_expert_composition_plan(
    *,
    current_base: CurrentExpertCompositionBase,
    approved_sources: tuple[ApprovedExpertCompositionSource, ...],
    composition_policy_version: str,
    composition_source_limit: int,
    candidate_entry_limit: int,
    candidate_byte_limit: int,
) -> ExpertCompositionPlan:
    """Derive the only plan implied by sealed base and source capabilities."""

    if (
        type(current_base) is not CurrentExpertCompositionBase
        or type(approved_sources) is not tuple
        or not approved_sources
        or any(
            type(source) is not ApprovedExpertCompositionSource
            for source in approved_sources
        )
    ):
        raise ExpertCompositionAdmissionError(
            "composition planning requires sealed current base and source capabilities"
        )
    ordered_sources = tuple(
        sorted(
            approved_sources,
            key=lambda source: (
                source.source_reference.candidate_id,
                source.source_reference.source_reference_id,
            ),
        )
    )
    if ordered_sources != approved_sources:
        raise ExpertCompositionAdmissionError(
            "composition source capabilities must be canonical"
        )
    active_task_bindings = compose_expert_active_task_bindings(
        tuple(
            source.stored_candidate.closure.validation_context.active_task_bindings
            for source in approved_sources
        )
    )
    source_references = tuple(source.source_reference for source in approved_sources)
    base = current_base.closure
    authorities = {
        base.scope_contract.scope_contract_id,
        base.reference.base_reference_id,
        *base.reference.stable_authority_ids,
        *(reference.source_reference_id for reference in source_references),
        *(
            authority_id
            for reference in source_references
            for authority_id in reference.stable_authority_ids
        ),
    }
    if base.scope_contract.supersedes_scope_contract_id is not None:
        authorities.add(base.scope_contract.supersedes_scope_contract_id)
    return ExpertCompositionPlan.mint(
        scope_contract=base.scope_contract,
        current_base=base.reference,
        sources=source_references,
        active_task_bindings=active_task_bindings,
        composition_policy_version=composition_policy_version,
        composition_source_limit=composition_source_limit,
        candidate_entry_limit=candidate_entry_limit,
        candidate_byte_limit=candidate_byte_limit,
        configuration_fingerprint=expert_composition_configuration_fingerprint(
            composition_policy_version=composition_policy_version,
            composition_source_limit=composition_source_limit,
            candidate_entry_limit=candidate_entry_limit,
            candidate_byte_limit=candidate_byte_limit,
        ),
        stable_authority_ids=tuple(sorted(authorities)),
    )


class ExpertCompositionAdmissionCoordinator:
    """Build and persist one composition under fresh independent authorities."""

    def __init__(
        self,
        *,
        candidate_store: ExpertCandidateStore,
        base_provider: GitHubExpertCompositionBaseProvider,
        source_resolver: ExpertCompositionSourceResolver,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertCompositionAdmissionDenylistAuthority,
    ) -> None:
        if (
            type(candidate_store) is not ExpertCandidateStore
            or type(base_provider) is not GitHubExpertCompositionBaseProvider
            or type(source_resolver) is not ExpertCompositionSourceResolver
            or source_resolver.candidate_store is not candidate_store
            or base_provider._settings != candidate_store.validator.settings
            or task_adapter_authority
            is not source_resolver.validation_store.reducer.task_adapter_provider
        ):
            raise ExpertCompositionAdmissionError(
                "composition admission components do not share one exact authority"
            )
        self.candidate_store = candidate_store
        self.base_provider = base_provider
        self.source_resolver = source_resolver
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority
        settings = candidate_store.validator.settings
        self.reducer = ExpertCompositionReducer(
            candidate_entry_limit=settings.candidate_entry_limit,
            candidate_byte_limit=settings.candidate_byte_limit,
        )
        self._authority = _seal_expert_composition_admission_authority(
            coordinator=self,
            candidate_store=candidate_store,
        )
        self.candidate_store._bind_composition_admission_authority(self._authority)

    def compose_and_persist(
        self,
        *,
        scope_contract: ExpertScopeContract,
        source_candidate_ids: tuple[str, ...],
    ) -> StoredExpertCandidate:
        if (
            type(scope_contract) is not ExpertScopeContract
            or type(source_candidate_ids) is not tuple
            or not source_candidate_ids
            or source_candidate_ids != tuple(sorted(set(source_candidate_ids)))
        ):
            raise ExpertCompositionAdmissionError(
                "composition admission requires one scope and canonical source IDs"
            )
        settings = self.candidate_store.validator.settings
        if len(source_candidate_ids) > settings.composition_source_limit:
            raise ExpertCompositionAdmissionError(
                "composition admission exceeds its configured source limit"
            )
        current_base = self.base_provider.resolve_current(scope_contract)
        sources = tuple(
            self.source_resolver.resolve(candidate_id)
            for candidate_id in source_candidate_ids
        )
        plan = build_expert_composition_plan(
            current_base=current_base,
            approved_sources=sources,
            composition_policy_version=settings.composition_policy_version,
            composition_source_limit=settings.composition_source_limit,
            candidate_entry_limit=settings.candidate_entry_limit,
            candidate_byte_limit=settings.candidate_byte_limit,
        )
        reduction = self.reducer.reduce(
            plan=plan,
            current_base=current_base.closure,
            sources=tuple(source.reduction_source for source in sources),
        )
        self._require_clean_reduction(reduction)
        closure = project_deterministic_composition_candidate(
            reduction=reduction,
            current_base=current_base.closure,
            approved_sources=sources,
            sanitizer=self.candidate_store.validator.sanitizer,
        )
        commit_record = self.candidate_store.preview_composition_commit(closure)
        source_authorities = self._source_authorities(sources)
        freshness_context = _ExpertCompositionAdmissionFreshness(
            current_base=current_base,
            approved_sources=sources,
            source_authorities=source_authorities,
        )
        with self.source_resolver.lease_current_approvals(sources) as approval_lease:
            admission = self.candidate_store._seal_composition_admission(
                authority=self._authority,
                approval_lease=approval_lease,
                closure=closure,
                freshness_context=freshness_context,
            )
            return self.candidate_store._commit_composition_admission(
                authority=self._authority,
                admission=admission,
            )

    def _finalize_composition_admission_under_store_lock(
        self,
        *,
        freshness_context: object,
        closure: ExpertCandidateClosure,
        commit_record: ExpertCandidateCommitRecord,
    ) -> ExpertCompositionAdmissionFence:
        if type(freshness_context) is not _ExpertCompositionAdmissionFreshness:
            raise ExpertCompositionAdmissionError(
                "composition admission finalization lacks its sealed freshness context"
            )
        derivation = closure.derivation
        materialization = derivation.materialization
        plan = materialization.composition_assessment.composition_plan
        sources = freshness_context.approved_sources
        adapter_observations = self._reverify_adapters(
            sources=sources,
            active_task_bindings=plan.active_task_bindings,
            scope_contract_id=plan.scope_contract.scope_contract_id,
        )
        current_before = self.base_provider.require_current(
            freshness_context.current_base
        )
        base_security_subject_ids = expert_composition_base_security_subject_ids(
            freshness_context.current_base.closure,
            current_before,
        )
        security_subject_ids = composition_admission_security_subject_ids(
            closure=closure,
            commit_record=commit_record,
            base_security_subject_ids=base_security_subject_ids,
            source_authorities=freshness_context.source_authorities,
            current_release_observation=current_before,
            task_adapter_trust_observations=adapter_observations,
        )
        denylist = self.security_denylist_authority.observe_exact(
            scope_id=plan.scope_contract.scope_id,
            scope_contract_id=plan.scope_contract.scope_contract_id,
            checked_subject_ids=security_subject_ids,
        )
        if (
            type(denylist) is not SecurityDenylistObservation
            or denylist.checked_subject_ids != security_subject_ids
            or denylist.matched_revocations
        ):
            raise ExpertCompositionAdmissionError(
                "composition admission denylist rejected the exact authority closure"
            )
        current_after = self.base_provider.require_current(
            freshness_context.current_base
        )
        if current_after != current_before:
            raise ExpertCompositionAdmissionError(
                "expert CURRENT changed during composition admission"
            )
        return ExpertCompositionAdmissionFence.mint(
            candidate_id=closure.manifest.candidate_id,
            candidate_commit_record_id=commit_record.commit_record_id,
            candidate_tree_hash=closure.manifest.candidate_tree_hash,
            scope_id=plan.scope_contract.scope_id,
            scope_contract_id=plan.scope_contract.scope_contract_id,
            expected_current_release_id=plan.current_base.release_id,
            composition_plan_id=plan.composition_plan_id,
            composition_materialization_id=materialization.materialization_id,
            base_reference_id=plan.current_base.base_reference_id,
            base_security_subject_ids=base_security_subject_ids,
            source_authorities=freshness_context.source_authorities,
            current_release_observation=current_after,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=denylist,
        )

    @staticmethod
    def _require_clean_reduction(reduction: ExpertCompositionReduction) -> None:
        if (
            type(reduction) is not ExpertCompositionReduction
            or reduction.assessment.disposition
            is not ExpertCompositionDisposition.CLEAN
            or reduction.materialization is None
        ):
            raise ExpertCompositionAdmissionError(
                "composition admission requires a clean deterministic reduction"
            )

    @staticmethod
    def _source_authorities(
        sources: tuple[ApprovedExpertCompositionSource, ...],
    ) -> tuple[ExpertCompositionSourceAdmissionAuthority, ...]:
        authorities = []
        for source in sources:
            reference = source.source_reference
            snapshot = source.approval_snapshot
            result = source.publication_eligibility_result
            fence = result.publication_authority_fence
            if fence is None or snapshot.latest_attempt is None:
                raise ExpertCompositionAdmissionError(
                    "composition source lacks terminal publication authority"
                )
            authorities.append(
                ExpertCompositionSourceAdmissionAuthority.mint(
                    source_reference_id=reference.source_reference_id,
                    candidate_id=reference.candidate_id,
                    candidate_commit_record_id=(reference.candidate_commit_record_id),
                    source_reference_authority_ids=(reference.stable_authority_ids),
                    approval_transition_id=snapshot.transition.transition_id,
                    approval_state_id=snapshot.state.validation_state_id,
                    validation_attempt_id=(
                        snapshot.latest_attempt.validation_attempt_id
                    ),
                    publication_eligibility_result_id=(result.stage_result_record_id),
                    publication_result_dependency_ids=(result.exact_dependency_ids),
                    publication_authority_fence_id=fence.fence_id,
                    publication_fence_security_subject_ids=(fence.security_subject_ids),
                    publication_fence_dependency_ids=fence.exact_dependency_ids,
                    security_subject_ids=source.security_subject_ids,
                )
            )
        return tuple(
            sorted(
                authorities,
                key=lambda authority: (
                    authority.candidate_id,
                    authority.source_reference_id,
                ),
            )
        )

    def _reverify_adapters(
        self,
        *,
        sources: tuple[ApprovedExpertCompositionSource, ...],
        active_task_bindings: tuple[CrossRunTaskBindingSettings, ...],
        scope_contract_id: str,
    ) -> tuple[TaskAdapterTrustObservation, ...]:
        expected_by_id = {
            observation.observation_id: observation
            for source in sources
            for observation in source.publication_eligibility_result.publication_authority_fence.task_adapter_trust_observations
        }
        if not expected_by_id:
            raise ExpertCompositionAdmissionError(
                "composition admission sources lack adapter trust authority"
            )
        observations = []
        resolved_bindings = set()
        for expected in expected_by_id.values():
            adapter = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=expected.task_adapter_manifest_id,
                verification_receipt_id=expected.verification_receipt_id,
            )
            if type(adapter) is not VerifiedTaskAdapter:
                raise ExpertCompositionAdmissionError(
                    "composition admission adapter authority returned another type"
                )
            observation = TaskAdapterTrustObservation.mint(
                task_adapter_manifest_id=adapter.manifest.task_adapter_manifest_id,
                verification_receipt_id=(
                    adapter.verification_receipt.verification_receipt_id
                ),
                verifier_id=adapter.verification_receipt.verifier_id,
                verifier_version=adapter.verification_receipt.verifier_version,
                dependency_ids=adapter.dependency_ids,
            )
            if (
                observation != expected
                or adapter.manifest.scope_contract_id != scope_contract_id
            ):
                raise ExpertCompositionAdmissionError(
                    "composition admission adapter differs from source approval"
                )
            observations.append(observation)
            resolved_bindings.add(
                (
                    adapter.manifest.task_family_id,
                    adapter.manifest.task_adapter_id,
                )
            )
        required_bindings = {
            (binding.task_family_id, binding.task_adapter_id)
            for binding in active_task_bindings
        }
        if not required_bindings.issubset(resolved_bindings):
            raise ExpertCompositionAdmissionError(
                "composition admission adapter authority omits an active binding"
            )
        return tuple(sorted(observations, key=lambda item: item.observation_id))
