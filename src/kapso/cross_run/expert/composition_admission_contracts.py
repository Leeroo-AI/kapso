"""Durable authority evidence for deterministic composition admission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import (
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    StrictContract,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertDeterministicCompositionDerivation,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.composition_base import (
    build_expert_composition_base_closure,
    expert_composition_base_security_subject_ids,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)


class ExpertCompositionAdmissionContractError(ValueError):
    """Composition admission evidence is incomplete or inconsistent."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertCompositionAdmissionContractError(
            f"{name} uses the wrong namespace"
        )


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (not allow_empty and not values) or values != tuple(sorted(set(values))):
        raise ExpertCompositionAdmissionContractError(
            f"{name} must be sorted and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertCompositionSourceAdmissionAuthority(StrictContract):
    """Persisted projection of one live, terminally approved source capability."""

    source_admission_authority_id: str
    source_reference_id: str
    candidate_id: str
    candidate_commit_record_id: str
    source_reference_authority_ids: tuple[str, ...]
    approval_transition_id: str
    approval_state_id: str
    validation_attempt_id: str
    publication_eligibility_result_id: str
    publication_result_dependency_ids: tuple[str, ...]
    publication_authority_fence_id: str
    publication_fence_security_subject_ids: tuple[str, ...]
    publication_fence_dependency_ids: tuple[str, ...]
    security_subject_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-source-admission-authority"
    IDENTITY_FIELD: ClassVar[str] = "source_admission_authority_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.source_reference_id,
                "expert-composition-source-reference",
                "composition admission source reference",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "composition admission source candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "composition admission source commit",
            ),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "composition admission approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "composition admission approval state",
            ),
            (
                self.publication_eligibility_result_id,
                "expert-publication-eligibility-stage-result",
                "composition admission publication eligibility result",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "composition admission validation attempt",
            ),
            (
                self.publication_authority_fence_id,
                "expert-publication-eligibility-authority-fence",
                "composition admission publication authority fence",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        for values, name in (
            (
                self.source_reference_authority_ids,
                "composition admission source reference authorities",
            ),
            (
                self.publication_result_dependency_ids,
                "composition admission publication result dependencies",
            ),
            (
                self.publication_fence_security_subject_ids,
                "composition admission publication fence security subjects",
            ),
            (
                self.publication_fence_dependency_ids,
                "composition admission publication fence dependencies",
            ),
        ):
            _require_sorted_content_ids(values, name)
        _require_sorted_content_ids(
            self.security_subject_ids,
            "composition admission source security subjects",
        )
        expected = {
            self.source_reference_id,
            *self.source_reference_authority_ids,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.approval_transition_id,
            self.approval_state_id,
            self.validation_attempt_id,
            self.publication_eligibility_result_id,
            *self.publication_result_dependency_ids,
            self.publication_authority_fence_id,
            *self.publication_fence_security_subject_ids,
            *self.publication_fence_dependency_ids,
        }
        if set(self.security_subject_ids) != expected:
            raise ExpertCompositionAdmissionContractError(
                "composition admission source security authority is not exact"
            )
        if self.validation_attempt_id not in self.publication_result_dependency_ids:
            raise ExpertCompositionAdmissionContractError(
                "composition admission source result omits its validation attempt"
            )


@dataclass(frozen=True)
class ExpertCompositionAdmissionFence(StrictContract):
    """Historical proof that one composition crossed fresh admission authorities."""

    admission_fence_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    scope_id: str
    scope_contract_id: str
    expected_parent_release_id: str
    composition_plan_id: str
    composition_materialization_id: str
    base_reference_id: str
    base_security_subject_ids: tuple[str, ...]
    source_authorities: tuple[ExpertCompositionSourceAdmissionAuthority, ...]
    current_release_observation: SourceReplayCurrentReleaseObservation
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...]
    security_denylist_observation: SecurityDenylistObservation

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-admission-fence"
    IDENTITY_FIELD: ClassVar[str] = "admission_fence_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "composition admission scope")
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "composition admission candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "composition admission candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "composition admission scope contract",
            ),
            (
                self.expected_parent_release_id,
                "expert-base-release",
                "composition admission expected parent",
            ),
            (
                self.composition_plan_id,
                "expert-composition-plan",
                "composition admission plan",
            ),
            (
                self.composition_materialization_id,
                "expert-composition-materialization",
                "composition admission materialization",
            ),
            (
                self.base_reference_id,
                "expert-composition-base-reference",
                "composition admission base reference",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if (
            not isinstance(self.candidate_tree_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.candidate_tree_hash) is None
        ):
            raise ExpertCompositionAdmissionContractError(
                "composition admission candidate tree hash is invalid"
            )
        _require_sorted_content_ids(
            self.base_security_subject_ids,
            "composition admission base security subjects",
        )
        if not {
            self.base_reference_id,
            self.scope_contract_id,
            self.expected_parent_release_id,
        }.issubset(self.base_security_subject_ids):
            raise ExpertCompositionAdmissionContractError(
                "composition admission base omits mandatory security authority"
            )
        source_keys = tuple(
            (authority.candidate_id, authority.source_reference_id)
            for authority in self.source_authorities
        )
        if (
            not self.source_authorities
            or any(
                type(authority) is not ExpertCompositionSourceAdmissionAuthority
                for authority in self.source_authorities
            )
            or source_keys != tuple(sorted(set(source_keys)))
        ):
            raise ExpertCompositionAdmissionContractError(
                "composition admission source authorities must be canonical"
            )
        observation_ids = tuple(
            observation.observation_id
            for observation in self.task_adapter_trust_observations
        )
        if (
            not observation_ids
            or any(
                type(observation) is not TaskAdapterTrustObservation
                for observation in self.task_adapter_trust_observations
            )
            or observation_ids != tuple(sorted(set(observation_ids)))
        ):
            raise ExpertCompositionAdmissionContractError(
                "composition admission adapter observations must be canonical"
            )
        current = self.current_release_observation
        denylist = self.security_denylist_observation
        if (
            type(current) is not SourceReplayCurrentReleaseObservation
            or current.scope_id != self.scope_id
            or current.release_id != self.expected_parent_release_id
            or type(denylist) is not SecurityDenylistObservation
            or denylist.scope_id != self.scope_id
            or denylist.scope_contract_id != self.scope_contract_id
            or denylist.matched_revocations
        ):
            raise ExpertCompositionAdmissionContractError(
                "composition admission external authorities do not share one safe scope"
            )

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        return self.security_denylist_observation.checked_subject_ids

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.candidate_id,
                    self.candidate_commit_record_id,
                    self.scope_contract_id,
                    self.expected_parent_release_id,
                    self.composition_plan_id,
                    self.composition_materialization_id,
                    self.base_reference_id,
                    self.current_release_observation.observation_id,
                    self.security_denylist_observation.observation_id,
                    *(
                        authority.source_admission_authority_id
                        for authority in self.source_authorities
                    ),
                    *(
                        observation.observation_id
                        for observation in self.task_adapter_trust_observations
                    ),
                }
            )
        )


def composition_admission_security_subject_ids(
    *,
    closure: ExpertCandidateClosure,
    commit_record: ExpertCandidateCommitRecord,
    base_security_subject_ids: tuple[str, ...],
    source_authorities: tuple[ExpertCompositionSourceAdmissionAuthority, ...],
    current_release_observation: SourceReplayCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    """Project the exact pre-fence revocation closure checked at admission."""

    if (
        type(closure) is not ExpertCandidateClosure
        or type(commit_record) is not ExpertCandidateCommitRecord
        or type(source_authorities) is not tuple
        or any(
            type(authority) is not ExpertCompositionSourceAdmissionAuthority
            for authority in source_authorities
        )
        or type(current_release_observation)
        is not SourceReplayCurrentReleaseObservation
        or type(task_adapter_trust_observations) is not tuple
        or any(
            type(observation) is not TaskAdapterTrustObservation
            for observation in task_adapter_trust_observations
        )
    ):
        raise ExpertCompositionAdmissionContractError(
            "composition admission security projection requires exact authorities"
        )
    derivation = closure.derivation
    manifest = closure.manifest
    if (
        manifest.derivation_kind
        is not ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
        or type(derivation) is not ExpertDeterministicCompositionDerivation
        or commit_record.candidate_id != manifest.candidate_id
    ):
        raise ExpertCompositionAdmissionContractError(
            "composition admission security projection requires one composition"
        )
    materialization = derivation.materialization
    assessment = materialization.composition_assessment
    plan = assessment.composition_plan
    subjects = {
        manifest.candidate_id,
        commit_record.commit_record_id,
        manifest.scope_contract_id,
        manifest.derivation_ref,
        manifest.validation_context_ref,
        manifest.patch_ref,
        manifest.candidate_tree_ref,
        manifest.proposed_repository_map_ref,
        manifest.sanitation_report_id,
        *manifest.module_contract_refs,
        *manifest.source_dependency_ids,
        *manifest.ancestor_candidate_ids,
        *closure.validation_context.stable_dependency_ids,
        derivation.record.derivation_id,
        *derivation.record.source_validation_context_ids,
        *derivation.record.source_validation_context_ids.values(),
        *derivation.record.source_dependency_ids,
        materialization.materialization_id,
        *materialization.stable_authority_ids,
        assessment.assessment_id,
        *assessment.stable_authority_ids,
        plan.composition_plan_id,
        *plan.stable_authority_ids,
        *base_security_subject_ids,
        current_release_observation.observation_id,
        current_release_observation.publication_id,
        *current_release_observation.validation_closure_ids,
    }
    if manifest.parent_release_id is not None:
        subjects.add(manifest.parent_release_id)
    if manifest.parent_repository_map_ref is not None:
        subjects.add(manifest.parent_repository_map_ref)
    for authority in source_authorities:
        subjects.update(
            {
                authority.source_admission_authority_id,
                *authority.security_subject_ids,
            }
        )
    for observation in task_adapter_trust_observations:
        subjects.update(
            {
                observation.observation_id,
                observation.task_adapter_manifest_id,
                observation.verification_receipt_id,
                observation.verifier_authority_subject_id,
                *observation.dependency_ids,
            }
        )
    ordered = tuple(sorted(subjects))
    for subject_id in ordered:
        require_content_id(subject_id, "composition admission security subject")
    return ordered


def validate_expert_composition_admission_fence(
    *,
    fence: ExpertCompositionAdmissionFence,
    closure: ExpertCandidateClosure,
    commit_record: ExpertCandidateCommitRecord,
) -> None:
    """Join a durable admission fence to its exact scientific package."""

    if type(fence) is not ExpertCompositionAdmissionFence:
        raise ExpertCompositionAdmissionContractError(
            "composition admission join requires one exact fence"
        )
    derivation = closure.derivation
    if type(derivation) is not ExpertDeterministicCompositionDerivation:
        raise ExpertCompositionAdmissionContractError(
            "composition admission fence cannot authorize another derivation"
        )
    materialization = derivation.materialization
    plan = materialization.composition_assessment.composition_plan
    source_keys = tuple(
        (authority.candidate_id, authority.source_reference_id)
        for authority in fence.source_authorities
    )
    expected_source_keys = tuple(
        (source.candidate_id, source.source_reference_id) for source in plan.sources
    )
    if (
        fence.candidate_id != closure.manifest.candidate_id
        or fence.candidate_commit_record_id != commit_record.commit_record_id
        or fence.candidate_tree_hash != closure.manifest.candidate_tree_hash
        or fence.scope_id != plan.scope_contract.scope_id
        or fence.scope_contract_id != plan.scope_contract.scope_contract_id
        or fence.expected_parent_release_id != plan.current_base.release_id
        or fence.composition_plan_id != plan.composition_plan_id
        or fence.composition_materialization_id != materialization.materialization_id
        or fence.base_reference_id != plan.current_base.base_reference_id
        or source_keys != expected_source_keys
    ):
        raise ExpertCompositionAdmissionContractError(
            "composition admission fence differs from its candidate closure"
        )
    for authority, source in zip(fence.source_authorities, plan.sources):
        if (
            authority.candidate_commit_record_id != source.candidate_commit_record_id
            or authority.source_reference_authority_ids != source.stable_authority_ids
        ):
            raise ExpertCompositionAdmissionContractError(
                "composition admission source authority differs from its stable reference"
            )
    context = closure.validation_context
    if (
        context.parent_scope_contract is None
        or context.parent_release is None
        or context.parent_tree_receipt is None
        or context.parent_repository_map is None
    ):
        raise ExpertCompositionAdmissionContractError(
            "composition admission candidate lacks its parent base closure"
        )
    parent_base = build_expert_composition_base_closure(
        scope_contract=context.parent_scope_contract,
        release_manifest=context.parent_release,
        parent_tree_receipt=context.parent_tree_receipt,
        repository_map=context.parent_repository_map,
        module_contracts=context.parent_module_contracts,
        source_contents=derivation.parent_contents,
    )
    expected_base_security_subject_ids = expert_composition_base_security_subject_ids(
        parent_base,
        fence.current_release_observation,
    )
    if fence.base_security_subject_ids != expected_base_security_subject_ids:
        raise ExpertCompositionAdmissionContractError(
            "composition admission base security closure is not exact"
        )
    expected_subject_ids = composition_admission_security_subject_ids(
        closure=closure,
        commit_record=commit_record,
        base_security_subject_ids=fence.base_security_subject_ids,
        source_authorities=fence.source_authorities,
        current_release_observation=fence.current_release_observation,
        task_adapter_trust_observations=fence.task_adapter_trust_observations,
    )
    if fence.security_subject_ids != expected_subject_ids:
        raise ExpertCompositionAdmissionContractError(
            "composition admission denylist does not cover the exact candidate authority"
        )
