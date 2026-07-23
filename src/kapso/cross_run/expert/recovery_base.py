"""Authenticated historical source selection for clean-forward recovery."""

from __future__ import annotations

import os
from typing import Protocol

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertScopeContract,
)
from kapso.cross_run.expert.recovery_contracts import (
    ExpertCleanForwardRecoveryPlan,
    ExpertRecoveryReleaseAssessment,
    recovery_assessment_dependency_ids,
)
from kapso.cross_run.expert.release_authority import (
    AuthenticatedExpertReleaseActivation,
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings


class ExpertRecoveryBaseError(ValueError):
    """Clean-forward source selection lacks complete current authority."""


class ExpertRecoveryCurrentReleaseAuthority(Protocol):
    def observe_task_evaluation_current(
        self,
        scope_id: str,
    ) -> TaskEvaluationCurrentReleaseObservation: ...


class ExpertRecoverySecurityAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


class ExpertRecoveryReleaseUseAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_contract: ExpertScopeContract,
        checked_release_ids: tuple[str, ...],
    ) -> ExpertReleaseUsePolicyObservation: ...


class ExpertRecoveryBaseSelection:
    """Runtime result retaining the selected provider-owned source capability."""

    __slots__ = (
        "_owner_process_id",
        "_plan",
        "_selected_activation",
        "_selector",
    )

    def __init__(
        self,
        seal: object,
        selector: ExpertRecoveryBaseSelector,
        *,
        plan: ExpertCleanForwardRecoveryPlan,
        selected_activation: AuthenticatedExpertReleaseActivation | None,
    ) -> None:
        if seal is not _EXPERT_RECOVERY_BASE_SELECTION_SEAL:
            raise ExpertRecoveryBaseError("recovery selection is not selector sealed")
        if type(plan) is not ExpertCleanForwardRecoveryPlan:
            raise ExpertRecoveryBaseError(
                "recovery selection requires one exact durable plan"
            )
        if plan.source_base_release_id is None:
            if selected_activation is not None:
                raise ExpertRecoveryBaseError(
                    "empty recovery selection cannot retain a release capability"
                )
        elif (
            type(selected_activation) is not AuthenticatedExpertReleaseActivation
            or selected_activation.manifest.release_id != plan.source_base_release_id
        ):
            raise ExpertRecoveryBaseError(
                "historical recovery selection differs from its live capability"
            )
        object.__setattr__(self, "_selector", selector)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_selected_activation", selected_activation)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertRecoveryBaseError("recovery selection is immutable")

    def __reduce__(self) -> object:
        raise ExpertRecoveryBaseError("recovery selection cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise ExpertRecoveryBaseError("recovery selection cannot be serialized")

    @property
    def plan(self) -> ExpertCleanForwardRecoveryPlan:
        self._require_owner_process()
        return self._plan

    @property
    def selected_activation(
        self,
    ) -> AuthenticatedExpertReleaseActivation | None:
        self._require_owner_process()
        return self._selected_activation

    def _require_bound(self, selector: ExpertRecoveryBaseSelector) -> None:
        self._require_owner_process()
        if self._selector is not selector:
            raise ExpertRecoveryBaseError(
                "recovery selection belongs to another selector"
            )

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise ExpertRecoveryBaseError("recovery selection is foreign")


_EXPERT_RECOVERY_BASE_SELECTION_SEAL = object()


class ExpertRecoveryBaseSelector:
    """Walk authenticated activation history and choose the newest clean source."""

    __slots__ = (
        "_activation_provider",
        "_current_authority",
        "_release_use_authority",
        "_security_authority",
        "_settings",
    )

    def __init__(
        self,
        *,
        settings: CrossRunSettings,
        activation_provider: GitHubExpertReleaseActivationProvider,
        current_authority: ExpertRecoveryCurrentReleaseAuthority,
        security_authority: ExpertRecoverySecurityAuthority,
        release_use_authority: ExpertRecoveryReleaseUseAuthority,
    ) -> None:
        if (
            type(settings) is not CrossRunSettings
            or type(activation_provider) is not GitHubExpertReleaseActivationProvider
        ):
            raise ExpertRecoveryBaseError(
                "recovery selector requires exact settings and activation provider"
            )
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_activation_provider", activation_provider)
        object.__setattr__(self, "_current_authority", current_authority)
        object.__setattr__(self, "_security_authority", security_authority)
        object.__setattr__(self, "_release_use_authority", release_use_authority)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertRecoveryBaseError("recovery selector authority is immutable")

    def select(
        self,
        scope_contract: ExpertScopeContract,
    ) -> ExpertRecoveryBaseSelection:
        if type(scope_contract) is not ExpertScopeContract:
            raise ExpertRecoveryBaseError(
                "recovery selection requires one exact scope contract"
            )
        current_before = self._current_authority.observe_task_evaluation_current(
            scope_contract.scope_id
        )
        if (
            type(current_before) is not TaskEvaluationCurrentReleaseObservation
            or current_before.scope_id != scope_contract.scope_id
            or current_before.release_id is None
        ):
            raise ExpertRecoveryBaseError(
                "clean-forward recovery requires an authenticated non-empty CURRENT"
            )
        release_id = current_before.release_id
        seen_release_ids: set[str] = set()
        assessments: list[ExpertRecoveryReleaseAssessment] = []
        selected_activation = None
        for sequence_index in range(self._settings.expert.recovery_lineage_limit):
            require_content_id(release_id, "recovery historical release")
            if release_id in seen_release_ids:
                raise ExpertRecoveryBaseError(
                    "recovery activation history contains a cycle"
                )
            seen_release_ids.add(release_id)
            activation = self._activation_provider.resolve_exact(
                scope_contract,
                release_id,
            )
            assessment = self._assess(
                scope_contract=scope_contract,
                sequence_index=sequence_index,
                activation=activation,
            )
            assessments.append(assessment)
            if sequence_index == 0 and not assessment.blocked:
                raise ExpertRecoveryBaseError(
                    "expert CURRENT is clear on both recovery policy planes"
                )
            if sequence_index > 0 and not assessment.blocked:
                selected_activation = activation
                return self._finalize(
                    scope_contract=scope_contract,
                    current_before=current_before,
                    assessments=tuple(assessments),
                    selected_activation=selected_activation,
                )
            predecessor_release_id = (
                assessment.manifest.lineage.activation_predecessor_release_id
            )
            if predecessor_release_id is None:
                return self._finalize(
                    scope_contract=scope_contract,
                    current_before=current_before,
                    assessments=tuple(assessments),
                    selected_activation=None,
                )
            release_id = predecessor_release_id
        raise ExpertRecoveryBaseError(
            "recovery activation history exceeds the configured lineage limit"
        )

    def require_fresh(
        self,
        selection: ExpertRecoveryBaseSelection,
    ) -> ExpertRecoveryBaseSelection:
        """Re-run every authority read before a recovery admission commits."""

        if type(selection) is not ExpertRecoveryBaseSelection:
            raise ExpertRecoveryBaseError(
                "recovery freshness requires one selector-owned selection"
            )
        selection._require_bound(self)
        original_plan = selection.plan
        current = self._current_authority.observe_task_evaluation_current(
            original_plan.scope_contract.scope_id
        )
        if current != original_plan.current_release_observation:
            raise ExpertRecoveryBaseError(
                "recovery selection became stale before admission"
            )
        refreshed = self.select(original_plan.scope_contract)
        if refreshed.plan != original_plan:
            raise ExpertRecoveryBaseError(
                "recovery source authority changed before admission"
            )
        return refreshed

    def _assess(
        self,
        *,
        scope_contract: ExpertScopeContract,
        sequence_index: int,
        activation: AuthenticatedExpertReleaseActivation,
    ) -> ExpertRecoveryReleaseAssessment:
        manifest = activation.manifest
        pointer = activation.pointer
        witness = activation.witness
        security_subject_ids = tuple(
            sorted(
                {
                    manifest.release_id,
                    pointer.publication_record.publication_id,
                    witness.witness_id,
                    *manifest.consumed_dependency_ids,
                }
            )
        )
        security_observation = self._security_authority.observe_exact(
            scope_id=scope_contract.scope_id,
            scope_contract_id=scope_contract.scope_contract_id,
            checked_subject_ids=security_subject_ids,
        )
        release_use_observation = self._release_use_authority.observe_exact(
            scope_contract=scope_contract,
            checked_release_ids=(manifest.release_id,),
        )
        return ExpertRecoveryReleaseAssessment.mint(
            sequence_index=sequence_index,
            scope_contract_id=scope_contract.scope_contract_id,
            manifest=manifest,
            publication_pointer=pointer,
            activation_witness=witness,
            cache_receipt=activation.cache_receipt,
            security_subject_ids=security_subject_ids,
            security_observation=security_observation,
            release_use_observation=release_use_observation,
            exact_dependency_ids=recovery_assessment_dependency_ids(
                scope_contract_id=scope_contract.scope_contract_id,
                manifest=manifest,
                pointer=pointer,
                witness=witness,
                security_observation=security_observation,
                release_use_observation=release_use_observation,
            ),
        )

    def _finalize(
        self,
        *,
        scope_contract: ExpertScopeContract,
        current_before: TaskEvaluationCurrentReleaseObservation,
        assessments: tuple[ExpertRecoveryReleaseAssessment, ...],
        selected_activation: AuthenticatedExpertReleaseActivation | None,
    ) -> ExpertRecoveryBaseSelection:
        if selected_activation is not None:
            self._activation_provider.require_exact(selected_activation)
            selected_manifest = selected_activation.manifest
            source_base_release_id = selected_manifest.release_id
            source_base_tree_hash = selected_manifest.candidate_tree_hash
            source_base_repository_map_ref = selected_manifest.repository_map_ref
            source_base_module_contract_refs = selected_manifest.module_contract_refs
        else:
            source_base_release_id = None
            source_base_tree_hash = EMPTY_EXPERT_TREE_DIGEST
            source_base_repository_map_ref = None
            source_base_module_contract_refs = ()
        current_after = self._current_authority.observe_task_evaluation_current(
            scope_contract.scope_id
        )
        if current_after != current_before:
            raise ExpertRecoveryBaseError(
                "expert CURRENT changed during recovery source selection"
            )
        dependencies = {
            scope_contract.scope_contract_id,
            current_before.observation_id,
            *current_before.validation_closure_ids,
            *(assessment.assessment_id for assessment in assessments),
            *(
                dependency_id
                for assessment in assessments
                for dependency_id in assessment.exact_dependency_ids
            ),
            *source_base_module_contract_refs,
        }
        if current_before.publication_id is not None:
            dependencies.add(current_before.publication_id)
        if source_base_release_id is not None:
            dependencies.add(source_base_release_id)
        if source_base_repository_map_ref is not None:
            dependencies.add(source_base_repository_map_ref)
        plan = ExpertCleanForwardRecoveryPlan.mint(
            configuration_fingerprint=self._settings.configuration_fingerprint,
            scope_contract=scope_contract,
            current_release_observation=current_before,
            assessments=assessments,
            source_base_release_id=source_base_release_id,
            source_base_tree_hash=source_base_tree_hash,
            source_base_repository_map_ref=source_base_repository_map_ref,
            source_base_module_contract_refs=source_base_module_contract_refs,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        return ExpertRecoveryBaseSelection(
            _EXPERT_RECOVERY_BASE_SELECTION_SEAL,
            self,
            plan=plan,
            selected_activation=selected_activation,
        )


__all__ = [
    "ExpertRecoveryBaseError",
    "ExpertRecoveryBaseSelection",
    "ExpertRecoveryBaseSelector",
    "ExpertRecoveryCurrentReleaseAuthority",
    "ExpertRecoveryReleaseUseAuthority",
    "ExpertRecoverySecurityAuthority",
]
