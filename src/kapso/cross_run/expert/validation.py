"""Deterministic expert-candidate enrollment and ordered validation reduction."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping, Protocol

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertAcceptedStageResultRef,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertEvaluatorAttestation,
    ExpertEvaluatorAttestationEnvelope,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertEvaluatorRun,
    ExpertPromotionState,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplaySelection,
    ExpertValidationAuthorityInvalidation,
    ExpertValidationAuthorityInvalidationKind,
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
    TaskAdapterPackagePin,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.replay import _derive_expert_source_replay_selection
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.settings import (
    ExpertEvaluatorSettings,
    ExpertValidationPolicy,
    ExpertValidationSettings,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
    task_adapter_binding_id,
)


class ExpertValidationError(ValueError):
    """An enrollment, evaluator result, or state transition is invalid."""


@dataclass(frozen=True)
class ExpertEligibilityResult:
    decision: ExpertCandidateEligibilityDecision
    policy: ExpertValidationPolicy


@dataclass(frozen=True)
class ExpertValidationStart:
    attempt: ExpertValidationAttempt | None
    state: ExpertCandidateValidationState


@dataclass(frozen=True)
class ExpertValidationAuthorityInvalidationResult:
    invalidation: ExpertValidationAuthorityInvalidation
    state: ExpertCandidateValidationState


def _expert_evaluator_base_input_ids(
    attempt: ExpertValidationAttempt,
) -> set[str]:
    input_ids = {
        attempt.validation_attempt_id,
        attempt.candidate_id,
        attempt.candidate_commit_record_id,
        attempt.scope_contract_id,
        attempt.eligibility_decision_id,
        attempt.validation_policy_id,
        *(pin.task_adapter_manifest_id for pin in attempt.task_adapter_pins),
        *(pin.verification_receipt_id for pin in attempt.task_adapter_pins),
        *attempt.eligibility_dependency_ids,
    }
    if attempt.parent_release_id is not None:
        input_ids.add(attempt.parent_release_id)
    return input_ids


def validate_source_replay_request_authority_shape(
    *,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    request: ExpertSourceReplayExecutionRequest,
    settings: ExpertValidationSettings,
    error_type: type[ValueError] = ExpertValidationError,
) -> None:
    evaluator_matches = tuple(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    selection = attempt.source_replay_selection
    if len(evaluator_matches) != 1 or selection is None:
        raise error_type("source replay request has no unique configured authority")
    evaluator = evaluator_matches[0]
    selected_by_episode = {
        episode_id: (
            selection_case.source_bundle_id,
            selection_case.episode_reason_codes[episode_id],
        )
        for selection_case in selection.cases
        for episode_id in selection_case.episode_ids
    }
    adapter_pin_by_episode = {
        episode_id: pin
        for pin in selection.source_adapter_pins
        for episode_id in pin.episode_ids
    }
    request_by_episode = {case.episode_id: case for case in request.cases}
    if (
        state.promotion_state is not ExpertPromotionState.VALIDATING
        or state.next_stage is not ExpertValidationStage.SOURCE_RUN_REPLAY
        or state.validation_attempt_id != attempt.validation_attempt_id
        or state.candidate_id != attempt.candidate_id
        or state.candidate_tree_hash != attempt.candidate_tree_hash
        or attempt.parent_release_id is None
        or attempt.validation_policy_id
        != settings.policy.validation_policy().validation_policy_id
        or attempt.configuration_fingerprint != settings.configuration_fingerprint
        or request.validation_attempt_id != attempt.validation_attempt_id
        or request.authorization_state_id != state.validation_state_id
        or request.source_replay_selection_id != selection.source_replay_selection_id
        or request.candidate_id != attempt.candidate_id
        or request.candidate_tree_hash != attempt.candidate_tree_hash
        or request.candidate_commit_record_id != attempt.candidate_commit_record_id
        or request.scope_contract_id != attempt.scope_contract_id
        or request.parent_release_id != attempt.parent_release_id
        or request.validation_policy_id != attempt.validation_policy_id
        or request.configuration_fingerprint != attempt.configuration_fingerprint
        or request.request_policy_version
        != settings.policy.source_replay_request_policy_version
        or (
            request.evaluator_id,
            request.evaluator_role,
            request.evaluator_version,
        )
        != (
            evaluator.evaluator_id,
            evaluator.evaluator_role,
            evaluator.evaluator_version,
        )
        or request.attempt_dependency_ids != attempt.eligibility_dependency_ids
        or set(request_by_episode) != set(selected_by_episode)
        or set(adapter_pin_by_episode) != set(selected_by_episode)
    ):
        raise error_type(
            "source replay request differs from current validation authority"
        )
    for episode_id, request_case in request_by_episode.items():
        source_bundle_id, reason_codes = selected_by_episode[episode_id]
        adapter_pin = adapter_pin_by_episode[episode_id]
        if (
            request_case.source_bundle_id != source_bundle_id
            or request_case.episode_reason_codes != reason_codes
            or request_case.adapter_binding_id
            != task_adapter_binding_id(
                adapter_pin.task_family_id,
                adapter_pin.task_adapter_id,
            )
            or request_case.task_adapter_manifest_id
            != adapter_pin.task_adapter_manifest_id
            or request_case.verification_receipt_id
            != adapter_pin.verification_receipt_id
        ):
            raise error_type(
                "source replay request cases differ from the selected evidence"
            )


class ExpertAttestationVerifier(Protocol):
    """Verify one evaluator signature against its declared trust root."""

    def verify(self, envelope: ExpertEvaluatorAttestationEnvelope) -> None: ...


class ExpertCandidateReader(Protocol):
    """Reopen one exact immutable M7 candidate package."""

    def read(self, candidate_id: str) -> StoredExpertCandidate: ...


class ExpertCurrentReleaseProvider(Protocol):
    """Resolve the active scientific release identity for one scope."""

    def current_release_id(self, scope_id: str) -> str | None: ...


@dataclass(frozen=True)
class ExpertValidationPredecessor:
    latest_attempt: ExpertValidationAttempt | None
    state: ExpertCandidateValidationState

    def __post_init__(self) -> None:
        if self.latest_attempt is None:
            if self.state.validation_attempt_id is not None:
                raise ExpertValidationError(
                    "validation predecessor state requires its latest attempt"
                )
            return
        if (
            self.latest_attempt.candidate_id != self.state.candidate_id
            or self.latest_attempt.candidate_tree_hash != self.state.candidate_tree_hash
        ):
            raise ExpertValidationError(
                "latest validation attempt belongs to another candidate"
            )
        if (
            self.state.validation_attempt_id is not None
            and self.latest_attempt.validation_attempt_id
            != self.state.validation_attempt_id
        ):
            raise ExpertValidationError(
                "validation predecessor state does not reference its latest attempt"
            )


class ExpertValidationStateProvider(Protocol):
    """Resolve the exact current persisted validation state for a candidate."""

    def current(self, candidate_id: str) -> ExpertValidationPredecessor | None: ...


class ExpertCandidateEligibilityEvaluator:
    """Reopen one M7 candidate and derive its validation track and stage plan."""

    def __init__(
        self,
        settings: ExpertValidationSettings,
        candidate_store: ExpertCandidateReader,
        task_adapter_provider: VerifiedTaskAdapterProvider,
        current_release_provider: ExpertCurrentReleaseProvider,
    ) -> None:
        self.settings = settings
        self.candidate_store = candidate_store
        self.task_adapter_provider = task_adapter_provider
        self.current_release_provider = current_release_provider

    def decide(
        self,
        *,
        candidate_id: str,
    ) -> ExpertEligibilityResult:
        require_content_id(candidate_id, "candidate_id")
        stored = self.candidate_store.read(candidate_id)
        task_adapters = tuple(
            self.task_adapter_provider.resolve_active(
                scope_contract_id=stored.closure.manifest.scope_contract_id,
                task_family_id=binding.task_family_id,
                task_adapter_id=binding.task_adapter_id,
            )
            for binding in stored.closure.trigger_packet.active_task_bindings
        )
        return self._decide(stored, task_adapters)

    def replay(
        self,
        *,
        candidate_id: str,
        task_adapter_pins: tuple[TaskAdapterPackagePin, ...],
    ) -> ExpertEligibilityResult:
        require_content_id(candidate_id, "candidate_id")
        stored = self.candidate_store.read(candidate_id)
        task_adapters = tuple(
            self.task_adapter_provider.resolve_exact(
                task_adapter_manifest_id=pin.task_adapter_manifest_id,
                verification_receipt_id=pin.verification_receipt_id,
            )
            for pin in task_adapter_pins
        )
        return self._decide(stored, task_adapters)

    def _decide(
        self,
        stored: StoredExpertCandidate,
        task_adapters: tuple[VerifiedTaskAdapter, ...],
    ) -> ExpertEligibilityResult:
        current_parent_release_id = self.current_release_provider.current_release_id(
            stored.closure.trigger_packet.scope_contract.scope_id
        )
        if current_parent_release_id is not None:
            require_content_id(
                current_parent_release_id,
                "current_parent_release_id",
            )
        (
            adapter_pins,
            adapter_verification_ids,
            configured_task_family_ids,
        ) = self._adapter_bindings(
            stored,
            task_adapters,
        )
        policy = self.settings.policy.validation_policy()
        validation_track = self._validation_track(stored)
        stage_plan = self.settings.policy.required_stages(
            validation_track,
            configured_task_family_ids,
            has_parent_release=stored.closure.manifest.parent_release_id is not None,
        )
        parent_matches = (
            stored.closure.manifest.parent_release_id == current_parent_release_id
        )
        infrastructure_available = self.settings.policy.can_validate(
            validation_track,
            configured_task_family_ids,
            has_parent_release=stored.closure.manifest.parent_release_id is not None,
        )
        eligible = parent_matches and infrastructure_available
        if not parent_matches:
            reason_code = "stale_parent_release"
        elif not infrastructure_available:
            reason_code = "required_validation_infrastructure_unavailable"
        else:
            reason_code = "eligible"
        source_replay_selection = None
        source_adapter_dependency_ids: tuple[str, ...] = ()
        if eligible and ExpertValidationStage.SOURCE_RUN_REPLAY in stage_plan:
            replay_result = _derive_expert_source_replay_selection(
                stored_candidate=stored,
                settings=self.settings,
            )
            source_replay_selection = replay_result.selection
            if source_replay_selection is None:
                eligible = False
                reason_code = replay_result.reason_code
            else:
                source_adapter_dependency_ids = self._verify_source_replay_adapters(
                    source_replay_selection
                )
        manifest = stored.closure.manifest
        dependencies = {
            manifest.candidate_id,
            stored.commit_record.commit_record_id,
            manifest.scope_contract_id,
            policy.validation_policy_id,
            manifest.trigger_decision_id,
            manifest.trigger_evidence_packet_id,
            manifest.sanitation_report_id,
            *(pin.task_adapter_manifest_id for pin in adapter_pins),
            *(pin.verification_receipt_id for pin in adapter_pins),
            *adapter_verification_ids,
            *source_adapter_dependency_ids,
        }
        if manifest.parent_release_id is not None:
            dependencies.add(manifest.parent_release_id)
        if source_replay_selection is not None:
            dependencies.update(
                {
                    source_replay_selection.source_replay_selection_id,
                    *source_replay_selection.exact_dependency_ids,
                }
            )
        decision = ExpertCandidateEligibilityDecision.mint(
            candidate_id=manifest.candidate_id,
            candidate_tree_hash=manifest.candidate_tree_hash,
            candidate_commit_record_id=stored.commit_record.commit_record_id,
            scope_contract_id=manifest.scope_contract_id,
            parent_release_id=manifest.parent_release_id,
            validation_policy_id=policy.validation_policy_id,
            configuration_fingerprint=self.settings.configuration_fingerprint,
            eligible=eligible,
            validation_track=validation_track,
            required_stages=stage_plan if eligible else (),
            configured_task_family_ids=configured_task_family_ids,
            task_adapter_pins=adapter_pins,
            source_replay_selection=source_replay_selection,
            exact_dependency_ids=tuple(sorted(dependencies)),
            reason_code=reason_code,
        )
        return ExpertEligibilityResult(decision=decision, policy=policy)

    def _verify_source_replay_adapters(
        self,
        selection: ExpertSourceReplaySelection,
    ) -> tuple[str, ...]:
        dependency_ids: set[str] = set()
        for pin in selection.source_adapter_pins:
            adapter = self.task_adapter_provider.resolve_exact(
                task_adapter_manifest_id=pin.task_adapter_manifest_id,
                verification_receipt_id=pin.verification_receipt_id,
            )
            if not isinstance(adapter, VerifiedTaskAdapter):
                raise ExpertValidationError(
                    "historical source replay adapter is not a verified package"
                )
            manifest = adapter.manifest
            if (
                manifest.scope_contract_id != pin.scope_contract_id
                or manifest.task_family_id != pin.task_family_id
                or manifest.task_adapter_id != pin.task_adapter_id
                or manifest.task_adapter_manifest_id != pin.task_adapter_manifest_id
                or adapter.verification_receipt.verification_receipt_id
                != pin.verification_receipt_id
            ):
                raise ExpertValidationError(
                    "historical source replay adapter differs from its exact pin"
                )
            dependency_ids.update(adapter.dependency_ids)
        return tuple(sorted(dependency_ids))

    @staticmethod
    def _validation_track(
        stored: StoredExpertCandidate,
    ) -> ExpertValidationTrack:
        manifest = stored.closure.manifest
        if manifest.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE:
            return ExpertValidationTrack.REPOSITORY_ARCHITECTURE
        if stored.closure.trigger_decision.reason_code == "mechanically_general_fix":
            return ExpertValidationTrack.MECHANICAL_GENERAL_FIX
        return ExpertValidationTrack.BEHAVIORAL_CAPABILITY

    @staticmethod
    def _adapter_bindings(
        stored: StoredExpertCandidate,
        task_adapters: tuple[VerifiedTaskAdapter, ...],
    ) -> tuple[
        tuple[TaskAdapterPackagePin, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if not task_adapters or any(
            not isinstance(adapter, VerifiedTaskAdapter) for adapter in task_adapters
        ):
            raise ExpertValidationError(
                "task adapter provider returned an unverified package"
            )
        ordering = tuple(
            (adapter.manifest.task_family_id, adapter.manifest.task_adapter_id)
            for adapter in task_adapters
        )
        if len(ordering) != len(set(ordering)):
            raise ExpertValidationError("task adapters must be non-empty and unique")
        manifest = stored.closure.manifest
        scope_dimension_ids = {
            schema.dimension_id
            for schema in stored.closure.trigger_packet.scope_contract.context_dimension_schemas
        }
        expected_bindings = {
            (binding.task_family_id, binding.task_adapter_id)
            for binding in stored.closure.trigger_packet.active_task_bindings
        }
        if expected_bindings != set(ordering):
            raise ExpertValidationError(
                "task adapters differ from the candidate trigger bindings"
            )
        pins: list[TaskAdapterPackagePin] = []
        for verified_adapter in task_adapters:
            adapter = verified_adapter.manifest
            if (
                adapter.scope_contract_id != manifest.scope_contract_id
                or (adapter.task_family_id, adapter.task_adapter_id)
                not in expected_bindings
                or not set(adapter.context_binding.consumed_dimension_ids).issubset(
                    scope_dimension_ids
                )
            ):
                raise ExpertValidationError(
                    "task adapter does not match candidate scope and trigger"
                )
            binding_id = task_adapter_binding_id(
                adapter.task_family_id,
                adapter.task_adapter_id,
            )
            pins.append(
                TaskAdapterPackagePin(
                    adapter_binding_id=binding_id,
                    task_adapter_manifest_id=adapter.task_adapter_manifest_id,
                    verification_receipt_id=(
                        verified_adapter.verification_receipt.verification_receipt_id
                    ),
                )
            )
        verification_ids = tuple(
            sorted(
                {
                    dependency_id
                    for adapter in task_adapters
                    for dependency_id in adapter.dependency_ids
                }
            )
        )
        configured_task_family_ids = tuple(
            sorted({task_family_id for task_family_id, _ in expected_bindings})
        )
        return (
            tuple(sorted(pins, key=lambda pin: pin.adapter_binding_id)),
            verification_ids,
            configured_task_family_ids,
        )


class ExpertEvaluatorRunBuilder:
    """Construct one bounded, fully attributed executable-stage result."""

    def __init__(self, settings: ExpertValidationSettings) -> None:
        self.settings = settings

    def build(
        self,
        *,
        attempt: ExpertValidationAttempt,
        stage: ExpertValidationStage,
        exact_additional_input_ids: tuple[str, ...],
        output_payloads: Mapping[str, bytes],
        measurements: Mapping[str, float],
        costs: Mapping[str, float],
        duration_seconds: float,
        outcome: ExpertEvaluatorOutcome,
        signature: str,
    ) -> ExpertEvaluatorResultRecord:
        policy = self.settings.policy.validation_policy()
        if (
            attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "validation attempt differs from evaluator configuration"
            )
        if stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            raise ExpertValidationError(
                "source replay requires a typed execution receipt"
            )
        if stage in {
            ExpertValidationStage.AUTOMATED_REVIEW,
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        }:
            raise ExpertValidationError(
                f"stage {stage.value} requires a typed stage path"
            )
        evaluator = self._evaluator(stage)
        self._validate_outputs(output_payloads)
        for input_id in exact_additional_input_ids:
            require_content_id(input_id, "exact_additional_input_ids")
        if exact_additional_input_ids != tuple(sorted(set(exact_additional_input_ids))):
            raise ExpertValidationError(
                "additional evaluator inputs must be sorted and unique"
            )
        stages_requiring_external_evidence = {
            ExpertValidationStage.DEVELOPMENT_ANCHORS,
            ExpertValidationStage.CROSS_FAMILY_TRANSFER,
            ExpertValidationStage.SEALED_CANARY,
        }
        if (
            stage in stages_requiring_external_evidence
            and not exact_additional_input_ids
        ):
            raise ExpertValidationError(
                f"stage {stage.value} requires exact external evidence inputs"
            )
        exact_input_ids = {
            *_expert_evaluator_base_input_ids(attempt),
            *exact_additional_input_ids,
        }
        checksums = {
            path: tree_or_blob_digest(payload)
            for path, payload in sorted(output_payloads.items())
        }
        run = ExpertEvaluatorRun.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            stage=stage,
            evaluator_id=evaluator.evaluator_id,
            evaluator_role=evaluator.evaluator_role,
            evaluator_version=evaluator.evaluator_version,
            exact_input_ids=tuple(sorted(exact_input_ids)),
            output_payloads_base64={
                path: base64.b64encode(payload).decode("ascii")
                for path, payload in sorted(output_payloads.items())
            },
            output_checksums=checksums,
            measurements=measurements,
            costs=costs,
            duration_seconds=duration_seconds,
            outcome=outcome,
        )
        trust_root_id = (
            self.settings.policy.sealed_canary_trust_root
            if stage is ExpertValidationStage.SEALED_CANARY
            else None
        )
        if stage is ExpertValidationStage.SEALED_CANARY and trust_root_id is None:
            raise ExpertValidationError("sealed canary trust root is unavailable")
        attestation = ExpertEvaluatorAttestation.mint(
            evaluator_run_id=run.evaluator_run_id,
            issuer_id=evaluator.evaluator_id,
            trust_root_id=trust_root_id,
            predicate_digest=tree_or_blob_digest(run.to_json_bytes()),
        )
        return ExpertEvaluatorResultRecord.mint(
            evaluator_run=run,
            attestation_envelope=ExpertEvaluatorAttestationEnvelope(
                attestation=attestation,
                signature=signature,
            ),
        )

    def _evaluator(self, stage: ExpertValidationStage) -> ExpertEvaluatorSettings:
        matches = tuple(
            evaluator
            for evaluator in self.settings.policy.evaluators
            if evaluator.stage is stage
        )
        if len(matches) != 1:
            raise ExpertValidationError(
                f"stage {stage.value} requires a dedicated decision path"
            )
        return matches[0]

    def _validate_outputs(self, output_payloads: Mapping[str, bytes]) -> None:
        if not output_payloads:
            raise ExpertValidationError("evaluator outputs must not be empty")
        if len(output_payloads) > self.settings.policy.artifact_entry_limit:
            raise ExpertValidationError("evaluator output entry limit exceeded")
        total_bytes = 0
        for path, payload in output_payloads.items():
            if not isinstance(path, str) or not isinstance(payload, bytes):
                raise ExpertValidationError(
                    "evaluator outputs must map paths to exact bytes"
                )
            total_bytes += len(payload)
        if total_bytes > self.settings.policy.artifact_byte_limit:
            raise ExpertValidationError("evaluator output byte limit exceeded")


class ExpertValidationReducer:
    """Mint only exact enrollment and executable-stage state transitions."""

    def __init__(
        self,
        settings: ExpertValidationSettings,
        candidate_store: ExpertCandidateReader,
        attestation_verifier: ExpertAttestationVerifier,
        task_adapter_provider: VerifiedTaskAdapterProvider,
        current_release_provider: ExpertCurrentReleaseProvider,
        validation_state_provider: ExpertValidationStateProvider,
    ) -> None:
        self.settings = settings
        self.candidate_store = candidate_store
        self.attestation_verifier = attestation_verifier
        self.task_adapter_provider = task_adapter_provider
        self.current_release_provider = current_release_provider
        self.validation_state_provider = validation_state_provider

    def start(
        self,
        *,
        eligibility: ExpertEligibilityResult,
    ) -> ExpertValidationStart:
        predecessor = self.validation_state_provider.current(
            eligibility.decision.candidate_id
        )
        return self.start_from_predecessor(
            eligibility=eligibility,
            predecessor=predecessor,
        )

    def start_from_predecessor(
        self,
        *,
        eligibility: ExpertEligibilityResult,
        predecessor: ExpertValidationPredecessor | None,
    ) -> ExpertValidationStart:
        expected = ExpertCandidateEligibilityEvaluator(
            self.settings,
            self.candidate_store,
            self.task_adapter_provider,
            self.current_release_provider,
        ).replay(
            candidate_id=eligibility.decision.candidate_id,
            task_adapter_pins=eligibility.decision.task_adapter_pins,
        )
        if expected != eligibility:
            raise ExpertValidationError(
                "eligibility differs from deterministic candidate enrollment"
            )
        predecessor_attempt = (
            None if predecessor is None else predecessor.latest_attempt
        )
        predecessor_state = None if predecessor is None else predecessor.state
        self._validate_predecessors(
            eligibility.decision.candidate_id,
            predecessor_attempt,
            predecessor_state,
        )
        predecessor_state_id = (
            None if predecessor_state is None else predecessor_state.validation_state_id
        )
        if not eligibility.decision.eligible:
            state = ExpertCandidateValidationState.mint(
                validation_attempt_id=None,
                candidate_id=eligibility.decision.candidate_id,
                candidate_tree_hash=eligibility.decision.candidate_tree_hash,
                predecessor_state_id=predecessor_state_id,
                promotion_state=ExpertPromotionState.INELIGIBLE,
                accepted_stage_results=(),
                next_stage=None,
                review_assertion_ids=(),
                terminal_evidence_ids=(eligibility.decision.eligibility_decision_id,),
                transition_evidence_id=eligibility.decision.eligibility_decision_id,
                reason=eligibility.decision.reason_code,
            )
            return ExpertValidationStart(attempt=None, state=state)
        attempt = ExpertValidationAttempt.mint(
            candidate_id=eligibility.decision.candidate_id,
            candidate_tree_hash=eligibility.decision.candidate_tree_hash,
            candidate_commit_record_id=(
                eligibility.decision.candidate_commit_record_id
            ),
            scope_contract_id=eligibility.decision.scope_contract_id,
            parent_release_id=eligibility.decision.parent_release_id,
            eligibility_decision_id=(eligibility.decision.eligibility_decision_id),
            validation_policy_id=eligibility.decision.validation_policy_id,
            configuration_fingerprint=(eligibility.decision.configuration_fingerprint),
            validation_track=eligibility.decision.validation_track,
            attempt_number=(
                1
                if predecessor_attempt is None
                else predecessor_attempt.attempt_number + 1
            ),
            predecessor_attempt_id=(
                None
                if predecessor_attempt is None
                else predecessor_attempt.validation_attempt_id
            ),
            required_stages=eligibility.decision.required_stages,
            configured_task_family_ids=(
                eligibility.decision.configured_task_family_ids
            ),
            task_adapter_pins=eligibility.decision.task_adapter_pins,
            source_replay_selection=(eligibility.decision.source_replay_selection),
            eligibility_dependency_ids=tuple(
                sorted(
                    {
                        eligibility.decision.eligibility_decision_id,
                        *eligibility.decision.exact_dependency_ids,
                    }
                )
            ),
        )
        state = ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            predecessor_state_id=predecessor_state_id,
            promotion_state=ExpertPromotionState.VALIDATING,
            accepted_stage_results=(),
            next_stage=attempt.required_stages[0],
            review_assertion_ids=(),
            terminal_evidence_ids=(),
            transition_evidence_id=eligibility.decision.eligibility_decision_id,
            reason="validation_attempt_started",
        )
        return ExpertValidationStart(attempt=attempt, state=state)

    def invalidate_parent_authority(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
    ) -> ExpertValidationAuthorityInvalidationResult:
        if (
            state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.candidate_id != attempt.candidate_id
            or state.candidate_tree_hash != attempt.candidate_tree_hash
            or attempt.parent_release_id is None
        ):
            raise ExpertValidationError(
                "only an active parent-bound attempt may be invalidated"
            )
        policy = self.settings.policy.validation_policy()
        if (
            attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "active attempt differs from reducer configuration"
            )
        stored = self.candidate_store.read(attempt.candidate_id)
        manifest = stored.closure.manifest
        packet = stored.closure.trigger_packet
        if (
            manifest.candidate_id != attempt.candidate_id
            or manifest.candidate_tree_hash != attempt.candidate_tree_hash
            or stored.commit_record.commit_record_id
            != attempt.candidate_commit_record_id
            or manifest.scope_contract_id != attempt.scope_contract_id
            or manifest.parent_release_id != attempt.parent_release_id
            or packet.scope_contract.scope_contract_id != attempt.scope_contract_id
        ):
            raise ExpertValidationError(
                "active attempt differs from its immutable candidate closure"
            )
        observed_parent_release_id = self.current_release_provider.current_release_id(
            packet.scope_contract.scope_id
        )
        if observed_parent_release_id is not None:
            require_content_id(
                observed_parent_release_id,
                "observed_parent_release_id",
            )
        if observed_parent_release_id == attempt.parent_release_id:
            raise ExpertValidationError(
                "parent authority has not changed for the active attempt"
            )
        dependencies = {
            attempt.validation_attempt_id,
            state.validation_state_id,
            attempt.candidate_id,
            attempt.scope_contract_id,
            attempt.parent_release_id,
        }
        if observed_parent_release_id is not None:
            dependencies.add(observed_parent_release_id)
        invalidation = ExpertValidationAuthorityInvalidation.mint(
            kind=(ExpertValidationAuthorityInvalidationKind.PARENT_RELEASE_CHANGED),
            validation_attempt_id=attempt.validation_attempt_id,
            authorization_state_id=state.validation_state_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            scope_contract_id=attempt.scope_contract_id,
            expected_parent_release_id=attempt.parent_release_id,
            observed_parent_release_id=observed_parent_release_id,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        target_state = ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            predecessor_state_id=state.validation_state_id,
            promotion_state=ExpertPromotionState.FAILED,
            accepted_stage_results=state.accepted_stage_results,
            next_stage=None,
            review_assertion_ids=state.review_assertion_ids,
            terminal_evidence_ids=(invalidation.authority_invalidation_id,),
            transition_evidence_id=invalidation.authority_invalidation_id,
            reason="validation_parent_release_changed",
        )
        return ExpertValidationAuthorityInvalidationResult(
            invalidation=invalidation,
            state=target_state,
        )

    def validate_source_replay_request(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[ExpertEvaluatorResultRecord, ...],
        request: ExpertSourceReplayExecutionRequest,
    ) -> None:
        validate_source_replay_request_authority_shape(
            state=state,
            attempt=attempt,
            request=request,
            settings=self.settings,
        )
        stored = self.candidate_store.read(attempt.candidate_id)
        manifest = stored.closure.manifest
        packet = stored.closure.trigger_packet
        parent_receipt = packet.parent_tree_receipt
        current_parent = self.current_release_provider.current_release_id(
            packet.scope_contract.scope_id
        )
        if current_parent is not None:
            require_content_id(current_parent, "current source replay parent release")
        if (
            manifest.candidate_id != attempt.candidate_id
            or manifest.candidate_tree_hash != attempt.candidate_tree_hash
            or stored.commit_record.commit_record_id
            != attempt.candidate_commit_record_id
            or request.candidate_source_tree_manifest_id
            != stored.closure.candidate_tree.source_tree_manifest_id
            or manifest.scope_contract_id != attempt.scope_contract_id
            or manifest.parent_release_id != attempt.parent_release_id
            or parent_receipt is None
            or request.parent_tree_receipt_id != parent_receipt.parent_tree_receipt_id
            or request.parent_source_extraction_receipt_id
            != parent_receipt.source_extraction_receipt.extraction_receipt_id
            or request.parent_tree_hash != parent_receipt.parent_tree_hash
            or current_parent != attempt.parent_release_id
        ):
            raise ExpertValidationError(
                "source replay request differs from current validation authority"
            )
        request_cases = {case.episode_id: case for case in request.cases}
        selection = attempt.source_replay_selection
        if selection is None:
            raise ExpertValidationError(
                "source replay request has no selected adapter authority"
            )
        packet_episodes = {episode.episode_id: episode for episode in packet.episodes}
        for episode_id, request_case in request_cases.items():
            episode = packet_episodes.get(episode_id)
            if episode is None:
                raise ExpertValidationError(
                    "source replay request episode is absent from candidate evidence"
                )
            terminal_attempt = episode.attempts[episode.terminal_attempt_revision]
            context = episode.task_context_binding
            environment = episode.artifact_environment
            if (
                request_case.source_bundle_id != episode.source_bundle_id
                or request_case.source_node_id != episode.source["node_id"]
                or request_case.source_execution_revision
                != episode.terminal_attempt_revision
                or request_case.source_evaluation_fingerprint_ids
                != tuple(
                    fingerprint.evaluation_fingerprint_id
                    for fingerprint in terminal_attempt.evaluation_fingerprints
                )
                or request_case.task_context_binding_id
                != context.task_context_binding_id
                or request_case.source_expert_base_release_id
                != environment.expert_base_release_id
                or request_case.starting_artifact_content_ids
                != tuple(sorted(environment.starting_artifact_content_ids.values()))
                or request_case.task_adapter_manifest_id
                != environment.task_adapter_manifest_id
                or request_case.verification_receipt_id
                != environment.task_adapter_verification_receipt_id
            ):
                raise ExpertValidationError(
                    "source replay request differs from exact candidate evidence"
                )
        for pin in selection.source_adapter_pins:
            adapter = self.task_adapter_provider.resolve_exact(
                task_adapter_manifest_id=pin.task_adapter_manifest_id,
                verification_receipt_id=pin.verification_receipt_id,
            )
            if (
                not isinstance(adapter, VerifiedTaskAdapter)
                or adapter.manifest.scope_contract_id != pin.scope_contract_id
                or adapter.manifest.task_family_id != pin.task_family_id
                or adapter.manifest.task_adapter_id != pin.task_adapter_id
                or adapter.manifest.task_adapter_manifest_id
                != pin.task_adapter_manifest_id
                or adapter.verification_receipt.verification_receipt_id
                != pin.verification_receipt_id
                or any(
                    request_cases[episode_id].task_adapter_source_tree_hash
                    != adapter.manifest.tree_hash
                    or request_cases[episode_id].task_evaluator_digest
                    != tree_or_blob_digest(
                        adapter.manifest.task_evaluator.to_json_bytes()
                    )
                    or request_cases[episode_id].task_adapter_runtime_digest
                    != tree_or_blob_digest(adapter.manifest.runtime.to_json_bytes())
                    or request_cases[episode_id].task_adapter_context_binding_digest
                    != tree_or_blob_digest(
                        adapter.manifest.context_binding.to_json_bytes()
                    )
                    or request_cases[episode_id].task_adapter_dependency_ids
                    != adapter.dependency_ids
                    for episode_id in pin.episode_ids
                )
            ):
                raise ExpertValidationError(
                    "historical source replay adapter differs from its exact pin"
                )
        self._validate_accepted_history(state, attempt, accepted_results)

    def advance_evaluator_stage(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord,
            ...,
        ],
        result: ExpertEvaluatorResultRecord,
    ) -> ExpertCandidateValidationState:
        if (
            state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.candidate_id != attempt.candidate_id
            or state.candidate_tree_hash != attempt.candidate_tree_hash
        ):
            raise ExpertValidationError("only the matching active attempt may advance")
        policy = self.settings.policy.validation_policy()
        if (
            attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "active attempt differs from reducer configuration"
            )
        self._validate_accepted_history(state, attempt, accepted_results)
        accepted_count = len(state.accepted_stage_results)
        if accepted_count >= len(attempt.required_stages):
            raise ExpertValidationError("validation attempt has no remaining stage")
        expected_stage = attempt.required_stages[accepted_count]
        run = result.evaluator_run
        envelope = result.attestation_envelope
        if expected_stage in {
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            ExpertValidationStage.AUTOMATED_REVIEW,
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        } or run.stage in {
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            ExpertValidationStage.AUTOMATED_REVIEW,
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        }:
            raise ExpertValidationError(
                "typed stage cannot consume a generic evaluator result"
            )
        if state.next_stage is not expected_stage or run.stage is not expected_stage:
            raise ExpertValidationError("evaluator result is out of order")
        evaluator = ExpertEvaluatorRunBuilder(self.settings)._evaluator(expected_stage)
        if (
            run.validation_attempt_id != attempt.validation_attempt_id
            or run.candidate_id != attempt.candidate_id
            or run.candidate_tree_hash != attempt.candidate_tree_hash
            or run.evaluator_id != evaluator.evaluator_id
            or run.evaluator_role != evaluator.evaluator_role
            or run.evaluator_version != evaluator.evaluator_version
        ):
            raise ExpertValidationError(
                "evaluator result differs from the active configured stage"
            )
        self._validate_result_closure(attempt, result)
        self.attestation_verifier.verify(envelope)
        evidence = ExpertAcceptedStageResultRef(
            stage=run.stage,
            stage_result_record_id=result.evaluator_result_record_id,
        )
        if run.outcome is not ExpertEvaluatorOutcome.PASSED:
            terminal_ids = tuple(
                sorted(
                    {
                        run.evaluator_run_id,
                        envelope.attestation.evaluator_attestation_id,
                    }
                )
            )
            return ExpertCandidateValidationState.mint(
                validation_attempt_id=attempt.validation_attempt_id,
                candidate_id=attempt.candidate_id,
                candidate_tree_hash=attempt.candidate_tree_hash,
                predecessor_state_id=state.validation_state_id,
                promotion_state=ExpertPromotionState.FAILED,
                accepted_stage_results=state.accepted_stage_results,
                next_stage=None,
                review_assertion_ids=state.review_assertion_ids,
                terminal_evidence_ids=terminal_ids,
                transition_evidence_id=(envelope.attestation.evaluator_attestation_id),
                reason=f"stage_{run.stage.value}_{run.outcome.value}",
            )
        accepted = (*state.accepted_stage_results, evidence)
        next_position = len(accepted)
        if next_position >= len(attempt.required_stages):
            raise ExpertValidationError(
                "final promotion stages require a typed promotion decision"
            )
        return ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            predecessor_state_id=state.validation_state_id,
            promotion_state=ExpertPromotionState.VALIDATING,
            accepted_stage_results=accepted,
            next_stage=attempt.required_stages[next_position],
            review_assertion_ids=state.review_assertion_ids,
            terminal_evidence_ids=(),
            transition_evidence_id=(envelope.attestation.evaluator_attestation_id),
            reason=f"stage_{run.stage.value}_passed",
        )

    def advance_source_replay_stage(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord,
            ...,
        ],
        result: ExpertSourceReplayStageResultRecord,
    ) -> ExpertCandidateValidationState:
        if (
            state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.candidate_id != attempt.candidate_id
            or state.candidate_tree_hash != attempt.candidate_tree_hash
        ):
            raise ExpertValidationError("only the matching active attempt may advance")
        policy = self.settings.policy.validation_policy()
        if (
            attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "active attempt differs from reducer configuration"
            )
        self._validate_accepted_history(state, attempt, accepted_results)
        accepted_count = len(state.accepted_stage_results)
        if accepted_count >= len(attempt.required_stages):
            raise ExpertValidationError("validation attempt has no remaining stage")
        expected_stage = attempt.required_stages[accepted_count]
        if type(result) is not ExpertSourceReplayStageResultRecord:
            raise ExpertValidationError(
                "source replay stage requires its typed result record"
            )
        fence = result.publication_authority_fence
        if (
            state.next_stage is not ExpertValidationStage.SOURCE_RUN_REPLAY
            or expected_stage is not ExpertValidationStage.SOURCE_RUN_REPLAY
            or result.validation_attempt_id != attempt.validation_attempt_id
            or result.authorization_state_id != state.validation_state_id
            or result.candidate_id != attempt.candidate_id
            or result.candidate_tree_hash != attempt.candidate_tree_hash
            or result.validation_policy_id != attempt.validation_policy_id
            or result.configuration_fingerprint != attempt.configuration_fingerprint
            or fence.scope_contract_id != attempt.scope_contract_id
            or fence.expected_parent_release_id != attempt.parent_release_id
        ):
            raise ExpertValidationError(
                "source replay result differs from the active configured stage"
            )
        if result.outcome is ExpertEvaluatorOutcome.CANDIDATE_FAILED:
            return ExpertCandidateValidationState.mint(
                validation_attempt_id=attempt.validation_attempt_id,
                candidate_id=attempt.candidate_id,
                candidate_tree_hash=attempt.candidate_tree_hash,
                predecessor_state_id=state.validation_state_id,
                promotion_state=ExpertPromotionState.FAILED,
                accepted_stage_results=state.accepted_stage_results,
                next_stage=None,
                review_assertion_ids=state.review_assertion_ids,
                terminal_evidence_ids=(result.stage_result_record_id,),
                transition_evidence_id=result.stage_result_record_id,
                reason="stage_source_run_replay_candidate_failed",
            )
        if result.outcome is not ExpertEvaluatorOutcome.PASSED:
            raise ExpertValidationError("source replay result outcome is unsupported")
        accepted = (
            *state.accepted_stage_results,
            ExpertAcceptedStageResultRef(
                stage=ExpertValidationStage.SOURCE_RUN_REPLAY,
                stage_result_record_id=result.stage_result_record_id,
            ),
        )
        next_position = len(accepted)
        if next_position >= len(attempt.required_stages):
            raise ExpertValidationError(
                "final promotion stages require a typed promotion decision"
            )
        return ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            predecessor_state_id=state.validation_state_id,
            promotion_state=ExpertPromotionState.VALIDATING,
            accepted_stage_results=accepted,
            next_stage=attempt.required_stages[next_position],
            review_assertion_ids=state.review_assertion_ids,
            terminal_evidence_ids=(),
            transition_evidence_id=result.stage_result_record_id,
            reason="stage_source_run_replay_passed",
        )

    def advance_automated_review_stage(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord,
            ...,
        ],
        result: ExpertAutomatedReviewStageResultRecord,
    ) -> ExpertCandidateValidationState:
        if (
            state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.candidate_id != attempt.candidate_id
            or state.candidate_tree_hash != attempt.candidate_tree_hash
        ):
            raise ExpertValidationError("only the matching active attempt may advance")
        policy = self.settings.policy.validation_policy()
        if (
            attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "active attempt differs from reducer configuration"
            )
        self._validate_accepted_history(state, attempt, accepted_results)
        accepted_count = len(state.accepted_stage_results)
        if accepted_count >= len(attempt.required_stages):
            raise ExpertValidationError("validation attempt has no remaining stage")
        expected_stage = attempt.required_stages[accepted_count]
        if (
            type(result) is not ExpertAutomatedReviewStageResultRecord
            or state.next_stage is not ExpertValidationStage.AUTOMATED_REVIEW
            or expected_stage is not ExpertValidationStage.AUTOMATED_REVIEW
            or result.validation_attempt_id != attempt.validation_attempt_id
            or result.authorization_state_id != state.validation_state_id
            or result.candidate_id != attempt.candidate_id
            or result.candidate_tree_hash != attempt.candidate_tree_hash
            or result.scope_contract_id != attempt.scope_contract_id
            or result.parent_release_id != attempt.parent_release_id
            or result.validation_policy_id != attempt.validation_policy_id
            or result.configuration_fingerprint != attempt.configuration_fingerprint
        ):
            raise ExpertValidationError(
                "automated review result differs from the active configured stage"
            )
        assertion_ids = result.assertion_ids
        if result.outcome is ExpertAutomatedReviewOutcome.REJECTED:
            return ExpertCandidateValidationState.mint(
                validation_attempt_id=attempt.validation_attempt_id,
                candidate_id=attempt.candidate_id,
                candidate_tree_hash=attempt.candidate_tree_hash,
                predecessor_state_id=state.validation_state_id,
                promotion_state=ExpertPromotionState.FAILED,
                accepted_stage_results=state.accepted_stage_results,
                next_stage=None,
                review_assertion_ids=assertion_ids,
                terminal_evidence_ids=(result.stage_result_record_id,),
                transition_evidence_id=result.stage_result_record_id,
                reason="stage_automated_review_rejected",
            )
        if result.outcome is ExpertAutomatedReviewOutcome.DISPUTED:
            if len(assertion_ids) < 2:
                raise ExpertValidationError(
                    "disputed automated review requires multiple assertions"
                )
            return ExpertCandidateValidationState.mint(
                validation_attempt_id=attempt.validation_attempt_id,
                candidate_id=attempt.candidate_id,
                candidate_tree_hash=attempt.candidate_tree_hash,
                predecessor_state_id=state.validation_state_id,
                promotion_state=ExpertPromotionState.DISPUTED,
                accepted_stage_results=state.accepted_stage_results,
                next_stage=None,
                review_assertion_ids=assertion_ids,
                terminal_evidence_ids=(result.stage_result_record_id,),
                transition_evidence_id=result.stage_result_record_id,
                reason="stage_automated_review_disputed",
            )
        if result.outcome is not ExpertAutomatedReviewOutcome.PASSED:
            raise ExpertValidationError("automated review outcome is unsupported")
        accepted = (
            *state.accepted_stage_results,
            ExpertAcceptedStageResultRef(
                stage=ExpertValidationStage.AUTOMATED_REVIEW,
                stage_result_record_id=result.stage_result_record_id,
            ),
        )
        next_position = len(accepted)
        if next_position >= len(attempt.required_stages):
            raise ExpertValidationError(
                "final promotion stages require a typed promotion decision"
            )
        return ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            predecessor_state_id=state.validation_state_id,
            promotion_state=ExpertPromotionState.VALIDATING,
            accepted_stage_results=accepted,
            next_stage=attempt.required_stages[next_position],
            review_assertion_ids=assertion_ids,
            terminal_evidence_ids=(),
            transition_evidence_id=result.stage_result_record_id,
            reason="stage_automated_review_passed",
        )

    def _validate_result_closure(
        self,
        attempt: ExpertValidationAttempt,
        result: ExpertEvaluatorResultRecord,
    ) -> None:
        run = result.evaluator_run
        if run.stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            raise ExpertValidationError(
                "source replay requires a typed execution receipt"
            )
        if run.stage in {
            ExpertValidationStage.AUTOMATED_REVIEW,
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        }:
            raise ExpertValidationError(
                f"stage {run.stage.value} requires a typed stage path"
            )
        envelope = result.attestation_envelope
        attestation = envelope.attestation
        required_inputs = _expert_evaluator_base_input_ids(attempt)
        if not required_inputs.issubset(run.exact_input_ids):
            raise ExpertValidationError("evaluator input closure is incomplete")
        decoded_outputs = {
            path: base64.b64decode(payload, validate=True)
            for path, payload in run.output_payloads_base64.items()
        }
        if (
            attestation.evaluator_run_id != run.evaluator_run_id
            or attestation.issuer_id != run.evaluator_id
            or attestation.predicate_digest != tree_or_blob_digest(run.to_json_bytes())
        ):
            raise ExpertValidationError("evaluator attestation does not bind the run")
        expected_trust_root = (
            self.settings.policy.sealed_canary_trust_root
            if run.stage is ExpertValidationStage.SEALED_CANARY
            else None
        )
        if attestation.trust_root_id != expected_trust_root:
            raise ExpertValidationError("evaluator attestation trust root is invalid")
        output_bytes = sum(len(payload) for payload in decoded_outputs.values())
        if (
            len(run.output_payloads_base64) > self.settings.policy.artifact_entry_limit
            or output_bytes > self.settings.policy.artifact_byte_limit
        ):
            raise ExpertValidationError("evaluator output limits are exceeded")

    def _validate_accepted_history(
        self,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord,
            ...,
        ],
    ) -> None:
        if len(accepted_results) != len(state.accepted_stage_results):
            raise ExpertValidationError("accepted evaluator history is incomplete")
        if len(accepted_results) >= len(attempt.required_stages):
            raise ExpertValidationError(
                "accepted evaluator history exhausts the stage plan"
            )
        for position, (evidence, accepted_result) in enumerate(
            zip(state.accepted_stage_results, accepted_results)
        ):
            expected_stage = attempt.required_stages[position]
            if expected_stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
                if (
                    type(accepted_result) is not ExpertSourceReplayStageResultRecord
                    or evidence.stage is not expected_stage
                    or evidence.stage_result_record_id
                    != accepted_result.stage_result_record_id
                    or accepted_result.outcome is not ExpertEvaluatorOutcome.PASSED
                    or accepted_result.validation_attempt_id
                    != attempt.validation_attempt_id
                    or accepted_result.candidate_id != attempt.candidate_id
                    or accepted_result.candidate_tree_hash
                    != attempt.candidate_tree_hash
                    or accepted_result.validation_policy_id
                    != attempt.validation_policy_id
                    or accepted_result.configuration_fingerprint
                    != attempt.configuration_fingerprint
                    or accepted_result.publication_authority_fence.scope_contract_id
                    != attempt.scope_contract_id
                    or accepted_result.publication_authority_fence.expected_parent_release_id
                    != attempt.parent_release_id
                ):
                    raise ExpertValidationError(
                        "accepted source replay differs from the stage prefix"
                    )
                continue
            if expected_stage is ExpertValidationStage.AUTOMATED_REVIEW:
                if (
                    type(accepted_result) is not ExpertAutomatedReviewStageResultRecord
                    or evidence.stage is not expected_stage
                    or evidence.stage_result_record_id
                    != accepted_result.stage_result_record_id
                    or accepted_result.outcome
                    is not ExpertAutomatedReviewOutcome.PASSED
                    or accepted_result.validation_attempt_id
                    != attempt.validation_attempt_id
                    or accepted_result.candidate_id != attempt.candidate_id
                    or accepted_result.candidate_tree_hash
                    != attempt.candidate_tree_hash
                    or accepted_result.scope_contract_id != attempt.scope_contract_id
                    or accepted_result.parent_release_id != attempt.parent_release_id
                    or accepted_result.validation_policy_id
                    != attempt.validation_policy_id
                    or accepted_result.configuration_fingerprint
                    != attempt.configuration_fingerprint
                    or state.review_assertion_ids != accepted_result.assertion_ids
                ):
                    raise ExpertValidationError(
                        "accepted automated review differs from the stage prefix"
                    )
                continue
            if type(accepted_result) is not ExpertEvaluatorResultRecord:
                raise ExpertValidationError(
                    "accepted evaluator history uses another stage result type"
                )
            run = accepted_result.evaluator_run
            envelope = accepted_result.attestation_envelope
            evaluator = ExpertEvaluatorRunBuilder(self.settings)._evaluator(
                expected_stage
            )
            if (
                evidence.stage is not expected_stage
                or evidence.stage_result_record_id
                != accepted_result.evaluator_result_record_id
                or run.outcome is not ExpertEvaluatorOutcome.PASSED
                or run.stage is not expected_stage
                or run.validation_attempt_id != attempt.validation_attempt_id
                or run.candidate_id != attempt.candidate_id
                or run.candidate_tree_hash != attempt.candidate_tree_hash
                or run.evaluator_id != evaluator.evaluator_id
                or run.evaluator_role != evaluator.evaluator_role
                or run.evaluator_version != evaluator.evaluator_version
            ):
                raise ExpertValidationError(
                    "accepted evaluator history differs from the stage prefix"
                )
            self._validate_result_closure(attempt, accepted_result)
            self.attestation_verifier.verify(envelope)
        expected_next_stage = attempt.required_stages[len(accepted_results)]
        if state.next_stage is not expected_next_stage:
            raise ExpertValidationError(
                "validation state next stage differs from accepted history"
            )

    @staticmethod
    def _validate_predecessors(
        candidate_id: str,
        predecessor_attempt: ExpertValidationAttempt | None,
        predecessor_state: ExpertCandidateValidationState | None,
    ) -> None:
        if predecessor_state is None:
            if predecessor_attempt is not None:
                raise ExpertValidationError(
                    "attempt predecessor requires a state predecessor"
                )
            return
        if predecessor_state.candidate_id != candidate_id:
            raise ExpertValidationError(
                "validation predecessor belongs to another candidate"
            )
        if predecessor_state.promotion_state is ExpertPromotionState.INELIGIBLE:
            if predecessor_attempt is not None and (
                predecessor_attempt.candidate_id != candidate_id
                or predecessor_attempt.candidate_tree_hash
                != predecessor_state.candidate_tree_hash
            ):
                raise ExpertValidationError(
                    "historical validation attempt belongs to another candidate"
                )
            return
        if predecessor_attempt is None:
            raise ExpertValidationError(
                "validation predecessor state requires a historical attempt"
            )
        if (
            predecessor_attempt.candidate_id != candidate_id
            or predecessor_state.validation_attempt_id
            != predecessor_attempt.validation_attempt_id
            or predecessor_state.promotion_state
            not in {
                ExpertPromotionState.FAILED,
                ExpertPromotionState.DISPUTED,
                ExpertPromotionState.PARETO_RETAINED,
            }
        ):
            raise ExpertValidationError(
                "new validation attempt requires one matching terminal predecessor"
            )
