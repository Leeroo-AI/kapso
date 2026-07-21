"""Deterministic expert-candidate enrollment and ordered validation reduction."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping, Protocol

from kapso.cross_run.canonical import (
    content_id,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertEvaluatorAttestation,
    ExpertEvaluatorAttestationEnvelope,
    ExpertEvaluatorEvidenceRef,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorRun,
    ExpertPromotionState,
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
    TaskAdapterPackagePin,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.replay import _derive_expert_source_replay_selection
from kapso.cross_run.settings import (
    ExpertEvaluatorSettings,
    ExpertValidationPolicy,
    ExpertValidationSettings,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)


class ExpertValidationError(ValueError):
    """An enrollment, evaluator result, or state transition is invalid."""


@dataclass(frozen=True)
class ExpertEligibilityResult:
    decision: ExpertCandidateEligibilityDecision
    policy: ExpertValidationPolicy


@dataclass(frozen=True)
class ExpertEvaluatorResult:
    evaluator_run: ExpertEvaluatorRun
    attestation_envelope: ExpertEvaluatorAttestationEnvelope


@dataclass(frozen=True)
class ExpertValidationStart:
    attempt: ExpertValidationAttempt | None
    state: ExpertCandidateValidationState


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
        if eligible and ExpertValidationStage.SOURCE_RUN_REPLAY in stage_plan:
            replay_result = _derive_expert_source_replay_selection(
                stored_candidate=stored,
                settings=self.settings,
            )
            source_replay_selection = replay_result.selection
            if source_replay_selection is None:
                eligible = False
                reason_code = replay_result.reason_code
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
        ordering = tuple(
            (adapter.manifest.task_family_id, adapter.manifest.task_adapter_id)
            for adapter in task_adapters
        )
        if not task_adapters or len(ordering) != len(set(ordering)):
            raise ExpertValidationError("task adapters must be non-empty and unique")
        manifest = stored.closure.manifest
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
            ):
                raise ExpertValidationError(
                    "task adapter does not match candidate scope and trigger"
                )
            binding_id = content_id(
                "task-adapter-binding",
                {
                    "task_family_id": adapter.task_family_id,
                    "task_adapter_id": adapter.task_adapter_id,
                },
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
    ) -> ExpertEvaluatorResult:
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
        evaluator = self._evaluator(stage)
        self._validate_outputs(output_payloads)
        for input_id in exact_additional_input_ids:
            require_content_id(input_id, "exact_additional_input_ids")
        if exact_additional_input_ids != tuple(sorted(set(exact_additional_input_ids))):
            raise ExpertValidationError(
                "additional evaluator inputs must be sorted and unique"
            )
        stages_requiring_external_evidence = {
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            ExpertValidationStage.DEVELOPMENT_ANCHORS,
            ExpertValidationStage.CROSS_FAMILY_TRANSFER,
            ExpertValidationStage.SEALED_CANARY,
            ExpertValidationStage.RELEASE_MATRIX,
        }
        if (
            stage in stages_requiring_external_evidence
            and not exact_additional_input_ids
        ):
            raise ExpertValidationError(
                f"stage {stage.value} requires exact external evidence inputs"
            )
        exact_input_ids = {
            attempt.validation_attempt_id,
            attempt.candidate_id,
            attempt.candidate_commit_record_id,
            attempt.scope_contract_id,
            attempt.eligibility_decision_id,
            attempt.validation_policy_id,
            *(pin.task_adapter_manifest_id for pin in attempt.task_adapter_pins),
            *(pin.verification_receipt_id for pin in attempt.task_adapter_pins),
            *attempt.eligibility_dependency_ids,
            *exact_additional_input_ids,
        }
        if attempt.parent_release_id is not None:
            exact_input_ids.add(attempt.parent_release_id)
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
        return ExpertEvaluatorResult(
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
                accepted_evaluator_evidence=(),
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
            accepted_evaluator_evidence=(),
            next_stage=attempt.required_stages[0],
            review_assertion_ids=(),
            terminal_evidence_ids=(),
            transition_evidence_id=eligibility.decision.eligibility_decision_id,
            reason="validation_attempt_started",
        )
        return ExpertValidationStart(attempt=attempt, state=state)

    def advance(
        self,
        *,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[ExpertEvaluatorResult, ...],
        result: ExpertEvaluatorResult,
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
        accepted_count = len(state.accepted_evaluator_evidence)
        if accepted_count >= len(attempt.required_stages):
            raise ExpertValidationError("validation attempt has no remaining stage")
        expected_stage = attempt.required_stages[accepted_count]
        run = result.evaluator_run
        envelope = result.attestation_envelope
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
        evidence = ExpertEvaluatorEvidenceRef(
            evaluator_run_id=run.evaluator_run_id,
            evaluator_attestation_id=(envelope.attestation.evaluator_attestation_id),
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
                accepted_evaluator_evidence=state.accepted_evaluator_evidence,
                next_stage=None,
                review_assertion_ids=state.review_assertion_ids,
                terminal_evidence_ids=terminal_ids,
                transition_evidence_id=(envelope.attestation.evaluator_attestation_id),
                reason=f"stage_{run.stage.value}_{run.outcome.value}",
            )
        accepted = (*state.accepted_evaluator_evidence, evidence)
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
            accepted_evaluator_evidence=accepted,
            next_stage=attempt.required_stages[next_position],
            review_assertion_ids=state.review_assertion_ids,
            terminal_evidence_ids=(),
            transition_evidence_id=(envelope.attestation.evaluator_attestation_id),
            reason=f"stage_{run.stage.value}_passed",
        )

    def _validate_result_closure(
        self,
        attempt: ExpertValidationAttempt,
        result: ExpertEvaluatorResult,
    ) -> None:
        run = result.evaluator_run
        if run.stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            raise ExpertValidationError(
                "source replay requires a typed execution receipt"
            )
        envelope = result.attestation_envelope
        attestation = envelope.attestation
        required_inputs = {
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
            required_inputs.add(attempt.parent_release_id)
        if not required_inputs.issubset(run.exact_input_ids):
            raise ExpertValidationError("evaluator input closure is incomplete")
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
        output_bytes = sum(
            len(base64.b64decode(payload, validate=True))
            for payload in run.output_payloads_base64.values()
        )
        if (
            len(run.output_payloads_base64) > self.settings.policy.artifact_entry_limit
            or output_bytes > self.settings.policy.artifact_byte_limit
        ):
            raise ExpertValidationError("evaluator output limits are exceeded")

    def _validate_accepted_history(
        self,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt,
        accepted_results: tuple[ExpertEvaluatorResult, ...],
    ) -> None:
        if len(accepted_results) != len(state.accepted_evaluator_evidence):
            raise ExpertValidationError("accepted evaluator history is incomplete")
        if len(accepted_results) >= len(attempt.required_stages):
            raise ExpertValidationError(
                "accepted evaluator history exhausts the stage plan"
            )
        for position, (evidence, accepted_result) in enumerate(
            zip(state.accepted_evaluator_evidence, accepted_results)
        ):
            run = accepted_result.evaluator_run
            envelope = accepted_result.attestation_envelope
            expected_stage = attempt.required_stages[position]
            evaluator = ExpertEvaluatorRunBuilder(self.settings)._evaluator(
                expected_stage
            )
            if (
                evidence.evaluator_run_id != run.evaluator_run_id
                or evidence.evaluator_attestation_id
                != envelope.attestation.evaluator_attestation_id
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
