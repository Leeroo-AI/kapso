"""Durable fresh-authority evidence for source-replay stage publication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayPairedComparisonReceipt,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayStageDecision,
)
from kapso.cross_run.expert.replay_decision import decide_expert_source_replay_stage

if TYPE_CHECKING:
    from kapso.cross_run.expert.replay_execution_store import (
        SourceReplayExecutionJournalEvent,
    )
    from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest


class ExpertSourceReplayPublicationError(ValueError):
    """Source replay lacks exact current authority for validation publication."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertSourceReplayPublicationError(f"{name} uses the wrong namespace")


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertSourceReplayPublicationError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class SourceReplayDecisionPublicationFence(StrictContract):
    """Fresh external authority observed immediately before validation CAS."""

    fence_id: str
    reservation_id: str
    execution_request_id: str
    authorization_transition_id: str
    authorization_state_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    scope_id: str
    scope_contract_id: str
    expected_current_release_id: str
    validation_policy_id: str
    configuration_fingerprint: str
    paired_comparison_receipt_id: str
    source_replay_stage_decision_id: str
    outcome: ExpertEvaluatorOutcome
    current_release_observation: SourceReplayCurrentReleaseObservation
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...]
    security_denylist_observation: SecurityDenylistObservation

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-decision-publication-fence"
    IDENTITY_FIELD: ClassVar[str] = "fence_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "source replay publication scope_id")
        for value, namespace, name in (
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "source replay publication reservation_id",
            ),
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "source replay publication execution_request_id",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "source replay publication authorization_transition_id",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "source replay publication authorization_state_id",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "source replay publication validation_attempt_id",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "source replay publication candidate_id",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "source replay publication scope_contract_id",
            ),
            (
                self.expected_current_release_id,
                "expert-base-release",
                "source replay publication expected_current_release_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "source replay publication validation_policy_id",
            ),
            (
                self.paired_comparison_receipt_id,
                "expert-source-replay-paired-comparison",
                "source replay publication paired_comparison_receipt_id",
            ),
            (
                self.source_replay_stage_decision_id,
                "expert-source-replay-stage-decision",
                "source replay publication source_replay_stage_decision_id",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        for value, name in (
            (self.candidate_tree_hash, "source replay publication candidate tree"),
            (
                self.configuration_fingerprint,
                "source replay publication configuration fingerprint",
            ),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ExpertSourceReplayPublicationError(f"{name} is invalid")
        if self.outcome not in {
            ExpertEvaluatorOutcome.PASSED,
            ExpertEvaluatorOutcome.CANDIDATE_FAILED,
        }:
            raise ExpertSourceReplayPublicationError(
                "source replay publication outcome is unsupported"
            )
        current = self.current_release_observation
        if (
            current.scope_id != self.scope_id
            or current.release_id != self.expected_current_release_id
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay publication release differs from expected CURRENT"
            )
        observation_ids = tuple(
            observation.observation_id
            for observation in self.task_adapter_trust_observations
        )
        if not observation_ids or observation_ids != tuple(
            sorted(set(observation_ids))
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay publication adapter observations must be sorted and unique"
            )
        denylist = self.security_denylist_observation
        if (
            denylist.scope_id != self.scope_id
            or denylist.scope_contract_id != self.scope_contract_id
            or denylist.matched_revocations
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay publication denylist authority rejected the fence"
            )
        required_subjects = {
            self.reservation_id,
            self.execution_request_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.scope_contract_id,
            self.expected_current_release_id,
            self.validation_policy_id,
            self.paired_comparison_receipt_id,
            self.source_replay_stage_decision_id,
            current.observation_id,
            current.publication_id,
            *current.validation_closure_ids,
        }
        for observation in self.task_adapter_trust_observations:
            required_subjects.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        if not required_subjects.issubset(denylist.checked_subject_ids):
            raise ExpertSourceReplayPublicationError(
                "source replay publication fence omits mandatory security subjects"
            )

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        return self.security_denylist_observation.checked_subject_ids

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        current = self.current_release_observation
        denylist = self.security_denylist_observation
        dependencies = {
            self.reservation_id,
            self.execution_request_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.scope_contract_id,
            self.expected_current_release_id,
            self.validation_policy_id,
            self.paired_comparison_receipt_id,
            self.source_replay_stage_decision_id,
            current.observation_id,
            current.publication_id,
            *current.validation_closure_ids,
            denylist.observation_id,
            denylist.snapshot_id,
            denylist.publication_id,
            *denylist.checked_subject_ids,
        }
        for observation in self.task_adapter_trust_observations:
            dependencies.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        return tuple(sorted(dependencies))


@dataclass(frozen=True)
class ExpertSourceReplayStageResultRecord(StrictContract):
    """Self-contained factual, policy, and fresh-authority source-stage result."""

    stage_result_record_id: str
    validation_attempt_id: str
    authorization_transition_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    execution_request_id: str
    reservation_id: str
    validation_policy_id: str
    configuration_fingerprint: str
    paired_comparison_receipt: ExpertSourceReplayPairedComparisonReceipt
    stage_decision: ExpertSourceReplayStageDecision
    publication_authority_fence: SourceReplayDecisionPublicationFence
    outcome: ExpertEvaluatorOutcome
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-stage-result"
    IDENTITY_FIELD: ClassVar[str] = "stage_result_record_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "source replay result validation_attempt_id",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "source replay result authorization_transition_id",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "source replay result authorization_state_id",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "source replay result candidate_id",
            ),
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "source replay result execution_request_id",
            ),
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "source replay result reservation_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "source replay result validation_policy_id",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        for value, name in (
            (self.candidate_tree_hash, "source replay result candidate tree"),
            (
                self.configuration_fingerprint,
                "source replay result configuration fingerprint",
            ),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ExpertSourceReplayPublicationError(f"{name} is invalid")
        receipt = self.paired_comparison_receipt
        decision = self.stage_decision
        fence = self.publication_authority_fence
        if (
            self.outcome
            not in {
                ExpertEvaluatorOutcome.PASSED,
                ExpertEvaluatorOutcome.CANDIDATE_FAILED,
            }
            or receipt.reservation_id != self.reservation_id
            or receipt.execution_request_id != self.execution_request_id
            or decision.paired_comparison_receipt_id
            != receipt.paired_comparison_receipt_id
            or decision.validation_policy_id != self.validation_policy_id
            or decision.outcome is not self.outcome
            or fence.reservation_id != self.reservation_id
            or fence.execution_request_id != self.execution_request_id
            or fence.authorization_transition_id != self.authorization_transition_id
            or fence.authorization_state_id != self.authorization_state_id
            or fence.validation_attempt_id != self.validation_attempt_id
            or fence.candidate_id != self.candidate_id
            or fence.candidate_tree_hash != self.candidate_tree_hash
            or fence.validation_policy_id != self.validation_policy_id
            or fence.configuration_fingerprint != self.configuration_fingerprint
            or fence.paired_comparison_receipt_id
            != receipt.paired_comparison_receipt_id
            or fence.source_replay_stage_decision_id
            != decision.source_replay_stage_decision_id
            or fence.outcome is not self.outcome
            or not {
                receipt.paired_comparison_receipt_id,
                *receipt.exact_dependency_ids,
                decision.source_replay_stage_decision_id,
                *decision.exact_dependency_ids,
            }.issubset(fence.security_subject_ids)
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay stage result closure is inconsistent"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "source replay stage result dependencies",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.execution_request_id,
            self.reservation_id,
            self.validation_policy_id,
            receipt.paired_comparison_receipt_id,
            *receipt.exact_dependency_ids,
            decision.source_replay_stage_decision_id,
            *decision.exact_dependency_ids,
            fence.fence_id,
            *fence.exact_dependency_ids,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertSourceReplayPublicationError(
                "source replay stage result dependency closure is not exact"
            )


def _build_expert_source_replay_stage_result_record(
    *,
    reservation: ExpertSourceReplayExecutionReservation,
    prepared_request: PreparedExpertSourceReplayRequest,
    paired_comparison_receipt: ExpertSourceReplayPairedComparisonReceipt,
    stage_decision: ExpertSourceReplayStageDecision,
    publication_authority_fence: SourceReplayDecisionPublicationFence,
) -> ExpertSourceReplayStageResultRecord:
    """Bind the exact prepared validation authority to its source-stage evidence."""

    request = prepared_request.request
    source_base_release = prepared_request.source_base.release_manifest
    expected_adapter_observations = source_replay_task_adapter_trust_observations(
        prepared_request
    )
    if (
        reservation.execution_request_id != request.execution_request_id
        or reservation.validation_attempt_id != request.validation_attempt_id
        or reservation.authorization_state_id != request.authorization_state_id
        or reservation.candidate_id != request.candidate_id
        or reservation.candidate_tree_hash != request.candidate_tree_hash
        or reservation.expected_current_release_id != request.source_base_release_id
        or reservation.authorization_transition_id
        != publication_authority_fence.authorization_transition_id
        or publication_authority_fence.authorization_state_id
        != request.authorization_state_id
        or publication_authority_fence.validation_attempt_id
        != request.validation_attempt_id
        or publication_authority_fence.scope_id != source_base_release.scope_id
        or publication_authority_fence.scope_contract_id != request.scope_contract_id
        or publication_authority_fence.expected_current_release_id
        != request.source_base_release_id
        or publication_authority_fence.task_adapter_trust_observations
        != expected_adapter_observations
        or stage_decision
        != decide_expert_source_replay_stage(
            paired_comparison_receipt=paired_comparison_receipt,
            prepared_request=prepared_request,
        )
    ):
        raise ExpertSourceReplayPublicationError(
            "source replay result reservation differs from prepared authority"
        )
    dependencies = {
        reservation.validation_attempt_id,
        reservation.authorization_transition_id,
        reservation.authorization_state_id,
        reservation.candidate_id,
        request.execution_request_id,
        reservation.reservation_id,
        request.validation_policy_id,
        paired_comparison_receipt.paired_comparison_receipt_id,
        *paired_comparison_receipt.exact_dependency_ids,
        stage_decision.source_replay_stage_decision_id,
        *stage_decision.exact_dependency_ids,
        publication_authority_fence.fence_id,
        *publication_authority_fence.exact_dependency_ids,
    }
    return ExpertSourceReplayStageResultRecord.mint(
        validation_attempt_id=reservation.validation_attempt_id,
        authorization_transition_id=reservation.authorization_transition_id,
        authorization_state_id=reservation.authorization_state_id,
        candidate_id=reservation.candidate_id,
        candidate_tree_hash=reservation.candidate_tree_hash,
        execution_request_id=request.execution_request_id,
        reservation_id=reservation.reservation_id,
        validation_policy_id=request.validation_policy_id,
        configuration_fingerprint=request.configuration_fingerprint,
        paired_comparison_receipt=paired_comparison_receipt,
        stage_decision=stage_decision,
        publication_authority_fence=publication_authority_fence,
        outcome=stage_decision.outcome,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def source_replay_publication_security_subject_ids(
    *,
    prepared_request: PreparedExpertSourceReplayRequest,
    reservation: ExpertSourceReplayExecutionReservation,
    paired_comparison_receipt: ExpertSourceReplayPairedComparisonReceipt,
    stage_decision: ExpertSourceReplayStageDecision,
    execution_events: tuple[SourceReplayExecutionJournalEvent, ...],
    current_release_observation: SourceReplayCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    """Expand the exact revocation closure checked before final publication."""

    request = prepared_request.request
    candidate = prepared_request.candidate.manifest
    source_base_release = prepared_request.source_base.release_manifest
    if (
        reservation.execution_request_id != request.execution_request_id
        or paired_comparison_receipt.reservation_id != reservation.reservation_id
        or paired_comparison_receipt.execution_request_id
        != request.execution_request_id
        or stage_decision.paired_comparison_receipt_id
        != paired_comparison_receipt.paired_comparison_receipt_id
        or stage_decision.validation_policy_id != request.validation_policy_id
    ):
        raise ExpertSourceReplayPublicationError(
            "source replay publication inputs do not share one reservation"
        )
    event_ids = tuple(event.event_id for event in execution_events)
    if (
        not event_ids
        or event_ids != paired_comparison_receipt.execution_journal_event_ids
    ):
        raise ExpertSourceReplayPublicationError(
            "source replay publication events differ from the factual receipt"
        )
    subjects = {
        reservation.reservation_id,
        *reservation.exact_dependency_ids,
        request.execution_request_id,
        *request.exact_dependency_ids,
        paired_comparison_receipt.paired_comparison_receipt_id,
        *paired_comparison_receipt.exact_dependency_ids,
        stage_decision.source_replay_stage_decision_id,
        *stage_decision.exact_dependency_ids,
        current_release_observation.observation_id,
        current_release_observation.publication_id,
        *current_release_observation.validation_closure_ids,
        *source_base_release.consumed_dependency_ids,
        *candidate.source_dependency_ids,
        *candidate.ancestor_candidate_ids,
        candidate.sanitation_report_id,
        *event_ids,
    }
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
    for event in execution_events:
        if event.predecessor_event_id is not None:
            subjects.add(event.predecessor_event_id)
        subjects.update(
            {
                event.reservation_id,
                event.execution_request_id,
                event.execution_case_id,
                event.execution_leg_id,
            }
        )
        if event.provider_execution_handle is not None:
            subjects.add(event.provider_execution_handle.provider_handle_id)
        spawn_fence = event.spawn_authority_fence
        if spawn_fence is None:
            continue
        spawn_current = spawn_fence.current_release_observation
        spawn_denylist = spawn_fence.security_denylist_observation
        subjects.update(
            {
                spawn_fence.fence_id,
                *spawn_fence.security_subject_ids,
                spawn_current.observation_id,
                spawn_current.publication_id,
                *spawn_current.validation_closure_ids,
                spawn_denylist.observation_id,
                spawn_denylist.snapshot_id,
                spawn_denylist.publication_id,
            }
        )
        for observation in spawn_fence.task_adapter_trust_observations:
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
    _require_sorted_content_ids(
        ordered,
        "source replay publication security subjects",
    )
    return ordered
