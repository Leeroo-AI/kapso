"""Trusted factual reduction of one completed expert source replay journal."""

from __future__ import annotations

from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayExecutionReservation,
    ObjectiveDirection,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayCaseComparison,
    ExpertSourceReplayFingerprintComparison,
    ExpertSourceReplayPairedComparisonReceipt,
    normalized_zero,
)
from kapso.cross_run.expert.replay_execution_store import (
    CompletedExpertSourceReplayExecution,
    ExpertSourceReplayExecutionStore,
    SourceReplayExecutionJournalEventKind,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest


class ExpertSourceReplayComparisonError(ValueError):
    """A completed replay journal cannot produce one exact factual receipt."""


def build_expert_source_replay_paired_comparison_receipt(
    *,
    completed_execution: CompletedExpertSourceReplayExecution,
    execution_store: ExpertSourceReplayExecutionStore,
    reservation: ExpertSourceReplayExecutionReservation,
    prepared_request: PreparedExpertSourceReplayRequest,
) -> ExpertSourceReplayPairedComparisonReceipt:
    if type(completed_execution) is not CompletedExpertSourceReplayExecution:
        raise ExpertSourceReplayComparisonError(
            "source replay comparison requires a sealed completed execution"
        )
    events = completed_execution.require_exact(
        execution_store,
        reservation,
        prepared_request,
    )
    request = prepared_request.request
    accepted_events = tuple(
        event
        for event in events
        if event.event_kind is SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED
    )
    accepted_by_case_and_leg = {
        (event.execution_case_id, event.execution_leg_id): event
        for event in accepted_events
    }
    expected_leg_keys = {
        (case.execution_case_id, leg.execution_leg_id)
        for case in request.cases
        for leg in (case.control_leg, case.candidate_leg)
    }
    if (
        len(accepted_by_case_and_leg) != len(accepted_events)
        or set(accepted_by_case_and_leg) != expected_leg_keys
    ):
        raise ExpertSourceReplayComparisonError(
            "source replay accepted results differ from the exact case-leg schedule"
        )
    materialized_cases = {
        item.request_case.execution_case_id: item for item in prepared_request.cases
    }
    if set(materialized_cases) != {case.execution_case_id for case in request.cases}:
        raise ExpertSourceReplayComparisonError(
            "source replay prepared cases differ from the execution request"
        )
    case_comparisons = []
    for request_case in sorted(request.cases, key=lambda case: case.execution_case_id):
        materialized_case = materialized_cases[request_case.execution_case_id]
        terminal_attempt = materialized_case.episode.attempts[
            materialized_case.episode.terminal_attempt_revision
        ]
        fingerprints = {
            fingerprint.evaluation_fingerprint_id: fingerprint
            for fingerprint in terminal_attempt.evaluation_fingerprints
        }
        comparison_bindings = {
            (binding.evaluator_fingerprint, binding.metric_name): binding
            for binding in (
                materialized_case.task_adapter.manifest.task_evaluator.metric_comparison_bindings
            )
        }
        legs_by_kind = {
            request_case.control_leg.kind: request_case.control_leg,
            request_case.candidate_leg.kind: request_case.candidate_leg,
        }
        control_event = accepted_by_case_and_leg[
            (
                request_case.execution_case_id,
                legs_by_kind[
                    ExpertSourceReplayExecutionLegKind.CONTROL_PARENT
                ].execution_leg_id,
            )
        ]
        candidate_event = accepted_by_case_and_leg[
            (
                request_case.execution_case_id,
                legs_by_kind[
                    ExpertSourceReplayExecutionLegKind.CANDIDATE
                ].execution_leg_id,
            )
        ]
        control_results = {
            result.evaluation_fingerprint_id: result
            for result in control_event.task_evaluator_result.fingerprint_results
        }
        candidate_results = {
            result.evaluation_fingerprint_id: result
            for result in candidate_event.task_evaluator_result.fingerprint_results
        }
        expected_fingerprint_ids = set(request_case.source_evaluation_fingerprint_ids)
        if (
            set(fingerprints) != expected_fingerprint_ids
            or set(control_results) != expected_fingerprint_ids
            or set(candidate_results) != expected_fingerprint_ids
        ):
            raise ExpertSourceReplayComparisonError(
                "source replay comparison fingerprint sets differ from the request"
            )
        fingerprint_comparisons = []
        for fingerprint_id in sorted(expected_fingerprint_ids):
            fingerprint = fingerprints[fingerprint_id]
            binding = comparison_bindings.get(
                (fingerprint.evaluator_fingerprint, fingerprint.metric_name)
            )
            if binding is None:
                raise ExpertSourceReplayComparisonError(
                    "source replay comparison lacks exact adapter metric authority"
                )
            control_result = control_results[fingerprint_id]
            candidate_result = candidate_results[fingerprint_id]
            raw_delta = normalized_zero(
                candidate_result.aggregate_value - control_result.aggregate_value
            )
            direction_aligned_delta = normalized_zero(
                raw_delta
                if fingerprint.objective_direction is ObjectiveDirection.MAXIMIZE
                else -raw_delta
            )
            normalized_effect = normalized_zero(
                direction_aligned_delta / binding.comparison_scale
            )
            fingerprint_comparisons.append(
                ExpertSourceReplayFingerprintComparison(
                    evaluation_fingerprint=fingerprint,
                    metric_comparison_binding=binding,
                    control_result=control_result,
                    candidate_result=candidate_result,
                    aggregate_raw_delta=raw_delta,
                    aggregate_direction_aligned_delta=direction_aligned_delta,
                    aggregate_normalized_effect=normalized_effect,
                )
            )
        case_comparisons.append(
            ExpertSourceReplayCaseComparison(
                execution_case_id=request_case.execution_case_id,
                score_of_record_fingerprint_id=(
                    request_case.source_score_of_record_fingerprint_id
                ),
                control_result_accepted_event_id=control_event.event_id,
                candidate_result_accepted_event_id=candidate_event.event_id,
                fingerprint_comparisons=tuple(fingerprint_comparisons),
            )
        )
    event_ids = tuple(event.event_id for event in events)
    exact_dependency_ids = tuple(
        sorted(
            {
                reservation.reservation_id,
                *reservation.exact_dependency_ids,
                request.execution_request_id,
                *request.exact_dependency_ids,
                *event_ids,
            }
        )
    )
    return ExpertSourceReplayPairedComparisonReceipt.mint(
        reservation_id=reservation.reservation_id,
        execution_request_id=request.execution_request_id,
        aggregate_recomputation_tolerance=(
            prepared_request.settings.policy.task_evaluation_aggregate_tolerance
        ),
        execution_journal_event_ids=event_ids,
        reservation_dependency_ids=reservation.exact_dependency_ids,
        request_dependency_ids=request.exact_dependency_ids,
        case_comparisons=tuple(case_comparisons),
        exact_dependency_ids=exact_dependency_ids,
    )
