"""Resolver-backed authority for precommitted expert release matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kapso.cross_run.contracts import (
    ExpertCandidateValidationState,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertSourceReplayExecutionRequest,
    ExpertValidationAttempt,
    ExpertValidationStage,
    TaskAdapterPackagePin,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixAdapterAuthority,
    ExpertReleaseMatrixEvaluationCell,
    ExpertReleaseMatrixEvaluationPlan,
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceBinding,
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.settings import ExpertValidationPolicy, ExpertValidationSettings
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
    task_adapter_binding_id,
)


class ExpertReleaseMatrixPlanError(ValueError):
    """A plan is not derived from its complete trusted inputs."""


class ExpertReleaseMatrixCandidateProvider(Protocol):
    def read(self, candidate_id: str) -> StoredExpertCandidate: ...


class ExpertReleaseMatrixCurrentReleaseProvider(Protocol):
    def current_release_id(self, scope_id: str) -> str | None: ...


ExpertReleaseMatrixAcceptedResult = (
    ExpertEvaluatorResultRecord
    | ExpertSourceReplayStageResultRecord
    | ExpertAutomatedReviewStageResultRecord
)


def _attempt_release_matrix_mode(
    attempt: ExpertValidationAttempt,
) -> ExpertReleaseMatrixMode:
    if attempt.recovery_plan_id is not None:
        return ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    if attempt.source_base_release_id is None:
        return ExpertReleaseMatrixMode.BOOTSTRAP
    return ExpertReleaseMatrixMode.CONTROL_COMPARISON


def _verified_adapter_key(adapter: VerifiedTaskAdapter) -> tuple[str, str]:
    return (
        adapter.manifest.task_adapter_manifest_id,
        adapter.verification_receipt.verification_receipt_id,
    )


def _canonical_verified_adapters(
    adapters: tuple[VerifiedTaskAdapter, ...],
) -> tuple[VerifiedTaskAdapter, ...]:
    package_keys = tuple(_verified_adapter_key(adapter) for adapter in adapters)
    if len(package_keys) != len(set(package_keys)):
        raise ExpertReleaseMatrixPlanError(
            "prepared release matrix adapter packages are not unique"
        )
    return tuple(sorted(adapters, key=_verified_adapter_key))


def _result_projection(
    result: ExpertReleaseMatrixAcceptedResult,
) -> tuple[ExpertValidationStage, str, bool]:
    if type(result) is ExpertEvaluatorResultRecord:
        return (
            result.evaluator_run.stage,
            result.evaluator_result_record_id,
            result.evaluator_run.outcome is ExpertEvaluatorOutcome.PASSED,
        )
    if type(result) is ExpertSourceReplayStageResultRecord:
        return (
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            result.stage_result_record_id,
            result.outcome is ExpertEvaluatorOutcome.PASSED,
        )
    if type(result) is ExpertAutomatedReviewStageResultRecord:
        return (
            ExpertValidationStage.AUTOMATED_REVIEW,
            result.stage_result_record_id,
            result.outcome is ExpertAutomatedReviewOutcome.PASSED,
        )
    raise ExpertReleaseMatrixPlanError(
        "release matrix history contains an unsupported result"
    )


def _source_result(
    accepted_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
) -> ExpertSourceReplayStageResultRecord | None:
    results = tuple(
        result
        for result in accepted_results
        if type(result) is ExpertSourceReplayStageResultRecord
    )
    if len(results) > 1:
        raise ExpertReleaseMatrixPlanError(
            "release matrix accepts at most one source replay result"
        )
    return None if not results else results[0]


def _validate_active_prefix(
    plan: ExpertReleaseMatrixEvaluationPlan,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    accepted_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
) -> None:
    projections = tuple(_result_projection(result) for result in accepted_results)
    state_refs = tuple(
        (reference.stage, reference.stage_result_record_id)
        for reference in state.accepted_stage_results
    )
    if (
        state.promotion_state is not ExpertPromotionState.VALIDATING
        or state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
        or state.validation_attempt_id != attempt.validation_attempt_id
        or state.candidate_id != attempt.candidate_id
        or state.candidate_tree_hash != attempt.candidate_tree_hash
        or tuple((stage, record_id) for stage, record_id, _passed in projections)
        != state_refs
        or tuple(stage for stage, _record_id, _passed in projections)
        != attempt.required_stages[: len(projections)]
        or len(projections) >= len(attempt.required_stages)
        or attempt.required_stages[len(projections)]
        is not ExpertValidationStage.RELEASE_MATRIX
        or any(not passed for _stage, _record_id, passed in projections)
        or (
            plan.validation_attempt_id,
            plan.candidate_id,
            plan.candidate_tree_hash,
            plan.candidate_commit_record_id,
            plan.scope_contract_id,
            plan.source_base_release_id,
            plan.expected_current_release_id,
            plan.recovery_plan_id,
            plan.control_dependency_ids,
            plan.validation_policy_id,
            plan.configuration_fingerprint,
        )
        != (
            attempt.validation_attempt_id,
            attempt.candidate_id,
            attempt.candidate_tree_hash,
            attempt.candidate_commit_record_id,
            attempt.scope_contract_id,
            attempt.source_base_release_id,
            attempt.expected_current_release_id,
            attempt.recovery_plan_id,
            attempt.control_dependency_ids,
            attempt.validation_policy_id,
            attempt.configuration_fingerprint,
        )
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix plan differs from its active accepted-stage prefix"
        )


def validate_expert_release_matrix_source_joins(
    plan: ExpertReleaseMatrixEvaluationPlan,
    source_result: ExpertSourceReplayStageResultRecord,
    request: ExpertSourceReplayExecutionRequest,
) -> None:
    receipt = source_result.paired_comparison_receipt
    request_cases = {case.execution_case_id: case for case in request.cases}
    comparison_cases = {
        comparison.execution_case_id: comparison
        for comparison in receipt.case_comparisons
    }
    provenances = tuple(
        provenance
        for provenance in plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    )
    provenance_case_ids = tuple(
        provenance.source_execution_case_id for provenance in provenances
    )
    if (
        plan.mode
        not in {
            ExpertReleaseMatrixMode.CONTROL_COMPARISON,
            ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
        }
        or set(provenance_case_ids) != set(request_cases)
        or len(provenance_case_ids) != len(set(provenance_case_ids))
        or set(comparison_cases) != set(request_cases)
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source case coverage is not exact"
        )
    authorities = {
        authority.adapter_authority_id: authority
        for authority in plan.adapter_authorities
    }
    for provenance in provenances:
        request_case = request_cases[provenance.source_execution_case_id]
        comparison_case = comparison_cases[provenance.source_execution_case_id]
        comparisons = {
            item.evaluation_fingerprint.evaluation_fingerprint_id: item
            for item in comparison_case.fingerprint_comparisons
        }
        authority = authorities[provenance.adapter_authority_id]
        pin = authority.task_adapter_pin
        cells = tuple(
            cell
            for cell in plan.evaluation_cells
            if cell.provenance_binding_id == provenance.provenance_binding_id
        )
        if (
            provenance.source_replay_stage_result_id
            != source_result.stage_result_record_id
            or provenance.paired_comparison_receipt_id
            != receipt.paired_comparison_receipt_id
            or provenance.source_replay_selection_id
            != request.source_replay_selection_id
            or provenance.source_bundle_id != request_case.source_bundle_id
            or provenance.bundle_lineage_ids != request_case.bundle_lineage_ids
            or provenance.source_episode_id != request_case.episode_id
            or provenance.task_context_binding.task_context_binding_id
            != request_case.task_context_binding_id
            or provenance.context_materialization_receipt_id
            != request_case.context_materialization_receipt_id
            or provenance.starting_artifact_ids
            != request_case.starting_artifact_content_ids
            or provenance.evaluation_fingerprint_ids
            != request_case.source_evaluation_fingerprint_ids
            or set(provenance.evaluation_fingerprint_ids) != set(comparisons)
            or pin.adapter_binding_id != request_case.adapter_binding_id
            or pin.task_adapter_manifest_id != request_case.task_adapter_manifest_id
            or pin.verification_receipt_id != request_case.verification_receipt_id
            or authority.task_adapter_dependency_ids
            != request_case.task_adapter_dependency_ids
            or len(cells) != len(comparisons)
        ):
            raise ExpertReleaseMatrixPlanError(
                "release matrix provenance differs from its accepted source case"
            )
        for cell in cells:
            comparison = comparisons[
                cell.evaluation_fingerprint.evaluation_fingerprint_id
            ]
            if (
                cell.evaluation_fingerprint != comparison.evaluation_fingerprint
                or cell.metric_comparison_binding
                != comparison.metric_comparison_binding
            ):
                raise ExpertReleaseMatrixPlanError(
                    "release matrix cell differs from its accepted source comparison"
                )


def _validate_adapter_case_joins(
    plan: ExpertReleaseMatrixEvaluationPlan,
    attempt: ExpertValidationAttempt,
) -> None:
    authorities = {
        authority.adapter_authority_id: authority
        for authority in plan.adapter_authorities
    }
    active_pins = {
        (pin.task_adapter_manifest_id, pin.verification_receipt_id): pin
        for pin in attempt.task_adapter_pins
    }
    authority_packages = {
        (
            authority.task_adapter_manifest.task_adapter_manifest_id,
            authority.verification_receipt.verification_receipt_id,
        ): authority
        for authority in plan.adapter_authorities
    }
    if not set(active_pins).issubset(authority_packages) or any(
        authority_packages[key].task_adapter_pin != pin
        for key, pin in active_pins.items()
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix active adapter authority coverage is not exact"
        )
    expected_cases = {
        (authority.adapter_authority_id, case.release_matrix_case_id): case
        for authority in plan.adapter_authorities
        if (
            authority.task_adapter_manifest.task_adapter_manifest_id,
            authority.verification_receipt.verification_receipt_id,
        )
        in active_pins
        for case in authority.task_adapter_manifest.release_matrix_cases
    }
    adapter_provenances = tuple(
        provenance
        for provenance in plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )
    observed_cases = {
        (provenance.adapter_authority_id, provenance.provenance_case_id): (
            provenance.adapter_case
        )
        for provenance in adapter_provenances
    }
    if (
        set(observed_cases) != set(expected_cases)
        or len(observed_cases) != len(adapter_provenances)
        or any(
            observed_cases[key] != expected_case
            for key, expected_case in expected_cases.items()
        )
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix active adapter case coverage is not exact"
        )
    for provenance in adapter_provenances:
        authority = authorities[provenance.adapter_authority_id]
        case = provenance.adapter_case
        cells = tuple(
            cell
            for cell in plan.evaluation_cells
            if cell.provenance_binding_id == provenance.provenance_binding_id
        )
        if case is None or (
            provenance.task_context_binding != case.task_context_binding
            or provenance.evaluation_fingerprint_ids != case.evaluation_fingerprint_ids
            or provenance.starting_artifact_ids != case.starting_artifact_ids
            or len(cells) != len(case.evaluation_fingerprints)
        ):
            raise ExpertReleaseMatrixPlanError(
                "release matrix adapter provenance differs from its signed case"
            )
        fingerprints = {
            fingerprint.evaluation_fingerprint_id: fingerprint
            for fingerprint in case.evaluation_fingerprints
        }
        comparison_bindings = {
            (binding.evaluator_fingerprint, binding.metric_name): binding
            for binding in authority.task_adapter_manifest.task_evaluator.metric_comparison_bindings
        }
        for cell in cells:
            fingerprint = fingerprints.get(
                cell.evaluation_fingerprint.evaluation_fingerprint_id
            )
            binding = comparison_bindings.get(
                (
                    cell.evaluation_fingerprint.evaluator_fingerprint,
                    cell.evaluation_fingerprint.metric_name,
                )
            )
            if (
                fingerprint != cell.evaluation_fingerprint
                or binding != cell.metric_comparison_binding
            ):
                raise ExpertReleaseMatrixPlanError(
                    "release matrix adapter cell differs from its signed case"
                )


def _validate_policy(
    plan: ExpertReleaseMatrixEvaluationPlan,
    policy: ExpertValidationPolicy,
) -> None:
    dimensions = {
        dimension.dimension_id: dimension
        for dimension in policy.policy.promotion.pareto_dimensions
    }
    observed = {
        cell.metric_comparison_binding.comparison_dimension_id
        for cell in plan.evaluation_cells
    }
    if observed != set(dimensions):
        raise ExpertReleaseMatrixPlanError(
            "release matrix Pareto dimension coverage is not exact"
        )
    if any(
        cell.metric_comparison_binding.objective_direction
        is not dimensions[
            cell.metric_comparison_binding.comparison_dimension_id
        ].direction
        for cell in plan.evaluation_cells
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix metric direction differs from promotion policy"
        )


def validate_expert_release_matrix_plan_store_shape(
    *,
    plan: ExpertReleaseMatrixEvaluationPlan,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    accepted_stage_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
    source_replay_request: ExpertSourceReplayExecutionRequest | None,
    validation_policy: ExpertValidationPolicy,
    validation_settings: ExpertValidationSettings,
) -> None:
    """Re-derive durable joins without provider or candidate-store access."""

    if (
        validation_policy != validation_settings.policy.validation_policy()
        or attempt.validation_policy_id != validation_policy.validation_policy_id
        or attempt.configuration_fingerprint
        != validation_settings.configuration_fingerprint
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix plan differs from persisted validation policy"
        )
    _validate_active_prefix(plan, state, attempt, accepted_stage_results)
    result = _source_result(accepted_stage_results)
    expected_mode = _attempt_release_matrix_mode(attempt)
    source_replay_required = attempt.source_base_release_id is not None
    if (
        plan.mode is not expected_mode
        or ((result is None) != (source_replay_request is None))
        or source_replay_required != (source_replay_request is not None)
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source authority differs from its active attempt"
        )
    if result is not None and source_replay_request is not None:
        receipt = result.paired_comparison_receipt
        if (
            plan.mode
            not in {
                ExpertReleaseMatrixMode.CONTROL_COMPARISON,
                ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
            }
            or result.outcome is not ExpertEvaluatorOutcome.PASSED
            or result.validation_attempt_id != attempt.validation_attempt_id
            or result.candidate_id != attempt.candidate_id
            or result.candidate_tree_hash != attempt.candidate_tree_hash
            or result.execution_request_id != source_replay_request.execution_request_id
            or result.validation_policy_id != attempt.validation_policy_id
            or result.configuration_fingerprint != attempt.configuration_fingerprint
            or receipt.execution_request_id
            != source_replay_request.execution_request_id
            or source_replay_request.validation_attempt_id
            != attempt.validation_attempt_id
            or source_replay_request.candidate_id != attempt.candidate_id
            or source_replay_request.candidate_tree_hash != attempt.candidate_tree_hash
            or source_replay_request.candidate_commit_record_id
            != attempt.candidate_commit_record_id
            or source_replay_request.scope_contract_id != attempt.scope_contract_id
            or source_replay_request.source_base_release_id
            != attempt.source_base_release_id
            or source_replay_request.expected_current_release_id
            != attempt.expected_current_release_id
            or source_replay_request.recovery_plan_id != attempt.recovery_plan_id
            or source_replay_request.control_dependency_ids
            != attempt.control_dependency_ids
            or source_replay_request.source_base_tree_hash != plan.source_base_tree_hash
            or source_replay_request.validation_policy_id
            != attempt.validation_policy_id
            or source_replay_request.configuration_fingerprint
            != attempt.configuration_fingerprint
        ):
            raise ExpertReleaseMatrixPlanError(
                "release matrix source authority differs from its active attempt"
            )
        validate_expert_release_matrix_source_joins(
            plan,
            result,
            source_replay_request,
        )
    elif any(
        provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        for provenance in plan.provenance_bindings
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source provenance lacks accepted replay authority"
        )
    _validate_adapter_case_joins(plan, attempt)
    _validate_policy(plan, validation_policy)


def validate_expert_release_matrix_plan_durable_shape(
    *,
    plan: ExpertReleaseMatrixEvaluationPlan,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    accepted_stage_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
    source_replay_request: ExpertSourceReplayExecutionRequest | None,
    stored_candidate: StoredExpertCandidate,
    verified_adapters: tuple[VerifiedTaskAdapter, ...],
    validation_policy: ExpertValidationPolicy,
    validation_settings: ExpertValidationSettings,
) -> None:
    """Add immutable candidate evidence and verified packages to store shape."""

    validate_expert_release_matrix_plan_store_shape(
        plan=plan,
        state=state,
        attempt=attempt,
        accepted_stage_results=accepted_stage_results,
        source_replay_request=source_replay_request,
        validation_policy=validation_policy,
        validation_settings=validation_settings,
    )
    manifest = stored_candidate.closure.manifest
    context = stored_candidate.closure.validation_context
    selection = attempt.source_replay_selection
    if (
        manifest.candidate_id != attempt.candidate_id
        or manifest.candidate_tree_hash != attempt.candidate_tree_hash
        or stored_candidate.commit_record.commit_record_id
        != attempt.candidate_commit_record_id
        or manifest.scope_contract_id != attempt.scope_contract_id
        or manifest.source_base_release_id != attempt.source_base_release_id
        or plan.source_base_tree_hash
        != (
            None
            if attempt.source_base_release_id is None
            else manifest.source_base_tree_hash
        )
        or (
            attempt.source_base_release_id is not None
            and context.source_base_tree_hash != manifest.source_base_tree_hash
        )
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix plan differs from immutable candidate evidence"
        )
    if source_replay_request is None:
        if selection is not None:
            raise ExpertReleaseMatrixPlanError(
                "release matrix source selection lacks an execution request"
            )
    else:
        if (
            stored_candidate.closure.candidate_tree.source_tree_manifest_id
            != source_replay_request.candidate_source_tree_manifest_id
            or selection is None
            or selection.source_replay_selection_id
            != source_replay_request.source_replay_selection_id
        ):
            raise ExpertReleaseMatrixPlanError(
                "release matrix plan differs from immutable source evidence"
            )
        episodes = {
            episode.episode_id: episode for episode in context.replay_evidence.episodes
        }
        provenances = {
            provenance.source_execution_case_id: provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        }
        for request_case in source_replay_request.cases:
            episode = episodes.get(request_case.episode_id)
            provenance = provenances[request_case.execution_case_id]
            if episode is None:
                raise ExpertReleaseMatrixPlanError(
                    "release matrix source case is absent from candidate evidence"
                )
            terminal_attempt = episode.attempts[episode.terminal_attempt_revision]
            environment = episode.artifact_environment
            if (
                episode.source_bundle_id != request_case.source_bundle_id
                or episode.source["node_id"] != request_case.source_node_id
                or episode.task_context_binding != provenance.task_context_binding
                or episode.terminal_attempt_revision
                != request_case.source_execution_revision
                or tuple(
                    fingerprint.evaluation_fingerprint_id
                    for fingerprint in terminal_attempt.evaluation_fingerprints
                )
                != request_case.source_evaluation_fingerprint_ids
                or environment.task_adapter_manifest_id
                != request_case.task_adapter_manifest_id
                or environment.task_adapter_verification_receipt_id
                != request_case.verification_receipt_id
                or tuple(sorted(environment.starting_artifact_content_ids.values()))
                != request_case.starting_artifact_content_ids
            ):
                raise ExpertReleaseMatrixPlanError(
                    "release matrix source case differs from candidate episode evidence"
                )
    verified = {
        (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        ): adapter
        for adapter in verified_adapters
    }
    authority_packages = {
        (
            authority.task_adapter_manifest.task_adapter_manifest_id,
            authority.verification_receipt.verification_receipt_id,
        )
        for authority in plan.adapter_authorities
    }
    active_pins = {
        (pin.task_adapter_manifest_id, pin.verification_receipt_id): pin
        for pin in attempt.task_adapter_pins
    }
    if set(verified) != authority_packages or not set(active_pins).issubset(
        authority_packages
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix verified adapter coverage is not exact"
        )
    for authority in plan.adapter_authorities:
        adapter = verified[
            (
                authority.task_adapter_manifest.task_adapter_manifest_id,
                authority.verification_receipt.verification_receipt_id,
            )
        ]
        if (
            adapter.manifest != authority.task_adapter_manifest
            or adapter.verification_receipt != authority.verification_receipt
            or adapter.dependency_ids != authority.task_adapter_dependency_ids
        ):
            raise ExpertReleaseMatrixPlanError(
                "release matrix adapter authority differs from verified package"
            )
        package_key = (
            authority.task_adapter_manifest.task_adapter_manifest_id,
            authority.verification_receipt.verification_receipt_id,
        )
        if package_key in active_pins:
            if authority.task_adapter_pin != active_pins[package_key]:
                raise ExpertReleaseMatrixPlanError(
                    "release matrix active adapter authority differs from its pin"
                )
            for case in authority.task_adapter_manifest.release_matrix_cases:
                case.task_context_binding.validate_against(context.scope_contract)


@dataclass(frozen=True)
class PreparedExpertReleaseMatrixPlan:
    """Runtime authority whose complete closure passed admission checks."""

    plan: ExpertReleaseMatrixEvaluationPlan
    state: ExpertCandidateValidationState
    attempt: ExpertValidationAttempt
    accepted_stage_results: tuple[ExpertReleaseMatrixAcceptedResult, ...]
    source_replay_request: ExpertSourceReplayExecutionRequest | None
    stored_candidate: StoredExpertCandidate
    verified_adapters: tuple[VerifiedTaskAdapter, ...]
    validation_policy: ExpertValidationPolicy
    validation_settings: ExpertValidationSettings

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "accepted_stage_results", tuple(self.accepted_stage_results)
        )
        adapters = tuple(self.verified_adapters)
        if (
            type(self.plan) is not ExpertReleaseMatrixEvaluationPlan
            or type(self.state) is not ExpertCandidateValidationState
            or type(self.attempt) is not ExpertValidationAttempt
            or (
                self.source_replay_request is not None
                and type(self.source_replay_request)
                is not ExpertSourceReplayExecutionRequest
            )
            or not isinstance(self.stored_candidate, StoredExpertCandidate)
            or any(
                not isinstance(adapter, VerifiedTaskAdapter)
                for adapter in self.verified_adapters
            )
            or type(self.validation_policy) is not ExpertValidationPolicy
            or type(self.validation_settings) is not ExpertValidationSettings
        ):
            raise ExpertReleaseMatrixPlanError(
                "prepared release matrix plan closure is not typed"
            )
        object.__setattr__(
            self,
            "verified_adapters",
            _canonical_verified_adapters(adapters),
        )
        validate_expert_release_matrix_plan_durable_shape(
            plan=self.plan,
            state=self.state,
            attempt=self.attempt,
            accepted_stage_results=self.accepted_stage_results,
            source_replay_request=self.source_replay_request,
            stored_candidate=self.stored_candidate,
            verified_adapters=self.verified_adapters,
            validation_policy=self.validation_policy,
            validation_settings=self.validation_settings,
        )


def prepare_expert_release_matrix_plan_for_admission(
    *,
    prepared_plan: PreparedExpertReleaseMatrixPlan,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    accepted_stage_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
    source_replay_request: ExpertSourceReplayExecutionRequest | None,
    candidate_store: ExpertReleaseMatrixCandidateProvider,
    current_release_provider: ExpertReleaseMatrixCurrentReleaseProvider,
    task_adapter_provider: VerifiedTaskAdapterProvider,
    validation_policy: ExpertValidationPolicy,
    validation_settings: ExpertValidationSettings,
) -> PreparedExpertReleaseMatrixPlan:
    """Freshly reopen external authority immediately before reservation.

    Reservation performs no evaluator work. Bootstrap absence is checked on both
    sides of exact adapter resolution; durable alias replay stays offline. The
    later execution and publication boundaries must reacquire current authority.
    """

    if type(prepared_plan) is not PreparedExpertReleaseMatrixPlan:
        raise ExpertReleaseMatrixPlanError(
            "release matrix admission requires a prepared plan"
        )
    plan = prepared_plan.plan
    stored_candidate = candidate_store.read(plan.candidate_id)
    scope_id = stored_candidate.closure.validation_context.scope_id
    observed_current_before_adapter_resolution = (
        current_release_provider.current_release_id(scope_id)
    )
    if (
        observed_current_before_adapter_resolution
        != attempt.expected_current_release_id
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source-base authority changed before reservation"
        )
    verified_adapters = tuple(
        task_adapter_provider.resolve_exact(
            task_adapter_manifest_id=(
                authority.task_adapter_pin.task_adapter_manifest_id
            ),
            verification_receipt_id=(
                authority.task_adapter_pin.verification_receipt_id
            ),
        )
        for authority in plan.adapter_authorities
    )
    observed_current_after_adapter_resolution = (
        current_release_provider.current_release_id(scope_id)
    )
    if observed_current_after_adapter_resolution != attempt.expected_current_release_id:
        raise ExpertReleaseMatrixPlanError(
            "release matrix source-base authority changed during adapter resolution"
        )
    return PreparedExpertReleaseMatrixPlan(
        plan=plan,
        state=state,
        attempt=attempt,
        accepted_stage_results=accepted_stage_results,
        source_replay_request=source_replay_request,
        stored_candidate=stored_candidate,
        verified_adapters=verified_adapters,
        validation_policy=validation_policy,
        validation_settings=validation_settings,
    )


def _authority(adapter: VerifiedTaskAdapter) -> ExpertReleaseMatrixAdapterAuthority:
    manifest = adapter.manifest
    receipt = adapter.verification_receipt
    return ExpertReleaseMatrixAdapterAuthority.mint(
        task_adapter_pin=TaskAdapterPackagePin(
            adapter_binding_id=task_adapter_binding_id(
                manifest.task_family_id, manifest.task_adapter_id
            ),
            task_adapter_manifest_id=manifest.task_adapter_manifest_id,
            verification_receipt_id=receipt.verification_receipt_id,
        ),
        task_adapter_manifest=manifest,
        verification_receipt=receipt,
        task_adapter_dependency_ids=adapter.dependency_ids,
    )


def derive_expert_release_matrix_plan(
    *,
    state: ExpertCandidateValidationState,
    attempt: ExpertValidationAttempt,
    accepted_stage_results: tuple[ExpertReleaseMatrixAcceptedResult, ...],
    source_replay_request: ExpertSourceReplayExecutionRequest | None,
    stored_candidate: StoredExpertCandidate,
    verified_adapters: tuple[VerifiedTaskAdapter, ...],
    validation_policy: ExpertValidationPolicy,
    validation_settings: ExpertValidationSettings,
) -> PreparedExpertReleaseMatrixPlan:
    """Derive every signed active case and any exact accepted source evidence."""

    result = _source_result(accepted_stage_results)
    if (result is None) != (source_replay_request is None):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source result and execution request must appear together"
        )
    mode = _attempt_release_matrix_mode(attempt)
    if mode is ExpertReleaseMatrixMode.BOOTSTRAP and source_replay_request is not None:
        raise ExpertReleaseMatrixPlanError(
            "bootstrap release matrix cannot reuse control comparison evidence"
        )
    if attempt.source_base_release_id is not None and source_replay_request is None:
        raise ExpertReleaseMatrixPlanError(
            "source-base release matrix requires accepted source replay authority"
        )
    source_base_tree_hash = (
        None
        if attempt.source_base_release_id is None
        else stored_candidate.closure.manifest.source_base_tree_hash
    )
    if (
        source_replay_request is not None
        and source_replay_request.source_base_tree_hash != source_base_tree_hash
    ):
        raise ExpertReleaseMatrixPlanError(
            "release matrix source request differs from the candidate source-base tree"
        )
    canonical_adapters = _canonical_verified_adapters(verified_adapters)
    episodes = {
        episode.episode_id: episode
        for episode in (
            stored_candidate.closure.validation_context.replay_evidence.episodes
        )
    }
    adapters = {
        (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        ): adapter
        for adapter in canonical_adapters
    }
    active_pins = {
        (pin.task_adapter_manifest_id, pin.verification_receipt_id): pin
        for pin in attempt.task_adapter_pins
    }
    if len(active_pins) != len(attempt.task_adapter_pins):
        raise ExpertReleaseMatrixPlanError(
            "release matrix active adapter pins are not unique"
        )
    source_package_keys = (
        set()
        if source_replay_request is None
        else {
            (case.task_adapter_manifest_id, case.verification_receipt_id)
            for case in source_replay_request.cases
        }
    )
    required_package_keys = {*active_pins, *source_package_keys}
    if set(adapters) != required_package_keys:
        raise ExpertReleaseMatrixPlanError(
            "release matrix verified adapter closure is not exact"
        )
    authorities = {
        key: _authority(adapters[key]) for key in sorted(required_package_keys)
    }
    for package_key, pin in active_pins.items():
        if authorities[package_key].task_adapter_pin != pin:
            raise ExpertReleaseMatrixPlanError(
                "release matrix active adapter differs from its attempt pin"
            )
    provenances = []
    cells = []

    def append_cells(authority, provenance, fingerprint_bindings) -> None:
        for fingerprint, comparison_binding in fingerprint_bindings:
            dependencies = {
                attempt.validation_attempt_id,
                attempt.candidate_id,
                authority.adapter_authority_id,
                provenance.provenance_binding_id,
                provenance.task_context_binding.task_context_binding_id,
                provenance.independence_identity_id,
                fingerprint.evaluation_fingerprint_id,
            }
            if attempt.source_base_release_id is not None:
                dependencies.add(attempt.source_base_release_id)
            cells.append(
                ExpertReleaseMatrixEvaluationCell.mint(
                    mode=mode,
                    validation_attempt_id=attempt.validation_attempt_id,
                    candidate_id=attempt.candidate_id,
                    candidate_tree_hash=attempt.candidate_tree_hash,
                    source_base_release_id=attempt.source_base_release_id,
                    source_base_tree_hash=source_base_tree_hash,
                    adapter_authority_id=authority.adapter_authority_id,
                    provenance_binding_id=provenance.provenance_binding_id,
                    task_context_binding=provenance.task_context_binding,
                    independence_identity_id=provenance.independence_identity_id,
                    evaluation_fingerprint=fingerprint,
                    metric_comparison_binding=comparison_binding,
                    exact_dependency_ids=tuple(sorted(dependencies)),
                )
            )

    if source_replay_request is not None and result is not None:
        comparisons = {
            comparison.execution_case_id: comparison
            for comparison in result.paired_comparison_receipt.case_comparisons
        }
        for request_case in source_replay_request.cases:
            package_key = (
                request_case.task_adapter_manifest_id,
                request_case.verification_receipt_id,
            )
            authority = authorities[package_key]
            episode = episodes.get(request_case.episode_id)
            comparison_case = comparisons.get(request_case.execution_case_id)
            if episode is None or comparison_case is None:
                raise ExpertReleaseMatrixPlanError(
                    "release matrix derivation lacks source case authority"
                )
            fingerprint_ids = tuple(
                item.evaluation_fingerprint.evaluation_fingerprint_id
                for item in comparison_case.fingerprint_comparisons
            )
            provenance_dependencies = {
                authority.adapter_authority_id,
                episode.task_context_binding.task_context_binding_id,
                *fingerprint_ids,
                result.stage_result_record_id,
                result.paired_comparison_receipt.paired_comparison_receipt_id,
                request_case.execution_case_id,
                source_replay_request.source_replay_selection_id,
                *request_case.bundle_lineage_ids,
                request_case.episode_id,
                request_case.context_materialization_receipt_id,
                *request_case.starting_artifact_content_ids,
            }
            provenance = ExpertReleaseMatrixProvenanceBinding.mint(
                provenance_kind=ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY,
                adapter_authority_id=authority.adapter_authority_id,
                task_context_binding=episode.task_context_binding,
                evaluation_fingerprint_ids=fingerprint_ids,
                adapter_case=None,
                source_replay_stage_result_id=result.stage_result_record_id,
                paired_comparison_receipt_id=(
                    result.paired_comparison_receipt.paired_comparison_receipt_id
                ),
                source_execution_case_id=request_case.execution_case_id,
                source_replay_selection_id=(
                    source_replay_request.source_replay_selection_id
                ),
                source_bundle_id=request_case.source_bundle_id,
                bundle_lineage_ids=request_case.bundle_lineage_ids,
                source_episode_id=request_case.episode_id,
                context_materialization_receipt_id=(
                    request_case.context_materialization_receipt_id
                ),
                starting_artifact_ids=request_case.starting_artifact_content_ids,
                exact_dependency_ids=tuple(sorted(provenance_dependencies)),
            )
            provenances.append(provenance)
            append_cells(
                authority,
                provenance,
                tuple(
                    (
                        comparison.evaluation_fingerprint,
                        comparison.metric_comparison_binding,
                    )
                    for comparison in comparison_case.fingerprint_comparisons
                ),
            )

    for package_key, pin in sorted(
        active_pins.items(), key=lambda item: item[1].adapter_binding_id
    ):
        authority = authorities[package_key]
        manifest = authority.task_adapter_manifest
        comparison_bindings = {
            (binding.evaluator_fingerprint, binding.metric_name): binding
            for binding in manifest.task_evaluator.metric_comparison_bindings
        }
        for case in manifest.release_matrix_cases:
            provenance_dependencies = {
                authority.adapter_authority_id,
                case.task_context_binding.task_context_binding_id,
                case.release_matrix_case_id,
                case.independence_group.independence_group_id,
                *case.evaluation_fingerprint_ids,
                *case.starting_artifact_ids,
            }
            provenance = ExpertReleaseMatrixProvenanceBinding.mint(
                provenance_kind=ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE,
                adapter_authority_id=authority.adapter_authority_id,
                task_context_binding=case.task_context_binding,
                evaluation_fingerprint_ids=case.evaluation_fingerprint_ids,
                adapter_case=case,
                source_replay_stage_result_id=None,
                paired_comparison_receipt_id=None,
                source_execution_case_id=None,
                source_replay_selection_id=None,
                source_bundle_id=None,
                bundle_lineage_ids=(),
                source_episode_id=None,
                context_materialization_receipt_id=None,
                starting_artifact_ids=case.starting_artifact_ids,
                exact_dependency_ids=tuple(sorted(provenance_dependencies)),
            )
            fingerprint_bindings = tuple(
                (
                    fingerprint,
                    comparison_bindings.get(
                        (
                            fingerprint.evaluator_fingerprint,
                            fingerprint.metric_name,
                        )
                    ),
                )
                for fingerprint in case.evaluation_fingerprints
            )
            if any(binding is None for _fingerprint, binding in fingerprint_bindings):
                raise ExpertReleaseMatrixPlanError(
                    "release matrix adapter case lacks metric comparison authority"
                )
            provenances.append(provenance)
            append_cells(authority, provenance, fingerprint_bindings)
    ordered_authorities = tuple(
        sorted(authorities.values(), key=lambda item: item.canonical_key)
    )
    ordered_provenances = tuple(
        sorted(provenances, key=lambda item: item.canonical_key)
    )
    ordered_cells = tuple(sorted(cells, key=lambda item: item.canonical_key))
    internal_ids = {
        *(item.adapter_authority_id for item in ordered_authorities),
        *(item.provenance_binding_id for item in ordered_provenances),
        *(item.evaluation_cell_id for item in ordered_cells),
    }
    external = {
        dependency
        for authority in ordered_authorities
        for dependency in authority.exact_dependency_ids
    }
    external.update(
        dependency
        for provenance in ordered_provenances
        for dependency in provenance.exact_dependency_ids
        if dependency not in internal_ids
    )
    external.update(
        dependency
        for cell in ordered_cells
        for dependency in cell.exact_dependency_ids
        if dependency not in internal_ids
    )
    external.update(
        {
            attempt.validation_attempt_id,
            attempt.candidate_id,
            attempt.candidate_commit_record_id,
            attempt.scope_contract_id,
            attempt.validation_policy_id,
        }
    )
    if attempt.source_base_release_id is not None:
        external.add(attempt.source_base_release_id)
    if attempt.expected_current_release_id is not None:
        external.add(attempt.expected_current_release_id)
    if attempt.recovery_plan_id is not None:
        external.add(attempt.recovery_plan_id)
    external.update(attempt.control_dependency_ids)
    plan = ExpertReleaseMatrixEvaluationPlan.mint(
        mode=mode,
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_commit_record_id=attempt.candidate_commit_record_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        scope_contract_id=attempt.scope_contract_id,
        source_base_release_id=attempt.source_base_release_id,
        source_base_tree_hash=source_base_tree_hash,
        expected_current_release_id=attempt.expected_current_release_id,
        recovery_plan_id=attempt.recovery_plan_id,
        control_dependency_ids=attempt.control_dependency_ids,
        validation_policy_id=attempt.validation_policy_id,
        configuration_fingerprint=attempt.configuration_fingerprint,
        adapter_authorities=ordered_authorities,
        provenance_bindings=ordered_provenances,
        evaluation_cells=ordered_cells,
        external_dependency_ids=tuple(sorted(external)),
    )
    return PreparedExpertReleaseMatrixPlan(
        plan=plan,
        state=state,
        attempt=attempt,
        accepted_stage_results=accepted_stage_results,
        source_replay_request=source_replay_request,
        stored_candidate=stored_candidate,
        verified_adapters=canonical_adapters,
        validation_policy=validation_policy,
        validation_settings=validation_settings,
    )
