"""Join task-evaluation requests to their reserved release-matrix authority."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.contracts import ExpertValidationStage
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.task_evaluation_contracts import (
    TASK_EVALUATION_REQUEST_CONTRACT_VERSION,
    TaskEvaluationCase,
    TaskEvaluationComputeBinding,
    TaskEvaluationExpertLeg,
    TaskEvaluationLegKind,
    TaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_compute import (
    derive_release_matrix_compute_bindings,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)
from kapso.cross_run.settings import ExpertValidationSettings


class TaskEvaluationRequestPreparationError(ValueError):
    """A task-evaluation request differs from its reserved matrix plan."""


@dataclass(frozen=True)
class PlanJoinedTaskEvaluationRequest:
    """Runtime proof that a request is the exact adapter-case plan projection.

    This closure proves plan and configured-evaluator joins only. It is not an
    execution capability: dispatch additionally requires materialized package
    bytes, configured compute bindings, and fresh adapter authority.
    """

    request: TaskEvaluationRequest
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot
    settings: ExpertValidationSettings

    def __post_init__(self) -> None:
        if (
            type(self.request) is not TaskEvaluationRequest
            or type(self.plan_reservation)
            is not ExpertReleaseMatrixPlanReservationSnapshot
            or type(self.settings) is not ExpertValidationSettings
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation plan join requires exact typed inputs"
            )
        self._validate_plan_subjects()
        self._validate_evaluator()
        self._validate_adapter_cases()

    def _validate_plan_subjects(self) -> None:
        request = self.request
        reservation = self.plan_reservation
        plan = reservation.evaluation_plan
        snapshot = reservation.snapshot
        attempt = snapshot.latest_attempt
        if attempt is None:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation plan join lacks a validation attempt"
            )
        if (
            request.plan_reservation_operation_id != reservation.operation.operation_id
            or request.evaluation_plan_id != plan.evaluation_plan_id
            or request.mode is not plan.mode
            or request.authorization_transition_id != snapshot.transition.transition_id
            or request.authorization_state_id != snapshot.state.validation_state_id
            or request.validation_attempt_id != plan.validation_attempt_id
            or request.validation_attempt_id != attempt.validation_attempt_id
            or request.candidate_id != plan.candidate_id
            or request.candidate_commit_record_id != plan.candidate_commit_record_id
            or request.candidate_tree_hash != plan.candidate_tree_hash
            or request.scope_contract_id != plan.scope_contract_id
            or {request.scope_id}
            != {
                provenance.task_context_binding.scope_id
                for provenance in plan.provenance_bindings
            }
            or request.parent_release_id != plan.parent_release_id
            or request.parent_tree_hash != plan.parent_tree_hash
            or request.validation_policy_id != plan.validation_policy_id
            or request.configuration_fingerprint != plan.configuration_fingerprint
            or request.configuration_fingerprint
            != self.settings.configuration_fingerprint
            or request.plan_dependency_ids != plan.exact_dependency_ids
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation request differs from its reserved plan subjects"
            )

    def _validate_evaluator(self) -> None:
        evaluators = tuple(
            evaluator
            for evaluator in self.settings.policy.evaluators
            if evaluator.stage is ExpertValidationStage.RELEASE_MATRIX
        )
        if len(evaluators) != 1:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation requires one configured release-matrix evaluator"
            )
        evaluator = evaluators[0]
        request = self.request
        if (
            request.release_matrix_evaluator_id != evaluator.evaluator_id
            or request.release_matrix_evaluator_role != evaluator.evaluator_role
            or request.release_matrix_evaluator_version != evaluator.evaluator_version
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation evaluator differs from configuration"
            )

    def _validate_adapter_cases(self) -> None:
        plan = self.plan_reservation.evaluation_plan
        expected_provenances = tuple(
            provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
        )
        cases_by_provenance_id = {
            case.provenance_binding_id: case for case in self.request.cases
        }
        if len(cases_by_provenance_id) != len(self.request.cases) or set(
            cases_by_provenance_id
        ) != {provenance.provenance_binding_id for provenance in expected_provenances}:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation request adapter provenance coverage is not exact"
            )
        compute_bindings = derive_release_matrix_compute_bindings(
            settings=self.settings,
            mode=plan.mode,
            provenance_binding_ids=tuple(
                provenance.provenance_binding_id for provenance in expected_provenances
            ),
        )
        cells_by_provenance_id = {
            provenance.provenance_binding_id: tuple(
                cell
                for cell in plan.evaluation_cells
                if cell.provenance_binding_id == provenance.provenance_binding_id
            )
            for provenance in expected_provenances
        }
        for provenance in expected_provenances:
            adapter_case = provenance.adapter_case
            if adapter_case is None:
                raise TaskEvaluationRequestPreparationError(
                    "task evaluation adapter provenance lacks its signed case"
                )
            case = cases_by_provenance_id[provenance.provenance_binding_id]
            cells = cells_by_provenance_id[provenance.provenance_binding_id]
            if (
                case.adapter_authority_id != provenance.adapter_authority_id
                or case.release_matrix_case_id != adapter_case.release_matrix_case_id
                or case.task_context_binding_id
                != provenance.task_context_binding.task_context_binding_id
                or case.independence_group_id
                != adapter_case.independence_group.independence_group_id
                or case.evaluation_cell_ids
                != tuple(sorted(cell.evaluation_cell_id for cell in cells))
                or case.evaluation_fingerprint_ids
                != provenance.evaluation_fingerprint_ids
                or case.evaluation_fingerprint_ids
                != tuple(
                    sorted(
                        cell.evaluation_fingerprint.evaluation_fingerprint_id
                        for cell in cells
                    )
                )
                or case.starting_artifact_ids != provenance.starting_artifact_ids
                or case.compute_binding
                != compute_bindings[provenance.provenance_binding_id]
            ):
                raise TaskEvaluationRequestPreparationError(
                    "task evaluation case differs from its reserved provenance"
                )


def prepare_task_evaluation_request(
    *,
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    settings: ExpertValidationSettings,
    stored_candidate: StoredExpertCandidate,
    candidate: VerifiedTaskEvaluationCandidate,
    parent: VerifiedTaskEvaluationParent | None,
) -> PlanJoinedTaskEvaluationRequest:
    """Derive the only request admitted by one reserved matrix plan."""

    validate_task_evaluation_candidate_authority(
        plan_reservation=plan_reservation,
        settings=settings,
        stored_candidate=stored_candidate,
        candidate=candidate,
    )
    plan = plan_reservation.evaluation_plan
    context = stored_candidate.closure.validation_context
    if (plan.parent_release_id is None) != (parent is None):
        raise TaskEvaluationRequestPreparationError(
            "task evaluation parent authority differs from matrix mode"
        )
    if parent is not None and (
        type(parent) is not VerifiedTaskEvaluationParent
        or parent.release_manifest != context.parent_release
        or parent.parent_tree_receipt != context.parent_tree_receipt
        or parent.release_manifest.release_id != plan.parent_release_id
        or parent.release_manifest.scope_contract_id != plan.scope_contract_id
        or parent.parent_tree_receipt.parent_tree_hash != plan.parent_tree_hash
    ):
        raise TaskEvaluationRequestPreparationError(
            "task evaluation parent differs from reserved plan authority"
        )
    if parent is None and (
        context.parent_release is not None or context.parent_tree_receipt is not None
    ):
        raise TaskEvaluationRequestPreparationError(
            "task evaluation bootstrap candidate contains parent authority"
        )
    evaluators = tuple(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.RELEASE_MATRIX
    )
    if len(evaluators) != 1:
        raise TaskEvaluationRequestPreparationError(
            "task evaluation requires one configured release-matrix evaluator"
        )
    evaluator = evaluators[0]
    legs = _task_evaluation_legs(
        candidate=candidate,
        parent=parent,
    )
    provenances = tuple(
        provenance
        for provenance in plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )
    compute_bindings = derive_release_matrix_compute_bindings(
        settings=settings,
        mode=plan.mode,
        provenance_binding_ids=tuple(
            provenance.provenance_binding_id for provenance in provenances
        ),
    )
    cases = tuple(
        sorted(
            (
                _task_evaluation_case(
                    plan_reservation=plan_reservation,
                    provenance_binding_id=provenance.provenance_binding_id,
                    compute_binding=compute_bindings[provenance.provenance_binding_id],
                    legs=legs,
                )
                for provenance in provenances
            ),
            key=lambda case: case.canonical_key,
        )
    )
    snapshot = plan_reservation.snapshot
    dependencies = {
        plan_reservation.operation.operation_id,
        plan.evaluation_plan_id,
        snapshot.transition.transition_id,
        snapshot.state.validation_state_id,
        plan.validation_attempt_id,
        plan.candidate_id,
        plan.candidate_commit_record_id,
        plan.scope_contract_id,
        plan.validation_policy_id,
        *plan.exact_dependency_ids,
        *(case.evaluation_case_id for case in cases),
        *(
            dependency_id
            for case in cases
            for dependency_id in case.exact_dependency_ids
        ),
    }
    if plan.parent_release_id is not None:
        dependencies.add(plan.parent_release_id)
    request = TaskEvaluationRequest.mint(
        request_contract_version=TASK_EVALUATION_REQUEST_CONTRACT_VERSION,
        plan_reservation_operation_id=plan_reservation.operation.operation_id,
        evaluation_plan_id=plan.evaluation_plan_id,
        mode=plan.mode,
        authorization_transition_id=snapshot.transition.transition_id,
        authorization_state_id=snapshot.state.validation_state_id,
        validation_attempt_id=plan.validation_attempt_id,
        candidate_id=plan.candidate_id,
        candidate_commit_record_id=plan.candidate_commit_record_id,
        candidate_tree_hash=plan.candidate_tree_hash,
        scope_contract_id=plan.scope_contract_id,
        scope_id=context.scope_id,
        parent_release_id=plan.parent_release_id,
        parent_tree_hash=plan.parent_tree_hash,
        validation_policy_id=plan.validation_policy_id,
        configuration_fingerprint=plan.configuration_fingerprint,
        release_matrix_evaluator_id=evaluator.evaluator_id,
        release_matrix_evaluator_role=evaluator.evaluator_role,
        release_matrix_evaluator_version=evaluator.evaluator_version,
        plan_dependency_ids=plan.exact_dependency_ids,
        cases=cases,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
    return PlanJoinedTaskEvaluationRequest(
        request=request,
        plan_reservation=plan_reservation,
        settings=settings,
    )


def validate_task_evaluation_candidate_authority(
    *,
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    settings: ExpertValidationSettings,
    stored_candidate: StoredExpertCandidate,
    candidate: VerifiedTaskEvaluationCandidate,
) -> None:
    """Reject a candidate/configuration substitution before external reads."""

    if (
        type(plan_reservation) is not ExpertReleaseMatrixPlanReservationSnapshot
        or type(settings) is not ExpertValidationSettings
        or type(stored_candidate) is not StoredExpertCandidate
        or type(candidate) is not VerifiedTaskEvaluationCandidate
    ):
        raise TaskEvaluationRequestPreparationError(
            "task evaluation request derivation requires exact typed authority"
        )
    plan = plan_reservation.evaluation_plan
    closure = stored_candidate.closure
    context = closure.validation_context
    if (
        candidate.manifest != closure.manifest
        or candidate.commit_record != stored_candidate.commit_record
        or candidate.source_tree != closure.candidate_tree
        or candidate.source_contents != closure.candidate_contents
        or candidate.manifest.candidate_id != plan.candidate_id
        or candidate.manifest.validation_context_ref != context.validation_context_id
        or candidate.manifest.scope_contract_id
        != context.scope_contract.scope_contract_id
        or candidate.manifest.parent_release_id
        != (
            None
            if context.parent_release is None
            else context.parent_release.release_id
        )
        or candidate.manifest.parent_tree_hash != context.parent_tree_hash
        or candidate.commit_record.commit_record_id != plan.candidate_commit_record_id
        or candidate.source_tree.tree_hash != plan.candidate_tree_hash
        or candidate.manifest.scope_contract_id != plan.scope_contract_id
        or candidate.manifest.parent_release_id != plan.parent_release_id
        or plan.configuration_fingerprint != settings.configuration_fingerprint
        or (
            plan.parent_tree_hash is not None
            and candidate.manifest.parent_tree_hash != plan.parent_tree_hash
        )
    ):
        raise TaskEvaluationRequestPreparationError(
            "task evaluation candidate differs from reserved plan authority"
        )


def _task_evaluation_legs(
    *,
    candidate: VerifiedTaskEvaluationCandidate,
    parent: VerifiedTaskEvaluationParent | None,
) -> tuple[TaskEvaluationExpertLeg, ...]:
    candidate_leg = TaskEvaluationExpertLeg.mint(
        kind=TaskEvaluationLegKind.CANDIDATE,
        expert_artifact_id=candidate.manifest.candidate_id,
        expert_source_receipt_id=candidate.commit_record.commit_record_id,
        expert_tree_hash=candidate.source_tree.tree_hash,
        exact_dependency_ids=tuple(
            sorted(
                (
                    candidate.manifest.candidate_id,
                    candidate.commit_record.commit_record_id,
                )
            )
        ),
    )
    if parent is None:
        return (candidate_leg,)
    parent_receipt = parent.parent_tree_receipt
    parent_leg = TaskEvaluationExpertLeg.mint(
        kind=TaskEvaluationLegKind.PARENT_CONTROL,
        expert_artifact_id=parent.release_manifest.release_id,
        expert_source_receipt_id=parent_receipt.parent_tree_receipt_id,
        expert_tree_hash=parent_receipt.parent_tree_hash,
        exact_dependency_ids=tuple(
            sorted(
                (
                    parent.release_manifest.release_id,
                    parent_receipt.parent_tree_receipt_id,
                )
            )
        ),
    )
    return tuple(sorted((candidate_leg, parent_leg), key=lambda leg: leg.leg_id))


def _task_evaluation_case(
    *,
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    provenance_binding_id: str,
    compute_binding: TaskEvaluationComputeBinding,
    legs: tuple[TaskEvaluationExpertLeg, ...],
) -> TaskEvaluationCase:
    plan = plan_reservation.evaluation_plan
    provenances = tuple(
        provenance
        for provenance in plan.provenance_bindings
        if provenance.provenance_binding_id == provenance_binding_id
    )
    if len(provenances) != 1 or provenances[0].adapter_case is None:
        raise TaskEvaluationRequestPreparationError(
            "task evaluation case lacks unique adapter provenance"
        )
    provenance = provenances[0]
    signed_case = provenance.adapter_case
    cells = tuple(
        cell
        for cell in plan.evaluation_cells
        if cell.provenance_binding_id == provenance_binding_id
    )
    dependencies = {
        provenance.adapter_authority_id,
        provenance.provenance_binding_id,
        signed_case.release_matrix_case_id,
        provenance.task_context_binding.task_context_binding_id,
        signed_case.independence_group.independence_group_id,
        *(cell.evaluation_cell_id for cell in cells),
        *provenance.evaluation_fingerprint_ids,
        *provenance.starting_artifact_ids,
        compute_binding.compute_binding_id,
        *(leg.leg_id for leg in legs),
        *(dependency_id for leg in legs for dependency_id in leg.exact_dependency_ids),
    }
    return TaskEvaluationCase.mint(
        adapter_authority_id=provenance.adapter_authority_id,
        provenance_binding_id=provenance.provenance_binding_id,
        release_matrix_case_id=signed_case.release_matrix_case_id,
        task_context_binding_id=(
            provenance.task_context_binding.task_context_binding_id
        ),
        independence_group_id=(signed_case.independence_group.independence_group_id),
        evaluation_cell_ids=tuple(sorted(cell.evaluation_cell_id for cell in cells)),
        evaluation_fingerprint_ids=provenance.evaluation_fingerprint_ids,
        starting_artifact_ids=provenance.starting_artifact_ids,
        compute_binding=compute_binding,
        legs=legs,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
