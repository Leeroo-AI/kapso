"""Sealed publication of factual expert release-matrix evidence."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from kapso.cross_run.expert.promotion_evidence import (
    derive_expert_release_matrix_report,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    CompletedTaskEvaluationExecution,
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertReleaseMatrixStageCommitResult,
        ExpertValidationStore,
    )


class ExpertReleaseMatrixStageError(ValueError):
    """Release-matrix stage publication lacks exact sealed authority."""


_RELEASE_MATRIX_STAGE_EXECUTION_SEAL = object()


class ExpertReleaseMatrixStageExecution:
    """One-shot process-local authority over one factual matrix reduction."""

    __slots__ = (
        "_consumed",
        "_coordinator",
        "_execution_store",
        "_owner_process_id",
        "_validation_store",
        "completed_execution",
        "prepared_request",
        "reservation_snapshot",
        "stage_result",
    )

    def __init__(
        self,
        seal: object,
        coordinator: ExpertReleaseMatrixStageCoordinator,
        *,
        validation_store: ExpertValidationStore,
        execution_store: ExpertTaskEvaluationExecutionStore,
        completed_execution: CompletedTaskEvaluationExecution,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
        stage_result: ExpertReleaseMatrixStageResultRecord,
    ) -> None:
        if seal is not _RELEASE_MATRIX_STAGE_EXECUTION_SEAL:
            raise ExpertReleaseMatrixStageError(
                "release matrix stage execution is not coordinator sealed"
            )
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_validation_store", validation_store)
        object.__setattr__(self, "_execution_store", execution_store)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "completed_execution", completed_execution)
        object.__setattr__(self, "reservation_snapshot", reservation_snapshot)
        object.__setattr__(self, "prepared_request", prepared_request)
        object.__setattr__(self, "stage_result", stage_result)

    def __setattr__(self, name, value) -> None:
        raise ExpertReleaseMatrixStageError(
            "release matrix stage execution is immutable"
        )

    def _require_bound(
        self,
        coordinator: object,
        validation_store: object,
        execution_store: object,
    ) -> None:
        if (
            self._consumed
            or self._coordinator is not coordinator
            or self._validation_store is not validation_store
            or self._execution_store is not execution_store
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertReleaseMatrixStageError(
                "release matrix stage execution is consumed or foreign"
            )

    def _consume(
        self,
        coordinator: object,
        validation_store: object,
        execution_store: object,
    ) -> None:
        self._require_bound(coordinator, validation_store, execution_store)
        object.__setattr__(self, "_consumed", True)


class ExpertReleaseMatrixStageCoordinator:
    """Derive a factual matrix and atomically publish its typed stage result."""

    def __init__(
        self,
        *,
        validation_store: ExpertValidationStore,
        execution_store: ExpertTaskEvaluationExecutionStore,
    ) -> None:
        if type(execution_store) is not ExpertTaskEvaluationExecutionStore:
            raise ExpertReleaseMatrixStageError(
                "release matrix stage requires the canonical execution store"
            )
        self.validation_store = validation_store
        self.execution_store = execution_store
        validation_store._bind_release_matrix_stage_authority(self)

    def publish_completed(
        self,
        *,
        completed_execution: CompletedTaskEvaluationExecution,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertReleaseMatrixStageCommitResult:
        if (
            type(completed_execution) is not CompletedTaskEvaluationExecution
            or type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
            or type(prepared_request) is not PreparedTaskEvaluationRequest
        ):
            raise ExpertReleaseMatrixStageError(
                "release matrix stage requires exact completed authorities"
            )
        replayed = self.validation_store.reopen_or_replay_release_matrix_stage(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared_request,
        )
        if replayed is not None:
            return replayed
        report = derive_expert_release_matrix_report(
            validation_store=self.validation_store,
            execution_store=self.execution_store,
            completed_execution=completed_execution,
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared_request,
        )
        reservation = reservation_snapshot.reservation
        request = reservation_snapshot.request
        dependencies = {
            reservation.authorization_transition_id,
            reservation.authorization_state_id,
            reservation.validation_attempt_id,
            reservation.candidate_id,
            reservation.scope_contract_id,
            reservation.plan_reservation_operation_id,
            reservation.reservation_id,
            request.validation_policy_id,
            report.release_matrix_report_id,
            *report.exact_dependency_ids,
        }
        if reservation.observed_current_release_id is not None:
            dependencies.add(reservation.observed_current_release_id)
        stage_result = ExpertReleaseMatrixStageResultRecord.mint(
            validation_attempt_id=reservation.validation_attempt_id,
            authorization_transition_id=reservation.authorization_transition_id,
            authorization_state_id=reservation.authorization_state_id,
            candidate_id=reservation.candidate_id,
            candidate_tree_hash=reservation.candidate_tree_hash,
            scope_contract_id=reservation.scope_contract_id,
            source_base_release_id=reservation.observed_current_release_id,
            validation_policy_id=request.validation_policy_id,
            configuration_fingerprint=request.configuration_fingerprint,
            plan_reservation_operation_id=(reservation.plan_reservation_operation_id),
            task_evaluation_reservation_id=reservation.reservation_id,
            release_matrix_report=report,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        execution = ExpertReleaseMatrixStageExecution(
            _RELEASE_MATRIX_STAGE_EXECUTION_SEAL,
            self,
            validation_store=self.validation_store,
            execution_store=self.execution_store,
            completed_execution=completed_execution,
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared_request,
            stage_result=stage_result,
        )
        return self.validation_store.publish_release_matrix_stage(execution)
