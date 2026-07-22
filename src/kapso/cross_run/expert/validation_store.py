"""Crash-atomic expert validation history and operation replay."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    ExpertValidationAuthorityInvalidation,
    ExpertValidationAttempt,
    ExpertValidationStage,
    StrictContract,
)
from kapso.cross_run.expert.validation import (
    ExpertEligibilityResult,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
    validate_source_replay_request_authority_shape,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayPairedComparisonReceipt,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayStageDecision,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
    SourceReplayDecisionPublicationFence,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest
from kapso.cross_run.settings import (
    ExpertValidationPolicy,
    ExpertValidationSettings,
)


class ExpertValidationStoreError(ValueError):
    """Persisted validation history is incomplete, corrupt, or conflicting."""


class ExpertValidationCompareAndSwapError(ExpertValidationStoreError):
    """A validation operation was reduced from a stale candidate head."""


class ExpertValidationOperationKind(str, Enum):
    START = "start"
    EVALUATOR_RESULT = "evaluator_result"
    SOURCE_REPLAY_RESERVATION = "source_replay_reservation"
    SOURCE_REPLAY_STAGE_RESULT = "source_replay_stage_result"
    AUTHORITY_INVALIDATION = "authority_invalidation"


@dataclass(frozen=True)
class ExpertValidationOperation(StrictContract):
    operation_id: str
    operation_kind: ExpertValidationOperationKind
    candidate_id: str
    expected_transition_id: str | None
    request_record_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_id"

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "operation candidate_id")
        if self.expected_transition_id is not None:
            require_content_id(
                self.expected_transition_id,
                "operation expected_transition_id",
            )
        require_content_id(self.request_record_id, "operation request_record_id")


@dataclass(frozen=True)
class ExpertValidationTransition(StrictContract):
    transition_id: str
    candidate_id: str
    candidate_tree_hash: str
    transition_number: int
    predecessor_transition_id: str | None
    predecessor_state_id: str | None
    target_state_id: str
    latest_attempt_id: str | None
    operation_id: str
    validation_policy_id: str
    configuration_fingerprint: str
    eligibility_decision_id: str | None
    created_attempt_id: str | None
    accepted_stage_result_record_ids: tuple[str, ...]
    transition_stage_result_record_id: str | None
    transition_authority_invalidation_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-transition"
    IDENTITY_FIELD: ClassVar[str] = "transition_id"

    def _validate(self) -> None:
        for value, name in (
            (self.candidate_id, "transition candidate_id"),
            (self.target_state_id, "transition target_state_id"),
            (self.operation_id, "transition operation_id"),
            (self.validation_policy_id, "transition validation_policy_id"),
        ):
            require_content_id(value, name)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.candidate_tree_hash) is None:
            raise ContractValidationError("transition candidate tree hash is invalid")
        if self.transition_number <= 0:
            raise ContractValidationError("transition number must be positive")
        first = self.transition_number == 1
        if (self.predecessor_transition_id is None) != (
            self.predecessor_state_id is None
        ) or first != (self.predecessor_transition_id is None):
            raise ContractValidationError(
                "only the first transition may omit both predecessors"
            )
        for value, name in (
            (self.predecessor_transition_id, "predecessor_transition_id"),
            (self.predecessor_state_id, "predecessor_state_id"),
            (self.latest_attempt_id, "latest_attempt_id"),
            (self.eligibility_decision_id, "eligibility_decision_id"),
            (self.created_attempt_id, "created_attempt_id"),
            (
                self.transition_stage_result_record_id,
                "transition_stage_result_record_id",
            ),
            (
                self.transition_authority_invalidation_id,
                "transition_authority_invalidation_id",
            ),
        ):
            if value is not None:
                require_content_id(value, name)
        if (
            re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                self.configuration_fingerprint,
            )
            is None
        ):
            raise ContractValidationError(
                "transition configuration fingerprint is invalid"
            )
        if len(self.accepted_stage_result_record_ids) != len(
            set(self.accepted_stage_result_record_ids)
        ):
            raise ContractValidationError(
                "accepted stage result records must be unique"
            )
        for result_record_id in self.accepted_stage_result_record_ids:
            require_content_id(
                result_record_id,
                "accepted_stage_result_record_ids",
            )
        request_count = sum(
            value is not None
            for value in (
                self.eligibility_decision_id,
                self.transition_stage_result_record_id,
                self.transition_authority_invalidation_id,
            )
        )
        if request_count != 1:
            raise ContractValidationError(
                "transition must contain exactly one start, result, or invalidation request"
            )
        start = self.eligibility_decision_id is not None
        if not start and self.created_attempt_id is not None:
            raise ContractValidationError(
                "only a start transition may create a validation attempt"
            )
        if self.created_attempt_id is not None and (
            self.latest_attempt_id != self.created_attempt_id
        ):
            raise ContractValidationError(
                "created validation attempt must become the latest attempt"
            )


@dataclass(frozen=True)
class ExpertValidationJournal(StrictContract):
    candidate_id: str
    candidate_tree_hash: str
    transition_ids: tuple[str, ...]
    operation_transition_ids: Mapping[str, str]

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "journal candidate_id")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.candidate_tree_hash) is None:
            raise ContractValidationError("journal candidate tree hash is invalid")
        if len(self.transition_ids) != len(set(self.transition_ids)):
            raise ContractValidationError("journal transitions must be unique")
        for transition_id in self.transition_ids:
            require_content_id(transition_id, "journal transition_ids")
        for operation_id, transition_id in self.operation_transition_ids.items():
            require_content_id(operation_id, "journal operation ID")
            require_content_id(transition_id, "journal operation transition ID")
            if transition_id not in self.transition_ids:
                raise ContractValidationError(
                    "journal operation names an absent transition"
                )


@dataclass(frozen=True)
class ExpertValidationSnapshot:
    transition: ExpertValidationTransition
    state: ExpertCandidateValidationState
    latest_attempt: ExpertValidationAttempt | None
    accepted_stage_results: tuple[
        ExpertEvaluatorResultRecord | ExpertSourceReplayStageResultRecord, ...
    ]

    @property
    def predecessor(self) -> ExpertValidationPredecessor:
        return ExpertValidationPredecessor(
            latest_attempt=self.latest_attempt,
            state=self.state,
        )


@dataclass(frozen=True)
class ExpertValidationCommitResult:
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertSourceReplayReservationCommitResult:
    reservation: ExpertSourceReplayExecutionReservation
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertSourceReplayReservationSnapshot:
    reservation: ExpertSourceReplayExecutionReservation
    request: ExpertSourceReplayExecutionRequest
    snapshot: ExpertValidationSnapshot

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.reservation,
                ExpertSourceReplayExecutionReservation,
            )
            or not isinstance(self.request, ExpertSourceReplayExecutionRequest)
            or not isinstance(self.snapshot, ExpertValidationSnapshot)
            or self.snapshot.latest_attempt is None
        ):
            raise ExpertValidationStoreError(
                "source replay reservation snapshot is incomplete"
            )
        transition = self.snapshot.transition
        state = self.snapshot.state
        attempt = self.snapshot.latest_attempt
        if (
            self.reservation.execution_request_id != self.request.execution_request_id
            or self.reservation.authorization_transition_id != transition.transition_id
            or self.reservation.validation_attempt_id != attempt.validation_attempt_id
            or self.reservation.validation_attempt_id
            != self.request.validation_attempt_id
            or self.reservation.authorization_state_id != state.validation_state_id
            or self.reservation.authorization_state_id
            != self.request.authorization_state_id
            or self.reservation.candidate_id != self.request.candidate_id
            or self.reservation.candidate_id != state.candidate_id
            or self.reservation.candidate_id != transition.candidate_id
            or self.reservation.candidate_tree_hash != self.request.candidate_tree_hash
            or self.reservation.candidate_tree_hash != state.candidate_tree_hash
            or self.reservation.observed_parent_release_id
            != self.request.parent_release_id
        ):
            raise ExpertValidationStoreError(
                "source replay reservation snapshot authority is inconsistent"
            )


@dataclass(frozen=True)
class ExpertSourceReplayStageCommitResult:
    stage_result: ExpertSourceReplayStageResultRecord
    snapshot: ExpertValidationSnapshot
    replayed: bool


_SOURCE_REPLAY_PUBLICATION_PERMIT_SEAL = object()


class SourceReplayDecisionPublicationPermit:
    """One-shot process-local authority for source-stage validation CAS."""

    __slots__ = (
        "_store",
        "_coordinator",
        "_owner_process_id",
        "_consumed",
        "reservation_snapshot",
        "prepared_request",
        "stage_result",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertValidationStore,
        coordinator: object,
        reservation_snapshot: ExpertSourceReplayReservationSnapshot,
        prepared_request: PreparedExpertSourceReplayRequest,
        stage_result: ExpertSourceReplayStageResultRecord,
    ) -> None:
        if seal is not _SOURCE_REPLAY_PUBLICATION_PERMIT_SEAL:
            raise ExpertValidationStoreError(
                "source replay publication permit is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "reservation_snapshot", reservation_snapshot)
        object.__setattr__(self, "prepared_request", prepared_request)
        object.__setattr__(self, "stage_result", stage_result)

    def __setattr__(self, name, value) -> None:
        raise ExpertValidationStoreError(
            "source replay publication permit is immutable"
        )

    def _consume(
        self,
        store: ExpertValidationStore,
        coordinator: object,
    ) -> None:
        self._require_bound(store, coordinator)
        object.__setattr__(self, "_consumed", True)

    def _require_bound(
        self,
        store: ExpertValidationStore,
        coordinator: object,
    ) -> None:
        if (
            self._consumed
            or self._store is not store
            or self._coordinator is not coordinator
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertValidationStoreError(
                "source replay publication permit is consumed or foreign"
            )


class ExpertValidationStore:
    """Publish linear validation transitions through one atomic candidate journal."""

    def __init__(
        self,
        root: Path,
        state_root: Path,
        settings: ExpertValidationSettings,
        reducer: ExpertValidationReducer,
    ) -> None:
        self._validate_state_root(state_root)
        if (
            not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != state_root
        ):
            raise ExpertValidationStoreError(
                "validation store must be a direct normalized child of its state root"
            )
        if reducer.settings != settings:
            raise ExpertValidationStoreError(
                "validation reducer differs from store configuration"
            )
        self.root = root
        self.state_root = state_root
        self.settings = settings
        self.reducer = reducer
        self.object_root = root / "objects"
        self.configuration_root = root / "configurations"
        self.journal_root = root / "journals"
        self.staging_root = root / "staging"
        self._source_replay_publication_coordinator = None
        initialization_lock = state_root / f".{root.name}.initialization.lock"
        with _ValidationStoreLock(initialization_lock, exclusive=True, create=True):
            self._prepare_layout()

    def current(self, candidate_id: str) -> ExpertValidationPredecessor | None:
        snapshot = self.snapshot(candidate_id)
        return None if snapshot is None else snapshot.predecessor

    def snapshot(self, candidate_id: str) -> ExpertValidationSnapshot | None:
        require_content_id(candidate_id, "candidate_id")
        with self._lock(exclusive=False):
            return self._snapshot_unlocked(candidate_id)

    def reopen_or_replay_source_replay_publication(
        self,
        *,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[
        ExpertSourceReplayStageCommitResult | None,
        ExpertSourceReplayReservationSnapshot | None,
    ]:
        prepared = self._require_exact_reservation_prepared(
            reservation,
            prepared_request,
        )
        operation = self._source_replay_stage_operation(reservation)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            snapshot = self._resolved_operation_unlocked(journal, operation)
            if snapshot is not None:
                result = self._source_stage_result_for_transition_unlocked(
                    snapshot.transition
                )
                if result.execution_request_id != prepared.request.execution_request_id:
                    raise ExpertValidationStoreError(
                        "replayed source stage result differs from prepared request"
                    )
                return (
                    ExpertSourceReplayStageCommitResult(
                        stage_result=result,
                        snapshot=snapshot,
                        replayed=True,
                    ),
                    None,
                )
            current_reservation = self._current_source_replay_reservation_unlocked(
                journal,
                reservation.authorization_transition_id,
            )
            if current_reservation.reservation != reservation:
                raise ExpertValidationCompareAndSwapError(
                    "another source replay reservation owns the validation head"
                )
            return None, current_reservation

    def _bind_source_replay_publication_authority(
        self,
        coordinator: ExpertSourceReplayDecisionPublicationCoordinator,
    ) -> None:
        if type(
            coordinator
        ) is not ExpertSourceReplayDecisionPublicationCoordinator or (
            self._source_replay_publication_coordinator is not None
            and self._source_replay_publication_coordinator is not coordinator
        ):
            raise ExpertValidationStoreError(
                "validation store already has another publication coordinator"
            )
        self._source_replay_publication_coordinator = coordinator

    def _seal_source_replay_publication_authority(
        self,
        *,
        coordinator: object,
        reservation_snapshot: ExpertSourceReplayReservationSnapshot,
        prepared_request: PreparedExpertSourceReplayRequest,
        stage_result: ExpertSourceReplayStageResultRecord,
    ) -> SourceReplayDecisionPublicationPermit:
        if (
            self._source_replay_publication_coordinator is not coordinator
            or type(coordinator) is not ExpertSourceReplayDecisionPublicationCoordinator
            or not isinstance(
                reservation_snapshot,
                ExpertSourceReplayReservationSnapshot,
            )
            or type(stage_result) is not ExpertSourceReplayStageResultRecord
        ):
            raise ExpertValidationStoreError(
                "source replay publication lacks its bound coordinator authority"
            )
        prepared = self._require_exact_reservation_prepared(
            reservation_snapshot.reservation,
            prepared_request,
        )
        reservation = reservation_snapshot.reservation
        request = prepared.request
        if (
            reservation_snapshot.request != request
            or stage_result.reservation_id != reservation.reservation_id
            or stage_result.execution_request_id != request.execution_request_id
            or stage_result.authorization_transition_id
            != reservation.authorization_transition_id
            or stage_result.authorization_state_id != reservation.authorization_state_id
            or stage_result.validation_attempt_id != reservation.validation_attempt_id
            or stage_result.candidate_id != reservation.candidate_id
            or stage_result.candidate_tree_hash != reservation.candidate_tree_hash
            or stage_result.validation_policy_id != request.validation_policy_id
            or stage_result.configuration_fingerprint
            != request.configuration_fingerprint
        ):
            raise ExpertValidationStoreError(
                "source replay publication result differs from its reservation"
            )
        return SourceReplayDecisionPublicationPermit(
            _SOURCE_REPLAY_PUBLICATION_PERMIT_SEAL,
            self,
            coordinator,
            reservation_snapshot,
            prepared,
            stage_result,
        )

    def _commit_source_replay_publication(
        self,
        *,
        coordinator: object,
        publication_permit: SourceReplayDecisionPublicationPermit,
    ) -> ExpertSourceReplayStageCommitResult:
        if type(publication_permit) is not SourceReplayDecisionPublicationPermit:
            raise ExpertValidationStoreError(
                "source replay publication requires its live one-shot permit"
            )
        publication_permit._require_bound(self, coordinator)
        reservation_snapshot = publication_permit.reservation_snapshot
        reservation = reservation_snapshot.reservation
        prepared = self._require_exact_reservation_prepared(
            reservation,
            publication_permit.prepared_request,
        )
        result = publication_permit.stage_result
        operation = self._source_replay_stage_operation(reservation)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertSourceReplayStageCommitResult(
                    stage_result=(
                        self._source_stage_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            observed_reservation = self._current_source_replay_reservation_unlocked(
                journal,
                reservation.authorization_transition_id,
            )
            if observed_reservation != reservation_snapshot:
                raise ExpertValidationCompareAndSwapError(
                    "source replay reservation changed before publication"
                )
        if observed_reservation.snapshot.latest_attempt is None:
            raise ExpertValidationStoreError(
                "source replay publication has no active validation attempt"
            )
        target_state = self.reducer.advance_source_replay_stage(
            state=observed_reservation.snapshot.state,
            attempt=observed_reservation.snapshot.latest_attempt,
            accepted_results=(observed_reservation.snapshot.accepted_stage_results),
            result=result,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertSourceReplayStageCommitResult(
                    stage_result=(
                        self._source_stage_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current_reservation = self._current_source_replay_reservation_unlocked(
                journal,
                reservation.authorization_transition_id,
            )
            if current_reservation != observed_reservation:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during source replay publication"
                )
            publication_permit._consume(self, coordinator)
            current = current_reservation.snapshot
            if current.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "source replay publication lost its validation attempt"
                )
            accepted_ids = current.transition.accepted_stage_result_record_ids
            if result.outcome is ExpertEvaluatorOutcome.PASSED:
                accepted_ids = (*accepted_ids, result.stage_result_record_id)
            transition = ExpertValidationTransition.mint(
                candidate_id=reservation.candidate_id,
                candidate_tree_hash=reservation.candidate_tree_hash,
                transition_number=len(journal.transition_ids) + 1,
                predecessor_transition_id=current.transition.transition_id,
                predecessor_state_id=current.state.validation_state_id,
                target_state_id=target_state.validation_state_id,
                latest_attempt_id=current.latest_attempt.validation_attempt_id,
                operation_id=operation.operation_id,
                validation_policy_id=current.latest_attempt.validation_policy_id,
                configuration_fingerprint=(
                    current.latest_attempt.configuration_fingerprint
                ),
                eligibility_decision_id=None,
                created_attempt_id=None,
                accepted_stage_result_record_ids=accepted_ids,
                transition_stage_result_record_id=result.stage_result_record_id,
                transition_authority_invalidation_id=None,
            )
            self._write_contract_unlocked(result.paired_comparison_receipt)
            self._write_contract_unlocked(result.stage_decision)
            self._write_contract_unlocked(result.publication_authority_fence)
            self._write_contract_unlocked(result)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertSourceReplayStageCommitResult(
                stage_result=result,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def publish_start(
        self,
        *,
        expected_transition_id: str | None,
        eligibility: ExpertEligibilityResult,
    ) -> ExpertValidationCommitResult:
        if expected_transition_id is not None:
            require_content_id(expected_transition_id, "expected_transition_id")
        decision = eligibility.decision
        operation = ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.START,
            candidate_id=decision.candidate_id,
            expected_transition_id=expected_transition_id,
            request_record_id=decision.eligibility_decision_id,
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(decision.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            observed = self._current_from_journal_unlocked(journal)
            self._require_expected_head(observed, expected_transition_id)
            predecessor = None if observed is None else observed.predecessor
        start = self.reducer.start_from_predecessor(
            eligibility=eligibility,
            predecessor=predecessor,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(decision.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current != observed:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during enrollment checks"
                )
            if (
                current is not None
                and start.attempt is None
                and current.state.promotion_state is ExpertPromotionState.INELIGIBLE
                and current.transition.eligibility_decision_id
                == decision.eligibility_decision_id
            ):
                updated = self._bind_operation(journal, operation, current.transition)
                self._write_contract_unlocked(operation)
                self._publish_journal_unlocked(updated)
                replay = self._snapshot_at_unlocked(
                    updated,
                    current.transition.transition_id,
                )
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            transition = self._start_transition(
                journal,
                operation,
                eligibility,
                start.state,
                start.attempt,
                predecessor,
            )
            self._write_configuration_unlocked()
            self._write_contract_unlocked(eligibility.policy)
            self._write_contract_unlocked(decision)
            if start.attempt is not None:
                self._write_contract_unlocked(start.attempt)
            self._write_contract_unlocked(start.state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertValidationCommitResult(
                snapshot=self._snapshot_at_unlocked(updated, transition.transition_id),
                replayed=False,
            )

    def publish_evaluator_result(
        self,
        *,
        candidate_id: str,
        expected_transition_id: str,
        result: ExpertEvaluatorResultRecord,
    ) -> ExpertValidationCommitResult:
        require_content_id(candidate_id, "candidate_id")
        require_content_id(expected_transition_id, "expected_transition_id")
        operation = ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.EVALUATOR_RESULT,
            candidate_id=candidate_id,
            expected_transition_id=expected_transition_id,
            request_record_id=result.evaluator_result_record_id,
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            observed = self._current_from_journal_unlocked(journal)
            self._require_expected_head(observed, expected_transition_id)
            if observed is None or observed.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "evaluator result requires a current validation attempt"
                )
        target_state = self.reducer.advance_evaluator_stage(
            state=observed.state,
            attempt=observed.latest_attempt,
            accepted_results=observed.accepted_stage_results,
            result=result,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current != observed:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during evaluator checks"
                )
            accepted_ids = current.transition.accepted_stage_result_record_ids
            if result.evaluator_run.outcome is ExpertEvaluatorOutcome.PASSED:
                accepted_ids = (
                    *accepted_ids,
                    result.evaluator_result_record_id,
                )
            transition = ExpertValidationTransition.mint(
                candidate_id=candidate_id,
                candidate_tree_hash=current.state.candidate_tree_hash,
                transition_number=len(journal.transition_ids) + 1,
                predecessor_transition_id=current.transition.transition_id,
                predecessor_state_id=current.state.validation_state_id,
                target_state_id=target_state.validation_state_id,
                latest_attempt_id=current.latest_attempt.validation_attempt_id,
                operation_id=operation.operation_id,
                validation_policy_id=current.latest_attempt.validation_policy_id,
                configuration_fingerprint=(
                    current.latest_attempt.configuration_fingerprint
                ),
                eligibility_decision_id=None,
                created_attempt_id=None,
                accepted_stage_result_record_ids=accepted_ids,
                transition_stage_result_record_id=(result.evaluator_result_record_id),
                transition_authority_invalidation_id=None,
            )
            self._write_contract_unlocked(result)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertValidationCommitResult(
                snapshot=self._snapshot_at_unlocked(updated, transition.transition_id),
                replayed=False,
            )

    def reserve_source_replay(
        self,
        *,
        expected_transition_id: str,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayReservationCommitResult:
        require_content_id(expected_transition_id, "expected_transition_id")
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertValidationStoreError(
                "source replay reservation requires a verified prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        request = prepared.request
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(request.candidate_id)
            existing = self._source_replay_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if existing is not None:
                reservation, stored_request = existing
                if stored_request != request:
                    raise ExpertValidationCompareAndSwapError(
                        "validation head already reserves another source replay request"
                    )
                current = self._current_from_journal_unlocked(journal)
                self._require_reservation_replay_authority_unlocked(
                    journal,
                    current,
                    reservation,
                    expected_transition_id,
                )
                return ExpertSourceReplayReservationCommitResult(
                    reservation=reservation,
                    snapshot=self._snapshot_at_unlocked(
                        journal,
                        expected_transition_id,
                    ),
                    replayed=True,
                )
            observed = self._current_from_journal_unlocked(journal)
            self._require_expected_head(observed, expected_transition_id)
            if observed is None or observed.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "source replay reservation requires a current validation attempt"
                )
        self.reducer.validate_source_replay_request(
            state=observed.state,
            attempt=observed.latest_attempt,
            accepted_results=observed.accepted_stage_results,
            request=request,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(request.candidate_id)
            existing = self._source_replay_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if existing is not None:
                reservation, stored_request = existing
                if stored_request != request:
                    raise ExpertValidationCompareAndSwapError(
                        "validation head already reserves another source replay request"
                    )
                current = self._current_from_journal_unlocked(journal)
                self._require_reservation_replay_authority_unlocked(
                    journal,
                    current,
                    reservation,
                    expected_transition_id,
                )
                return ExpertSourceReplayReservationCommitResult(
                    reservation=reservation,
                    snapshot=self._snapshot_at_unlocked(
                        journal,
                        expected_transition_id,
                    ),
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current != observed:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during source replay reservation checks"
                )
            reservation = ExpertSourceReplayExecutionReservation.mint(
                execution_request_id=request.execution_request_id,
                authorization_transition_id=current.transition.transition_id,
                validation_attempt_id=current.latest_attempt.validation_attempt_id,
                authorization_state_id=current.state.validation_state_id,
                candidate_id=current.state.candidate_id,
                candidate_tree_hash=current.state.candidate_tree_hash,
                observed_parent_release_id=request.parent_release_id,
                exact_dependency_ids=tuple(
                    sorted(
                        {
                            request.execution_request_id,
                            current.transition.transition_id,
                            current.latest_attempt.validation_attempt_id,
                            current.state.validation_state_id,
                            current.state.candidate_id,
                            request.parent_release_id,
                        }
                    )
                ),
            )
            operation = ExpertValidationOperation.mint(
                operation_kind=(
                    ExpertValidationOperationKind.SOURCE_REPLAY_RESERVATION
                ),
                candidate_id=request.candidate_id,
                expected_transition_id=current.transition.transition_id,
                request_record_id=reservation.reservation_id,
            )
            self._write_contract_unlocked(request)
            self._write_contract_unlocked(reservation)
            self._write_contract_unlocked(operation)
            updated = self._bind_operation(journal, operation, current.transition)
            self._publish_journal_unlocked(updated)
            return ExpertSourceReplayReservationCommitResult(
                reservation=reservation,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    current.transition.transition_id,
                ),
                replayed=False,
            )

    def existing_source_replay_reservation(
        self,
        *,
        expected_transition_id: str,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayReservationCommitResult | None:
        """Read an exact existing reservation without creating durable state."""

        require_content_id(expected_transition_id, "expected_transition_id")
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertValidationStoreError(
                "source replay reservation lookup requires a prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(prepared.request.candidate_id)
            existing = self._source_replay_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if existing is None:
                return None
            reservation, stored_request = existing
            if stored_request != prepared.request:
                raise ExpertValidationCompareAndSwapError(
                    "validation head already reserves another source replay request"
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_reservation_replay_authority_unlocked(
                journal,
                current,
                reservation,
                expected_transition_id,
            )
            return ExpertSourceReplayReservationCommitResult(
                reservation=reservation,
                snapshot=self._snapshot_at_unlocked(
                    journal,
                    expected_transition_id,
                ),
                replayed=True,
            )

    def _require_reservation_replay_authority_unlocked(
        self,
        journal: ExpertValidationJournal,
        current: ExpertValidationSnapshot | None,
        reservation: ExpertSourceReplayExecutionReservation,
        expected_transition_id: str,
    ) -> None:
        if (
            current is not None
            and current.transition.transition_id == expected_transition_id
        ):
            return
        publication_operation = self._source_replay_stage_operation(reservation)
        published = self._resolved_operation_unlocked(
            journal,
            publication_operation,
        )
        if published is None or current != published:
            self._require_expected_head(current, expected_transition_id)
            return
        result = self._source_stage_result_for_transition_unlocked(published.transition)
        if (
            result.reservation_id != reservation.reservation_id
            or result.execution_request_id != reservation.execution_request_id
        ):
            raise ExpertValidationStoreError(
                "published source replay result differs from reservation replay"
            )

    def reopen_source_replay_reservation(
        self,
        *,
        reservation_id: str,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayReservationSnapshot:
        require_content_id(reservation_id, "source replay reservation_id")
        if reservation_id.split(":sha256:", 1)[0] != (
            "expert-source-replay-execution-reservation"
        ):
            raise ExpertValidationStoreError(
                "source replay reservation_id uses the wrong namespace"
            )
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertValidationStoreError(
                "source replay reopen requires a verified prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        request = prepared.request
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(request.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None:
                raise ExpertValidationStoreError(
                    "source replay reservation has no current validation state"
                )
            stored = self._source_replay_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            if stored is None:
                raise ExpertValidationStoreError(
                    "source replay reservation is not bound to the current head"
                )
            reservation, stored_request = stored
            if reservation.reservation_id != reservation_id:
                raise ExpertValidationStoreError(
                    "source replay reservation identity is not current"
                )
            if stored_request != request:
                raise ExpertValidationStoreError(
                    "source replay stored request differs from its prepared closure"
                )
            return ExpertSourceReplayReservationSnapshot(
                reservation=reservation,
                request=stored_request,
                snapshot=current,
            )

    def publish_parent_authority_invalidation(
        self,
        *,
        candidate_id: str,
        expected_validation_state_id: str,
    ) -> ExpertValidationCommitResult:
        require_content_id(candidate_id, "candidate_id")
        require_content_id(
            expected_validation_state_id,
            "expected_validation_state_id",
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            self._validate_journal_unlocked(journal)
            replayed = self._parent_authority_invalidation_snapshot_unlocked(
                journal,
                expected_validation_state_id,
            )
            if replayed is not None:
                return ExpertValidationCommitResult(snapshot=replayed, replayed=True)
            observed = self._current_from_journal_unlocked(journal)
            if observed is None or observed.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "authority invalidation requires a current validation attempt"
                )
            if observed.state.validation_state_id != expected_validation_state_id:
                raise ExpertValidationCompareAndSwapError(
                    "validation candidate head changed before publication"
                )
        reduced = self.reducer.invalidate_parent_authority(
            state=observed.state,
            attempt=observed.latest_attempt,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(candidate_id)
            self._validate_journal_unlocked(journal)
            replayed = self._parent_authority_invalidation_snapshot_unlocked(
                journal,
                expected_validation_state_id,
            )
            if replayed is not None:
                return ExpertValidationCommitResult(snapshot=replayed, replayed=True)
            current = self._current_from_journal_unlocked(journal)
            if (
                current is None
                or current.latest_attempt is None
                or current.state.validation_state_id != expected_validation_state_id
                or current != observed
            ):
                raise ExpertValidationCompareAndSwapError(
                    "validation candidate head changed during authority checks"
                )
            invalidation = reduced.invalidation
            operation = ExpertValidationOperation.mint(
                operation_kind=(ExpertValidationOperationKind.AUTHORITY_INVALIDATION),
                candidate_id=candidate_id,
                expected_transition_id=current.transition.transition_id,
                request_record_id=invalidation.authority_invalidation_id,
            )
            transition = ExpertValidationTransition.mint(
                candidate_id=candidate_id,
                candidate_tree_hash=current.state.candidate_tree_hash,
                transition_number=len(journal.transition_ids) + 1,
                predecessor_transition_id=current.transition.transition_id,
                predecessor_state_id=current.state.validation_state_id,
                target_state_id=reduced.state.validation_state_id,
                latest_attempt_id=current.latest_attempt.validation_attempt_id,
                operation_id=operation.operation_id,
                validation_policy_id=current.latest_attempt.validation_policy_id,
                configuration_fingerprint=(
                    current.latest_attempt.configuration_fingerprint
                ),
                eligibility_decision_id=None,
                created_attempt_id=None,
                accepted_stage_result_record_ids=(
                    current.transition.accepted_stage_result_record_ids
                ),
                transition_stage_result_record_id=None,
                transition_authority_invalidation_id=(
                    invalidation.authority_invalidation_id
                ),
            )
            self._write_contract_unlocked(invalidation)
            self._write_contract_unlocked(reduced.state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertValidationCommitResult(
                snapshot=self._snapshot_at_unlocked(updated, transition.transition_id),
                replayed=False,
            )

    def _parent_authority_invalidation_snapshot_unlocked(
        self,
        journal: ExpertValidationJournal,
        expected_validation_state_id: str,
    ) -> ExpertValidationSnapshot | None:
        for transition_id in journal.transition_ids:
            transition = self._read_contract_unlocked(
                transition_id,
                ExpertValidationTransition,
            )
            if (
                transition.predecessor_state_id == expected_validation_state_id
                and transition.transition_authority_invalidation_id is not None
            ):
                return self._snapshot_at_unlocked(journal, transition_id)
        return None

    def _start_transition(
        self,
        journal: ExpertValidationJournal,
        operation: ExpertValidationOperation,
        eligibility: ExpertEligibilityResult,
        state: ExpertCandidateValidationState,
        attempt: ExpertValidationAttempt | None,
        predecessor: ExpertValidationPredecessor | None,
    ) -> ExpertValidationTransition:
        previous = self._current_from_journal_unlocked(journal)
        latest_attempt = attempt
        if latest_attempt is None and predecessor is not None:
            latest_attempt = predecessor.latest_attempt
        return ExpertValidationTransition.mint(
            candidate_id=state.candidate_id,
            candidate_tree_hash=state.candidate_tree_hash,
            transition_number=len(journal.transition_ids) + 1,
            predecessor_transition_id=(
                None if previous is None else previous.transition.transition_id
            ),
            predecessor_state_id=state.predecessor_state_id,
            target_state_id=state.validation_state_id,
            latest_attempt_id=(
                None if latest_attempt is None else latest_attempt.validation_attempt_id
            ),
            operation_id=operation.operation_id,
            validation_policy_id=eligibility.decision.validation_policy_id,
            configuration_fingerprint=(eligibility.decision.configuration_fingerprint),
            eligibility_decision_id=(eligibility.decision.eligibility_decision_id),
            created_attempt_id=(
                None if attempt is None else attempt.validation_attempt_id
            ),
            accepted_stage_result_record_ids=(),
            transition_stage_result_record_id=None,
            transition_authority_invalidation_id=None,
        )

    @staticmethod
    def _require_exact_reservation_prepared(
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> PreparedExpertSourceReplayRequest:
        if not isinstance(
            reservation,
            ExpertSourceReplayExecutionReservation,
        ) or not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertValidationStoreError(
                "source replay publication requires typed reservation authority"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        request = prepared.request
        if (
            reservation.execution_request_id != request.execution_request_id
            or reservation.validation_attempt_id != request.validation_attempt_id
            or reservation.authorization_state_id != request.authorization_state_id
            or reservation.candidate_id != request.candidate_id
            or reservation.candidate_tree_hash != request.candidate_tree_hash
            or reservation.observed_parent_release_id != request.parent_release_id
        ):
            raise ExpertValidationStoreError(
                "source replay reservation differs from prepared request"
            )
        return prepared

    @staticmethod
    def _source_replay_stage_operation(
        reservation: ExpertSourceReplayExecutionReservation,
    ) -> ExpertValidationOperation:
        return ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.SOURCE_REPLAY_STAGE_RESULT,
            candidate_id=reservation.candidate_id,
            expected_transition_id=reservation.authorization_transition_id,
            request_record_id=reservation.reservation_id,
        )

    def _current_source_replay_reservation_unlocked(
        self,
        journal: ExpertValidationJournal,
        authorization_transition_id: str,
    ) -> ExpertSourceReplayReservationSnapshot:
        current = self._current_from_journal_unlocked(journal)
        self._require_expected_head(current, authorization_transition_id)
        if current is None:
            raise ExpertValidationStoreError(
                "source replay reservation has no validation head"
            )
        stored = self._source_replay_reservation_unlocked(
            journal,
            authorization_transition_id,
        )
        if stored is None:
            raise ExpertValidationStoreError(
                "source replay reservation is absent from its authorization head"
            )
        reservation, request = stored
        return ExpertSourceReplayReservationSnapshot(
            reservation=reservation,
            request=request,
            snapshot=current,
        )

    def _source_stage_result_for_transition_unlocked(
        self,
        transition: ExpertValidationTransition,
    ) -> ExpertSourceReplayStageResultRecord:
        result_record_id = transition.transition_stage_result_record_id
        if (
            result_record_id is None
            or result_record_id.split(":sha256:", 1)[0]
            != "expert-source-replay-stage-result"
        ):
            raise ExpertValidationStoreError(
                "validation transition does not contain a source replay result"
            )
        return self._read_contract_unlocked(
            result_record_id,
            ExpertSourceReplayStageResultRecord,
        )

    def _read_stage_result_unlocked(
        self,
        result_record_id: str,
    ) -> ExpertEvaluatorResultRecord | ExpertSourceReplayStageResultRecord:
        namespace = result_record_id.split(":sha256:", 1)[0]
        if namespace == "expert-evaluator-result-record":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertEvaluatorResultRecord,
            )
        if namespace == "expert-source-replay-stage-result":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertSourceReplayStageResultRecord,
            )
        raise ExpertValidationStoreError(
            "validation stage result uses an unsupported namespace"
        )

    @staticmethod
    def _stage_result_projection(
        result: ExpertEvaluatorResultRecord | ExpertSourceReplayStageResultRecord,
    ) -> tuple[
        ExpertValidationStage,
        str,
        ExpertEvaluatorOutcome,
        str,
        str,
        str,
    ]:
        if type(result) is ExpertEvaluatorResultRecord:
            run = result.evaluator_run
            return (
                run.stage,
                result.evaluator_result_record_id,
                run.outcome,
                run.validation_attempt_id,
                run.candidate_id,
                run.candidate_tree_hash,
            )
        if type(result) is ExpertSourceReplayStageResultRecord:
            return (
                ExpertValidationStage.SOURCE_RUN_REPLAY,
                result.stage_result_record_id,
                result.outcome,
                result.validation_attempt_id,
                result.candidate_id,
                result.candidate_tree_hash,
            )
        raise ExpertValidationStoreError("validation stage result type is unsupported")

    def _snapshot_unlocked(
        self,
        candidate_id: str,
    ) -> ExpertValidationSnapshot | None:
        journal = self._read_journal_unlocked(candidate_id)
        return self._current_from_journal_unlocked(journal)

    def _current_from_journal_unlocked(
        self,
        journal: ExpertValidationJournal,
    ) -> ExpertValidationSnapshot | None:
        self._validate_journal_unlocked(journal)
        if not journal.transition_ids:
            return None
        return self._snapshot_at_unlocked(journal, journal.transition_ids[-1])

    def _snapshot_at_unlocked(
        self,
        journal: ExpertValidationJournal,
        transition_id: str,
    ) -> ExpertValidationSnapshot:
        if transition_id not in journal.transition_ids:
            raise ExpertValidationStoreError(
                "validation snapshot transition is absent from its journal"
            )
        transition = self._read_contract_unlocked(
            transition_id,
            ExpertValidationTransition,
        )
        state = self._read_contract_unlocked(
            transition.target_state_id,
            ExpertCandidateValidationState,
        )
        latest_attempt = (
            None
            if transition.latest_attempt_id is None
            else self._read_contract_unlocked(
                transition.latest_attempt_id,
                ExpertValidationAttempt,
            )
        )
        active_attempt_transition = (
            transition.created_attempt_id is not None
            or transition.transition_stage_result_record_id is not None
            or transition.transition_authority_invalidation_id is not None
        )
        if latest_attempt is not None and (
            latest_attempt.candidate_id != transition.candidate_id
            or latest_attempt.candidate_tree_hash != transition.candidate_tree_hash
            or (
                active_attempt_transition
                and (
                    latest_attempt.validation_policy_id
                    != transition.validation_policy_id
                    or latest_attempt.configuration_fingerprint
                    != transition.configuration_fingerprint
                )
            )
        ):
            raise ExpertValidationStoreError(
                "latest validation attempt differs from its transition"
            )
        accepted_records = tuple(
            self._read_stage_result_unlocked(result_record_id)
            for result_record_id in transition.accepted_stage_result_record_ids
        )
        return ExpertValidationSnapshot(
            transition=transition,
            state=state,
            latest_attempt=latest_attempt,
            accepted_stage_results=accepted_records,
        )

    def _validate_journal_unlocked(self, journal: ExpertValidationJournal) -> None:
        previous_transition = None
        previous_state = None
        previous_latest_attempt = None
        for position, transition_id in enumerate(journal.transition_ids, start=1):
            transition = self._read_contract_unlocked(
                transition_id,
                ExpertValidationTransition,
            )
            state = self._read_contract_unlocked(
                transition.target_state_id,
                ExpertCandidateValidationState,
            )
            operation = self._read_contract_unlocked(
                transition.operation_id,
                ExpertValidationOperation,
            )
            if (
                transition.transition_number != position
                or transition.candidate_id != journal.candidate_id
                or transition.candidate_tree_hash != journal.candidate_tree_hash
                or state.candidate_id != journal.candidate_id
                or state.candidate_tree_hash != journal.candidate_tree_hash
                or operation.candidate_id != journal.candidate_id
                or journal.operation_transition_ids.get(operation.operation_id)
                != transition.transition_id
                or transition.predecessor_transition_id
                != (
                    None
                    if previous_transition is None
                    else previous_transition.transition_id
                )
                or transition.predecessor_state_id
                != (
                    None
                    if previous_state is None
                    else previous_state.validation_state_id
                )
                or state.predecessor_state_id != transition.predecessor_state_id
            ):
                raise ExpertValidationStoreError(
                    "validation journal transition lineage is inconsistent"
                )
            self._validate_transition_closure_unlocked(
                journal,
                transition,
                state,
                operation,
                previous_transition,
                previous_latest_attempt,
            )
            previous_transition = transition
            previous_state = state
            previous_latest_attempt = (
                None
                if transition.latest_attempt_id is None
                else self._read_contract_unlocked(
                    transition.latest_attempt_id,
                    ExpertValidationAttempt,
                )
            )
        transitions = {
            transition_id: self._read_contract_unlocked(
                transition_id,
                ExpertValidationTransition,
            )
            for transition_id in journal.transition_ids
        }
        reserved_transition_ids: set[str] = set()
        for operation_id, transition_id in journal.operation_transition_ids.items():
            operation = self._read_contract_unlocked(
                operation_id,
                ExpertValidationOperation,
            )
            transition = transitions[transition_id]
            if operation.candidate_id != journal.candidate_id:
                raise ExpertValidationStoreError(
                    "validation operation belongs to another candidate"
                )
            if operation_id == transition.operation_id:
                continue
            ineligible_start_replay = (
                operation.operation_kind is ExpertValidationOperationKind.START
                and transition.eligibility_decision_id is not None
                and operation.request_record_id == transition.eligibility_decision_id
                and operation.expected_transition_id == transition.transition_id
                and self._read_contract_unlocked(
                    transition.target_state_id,
                    ExpertCandidateValidationState,
                ).promotion_state
                is ExpertPromotionState.INELIGIBLE
            )
            if ineligible_start_replay:
                continue
            if (
                operation.operation_kind
                is ExpertValidationOperationKind.SOURCE_REPLAY_RESERVATION
            ):
                if transition_id in reserved_transition_ids:
                    raise ExpertValidationStoreError(
                        "validation transition has multiple source replay reservations"
                    )
                self._validate_source_replay_reservation_alias_unlocked(
                    operation,
                    transition,
                )
                reserved_transition_ids.add(transition_id)
                continue
            raise ExpertValidationStoreError(
                "validation replay operation does not bind its transition"
            )

    def _validate_source_replay_reservation_alias_unlocked(
        self,
        operation: ExpertValidationOperation,
        transition: ExpertValidationTransition,
    ) -> None:
        reservation = self._read_contract_unlocked(
            operation.request_record_id,
            ExpertSourceReplayExecutionReservation,
        )
        request = self._read_contract_unlocked(
            reservation.execution_request_id,
            ExpertSourceReplayExecutionRequest,
        )
        state = self._read_contract_unlocked(
            transition.target_state_id,
            ExpertCandidateValidationState,
        )
        if transition.latest_attempt_id is None:
            raise ExpertValidationStoreError(
                "source replay reservation requires a validation attempt"
            )
        attempt = self._read_contract_unlocked(
            transition.latest_attempt_id,
            ExpertValidationAttempt,
        )
        persisted_settings = self._read_configuration_unlocked(
            transition.configuration_fingerprint
        )
        validate_source_replay_request_authority_shape(
            state=state,
            attempt=attempt,
            request=request,
            settings=persisted_settings,
            error_type=ExpertValidationStoreError,
        )
        if (
            operation.expected_transition_id != transition.transition_id
            or operation.candidate_id != transition.candidate_id
            or reservation.authorization_transition_id != transition.transition_id
            or reservation.validation_attempt_id != attempt.validation_attempt_id
            or reservation.authorization_state_id != state.validation_state_id
            or reservation.candidate_id != transition.candidate_id
            or reservation.candidate_tree_hash != transition.candidate_tree_hash
            or reservation.observed_parent_release_id != request.parent_release_id
        ):
            raise ExpertValidationStoreError(
                "source replay reservation alias closure is inconsistent"
            )

    def _source_replay_reservation_unlocked(
        self,
        journal: ExpertValidationJournal,
        authorization_transition_id: str,
    ) -> (
        tuple[
            ExpertSourceReplayExecutionReservation,
            ExpertSourceReplayExecutionRequest,
        ]
        | None
    ):
        matches = []
        for operation_id, transition_id in journal.operation_transition_ids.items():
            if transition_id != authorization_transition_id:
                continue
            operation = self._read_contract_unlocked(
                operation_id,
                ExpertValidationOperation,
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.SOURCE_REPLAY_RESERVATION
            ):
                continue
            reservation = self._read_contract_unlocked(
                operation.request_record_id,
                ExpertSourceReplayExecutionReservation,
            )
            request = self._read_contract_unlocked(
                reservation.execution_request_id,
                ExpertSourceReplayExecutionRequest,
            )
            matches.append((reservation, request))
        if len(matches) > 1:
            raise ExpertValidationStoreError(
                "validation transition has multiple source replay reservations"
            )
        return None if not matches else matches[0]

    def _validate_source_stage_transition_unlocked(
        self,
        *,
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
        state: ExpertCandidateValidationState,
        operation: ExpertValidationOperation,
        latest_attempt: ExpertValidationAttempt | None,
        previous_accepted: tuple[str, ...],
        result_record: ExpertSourceReplayStageResultRecord,
    ) -> None:
        if transition.predecessor_state_id is None:
            raise ExpertValidationStoreError(
                "source replay result requires its authorization state"
            )
        predecessor_state = self._read_contract_unlocked(
            transition.predecessor_state_id,
            ExpertCandidateValidationState,
        )
        reservation = self._read_contract_unlocked(
            result_record.reservation_id,
            ExpertSourceReplayExecutionReservation,
        )
        request = self._read_contract_unlocked(
            result_record.execution_request_id,
            ExpertSourceReplayExecutionRequest,
        )
        stored_reservation = self._source_replay_reservation_unlocked(
            journal,
            result_record.authorization_transition_id,
        )
        receipt = self._read_contract_unlocked(
            result_record.paired_comparison_receipt.paired_comparison_receipt_id,
            ExpertSourceReplayPairedComparisonReceipt,
        )
        decision = self._read_contract_unlocked(
            result_record.stage_decision.source_replay_stage_decision_id,
            ExpertSourceReplayStageDecision,
        )
        fence = self._read_contract_unlocked(
            result_record.publication_authority_fence.fence_id,
            SourceReplayDecisionPublicationFence,
        )
        common_invalid = (
            operation.operation_kind
            is not ExpertValidationOperationKind.SOURCE_REPLAY_STAGE_RESULT
            or operation.request_record_id != reservation.reservation_id
            or operation.expected_transition_id
            != reservation.authorization_transition_id
            or transition.predecessor_transition_id
            != reservation.authorization_transition_id
            or latest_attempt is None
            or predecessor_state.promotion_state is not ExpertPromotionState.VALIDATING
            or predecessor_state.next_stage
            is not ExpertValidationStage.SOURCE_RUN_REPLAY
            or predecessor_state.validation_attempt_id
            != latest_attempt.validation_attempt_id
            or tuple(
                item.stage_result_record_id
                for item in predecessor_state.accepted_stage_results
            )
            != previous_accepted
            or state.review_assertion_ids != predecessor_state.review_assertion_ids
            or result_record.validation_attempt_id
            != latest_attempt.validation_attempt_id
            or result_record.authorization_transition_id
            != transition.predecessor_transition_id
            or result_record.authorization_state_id
            != predecessor_state.validation_state_id
            or result_record.candidate_id != transition.candidate_id
            or result_record.candidate_tree_hash != transition.candidate_tree_hash
            or result_record.validation_policy_id != latest_attempt.validation_policy_id
            or result_record.configuration_fingerprint
            != latest_attempt.configuration_fingerprint
            or reservation.execution_request_id != request.execution_request_id
            or reservation.validation_attempt_id != latest_attempt.validation_attempt_id
            or reservation.authorization_state_id
            != predecessor_state.validation_state_id
            or reservation.candidate_id != transition.candidate_id
            or reservation.candidate_tree_hash != transition.candidate_tree_hash
            or request.authorization_state_id != predecessor_state.validation_state_id
            or request.validation_attempt_id != latest_attempt.validation_attempt_id
            or request.validation_policy_id != latest_attempt.validation_policy_id
            or request.configuration_fingerprint
            != latest_attempt.configuration_fingerprint
            or stored_reservation != (reservation, request)
            or receipt != result_record.paired_comparison_receipt
            or decision != result_record.stage_decision
            or fence != result_record.publication_authority_fence
            or state.transition_evidence_id != result_record.stage_result_record_id
        )
        if common_invalid:
            raise ExpertValidationStoreError(
                "source replay result transition closure is inconsistent"
            )
        if result_record.outcome is ExpertEvaluatorOutcome.PASSED:
            valid_state = (
                state.promotion_state is ExpertPromotionState.VALIDATING
                and not state.terminal_evidence_ids
                and state.reason == "stage_source_run_replay_passed"
            )
        else:
            valid_state = (
                result_record.outcome is ExpertEvaluatorOutcome.CANDIDATE_FAILED
                and state.promotion_state is ExpertPromotionState.FAILED
                and state.next_stage is None
                and state.terminal_evidence_ids
                == (result_record.stage_result_record_id,)
                and state.reason == "stage_source_run_replay_candidate_failed"
                and transition.accepted_stage_result_record_ids == previous_accepted
            )
        if not valid_state:
            raise ExpertValidationStoreError(
                "source replay result state semantics are inconsistent"
            )

    def _validate_transition_closure_unlocked(
        self,
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
        state: ExpertCandidateValidationState,
        operation: ExpertValidationOperation,
        previous_transition: ExpertValidationTransition | None,
        previous_latest_attempt: ExpertValidationAttempt | None,
    ) -> None:
        persisted_settings = self._read_configuration_unlocked(
            transition.configuration_fingerprint
        )
        policy = self._read_contract_unlocked(
            transition.validation_policy_id,
            ExpertValidationPolicy,
        )
        if policy != persisted_settings.policy.validation_policy():
            raise ExpertValidationStoreError(
                "persisted validation policy differs from store configuration"
            )
        latest_attempt = (
            None
            if transition.latest_attempt_id is None
            else self._read_contract_unlocked(
                transition.latest_attempt_id,
                ExpertValidationAttempt,
            )
        )
        if transition.created_attempt_id is not None:
            if latest_attempt is None or (
                latest_attempt.predecessor_attempt_id
                != (
                    None
                    if previous_latest_attempt is None
                    else previous_latest_attempt.validation_attempt_id
                )
                or latest_attempt.attempt_number
                != (
                    1
                    if previous_latest_attempt is None
                    else previous_latest_attempt.attempt_number + 1
                )
            ):
                raise ExpertValidationStoreError(
                    "validation attempt lineage is not gap-free"
                )
        elif latest_attempt != previous_latest_attempt:
            raise ExpertValidationStoreError(
                "transition changed the latest attempt without creating one"
            )
        if state.validation_attempt_id is not None and (
            latest_attempt is None
            or state.validation_attempt_id != latest_attempt.validation_attempt_id
        ):
            raise ExpertValidationStoreError(
                "validation state differs from the transition latest attempt"
            )
        accepted_records = tuple(
            self._read_stage_result_unlocked(result_record_id)
            for result_record_id in transition.accepted_stage_result_record_ids
        )
        accepted_projections = tuple(
            self._stage_result_projection(record) for record in accepted_records
        )
        accepted_refs = tuple(
            (stage, record_id)
            for stage, record_id, _outcome, _attempt_id, _candidate_id, _tree_hash in (
                accepted_projections
            )
        )
        state_refs = tuple(
            (item.stage, item.stage_result_record_id)
            for item in state.accepted_stage_results
        )
        if (
            accepted_refs != state_refs
            or (
                latest_attempt is not None
                and tuple(
                    stage
                    for stage, _record_id, _outcome, _attempt_id, _candidate_id, _tree_hash in accepted_projections
                )
                != latest_attempt.required_stages[: len(accepted_records)]
            )
            or (
                state.promotion_state is ExpertPromotionState.VALIDATING
                and (
                    latest_attempt is None
                    or len(accepted_records) >= len(latest_attempt.required_stages)
                    or state.next_stage
                    is not latest_attempt.required_stages[len(accepted_records)]
                )
            )
            or any(
                outcome is not ExpertEvaluatorOutcome.PASSED
                for _stage, _record_id, outcome, _attempt_id, _candidate_id, _tree_hash in accepted_projections
            )
            or any(
                latest_attempt is None
                or attempt_id != latest_attempt.validation_attempt_id
                or candidate_id != transition.candidate_id
                or candidate_tree_hash != transition.candidate_tree_hash
                for _stage, _record_id, _outcome, attempt_id, candidate_id, candidate_tree_hash in accepted_projections
            )
        ):
            raise ExpertValidationStoreError(
                "validation state accepted evidence closure is inconsistent"
            )
        if transition.eligibility_decision_id is not None:
            decision = self._read_contract_unlocked(
                transition.eligibility_decision_id,
                ExpertCandidateEligibilityDecision,
            )
            if (
                operation.operation_kind is not ExpertValidationOperationKind.START
                or operation.request_record_id != decision.eligibility_decision_id
                or decision.candidate_id != state.candidate_id
                or decision.candidate_tree_hash != state.candidate_tree_hash
                or decision.validation_policy_id != transition.validation_policy_id
                or decision.configuration_fingerprint
                != transition.configuration_fingerprint
                or state.transition_evidence_id != decision.eligibility_decision_id
                or operation.expected_transition_id
                != transition.predecessor_transition_id
                or transition.accepted_stage_result_record_ids
            ):
                raise ExpertValidationStoreError(
                    "validation start transition closure is inconsistent"
                )
            if transition.created_attempt_id is not None and (
                latest_attempt is None
                or latest_attempt.eligibility_decision_id
                != decision.eligibility_decision_id
                or latest_attempt.candidate_id != decision.candidate_id
                or latest_attempt.candidate_tree_hash != decision.candidate_tree_hash
                or latest_attempt.candidate_commit_record_id
                != decision.candidate_commit_record_id
                or latest_attempt.scope_contract_id != decision.scope_contract_id
                or latest_attempt.parent_release_id != decision.parent_release_id
                or latest_attempt.validation_policy_id != decision.validation_policy_id
                or latest_attempt.configuration_fingerprint
                != decision.configuration_fingerprint
                or latest_attempt.validation_track != decision.validation_track
                or latest_attempt.required_stages != decision.required_stages
                or latest_attempt.configured_task_family_ids
                != decision.configured_task_family_ids
                or latest_attempt.task_adapter_pins != decision.task_adapter_pins
                or latest_attempt.source_replay_selection
                != decision.source_replay_selection
                or set(latest_attempt.eligibility_dependency_ids)
                != {
                    decision.eligibility_decision_id,
                    *decision.exact_dependency_ids,
                }
            ):
                raise ExpertValidationStoreError(
                    "validation start attempt differs from its eligibility decision"
                )
        elif transition.transition_stage_result_record_id is not None:
            previous_accepted = (
                ()
                if previous_transition is None
                else previous_transition.accepted_stage_result_record_ids
            )
            result_record = self._read_stage_result_unlocked(
                transition.transition_stage_result_record_id
            )
            if type(result_record) is ExpertEvaluatorResultRecord:
                if (
                    operation.operation_kind
                    is not ExpertValidationOperationKind.EVALUATOR_RESULT
                    or operation.request_record_id
                    != result_record.evaluator_result_record_id
                    or latest_attempt is None
                    or result_record.evaluator_run.validation_attempt_id
                    != latest_attempt.validation_attempt_id
                    or result_record.evaluator_run.candidate_id
                    != transition.candidate_id
                    or result_record.evaluator_run.candidate_tree_hash
                    != transition.candidate_tree_hash
                    or state.transition_evidence_id
                    != result_record.attestation_envelope.attestation.evaluator_attestation_id
                    or operation.expected_transition_id
                    != transition.predecessor_transition_id
                ):
                    raise ExpertValidationStoreError(
                        "validation result transition closure is inconsistent"
                    )
                expected_accepted = previous_accepted
                if result_record.evaluator_run.outcome is ExpertEvaluatorOutcome.PASSED:
                    expected_accepted = (
                        *previous_accepted,
                        result_record.evaluator_result_record_id,
                    )
            else:
                self._validate_source_stage_transition_unlocked(
                    journal=journal,
                    transition=transition,
                    state=state,
                    operation=operation,
                    latest_attempt=latest_attempt,
                    previous_accepted=previous_accepted,
                    result_record=result_record,
                )
                expected_accepted = previous_accepted
                if result_record.outcome is ExpertEvaluatorOutcome.PASSED:
                    expected_accepted = (
                        *previous_accepted,
                        result_record.stage_result_record_id,
                    )
            if transition.accepted_stage_result_record_ids != expected_accepted:
                raise ExpertValidationStoreError(
                    "validation accepted result prefix is not gap-free"
                )
        else:
            invalidation = self._read_contract_unlocked(
                transition.transition_authority_invalidation_id,
                ExpertValidationAuthorityInvalidation,
            )
            if transition.predecessor_state_id is None:
                raise ExpertValidationStoreError(
                    "authority invalidation requires a predecessor state"
                )
            predecessor_state = self._read_contract_unlocked(
                transition.predecessor_state_id,
                ExpertCandidateValidationState,
            )
            previous_accepted = (
                ()
                if previous_transition is None
                else previous_transition.accepted_stage_result_record_ids
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.AUTHORITY_INVALIDATION
                or operation.request_record_id != invalidation.authority_invalidation_id
                or operation.expected_transition_id
                != transition.predecessor_transition_id
                or latest_attempt is None
                or predecessor_state.promotion_state
                is not ExpertPromotionState.VALIDATING
                or predecessor_state.validation_attempt_id
                != latest_attempt.validation_attempt_id
                or invalidation.validation_attempt_id
                != latest_attempt.validation_attempt_id
                or invalidation.authorization_state_id
                != predecessor_state.validation_state_id
                or invalidation.candidate_id != latest_attempt.candidate_id
                or invalidation.candidate_tree_hash
                != latest_attempt.candidate_tree_hash
                or invalidation.scope_contract_id != latest_attempt.scope_contract_id
                or invalidation.expected_parent_release_id
                != latest_attempt.parent_release_id
                or transition.validation_policy_id
                != latest_attempt.validation_policy_id
                or transition.configuration_fingerprint
                != latest_attempt.configuration_fingerprint
                or state.promotion_state is not ExpertPromotionState.FAILED
                or state.accepted_stage_results
                != predecessor_state.accepted_stage_results
                or state.review_assertion_ids != predecessor_state.review_assertion_ids
                or state.terminal_evidence_ids
                != (invalidation.authority_invalidation_id,)
                or state.transition_evidence_id
                != invalidation.authority_invalidation_id
                or state.reason != "validation_parent_release_changed"
                or transition.accepted_stage_result_record_ids != previous_accepted
            ):
                raise ExpertValidationStoreError(
                    "validation authority invalidation closure is inconsistent"
                )

    def _resolved_operation_unlocked(
        self,
        journal: ExpertValidationJournal,
        operation: ExpertValidationOperation,
    ) -> ExpertValidationSnapshot | None:
        transition_id = journal.operation_transition_ids.get(operation.operation_id)
        if transition_id is None:
            return None
        stored_operation = self._read_contract_unlocked(
            operation.operation_id,
            ExpertValidationOperation,
        )
        if stored_operation != operation:
            raise ExpertValidationStoreError(
                "validation operation identity conflicts with persisted input"
            )
        return self._snapshot_at_unlocked(journal, transition_id)

    @staticmethod
    def _require_expected_head(
        current: ExpertValidationSnapshot | None,
        expected_transition_id: str | None,
    ) -> None:
        current_id = None if current is None else current.transition.transition_id
        if current_id != expected_transition_id:
            raise ExpertValidationCompareAndSwapError(
                "validation candidate head changed before publication"
            )

    @staticmethod
    def _append_transition(
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
    ) -> ExpertValidationJournal:
        operations = dict(journal.operation_transition_ids)
        operations[transition.operation_id] = transition.transition_id
        return ExpertValidationJournal(
            candidate_id=journal.candidate_id,
            candidate_tree_hash=(
                transition.candidate_tree_hash
                if not journal.transition_ids
                else journal.candidate_tree_hash
            ),
            transition_ids=(*journal.transition_ids, transition.transition_id),
            operation_transition_ids=operations,
        )

    @staticmethod
    def _bind_operation(
        journal: ExpertValidationJournal,
        operation: ExpertValidationOperation,
        transition: ExpertValidationTransition,
    ) -> ExpertValidationJournal:
        operations = dict(journal.operation_transition_ids)
        operations[operation.operation_id] = transition.transition_id
        return ExpertValidationJournal(
            candidate_id=journal.candidate_id,
            candidate_tree_hash=journal.candidate_tree_hash,
            transition_ids=journal.transition_ids,
            operation_transition_ids=operations,
        )

    def _read_journal_unlocked(
        self,
        candidate_id: str,
    ) -> ExpertValidationJournal:
        path = self._journal_path(candidate_id, create_namespace=False)
        if not os.path.lexists(path):
            return ExpertValidationJournal(
                candidate_id=candidate_id,
                candidate_tree_hash="sha256:" + "0" * 64,
                transition_ids=(),
                operation_transition_ids={},
            )
        payload = self._read_private_file(path, "validation journal")
        journal = ExpertValidationJournal.from_json_bytes(payload)
        if payload != journal.to_json_bytes() or journal.candidate_id != candidate_id:
            raise ExpertValidationStoreError(
                "validation journal bytes or identity are invalid"
            )
        self._validate_journal_unlocked(journal)
        return journal

    def _write_journal_unlocked(self, journal: ExpertValidationJournal) -> None:
        self._atomic_replace(
            self._journal_path(journal.candidate_id, create_namespace=True),
            journal.to_json_bytes(),
        )

    def _publish_journal_unlocked(self, journal: ExpertValidationJournal) -> None:
        self._validate_journal_unlocked(journal)
        self._write_journal_unlocked(journal)

    def _write_configuration_unlocked(self) -> None:
        payload = self.settings.to_json_bytes()
        fingerprint = self.settings.configuration_fingerprint
        if tree_or_blob_digest(payload) != fingerprint:
            raise ExpertValidationStoreError(
                "validation settings fingerprint differs from canonical bytes"
            )
        self._write_once(
            self._configuration_path(fingerprint),
            payload,
        )

    def _read_configuration_unlocked(
        self,
        fingerprint: str,
    ) -> ExpertValidationSettings:
        payload = self._read_private_file(
            self._configuration_path(fingerprint),
            "validation configuration",
        )
        settings = ExpertValidationSettings.from_json_bytes(payload)
        if (
            payload != settings.to_json_bytes()
            or settings.configuration_fingerprint != fingerprint
        ):
            raise ExpertValidationStoreError(
                "persisted validation configuration is invalid"
            )
        return settings

    def _write_contract_unlocked(self, contract: StrictContract) -> None:
        identity_field = contract.IDENTITY_FIELD
        if identity_field is None:
            raise ExpertValidationStoreError(
                "validation object must be content identified"
            )
        identity = getattr(contract, identity_field)
        self._write_once(
            self._object_path(identity, create_namespace=True),
            contract.to_json_bytes(),
        )

    def _read_contract_unlocked(self, identity: str, contract_type):
        payload = self._read_private_file(
            self._object_path(identity, create_namespace=False),
            "validation object",
        )
        contract = contract_type.from_json_bytes(payload)
        identity_field = contract.IDENTITY_FIELD
        if (
            identity_field is None
            or getattr(contract, identity_field) != identity
            or payload != contract.to_json_bytes()
        ):
            raise ExpertValidationStoreError(
                "validation object bytes or identity are invalid"
            )
        return contract

    def _object_path(self, identity: str, *, create_namespace: bool) -> Path:
        require_content_id(identity, "validation object ID")
        namespace, digest = identity.split(":sha256:", 1)
        namespace_root = self.object_root / namespace
        if not os.path.lexists(namespace_root) and create_namespace:
            os.mkdir(namespace_root, mode=0o700)
            self._fsync_directory(self.object_root)
        if not os.path.lexists(namespace_root):
            raise ExpertValidationStoreError("validation object namespace is missing")
        self._validate_private_directory(namespace_root, "object namespace")
        return namespace_root / f"{digest}.json"

    def _configuration_path(self, fingerprint: str) -> Path:
        if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
            raise ExpertValidationStoreError(
                "validation configuration fingerprint is invalid"
            )
        return self.configuration_root / f"{fingerprint[7:]}.json"

    def _journal_path(
        self,
        candidate_id: str,
        *,
        create_namespace: bool,
    ) -> Path:
        require_content_id(candidate_id, "candidate_id")
        namespace, digest = candidate_id.split(":sha256:", 1)
        namespace_root = self.journal_root / namespace
        if not os.path.lexists(namespace_root) and create_namespace:
            os.mkdir(namespace_root, mode=0o700)
            self._fsync_directory(self.journal_root)
        if os.path.lexists(namespace_root):
            self._validate_private_directory(namespace_root, "journal namespace")
        return namespace_root / f"{digest}.json"

    def _write_once(self, path: Path, payload: bytes) -> None:
        if os.path.lexists(path):
            existing = self._read_private_file(path, "validation object")
            if existing != payload:
                raise ExpertValidationStoreError(
                    "validation object identity conflicts with persisted bytes"
                )
            return
        self._atomic_replace(path, payload)

    def _atomic_replace(self, path: Path, payload: bytes) -> None:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.staging_root,
            prefix=".validation-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
        self._fsync_directory(path.parent)

    def _prepare_layout(self) -> None:
        if not os.path.lexists(self.root):
            os.mkdir(self.root, mode=0o700)
            self._fsync_directory(self.state_root)
        self._validate_private_directory(self.root, "validation store")
        for path in (
            self.object_root,
            self.configuration_root,
            self.journal_root,
            self.staging_root,
        ):
            if not os.path.lexists(path):
                os.mkdir(path, mode=0o700)
            self._validate_private_directory(path, "validation store child")
        self._fsync_directory(self.root)

    def _lock(self, *, exclusive: bool) -> _ValidationStoreLock:
        return _ValidationStoreLock(
            self.root / "validation.lock",
            exclusive=exclusive,
            create=True,
        )

    @staticmethod
    def _validate_state_root(path: Path) -> None:
        if (
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise ExpertValidationStoreError(
                "validation state root must be an authorized real directory"
            )
        ExpertValidationStore._validate_private_directory(
            path,
            "validation state root",
        )

    @staticmethod
    def _validate_private_directory(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ExpertValidationStoreError(f"{name} must be a real directory")
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
            0o077 | stat.S_ISUID | stat.S_ISGID
        ):
            raise ExpertValidationStoreError(f"{name} must be private")

    @staticmethod
    def _read_private_file(path: Path, name: str) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise ExpertValidationStoreError(f"{name} must be a regular file")
        metadata = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            raise ExpertValidationStoreError(
                f"{name} must be a private independent file"
            )
        return path.read_bytes()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(descriptor)
        os.close(descriptor)


class _ValidationStoreLock:
    def __init__(self, path: Path, *, exclusive: bool, create: bool) -> None:
        self.path = path
        self.exclusive = exclusive
        self.create = create
        self.handle = None

    def __enter__(self):
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
        if self.create:
            flags |= os.O_CREAT
        descriptor = os.open(self.path, flags, 0o600)
        self.handle = os.fdopen(descriptor, "r+b")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            self.handle.close()
            raise ExpertValidationStoreError(
                "validation lock must be a private independent file"
            )
        fcntl.flock(
            self.handle.fileno(),
            fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH,
        )
        return self

    def __exit__(self, exception_type, exception, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        return False
