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
    ExpertEvaluatorAttestationEnvelope,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorRun,
    ExpertPromotionState,
    ExpertValidationAttempt,
    StrictContract,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.validation import (
    ExpertEligibilityResult,
    ExpertEvaluatorResult,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
)
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


@dataclass(frozen=True)
class ExpertEvaluatorResultRecord(StrictContract):
    evaluator_result_record_id: str
    evaluator_run: ExpertEvaluatorRun
    attestation_envelope: ExpertEvaluatorAttestationEnvelope

    CONTENT_NAMESPACE: ClassVar[str] = "expert-evaluator-result-record"
    IDENTITY_FIELD: ClassVar[str] = "evaluator_result_record_id"

    def _validate(self) -> None:
        attestation = self.attestation_envelope.attestation
        if (
            attestation.evaluator_run_id != self.evaluator_run.evaluator_run_id
            or attestation.issuer_id != self.evaluator_run.evaluator_id
            or attestation.predicate_digest
            != tree_or_blob_digest(self.evaluator_run.to_json_bytes())
        ):
            raise ContractValidationError(
                "evaluator result record attestation does not bind its run"
            )


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
    accepted_result_record_ids: tuple[str, ...]
    transition_result_record_id: str | None

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
            (self.transition_result_record_id, "transition_result_record_id"),
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
        if len(self.accepted_result_record_ids) != len(
            set(self.accepted_result_record_ids)
        ):
            raise ContractValidationError(
                "accepted evaluator result records must be unique"
            )
        for result_record_id in self.accepted_result_record_ids:
            require_content_id(result_record_id, "accepted_result_record_ids")
        start = self.eligibility_decision_id is not None
        if start == (self.transition_result_record_id is not None):
            raise ContractValidationError(
                "transition must contain exactly one start or result request"
            )
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
    journal: ExpertValidationJournal
    transition: ExpertValidationTransition
    state: ExpertCandidateValidationState
    latest_attempt: ExpertValidationAttempt | None
    accepted_results: tuple[ExpertEvaluatorResult, ...]

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
        self.root = root
        self.state_root = state_root
        self.settings = settings
        self.reducer = reducer
        self.object_root = root / "objects"
        self.configuration_root = root / "configurations"
        self.journal_root = root / "journals"
        self.staging_root = root / "staging"
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

    def publish_start(
        self,
        *,
        expected_transition_id: str | None,
        stored_candidate: StoredExpertCandidate,
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
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(decision.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            predecessor = None if current is None else current.predecessor
            start = self.reducer.start_from_predecessor(
                stored_candidate=stored_candidate,
                eligibility=eligibility,
                predecessor=predecessor,
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
                self._write_journal_unlocked(updated)
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
            self._write_journal_unlocked(updated)
            return ExpertValidationCommitResult(
                snapshot=self._snapshot_at_unlocked(updated, transition.transition_id),
                replayed=False,
            )

    def publish_result(
        self,
        *,
        candidate_id: str,
        expected_transition_id: str,
        result: ExpertEvaluatorResult,
    ) -> ExpertValidationCommitResult:
        require_content_id(candidate_id, "candidate_id")
        require_content_id(expected_transition_id, "expected_transition_id")
        result_record = ExpertEvaluatorResultRecord.mint(
            evaluator_run=result.evaluator_run,
            attestation_envelope=result.attestation_envelope,
        )
        operation = ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.EVALUATOR_RESULT,
            candidate_id=candidate_id,
            expected_transition_id=expected_transition_id,
            request_record_id=result_record.evaluator_result_record_id,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertValidationCommitResult(snapshot=replay, replayed=True)
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current is None or current.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "evaluator result requires a current validation attempt"
                )
            target_state = self.reducer.advance(
                state=current.state,
                attempt=current.latest_attempt,
                accepted_results=current.accepted_results,
                result=result,
            )
            accepted_ids = current.transition.accepted_result_record_ids
            if result.evaluator_run.outcome is ExpertEvaluatorOutcome.PASSED:
                accepted_ids = (
                    *accepted_ids,
                    result_record.evaluator_result_record_id,
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
                accepted_result_record_ids=accepted_ids,
                transition_result_record_id=(result_record.evaluator_result_record_id),
            )
            self._write_contract_unlocked(result_record)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._write_journal_unlocked(updated)
            return ExpertValidationCommitResult(
                snapshot=self._snapshot_at_unlocked(updated, transition.transition_id),
                replayed=False,
            )

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
            accepted_result_record_ids=(),
            transition_result_record_id=None,
        )

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
            or transition.transition_result_record_id is not None
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
            self._read_contract_unlocked(
                result_record_id,
                ExpertEvaluatorResultRecord,
            )
            for result_record_id in transition.accepted_result_record_ids
        )
        return ExpertValidationSnapshot(
            journal=journal,
            transition=transition,
            state=state,
            latest_attempt=latest_attempt,
            accepted_results=tuple(
                ExpertEvaluatorResult(
                    evaluator_run=record.evaluator_run,
                    attestation_envelope=record.attestation_envelope,
                )
                for record in accepted_records
            ),
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
            if operation_id != transition.operation_id and (
                operation.operation_kind is not ExpertValidationOperationKind.START
                or transition.eligibility_decision_id is None
                or operation.request_record_id != transition.eligibility_decision_id
                or operation.expected_transition_id != transition.transition_id
                or self._read_contract_unlocked(
                    transition.target_state_id,
                    ExpertCandidateValidationState,
                ).promotion_state
                is not ExpertPromotionState.INELIGIBLE
            ):
                raise ExpertValidationStoreError(
                    "validation replay operation does not bind its transition"
                )

    def _validate_transition_closure_unlocked(
        self,
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
            self._read_contract_unlocked(
                result_record_id,
                ExpertEvaluatorResultRecord,
            )
            for result_record_id in transition.accepted_result_record_ids
        )
        evidence = tuple(
            (
                record.evaluator_run.evaluator_run_id,
                record.attestation_envelope.attestation.evaluator_attestation_id,
            )
            for record in accepted_records
        )
        state_evidence = tuple(
            (item.evaluator_run_id, item.evaluator_attestation_id)
            for item in state.accepted_evaluator_evidence
        )
        if (
            evidence != state_evidence
            or any(
                record.evaluator_run.outcome is not ExpertEvaluatorOutcome.PASSED
                for record in accepted_records
            )
            or any(
                latest_attempt is None
                or record.evaluator_run.validation_attempt_id
                != latest_attempt.validation_attempt_id
                or record.evaluator_run.candidate_id != transition.candidate_id
                or record.evaluator_run.candidate_tree_hash
                != transition.candidate_tree_hash
                for record in accepted_records
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
                or transition.accepted_result_record_ids
            ):
                raise ExpertValidationStoreError(
                    "validation start transition closure is inconsistent"
                )
            if transition.created_attempt_id is not None and (
                latest_attempt is None
                or latest_attempt.eligibility_decision_id
                != decision.eligibility_decision_id
            ):
                raise ExpertValidationStoreError(
                    "validation start attempt differs from its eligibility decision"
                )
        else:
            result_record = self._read_contract_unlocked(
                transition.transition_result_record_id,
                ExpertEvaluatorResultRecord,
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.EVALUATOR_RESULT
                or operation.request_record_id
                != result_record.evaluator_result_record_id
                or latest_attempt is None
                or result_record.evaluator_run.validation_attempt_id
                != latest_attempt.validation_attempt_id
                or result_record.evaluator_run.candidate_id != transition.candidate_id
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
            previous_accepted = (
                ()
                if previous_transition is None
                else previous_transition.accepted_result_record_ids
            )
            expected_accepted = previous_accepted
            if result_record.evaluator_run.outcome is ExpertEvaluatorOutcome.PASSED:
                expected_accepted = (
                    *previous_accepted,
                    result_record.evaluator_result_record_id,
                )
            if transition.accepted_result_record_ids != expected_accepted:
                raise ExpertValidationStoreError(
                    "validation accepted result prefix is not gap-free"
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
