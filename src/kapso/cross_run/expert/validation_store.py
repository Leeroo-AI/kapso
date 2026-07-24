"""Crash-atomic expert validation history and operation replay."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ContractValidationError,
    ExpertCandidateDerivationKind,
    ExpertCandidateOperationRecord,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertBaseReleaseManifest,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    ExpertValidationAuthorityInvalidation,
    ExpertValidationAuthorityInvalidationKind,
    ExpertValidationAttempt,
    ExpertValidationStage,
    PublicationArtifactKind,
    StrictContract,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertAgentProposalDerivationRecord,
    ExpertDeterministicCompositionDerivationRecord,
    ExpertDeterministicRecoveryRestoreDerivationRecord,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionMaterialization,
)
from kapso.cross_run.expert.triggers import ExpertTriggerEvidencePacket
from kapso.cross_run.expert.validation import (
    ExpertEligibilityResult,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
    validate_source_replay_request_authority_shape,
)
from kapso.cross_run.expert.validation_operation_contracts import (
    ExpertValidationOperation,
    ExpertValidationOperationKind,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertPublicationEligibilitySnapshot,
    ExpertReleaseMatrixPlanReservationSnapshot,
    ExpertReleaseMatrixSourceEvidenceSnapshot,
    ExpertValidationSnapshot,
    ExpertValidationTransition,
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
from kapso.cross_run.expert.recovery_candidate_contracts import (
    ExpertRecoveryCandidateAdmission,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixEvaluationPlan,
    ExpertReleaseMatrixReport,
)
from kapso.cross_run.expert.promotion_plan import (
    PreparedExpertReleaseMatrixPlan,
    prepare_expert_release_matrix_plan_for_admission,
    validate_expert_release_matrix_plan_store_shape,
)
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
    ExpertReleaseMatrixStageExecution,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityCoordinator,
    ExpertPublicationEligibilityExecution,
    build_publication_eligibility_stage_result,
    publication_eligibility_security_subject_ids,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertCandidateReleaseUseDecision,
    ExpertCandidateReleaseUseOutcome,
    ExpertPublicationEligibilityAuthorityFence,
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.promotion import (
    decide_expert_release_matrix_promotion,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixPromotionDecision,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
    ExpertAutomatedReviewExecution,
    PreparedExpertAutomatedReviewPacket,
    adjudicate_expert_automated_review,
    build_expert_automated_review_stage_result,
    validate_expert_automated_review_facts,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewAssertion,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewPacket,
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleaseActivationReceipt,
    ExpertReleasePublicationIntent,
    ExpertReleasePublicationPlan,
    ExpertReleasePublicationStaleResolution,
)
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
    ExpertReleasePackage,
)
from kapso.cross_run.expert.publisher import ExpertReleasePublisher
from kapso.cross_run.expert.revocation import ExpertReleaseRevocationCoordinator
from kapso.cross_run.expert.revocation_contracts import (
    ExpertReleaseRevocationReceipt,
    expert_release_revocation_security_subject_ids,
)
from kapso.cross_run.git_refs import git_object_sha
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationRequest,
    TaskEvaluationReservation,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluation_request import (
    PlanJoinedTaskEvaluationRequest,
)
from kapso.cross_run.settings import (
    ExpertValidationPolicy,
    ExpertValidationSettings,
)


class ExpertValidationStoreError(ValueError):
    """Persisted validation history is incomplete, corrupt, or conflicting."""


class ExpertValidationCompareAndSwapError(ExpertValidationStoreError):
    """A validation operation was reduced from a stale candidate head."""


@dataclass(frozen=True)
class ExpertValidationJournal(StrictContract):
    candidate_id: str
    candidate_tree_hash: str
    transition_ids: tuple[str, ...]
    operation_transition_ids: Mapping[str, str]
    release_publication_intent_id: str | None
    release_publication_stale_resolution_id: str | None

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
        if self.release_publication_intent_id is not None:
            require_content_id(
                self.release_publication_intent_id,
                "journal release publication intent",
            )
            if self.release_publication_intent_id.split(":sha256:", 1)[0] != (
                "expert-release-publication-intent"
            ):
                raise ContractValidationError(
                    "journal release publication intent uses the wrong namespace"
                )
        if self.release_publication_stale_resolution_id is not None:
            require_content_id(
                self.release_publication_stale_resolution_id,
                "journal stale release publication resolution",
            )
            if (
                self.release_publication_stale_resolution_id.split(":sha256:", 1)[0]
                != "expert-release-publication-stale-resolution"
            ):
                raise ContractValidationError(
                    "journal stale release publication resolution uses the wrong namespace"
                )
        if (
            self.release_publication_intent_id is not None
            and self.release_publication_stale_resolution_id is not None
        ):
            raise ContractValidationError(
                "journal cannot contain pending and stale release publication authority"
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
            or self.reservation.expected_current_release_id
            != self.request.expected_current_release_id
            or self.request.validation_attempt_id != attempt.validation_attempt_id
            or self.request.candidate_id != attempt.candidate_id
            or self.request.candidate_tree_hash != attempt.candidate_tree_hash
            or self.request.candidate_commit_record_id
            != attempt.candidate_commit_record_id
            or self.request.scope_contract_id != attempt.scope_contract_id
            or self.request.source_base_release_id != attempt.source_base_release_id
            or self.request.expected_current_release_id
            != attempt.expected_current_release_id
            or self.request.recovery_plan_id != attempt.recovery_plan_id
            or self.request.control_dependency_ids != attempt.control_dependency_ids
            or self.request.attempt_dependency_ids != attempt.eligibility_dependency_ids
        ):
            raise ExpertValidationStoreError(
                "source replay reservation snapshot authority is inconsistent"
            )


@dataclass(frozen=True)
class ExpertSourceReplayStageCommitResult:
    stage_result: ExpertSourceReplayStageResultRecord
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertAutomatedReviewStageCommitResult:
    stage_result: ExpertAutomatedReviewStageResultRecord
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertReleaseMatrixStageCommitResult:
    stage_result: ExpertReleaseMatrixStageResultRecord
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertPublicationEligibilityStageCommitResult:
    stage_result: ExpertPublicationEligibilityStageResultRecord
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertReleaseUseBlockCommitResult:
    decision: ExpertCandidateReleaseUseDecision
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertReleasePublicationReservation:
    intent: ExpertReleasePublicationIntent
    plan: ExpertReleasePublicationPlan
    manifest: ExpertBaseReleaseManifest
    snapshot: ExpertValidationSnapshot

    def __post_init__(self) -> None:
        if (
            type(self.intent) is not ExpertReleasePublicationIntent
            or type(self.plan) is not ExpertReleasePublicationPlan
            or type(self.manifest) is not ExpertBaseReleaseManifest
            or type(self.snapshot) is not ExpertValidationSnapshot
            or self.intent.publication_plan_id != self.plan.publication_plan_id
            or self.plan.release_id != self.manifest.release_id
            or self.plan.candidate_id != self.snapshot.state.candidate_id
            or self.plan.approval_transition_id
            != self.snapshot.transition.transition_id
            or self.plan.approval_state_id != self.snapshot.state.validation_state_id
        ):
            raise ExpertValidationStoreError(
                "release publication reservation authority is inconsistent"
            )


@dataclass(frozen=True)
class ExpertReleasePublicationReservationCommitResult:
    reservation: ExpertReleasePublicationReservation
    replayed: bool


@dataclass(frozen=True)
class ExpertReleaseActivationCommitResult:
    receipt: ExpertReleaseActivationReceipt
    snapshot: ExpertValidationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertReleaseRevocationTarget:
    activation: ExpertReleaseActivationCommitResult
    manifest: ExpertBaseReleaseManifest
    security_subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.activation) is not ExpertReleaseActivationCommitResult
            or type(self.manifest) is not ExpertBaseReleaseManifest
        ):
            raise ExpertValidationStoreError(
                "release revocation target authority is not typed"
            )
        snapshot = self.activation.snapshot
        attempt = snapshot.latest_attempt
        if (
            attempt is None
            or self.security_subject_ids
            != expert_release_revocation_security_subject_ids(
                authorization_transition_id=snapshot.transition.transition_id,
                released_state=snapshot.state,
                validation_attempt=attempt,
                activation_receipt=self.activation.receipt,
                release_manifest=self.manifest,
            )
        ):
            raise ExpertValidationStoreError(
                "release revocation target authority is inconsistent"
            )


@dataclass(frozen=True)
class ExpertReleaseRevocationCommitResult:
    receipt: ExpertReleaseRevocationReceipt
    snapshot: ExpertValidationSnapshot
    replayed: bool


_RELEASE_PUBLICATION_STALE_PERMIT_SEAL = object()
_RELEASE_USE_BLOCK_PERMIT_SEAL = object()


class ExpertReleaseUseBlockPermit:
    """One-shot publisher authority for an approved-to-blocked policy transition."""

    __slots__ = (
        "_approval_snapshot",
        "_consumed",
        "_decision",
        "_owner_process_id",
        "_publisher",
        "_store",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertValidationStore,
        publisher: object,
        approval_snapshot: ExpertValidationSnapshot,
        decision: ExpertCandidateReleaseUseDecision,
    ) -> None:
        if seal is not _RELEASE_USE_BLOCK_PERMIT_SEAL:
            raise ExpertValidationStoreError(
                "release-use block permit is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_publisher", publisher)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_approval_snapshot", approval_snapshot)
        object.__setattr__(self, "_decision", decision)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertValidationStoreError("release-use block permit is immutable")

    def _require_bound(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        if (
            self._consumed
            or self._store is not store
            or self._publisher is not publisher
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertValidationStoreError(
                "release-use block permit is consumed or foreign"
            )

    def _consume(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        self._require_bound(store, publisher)
        object.__setattr__(self, "_consumed", True)


class ExpertReleasePublicationStalePermit:
    """One-shot authority over a publisher-classified losing reservation."""

    __slots__ = (
        "_consumed",
        "_observed_current",
        "_observed_current_activation_witness",
        "_own_github_publication_intent",
        "_own_github_publication_pointer",
        "_own_github_activation_preparation_commit_sha",
        "_owner_process_id",
        "_publisher",
        "_reservation",
        "_resolved_at",
        "_store",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertValidationStore,
        publisher: object,
        reservation: ExpertReleasePublicationReservation,
        observed_current: TaskEvaluationCurrentReleaseObservation,
        observed_current_activation_witness: GitHubArtifactActivationWitness,
        own_github_publication_intent: ArtifactPublicationIntent | None,
        own_github_publication_pointer: CurrentArtifactPointer | None,
        own_github_activation_preparation_commit_sha: str | None,
        resolved_at: str,
    ) -> None:
        if seal is not _RELEASE_PUBLICATION_STALE_PERMIT_SEAL:
            raise ExpertValidationStoreError(
                "release publication stale permit is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_publisher", publisher)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_reservation", reservation)
        object.__setattr__(self, "_observed_current", observed_current)
        object.__setattr__(
            self,
            "_observed_current_activation_witness",
            observed_current_activation_witness,
        )
        object.__setattr__(
            self,
            "_own_github_publication_intent",
            own_github_publication_intent,
        )
        object.__setattr__(
            self,
            "_own_github_publication_pointer",
            own_github_publication_pointer,
        )
        object.__setattr__(
            self,
            "_own_github_activation_preparation_commit_sha",
            own_github_activation_preparation_commit_sha,
        )
        object.__setattr__(self, "_resolved_at", resolved_at)

    def __setattr__(self, name, value) -> None:
        raise ExpertValidationStoreError(
            "release publication stale permit is immutable"
        )

    def _consume(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        self._require_bound(store, publisher)
        object.__setattr__(self, "_consumed", True)

    def _require_bound(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        if (
            self._consumed
            or self._store is not store
            or self._publisher is not publisher
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertValidationStoreError(
                "release publication stale permit is consumed or foreign"
            )


_RELEASE_ACTIVATION_PERMIT_SEAL = object()


class ExpertReleaseActivationPermit:
    """One-shot authority over an exactly observed GitHub activation."""

    __slots__ = (
        "_activation_witness",
        "_consumed",
        "_github_publication_intent",
        "_github_publication_pointer",
        "_observed_current",
        "_owner_process_id",
        "_publisher",
        "_reservation",
        "_store",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertValidationStore,
        publisher: object,
        reservation: ExpertReleasePublicationReservation,
        github_publication_intent: ArtifactPublicationIntent,
        github_publication_pointer: CurrentArtifactPointer,
        activation_witness: GitHubArtifactActivationWitness,
        observed_current: TaskEvaluationCurrentReleaseObservation,
    ) -> None:
        if seal is not _RELEASE_ACTIVATION_PERMIT_SEAL:
            raise ExpertValidationStoreError(
                "release activation permit is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_publisher", publisher)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_reservation", reservation)
        object.__setattr__(
            self, "_github_publication_intent", github_publication_intent
        )
        object.__setattr__(
            self, "_github_publication_pointer", github_publication_pointer
        )
        object.__setattr__(self, "_activation_witness", activation_witness)
        object.__setattr__(self, "_observed_current", observed_current)

    def __setattr__(self, name, value) -> None:
        raise ExpertValidationStoreError("release activation permit is immutable")

    def _require_bound(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        if (
            self._consumed
            or self._store is not store
            or self._publisher is not publisher
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertValidationStoreError(
                "release activation permit is consumed or foreign"
            )

    def _consume(
        self,
        store: ExpertValidationStore,
        publisher: object,
    ) -> None:
        self._require_bound(store, publisher)
        object.__setattr__(self, "_consumed", True)


_RELEASE_REVOCATION_PERMIT_SEAL = object()


class ExpertReleaseRevocationPermit:
    """One-shot authority over one freshly matched activated release closure."""

    __slots__ = (
        "_consumed",
        "_coordinator",
        "_observation",
        "_owner_process_id",
        "_revoked_at",
        "_store",
        "_target",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertValidationStore,
        coordinator: ExpertReleaseRevocationCoordinator,
        target: ExpertReleaseRevocationTarget,
        observation: SecurityDenylistObservation,
        revoked_at: str,
    ) -> None:
        if seal is not _RELEASE_REVOCATION_PERMIT_SEAL:
            raise ExpertValidationStoreError(
                "release revocation permit is not store sealed"
            )
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_observation", observation)
        object.__setattr__(self, "_revoked_at", revoked_at)

    def __setattr__(self, name, value) -> None:
        raise ExpertValidationStoreError("release revocation permit is immutable")

    def _require_bound(
        self,
        store: ExpertValidationStore,
        coordinator: ExpertReleaseRevocationCoordinator,
    ) -> None:
        if (
            self._consumed
            or self._store is not store
            or self._coordinator is not coordinator
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertValidationStoreError(
                "release revocation permit is consumed or foreign"
            )

    def _consume(
        self,
        store: ExpertValidationStore,
        coordinator: ExpertReleaseRevocationCoordinator,
    ) -> None:
        self._require_bound(store, coordinator)
        object.__setattr__(self, "_consumed", True)


@dataclass(frozen=True)
class ExpertReleaseMatrixPlanReservationCommitResult:
    reservation: ExpertReleaseMatrixPlanReservationSnapshot
    replayed: bool


@dataclass(frozen=True)
class ExpertTaskEvaluationReservationCommitResult:
    reservation: ExpertTaskEvaluationReservationSnapshot
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
        self._automated_review_coordinator = None
        self._release_matrix_stage_coordinator = None
        self._publication_eligibility_coordinator = None
        self._release_assembler = None
        self._release_publisher = None
        self._release_revocation_coordinator = None
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

    def _reserve_release_publication(
        self,
        publisher: object,
        *,
        plan: ExpertReleasePublicationPlan,
        package: ExpertReleasePackage,
        committed_at: str,
    ) -> ExpertReleasePublicationReservationCommitResult:
        """Freeze one terminal approval to a first-writer-wins publication intent."""

        self._require_bound_release_publisher_authority(publisher)
        if (
            type(plan) is not ExpertReleasePublicationPlan
            or type(package) is not ExpertReleasePackage
        ):
            raise ExpertValidationStoreError(
                "release publication reservation requires an exact plan and package"
            )
        assembler = self._require_bound_release_assembly_authority(publisher.assembler)
        expected_plan = assembler._derive_publication_plan(
            package=package,
            current_release_observation=plan.current_release_observation,
            activation_predecessor_pointer=plan.activation_predecessor_pointer,
        )
        if expected_plan != plan:
            raise ExpertValidationStoreError(
                "release publication plan differs from its deterministic package"
            )
        manifest = package.manifest
        proposed_intent = ExpertReleasePublicationIntent.mint(
            publication_plan_id=plan.publication_plan_id,
            committed_at=committed_at,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(plan.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None:
                raise ExpertValidationStoreError(
                    "release publication reservation has no validation head"
                )
            if journal.release_publication_stale_resolution_id is not None:
                raise ExpertValidationCompareAndSwapError(
                    "candidate already has a stale release publication resolution"
                )
            if journal.release_publication_intent_id is not None:
                stored_intent = self._read_contract_unlocked(
                    journal.release_publication_intent_id,
                    ExpertReleasePublicationIntent,
                )
                stored_plan = self._read_contract_unlocked(
                    stored_intent.publication_plan_id,
                    ExpertReleasePublicationPlan,
                )
                stored_manifest = self._read_contract_unlocked(
                    stored_plan.release_id,
                    ExpertBaseReleaseManifest,
                )
                if stored_plan != plan or stored_manifest != manifest:
                    raise ExpertValidationCompareAndSwapError(
                        "validation approval is reserved for another release plan"
                    )
                self._validate_release_publication_reservation_unlocked(
                    stored_intent,
                    stored_plan,
                    stored_manifest,
                    current,
                )
                return ExpertReleasePublicationReservationCommitResult(
                    reservation=ExpertReleasePublicationReservation(
                        intent=stored_intent,
                        plan=stored_plan,
                        manifest=stored_manifest,
                        snapshot=current,
                    ),
                    replayed=True,
                )
            self._validate_release_publication_reservation_unlocked(
                proposed_intent,
                plan,
                manifest,
                current,
            )
            self._write_contract_unlocked(manifest)
            self._write_contract_unlocked(plan)
            self._write_contract_unlocked(proposed_intent)
            updated = ExpertValidationJournal(
                candidate_id=journal.candidate_id,
                candidate_tree_hash=journal.candidate_tree_hash,
                transition_ids=journal.transition_ids,
                operation_transition_ids=journal.operation_transition_ids,
                release_publication_intent_id=(proposed_intent.publication_intent_id),
                release_publication_stale_resolution_id=None,
            )
            self._publish_journal_unlocked(updated)
            return ExpertReleasePublicationReservationCommitResult(
                reservation=ExpertReleasePublicationReservation(
                    intent=proposed_intent,
                    plan=plan,
                    manifest=manifest,
                    snapshot=current,
                ),
                replayed=False,
            )

    def reopen_release_publication(
        self,
        candidate_id: str,
    ) -> ExpertReleasePublicationReservation | None:
        """Reopen the exact frozen approval and publication intent."""

        require_content_id(candidate_id, "release publication candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            intent_id = journal.release_publication_intent_id
            if intent_id is None:
                return None
            intent = self._read_contract_unlocked(
                intent_id,
                ExpertReleasePublicationIntent,
            )
            plan = self._read_contract_unlocked(
                intent.publication_plan_id,
                ExpertReleasePublicationPlan,
            )
            manifest = self._read_contract_unlocked(
                plan.release_id,
                ExpertBaseReleaseManifest,
            )
            approval = self._snapshot_at_unlocked(
                journal,
                plan.approval_transition_id,
            )
            if approval is None:
                raise ExpertValidationStoreError(
                    "release publication reservation lost its validation head"
                )
            self._validate_release_publication_reservation_unlocked(
                intent,
                plan,
                manifest,
                approval,
            )
            return ExpertReleasePublicationReservation(
                intent=intent,
                plan=plan,
                manifest=manifest,
                snapshot=approval,
            )

    def resolve_stale_release_publication(
        self,
        stale_permit: ExpertReleasePublicationStalePermit,
    ) -> ExpertReleasePublicationStaleResolution:
        """Terminalize one publisher-classified losing publication intent."""

        if type(stale_permit) is not ExpertReleasePublicationStalePermit:
            raise ExpertValidationStoreError(
                "stale publication resolution requires a publisher-sealed permit"
            )
        publisher = self._require_bound_release_publisher_authority(
            self._release_publisher
        )
        stale_permit._require_bound(self, publisher)
        reservation = stale_permit._reservation
        plan = reservation.plan
        observed = stale_permit._observed_current
        dependencies = {
            reservation.intent.publication_intent_id,
            plan.publication_plan_id,
            plan.release_id,
            plan.candidate_id,
            plan.approval_transition_id,
            plan.approval_state_id,
            plan.current_release_observation.observation_id,
            observed.observation_id,
            *observed.validation_closure_ids,
        }
        if observed.release_id is not None:
            dependencies.add(observed.release_id)
        if observed.publication_id is not None:
            dependencies.add(observed.publication_id)
        own_pointer = stale_permit._own_github_publication_pointer
        if own_pointer is not None:
            dependencies.add(own_pointer.publication_record.publication_id)
        winner_witness = stale_permit._observed_current_activation_witness
        dependencies.add(winner_witness.witness_id)
        resolution = ExpertReleasePublicationStaleResolution.mint(
            publication_intent_id=reservation.intent.publication_intent_id,
            publication_plan_id=plan.publication_plan_id,
            release_id=plan.release_id,
            candidate_id=plan.candidate_id,
            approval_transition_id=plan.approval_transition_id,
            approval_state_id=plan.approval_state_id,
            planned_current_observation_id=(
                plan.current_release_observation.observation_id
            ),
            observed_current_release=observed,
            observed_current_activation_witness=winner_witness,
            own_github_publication_intent=(stale_permit._own_github_publication_intent),
            own_github_publication_pointer=own_pointer,
            own_github_activation_preparation_commit_sha=(
                stale_permit._own_github_activation_preparation_commit_sha
            ),
            resolved_at=stale_permit._resolved_at,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(plan.candidate_id)
            if journal.release_publication_stale_resolution_id is not None:
                stored = self._read_contract_unlocked(
                    journal.release_publication_stale_resolution_id,
                    ExpertReleasePublicationStaleResolution,
                )
                if (
                    stored.publication_intent_id
                    != reservation.intent.publication_intent_id
                    or stored.publication_plan_id != plan.publication_plan_id
                ):
                    raise ExpertValidationCompareAndSwapError(
                        "stale publication resolution conflicts with durable outcome"
                    )
                stale_permit._consume(self, publisher)
                return stored
            if journal.release_publication_intent_id != (
                reservation.intent.publication_intent_id
            ):
                raise ExpertValidationCompareAndSwapError(
                    "release publication intent changed before stale resolution"
                )
            stale_permit._consume(self, publisher)
            self._write_contract_unlocked(resolution)
            updated = ExpertValidationJournal(
                candidate_id=journal.candidate_id,
                candidate_tree_hash=journal.candidate_tree_hash,
                transition_ids=journal.transition_ids,
                operation_transition_ids=journal.operation_transition_ids,
                release_publication_intent_id=None,
                release_publication_stale_resolution_id=(
                    resolution.stale_resolution_id
                ),
            )
            self._publish_journal_unlocked(updated)
            return resolution

    def reopen_stale_release_publication(
        self,
        candidate_id: str,
    ) -> ExpertReleasePublicationStaleResolution:
        """Reopen the first durable terminal outcome for a losing reservation."""

        require_content_id(candidate_id, "stale release publication candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            resolution_id = journal.release_publication_stale_resolution_id
            if resolution_id is None:
                raise ExpertValidationStoreError(
                    "candidate has no stale release publication resolution"
                )
            return self._read_contract_unlocked(
                resolution_id,
                ExpertReleasePublicationStaleResolution,
            )

    def commit_release_activation(
        self,
        activation_permit: ExpertReleaseActivationPermit,
    ) -> ExpertReleaseActivationCommitResult:
        """Atomically publish one APPROVED-to-RELEASED lifecycle transition."""

        if type(activation_permit) is not ExpertReleaseActivationPermit:
            raise ExpertValidationStoreError(
                "release activation requires a publisher-sealed permit"
            )
        publisher = self._require_bound_release_publisher_authority(
            self._release_publisher
        )
        activation_permit._require_bound(self, publisher)
        reservation = activation_permit._reservation
        plan = reservation.plan
        observed = activation_permit._observed_current
        github_intent = activation_permit._github_publication_intent
        github_pointer = activation_permit._github_publication_pointer
        activation_witness = activation_permit._activation_witness
        consumed_dependencies = {
            reservation.intent.publication_intent_id,
            plan.publication_plan_id,
            plan.release_id,
            plan.candidate_id,
            plan.approval_transition_id,
            plan.approval_state_id,
            github_pointer.publication_record.publication_id,
            activation_witness.witness_id,
            plan.release_id,
            *plan.manifest_consumed_dependency_ids,
        }
        control_dependencies = {
            *plan.manifest_control_dependency_ids,
            plan.current_release_observation.observation_id,
            observed.observation_id,
            *observed.validation_closure_ids,
        }
        if observed.release_id is not None:
            control_dependencies.add(observed.release_id)
        if observed.publication_id is not None:
            control_dependencies.add(observed.publication_id)
        control_dependencies.difference_update(consumed_dependencies)
        receipt = ExpertReleaseActivationReceipt.mint(
            publication_intent_id=reservation.intent.publication_intent_id,
            publication_plan_id=plan.publication_plan_id,
            release_id=plan.release_id,
            candidate_id=plan.candidate_id,
            approval_transition_id=plan.approval_transition_id,
            approval_state_id=plan.approval_state_id,
            planned_current_observation_id=(
                plan.current_release_observation.observation_id
            ),
            github_publication_intent=github_intent,
            github_publication_pointer=github_pointer,
            activation_witness=activation_witness,
            observed_current_release=observed,
            consumed_dependency_ids=tuple(sorted(consumed_dependencies)),
            control_dependency_ids=tuple(sorted(control_dependencies)),
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(plan.candidate_id)
            replay = self._release_activation_result_unlocked(journal)
            if replay is not None:
                activation_permit._consume(self, publisher)
                return replay
            if journal.release_publication_stale_resolution_id is not None:
                raise ExpertValidationCompareAndSwapError(
                    "release publication already has a stale terminal outcome"
                )
            if journal.release_publication_intent_id != (
                reservation.intent.publication_intent_id
            ):
                raise ExpertValidationCompareAndSwapError(
                    "release publication intent changed before activation commit"
                )
            current = self._current_from_journal_unlocked(journal)
            if current is None or current.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "release activation lost its approved validation head"
                )
            approval_snapshot = self._snapshot_at_unlocked(
                journal,
                plan.approval_transition_id,
            )
            self._validate_release_publication_reservation_unlocked(
                reservation.intent,
                plan,
                reservation.manifest,
                approval_snapshot,
            )
            target_state = self.reducer.advance_release_activation(
                state=current.state,
                approval_state=approval_snapshot.state,
                attempt=current.latest_attempt,
                plan=plan,
                receipt=receipt,
            )
            operation = ExpertValidationOperation.mint(
                operation_kind=ExpertValidationOperationKind.RELEASE_ACTIVATION,
                candidate_id=plan.candidate_id,
                expected_transition_id=current.transition.transition_id,
                request_record_id=receipt.activation_receipt_id,
            )
            transition = ExpertValidationTransition.mint(
                candidate_id=plan.candidate_id,
                candidate_tree_hash=plan.candidate_tree_hash,
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
                accepted_stage_result_record_ids=(
                    current.transition.accepted_stage_result_record_ids
                ),
                transition_stage_result_record_id=None,
                transition_authority_invalidation_id=None,
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=(
                    receipt.activation_receipt_id
                ),
                transition_release_revocation_receipt_id=None,
            )
            activation_permit._consume(self, publisher)
            self._write_contract_unlocked(receipt)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_release_activation_transition(
                journal,
                transition,
            )
            self._publish_journal_unlocked(updated)
            return ExpertReleaseActivationCommitResult(
                receipt=receipt,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def reopen_release_activation(
        self,
        candidate_id: str,
    ) -> ExpertReleaseActivationCommitResult | None:
        """Reopen the first durable RELEASED outcome without remote access."""

        require_content_id(candidate_id, "release activation candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            return self._release_activation_result_unlocked(journal)

    def reopen_release_revocation_target(
        self,
        candidate_id: str,
    ) -> ExpertReleaseRevocationTarget:
        """Reopen the exact current RELEASED proof closure to check remotely."""

        require_content_id(candidate_id, "release revocation candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            return self._release_revocation_target_unlocked(journal)

    def _seal_release_revocation(
        self,
        *,
        coordinator: object,
        target: ExpertReleaseRevocationTarget,
        security_denylist_observation: SecurityDenylistObservation,
        revoked_at: str,
    ) -> ExpertReleaseRevocationPermit:
        authority = self._require_bound_release_revocation_authority(coordinator)
        if (
            type(target) is not ExpertReleaseRevocationTarget
            or type(security_denylist_observation) is not SecurityDenylistObservation
            or not security_denylist_observation.matched_revocations
            or security_denylist_observation.scope_id != target.manifest.scope_id
            or security_denylist_observation.scope_contract_id
            != target.manifest.scope_contract_id
            or security_denylist_observation.scope_repository_binding_hash
            != target.activation.receipt.activation_witness.scope_repository_binding_hash
            or security_denylist_observation.checked_subject_ids
            != target.security_subject_ids
        ):
            raise ExpertValidationStoreError(
                "release revocation sealing requires one exact emergency match"
            )
        self._mint_release_revocation_receipt(
            target,
            security_denylist_observation,
            revoked_at,
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(target.manifest.candidate_id)
            current_target = self._release_revocation_target_unlocked(journal)
            if current_target != target:
                raise ExpertValidationCompareAndSwapError(
                    "release revocation target changed during authority checks"
                )
        return ExpertReleaseRevocationPermit(
            _RELEASE_REVOCATION_PERMIT_SEAL,
            self,
            authority,
            target,
            security_denylist_observation,
            revoked_at,
        )

    def commit_release_revocation(
        self,
        revocation_permit: ExpertReleaseRevocationPermit,
    ) -> ExpertReleaseRevocationCommitResult:
        """Atomically append one RELEASED-to-REVOKED lifecycle transition."""

        if type(revocation_permit) is not ExpertReleaseRevocationPermit:
            raise ExpertValidationStoreError(
                "release revocation requires an authority-sealed permit"
            )
        coordinator = self._require_bound_release_revocation_authority(
            self._release_revocation_coordinator
        )
        revocation_permit._require_bound(self, coordinator)
        target = revocation_permit._target
        receipt = self._mint_release_revocation_receipt(
            target,
            revocation_permit._observation,
            revocation_permit._revoked_at,
        )
        operation = ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.RELEASE_REVOCATION,
            candidate_id=target.manifest.candidate_id,
            expected_transition_id=target.activation.snapshot.transition.transition_id,
            request_record_id=receipt.revocation_receipt_id,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(target.manifest.candidate_id)
            replay = self._release_revocation_result_unlocked(journal)
            if replay is not None:
                if (
                    replay.receipt.release_id != target.manifest.release_id
                    or replay.receipt.candidate_id != target.manifest.candidate_id
                    or replay.receipt.activation_receipt_id
                    != target.activation.receipt.activation_receipt_id
                    or replay.receipt.authorization_transition_id
                    != target.activation.snapshot.transition.transition_id
                ):
                    raise ExpertValidationCompareAndSwapError(
                        "release revocation conflicts with the durable outcome"
                    )
                revocation_permit._consume(self, coordinator)
                return replay
            current_target = self._release_revocation_target_unlocked(journal)
            if current_target != target:
                raise ExpertValidationCompareAndSwapError(
                    "release revocation target changed before commit"
                )
            activation_snapshot = target.activation.snapshot
            attempt = activation_snapshot.latest_attempt
            if attempt is None:
                raise ExpertValidationStoreError(
                    "release revocation target lacks its validation attempt"
                )
            target_state = self.reducer.advance_release_revocation(
                authorization_transition_id=(
                    activation_snapshot.transition.transition_id
                ),
                state=activation_snapshot.state,
                attempt=attempt,
                activation_receipt=target.activation.receipt,
                release_manifest=target.manifest,
                revocation_receipt=receipt,
            )
            transition = ExpertValidationTransition.mint(
                candidate_id=target_state.candidate_id,
                candidate_tree_hash=target_state.candidate_tree_hash,
                transition_number=len(journal.transition_ids) + 1,
                predecessor_transition_id=(
                    activation_snapshot.transition.transition_id
                ),
                predecessor_state_id=activation_snapshot.state.validation_state_id,
                target_state_id=target_state.validation_state_id,
                latest_attempt_id=attempt.validation_attempt_id,
                operation_id=operation.operation_id,
                validation_policy_id=attempt.validation_policy_id,
                configuration_fingerprint=attempt.configuration_fingerprint,
                eligibility_decision_id=None,
                created_attempt_id=None,
                accepted_stage_result_record_ids=(
                    activation_snapshot.transition.accepted_stage_result_record_ids
                ),
                transition_stage_result_record_id=None,
                transition_authority_invalidation_id=None,
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=(
                    receipt.revocation_receipt_id
                ),
            )
            revocation_permit._consume(self, coordinator)
            self._write_contract_unlocked(receipt)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertReleaseRevocationCommitResult(
                receipt=receipt,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def reopen_release_revocation(
        self,
        candidate_id: str,
    ) -> ExpertReleaseRevocationCommitResult | None:
        """Reopen the first durable REVOKED outcome without remote access."""

        require_content_id(candidate_id, "release revocation candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            return self._release_revocation_result_unlocked(journal)

    def reopen_or_replay_automated_review(
        self,
        packet: ExpertAutomatedReviewPacket,
    ) -> ExpertAutomatedReviewStageCommitResult | None:
        if type(packet) is not ExpertAutomatedReviewPacket:
            raise ExpertValidationStoreError(
                "automated review replay requires its typed packet"
            )
        operation = self._automated_review_operation(packet)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(packet.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertAutomatedReviewStageCommitResult(
                    stage_result=self._automated_review_result_for_transition_unlocked(
                        replay.transition
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                current,
                packet.authorization_transition_id,
            )
            if (
                current is None
                or current.latest_attempt is None
                or current.state.validation_state_id != packet.authorization_state_id
                or current.latest_attempt.validation_attempt_id
                != packet.validation_attempt_id
                or current.state.next_stage
                is not ExpertValidationStage.AUTOMATED_REVIEW
            ):
                raise ExpertValidationStoreError(
                    "automated review packet lacks current stage authority"
                )
        return None

    def automated_review_stage_lock(
        self,
        candidate_id: str,
    ) -> _ValidationStoreLock:
        """Serialize paid review work for one candidate across processes."""
        require_content_id(candidate_id, "candidate_id")
        candidate_digest = candidate_id.rsplit(":", 1)[-1]
        return _ValidationStoreLock(
            self.state_root
            / f".{self.root.name}.{candidate_digest}.automated-review.lock",
            exclusive=True,
            create=True,
        )

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

    def _bind_automated_review_publication_authority(
        self,
        coordinator: ExpertAutomatedReviewCoordinator,
    ) -> None:
        if type(coordinator) is not ExpertAutomatedReviewCoordinator:
            raise ExpertValidationStoreError(
                "validation store review coordinator type is invalid"
            )
        coordinator._require_runner_authority()
        if (
            self._automated_review_coordinator is not None
            and self._automated_review_coordinator is not coordinator
        ) or (
            self.reducer.candidate_store.validator.settings != coordinator.settings
            or self.settings != coordinator.settings.validation
            or self.reducer.candidate_store.root
            != coordinator.workspace_root / coordinator.settings.candidate_path
            or self.root
            != coordinator.workspace_root / coordinator.settings.validation.state_path
        ):
            raise ExpertValidationStoreError(
                "validation store has invalid or conflicting review authority"
            )
        self._automated_review_coordinator = coordinator

    def _require_bound_automated_review_authority(
        self,
        coordinator: object,
    ) -> ExpertAutomatedReviewCoordinator:
        if type(coordinator) is not ExpertAutomatedReviewCoordinator:
            raise ExpertValidationStoreError(
                "automated review publication lacks bound coordinator authority"
            )
        coordinator._require_runner_authority()
        if (
            coordinator is not self._automated_review_coordinator
            or self.reducer.candidate_store.validator.settings != coordinator.settings
            or self.settings != coordinator.settings.validation
            or self.reducer.candidate_store.root
            != coordinator.workspace_root / coordinator.settings.candidate_path
            or self.root
            != coordinator.workspace_root / coordinator.settings.validation.state_path
        ):
            raise ExpertValidationStoreError(
                "automated review publication authority changed after binding"
            )
        return coordinator

    def _bind_release_matrix_stage_authority(
        self,
        coordinator: ExpertReleaseMatrixStageCoordinator,
    ) -> None:
        if type(coordinator) is not ExpertReleaseMatrixStageCoordinator:
            raise ExpertValidationStoreError(
                "release matrix stage coordinator type is invalid"
            )
        execution_store = coordinator.execution_store
        if (
            coordinator.validation_store is not self
            or type(execution_store) is not ExpertTaskEvaluationExecutionStore
            or execution_store.root
            != ExpertTaskEvaluationExecutionStore.canonical_root(self.root).resolve()
            or execution_store.trusted_root != self.root
            or execution_store.policy_settings != self.settings.policy
            or (
                self._release_matrix_stage_coordinator is not None
                and self._release_matrix_stage_coordinator is not coordinator
            )
        ):
            raise ExpertValidationStoreError(
                "validation store has invalid or conflicting release matrix authority"
            )
        self._release_matrix_stage_coordinator = coordinator

    def _require_bound_release_matrix_stage_authority(
        self,
        coordinator: object,
    ) -> ExpertReleaseMatrixStageCoordinator:
        if (
            type(coordinator) is not ExpertReleaseMatrixStageCoordinator
            or coordinator is not self._release_matrix_stage_coordinator
            or coordinator.validation_store is not self
        ):
            raise ExpertValidationStoreError(
                "release matrix stage publication lacks bound coordinator authority"
            )
        execution_store = coordinator.execution_store
        if (
            type(execution_store) is not ExpertTaskEvaluationExecutionStore
            or execution_store.root
            != ExpertTaskEvaluationExecutionStore.canonical_root(self.root).resolve()
            or execution_store.trusted_root != self.root
            or execution_store.policy_settings != self.settings.policy
        ):
            raise ExpertValidationStoreError(
                "release matrix stage publication authority changed after binding"
            )
        return coordinator

    def _bind_publication_eligibility_authority(
        self,
        coordinator: ExpertPublicationEligibilityCoordinator,
    ) -> None:
        if (
            type(coordinator) is not ExpertPublicationEligibilityCoordinator
            or coordinator.validation_store is not self
            or (
                self._publication_eligibility_coordinator is not None
                and self._publication_eligibility_coordinator is not coordinator
            )
        ):
            raise ExpertValidationStoreError(
                "validation store has invalid publication eligibility authority"
            )
        self._publication_eligibility_coordinator = coordinator

    def _bind_release_assembly_authority(self, assembler: object) -> None:
        if type(assembler) is not ExpertReleaseAssembler:
            raise ExpertValidationStoreError(
                "release publication assembler is not the concrete authority"
            )
        if self._release_assembler is not None and self._release_assembler is not (
            assembler
        ):
            raise ExpertValidationStoreError(
                "validation store already has another release assembler"
            )
        self._release_assembler = assembler

    def _bind_release_publisher_authority(self, publisher: object) -> None:
        if type(publisher) is ExpertReleasePublisher:
            publisher._require_local_authority_join()
        if (
            type(publisher) is not ExpertReleasePublisher
            or publisher.validation_store is not self
            or publisher.assembler is not self._release_assembler
            or publisher.github_publisher.resolver is not publisher.resolver
            or publisher.github_publisher.client is not publisher.resolver.client
            or publisher.current_release_authority.resolver is not publisher.resolver
            or self.reducer.current_release_provider
            is not publisher.current_release_authority
            or self.reducer.task_adapter_provider
            is not publisher.task_adapter_authority
            or type(self._publication_eligibility_coordinator)
            is not ExpertPublicationEligibilityCoordinator
            or publisher.task_adapter_authority
            is not self._publication_eligibility_coordinator.task_adapter_authority
            or publisher.security_denylist_authority
            is not self._publication_eligibility_coordinator.security_denylist_authority
            or publisher.release_use_policy_authority
            is not self._publication_eligibility_coordinator.release_use_policy_authority
        ):
            raise ExpertValidationStoreError(
                "release publisher is not the concrete bound authority"
            )
        if self._release_publisher is not None and self._release_publisher is not (
            publisher
        ):
            raise ExpertValidationStoreError(
                "validation store already has another release publisher"
            )
        self._release_publisher = publisher

    def _require_bound_release_publisher_authority(
        self,
        publisher: object,
    ) -> ExpertReleasePublisher:
        if type(publisher) is ExpertReleasePublisher:
            publisher._require_local_authority_join()
        if (
            type(publisher) is not ExpertReleasePublisher
            or publisher is not self._release_publisher
            or publisher.validation_store is not self
            or publisher.assembler is not self._release_assembler
            or publisher.github_publisher.resolver is not publisher.resolver
            or publisher.github_publisher.client is not publisher.resolver.client
            or publisher.current_release_authority.resolver is not publisher.resolver
            or self.reducer.current_release_provider
            is not publisher.current_release_authority
            or self.reducer.task_adapter_provider
            is not publisher.task_adapter_authority
            or type(self._publication_eligibility_coordinator)
            is not ExpertPublicationEligibilityCoordinator
            or publisher.task_adapter_authority
            is not self._publication_eligibility_coordinator.task_adapter_authority
            or publisher.security_denylist_authority
            is not self._publication_eligibility_coordinator.security_denylist_authority
            or publisher.release_use_policy_authority
            is not self._publication_eligibility_coordinator.release_use_policy_authority
        ):
            raise ExpertValidationStoreError(
                "release publication lacks its bound publisher authority"
            )
        return publisher

    def _bind_release_revocation_authority(
        self,
        coordinator: ExpertReleaseRevocationCoordinator,
    ) -> None:
        if (
            type(coordinator) is not ExpertReleaseRevocationCoordinator
            or coordinator.validation_store is not self
            or (
                self._release_revocation_coordinator is not None
                and self._release_revocation_coordinator is not coordinator
            )
        ):
            raise ExpertValidationStoreError(
                "validation store has invalid release revocation authority"
            )
        self._release_revocation_coordinator = coordinator

    def _require_bound_release_revocation_authority(
        self,
        coordinator: object,
    ) -> ExpertReleaseRevocationCoordinator:
        if (
            type(coordinator) is not ExpertReleaseRevocationCoordinator
            or coordinator is not self._release_revocation_coordinator
            or coordinator.validation_store is not self
        ):
            raise ExpertValidationStoreError(
                "release revocation lacks its bound authority"
            )
        return coordinator

    def _seal_stale_release_publication(
        self,
        *,
        publisher: object,
        reservation: ExpertReleasePublicationReservation,
        observed_current: TaskEvaluationCurrentReleaseObservation,
        observed_current_activation_witness: GitHubArtifactActivationWitness,
        own_github_publication_intent: ArtifactPublicationIntent | None,
        own_github_publication_pointer: CurrentArtifactPointer | None,
        own_github_activation_preparation_commit_sha: str | None,
        resolved_at: str,
    ) -> ExpertReleasePublicationStalePermit:
        self._require_bound_release_publisher_authority(publisher)
        if (
            type(reservation) is not ExpertReleasePublicationReservation
            or type(observed_current) is not TaskEvaluationCurrentReleaseObservation
            or type(observed_current_activation_witness)
            is not GitHubArtifactActivationWitness
        ):
            raise ExpertValidationStoreError(
                "stale publication sealing requires exact reservation and CURRENT"
            )
        plan = reservation.plan
        self._validate_release_publication_remote_history(
            reservation.intent,
            plan,
            own_github_publication_intent,
            own_github_publication_pointer,
            self._release_assembler.github_settings.publisher_login,
        )
        planned = plan.current_release_observation
        repositories = publisher.resolver.repositories_for_scope(plan.scope_id)
        if (
            observed_current_activation_witness.scope_id != plan.scope_id
            or observed_current_activation_witness.scope_repository_binding_hash
            != repositories.binding_fingerprint
            or observed_current_activation_witness.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or observed_current_activation_witness.artifact_id
            != observed_current.release_id
            or observed_current_activation_witness.repository_full_name
            != repositories.expert_repository
            or observed_current_activation_witness.activation_commit_sha
            != observed_current.default_branch_head_commit_sha
            or observed_current_activation_witness.current_pointer_digest
            != observed_current.current_pointer_digest
        ):
            raise ExpertValidationStoreError(
                "stale publication winner lacks its activation witness"
            )
        if own_github_activation_preparation_commit_sha is not None and (
            own_github_publication_pointer is None
            or not re.fullmatch(
                r"[0-9a-f]{40}",
                own_github_activation_preparation_commit_sha,
            )
            or own_github_activation_preparation_commit_sha
            == observed_current_activation_witness.activation_commit_sha
        ):
            raise ExpertValidationStoreError(
                "stale publication preparation is inconsistent"
            )
        if (
            observed_current.scope_id != plan.scope_id
            or observed_current.repository_full_name != planned.repository_full_name
            or observed_current.repository_node_id != planned.repository_node_id
            or observed_current.release_id is None
            or observed_current.release_id == plan.release_id
            or observed_current.release_id
            == plan.lineage.activation_predecessor_release_id
            or observed_current == planned
        ):
            raise ExpertValidationStoreError(
                "stale publication sealing lacks displaced CURRENT authority"
            )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            current_intent_id = journal.release_publication_intent_id
            if current_intent_id != reservation.intent.publication_intent_id:
                raise ExpertValidationCompareAndSwapError(
                    "release publication intent changed before stale sealing"
                )
            current = self._current_from_journal_unlocked(journal)
            if current is None:
                raise ExpertValidationStoreError(
                    "stale publication reservation lost its validation head"
                )
            self._validate_release_publication_reservation_unlocked(
                reservation.intent,
                plan,
                reservation.manifest,
                current,
            )
        return ExpertReleasePublicationStalePermit(
            _RELEASE_PUBLICATION_STALE_PERMIT_SEAL,
            self,
            publisher,
            reservation,
            observed_current,
            observed_current_activation_witness,
            own_github_publication_intent,
            own_github_publication_pointer,
            own_github_activation_preparation_commit_sha,
            resolved_at,
        )

    def _seal_release_activation(
        self,
        *,
        publisher: object,
        reservation: ExpertReleasePublicationReservation,
        github_publication_intent: ArtifactPublicationIntent,
        github_publication_pointer: CurrentArtifactPointer,
        activation_witness: GitHubArtifactActivationWitness,
        observed_current: TaskEvaluationCurrentReleaseObservation,
    ) -> ExpertReleaseActivationPermit:
        self._require_bound_release_publisher_authority(publisher)
        if (
            type(reservation) is not ExpertReleasePublicationReservation
            or type(github_publication_intent) is not ArtifactPublicationIntent
            or type(github_publication_pointer) is not CurrentArtifactPointer
            or type(activation_witness) is not GitHubArtifactActivationWitness
            or type(observed_current) is not TaskEvaluationCurrentReleaseObservation
        ):
            raise ExpertValidationStoreError(
                "release activation sealing requires exact remote evidence"
            )
        plan = reservation.plan
        self._validate_release_publication_remote_history(
            reservation.intent,
            plan,
            github_publication_intent,
            github_publication_pointer,
            self._release_assembler.github_settings.publisher_login,
        )
        repositories = publisher.resolver.repositories_for_scope(plan.scope_id)
        if (
            github_publication_intent.artifact_id != plan.release_id
            or not github_publication_intent.binds(github_publication_pointer)
            or activation_witness.scope_id != plan.scope_id
            or activation_witness.scope_repository_binding_hash
            != repositories.binding_fingerprint
            or activation_witness.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or activation_witness.artifact_id != plan.release_id
            or activation_witness.repository_full_name != repositories.expert_repository
            or activation_witness.publication_intent_digest
            != github_publication_intent.digest
            or activation_witness.current_pointer_digest
            != tree_or_blob_digest(github_publication_pointer.to_json_bytes())
            or observed_current.scope_id != plan.scope_id
            or observed_current.repository_full_name != repositories.expert_repository
            or observed_current.repository_node_id
            != plan.current_release_observation.repository_node_id
        ):
            raise ExpertValidationStoreError(
                "release activation evidence does not prove the reserved release"
            )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            replay = self._release_activation_result_unlocked(journal)
            if replay is not None:
                if replay.receipt.publication_plan_id != plan.publication_plan_id:
                    raise ExpertValidationCompareAndSwapError(
                        "release activation conflicts with the durable outcome"
                    )
            else:
                if journal.release_publication_intent_id != (
                    reservation.intent.publication_intent_id
                ):
                    raise ExpertValidationCompareAndSwapError(
                        "release publication intent changed before activation sealing"
                    )
                current = self._current_from_journal_unlocked(journal)
                if current is None:
                    raise ExpertValidationStoreError(
                        "release activation reservation lost its validation head"
                    )
                if current.state.promotion_state not in {
                    ExpertPromotionState.APPROVED,
                    ExpertPromotionState.RELEASE_USE_BLOCKED,
                }:
                    raise ExpertValidationStoreError(
                        "release activation lacks approved or blocked recovery state"
                    )
                approval = self._snapshot_at_unlocked(
                    journal,
                    plan.approval_transition_id,
                )
                self._validate_release_publication_reservation_unlocked(
                    reservation.intent,
                    plan,
                    reservation.manifest,
                    approval,
                )
        return ExpertReleaseActivationPermit(
            _RELEASE_ACTIVATION_PERMIT_SEAL,
            self,
            publisher,
            reservation,
            github_publication_intent,
            github_publication_pointer,
            activation_witness,
            observed_current,
        )

    @staticmethod
    def _validate_release_publication_remote_history(
        reservation_intent: ExpertReleasePublicationIntent,
        plan: ExpertReleasePublicationPlan,
        own_intent: ArtifactPublicationIntent | None,
        own_pointer: CurrentArtifactPointer | None,
        publisher_login: str | None,
    ) -> None:
        if own_intent is None:
            if own_pointer is not None:
                raise ExpertValidationStoreError(
                    "stale publication pointer lacks its remote intent"
                )
            return
        if type(own_intent) is not ArtifactPublicationIntent:
            raise ExpertValidationStoreError(
                "stale publication remote intent is not exact"
            )
        planned = plan.current_release_observation
        expected_assets = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in plan.assets
        )
        observed_assets = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in own_intent.assets
        )
        if plan.activation_predecessor_pointer is None:
            preserved_current_matches = own_intent.preserved_current is None
        else:
            payload = plan.activation_predecessor_pointer.to_json_bytes()
            preserved = own_intent.preserved_current
            preserved_current_matches = preserved is not None and (
                preserved.relative_path == "CURRENT.json"
                and preserved.mode == "100644"
                and preserved.size == len(payload)
                and preserved.sha256 == tree_or_blob_digest(payload)
                and preserved.git_blob_sha == git_object_sha("blob", payload)
            )
        if (
            own_intent.scope_id != plan.scope_id
            or own_intent.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or own_intent.artifact_id != plan.release_id
            or own_intent.repository_node_id != planned.repository_node_id
            or own_intent.repository_full_name != planned.repository_full_name
            or own_intent.expected_parent_sha != planned.default_branch_head_commit_sha
            or own_intent.source_tree_digest != plan.publication_source_tree_digest
            or own_intent.manifest_relative_path != EXPERT_RELEASE_MANIFEST_PATH
            or own_intent.manifest_digest != plan.manifest_digest
            or own_intent.tag != plan.tag
            or observed_assets != expected_assets
            or own_intent.validation_closure_ids != plan.validation_closure_ids
            or (
                publisher_login is not None
                and own_intent.publisher_identity != publisher_login
            )
            or own_intent.committed_at != reservation_intent.committed_at
            or not preserved_current_matches
        ):
            raise ExpertValidationStoreError(
                "stale publication remote history differs from its reservation"
            )
        if own_pointer is None:
            return
        if type(own_pointer) is not CurrentArtifactPointer:
            raise ExpertValidationStoreError(
                "stale publication remote pointer is not exact"
            )
        record = own_pointer.publication_record
        pointer_assets = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in record.assets
        )
        if (
            not own_intent.binds(own_pointer)
            or own_pointer.scope_id != plan.scope_id
            or record.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or record.artifact_id != plan.release_id
            or record.repository_node_id != planned.repository_node_id
            or record.repository_full_name != planned.repository_full_name
            or record.tag != plan.tag
            or (
                publisher_login is not None
                and record.publisher_identity != publisher_login
            )
            or pointer_assets != expected_assets
            or own_pointer.source_tree_digest != plan.publication_source_tree_digest
            or own_pointer.manifest_relative_path != EXPERT_RELEASE_MANIFEST_PATH
            or own_pointer.manifest_digest != plan.manifest_digest
            or own_pointer.validation_closure_ids != plan.validation_closure_ids
        ):
            raise ExpertValidationStoreError(
                "stale publication remote pointer differs from its reservation"
            )

    def _require_bound_release_assembly_authority(
        self,
        assembler: object,
    ) -> object:
        if (
            assembler is None
            or assembler is not self._release_assembler
            or type(assembler) is not ExpertReleaseAssembler
        ):
            raise ExpertValidationStoreError(
                "release publication lacks its bound assembler authority"
            )
        return assembler

    def _require_bound_publication_eligibility_authority(
        self,
        coordinator: object,
    ) -> ExpertPublicationEligibilityCoordinator:
        if (
            type(coordinator) is not ExpertPublicationEligibilityCoordinator
            or coordinator is not self._publication_eligibility_coordinator
            or coordinator.validation_store is not self
            or coordinator.current_release_authority
            is not self.reducer.current_release_provider
            or coordinator.task_adapter_authority
            is not self.reducer.task_adapter_provider
            or coordinator.release_use_policy_authority is None
        ):
            raise ExpertValidationStoreError(
                "publication eligibility lacks bound coordinator authority"
            )
        return coordinator

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
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
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
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
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

    def publish_automated_review_stage(
        self,
        execution: ExpertAutomatedReviewExecution,
    ) -> ExpertAutomatedReviewStageCommitResult:
        if type(execution) is not ExpertAutomatedReviewExecution:
            raise ExpertValidationStoreError(
                "automated review publication requires its complete execution"
            )
        packet = execution.prepared_packet.packet
        operation = self._automated_review_operation(packet)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(packet.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertAutomatedReviewStageCommitResult(
                    stage_result=self._automated_review_result_for_transition_unlocked(
                        replay.transition
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            observed = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                observed,
                packet.authorization_transition_id,
            )
            if observed is None or observed.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "automated review requires a current validation attempt"
                )
        coordinator = self._require_bound_automated_review_authority(
            self._automated_review_coordinator
        )
        prepared, result, target_state = self._validate_automated_review_execution(
            execution,
            observed,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(packet.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertAutomatedReviewStageCommitResult(
                    stage_result=self._automated_review_result_for_transition_unlocked(
                        replay.transition
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                current,
                packet.authorization_transition_id,
            )
            if current != observed or current.latest_attempt is None:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during automated review checks"
                )
            execution._consume(coordinator)
            accepted_ids = current.transition.accepted_stage_result_record_ids
            if result.outcome is ExpertAutomatedReviewOutcome.PASSED:
                accepted_ids = (*accepted_ids, result.stage_result_record_id)
            transition = ExpertValidationTransition.mint(
                candidate_id=packet.candidate_id,
                candidate_tree_hash=packet.candidate_tree_hash,
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
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
            )
            self._write_contract_unlocked(prepared.candidate_input)
            self._write_automated_review_derivation_unlocked(prepared)
            self._write_contract_unlocked(packet)
            for review_operation in execution.operation_records:
                self._write_contract_unlocked(review_operation.operation_receipt)
                self._write_contract_unlocked(review_operation)
            for assertion in execution.assertions:
                self._write_contract_unlocked(assertion)
            self._write_contract_unlocked(execution.adjudication)
            self._write_contract_unlocked(result)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertAutomatedReviewStageCommitResult(
                stage_result=result,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def reopen_or_replay_release_matrix_stage(
        self,
        *,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertReleaseMatrixStageCommitResult | None:
        """Validate the reserved head or return its committed matrix result."""

        if (
            type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
            or type(prepared_request) is not PreparedTaskEvaluationRequest
        ):
            raise ExpertValidationStoreError(
                "release matrix stage replay requires exact task authorities"
            )
        prepared = PreparedTaskEvaluationRequest(
            plan_join=prepared_request.plan_join,
            stored_candidate=prepared_request.stored_candidate,
            candidate=prepared_request.candidate,
            source_base=prepared_request.source_base,
            current_release_observation=(prepared_request.current_release_observation),
            cases=prepared_request.cases,
        )
        reservation = reservation_snapshot.reservation
        operation = self._release_matrix_stage_operation(reservation)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            observed_reservation = (
                self._release_matrix_task_reservation_snapshot_unlocked(
                    journal,
                    reservation.authorization_transition_id,
                )
            )
            if (
                observed_reservation != reservation_snapshot
                or observed_reservation.request != prepared.plan_join.request
            ):
                raise ExpertValidationCompareAndSwapError(
                    "release matrix stage task reservation authority changed"
                )
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertReleaseMatrixStageCommitResult(
                    stage_result=(
                        self._release_matrix_stage_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                current,
                reservation.authorization_transition_id,
            )
            if (
                current != observed_reservation.plan_reservation.snapshot
                or current.state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
            ):
                raise ExpertValidationStoreError(
                    "release matrix stage lacks the current reserved stage head"
                )
        return None

    def publish_release_matrix_stage(
        self,
        execution: ExpertReleaseMatrixStageExecution,
    ) -> ExpertReleaseMatrixStageCommitResult:
        """CAS one sealed factual matrix into the accepted validation prefix."""

        if type(execution) is not ExpertReleaseMatrixStageExecution:
            raise ExpertValidationStoreError(
                "release matrix stage publication requires its sealed execution"
            )
        reservation_snapshot = execution.reservation_snapshot
        prepared = execution.prepared_request
        reservation = reservation_snapshot.reservation
        operation = self._release_matrix_stage_operation(reservation)
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertReleaseMatrixStageCommitResult(
                    stage_result=(
                        self._release_matrix_stage_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            observed_reservation = (
                self._release_matrix_task_reservation_snapshot_unlocked(
                    journal,
                    reservation.authorization_transition_id,
                )
            )
            if (
                observed_reservation != reservation_snapshot
                or observed_reservation.request != prepared.plan_join.request
            ):
                raise ExpertValidationCompareAndSwapError(
                    "release matrix task reservation changed before publication"
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                current,
                reservation.authorization_transition_id,
            )
            if current != observed_reservation.plan_reservation.snapshot:
                raise ExpertValidationCompareAndSwapError(
                    "release matrix validation head differs from its reservation"
                )
        coordinator = self._require_bound_release_matrix_stage_authority(
            self._release_matrix_stage_coordinator
        )
        execution._require_bound(
            coordinator,
            self,
            coordinator.execution_store,
        )
        execution.completed_execution.require_exact(
            coordinator.execution_store,
            reservation_snapshot,
            prepared,
        )
        result = execution.stage_result
        report = result.release_matrix_report
        if (
            result.task_evaluation_reservation_id != reservation.reservation_id
            or result.authorization_transition_id
            != reservation.authorization_transition_id
            or result.authorization_state_id != reservation.authorization_state_id
            or result.validation_attempt_id != reservation.validation_attempt_id
            or result.candidate_id != reservation.candidate_id
            or result.candidate_tree_hash != reservation.candidate_tree_hash
            or result.scope_contract_id != reservation.scope_contract_id
            or result.source_base_release_id
            != reservation_snapshot.request.source_base_release_id
            or result.plan_reservation_operation_id
            != reservation.plan_reservation_operation_id
            or result.validation_policy_id
            != reservation_snapshot.request.validation_policy_id
            or result.configuration_fingerprint
            != reservation_snapshot.request.configuration_fingerprint
            or report.evaluation_plan
            != reservation_snapshot.plan_reservation.evaluation_plan
        ):
            raise ExpertValidationStoreError(
                "release matrix stage result differs from its sealed reservation"
            )
        attempt = observed_reservation.plan_reservation.snapshot.latest_attempt
        if attempt is None:
            raise ExpertValidationStoreError(
                "release matrix stage publication has no active attempt"
            )
        target_state = self.reducer.advance_release_matrix_stage(
            state=observed_reservation.plan_reservation.snapshot.state,
            attempt=attempt,
            accepted_results=(
                observed_reservation.plan_reservation.snapshot.accepted_stage_results
            ),
            result=result,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(reservation.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertReleaseMatrixStageCommitResult(
                    stage_result=(
                        self._release_matrix_stage_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current_reservation = (
                self._release_matrix_task_reservation_snapshot_unlocked(
                    journal,
                    reservation.authorization_transition_id,
                )
            )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(
                current,
                reservation.authorization_transition_id,
            )
            if (
                current_reservation != observed_reservation
                or current != observed_reservation.plan_reservation.snapshot
                or current.latest_attempt is None
            ):
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during release matrix reduction"
                )
            execution._consume(
                coordinator,
                self,
                coordinator.execution_store,
            )
            accepted_ids = (
                *current.transition.accepted_stage_result_record_ids,
                result.stage_result_record_id,
            )
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
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
            )
            self._write_contract_unlocked(report)
            self._write_contract_unlocked(result)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertReleaseMatrixStageCommitResult(
                stage_result=result,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def reopen_or_replay_publication_eligibility(
        self,
        *,
        candidate_id: str,
        release_matrix_stage_result_id: str,
    ) -> tuple[
        ExpertPublicationEligibilityStageCommitResult | None,
        ExpertPublicationEligibilitySnapshot | None,
    ]:
        """Return a committed terminal result or the exact current matrix head."""

        require_content_id(candidate_id, "publication eligibility candidate_id")
        require_content_id(
            release_matrix_stage_result_id,
            "publication eligibility matrix result ID",
        )
        if release_matrix_stage_result_id.split(":sha256:", 1)[0] != (
            "expert-release-matrix-stage-result"
        ):
            raise ExpertValidationStoreError(
                "publication eligibility matrix result uses the wrong namespace"
            )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            self._validate_journal_unlocked(journal)
            replay = self._publication_eligibility_commit_unlocked(
                journal,
                release_matrix_stage_result_id,
            )
            if replay is not None:
                return replay, None
            current = self._current_from_journal_unlocked(journal)
            if current is None or current.latest_attempt is None:
                raise ExpertValidationStoreError(
                    "publication eligibility has no current validation attempt"
                )
            matrix_result = self._read_contract_unlocked(
                release_matrix_stage_result_id,
                ExpertReleaseMatrixStageResultRecord,
            )
            if (
                not current.accepted_stage_results
                or current.accepted_stage_results[-1] != matrix_result
            ):
                raise ExpertValidationStoreError(
                    "publication eligibility matrix is not the current accepted head"
                )
            return (
                None,
                ExpertPublicationEligibilitySnapshot(
                    snapshot=current,
                    release_matrix_stage_result=matrix_result,
                ),
            )

    def reopen_publication_eligibility_snapshot(
        self,
        input_snapshot: ExpertPublicationEligibilitySnapshot,
    ) -> ExpertPublicationEligibilitySnapshot:
        """Reopen an unchanged terminal input after fresh external checks."""

        if type(input_snapshot) is not ExpertPublicationEligibilitySnapshot:
            raise ExpertValidationStoreError(
                "publication eligibility reopen requires its exact snapshot"
            )
        candidate_id = input_snapshot.snapshot.state.candidate_id
        matrix_id = input_snapshot.release_matrix_stage_result.stage_result_record_id
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current != input_snapshot.snapshot:
                raise ExpertValidationCompareAndSwapError(
                    "publication eligibility validation head changed"
                )
            matrix_result = self._read_contract_unlocked(
                matrix_id,
                ExpertReleaseMatrixStageResultRecord,
            )
            reopened = ExpertPublicationEligibilitySnapshot(
                snapshot=current,
                release_matrix_stage_result=matrix_result,
            )
            if reopened != input_snapshot:
                raise ExpertValidationStoreError(
                    "publication eligibility matrix authority changed"
                )
            return reopened

    def publish_publication_eligibility(
        self,
        execution: ExpertPublicationEligibilityExecution,
    ) -> ExpertPublicationEligibilityStageCommitResult:
        """CAS one sealed Pareto decision into its terminal validation state."""

        if type(execution) is not ExpertPublicationEligibilityExecution:
            raise ExpertValidationStoreError(
                "publication eligibility requires a sealed execution"
            )
        coordinator = self._require_bound_publication_eligibility_authority(
            self._publication_eligibility_coordinator
        )
        execution._require_bound(coordinator, self)
        input_snapshot = execution.input_snapshot
        decision = execution.decision
        result = execution.stage_result
        operation = self._publication_eligibility_operation(
            input_snapshot,
            result,
        )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(
                input_snapshot.snapshot.state.candidate_id
            )
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertPublicationEligibilityStageCommitResult(
                    stage_result=(
                        self._publication_eligibility_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            if current != input_snapshot.snapshot or current.latest_attempt is None:
                raise ExpertValidationCompareAndSwapError(
                    "publication eligibility head changed before reduction"
                )
            stored_candidate = self.reducer.candidate_store.read(
                current.state.candidate_id
            )
            self._validate_publication_eligibility_execution(
                execution=execution,
                current=current,
                stored_candidate=stored_candidate,
            )
            target_state = self.reducer.advance_publication_eligibility_stage(
                state=current.state,
                attempt=current.latest_attempt,
                accepted_results=current.accepted_stage_results,
                result=result,
            )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(current.state.candidate_id)
            replay = self._resolved_operation_unlocked(journal, operation)
            if replay is not None:
                return ExpertPublicationEligibilityStageCommitResult(
                    stage_result=(
                        self._publication_eligibility_result_for_transition_unlocked(
                            replay.transition
                        )
                    ),
                    snapshot=replay,
                    replayed=True,
                )
            observed = self._current_from_journal_unlocked(journal)
            if observed != current or observed.latest_attempt is None:
                raise ExpertValidationCompareAndSwapError(
                    "publication eligibility head changed during reduction"
                )
            execution._consume(coordinator, self)
            accepted_ids = observed.transition.accepted_stage_result_record_ids
            if result.publication_authority_fence is not None:
                accepted_ids = (*accepted_ids, result.stage_result_record_id)
            transition = ExpertValidationTransition.mint(
                candidate_id=observed.state.candidate_id,
                candidate_tree_hash=observed.state.candidate_tree_hash,
                transition_number=len(journal.transition_ids) + 1,
                predecessor_transition_id=observed.transition.transition_id,
                predecessor_state_id=observed.state.validation_state_id,
                target_state_id=target_state.validation_state_id,
                latest_attempt_id=observed.latest_attempt.validation_attempt_id,
                operation_id=operation.operation_id,
                validation_policy_id=observed.latest_attempt.validation_policy_id,
                configuration_fingerprint=(
                    observed.latest_attempt.configuration_fingerprint
                ),
                eligibility_decision_id=None,
                created_attempt_id=None,
                accepted_stage_result_record_ids=accepted_ids,
                transition_stage_result_record_id=result.stage_result_record_id,
                transition_authority_invalidation_id=None,
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
            )
            self._write_contract_unlocked(decision)
            if result.release_use_decision is not None:
                self._write_contract_unlocked(
                    result.release_use_decision.policy_observation
                )
                for (
                    revocation
                ) in result.release_use_decision.policy_observation.matched_revocations:
                    self._write_contract_unlocked(revocation)
                self._write_contract_unlocked(result.release_use_decision)
            if result.publication_authority_fence is not None:
                self._write_contract_unlocked(result.publication_authority_fence)
            self._write_contract_unlocked(result)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertPublicationEligibilityStageCommitResult(
                stage_result=result,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def _seal_release_use_block(
        self,
        *,
        publisher: object,
        approval_snapshot: ExpertValidationSnapshot,
        decision: ExpertCandidateReleaseUseDecision,
    ) -> ExpertReleaseUseBlockPermit:
        """Seal one fresh matched policy read against an unchanged approval."""

        authority = self._require_bound_release_publisher_authority(publisher)
        if (
            type(approval_snapshot) is not ExpertValidationSnapshot
            or type(decision) is not ExpertCandidateReleaseUseDecision
            or decision.outcome is not ExpertCandidateReleaseUseOutcome.BLOCKED
        ):
            raise ExpertValidationStoreError(
                "release-use block sealing requires exact matched authority"
            )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(decision.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current != approval_snapshot or current.latest_attempt is None:
                raise ExpertValidationCompareAndSwapError(
                    "release-use block approval changed during policy checks"
                )
            publication_result = current.accepted_stage_results[-1]
            if (
                type(publication_result)
                is not ExpertPublicationEligibilityStageResultRecord
            ):
                raise ExpertValidationStoreError(
                    "release-use block lacks publication eligibility authority"
                )
            stored_candidate = self.reducer.candidate_store.read(
                current.state.candidate_id
            )
            if (
                decision.policy_observation.checked_release_ids
                != stored_candidate.closure.manifest.consumed_expert_release_ids
            ):
                raise ExpertValidationStoreError(
                    "release-use block checked another candidate release closure"
                )
            self.reducer.advance_release_use_block(
                state=current.state,
                attempt=current.latest_attempt,
                publication_result=publication_result,
                decision=decision,
            )
        return ExpertReleaseUseBlockPermit(
            _RELEASE_USE_BLOCK_PERMIT_SEAL,
            self,
            authority,
            approval_snapshot,
            decision,
        )

    def commit_release_use_block(
        self,
        permit: ExpertReleaseUseBlockPermit,
    ) -> ExpertReleaseUseBlockCommitResult:
        """Atomically append the first permanent late release-use block."""

        if type(permit) is not ExpertReleaseUseBlockPermit:
            raise ExpertValidationStoreError(
                "release-use block requires a publisher-sealed permit"
            )
        publisher = self._require_bound_release_publisher_authority(
            self._release_publisher
        )
        permit._require_bound(self, publisher)
        approval_snapshot = permit._approval_snapshot
        decision = permit._decision
        operation = ExpertValidationOperation.mint(
            operation_kind=ExpertValidationOperationKind.RELEASE_USE_BLOCK,
            candidate_id=decision.candidate_id,
            expected_transition_id=approval_snapshot.transition.transition_id,
            request_record_id=decision.release_use_decision_id,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(decision.candidate_id)
            replay = self._release_use_block_result_unlocked(journal)
            if replay is not None:
                permit._consume(self, publisher)
                return replay
            current = self._current_from_journal_unlocked(journal)
            if current != approval_snapshot or current.latest_attempt is None:
                raise ExpertValidationCompareAndSwapError(
                    "release-use block approval changed before commit"
                )
            publication_result = current.accepted_stage_results[-1]
            if (
                type(publication_result)
                is not ExpertPublicationEligibilityStageResultRecord
            ):
                raise ExpertValidationStoreError(
                    "release-use block lost publication eligibility authority"
                )
            target_state = self.reducer.advance_release_use_block(
                state=current.state,
                attempt=current.latest_attempt,
                publication_result=publication_result,
                decision=decision,
            )
            transition = ExpertValidationTransition.mint(
                candidate_id=decision.candidate_id,
                candidate_tree_hash=decision.candidate_tree_hash,
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
                accepted_stage_result_record_ids=(
                    current.transition.accepted_stage_result_record_ids
                ),
                transition_stage_result_record_id=None,
                transition_authority_invalidation_id=None,
                transition_release_use_block_decision_id=(
                    decision.release_use_decision_id
                ),
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
            )
            permit._consume(self, publisher)
            self._write_contract_unlocked(decision.policy_observation)
            for revocation in decision.policy_observation.matched_revocations:
                self._write_contract_unlocked(revocation)
            self._write_contract_unlocked(decision)
            self._write_contract_unlocked(target_state)
            self._write_contract_unlocked(operation)
            self._write_contract_unlocked(transition)
            updated = self._append_release_use_block_transition(journal, transition)
            self._publish_journal_unlocked(updated)
            return ExpertReleaseUseBlockCommitResult(
                decision=decision,
                snapshot=self._snapshot_at_unlocked(
                    updated,
                    transition.transition_id,
                ),
                replayed=False,
            )

    def reopen_release_use_block(
        self,
        candidate_id: str,
    ) -> ExpertReleaseUseBlockCommitResult | None:
        """Reopen the first durable release-use block without external reads."""

        require_content_id(candidate_id, "release-use block candidate_id")
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(candidate_id)
            return self._release_use_block_result_unlocked(journal)

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
            source_base=prepared_request.source_base,
            authorization_state=prepared_request.authorization_state,
            recovery_admission=prepared_request.recovery_admission,
            cases=prepared_request.cases,
        )
        request = prepared.request
        recovery_admission = prepared.recovery_admission
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
                expected_current_release_id=request.expected_current_release_id,
                exact_dependency_ids=tuple(
                    sorted(
                        {
                            request.execution_request_id,
                            current.transition.transition_id,
                            current.latest_attempt.validation_attempt_id,
                            current.state.validation_state_id,
                            current.state.candidate_id,
                            request.expected_current_release_id,
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
            if recovery_admission is not None:
                self._write_contract_unlocked(recovery_admission)
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

    def reserve_release_matrix_plan(
        self,
        *,
        expected_transition_id: str,
        prepared_plan: PreparedExpertReleaseMatrixPlan,
    ) -> ExpertReleaseMatrixPlanReservationCommitResult:
        require_content_id(expected_transition_id, "expected_transition_id")
        if type(prepared_plan) is not PreparedExpertReleaseMatrixPlan:
            raise ExpertValidationStoreError(
                "release matrix reservation requires a prepared plan"
            )
        plan = prepared_plan.plan
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            existing = self._release_matrix_plan_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if existing is not None:
                operation, stored_plan = existing
                if stored_plan != plan:
                    raise ExpertValidationCompareAndSwapError(
                        "validation head already reserves another release matrix plan"
                    )
                current = self._current_from_journal_unlocked(journal)
                self._require_expected_head(current, expected_transition_id)
                if current is None:
                    raise ExpertValidationStoreError(
                        "release matrix reservation has no validation head"
                    )
                return ExpertReleaseMatrixPlanReservationCommitResult(
                    reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                        operation=operation,
                        evaluation_plan=stored_plan,
                        snapshot=current,
                    ),
                    replayed=True,
                )
            observed = self._current_from_journal_unlocked(journal)
            self._require_expected_head(observed, expected_transition_id)
            if (
                observed is None
                or observed.latest_attempt is None
                or observed.state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
            ):
                raise ExpertValidationStoreError(
                    "release matrix reservation requires the active matrix stage"
                )
        admitted = prepare_expert_release_matrix_plan_for_admission(
            prepared_plan=prepared_plan,
            state=observed.state,
            attempt=observed.latest_attempt,
            accepted_stage_results=observed.accepted_stage_results,
            source_replay_request=prepared_plan.source_replay_request,
            candidate_store=self.reducer.candidate_store,
            current_release_provider=self.reducer.current_release_provider,
            task_adapter_provider=self.reducer.task_adapter_provider,
            validation_policy=self.settings.policy.validation_policy(),
            validation_settings=self.settings,
        )
        if admitted != prepared_plan:
            raise ExpertValidationStoreError(
                "release matrix plan differs from fresh admission authority"
            )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(plan.candidate_id)
            existing = self._release_matrix_plan_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if existing is not None:
                operation, stored_plan = existing
                if stored_plan != plan:
                    raise ExpertValidationCompareAndSwapError(
                        "validation head already reserves another release matrix plan"
                    )
                current = self._current_from_journal_unlocked(journal)
                self._require_expected_head(current, expected_transition_id)
                if current is None:
                    raise ExpertValidationStoreError(
                        "release matrix reservation lost its validation head"
                    )
                return ExpertReleaseMatrixPlanReservationCommitResult(
                    reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                        operation=operation,
                        evaluation_plan=stored_plan,
                        snapshot=current,
                    ),
                    replayed=True,
                )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current != observed:
                raise ExpertValidationCompareAndSwapError(
                    "validation head changed during release matrix plan admission"
                )
            operation = ExpertValidationOperation.mint(
                operation_kind=(
                    ExpertValidationOperationKind.RELEASE_MATRIX_PLAN_RESERVATION
                ),
                candidate_id=plan.candidate_id,
                expected_transition_id=expected_transition_id,
                request_record_id=plan.evaluation_plan_id,
            )
            self._write_contract_unlocked(plan)
            self._write_contract_unlocked(operation)
            updated = self._bind_operation(journal, operation, current.transition)
            self._publish_journal_unlocked(updated)
            return ExpertReleaseMatrixPlanReservationCommitResult(
                reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                    operation=operation,
                    evaluation_plan=plan,
                    snapshot=self._snapshot_at_unlocked(
                        updated,
                        current.transition.transition_id,
                    ),
                ),
                replayed=False,
            )

    def reopen_release_matrix_plan_reservation(
        self,
        *,
        evaluation_plan_id: str,
        prepared_plan: PreparedExpertReleaseMatrixPlan,
    ) -> ExpertReleaseMatrixPlanReservationSnapshot:
        require_content_id(evaluation_plan_id, "release matrix evaluation_plan_id")
        if evaluation_plan_id.split(":sha256:", 1)[0] != (
            "expert-release-matrix-evaluation-plan"
        ):
            raise ExpertValidationStoreError(
                "release matrix evaluation_plan_id uses the wrong namespace"
            )
        if type(prepared_plan) is not PreparedExpertReleaseMatrixPlan:
            raise ExpertValidationStoreError(
                "release matrix reopen requires a prepared plan"
            )
        plan = prepared_plan.plan
        if plan.evaluation_plan_id != evaluation_plan_id:
            raise ExpertValidationStoreError(
                "release matrix reopen plan identity differs"
            )
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None:
                raise ExpertValidationStoreError(
                    "release matrix reservation has no current validation head"
                )
            stored = self._release_matrix_plan_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            if stored is None:
                raise ExpertValidationStoreError(
                    "release matrix plan is not bound to the current head"
                )
            operation, stored_plan = stored
            if stored_plan != plan:
                raise ExpertValidationStoreError(
                    "stored release matrix plan differs from its prepared closure"
                )
            return ExpertReleaseMatrixPlanReservationSnapshot(
                operation=operation,
                evaluation_plan=stored_plan,
                snapshot=current,
            )

    def reopen_release_matrix_plan_reservation_snapshot(
        self,
        *,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    ) -> ExpertReleaseMatrixPlanReservationSnapshot:
        """Reopen one exact plan alias at the unchanged current head."""

        if type(plan_reservation) is not ExpertReleaseMatrixPlanReservationSnapshot:
            raise ExpertValidationStoreError(
                "release matrix snapshot reopen requires exact reserved authority"
            )
        plan = plan_reservation.evaluation_plan
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None or current != plan_reservation.snapshot:
                raise ExpertValidationStoreError(
                    "release matrix snapshot reopen head changed"
                )
            stored = self._release_matrix_plan_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            if stored != (
                plan_reservation.operation,
                plan_reservation.evaluation_plan,
            ):
                raise ExpertValidationStoreError(
                    "release matrix snapshot reopen alias changed"
                )
            return ExpertReleaseMatrixPlanReservationSnapshot(
                operation=stored[0],
                evaluation_plan=stored[1],
                snapshot=current,
            )

    def reserve_task_evaluation(
        self,
        *,
        expected_transition_id: str,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertTaskEvaluationReservationCommitResult:
        """Atomically bind one byte-closed task request to the current plan head."""

        require_content_id(expected_transition_id, "expected_transition_id")
        if type(prepared_request) is not PreparedTaskEvaluationRequest:
            raise ExpertValidationStoreError(
                "task evaluation reservation requires an exact prepared request"
            )
        prepared = PreparedTaskEvaluationRequest(
            plan_join=prepared_request.plan_join,
            stored_candidate=prepared_request.stored_candidate,
            candidate=prepared_request.candidate,
            source_base=prepared_request.source_base,
            current_release_observation=(prepared_request.current_release_observation),
            cases=prepared_request.cases,
        )
        request = prepared.plan_join.request
        recovery_admission = prepared.stored_candidate.recovery_admission
        supplied_plan_reservation = prepared.plan_join.plan_reservation
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(request.candidate_id)
            existing = self._task_evaluation_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            current = self._current_from_journal_unlocked(journal)
            self._require_expected_head(current, expected_transition_id)
            if current is None:
                raise ExpertValidationStoreError(
                    "task evaluation reservation has no validation head"
                )
            stored_plan = self._release_matrix_plan_reservation_unlocked(
                journal,
                expected_transition_id,
            )
            if (
                stored_plan
                != (
                    supplied_plan_reservation.operation,
                    supplied_plan_reservation.evaluation_plan,
                )
                or current != supplied_plan_reservation.snapshot
            ):
                raise ExpertValidationCompareAndSwapError(
                    "task evaluation reservation plan authority changed"
                )
            if existing is not None:
                operation, reservation, stored_request, observation = existing
                if stored_request != request:
                    raise ExpertValidationCompareAndSwapError(
                        "validation head already reserves another task evaluation"
                    )
                return ExpertTaskEvaluationReservationCommitResult(
                    reservation=ExpertTaskEvaluationReservationSnapshot(
                        operation=operation,
                        reservation=reservation,
                        request=stored_request,
                        current_release_observation=observation,
                        plan_reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                            operation=stored_plan[0],
                            evaluation_plan=stored_plan[1],
                            snapshot=current,
                        ),
                    ),
                    replayed=True,
                )
            observation = prepared.current_release_observation
            dependencies = {
                request.request_id,
                request.plan_reservation_operation_id,
                request.evaluation_plan_id,
                request.authorization_transition_id,
                request.authorization_state_id,
                request.validation_attempt_id,
                request.candidate_id,
                request.scope_contract_id,
                observation.observation_id,
            }
            if request.expected_current_release_id is not None:
                dependencies.add(request.expected_current_release_id)
            reservation = TaskEvaluationReservation.mint(
                request_id=request.request_id,
                plan_reservation_operation_id=(request.plan_reservation_operation_id),
                evaluation_plan_id=request.evaluation_plan_id,
                mode=request.mode,
                authorization_transition_id=request.authorization_transition_id,
                authorization_state_id=request.authorization_state_id,
                validation_attempt_id=request.validation_attempt_id,
                candidate_id=request.candidate_id,
                candidate_tree_hash=request.candidate_tree_hash,
                scope_contract_id=request.scope_contract_id,
                scope_id=request.scope_id,
                current_release_observation_id=observation.observation_id,
                observed_current_release_id=observation.release_id,
                exact_dependency_ids=tuple(sorted(dependencies)),
            )
            operation = ExpertValidationOperation.mint(
                operation_kind=(
                    ExpertValidationOperationKind.TASK_EVALUATION_RESERVATION
                ),
                candidate_id=request.candidate_id,
                expected_transition_id=current.transition.transition_id,
                request_record_id=reservation.reservation_id,
            )
            self._write_contract_unlocked(request)
            if recovery_admission is not None:
                self._write_contract_unlocked(recovery_admission)
            self._write_contract_unlocked(observation)
            self._write_contract_unlocked(reservation)
            self._write_contract_unlocked(operation)
            updated = self._bind_operation(journal, operation, current.transition)
            self._publish_journal_unlocked(updated)
            return ExpertTaskEvaluationReservationCommitResult(
                reservation=ExpertTaskEvaluationReservationSnapshot(
                    operation=operation,
                    reservation=reservation,
                    request=request,
                    current_release_observation=observation,
                    plan_reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                        operation=stored_plan[0],
                        evaluation_plan=stored_plan[1],
                        snapshot=self._snapshot_at_unlocked(
                            updated,
                            current.transition.transition_id,
                        ),
                    ),
                ),
                replayed=False,
            )

    def reopen_task_evaluation_reservation(
        self,
        *,
        reservation_id: str,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertTaskEvaluationReservationSnapshot:
        """Reopen one current task reservation from its exact prepared bytes."""

        require_content_id(reservation_id, "task evaluation reservation_id")
        if reservation_id.split(":sha256:", 1)[0] != "task-evaluation-reservation":
            raise ExpertValidationStoreError(
                "task evaluation reservation_id uses the wrong namespace"
            )
        if type(prepared_request) is not PreparedTaskEvaluationRequest:
            raise ExpertValidationStoreError(
                "task evaluation reopen requires an exact prepared request"
            )
        prepared = PreparedTaskEvaluationRequest(
            plan_join=prepared_request.plan_join,
            stored_candidate=prepared_request.stored_candidate,
            candidate=prepared_request.candidate,
            source_base=prepared_request.source_base,
            current_release_observation=(prepared_request.current_release_observation),
            cases=prepared_request.cases,
        )
        request = prepared.plan_join.request
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(request.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None:
                raise ExpertValidationStoreError(
                    "task evaluation reservation has no current validation head"
                )
            stored = self._task_evaluation_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            if stored is None:
                raise ExpertValidationStoreError(
                    "task evaluation reservation is not bound to the current head"
                )
            operation, reservation, stored_request, observation = stored
            if reservation.reservation_id != reservation_id:
                raise ExpertValidationStoreError(
                    "task evaluation reservation identity is not current"
                )
            if stored_request != request:
                raise ExpertValidationStoreError(
                    "task evaluation stored request differs from prepared bytes"
                )
            stored_plan = self._release_matrix_plan_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            supplied_plan = prepared.plan_join.plan_reservation
            if (
                stored_plan
                != (
                    supplied_plan.operation,
                    supplied_plan.evaluation_plan,
                )
                or current != supplied_plan.snapshot
            ):
                raise ExpertValidationStoreError(
                    "task evaluation stored plan differs from prepared bytes"
                )
            return ExpertTaskEvaluationReservationSnapshot(
                operation=operation,
                reservation=reservation,
                request=stored_request,
                current_release_observation=observation,
                plan_reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                    operation=stored_plan[0],
                    evaluation_plan=stored_plan[1],
                    snapshot=current,
                ),
            )

    def reopen_release_matrix_source_evidence(
        self,
        *,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    ) -> ExpertReleaseMatrixSourceEvidenceSnapshot:
        """Resolve accepted source facts without rerunning source preflight or work."""

        if type(plan_reservation) is not ExpertReleaseMatrixPlanReservationSnapshot:
            raise ExpertValidationStoreError(
                "release matrix source evidence requires a plan reservation"
            )
        plan = plan_reservation.evaluation_plan
        with self._lock(exclusive=False):
            journal = self._read_journal_unlocked(plan.candidate_id)
            current = self._current_from_journal_unlocked(journal)
            if current is None or current != plan_reservation.snapshot:
                raise ExpertValidationStoreError(
                    "release matrix source evidence head changed"
                )
            stored = self._release_matrix_plan_reservation_unlocked(
                journal,
                current.transition.transition_id,
            )
            if stored != (
                plan_reservation.operation,
                plan_reservation.evaluation_plan,
            ):
                raise ExpertValidationStoreError(
                    "release matrix source evidence plan reservation changed"
                )
            source_results = tuple(
                result
                for result in current.accepted_stage_results
                if type(result) is ExpertSourceReplayStageResultRecord
            )
            if len(source_results) != 1:
                raise ExpertValidationStoreError(
                    "release matrix source evidence requires one accepted source result"
                )
            stage_result = source_results[0]
            if stage_result.outcome is not ExpertEvaluatorOutcome.PASSED:
                raise ExpertValidationStoreError(
                    "release matrix source evidence result did not pass"
                )
            reservation = self._read_contract_unlocked(
                stage_result.reservation_id,
                ExpertSourceReplayExecutionReservation,
            )
            request = self._read_contract_unlocked(
                stage_result.execution_request_id,
                ExpertSourceReplayExecutionRequest,
            )
            historical_reservation = self._source_replay_reservation_unlocked(
                journal,
                stage_result.authorization_transition_id,
            )
            if historical_reservation != (reservation, request):
                raise ExpertValidationStoreError(
                    "release matrix source evidence reservation closure changed"
                )
            return ExpertReleaseMatrixSourceEvidenceSnapshot(
                plan_reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                    operation=stored[0],
                    evaluation_plan=stored[1],
                    snapshot=current,
                ),
                stage_result=stage_result,
                reservation=reservation,
                request=request,
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
            source_base=prepared_request.source_base,
            authorization_state=prepared_request.authorization_state,
            recovery_admission=prepared_request.recovery_admission,
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
            source_base=prepared_request.source_base,
            authorization_state=prepared_request.authorization_state,
            recovery_admission=prepared_request.recovery_admission,
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

    def publish_current_release_authority_invalidation(
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
            replayed = self._current_release_authority_invalidation_snapshot_unlocked(
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
        reduced = self.reducer.invalidate_current_release_authority(
            state=observed.state,
            attempt=observed.latest_attempt,
        )
        with self._lock(exclusive=True):
            journal = self._read_journal_unlocked(candidate_id)
            self._validate_journal_unlocked(journal)
            replayed = self._current_release_authority_invalidation_snapshot_unlocked(
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
                transition_release_use_block_decision_id=None,
                transition_release_activation_receipt_id=None,
                transition_release_revocation_receipt_id=None,
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

    def _current_release_authority_invalidation_snapshot_unlocked(
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
            transition_release_use_block_decision_id=None,
            transition_release_activation_receipt_id=None,
            transition_release_revocation_receipt_id=None,
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
            source_base=prepared_request.source_base,
            authorization_state=prepared_request.authorization_state,
            recovery_admission=prepared_request.recovery_admission,
            cases=prepared_request.cases,
        )
        request = prepared.request
        if (
            reservation.execution_request_id != request.execution_request_id
            or reservation.validation_attempt_id != request.validation_attempt_id
            or reservation.authorization_state_id != request.authorization_state_id
            or reservation.candidate_id != request.candidate_id
            or reservation.candidate_tree_hash != request.candidate_tree_hash
            or reservation.expected_current_release_id
            != request.expected_current_release_id
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

    @staticmethod
    def _automated_review_operation(
        packet: ExpertAutomatedReviewPacket,
    ) -> ExpertValidationOperation:
        return ExpertValidationOperation.mint(
            operation_kind=(
                ExpertValidationOperationKind.AUTOMATED_REVIEW_STAGE_RESULT
            ),
            candidate_id=packet.candidate_id,
            expected_transition_id=packet.authorization_transition_id,
            request_record_id=packet.review_packet_id,
        )

    @staticmethod
    def _release_matrix_stage_operation(
        reservation: TaskEvaluationReservation,
    ) -> ExpertValidationOperation:
        return ExpertValidationOperation.mint(
            operation_kind=(ExpertValidationOperationKind.RELEASE_MATRIX_STAGE_RESULT),
            candidate_id=reservation.candidate_id,
            expected_transition_id=reservation.authorization_transition_id,
            request_record_id=reservation.reservation_id,
        )

    @staticmethod
    def _publication_eligibility_operation(
        input_snapshot: ExpertPublicationEligibilitySnapshot,
        result: ExpertPublicationEligibilityStageResultRecord,
    ) -> ExpertValidationOperation:
        return ExpertValidationOperation.mint(
            operation_kind=(
                ExpertValidationOperationKind.PUBLICATION_ELIGIBILITY_STAGE_RESULT
            ),
            candidate_id=input_snapshot.snapshot.state.candidate_id,
            expected_transition_id=(input_snapshot.snapshot.transition.transition_id),
            request_record_id=result.stage_result_record_id,
        )

    def _validate_publication_eligibility_execution(
        self,
        *,
        execution: ExpertPublicationEligibilityExecution,
        current: ExpertValidationSnapshot,
        stored_candidate: StoredExpertCandidate,
    ) -> None:
        input_snapshot = execution.input_snapshot
        attempt = current.latest_attempt
        if attempt is None:
            raise ExpertValidationStoreError(
                "publication eligibility execution has no validation attempt"
            )
        expected_input = ExpertPublicationEligibilitySnapshot(
            snapshot=current,
            release_matrix_stage_result=(input_snapshot.release_matrix_stage_result),
        )
        expected_decision = decide_expert_release_matrix_promotion(
            stage_result=input_snapshot.release_matrix_stage_result,
            attempt=attempt,
            settings=self.settings,
        )
        if (
            input_snapshot != expected_input
            or execution.stored_candidate != stored_candidate
            or execution.decision != expected_decision
        ):
            raise ExpertValidationStoreError(
                "publication eligibility execution differs from stored authority"
            )
        release_use = execution.stage_result.release_use_decision
        fence = execution.stage_result.publication_authority_fence
        if expected_decision.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED:
            if release_use is None:
                raise ExpertValidationStoreError(
                    "approved publication eligibility lacks release-use authority"
                )
            if (
                release_use.policy_observation.checked_release_ids
                != stored_candidate.closure.manifest.consumed_expert_release_ids
            ):
                raise ExpertValidationStoreError(
                    "publication eligibility checked another release-use closure"
                )
            if release_use.outcome is ExpertCandidateReleaseUseOutcome.CLEARED:
                if fence is None:
                    raise ExpertValidationStoreError(
                        "cleared publication eligibility lacks fresh authority"
                    )
                expected_security_subject_ids = (
                    publication_eligibility_security_subject_ids(
                        input_snapshot=expected_input,
                        stored_candidate=stored_candidate,
                        decision=expected_decision,
                        current_release_observation=(fence.current_release_observation),
                        task_adapter_trust_observations=(
                            fence.task_adapter_trust_observations
                        ),
                    )
                )
                if fence.security_subject_ids != expected_security_subject_ids:
                    raise ExpertValidationStoreError(
                        "publication eligibility security closure is not exact"
                    )
            elif fence is not None:
                raise ExpertValidationStoreError(
                    "blocked publication eligibility cannot carry fresh authority"
                )
        elif release_use is not None or fence is not None:
            raise ExpertValidationStoreError(
                "non-approved publication eligibility cannot carry external authority"
            )
        expected_result = build_publication_eligibility_stage_result(
            input_snapshot=expected_input,
            stored_candidate=stored_candidate,
            decision=expected_decision,
            release_use_decision=release_use,
            publication_authority_fence=fence,
        )
        if execution.stage_result != expected_result:
            raise ExpertValidationStoreError(
                "publication eligibility result differs from sealed reduction"
            )

    def _validate_automated_review_execution(
        self,
        execution: ExpertAutomatedReviewExecution,
        observed: ExpertValidationSnapshot,
    ) -> tuple[
        PreparedExpertAutomatedReviewPacket,
        ExpertAutomatedReviewStageResultRecord,
        ExpertCandidateValidationState,
    ]:
        if observed.latest_attempt is None:
            raise ExpertValidationStoreError(
                "automated review execution has no active attempt"
            )
        supplied = execution.prepared_packet
        prepared = PreparedExpertAutomatedReviewPacket(
            packet=supplied.packet,
            candidate_input=supplied.candidate_input,
            candidate_derivation_record=supplied.candidate_derivation_record,
            candidate_operation=supplied.candidate_operation,
            composition_materialization=supplied.composition_materialization,
            recovery_replay_basis=supplied.recovery_replay_basis,
            validation_attempt=supplied.validation_attempt,
            authorization_state=supplied.authorization_state,
            validation_policy=supplied.validation_policy,
            accepted_stage_results=supplied.accepted_stage_results,
        )
        packet = prepared.packet
        policy = self.settings.policy.validation_policy()
        if (
            prepared != supplied
            or prepared.validation_attempt != observed.latest_attempt
            or prepared.authorization_state != observed.state
            or prepared.accepted_stage_results != observed.accepted_stage_results
            or prepared.validation_policy != policy
            or packet.authorization_transition_id != observed.transition.transition_id
        ):
            raise ExpertValidationStoreError(
                "automated review execution differs from the validation head"
            )
        adjudication = adjudicate_expert_automated_review(
            packet=packet,
            validation_policy=policy,
            assertions=execution.assertions,
            operation_records=execution.operation_records,
        )
        if adjudication != execution.adjudication:
            raise ExpertValidationStoreError(
                "automated review adjudication is not deterministic"
            )
        result = build_expert_automated_review_stage_result(
            prepared=prepared,
            assertions=execution.assertions,
            operation_records=execution.operation_records,
            adjudication=adjudication,
        )
        if result != execution.stage_result:
            raise ExpertValidationStoreError(
                "automated review stage result is not deterministic"
            )
        validate_expert_automated_review_facts(
            prepared=prepared,
            assertions=execution.assertions,
            operation_records=execution.operation_records,
            adjudication=adjudication,
            stage_result=result,
        )
        target_state = self.reducer.advance_automated_review_stage(
            state=observed.state,
            attempt=observed.latest_attempt,
            accepted_results=observed.accepted_stage_results,
            result=result,
        )
        return prepared, result, target_state

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

    def _release_matrix_task_reservation_snapshot_unlocked(
        self,
        journal: ExpertValidationJournal,
        authorization_transition_id: str,
    ) -> ExpertTaskEvaluationReservationSnapshot:
        stored = self._task_evaluation_reservation_unlocked(
            journal,
            authorization_transition_id,
        )
        plan_alias = self._release_matrix_plan_reservation_unlocked(
            journal,
            authorization_transition_id,
        )
        if stored is None or plan_alias is None:
            raise ExpertValidationStoreError(
                "release matrix stage lacks its task and plan reservations"
            )
        operation, reservation, request, observation = stored
        snapshot = self._snapshot_at_unlocked(
            journal,
            authorization_transition_id,
        )
        return ExpertTaskEvaluationReservationSnapshot(
            operation=operation,
            reservation=reservation,
            request=request,
            current_release_observation=observation,
            plan_reservation=ExpertReleaseMatrixPlanReservationSnapshot(
                operation=plan_alias[0],
                evaluation_plan=plan_alias[1],
                snapshot=snapshot,
            ),
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

    def _automated_review_result_for_transition_unlocked(
        self,
        transition: ExpertValidationTransition,
    ) -> ExpertAutomatedReviewStageResultRecord:
        result_record_id = transition.transition_stage_result_record_id
        if (
            result_record_id is None
            or result_record_id.split(":sha256:", 1)[0]
            != "expert-automated-review-stage-result"
        ):
            raise ExpertValidationStoreError(
                "validation transition does not contain an automated review result"
            )
        return self._read_contract_unlocked(
            result_record_id,
            ExpertAutomatedReviewStageResultRecord,
        )

    def _release_matrix_stage_result_for_transition_unlocked(
        self,
        transition: ExpertValidationTransition,
    ) -> ExpertReleaseMatrixStageResultRecord:
        result_record_id = transition.transition_stage_result_record_id
        if (
            result_record_id is None
            or result_record_id.split(":sha256:", 1)[0]
            != "expert-release-matrix-stage-result"
        ):
            raise ExpertValidationStoreError(
                "validation transition does not contain a release matrix result"
            )
        return self._read_contract_unlocked(
            result_record_id,
            ExpertReleaseMatrixStageResultRecord,
        )

    def _publication_eligibility_result_for_transition_unlocked(
        self,
        transition: ExpertValidationTransition,
    ) -> ExpertPublicationEligibilityStageResultRecord:
        result_record_id = transition.transition_stage_result_record_id
        if (
            result_record_id is None
            or result_record_id.split(":sha256:", 1)[0]
            != "expert-publication-eligibility-stage-result"
        ):
            raise ExpertValidationStoreError(
                "validation transition does not contain publication eligibility"
            )
        return self._read_contract_unlocked(
            result_record_id,
            ExpertPublicationEligibilityStageResultRecord,
        )

    def _publication_eligibility_commit_unlocked(
        self,
        journal: ExpertValidationJournal,
        release_matrix_stage_result_id: str,
    ) -> ExpertPublicationEligibilityStageCommitResult | None:
        matches = []
        for transition_id in journal.transition_ids:
            transition = self._read_contract_unlocked(
                transition_id,
                ExpertValidationTransition,
            )
            result_record_id = transition.transition_stage_result_record_id
            if (
                result_record_id is None
                or result_record_id.split(":sha256:", 1)[0]
                != "expert-publication-eligibility-stage-result"
            ):
                continue
            result = self._publication_eligibility_result_for_transition_unlocked(
                transition
            )
            if (
                result.promotion_decision.release_matrix_stage_result_id
                == release_matrix_stage_result_id
            ):
                matches.append((transition, result))
        if len(matches) > 1:
            raise ExpertValidationStoreError(
                "release matrix has multiple publication eligibility successors"
            )
        if not matches:
            return None
        transition, result = matches[0]
        return ExpertPublicationEligibilityStageCommitResult(
            stage_result=result,
            snapshot=self._snapshot_at_unlocked(journal, transition.transition_id),
            replayed=True,
        )

    def _read_stage_result_unlocked(
        self,
        result_record_id: str,
    ) -> (
        ExpertEvaluatorResultRecord
        | ExpertSourceReplayStageResultRecord
        | ExpertAutomatedReviewStageResultRecord
        | ExpertReleaseMatrixStageResultRecord
        | ExpertPublicationEligibilityStageResultRecord
    ):
        namespace = result_record_id.split(":sha256:", 1)[0]
        if namespace == "expert-evaluator-result-record":
            result = self._read_contract_unlocked(
                result_record_id,
                ExpertEvaluatorResultRecord,
            )
            if result.evaluator_run.stage in {
                ExpertValidationStage.SOURCE_RUN_REPLAY,
                ExpertValidationStage.AUTOMATED_REVIEW,
                ExpertValidationStage.RELEASE_MATRIX,
                ExpertValidationStage.PUBLICATION_ELIGIBILITY,
            }:
                raise ExpertValidationStoreError(
                    "typed validation stage cannot use a generic evaluator result"
                )
            return result
        if namespace == "expert-source-replay-stage-result":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertSourceReplayStageResultRecord,
            )
        if namespace == "expert-automated-review-stage-result":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertAutomatedReviewStageResultRecord,
            )
        if namespace == "expert-release-matrix-stage-result":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertReleaseMatrixStageResultRecord,
            )
        if namespace == "expert-publication-eligibility-stage-result":
            return self._read_contract_unlocked(
                result_record_id,
                ExpertPublicationEligibilityStageResultRecord,
            )
        raise ExpertValidationStoreError(
            "validation stage result uses an unsupported namespace"
        )

    @staticmethod
    def _stage_result_projection(
        result: (
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord
            | ExpertReleaseMatrixStageResultRecord
            | ExpertPublicationEligibilityStageResultRecord
        ),
    ) -> tuple[
        ExpertValidationStage,
        str,
        bool,
        str,
        str,
        str,
    ]:
        if type(result) is ExpertEvaluatorResultRecord:
            run = result.evaluator_run
            if run.stage in {
                ExpertValidationStage.SOURCE_RUN_REPLAY,
                ExpertValidationStage.AUTOMATED_REVIEW,
                ExpertValidationStage.RELEASE_MATRIX,
                ExpertValidationStage.PUBLICATION_ELIGIBILITY,
            }:
                raise ExpertValidationStoreError(
                    "typed validation stage cannot use a generic evaluator result"
                )
            return (
                run.stage,
                result.evaluator_result_record_id,
                run.outcome is ExpertEvaluatorOutcome.PASSED,
                run.validation_attempt_id,
                run.candidate_id,
                run.candidate_tree_hash,
            )
        if type(result) is ExpertSourceReplayStageResultRecord:
            return (
                ExpertValidationStage.SOURCE_RUN_REPLAY,
                result.stage_result_record_id,
                result.outcome is ExpertEvaluatorOutcome.PASSED,
                result.validation_attempt_id,
                result.candidate_id,
                result.candidate_tree_hash,
            )
        if type(result) is ExpertAutomatedReviewStageResultRecord:
            return (
                ExpertValidationStage.AUTOMATED_REVIEW,
                result.stage_result_record_id,
                result.outcome is ExpertAutomatedReviewOutcome.PASSED,
                result.validation_attempt_id,
                result.candidate_id,
                result.candidate_tree_hash,
            )
        if type(result) is ExpertReleaseMatrixStageResultRecord:
            return (
                ExpertValidationStage.RELEASE_MATRIX,
                result.stage_result_record_id,
                True,
                result.validation_attempt_id,
                result.candidate_id,
                result.candidate_tree_hash,
            )
        if type(result) is ExpertPublicationEligibilityStageResultRecord:
            return (
                ExpertValidationStage.PUBLICATION_ELIGIBILITY,
                result.stage_result_record_id,
                result.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED,
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
            or transition.transition_release_use_block_decision_id is not None
            or transition.transition_release_activation_receipt_id is not None
            or transition.transition_release_revocation_receipt_id is not None
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
        source_replay_reserved_transition_ids: set[str] = set()
        release_matrix_reserved_transition_ids: set[str] = set()
        task_evaluation_reserved_transition_ids: set[str] = set()
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
                if transition_id in source_replay_reserved_transition_ids:
                    raise ExpertValidationStoreError(
                        "validation transition has multiple source replay reservations"
                    )
                self._validate_source_replay_reservation_alias_unlocked(
                    operation,
                    transition,
                )
                source_replay_reserved_transition_ids.add(transition_id)
                continue
            if (
                operation.operation_kind
                is ExpertValidationOperationKind.RELEASE_MATRIX_PLAN_RESERVATION
            ):
                if transition_id in release_matrix_reserved_transition_ids:
                    raise ExpertValidationStoreError(
                        "validation transition has multiple release matrix plans"
                    )
                self._validate_release_matrix_plan_alias_unlocked(
                    operation,
                    transition,
                )
                release_matrix_reserved_transition_ids.add(transition_id)
                continue
            if (
                operation.operation_kind
                is ExpertValidationOperationKind.TASK_EVALUATION_RESERVATION
            ):
                if transition_id in task_evaluation_reserved_transition_ids:
                    raise ExpertValidationStoreError(
                        "validation transition has multiple task evaluations"
                    )
                self._validate_task_evaluation_reservation_alias_unlocked(
                    journal,
                    operation,
                    transition,
                )
                task_evaluation_reserved_transition_ids.add(transition_id)
                continue
            raise ExpertValidationStoreError(
                "validation replay operation does not bind its transition"
            )
        if journal.release_publication_intent_id is not None:
            if (
                previous_transition is None
                or previous_state is None
                or previous_latest_attempt is None
            ):
                raise ExpertValidationStoreError(
                    "release publication intent lacks a validation head"
                )
            intent = self._read_contract_unlocked(
                journal.release_publication_intent_id,
                ExpertReleasePublicationIntent,
            )
            plan = self._read_contract_unlocked(
                intent.publication_plan_id,
                ExpertReleasePublicationPlan,
            )
            manifest = self._read_contract_unlocked(
                plan.release_id,
                ExpertBaseReleaseManifest,
            )
            snapshot = self._snapshot_at_unlocked(
                journal,
                plan.approval_transition_id,
            )
            self._validate_release_publication_reservation_unlocked(
                intent,
                plan,
                manifest,
                snapshot,
            )
            if previous_state.promotion_state not in {
                ExpertPromotionState.APPROVED,
                ExpertPromotionState.RELEASE_USE_BLOCKED,
            }:
                raise ExpertValidationStoreError(
                    "release publication intent is not active or recoverable"
                )
        if journal.release_publication_stale_resolution_id is not None:
            resolution = self._read_contract_unlocked(
                journal.release_publication_stale_resolution_id,
                ExpertReleasePublicationStaleResolution,
            )
            intent = self._read_contract_unlocked(
                resolution.publication_intent_id,
                ExpertReleasePublicationIntent,
            )
            plan = self._read_contract_unlocked(
                intent.publication_plan_id,
                ExpertReleasePublicationPlan,
            )
            manifest = self._read_contract_unlocked(
                plan.release_id,
                ExpertBaseReleaseManifest,
            )
            if (
                resolution.publication_plan_id != plan.publication_plan_id
                or resolution.release_id != manifest.release_id
                or resolution.candidate_id != journal.candidate_id
                or resolution.candidate_id != manifest.candidate_id
                or resolution.approval_transition_id not in journal.transition_ids
                or resolution.approval_transition_id != manifest.approval_transition_id
                or resolution.approval_state_id != manifest.approval_state_id
                or resolution.planned_current_observation_id
                != plan.current_release_observation.observation_id
                or resolution.observed_current_release.scope_id != plan.scope_id
                or resolution.observed_current_release.repository_full_name
                != plan.current_release_observation.repository_full_name
                or resolution.observed_current_release.repository_node_id
                != plan.current_release_observation.repository_node_id
            ):
                raise ExpertValidationStoreError(
                    "stale release publication resolution closure is inconsistent"
                )
            self._validate_release_publication_remote_history(
                intent,
                plan,
                resolution.own_github_publication_intent,
                resolution.own_github_publication_pointer,
                None,
            )

    @staticmethod
    def _validate_release_publication_reservation_unlocked(
        intent: ExpertReleasePublicationIntent,
        plan: ExpertReleasePublicationPlan,
        manifest: ExpertBaseReleaseManifest,
        snapshot: ExpertValidationSnapshot,
    ) -> None:
        attempt = snapshot.latest_attempt
        publication_result = (
            None
            if not snapshot.accepted_stage_results
            else snapshot.accepted_stage_results[-1]
        )
        if (
            intent.publication_plan_id != plan.publication_plan_id
            or plan.release_id != manifest.release_id
            or plan.scope_contract_id != manifest.scope_contract_id
            or plan.scope_id != manifest.scope_id
            or plan.candidate_id != manifest.candidate_id
            or plan.candidate_tree_hash != manifest.candidate_tree_hash
            or plan.validation_attempt_id != manifest.validation_attempt_id
            or plan.approval_transition_id != manifest.approval_transition_id
            or plan.approval_state_id != manifest.approval_state_id
            or plan.publication_eligibility_result_id
            != manifest.publication_eligibility_result_id
            or plan.lineage != manifest.lineage
            or plan.manifest_digest != tree_or_blob_digest(manifest.to_json_bytes())
            or plan.manifest_consumed_dependency_ids != manifest.consumed_dependency_ids
            or plan.manifest_control_dependency_ids != manifest.control_dependency_ids
            or attempt is None
            or type(publication_result)
            is not ExpertPublicationEligibilityStageResultRecord
            or publication_result.promotion_decision.outcome
            is not ExpertReleaseMatrixDecisionOutcome.APPROVED
            or publication_result.publication_authority_fence is None
            or snapshot.state.promotion_state is not ExpertPromotionState.APPROVED
            or snapshot.state.next_stage is not None
            or plan.candidate_id != snapshot.state.candidate_id
            or plan.candidate_tree_hash != snapshot.state.candidate_tree_hash
            or plan.validation_attempt_id != attempt.validation_attempt_id
            or plan.approval_transition_id != snapshot.transition.transition_id
            or plan.approval_state_id != snapshot.state.validation_state_id
            or plan.publication_eligibility_result_id
            != publication_result.stage_result_record_id
            or plan.scope_contract_id != attempt.scope_contract_id
            or plan.scope_contract_id != publication_result.scope_contract_id
            or plan.scope_id != publication_result.scope_id
            or plan.lineage.source_base_release_id != attempt.source_base_release_id
            or plan.lineage.activation_predecessor_release_id
            != publication_result.expected_current_release_id
            or plan.current_release_observation
            != publication_result.publication_authority_fence.current_release_observation
            or publication_result.validation_attempt_id != attempt.validation_attempt_id
            or publication_result.candidate_id != plan.candidate_id
            or publication_result.candidate_tree_hash != plan.candidate_tree_hash
            or publication_result.validation_policy_id != attempt.validation_policy_id
            or publication_result.configuration_fingerprint
            != attempt.configuration_fingerprint
        ):
            raise ExpertValidationStoreError(
                "release publication intent differs from terminal approval authority"
            )

    def _validate_release_matrix_plan_alias_unlocked(
        self,
        operation: ExpertValidationOperation,
        transition: ExpertValidationTransition,
    ) -> None:
        plan = self._read_contract_unlocked(
            operation.request_record_id,
            ExpertReleaseMatrixEvaluationPlan,
        )
        state = self._read_contract_unlocked(
            transition.target_state_id,
            ExpertCandidateValidationState,
        )
        if transition.latest_attempt_id is None:
            raise ExpertValidationStoreError(
                "release matrix plan reservation requires a validation attempt"
            )
        attempt = self._read_contract_unlocked(
            transition.latest_attempt_id,
            ExpertValidationAttempt,
        )
        persisted_settings = self._read_configuration_unlocked(
            transition.configuration_fingerprint
        )
        validation_policy = self._read_contract_unlocked(
            transition.validation_policy_id,
            ExpertValidationPolicy,
        )
        accepted_results = tuple(
            self._read_stage_result_unlocked(result_record_id)
            for result_record_id in transition.accepted_stage_result_record_ids
        )
        source_replay_request = self._source_replay_request_for_results_unlocked(
            accepted_results
        )
        validate_expert_release_matrix_plan_store_shape(
            plan=plan,
            state=state,
            attempt=attempt,
            accepted_stage_results=accepted_results,
            source_replay_request=source_replay_request,
            validation_policy=validation_policy,
            validation_settings=persisted_settings,
        )
        if (
            operation.expected_transition_id != transition.transition_id
            or operation.candidate_id != transition.candidate_id
            or operation.request_record_id != plan.evaluation_plan_id
        ):
            raise ExpertValidationStoreError(
                "release matrix plan reservation alias closure is inconsistent"
            )

    def _source_replay_request_for_results_unlocked(
        self,
        accepted_results: tuple[
            ExpertEvaluatorResultRecord
            | ExpertSourceReplayStageResultRecord
            | ExpertAutomatedReviewStageResultRecord
            | ExpertReleaseMatrixStageResultRecord,
            ...,
        ],
    ) -> ExpertSourceReplayExecutionRequest | None:
        source_results = tuple(
            result
            for result in accepted_results
            if type(result) is ExpertSourceReplayStageResultRecord
        )
        if len(source_results) > 1:
            raise ExpertValidationStoreError(
                "release matrix plan accepts at most one source replay result"
            )
        if not source_results:
            return None
        return self._read_contract_unlocked(
            source_results[0].execution_request_id,
            ExpertSourceReplayExecutionRequest,
        )

    def _validate_task_evaluation_reservation_alias_unlocked(
        self,
        journal: ExpertValidationJournal,
        operation: ExpertValidationOperation,
        transition: ExpertValidationTransition,
    ) -> None:
        reservation = self._read_contract_unlocked(
            operation.request_record_id,
            TaskEvaluationReservation,
        )
        request = self._read_contract_unlocked(
            reservation.request_id,
            TaskEvaluationRequest,
        )
        observation = self._read_contract_unlocked(
            reservation.current_release_observation_id,
            TaskEvaluationCurrentReleaseObservation,
        )
        plan_alias = self._release_matrix_plan_reservation_unlocked(
            journal,
            transition.transition_id,
        )
        if plan_alias is None:
            raise ExpertValidationStoreError(
                "task evaluation reservation lacks its release matrix plan"
            )
        state = self._read_contract_unlocked(
            transition.target_state_id,
            ExpertCandidateValidationState,
        )
        if transition.latest_attempt_id is None:
            raise ExpertValidationStoreError(
                "task evaluation reservation requires a validation attempt"
            )
        attempt = self._read_contract_unlocked(
            transition.latest_attempt_id,
            ExpertValidationAttempt,
        )
        accepted_results = tuple(
            self._read_stage_result_unlocked(result_record_id)
            for result_record_id in transition.accepted_stage_result_record_ids
        )
        plan_reservation = ExpertReleaseMatrixPlanReservationSnapshot(
            operation=plan_alias[0],
            evaluation_plan=plan_alias[1],
            snapshot=ExpertValidationSnapshot(
                transition=transition,
                state=state,
                latest_attempt=attempt,
                accepted_stage_results=accepted_results,
            ),
        )
        persisted_settings = self._read_configuration_unlocked(
            transition.configuration_fingerprint
        )
        PlanJoinedTaskEvaluationRequest(
            request=request,
            plan_reservation=plan_reservation,
            settings=persisted_settings,
        )
        self._validate_recovery_security_authority_unlocked(
            candidate_id=request.candidate_id,
            candidate_commit_record_id=request.candidate_commit_record_id,
            source_base_release_id=request.source_base_release_id,
            expected_current_release_id=request.expected_current_release_id,
            recovery_plan_id=request.recovery_plan_id,
            control_dependency_ids=request.control_dependency_ids,
            allowed_control_security_subject_ids=(
                request.allowed_control_security_subject_ids
            ),
        )
        ExpertTaskEvaluationReservationSnapshot(
            operation=operation,
            reservation=reservation,
            request=request,
            current_release_observation=observation,
            plan_reservation=plan_reservation,
        )

    def _release_matrix_plan_reservation_unlocked(
        self,
        journal: ExpertValidationJournal,
        authorization_transition_id: str,
    ) -> tuple[ExpertValidationOperation, ExpertReleaseMatrixEvaluationPlan] | None:
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
                is not ExpertValidationOperationKind.RELEASE_MATRIX_PLAN_RESERVATION
            ):
                continue
            plan = self._read_contract_unlocked(
                operation.request_record_id,
                ExpertReleaseMatrixEvaluationPlan,
            )
            matches.append((operation, plan))
        if len(matches) > 1:
            raise ExpertValidationStoreError(
                "validation transition has multiple release matrix plans"
            )
        return None if not matches else matches[0]

    def _task_evaluation_reservation_unlocked(
        self,
        journal: ExpertValidationJournal,
        authorization_transition_id: str,
    ) -> (
        tuple[
            ExpertValidationOperation,
            TaskEvaluationReservation,
            TaskEvaluationRequest,
            TaskEvaluationCurrentReleaseObservation,
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
                is not ExpertValidationOperationKind.TASK_EVALUATION_RESERVATION
            ):
                continue
            reservation = self._read_contract_unlocked(
                operation.request_record_id,
                TaskEvaluationReservation,
            )
            request = self._read_contract_unlocked(
                reservation.request_id,
                TaskEvaluationRequest,
            )
            observation = self._read_contract_unlocked(
                reservation.current_release_observation_id,
                TaskEvaluationCurrentReleaseObservation,
            )
            matches.append((operation, reservation, request, observation))
        if len(matches) > 1:
            raise ExpertValidationStoreError(
                "validation transition has multiple task evaluation reservations"
            )
        return None if not matches else matches[0]

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
        self._validate_recovery_security_authority_unlocked(
            candidate_id=request.candidate_id,
            candidate_commit_record_id=request.candidate_commit_record_id,
            source_base_release_id=request.source_base_release_id,
            expected_current_release_id=request.expected_current_release_id,
            recovery_plan_id=request.recovery_plan_id,
            control_dependency_ids=request.control_dependency_ids,
            allowed_control_security_subject_ids=(
                request.allowed_control_security_subject_ids
            ),
        )
        if (
            operation.expected_transition_id != transition.transition_id
            or operation.candidate_id != transition.candidate_id
            or reservation.authorization_transition_id != transition.transition_id
            or reservation.validation_attempt_id != attempt.validation_attempt_id
            or reservation.authorization_state_id != state.validation_state_id
            or reservation.candidate_id != transition.candidate_id
            or reservation.candidate_tree_hash != transition.candidate_tree_hash
            or reservation.expected_current_release_id
            != request.expected_current_release_id
        ):
            raise ExpertValidationStoreError(
                "source replay reservation alias closure is inconsistent"
            )

    def _validate_recovery_security_authority_unlocked(
        self,
        *,
        candidate_id: str,
        candidate_commit_record_id: str,
        source_base_release_id: str | None,
        expected_current_release_id: str | None,
        recovery_plan_id: str | None,
        control_dependency_ids: tuple[str, ...],
        allowed_control_security_subject_ids: tuple[str, ...],
    ) -> None:
        recovery_admission_ids = tuple(
            dependency_id
            for dependency_id in control_dependency_ids
            if dependency_id.split(":sha256:", 1)[0]
            == "expert-recovery-candidate-admission"
        )
        if recovery_plan_id is None:
            if recovery_admission_ids or allowed_control_security_subject_ids:
                raise ExpertValidationStoreError(
                    "ordinary validation carries recovery security authority"
                )
            return
        if len(recovery_admission_ids) != 1:
            raise ExpertValidationStoreError(
                "recovery validation requires one durable candidate admission"
            )
        recovery_admission = self._read_contract_unlocked(
            recovery_admission_ids[0],
            ExpertRecoveryCandidateAdmission,
        )
        if (
            recovery_admission.candidate_id != candidate_id
            or recovery_admission.candidate_commit_record_id
            != candidate_commit_record_id
            or recovery_admission.recovery_plan.recovery_plan_id != recovery_plan_id
            or recovery_admission.recovery_plan.source_base_release_id
            != source_base_release_id
            or recovery_admission.recovery_plan.activation_predecessor_release_id
            != expected_current_release_id
            or recovery_admission.control_dependency_ids != control_dependency_ids
            or recovery_admission.allowed_control_security_subject_ids
            != allowed_control_security_subject_ids
        ):
            raise ExpertValidationStoreError(
                "recovery security authority differs from durable candidate admission"
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
            or fence.expected_current_release_id != request.expected_current_release_id
            or fence.allowed_control_security_subject_ids
            != request.allowed_control_security_subject_ids
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

    def _write_automated_review_derivation_unlocked(
        self,
        prepared: PreparedExpertAutomatedReviewPacket,
    ) -> None:
        self._write_contract_unlocked(prepared.candidate_derivation_record)
        if prepared.candidate_operation is not None:
            self._write_contract_unlocked(prepared.candidate_operation)
        if prepared.composition_materialization is not None:
            self._write_contract_unlocked(prepared.composition_materialization)
        if prepared.recovery_replay_basis is not None:
            self._write_contract_unlocked(prepared.recovery_replay_basis)

    def _read_automated_review_derivation_unlocked(
        self,
        packet: ExpertAutomatedReviewPacket,
    ) -> tuple[
        ExpertAgentProposalDerivationRecord
        | ExpertDeterministicCompositionDerivationRecord
        | ExpertDeterministicRecoveryRestoreDerivationRecord,
        ExpertCandidateOperationRecord | None,
        ExpertCompositionMaterialization | None,
        ExpertTriggerEvidencePacket | None,
    ]:
        if packet.candidate_derivation_kind in {
            ExpertCandidateDerivationKind.AGENT_PROPOSAL,
            ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP,
        }:
            derivation_record = self._read_contract_unlocked(
                packet.candidate_derivation_ref,
                ExpertAgentProposalDerivationRecord,
            )
            operation = self._read_contract_unlocked(
                derivation_record.operation_record_id,
                ExpertCandidateOperationRecord,
            )
            return derivation_record, operation, None, None
        if packet.candidate_derivation_kind is (
            ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
        ):
            derivation_record = self._read_contract_unlocked(
                packet.candidate_derivation_ref,
                ExpertDeterministicCompositionDerivationRecord,
            )
            materialization = self._read_contract_unlocked(
                derivation_record.composition_materialization_id,
                ExpertCompositionMaterialization,
            )
            return derivation_record, None, materialization, None
        if packet.candidate_derivation_kind is (
            ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE
        ):
            derivation_record = self._read_contract_unlocked(
                packet.candidate_derivation_ref,
                ExpertDeterministicRecoveryRestoreDerivationRecord,
            )
            replay_basis = self._read_contract_unlocked(
                derivation_record.replay_basis_packet_id,
                ExpertTriggerEvidencePacket,
            )
            return derivation_record, None, None, replay_basis
        raise ExpertValidationStoreError(
            "automated review packet uses an unknown candidate derivation"
        )

    def _validate_automated_review_transition_unlocked(
        self,
        *,
        transition: ExpertValidationTransition,
        state: ExpertCandidateValidationState,
        operation: ExpertValidationOperation,
        latest_attempt: ExpertValidationAttempt | None,
        previous_accepted: tuple[str, ...],
        result_record: ExpertAutomatedReviewStageResultRecord,
        validation_policy: ExpertValidationPolicy,
    ) -> None:
        if transition.predecessor_state_id is None or latest_attempt is None:
            raise ExpertValidationStoreError(
                "automated review result requires its active predecessor"
            )
        predecessor_state = self._read_contract_unlocked(
            transition.predecessor_state_id,
            ExpertCandidateValidationState,
        )
        if transition.predecessor_transition_id is None:
            raise ExpertValidationStoreError(
                "automated review result requires its authorization transition"
            )
        predecessor_transition = self._read_contract_unlocked(
            transition.predecessor_transition_id,
            ExpertValidationTransition,
        )
        packet = self._read_contract_unlocked(
            result_record.review_packet_id,
            ExpertAutomatedReviewPacket,
        )
        candidate_input = self._read_contract_unlocked(
            packet.candidate_input_id,
            ExpertCandidateAncestorInput,
        )
        (
            candidate_derivation_record,
            candidate_operation,
            composition_materialization,
            recovery_replay_basis,
        ) = self._read_automated_review_derivation_unlocked(packet)
        accepted_results = tuple(
            self._read_stage_result_unlocked(result_id)
            for result_id in previous_accepted
        )
        prepared = PreparedExpertAutomatedReviewPacket(
            packet=packet,
            candidate_input=candidate_input,
            candidate_derivation_record=candidate_derivation_record,
            candidate_operation=candidate_operation,
            composition_materialization=composition_materialization,
            recovery_replay_basis=recovery_replay_basis,
            validation_attempt=latest_attempt,
            authorization_state=predecessor_state,
            validation_policy=validation_policy,
            accepted_stage_results=accepted_results,
        )
        assertions = tuple(
            sorted(
                (
                    self._read_contract_unlocked(
                        assertion_id,
                        ExpertAutomatedReviewAssertion,
                    )
                    for assertion_id in result_record.assertion_ids
                ),
                key=lambda assertion: assertion.reviewer_id,
            )
        )
        review_operations = tuple(
            sorted(
                (
                    self._read_contract_unlocked(
                        operation_record_id,
                        ExpertAutomatedReviewOperationRecord,
                    )
                    for operation_record_id in result_record.operation_record_ids
                ),
                key=lambda review_operation: (
                    review_operation.operation_receipt.principal_id
                ),
            )
        )
        stored_receipts = tuple(
            self._read_contract_unlocked(
                receipt_id,
                CodingAgentOperationReceipt,
            )
            for receipt_id in result_record.operation_receipt_ids
        )
        adjudication = self._read_contract_unlocked(
            result_record.adjudication_id,
            ExpertAutomatedReviewAdjudication,
        )
        expected_adjudication = adjudicate_expert_automated_review(
            packet=packet,
            validation_policy=validation_policy,
            assertions=assertions,
            operation_records=review_operations,
        )
        expected_result = build_expert_automated_review_stage_result(
            prepared=prepared,
            assertions=assertions,
            operation_records=review_operations,
            adjudication=expected_adjudication,
        )
        validate_expert_automated_review_facts(
            prepared=prepared,
            assertions=assertions,
            operation_records=review_operations,
            adjudication=adjudication,
            stage_result=result_record,
        )
        common_invalid = (
            operation.operation_kind
            is not ExpertValidationOperationKind.AUTOMATED_REVIEW_STAGE_RESULT
            or operation.request_record_id != packet.review_packet_id
            or operation.expected_transition_id != packet.authorization_transition_id
            or transition.predecessor_transition_id
            != packet.authorization_transition_id
            or packet.authorization_state_id != predecessor_state.validation_state_id
            or packet.validation_attempt_id != latest_attempt.validation_attempt_id
            or packet.candidate_id != transition.candidate_id
            or packet.candidate_tree_hash != transition.candidate_tree_hash
            or packet.scope_contract_id != latest_attempt.scope_contract_id
            or packet.source_base_release_id != latest_attempt.source_base_release_id
            or packet.validation_policy_id != transition.validation_policy_id
            or packet.configuration_fingerprint != transition.configuration_fingerprint
            or predecessor_state.promotion_state is not ExpertPromotionState.VALIDATING
            or predecessor_state.next_stage
            is not ExpertValidationStage.AUTOMATED_REVIEW
            or tuple(
                item.stage_result_record_id
                for item in predecessor_state.accepted_stage_results
            )
            != previous_accepted
            or predecessor_transition.accepted_stage_result_record_ids
            != previous_accepted
            or result_record != expected_result
            or adjudication != expected_adjudication
            or tuple(
                sorted(
                    stored_receipts,
                    key=lambda receipt: receipt.operation_receipt_id,
                )
            )
            != tuple(
                sorted(
                    (
                        review_operation.operation_receipt
                        for review_operation in review_operations
                    ),
                    key=lambda receipt: receipt.operation_receipt_id,
                )
            )
            or state.review_assertion_ids != result_record.assertion_ids
            or state.transition_evidence_id != result_record.stage_result_record_id
        )
        if common_invalid:
            raise ExpertValidationStoreError(
                "automated review transition closure is inconsistent"
            )
        if result_record.outcome is ExpertAutomatedReviewOutcome.PASSED:
            valid_state = (
                state.promotion_state is ExpertPromotionState.VALIDATING
                and state.next_stage is ExpertValidationStage.RELEASE_MATRIX
                and not state.terminal_evidence_ids
                and state.reason == "stage_automated_review_passed"
                and transition.accepted_stage_result_record_ids
                == (*previous_accepted, result_record.stage_result_record_id)
            )
        elif result_record.outcome is ExpertAutomatedReviewOutcome.REJECTED:
            valid_state = (
                state.promotion_state is ExpertPromotionState.FAILED
                and state.next_stage is None
                and state.terminal_evidence_ids
                == (result_record.stage_result_record_id,)
                and state.reason == "stage_automated_review_rejected"
                and transition.accepted_stage_result_record_ids == previous_accepted
            )
        else:
            valid_state = (
                result_record.outcome is ExpertAutomatedReviewOutcome.DISPUTED
                and len(result_record.assertion_ids) >= 2
                and state.promotion_state is ExpertPromotionState.DISPUTED
                and state.next_stage is None
                and state.terminal_evidence_ids
                == (result_record.stage_result_record_id,)
                and state.reason == "stage_automated_review_disputed"
                and transition.accepted_stage_result_record_ids == previous_accepted
            )
        if not valid_state:
            raise ExpertValidationStoreError(
                "automated review state semantics are inconsistent"
            )

    def _validate_release_matrix_stage_transition_unlocked(
        self,
        *,
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
        state: ExpertCandidateValidationState,
        operation: ExpertValidationOperation,
        latest_attempt: ExpertValidationAttempt | None,
        previous_accepted: tuple[str, ...],
        result_record: ExpertReleaseMatrixStageResultRecord,
    ) -> None:
        if (
            transition.predecessor_transition_id is None
            or transition.predecessor_state_id is None
            or latest_attempt is None
        ):
            raise ExpertValidationStoreError(
                "release matrix result requires its active predecessor"
            )
        predecessor_transition = self._read_contract_unlocked(
            transition.predecessor_transition_id,
            ExpertValidationTransition,
        )
        predecessor_state = self._read_contract_unlocked(
            transition.predecessor_state_id,
            ExpertCandidateValidationState,
        )
        report = self._read_contract_unlocked(
            result_record.release_matrix_report.release_matrix_report_id,
            ExpertReleaseMatrixReport,
        )
        task_alias = self._task_evaluation_reservation_unlocked(
            journal,
            result_record.authorization_transition_id,
        )
        plan_alias = self._release_matrix_plan_reservation_unlocked(
            journal,
            result_record.authorization_transition_id,
        )
        if task_alias is None or plan_alias is None:
            raise ExpertValidationStoreError(
                "release matrix result lacks its durable reservations"
            )
        _reservation_operation, reservation, request, _observation = task_alias
        common_invalid = (
            operation.operation_kind
            is not ExpertValidationOperationKind.RELEASE_MATRIX_STAGE_RESULT
            or operation.request_record_id != reservation.reservation_id
            or operation.expected_transition_id
            != reservation.authorization_transition_id
            or transition.predecessor_transition_id
            != reservation.authorization_transition_id
            or predecessor_transition.accepted_stage_result_record_ids
            != previous_accepted
            or predecessor_state.promotion_state is not ExpertPromotionState.VALIDATING
            or predecessor_state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
            or predecessor_state.validation_attempt_id
            != latest_attempt.validation_attempt_id
            or tuple(
                item.stage_result_record_id
                for item in predecessor_state.accepted_stage_results
            )
            != previous_accepted
            or result_record.authorization_transition_id
            != transition.predecessor_transition_id
            or result_record.authorization_state_id
            != predecessor_state.validation_state_id
            or result_record.validation_attempt_id
            != latest_attempt.validation_attempt_id
            or result_record.candidate_id != transition.candidate_id
            or result_record.candidate_tree_hash != transition.candidate_tree_hash
            or result_record.scope_contract_id != latest_attempt.scope_contract_id
            or result_record.source_base_release_id
            != latest_attempt.source_base_release_id
            or result_record.validation_policy_id != latest_attempt.validation_policy_id
            or result_record.configuration_fingerprint
            != latest_attempt.configuration_fingerprint
            or report != result_record.release_matrix_report
            or report.candidate_commit_record_id
            != latest_attempt.candidate_commit_record_id
            or result_record.task_evaluation_reservation_id
            != reservation.reservation_id
            or result_record.plan_reservation_operation_id != plan_alias[0].operation_id
            or report.evaluation_plan != plan_alias[1]
            or request.authorization_transition_id
            != transition.predecessor_transition_id
            or request.authorization_state_id != predecessor_state.validation_state_id
            or request.validation_attempt_id != latest_attempt.validation_attempt_id
            or request.candidate_id != transition.candidate_id
            or request.candidate_tree_hash != transition.candidate_tree_hash
            or request.candidate_commit_record_id
            != latest_attempt.candidate_commit_record_id
            or request.scope_contract_id != latest_attempt.scope_contract_id
            or request.source_base_release_id != latest_attempt.source_base_release_id
            or request.validation_policy_id != latest_attempt.validation_policy_id
            or request.configuration_fingerprint
            != latest_attempt.configuration_fingerprint
            or state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.next_stage is not ExpertValidationStage.PUBLICATION_ELIGIBILITY
            or state.review_assertion_ids != predecessor_state.review_assertion_ids
            or state.terminal_evidence_ids
            or state.transition_evidence_id != result_record.stage_result_record_id
            or state.reason != "stage_release_matrix_passed"
        )
        if common_invalid:
            raise ExpertValidationStoreError(
                "release matrix result transition closure is inconsistent"
            )

    def _validate_publication_eligibility_transition_unlocked(
        self,
        *,
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
        state: ExpertCandidateValidationState,
        operation: ExpertValidationOperation,
        latest_attempt: ExpertValidationAttempt | None,
        previous_accepted: tuple[str, ...],
        result_record: ExpertPublicationEligibilityStageResultRecord,
        persisted_settings: ExpertValidationSettings,
    ) -> None:
        if (
            transition.predecessor_transition_id is None
            or transition.predecessor_state_id is None
            or latest_attempt is None
            or not previous_accepted
        ):
            raise ExpertValidationStoreError(
                "publication eligibility requires its accepted matrix predecessor"
            )
        predecessor_snapshot = self._snapshot_at_unlocked(
            journal,
            transition.predecessor_transition_id,
        )
        predecessor_state = predecessor_snapshot.state
        matrix_result = predecessor_snapshot.accepted_stage_results[-1]
        if type(matrix_result) is not ExpertReleaseMatrixStageResultRecord:
            raise ExpertValidationStoreError(
                "publication eligibility predecessor does not end in a matrix"
            )
        input_snapshot = ExpertPublicationEligibilitySnapshot(
            snapshot=predecessor_snapshot,
            release_matrix_stage_result=matrix_result,
        )
        decision = self._read_contract_unlocked(
            result_record.promotion_decision.promotion_decision_id,
            ExpertReleaseMatrixPromotionDecision,
        )
        expected_decision = decide_expert_release_matrix_promotion(
            stage_result=matrix_result,
            attempt=latest_attempt,
            settings=persisted_settings,
        )
        stored_candidate = self.reducer.candidate_store.read(
            latest_attempt.candidate_id
        )
        release_use = result_record.release_use_decision
        if release_use is not None:
            stored_release_use = self._read_contract_unlocked(
                release_use.release_use_decision_id,
                ExpertCandidateReleaseUseDecision,
            )
            if (
                stored_release_use != release_use
                or self._read_contract_unlocked(
                    release_use.policy_observation.observation_id,
                    type(release_use.policy_observation),
                )
                != release_use.policy_observation
                or release_use.policy_observation.checked_release_ids
                != stored_candidate.closure.manifest.consumed_expert_release_ids
            ):
                raise ExpertValidationStoreError(
                    "publication eligibility persisted release-use authority is invalid"
                )
            for revocation in release_use.policy_observation.matched_revocations:
                if (
                    self._read_contract_unlocked(
                        revocation.revocation_id,
                        type(revocation),
                    )
                    != revocation
                ):
                    raise ExpertValidationStoreError(
                        "publication eligibility release-use event is invalid"
                    )
        fence = result_record.publication_authority_fence
        if fence is not None:
            stored_fence = self._read_contract_unlocked(
                fence.fence_id,
                ExpertPublicationEligibilityAuthorityFence,
            )
            expected_security_subject_ids = (
                publication_eligibility_security_subject_ids(
                    input_snapshot=input_snapshot,
                    stored_candidate=stored_candidate,
                    decision=expected_decision,
                    current_release_observation=(
                        stored_fence.current_release_observation
                    ),
                    task_adapter_trust_observations=(
                        stored_fence.task_adapter_trust_observations
                    ),
                )
            )
            if (
                stored_fence != fence
                or stored_fence.security_subject_ids != expected_security_subject_ids
            ):
                raise ExpertValidationStoreError(
                    "publication eligibility persisted security closure is invalid"
                )
        expected_result = build_publication_eligibility_stage_result(
            input_snapshot=input_snapshot,
            stored_candidate=stored_candidate,
            decision=expected_decision,
            release_use_decision=release_use,
            publication_authority_fence=fence,
        )
        historical_reducer = ExpertValidationReducer(
            persisted_settings,
            self.reducer.candidate_store,
            self.reducer.attestation_verifier,
            self.reducer.task_adapter_provider,
            self.reducer.current_release_provider,
            self.reducer.validation_state_provider,
        )
        expected_state = historical_reducer.advance_publication_eligibility_stage(
            state=predecessor_state,
            attempt=latest_attempt,
            accepted_results=predecessor_snapshot.accepted_stage_results,
            result=expected_result,
        )
        if (
            operation.operation_kind
            is not ExpertValidationOperationKind.PUBLICATION_ELIGIBILITY_STAGE_RESULT
            or operation.request_record_id != result_record.stage_result_record_id
            or operation.expected_transition_id != transition.predecessor_transition_id
            or result_record != expected_result
            or result_record.promotion_decision != decision
            or decision != expected_decision
            or result_record.release_matrix_acceptance_transition_id
            != transition.predecessor_transition_id
            or result_record.release_matrix_acceptance_state_id
            != predecessor_state.validation_state_id
            or result_record.accepted_stage_results
            != predecessor_state.accepted_stage_results
            or result_record.validation_attempt_id
            != latest_attempt.validation_attempt_id
            or result_record.candidate_id != transition.candidate_id
            or result_record.candidate_tree_hash != transition.candidate_tree_hash
            or result_record.candidate_commit_record_id
            != latest_attempt.candidate_commit_record_id
            or result_record.scope_contract_id != latest_attempt.scope_contract_id
            or result_record.expected_current_release_id
            != latest_attempt.expected_current_release_id
            or result_record.validation_policy_id != latest_attempt.validation_policy_id
            or result_record.configuration_fingerprint
            != latest_attempt.configuration_fingerprint
            or state != expected_state
            or transition.accepted_stage_result_record_ids
            != tuple(
                item.stage_result_record_id for item in state.accepted_stage_results
            )
        ):
            raise ExpertValidationStoreError(
                "publication eligibility transition closure is inconsistent"
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
            for stage, record_id, _accepted, _attempt_id, _candidate_id, _tree_hash in (
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
                    for stage, _record_id, _accepted, _attempt_id, _candidate_id, _tree_hash in accepted_projections
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
                not accepted
                for _stage, _record_id, accepted, _attempt_id, _candidate_id, _tree_hash in accepted_projections
            )
            or any(
                latest_attempt is None
                or attempt_id != latest_attempt.validation_attempt_id
                or candidate_id != transition.candidate_id
                or candidate_tree_hash != transition.candidate_tree_hash
                for _stage, _record_id, _accepted, attempt_id, candidate_id, candidate_tree_hash in accepted_projections
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
                or latest_attempt.source_base_release_id
                != decision.source_base_release_id
                or latest_attempt.expected_current_release_id
                != decision.expected_current_release_id
                or latest_attempt.recovery_plan_id != decision.recovery_plan_id
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
                or latest_attempt.control_dependency_ids
                != decision.control_dependency_ids
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
                    result_record.evaluator_run.stage
                    in {
                        ExpertValidationStage.SOURCE_RUN_REPLAY,
                        ExpertValidationStage.AUTOMATED_REVIEW,
                        ExpertValidationStage.RELEASE_MATRIX,
                        ExpertValidationStage.PUBLICATION_ELIGIBILITY,
                    }
                    or operation.operation_kind
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
            elif type(result_record) is ExpertSourceReplayStageResultRecord:
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
            elif type(result_record) is ExpertAutomatedReviewStageResultRecord:
                self._validate_automated_review_transition_unlocked(
                    transition=transition,
                    state=state,
                    operation=operation,
                    latest_attempt=latest_attempt,
                    previous_accepted=previous_accepted,
                    result_record=result_record,
                    validation_policy=policy,
                )
                expected_accepted = previous_accepted
                if result_record.outcome is ExpertAutomatedReviewOutcome.PASSED:
                    expected_accepted = (
                        *previous_accepted,
                        result_record.stage_result_record_id,
                    )
            elif type(result_record) is ExpertReleaseMatrixStageResultRecord:
                self._validate_release_matrix_stage_transition_unlocked(
                    journal=journal,
                    transition=transition,
                    state=state,
                    operation=operation,
                    latest_attempt=latest_attempt,
                    previous_accepted=previous_accepted,
                    result_record=result_record,
                )
                expected_accepted = (
                    *previous_accepted,
                    result_record.stage_result_record_id,
                )
            elif type(result_record) is ExpertPublicationEligibilityStageResultRecord:
                self._validate_publication_eligibility_transition_unlocked(
                    journal=journal,
                    transition=transition,
                    state=state,
                    operation=operation,
                    latest_attempt=latest_attempt,
                    previous_accepted=previous_accepted,
                    result_record=result_record,
                    persisted_settings=persisted_settings,
                )
                expected_accepted = previous_accepted
                if result_record.publication_authority_fence is not None:
                    expected_accepted = (
                        *previous_accepted,
                        result_record.stage_result_record_id,
                    )
            else:
                raise ExpertValidationStoreError(
                    "validation stage result type is unsupported"
                )
            if transition.accepted_stage_result_record_ids != expected_accepted:
                raise ExpertValidationStoreError(
                    "validation accepted result prefix is not gap-free"
                )
        elif transition.transition_release_use_block_decision_id is not None:
            if previous_transition is None or latest_attempt is None:
                raise ExpertValidationStoreError(
                    "release-use block requires an approved predecessor"
                )
            predecessor_snapshot = self._snapshot_at_unlocked(
                journal,
                previous_transition.transition_id,
            )
            predecessor_state = predecessor_snapshot.state
            publication_result = predecessor_snapshot.accepted_stage_results[-1]
            if (
                type(publication_result)
                is not ExpertPublicationEligibilityStageResultRecord
            ):
                raise ExpertValidationStoreError(
                    "release-use block predecessor lacks publication eligibility"
                )
            decision = self._read_contract_unlocked(
                transition.transition_release_use_block_decision_id,
                ExpertCandidateReleaseUseDecision,
            )
            observation = decision.policy_observation
            if (
                self._read_contract_unlocked(
                    observation.observation_id,
                    type(observation),
                )
                != observation
            ):
                raise ExpertValidationStoreError(
                    "release-use block observation is inconsistent"
                )
            for revocation in observation.matched_revocations:
                if (
                    self._read_contract_unlocked(
                        revocation.revocation_id,
                        type(revocation),
                    )
                    != revocation
                ):
                    raise ExpertValidationStoreError(
                        "release-use block event is inconsistent"
                    )
            stored_candidate = self.reducer.candidate_store.read(
                latest_attempt.candidate_id
            )
            if (
                observation.checked_release_ids
                != stored_candidate.closure.manifest.consumed_expert_release_ids
            ):
                raise ExpertValidationStoreError(
                    "release-use block checked another candidate closure"
                )
            historical_reducer = ExpertValidationReducer(
                persisted_settings,
                self.reducer.candidate_store,
                self.reducer.attestation_verifier,
                self.reducer.task_adapter_provider,
                self.reducer.current_release_provider,
                self.reducer.validation_state_provider,
            )
            expected_state = historical_reducer.advance_release_use_block(
                state=predecessor_state,
                attempt=latest_attempt,
                publication_result=publication_result,
                decision=decision,
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.RELEASE_USE_BLOCK
                or operation.request_record_id != decision.release_use_decision_id
                or operation.expected_transition_id
                != transition.predecessor_transition_id
                or transition.predecessor_transition_id
                != previous_transition.transition_id
                or transition.predecessor_state_id
                != predecessor_state.validation_state_id
                or transition.accepted_stage_result_record_ids
                != previous_transition.accepted_stage_result_record_ids
                or state != expected_state
            ):
                raise ExpertValidationStoreError(
                    "release-use block transition closure is inconsistent"
                )
        elif transition.transition_release_activation_receipt_id is not None:
            if previous_transition is None or latest_attempt is None:
                raise ExpertValidationStoreError(
                    "release activation requires an approved predecessor"
                )
            predecessor_state = self._read_contract_unlocked(
                previous_transition.target_state_id,
                ExpertCandidateValidationState,
            )
            receipt = self._read_contract_unlocked(
                transition.transition_release_activation_receipt_id,
                ExpertReleaseActivationReceipt,
            )
            intent = self._read_contract_unlocked(
                receipt.publication_intent_id,
                ExpertReleasePublicationIntent,
            )
            plan = self._read_contract_unlocked(
                receipt.publication_plan_id,
                ExpertReleasePublicationPlan,
            )
            manifest = self._read_contract_unlocked(
                receipt.release_id,
                ExpertBaseReleaseManifest,
            )
            predecessor_snapshot = self._snapshot_at_unlocked(
                journal,
                previous_transition.transition_id,
            )
            approval_snapshot = self._snapshot_at_unlocked(
                journal,
                plan.approval_transition_id,
            )
            self._validate_release_publication_reservation_unlocked(
                intent,
                plan,
                manifest,
                approval_snapshot,
            )
            self._validate_release_publication_remote_history(
                intent,
                plan,
                receipt.github_publication_intent,
                receipt.github_publication_pointer,
                None,
            )
            historical_reducer = ExpertValidationReducer(
                persisted_settings,
                self.reducer.candidate_store,
                self.reducer.attestation_verifier,
                self.reducer.task_adapter_provider,
                self.reducer.current_release_provider,
                self.reducer.validation_state_provider,
            )
            expected_state = historical_reducer.advance_release_activation(
                state=predecessor_state,
                approval_state=approval_snapshot.state,
                attempt=latest_attempt,
                plan=plan,
                receipt=receipt,
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.RELEASE_ACTIVATION
                or operation.request_record_id != receipt.activation_receipt_id
                or operation.expected_transition_id
                != transition.predecessor_transition_id
                or receipt.publication_plan_id != plan.publication_plan_id
                or receipt.release_id != manifest.release_id
                or receipt.candidate_id != transition.candidate_id
                or receipt.approval_transition_id != plan.approval_transition_id
                or receipt.approval_state_id != plan.approval_state_id
                or transition.accepted_stage_result_record_ids
                != previous_transition.accepted_stage_result_record_ids
                or state != expected_state
            ):
                raise ExpertValidationStoreError(
                    "release activation transition closure is inconsistent"
                )
        elif transition.transition_release_revocation_receipt_id is not None:
            if previous_transition is None or latest_attempt is None:
                raise ExpertValidationStoreError(
                    "release revocation requires a released predecessor"
                )
            predecessor_state = self._read_contract_unlocked(
                previous_transition.target_state_id,
                ExpertCandidateValidationState,
            )
            revocation_receipt = self._read_contract_unlocked(
                transition.transition_release_revocation_receipt_id,
                ExpertReleaseRevocationReceipt,
            )
            activation_receipt_id = (
                previous_transition.transition_release_activation_receipt_id
            )
            if activation_receipt_id is None:
                raise ExpertValidationStoreError(
                    "release revocation predecessor lacks activation evidence"
                )
            activation_receipt = self._read_contract_unlocked(
                activation_receipt_id,
                ExpertReleaseActivationReceipt,
            )
            manifest = self._read_contract_unlocked(
                activation_receipt.release_id,
                ExpertBaseReleaseManifest,
            )
            historical_reducer = ExpertValidationReducer(
                persisted_settings,
                self.reducer.candidate_store,
                self.reducer.attestation_verifier,
                self.reducer.task_adapter_provider,
                self.reducer.current_release_provider,
                self.reducer.validation_state_provider,
            )
            expected_state = historical_reducer.advance_release_revocation(
                authorization_transition_id=previous_transition.transition_id,
                state=predecessor_state,
                attempt=latest_attempt,
                activation_receipt=activation_receipt,
                release_manifest=manifest,
                revocation_receipt=revocation_receipt,
            )
            if (
                operation.operation_kind
                is not ExpertValidationOperationKind.RELEASE_REVOCATION
                or operation.request_record_id
                != revocation_receipt.revocation_receipt_id
                or operation.expected_transition_id
                != transition.predecessor_transition_id
                or transition.predecessor_transition_id
                != previous_transition.transition_id
                or transition.predecessor_state_id
                != predecessor_state.validation_state_id
                or revocation_receipt.authorization_transition_id
                != previous_transition.transition_id
                or revocation_receipt.authorization_state_id
                != predecessor_state.validation_state_id
                or revocation_receipt.activation_receipt_id
                != activation_receipt.activation_receipt_id
                or transition.accepted_stage_result_record_ids
                != previous_transition.accepted_stage_result_record_ids
                or state != expected_state
            ):
                raise ExpertValidationStoreError(
                    "release revocation transition closure is inconsistent"
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
                or invalidation.kind
                is not ExpertValidationAuthorityInvalidationKind.CURRENT_RELEASE_AUTHORITY_CHANGED
                or invalidation.expected_current_release_id
                != latest_attempt.expected_current_release_id
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
                or state.reason != "validation_current_release_authority_changed"
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
        if journal.release_publication_intent_id is not None:
            raise ExpertValidationCompareAndSwapError(
                "validation approval is frozen for release publication"
            )
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
            release_publication_intent_id=None,
            release_publication_stale_resolution_id=(
                journal.release_publication_stale_resolution_id
            ),
        )

    @staticmethod
    def _append_release_use_block_transition(
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
    ) -> ExpertValidationJournal:
        if transition.transition_release_use_block_decision_id is None:
            raise ExpertValidationCompareAndSwapError(
                "release-use block transition lacks its decision"
            )
        operations = dict(journal.operation_transition_ids)
        operations[transition.operation_id] = transition.transition_id
        return ExpertValidationJournal(
            candidate_id=journal.candidate_id,
            candidate_tree_hash=journal.candidate_tree_hash,
            transition_ids=(*journal.transition_ids, transition.transition_id),
            operation_transition_ids=operations,
            release_publication_intent_id=journal.release_publication_intent_id,
            release_publication_stale_resolution_id=(
                journal.release_publication_stale_resolution_id
            ),
        )

    @staticmethod
    def _append_release_activation_transition(
        journal: ExpertValidationJournal,
        transition: ExpertValidationTransition,
    ) -> ExpertValidationJournal:
        if (
            journal.release_publication_intent_id is None
            or journal.release_publication_stale_resolution_id is not None
            or transition.transition_release_activation_receipt_id is None
        ):
            raise ExpertValidationCompareAndSwapError(
                "release activation lacks the frozen publication intent"
            )
        operations = dict(journal.operation_transition_ids)
        operations[transition.operation_id] = transition.transition_id
        return ExpertValidationJournal(
            candidate_id=journal.candidate_id,
            candidate_tree_hash=journal.candidate_tree_hash,
            transition_ids=(*journal.transition_ids, transition.transition_id),
            operation_transition_ids=operations,
            release_publication_intent_id=None,
            release_publication_stale_resolution_id=None,
        )

    def _release_activation_result_unlocked(
        self,
        journal: ExpertValidationJournal,
    ) -> ExpertReleaseActivationCommitResult | None:
        activation_transitions = tuple(
            transition
            for transition in (
                self._read_contract_unlocked(
                    transition_id,
                    ExpertValidationTransition,
                )
                for transition_id in journal.transition_ids
            )
            if transition.transition_release_activation_receipt_id is not None
        )
        if not activation_transitions:
            return None
        if len(activation_transitions) != 1:
            raise ExpertValidationStoreError(
                "candidate has multiple durable release activations"
            )
        transition = activation_transitions[0]
        receipt_id = transition.transition_release_activation_receipt_id
        if receipt_id is None:
            raise ExpertValidationStoreError("release activation receipt is missing")
        receipt = self._read_contract_unlocked(
            receipt_id,
            ExpertReleaseActivationReceipt,
        )
        snapshot = self._snapshot_at_unlocked(journal, transition.transition_id)
        if (
            snapshot.state.promotion_state is not ExpertPromotionState.RELEASED
            or receipt.candidate_id != journal.candidate_id
            or receipt.activation_receipt_id != snapshot.state.transition_evidence_id
        ):
            raise ExpertValidationStoreError(
                "durable release activation outcome is inconsistent"
            )
        return ExpertReleaseActivationCommitResult(
            receipt=receipt,
            snapshot=snapshot,
            replayed=True,
        )

    def _release_use_block_result_unlocked(
        self,
        journal: ExpertValidationJournal,
    ) -> ExpertReleaseUseBlockCommitResult | None:
        transitions = tuple(
            transition
            for transition in (
                self._read_contract_unlocked(
                    transition_id,
                    ExpertValidationTransition,
                )
                for transition_id in journal.transition_ids
            )
            if transition.transition_release_use_block_decision_id is not None
        )
        if not transitions:
            return None
        if len(transitions) != 1:
            raise ExpertValidationStoreError(
                "candidate has multiple durable release-use blocks"
            )
        transition = transitions[0]
        decision_id = transition.transition_release_use_block_decision_id
        if decision_id is None:
            raise ExpertValidationStoreError("release-use block decision is missing")
        return ExpertReleaseUseBlockCommitResult(
            decision=self._read_contract_unlocked(
                decision_id,
                ExpertCandidateReleaseUseDecision,
            ),
            snapshot=self._snapshot_at_unlocked(journal, transition.transition_id),
            replayed=True,
        )

    def _release_revocation_target_unlocked(
        self,
        journal: ExpertValidationJournal,
    ) -> ExpertReleaseRevocationTarget:
        activation = self._release_activation_result_unlocked(journal)
        current = self._current_from_journal_unlocked(journal)
        if (
            activation is None
            or current is None
            or current.transition.transition_id
            != activation.snapshot.transition.transition_id
            or current.state.promotion_state is not ExpertPromotionState.RELEASED
        ):
            raise ExpertValidationCompareAndSwapError(
                "release revocation requires the current durable RELEASED head"
            )
        manifest = self._read_contract_unlocked(
            activation.receipt.release_id,
            ExpertBaseReleaseManifest,
        )
        attempt = activation.snapshot.latest_attempt
        if attempt is None:
            raise ExpertValidationStoreError(
                "release revocation target lacks its validation attempt"
            )
        subjects = expert_release_revocation_security_subject_ids(
            authorization_transition_id=activation.snapshot.transition.transition_id,
            released_state=activation.snapshot.state,
            validation_attempt=attempt,
            activation_receipt=activation.receipt,
            release_manifest=manifest,
        )
        return ExpertReleaseRevocationTarget(
            activation=activation,
            manifest=manifest,
            security_subject_ids=subjects,
        )

    @staticmethod
    def _mint_release_revocation_receipt(
        target: ExpertReleaseRevocationTarget,
        observation: SecurityDenylistObservation,
        revoked_at: str,
    ) -> ExpertReleaseRevocationReceipt:
        snapshot = target.activation.snapshot
        attempt = snapshot.latest_attempt
        if attempt is None:
            raise ExpertValidationStoreError(
                "release revocation receipt lacks its validation attempt"
            )
        dependencies = {
            target.manifest.release_id,
            target.manifest.candidate_id,
            attempt.validation_attempt_id,
            snapshot.transition.transition_id,
            snapshot.state.validation_state_id,
            target.activation.receipt.activation_receipt_id,
            observation.observation_id,
            observation.scope_contract_id,
            observation.snapshot_id,
            observation.publication_id,
            *observation.checked_subject_ids,
            *(
                revocation.revocation_id
                for revocation in observation.matched_revocations
            ),
            *(revocation.subject_id for revocation in observation.matched_revocations),
            *(
                evidence_id
                for revocation in observation.matched_revocations
                for evidence_id in revocation.evidence_ids
            ),
        }
        return ExpertReleaseRevocationReceipt.mint(
            release_id=target.manifest.release_id,
            candidate_id=target.manifest.candidate_id,
            candidate_tree_hash=target.manifest.candidate_tree_hash,
            validation_attempt_id=attempt.validation_attempt_id,
            authorization_transition_id=snapshot.transition.transition_id,
            authorization_state_id=snapshot.state.validation_state_id,
            activation_receipt_id=(target.activation.receipt.activation_receipt_id),
            security_denylist_observation=observation,
            revoked_at=revoked_at,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )

    def _release_revocation_result_unlocked(
        self,
        journal: ExpertValidationJournal,
    ) -> ExpertReleaseRevocationCommitResult | None:
        self._validate_journal_unlocked(journal)
        revocation_transitions = tuple(
            transition
            for transition in (
                self._read_contract_unlocked(
                    transition_id,
                    ExpertValidationTransition,
                )
                for transition_id in journal.transition_ids
            )
            if transition.transition_release_revocation_receipt_id is not None
        )
        if not revocation_transitions:
            return None
        if len(revocation_transitions) != 1:
            raise ExpertValidationStoreError(
                "candidate has multiple durable release revocations"
            )
        transition = revocation_transitions[0]
        receipt_id = transition.transition_release_revocation_receipt_id
        if receipt_id is None:
            raise ExpertValidationStoreError("release revocation receipt is missing")
        receipt = self._read_contract_unlocked(
            receipt_id,
            ExpertReleaseRevocationReceipt,
        )
        snapshot = self._snapshot_at_unlocked(journal, transition.transition_id)
        if (
            snapshot.state.promotion_state is not ExpertPromotionState.REVOKED
            or receipt.candidate_id != journal.candidate_id
            or receipt.revocation_receipt_id != snapshot.state.transition_evidence_id
        ):
            raise ExpertValidationStoreError(
                "durable release revocation outcome is inconsistent"
            )
        return ExpertReleaseRevocationCommitResult(
            receipt=receipt,
            snapshot=snapshot,
            replayed=True,
        )

    @staticmethod
    def _bind_operation(
        journal: ExpertValidationJournal,
        operation: ExpertValidationOperation,
        transition: ExpertValidationTransition,
    ) -> ExpertValidationJournal:
        if journal.release_publication_intent_id is not None:
            raise ExpertValidationCompareAndSwapError(
                "validation approval is frozen for release publication"
            )
        operations = dict(journal.operation_transition_ids)
        operations[operation.operation_id] = transition.transition_id
        return ExpertValidationJournal(
            candidate_id=journal.candidate_id,
            candidate_tree_hash=journal.candidate_tree_hash,
            transition_ids=journal.transition_ids,
            operation_transition_ids=operations,
            release_publication_intent_id=None,
            release_publication_stale_resolution_id=(
                journal.release_publication_stale_resolution_id
            ),
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
                release_publication_intent_id=None,
                release_publication_stale_resolution_id=None,
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
