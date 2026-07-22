"""Evidence-triggered expert candidate proposal plane."""

from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertEvaluatorResultRecord,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    compile_expert_semantic_book,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.architect import ExpertRepositoryArchitect
from kapso.cross_run.expert.candidates import (
    ExpertCandidateClosure,
    ExpertCandidateValidationError,
    ExpertCandidateValidator,
)
from kapso.cross_run.expert.sanitation import ExpertCandidateSanitizer
from kapso.cross_run.expert.generalizer import ExpertCapabilityGeneralizer
from kapso.cross_run.expert.proposal import (
    ExpertCandidateProposalEngine,
    ExpertCandidateProposalResult,
)
from kapso.cross_run.expert.proposal_contract import (
    ExpertCandidateAncestorInput,
    ExpertProposalContractError,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.replay_context import (
    SourceReplayContextProvider,
    VerifiedSourceReplayContext,
    VerifiedSourceReplayStartingArtifact,
)
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayBundleProvider,
    ExpertSourceReplayCandidateReader,
    ExpertSourceReplayParentProvider,
    ExpertSourceReplayPreflightCoordinator,
    ExpertSourceReplayPreflightResult,
    ExpertSourceReplayRequestError,
    ExpertSourceReplayTaskAdapterProvider,
    ExpertSourceReplayValidationAuthority,
    MaterializedExpertSourceReplayCase,
    PreparedExpertSourceReplayRequest,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
    VerifiedTaskEvaluationStartingArtifact,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
    TaskEvaluationRequest,
    TaskEvaluationReservation,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    ExecutableTaskEvaluationCase,
    ExecutableTaskEvaluationLeg,
    ResolvedTaskEvaluationCase,
    TaskEvaluationExecutionError,
    TaskEvaluationExecutionProvider,
    TaskEvaluationExecutionProviderKey,
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationProviderCompletion,
    TaskEvaluationProviderExecutionHandle,
    TaskEvaluationProviderSupportRequirements,
    project_prepared_task_evaluation_cases,
    task_evaluation_provider_execution_handle,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationAuthorityError,
    TaskEvaluationCurrentReleaseObservation,
    TaskEvaluationSpawnAuthorityFence,
)
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
    task_evaluation_adapter_trust_observations,
    task_evaluation_allocation_case_leg,
    task_evaluation_spawn_security_subject_ids,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    MaterializedTaskEvaluationCase,
    PreparedTaskEvaluationRequest,
    TaskEvaluationAdapterProvider,
    TaskEvaluationCandidateReader,
    TaskEvaluationCurrentReleaseAuthority,
    TaskEvaluationParentProvider,
    TaskEvaluationPlanReservationAuthority,
    TaskEvaluationPreflightCoordinator,
    TaskEvaluationPreflightError,
    task_evaluation_materialization_usage,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayPublicationError,
    ExpertSourceReplayStageResultRecord,
    SourceReplayDecisionPublicationFence,
    source_replay_publication_security_subject_ids,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
    ExpertSourceReplayPublicationCurrentAuthority,
    ExpertSourceReplayPublicationDenylistAuthority,
)
from kapso.cross_run.expert.replay_stage import (
    ExpertSourceReplayPermanentlyInterruptedError,
    ExpertSourceReplayStageError,
    ExpertSourceReplayStageOrchestrator,
    SourceReplayProviderRegistryFactory,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
    ExpertAutomatedReviewExecution,
    PreparedExpertAutomatedReviewPacket,
)
from kapso.cross_run.expert.review_contracts import (
    EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION,
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewAssertion,
    ExpertAutomatedReviewError,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewPacket,
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.review_stage import (
    ExpertAutomatedReviewStageError,
    ExpertAutomatedReviewStageOrchestrator,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    ExpertCandidateStoreError,
    StoredExpertCandidate,
)
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertParentTreeReceipt,
    ExpertTriggerDecisionStore,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvidencePacketBuilder,
    ExpertTriggerError,
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.validation import (
    ExpertAttestationVerifier,
    ExpertCandidateEligibilityEvaluator,
    ExpertCurrentReleaseProvider,
    ExpertEligibilityResult,
    ExpertEvaluatorRunBuilder,
    ExpertValidationError,
    ExpertValidationAuthorityInvalidationResult,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
    ExpertValidationStart,
    ExpertValidationStateProvider,
)
from kapso.cross_run.expert.validation_store import (
    ExpertAutomatedReviewStageCommitResult,
    ExpertSourceReplayStageCommitResult,
    ExpertTaskEvaluationReservationCommitResult,
    ExpertValidationCommitResult,
    ExpertValidationCompareAndSwapError,
    ExpertValidationJournal,
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from kapso.cross_run.expert.validation_operation_contracts import (
    ExpertValidationOperation,
    ExpertValidationOperationKind,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertReleaseMatrixPlanReservationSnapshot,
    ExpertValidationSnapshot,
    ExpertValidationTransition,
)
from kapso.cross_run.expert.workspace import (
    ExpertCandidateWorkspaceError,
    ExpertCandidateWorkspaceLease,
    ExpertCandidateWorkspaceManager,
    PreparedExpertCandidateWorkspace,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)

__all__ = [
    "EXPERT_BOOK_PATH",
    "EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION",
    "EMPTY_EXPERT_TREE_DIGEST",
    "ExpertAutomatedReviewAdjudication",
    "ExpertAutomatedReviewAssertion",
    "ExpertAutomatedReviewCoordinator",
    "ExpertAutomatedReviewError",
    "ExpertAutomatedReviewExecution",
    "ExpertAutomatedReviewOperationRecord",
    "ExpertAutomatedReviewOutcome",
    "ExpertAutomatedReviewPacket",
    "ExpertAutomatedReviewStageResultRecord",
    "ExpertAutomatedReviewStageCommitResult",
    "ExpertAutomatedReviewStageError",
    "ExpertAutomatedReviewStageOrchestrator",
    "ExpertEvolutionTriggerDecision",
    "GitHubExpertCurrentReleaseProvider",
    "ExpertAttestationVerifier",
    "ExpertCandidateEligibilityEvaluator",
    "ExpertCurrentReleaseProvider",
    "ExpertCandidateClosure",
    "ExpertCandidateAncestorInput",
    "ExpertCandidateProposalEngine",
    "ExpertCandidateProposalResult",
    "ExpertCandidateValidationError",
    "ExpertCandidateValidator",
    "ExpertCapabilityGeneralizer",
    "ExpertEligibilityResult",
    "ExpertEvaluatorRunBuilder",
    "ExpertEvaluatorResultRecord",
    "ExpertCandidateWorkspaceError",
    "ExpertCandidateWorkspaceLease",
    "ExpertCandidateWorkspaceManager",
    "ExpertCandidateSanitizer",
    "ExpertCandidateStore",
    "ExpertCandidateStoreError",
    "ExpertParentTreeReceipt",
    "ExpertProposalContractError",
    "ExpertRepositoryArchitect",
    "ExpertReleaseMatrixPlanReservationSnapshot",
    "ExpertSourceReplayBundleProvider",
    "ExpertSourceReplayCandidateReader",
    "ExpertSourceReplayDecisionPublicationCoordinator",
    "ExpertSourceReplayParentProvider",
    "ExpertSourceReplayPreflightCoordinator",
    "ExpertSourceReplayPreflightResult",
    "ExpertSourceReplayPublicationError",
    "ExpertSourceReplayPublicationCurrentAuthority",
    "ExpertSourceReplayPublicationDenylistAuthority",
    "ExpertSourceReplayPermanentlyInterruptedError",
    "ExpertSourceReplayRequestError",
    "ExpertSourceReplayStageResultRecord",
    "ExpertSourceReplayStageCommitResult",
    "ExpertSourceReplayStageError",
    "ExpertSourceReplayStageOrchestrator",
    "ExpertSourceReplayTaskAdapterProvider",
    "ExpertSourceReplayValidationAuthority",
    "ExpertTriggerDecisionStore",
    "ExpertTriggerEvidencePacket",
    "ExpertTriggerEvidencePacketBuilder",
    "ExpertTriggerError",
    "ExpertTriggerEvaluator",
    "ExpertTriggerObservation",
    "ExpertTriggerObservationKind",
    "ExpertTaskEvaluationReservationCommitResult",
    "ExpertTaskEvaluationReservationSnapshot",
    "ExpertValidationError",
    "ExpertValidationAuthorityInvalidationResult",
    "ExpertValidationCommitResult",
    "ExpertValidationCompareAndSwapError",
    "ExpertValidationJournal",
    "ExpertValidationOperation",
    "ExpertValidationOperationKind",
    "ExpertValidationPredecessor",
    "ExpertValidationReducer",
    "ExpertValidationSnapshot",
    "ExpertValidationStart",
    "ExpertValidationStateProvider",
    "ExpertValidationStore",
    "ExpertValidationStoreError",
    "ExpertValidationTransition",
    "ExecutableTaskEvaluationCase",
    "ExecutableTaskEvaluationLeg",
    "VerifiedTaskAdapter",
    "VerifiedTaskAdapterProvider",
    "PreparedExpertCandidateWorkspace",
    "PreparedExpertAutomatedReviewPacket",
    "MaterializedExpertSourceReplayCase",
    "MaterializedTaskEvaluationCase",
    "PreparedExpertSourceReplayRequest",
    "PreparedTaskEvaluationRequest",
    "ResolvedTaskEvaluationCase",
    "SourceReplayContextProvider",
    "SourceReplayProviderRegistryFactory",
    "SourceReplayDecisionPublicationFence",
    "TaskEvaluationAdapterProvider",
    "TaskEvaluationAuthorityError",
    "TaskEvaluationCandidateReader",
    "TaskEvaluationCurrentReleaseAuthority",
    "TaskEvaluationCurrentReleaseObservation",
    "TaskEvaluationExecutionError",
    "TaskEvaluationExecutionProvider",
    "TaskEvaluationExecutionProviderKey",
    "TaskEvaluationExecutionProviderRegistry",
    "TaskEvaluationInvocationAllocation",
    "TaskEvaluationMaterializationLimits",
    "TaskEvaluationParentProvider",
    "TaskEvaluationPlanReservationAuthority",
    "TaskEvaluationPreflightCoordinator",
    "TaskEvaluationProviderCompletion",
    "TaskEvaluationProviderExecutionHandle",
    "TaskEvaluationProviderSupportRequirements",
    "TaskEvaluationRequest",
    "TaskEvaluationReservation",
    "TaskEvaluationSpawnAuthorityFence",
    "StoredExpertCandidate",
    "TaskEvaluationPreflightError",
    "VerifiedTaskEvaluationAdapterRuntime",
    "VerifiedTaskEvaluationCandidate",
    "VerifiedTaskEvaluationParent",
    "VerifiedTaskEvaluationStartingArtifact",
    "VerifiedSourceReplayContext",
    "VerifiedSourceReplayStartingArtifact",
    "materialize_task_evaluation_starting_artifacts",
    "project_prepared_task_evaluation_cases",
    "task_evaluation_adapter_trust_observations",
    "task_evaluation_allocation_case_leg",
    "task_evaluation_materialization_usage",
    "task_evaluation_provider_execution_handle",
    "task_evaluation_spawn_security_subject_ids",
    "build_task_evaluation_evaluator_request",
    "build_task_evaluation_spawn_authority_fence",
    "compile_expert_semantic_book",
    "expert_semantic_book_digest",
    "source_replay_publication_security_subject_ids",
]
