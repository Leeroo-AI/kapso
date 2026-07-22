"""Deterministic, byte-closed source replay preflight materialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.lineage import VerifiedRunBundleLineage
from kapso.cross_run.catalog.projector import ProjectionResult, RunBundleProjector
from kapso.cross_run.capture.bundle import StoredRunBundle
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertCandidateValidationState,
    ExpertPromotionState,
    ExpertSourceReplayCase,
    ExpertSourceReplayComputeBinding,
    ExpertSourceReplayExecutionCase,
    ExpertSourceReplayExecutionLeg,
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplaySelection,
    ExpertValidationAttempt,
    ExpertValidationStage,
    TransferEpisode,
    expert_source_replay_matched_compute_digest,
)
from kapso.cross_run.expert.replay import _derive_expert_source_replay_selection
from kapso.cross_run.expert.replay_context import (
    SourceReplayContextProvider,
    VerifiedSourceReplayContext,
)
from kapso.cross_run.expert.store import (
    StoredExpertCandidate,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
)
from kapso.cross_run.expert.triggers import ExpertParentTreeReceipt
from kapso.cross_run.expert.validation import (
    ExpertCurrentReleaseProvider,
    ExpertValidationPredecessor,
)
from kapso.cross_run.settings import ExpertValidationSettings
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    task_adapter_binding_id,
    task_adapter_materialization_usage,
)


class ExpertSourceReplayRequestError(ValueError):
    """A source replay preflight cannot be derived from exact trusted inputs."""


class ExpertSourceReplayCandidateReader(Protocol):
    def read(self, candidate_id: str) -> StoredExpertCandidate: ...


class ExpertSourceReplayBundleProvider(Protocol):
    def resolve_exact_bounded(
        self,
        bundle_id: str,
        *,
        maximum_entries: int,
        maximum_bytes: int,
        timeout_seconds: int,
        retained_bundles: Mapping[str, StoredRunBundle],
    ) -> VerifiedRunBundleLineage: ...


class ExpertSourceReplayValidationSnapshot(Protocol):
    state: ExpertCandidateValidationState


class ExpertSourceReplayValidationCommit(Protocol):
    snapshot: ExpertSourceReplayValidationSnapshot


class ExpertSourceReplayValidationAuthority(Protocol):
    def current(self, candidate_id: str) -> ExpertValidationPredecessor | None: ...

    def publish_parent_authority_invalidation(
        self,
        *,
        candidate_id: str,
        expected_validation_state_id: str,
    ) -> ExpertSourceReplayValidationCommit: ...


class ExpertSourceReplayParentProvider(Protocol):
    def materialize_exact(
        self,
        release_manifest: ExpertBaseReleaseManifest,
        parent_tree_receipt: ExpertParentTreeReceipt,
        limits: TaskEvaluationMaterializationLimits,
    ) -> VerifiedTaskEvaluationParent: ...


class ExpertSourceReplayTaskAdapterProvider(Protocol):
    def resolve_exact_bounded(
        self,
        *,
        task_adapter_manifest_id: str,
        verification_receipt_id: str,
        maximum_entries: int,
        maximum_bytes: int,
        timeout_seconds: int,
    ) -> VerifiedTaskAdapter: ...


def _source_replay_compute_bindings(
    settings: ExpertValidationSettings,
    episode_ids: tuple[str, ...],
) -> Mapping[str, ExpertSourceReplayComputeBinding]:
    ordered_episode_ids = tuple(sorted(episode_ids))
    if not ordered_episode_ids or len(ordered_episode_ids) != len(
        set(ordered_episode_ids)
    ):
        raise ExpertSourceReplayRequestError(
            "source replay compute schedule requires unique selected episodes"
        )
    policy = settings.policy
    evaluator = _source_replay_evaluator(settings)
    order_digest = tree_or_blob_digest(
        canonical_json_bytes(
            {
                "episode_ids": ordered_episode_ids,
                "paired_execution_protocol_version": (
                    policy.task_evaluation_execution_protocol_version
                ),
            }
        )
    )
    control_first = (
        ExpertSourceReplayExecutionLegKind.CONTROL_PARENT,
        ExpertSourceReplayExecutionLegKind.CANDIDATE,
    )
    candidate_first = tuple(reversed(control_first))
    starting_offset = int(order_digest[-1], 16) % 2
    return MappingProxyType(
        {
            episode_id: ExpertSourceReplayComputeBinding.mint(
                paired_execution_protocol_version=(
                    policy.task_evaluation_execution_protocol_version
                ),
                execution_provider_id=policy.task_evaluation_execution_provider_id,
                execution_provider_version=(
                    policy.task_evaluation_execution_provider_version
                ),
                execution_provider_settings_digest=tree_or_blob_digest(
                    settings.task_evaluation_provider.to_json_bytes()
                ),
                sandbox_policy_version=policy.task_evaluation_sandbox_policy_version,
                leg_wall_time_limit_seconds=evaluator.timeout_seconds,
                termination_grace_seconds=(
                    policy.task_evaluation_termination_grace_seconds
                ),
                cpu_millicore_limit=policy.task_evaluation_cpu_millicore_limit,
                memory_byte_limit=policy.task_evaluation_memory_byte_limit,
                shared_memory_byte_limit=(
                    policy.task_evaluation_shared_memory_byte_limit
                ),
                process_limit=policy.task_evaluation_process_limit,
                open_file_limit=policy.task_evaluation_open_file_limit,
                writable_inode_limit=policy.task_evaluation_writable_inode_limit,
                writable_storage_byte_limit=(
                    policy.task_evaluation_writable_storage_byte_limit
                ),
                output_entry_limit=policy.artifact_entry_limit,
                output_byte_limit=policy.artifact_byte_limit,
                stdout_byte_limit=policy.task_evaluation_stdout_byte_limit,
                stderr_byte_limit=policy.task_evaluation_stderr_byte_limit,
                accelerator_class_id=policy.task_evaluation_accelerator_class_id,
                accelerator_count=policy.task_evaluation_accelerator_count,
                leg_order=(
                    control_first
                    if (position + starting_offset) % 2 == 0
                    else candidate_first
                ),
            )
            for position, episode_id in enumerate(ordered_episode_ids)
        }
    )


@dataclass(frozen=True)
class MaterializedExpertSourceReplayCase:
    request_case: ExpertSourceReplayExecutionCase
    selection_case: ExpertSourceReplayCase
    bundle_lineage: VerifiedRunBundleLineage
    episode: TransferEpisode
    task_adapter: VerifiedTaskAdapter
    task_context: VerifiedSourceReplayContext

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_case, ExpertSourceReplayExecutionCase)
            or not isinstance(self.selection_case, ExpertSourceReplayCase)
            or not isinstance(self.episode, TransferEpisode)
            or not isinstance(self.task_adapter, VerifiedTaskAdapter)
            or not isinstance(self.task_context, VerifiedSourceReplayContext)
        ):
            raise ExpertSourceReplayRequestError(
                "materialized source replay case contains an unverified authority"
            )
        case = self.request_case
        if not isinstance(
            self.bundle_lineage, VerifiedRunBundleLineage
        ) or not isinstance(self.bundle_lineage.tip_projection, ProjectionResult):
            raise ExpertSourceReplayRequestError(
                "source replay requires a verified bundle lineage"
            )
        if any(
            not isinstance(bundle, StoredRunBundle)
            for bundle in self.bundle_lineage.bundles
        ):
            raise ExpertSourceReplayRequestError(
                "source replay requires the exact root-to-tip bundle byte closure"
            )
        bundle = self.bundle_lineage.tip_bundle.manifest
        projection = self.bundle_lineage.tip_projection
        projection_manifest = projection.projection_manifest
        projected_episodes = {
            episode.episode_id: episode for episode in projection.episodes
        }
        context = self.episode.task_context_binding
        receipt = self.task_context.receipt
        adapter = self.task_adapter
        terminal_attempt = self.episode.attempts[self.episode.terminal_attempt_revision]
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in terminal_attempt.evaluation_fingerprints
        )
        score_of_record_fingerprint_id = terminal_attempt.score_of_record_fingerprint_id
        if score_of_record_fingerprint_id is None:
            raise ExpertSourceReplayRequestError(
                "source replay requires a score-of-record fingerprint"
            )
        if any(
            fingerprint.evaluator_fingerprint
            not in adapter.manifest.task_evaluator.supported_evaluator_fingerprints
            for fingerprint in terminal_attempt.evaluation_fingerprints
        ):
            raise ExpertSourceReplayRequestError(
                "source replay fingerprint differs from the exact task evaluator"
            )
        comparison_bindings = {
            (binding.evaluator_fingerprint, binding.metric_name): binding
            for binding in adapter.manifest.task_evaluator.metric_comparison_bindings
        }
        if any(
            comparison_bindings.get(
                (fingerprint.evaluator_fingerprint, fingerprint.metric_name)
            )
            is None
            or comparison_bindings[
                (fingerprint.evaluator_fingerprint, fingerprint.metric_name)
            ].objective_direction
            is not fingerprint.objective_direction
            for fingerprint in terminal_attempt.evaluation_fingerprints
        ):
            raise ExpertSourceReplayRequestError(
                "source replay fingerprint lacks its exact metric comparison authority"
            )
        artifact_ids = tuple(
            artifact.starting_artifact_content_id
            for artifact in receipt.starting_artifacts
        )
        adapter_binding = task_adapter_binding_id(
            context.task_family_id,
            context.task_adapter_id,
        )
        adapter_dependencies = adapter.dependency_ids
        captured_artifact_ids = dict(
            self.episode.artifact_environment.starting_artifact_content_ids
        )
        materialized_artifact_ids = {
            artifact.starting_artifact_ref: artifact.starting_artifact_content_id
            for artifact in receipt.starting_artifacts
        }
        expected_dependencies = {
            *self.bundle_lineage.bundle_ids,
            projection_manifest.projection_manifest_id,
            self.episode.episode_id,
            *fingerprint_ids,
            context.task_context_binding_id,
            self.episode.artifact_environment.expert_base_release_id,
            receipt.context_materialization_receipt_id,
            *artifact_ids,
            adapter_binding,
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
            *adapter_dependencies,
            case.compute_binding.compute_binding_id,
            case.control_leg.execution_leg_id,
            *case.control_leg.exact_dependency_ids,
            case.candidate_leg.execution_leg_id,
            *case.candidate_leg.exact_dependency_ids,
        }
        matched_compute_digest = expert_source_replay_matched_compute_digest(
            bundle_lineage_ids=self.bundle_lineage.bundle_ids,
            projection_manifest_id=projection_manifest.projection_manifest_id,
            episode_id=self.episode.episode_id,
            source_execution_revision=self.episode.terminal_attempt_revision,
            source_evaluation_fingerprint_ids=fingerprint_ids,
            source_score_of_record_fingerprint_id=score_of_record_fingerprint_id,
            task_context_binding_id=context.task_context_binding_id,
            context_materialization_receipt_id=(
                receipt.context_materialization_receipt_id
            ),
            starting_artifact_content_ids=artifact_ids,
            task_adapter_manifest_id=adapter.manifest.task_adapter_manifest_id,
            verification_receipt_id=(
                adapter.verification_receipt.verification_receipt_id
            ),
            task_adapter_source_tree_hash=adapter.manifest.tree_hash,
            task_evaluator_digest=tree_or_blob_digest(
                adapter.manifest.task_evaluator.to_json_bytes()
            ),
            task_adapter_runtime_digest=tree_or_blob_digest(
                adapter.manifest.runtime.to_json_bytes()
            ),
            task_adapter_context_binding_digest=tree_or_blob_digest(
                adapter.manifest.context_binding.to_json_bytes()
            ),
            compute_binding_id=case.compute_binding.compute_binding_id,
        )
        if (
            self.bundle_lineage.bundle_ids != case.bundle_lineage_ids
            or bundle.bundle_id != case.source_bundle_id
            or projection.source_bundle != bundle
            or projection_manifest.projection_manifest_id != case.projection_manifest_id
            or case.episode_id not in projection_manifest.episode_ids
            or projected_episodes.get(case.episode_id) != self.episode
            or self.episode.source_bundle_id != case.source_bundle_id
            or self.episode.artifact_environment != bundle.artifact_environment
            or context != bundle.task_context_binding
            or receipt.task_context_binding_id != context.task_context_binding_id
            or receipt.input_contract_fingerprint != context.input_contract_fingerprint
            or receipt.target_contract_fingerprint
            != context.target_contract_fingerprint
            or materialized_artifact_ids != captured_artifact_ids
            or set(materialized_artifact_ids) != set(context.starting_artifact_refs)
            or not set(
                adapter.manifest.context_binding.consumed_dimension_ids
            ).issubset(context.transfer_dimensions)
            or self.episode.episode_id not in self.selection_case.episode_ids
            or case.episode_reason_codes
            != self.selection_case.episode_reason_codes[self.episode.episode_id]
            or self.episode.source["node_id"] != case.source_node_id
            or self.episode.terminal_attempt_revision != case.source_execution_revision
            or fingerprint_ids != case.source_evaluation_fingerprint_ids
            or score_of_record_fingerprint_id
            != case.source_score_of_record_fingerprint_id
            or context.task_context_binding_id != case.task_context_binding_id
            or self.episode.artifact_environment.expert_base_release_id
            != case.source_expert_base_release_id
            or receipt.context_materialization_receipt_id
            != case.context_materialization_receipt_id
            or artifact_ids != case.starting_artifact_content_ids
            or adapter_binding != case.adapter_binding_id
            or adapter.manifest.task_adapter_manifest_id
            != case.task_adapter_manifest_id
            or adapter.verification_receipt.verification_receipt_id
            != case.verification_receipt_id
            or adapter.manifest.tree_hash != case.task_adapter_source_tree_hash
            or tree_or_blob_digest(adapter.manifest.task_evaluator.to_json_bytes())
            != case.task_evaluator_digest
            or tree_or_blob_digest(adapter.manifest.runtime.to_json_bytes())
            != case.task_adapter_runtime_digest
            or tree_or_blob_digest(adapter.manifest.context_binding.to_json_bytes())
            != case.task_adapter_context_binding_digest
            or adapter_dependencies != case.task_adapter_dependency_ids
            or matched_compute_digest != case.matched_compute_binding_digest
            or expected_dependencies != set(case.exact_dependency_ids)
        ):
            raise ExpertSourceReplayRequestError(
                "materialized source replay case differs from its request"
            )


@dataclass(frozen=True)
class PreparedExpertSourceReplayRequest:
    request: ExpertSourceReplayExecutionRequest
    settings: ExpertValidationSettings
    attempt: ExpertValidationAttempt
    selection: ExpertSourceReplaySelection
    candidate: VerifiedTaskEvaluationCandidate
    parent: VerifiedTaskEvaluationParent
    authorization_state: ExpertCandidateValidationState
    cases: tuple[MaterializedExpertSourceReplayCase, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request, ExpertSourceReplayExecutionRequest)
            or not isinstance(self.settings, ExpertValidationSettings)
            or not isinstance(self.attempt, ExpertValidationAttempt)
            or not isinstance(self.selection, ExpertSourceReplaySelection)
            or not isinstance(self.candidate, VerifiedTaskEvaluationCandidate)
            or not isinstance(self.parent, VerifiedTaskEvaluationParent)
            or not isinstance(
                self.authorization_state,
                ExpertCandidateValidationState,
            )
            or any(
                not isinstance(item, MaterializedExpertSourceReplayCase)
                for item in self.cases
            )
        ):
            raise ExpertSourceReplayRequestError(
                "prepared source replay request contains an unverified authority"
            )
        evaluator = _source_replay_evaluator(self.settings)
        policy = self.settings.policy.validation_policy()
        pareto_dimensions = {
            dimension.dimension_id: dimension
            for dimension in self.settings.policy.promotion.pareto_dimensions
        }
        if any(
            binding.comparison_dimension_id not in pareto_dimensions
            or pareto_dimensions[binding.comparison_dimension_id].direction
            is not binding.objective_direction
            for item in self.cases
            for binding in (
                item.task_adapter.manifest.task_evaluator.metric_comparison_bindings
            )
        ):
            raise ExpertSourceReplayRequestError(
                "source replay metric comparison authority differs from central "
                "promotion policy"
            )
        request_cases = tuple(item.request_case for item in self.cases)
        adapters, lineages, contexts = _deduplicated_materialized_authorities(
            self.cases
        )
        projector = RunBundleProjector(
            self.settings.policy.task_evaluation_aggregate_tolerance
        )
        if any(
            _replay_bundle_lineage_from_bytes(
                lineage.bundle_ids[-1],
                lineage,
                projector,
            )
            != lineage
            for lineage in lineages
        ):
            raise ExpertSourceReplayRequestError(
                "prepared source replay lineage differs from exact bundle bytes"
            )
        entry_count, byte_count = _source_replay_materialization_usage(
            candidate=self.candidate,
            parent=self.parent,
            adapters=adapters,
            lineages=lineages,
            contexts=contexts,
        )
        parent_receipt = self.parent.parent_tree_receipt
        expected_dependencies = {
            self.attempt.validation_attempt_id,
            self.authorization_state.validation_state_id,
            self.selection.source_replay_selection_id,
            self.candidate.manifest.candidate_id,
            self.candidate.commit_record.commit_record_id,
            self.candidate.source_tree.source_tree_manifest_id,
            self.attempt.scope_contract_id,
            self.parent.release_manifest.release_id,
            parent_receipt.parent_tree_receipt_id,
            parent_receipt.source_extraction_receipt.extraction_receipt_id,
            self.attempt.validation_policy_id,
            *self.attempt.eligibility_dependency_ids,
            *(
                dependency_id
                for case in request_cases
                for dependency_id in (
                    case.execution_case_id,
                    *case.exact_dependency_ids,
                )
            ),
        }
        control_leg = _control_leg(self.parent)
        candidate_leg = _candidate_leg(self.candidate)
        selected_episode_assignments = {
            (selection_case.source_bundle_id, episode_id)
            for selection_case in self.selection.cases
            for episode_id in selection_case.episode_ids
        }
        materialized_episode_assignments = {
            (item.selection_case.source_bundle_id, item.episode.episode_id)
            for item in self.cases
        }
        selection_cases_by_bundle = {
            selection_case.source_bundle_id: selection_case
            for selection_case in self.selection.cases
        }
        if (
            self.attempt.source_replay_selection != self.selection
            or self.attempt.validation_policy_id != policy.validation_policy_id
            or self.attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
            or self.attempt.candidate_id != self.candidate.manifest.candidate_id
            or self.attempt.candidate_tree_hash
            != self.candidate.manifest.candidate_tree_hash
            or self.attempt.candidate_commit_record_id
            != self.candidate.commit_record.commit_record_id
            or self.attempt.parent_release_id != self.parent.release_manifest.release_id
            or self.authorization_state.validation_attempt_id
            != self.attempt.validation_attempt_id
            or self.authorization_state.candidate_id != self.attempt.candidate_id
            or self.authorization_state.candidate_tree_hash
            != self.attempt.candidate_tree_hash
            or self.authorization_state.promotion_state
            is not ExpertPromotionState.VALIDATING
            or self.authorization_state.next_stage
            is not ExpertValidationStage.SOURCE_RUN_REPLAY
            or materialized_episode_assignments != selected_episode_assignments
            or len(self.cases) != len(materialized_episode_assignments)
            or len(selection_cases_by_bundle) != len(self.selection.cases)
            or any(
                selection_cases_by_bundle.get(item.selection_case.source_bundle_id)
                != item.selection_case
                for item in self.cases
            )
            or entry_count
            > self.settings.policy.task_evaluation_materialization_entry_limit
            or byte_count
            > self.settings.policy.task_evaluation_materialization_byte_limit
            or any(
                item.task_context.receipt.materializer_id
                != self.settings.policy.source_replay_context_materializer_id
                or item.task_context.receipt.materializer_version
                != self.settings.policy.source_replay_context_materializer_version
                for item in self.cases
            )
            or self.request.validation_attempt_id != self.attempt.validation_attempt_id
            or self.request.authorization_state_id
            != self.authorization_state.validation_state_id
            or self.request.source_replay_selection_id
            != self.selection.source_replay_selection_id
            or self.request.candidate_id != self.candidate.manifest.candidate_id
            or self.request.candidate_tree_hash
            != self.candidate.manifest.candidate_tree_hash
            or self.request.candidate_commit_record_id
            != self.candidate.commit_record.commit_record_id
            or self.request.candidate_source_tree_manifest_id
            != self.candidate.source_tree.source_tree_manifest_id
            or self.request.scope_contract_id != self.attempt.scope_contract_id
            or self.request.parent_release_id != self.parent.release_manifest.release_id
            or self.request.parent_tree_receipt_id
            != parent_receipt.parent_tree_receipt_id
            or self.request.parent_source_extraction_receipt_id
            != parent_receipt.source_extraction_receipt.extraction_receipt_id
            or self.request.parent_tree_hash != parent_receipt.parent_tree_hash
            or self.request.validation_policy_id != self.attempt.validation_policy_id
            or self.request.configuration_fingerprint
            != self.attempt.configuration_fingerprint
            or self.request.request_policy_version
            != self.settings.policy.source_replay_request_policy_version
            or (
                self.request.evaluator_id,
                self.request.evaluator_role,
                self.request.evaluator_version,
            )
            != (
                evaluator.evaluator_id,
                evaluator.evaluator_role,
                evaluator.evaluator_version,
            )
            or self.request.attempt_dependency_ids
            != self.attempt.eligibility_dependency_ids
            or request_cases != self.request.cases
            or any(
                case.control_leg != control_leg or case.candidate_leg != candidate_leg
                for case in request_cases
            )
            or {case.episode_id: case.compute_binding for case in request_cases}
            != _source_replay_compute_bindings(
                self.settings,
                tuple(case.episode_id for case in request_cases),
            )
            or expected_dependencies != set(self.request.exact_dependency_ids)
        ):
            raise ExpertSourceReplayRequestError(
                "prepared source replay closure differs from its request"
            )


@dataclass(frozen=True)
class ExpertSourceReplayPreflightResult:
    prepared_request: PreparedExpertSourceReplayRequest | None
    invalidated_state: ExpertCandidateValidationState | None

    def __post_init__(self) -> None:
        if (self.prepared_request is None) == (self.invalidated_state is None):
            raise ExpertSourceReplayRequestError(
                "source replay preflight must prepare or invalidate exactly once"
            )
        if self.prepared_request is not None and not isinstance(
            self.prepared_request,
            PreparedExpertSourceReplayRequest,
        ):
            raise ExpertSourceReplayRequestError(
                "source replay preflight prepared an unverified request"
            )
        if self.invalidated_state is not None and (
            not isinstance(
                self.invalidated_state,
                ExpertCandidateValidationState,
            )
            or self.invalidated_state.promotion_state is not ExpertPromotionState.FAILED
            or self.invalidated_state.reason != "validation_parent_release_changed"
        ):
            raise ExpertSourceReplayRequestError(
                "source replay preflight returned an invalid invalidation state"
            )


class ExpertSourceReplayPreflightCoordinator:
    """Prepare exact replay bytes or terminally invalidate stale parent authority."""

    def __init__(
        self,
        settings: ExpertValidationSettings,
        candidate_store: ExpertSourceReplayCandidateReader,
        validation_authority: ExpertSourceReplayValidationAuthority,
        current_release_provider: ExpertCurrentReleaseProvider,
        parent_provider: ExpertSourceReplayParentProvider,
        bundle_provider: ExpertSourceReplayBundleProvider,
        task_adapter_provider: ExpertSourceReplayTaskAdapterProvider,
        task_context_provider: SourceReplayContextProvider,
        monotonic_clock: Callable[[], float],
    ) -> None:
        self.settings = settings
        self.candidate_store = candidate_store
        self.validation_authority = validation_authority
        self.current_release_provider = current_release_provider
        self.parent_provider = parent_provider
        self.bundle_provider = bundle_provider
        self.bundle_projector = RunBundleProjector(
            settings.policy.task_evaluation_aggregate_tolerance
        )
        self.task_adapter_provider = task_adapter_provider
        self.task_context_provider = task_context_provider
        self.monotonic_clock = monotonic_clock

    def build(
        self,
        attempt: ExpertValidationAttempt,
    ) -> ExpertSourceReplayPreflightResult:
        deadline = (
            self.monotonic_clock()
            + self.settings.policy.task_evaluation_materialization_timeout_seconds
        )
        state = self._authorized_state(attempt)
        stored_candidate = self.candidate_store.read(attempt.candidate_id)
        self._require_deadline(deadline)
        selection = self._validated_selection(attempt, stored_candidate)
        if not self._parent_is_current(attempt, stored_candidate):
            return self._invalidate_parent_authority(attempt, state)
        candidate = VerifiedTaskEvaluationCandidate(
            manifest=stored_candidate.closure.manifest,
            commit_record=stored_candidate.commit_record,
            source_tree=stored_candidate.closure.candidate_tree,
            source_contents=stored_candidate.closure.candidate_contents,
        )
        parent = self._materialize_parent(
            stored_candidate,
            self._remaining_materialization_limits(
                candidate=candidate,
                parent=None,
                adapters=(),
                lineages=(),
                contexts=(),
                deadline=deadline,
            ),
        )
        self._require_deadline(deadline)
        adapters_by_episode = self._resolve_source_adapters(
            selection,
            candidate,
            parent,
            deadline,
        )
        unique_adapters = tuple(
            {
                (
                    adapter.manifest.task_adapter_manifest_id,
                    adapter.verification_receipt.verification_receipt_id,
                ): adapter
                for adapter in adapters_by_episode.values()
            }.values()
        )
        packet_episodes = {
            episode.episode_id: episode
            for episode in stored_candidate.closure.trigger_packet.episodes
        }
        lineages: dict[str, VerifiedRunBundleLineage] = {}
        contexts: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            VerifiedSourceReplayContext,
        ] = {}
        materialized_cases: list[MaterializedExpertSourceReplayCase] = []
        control_leg = _control_leg(parent)
        candidate_leg = _candidate_leg(candidate)
        selected_episode_ids = tuple(
            episode_id
            for selected_case in selection.cases
            for episode_id in selected_case.episode_ids
        )
        compute_bindings = _source_replay_compute_bindings(
            self.settings,
            selected_episode_ids,
        )
        for selected_case in selection.cases:
            lineage = lineages.get(selected_case.source_bundle_id)
            if lineage is None:
                remaining = self._remaining_materialization_limits(
                    candidate=candidate,
                    parent=parent,
                    adapters=unique_adapters,
                    lineages=tuple(lineages.values()),
                    contexts=tuple(contexts.values()),
                    deadline=deadline,
                )
                supplied_lineage = self.bundle_provider.resolve_exact_bounded(
                    selected_case.source_bundle_id,
                    maximum_entries=remaining.maximum_entries,
                    maximum_bytes=remaining.maximum_bytes,
                    timeout_seconds=remaining.timeout_seconds,
                    retained_bundles=_retained_bundle_closures(
                        tuple(lineages.values())
                    ),
                )
                self._require_deadline(deadline)
                lineage = self._replay_bundle_lineage(
                    selected_case.source_bundle_id,
                    supplied_lineage,
                )
                lineages[selected_case.source_bundle_id] = lineage
            projected_episodes = {
                episode.episode_id: episode
                for episode in lineage.tip_projection.episodes
            }
            for episode_id in selected_case.episode_ids:
                episode = projected_episodes.get(episode_id)
                if episode is None or packet_episodes.get(episode_id) != episode:
                    raise ExpertSourceReplayRequestError(
                        "selected replay episode differs from the exact projection"
                    )
                materialized_cases.append(
                    self._build_case(
                        attempt=attempt,
                        selected_case=selected_case,
                        lineage=lineage,
                        episode=episode,
                        adapter=adapters_by_episode[episode_id],
                        contexts=contexts,
                        candidate=candidate,
                        parent=parent,
                        adapters=unique_adapters,
                        lineages=tuple(lineages.values()),
                        control_leg=control_leg,
                        candidate_leg=candidate_leg,
                        compute_binding=compute_bindings[episode_id],
                        deadline=deadline,
                    )
                )
                self._check_materialization_totals(
                    candidate=candidate,
                    parent=parent,
                    adapters=unique_adapters,
                    lineages=tuple(lineages.values()),
                    contexts=tuple(contexts.values()),
                )
                self._require_deadline(deadline)
        ordered_cases = tuple(
            sorted(materialized_cases, key=lambda item: item.request_case.episode_id)
        )
        if {item.request_case.episode_id for item in ordered_cases} != {
            *selection.causal_episode_ids,
            *selection.coverage_episode_ids,
        }:
            raise ExpertSourceReplayRequestError(
                "source replay request does not cover the selection exactly"
            )
        if self._authorized_state(attempt) != state:
            raise ExpertSourceReplayRequestError(
                "source replay validation authority changed during materialization"
            )
        self._require_deadline(deadline)
        if not self._parent_is_current(attempt, stored_candidate):
            return self._invalidate_parent_authority(attempt, state)
        evaluator = _source_replay_evaluator(self.settings)
        request_cases = tuple(item.request_case for item in ordered_cases)
        parent_receipt = parent.parent_tree_receipt
        dependencies = {
            attempt.validation_attempt_id,
            state.validation_state_id,
            selection.source_replay_selection_id,
            attempt.candidate_id,
            attempt.candidate_commit_record_id,
            candidate.source_tree.source_tree_manifest_id,
            attempt.scope_contract_id,
            attempt.parent_release_id,
            parent_receipt.parent_tree_receipt_id,
            parent_receipt.source_extraction_receipt.extraction_receipt_id,
            attempt.validation_policy_id,
            *attempt.eligibility_dependency_ids,
            *(
                dependency_id
                for request_case in request_cases
                for dependency_id in (
                    request_case.execution_case_id,
                    *request_case.exact_dependency_ids,
                )
            ),
        }
        request = ExpertSourceReplayExecutionRequest.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            authorization_state_id=state.validation_state_id,
            source_replay_selection_id=selection.source_replay_selection_id,
            candidate_id=attempt.candidate_id,
            candidate_tree_hash=attempt.candidate_tree_hash,
            candidate_commit_record_id=attempt.candidate_commit_record_id,
            candidate_source_tree_manifest_id=(
                candidate.source_tree.source_tree_manifest_id
            ),
            scope_contract_id=attempt.scope_contract_id,
            parent_release_id=attempt.parent_release_id,
            parent_tree_receipt_id=parent_receipt.parent_tree_receipt_id,
            parent_source_extraction_receipt_id=(
                parent_receipt.source_extraction_receipt.extraction_receipt_id
            ),
            parent_tree_hash=parent_receipt.parent_tree_hash,
            validation_policy_id=attempt.validation_policy_id,
            configuration_fingerprint=attempt.configuration_fingerprint,
            request_policy_version=(
                self.settings.policy.source_replay_request_policy_version
            ),
            evaluator_id=evaluator.evaluator_id,
            evaluator_role=evaluator.evaluator_role,
            evaluator_version=evaluator.evaluator_version,
            attempt_dependency_ids=attempt.eligibility_dependency_ids,
            cases=request_cases,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        prepared = PreparedExpertSourceReplayRequest(
            request=request,
            settings=self.settings,
            attempt=attempt,
            selection=selection,
            candidate=candidate,
            parent=parent,
            authorization_state=state,
            cases=ordered_cases,
        )
        self._require_deadline(deadline)
        return ExpertSourceReplayPreflightResult(
            prepared_request=prepared,
            invalidated_state=None,
        )

    def _authorized_state(
        self,
        attempt: ExpertValidationAttempt,
    ) -> ExpertCandidateValidationState:
        current = self.validation_authority.current(attempt.candidate_id)
        if (
            current is None
            or current.latest_attempt != attempt
            or current.state.validation_attempt_id != attempt.validation_attempt_id
            or current.state.promotion_state is not ExpertPromotionState.VALIDATING
            or current.state.next_stage is not ExpertValidationStage.SOURCE_RUN_REPLAY
        ):
            raise ExpertSourceReplayRequestError(
                "source replay is not the current authorized validation stage"
            )
        return current.state

    def _validated_selection(
        self,
        attempt: ExpertValidationAttempt,
        candidate: StoredExpertCandidate,
    ) -> ExpertSourceReplaySelection:
        policy = self.settings.policy.validation_policy()
        manifest = candidate.closure.manifest
        if (
            attempt.parent_release_id is None
            or attempt.validation_policy_id != policy.validation_policy_id
            or attempt.configuration_fingerprint
            != self.settings.configuration_fingerprint
            or attempt.candidate_id != manifest.candidate_id
            or attempt.candidate_tree_hash != manifest.candidate_tree_hash
            or attempt.candidate_commit_record_id
            != candidate.commit_record.commit_record_id
            or attempt.scope_contract_id != manifest.scope_contract_id
            or attempt.parent_release_id != manifest.parent_release_id
            or ExpertValidationStage.SOURCE_RUN_REPLAY not in attempt.required_stages
            or attempt.source_replay_selection is None
        ):
            raise ExpertSourceReplayRequestError(
                "source replay attempt differs from candidate or policy authority"
            )
        expected = _derive_expert_source_replay_selection(
            stored_candidate=candidate,
            settings=self.settings,
        ).selection
        if expected is None or expected != attempt.source_replay_selection:
            raise ExpertSourceReplayRequestError(
                "source replay selection differs from deterministic enrollment"
            )
        return expected

    def _parent_is_current(
        self,
        attempt: ExpertValidationAttempt,
        candidate: StoredExpertCandidate,
    ) -> bool:
        scope_id = candidate.closure.trigger_packet.scope_contract.scope_id
        return (
            self.current_release_provider.current_release_id(scope_id)
            == attempt.parent_release_id
        )

    def _invalidate_parent_authority(
        self,
        attempt: ExpertValidationAttempt,
        state: ExpertCandidateValidationState,
    ) -> ExpertSourceReplayPreflightResult:
        committed = self.validation_authority.publish_parent_authority_invalidation(
            candidate_id=attempt.candidate_id,
            expected_validation_state_id=state.validation_state_id,
        )
        invalidated = committed.snapshot.state
        if (
            invalidated.validation_attempt_id != attempt.validation_attempt_id
            or invalidated.candidate_id != attempt.candidate_id
            or invalidated.candidate_tree_hash != attempt.candidate_tree_hash
            or invalidated.predecessor_state_id != state.validation_state_id
            or invalidated.promotion_state is not ExpertPromotionState.FAILED
            or invalidated.reason != "validation_parent_release_changed"
        ):
            raise ExpertSourceReplayRequestError(
                "validation authority returned another parent invalidation"
            )
        return ExpertSourceReplayPreflightResult(
            prepared_request=None,
            invalidated_state=invalidated,
        )

    def _materialize_parent(
        self,
        candidate: StoredExpertCandidate,
        limits: TaskEvaluationMaterializationLimits,
    ) -> VerifiedTaskEvaluationParent:
        packet = candidate.closure.trigger_packet
        if packet.parent_release is None or packet.parent_tree_receipt is None:
            raise ExpertSourceReplayRequestError(
                "source replay requires an exact materialized parent release"
            )
        parent = self.parent_provider.materialize_exact(
            packet.parent_release,
            packet.parent_tree_receipt,
            limits,
        )
        if type(parent) is not VerifiedTaskEvaluationParent:
            raise ExpertSourceReplayRequestError(
                "parent provider returned an unverified source closure"
            )
        if (
            parent.release_manifest != packet.parent_release
            or parent.parent_tree_receipt != packet.parent_tree_receipt
            or parent.parent_tree_receipt.source_extraction_receipt.source_tree_files
            != candidate.closure.parent_files
        ):
            raise ExpertSourceReplayRequestError(
                "materialized parent differs from the candidate parent authority"
            )
        return parent

    def _resolve_source_adapters(
        self,
        selection: ExpertSourceReplaySelection,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent,
        deadline: float,
    ) -> dict[str, VerifiedTaskAdapter]:
        adapters_by_episode: dict[str, VerifiedTaskAdapter] = {}
        for pin in selection.source_adapter_pins:
            unique_adapters = tuple(
                {
                    (
                        adapter.manifest.task_adapter_manifest_id,
                        adapter.verification_receipt.verification_receipt_id,
                    ): adapter
                    for adapter in adapters_by_episode.values()
                }.values()
            )
            remaining = self._remaining_materialization_limits(
                candidate=candidate,
                parent=parent,
                adapters=unique_adapters,
                lineages=(),
                contexts=(),
                deadline=deadline,
            )
            adapter = self.task_adapter_provider.resolve_exact_bounded(
                task_adapter_manifest_id=pin.task_adapter_manifest_id,
                verification_receipt_id=pin.verification_receipt_id,
                maximum_entries=remaining.maximum_entries,
                maximum_bytes=remaining.maximum_bytes,
                timeout_seconds=remaining.timeout_seconds,
            )
            self._require_deadline(deadline)
            if not isinstance(adapter, VerifiedTaskAdapter):
                raise ExpertSourceReplayRequestError(
                    "historical source adapter is not a verified package"
                )
            manifest = adapter.manifest
            if (
                manifest.scope_contract_id != pin.scope_contract_id
                or manifest.task_family_id != pin.task_family_id
                or manifest.task_adapter_id != pin.task_adapter_id
                or manifest.task_adapter_manifest_id != pin.task_adapter_manifest_id
                or adapter.verification_receipt.verification_receipt_id
                != pin.verification_receipt_id
                or any(
                    episode_id in adapters_by_episode for episode_id in pin.episode_ids
                )
            ):
                raise ExpertSourceReplayRequestError(
                    "historical source adapter differs from its exact selection pin"
                )
            for episode_id in pin.episode_ids:
                adapters_by_episode[episode_id] = adapter
        return adapters_by_episode

    def _replay_bundle_lineage(
        self,
        expected_tip_bundle_id: str,
        supplied: VerifiedRunBundleLineage,
    ) -> VerifiedRunBundleLineage:
        return _replay_bundle_lineage_from_bytes(
            expected_tip_bundle_id,
            supplied,
            self.bundle_projector,
        )

    def _build_case(
        self,
        *,
        attempt: ExpertValidationAttempt,
        selected_case: ExpertSourceReplayCase,
        lineage: VerifiedRunBundleLineage,
        episode: TransferEpisode,
        adapter: VerifiedTaskAdapter,
        contexts: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            VerifiedSourceReplayContext,
        ],
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent,
        adapters: tuple[VerifiedTaskAdapter, ...],
        lineages: tuple[VerifiedRunBundleLineage, ...],
        control_leg: ExpertSourceReplayExecutionLeg,
        candidate_leg: ExpertSourceReplayExecutionLeg,
        compute_binding: ExpertSourceReplayComputeBinding,
        deadline: float,
    ) -> MaterializedExpertSourceReplayCase:
        bundle = lineage.tip_bundle.manifest
        projection = lineage.tip_projection
        context = episode.task_context_binding
        environment = episode.artifact_environment
        if (
            lineage.bundle_ids[-1] != selected_case.source_bundle_id
            or projection.source_bundle != bundle
            or projection.projection_manifest.source_bundle_id
            != selected_case.source_bundle_id
            or episode.episode_id not in projection.projection_manifest.episode_ids
            or episode.source_bundle_id != selected_case.source_bundle_id
            or context != bundle.task_context_binding
            or environment != bundle.artifact_environment
            or context.scope_contract_id != attempt.scope_contract_id
        ):
            raise ExpertSourceReplayRequestError(
                "source replay episode differs from its verified bundle projection"
            )
        binding_id = task_adapter_binding_id(
            context.task_family_id,
            context.task_adapter_id,
        )
        if (
            adapter.manifest.scope_contract_id != context.scope_contract_id
            or adapter.manifest.task_family_id != context.task_family_id
            or adapter.manifest.task_adapter_id != context.task_adapter_id
            or adapter.manifest.task_adapter_manifest_id
            != environment.task_adapter_manifest_id
            or adapter.verification_receipt.verification_receipt_id
            != environment.task_adapter_verification_receipt_id
        ):
            raise ExpertSourceReplayRequestError(
                "source replay episode lacks its exact historical adapter package"
            )
        context_key = (
            context.task_context_binding_id,
            tuple(sorted(environment.starting_artifact_content_ids.items())),
        )
        verified_context = contexts.get(context_key)
        if verified_context is None:
            verified_context = self.task_context_provider.materialize_exact(
                context,
                environment.starting_artifact_content_ids,
                self._remaining_materialization_limits(
                    candidate=candidate,
                    parent=parent,
                    adapters=adapters,
                    lineages=lineages,
                    contexts=tuple(contexts.values()),
                    deadline=deadline,
                ),
            )
            self._require_deadline(deadline)
            if not isinstance(verified_context, VerifiedSourceReplayContext):
                raise ExpertSourceReplayRequestError(
                    "context provider returned an unverified artifact closure"
                )
            self._validate_context(context, environment, verified_context)
            contexts[context_key] = verified_context
        terminal_attempt = episode.attempts[episode.terminal_attempt_revision]
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in terminal_attempt.evaluation_fingerprints
        )
        score_of_record_fingerprint_id = terminal_attempt.score_of_record_fingerprint_id
        if score_of_record_fingerprint_id is None:
            raise ExpertSourceReplayRequestError(
                "source replay requires a score-of-record fingerprint"
            )
        if any(
            fingerprint.evaluator_fingerprint
            not in adapter.manifest.task_evaluator.supported_evaluator_fingerprints
            for fingerprint in terminal_attempt.evaluation_fingerprints
        ):
            raise ExpertSourceReplayRequestError(
                "source replay fingerprint differs from the exact task evaluator"
            )
        projection_manifest_id = projection.projection_manifest.projection_manifest_id
        context_receipt = verified_context.receipt
        artifact_ids = tuple(
            artifact.starting_artifact_content_id
            for artifact in context_receipt.starting_artifacts
        )
        adapter_dependencies = adapter.dependency_ids
        dependencies = {
            *lineage.bundle_ids,
            projection_manifest_id,
            episode.episode_id,
            *fingerprint_ids,
            context.task_context_binding_id,
            environment.expert_base_release_id,
            context_receipt.context_materialization_receipt_id,
            *artifact_ids,
            binding_id,
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
            *adapter_dependencies,
            compute_binding.compute_binding_id,
            control_leg.execution_leg_id,
            *control_leg.exact_dependency_ids,
            candidate_leg.execution_leg_id,
            *candidate_leg.exact_dependency_ids,
        }
        request_case = ExpertSourceReplayExecutionCase.mint(
            source_bundle_id=selected_case.source_bundle_id,
            bundle_lineage_ids=lineage.bundle_ids,
            projection_manifest_id=projection_manifest_id,
            episode_id=episode.episode_id,
            source_node_id=episode.source["node_id"],
            source_execution_revision=episode.terminal_attempt_revision,
            source_evaluation_fingerprint_ids=fingerprint_ids,
            source_score_of_record_fingerprint_id=score_of_record_fingerprint_id,
            episode_reason_codes=selected_case.episode_reason_codes[episode.episode_id],
            task_context_binding_id=context.task_context_binding_id,
            source_expert_base_release_id=environment.expert_base_release_id,
            context_materialization_receipt_id=(
                context_receipt.context_materialization_receipt_id
            ),
            starting_artifact_content_ids=artifact_ids,
            adapter_binding_id=binding_id,
            task_adapter_manifest_id=adapter.manifest.task_adapter_manifest_id,
            verification_receipt_id=(
                adapter.verification_receipt.verification_receipt_id
            ),
            task_adapter_source_tree_hash=adapter.manifest.tree_hash,
            task_evaluator_digest=tree_or_blob_digest(
                adapter.manifest.task_evaluator.to_json_bytes()
            ),
            task_adapter_runtime_digest=tree_or_blob_digest(
                adapter.manifest.runtime.to_json_bytes()
            ),
            task_adapter_context_binding_digest=tree_or_blob_digest(
                adapter.manifest.context_binding.to_json_bytes()
            ),
            task_adapter_dependency_ids=adapter_dependencies,
            compute_binding=compute_binding,
            matched_compute_binding_digest=expert_source_replay_matched_compute_digest(
                bundle_lineage_ids=lineage.bundle_ids,
                projection_manifest_id=projection_manifest_id,
                episode_id=episode.episode_id,
                source_execution_revision=episode.terminal_attempt_revision,
                source_evaluation_fingerprint_ids=fingerprint_ids,
                source_score_of_record_fingerprint_id=(score_of_record_fingerprint_id),
                task_context_binding_id=context.task_context_binding_id,
                context_materialization_receipt_id=(
                    context_receipt.context_materialization_receipt_id
                ),
                starting_artifact_content_ids=artifact_ids,
                task_adapter_manifest_id=(adapter.manifest.task_adapter_manifest_id),
                verification_receipt_id=(
                    adapter.verification_receipt.verification_receipt_id
                ),
                task_adapter_source_tree_hash=adapter.manifest.tree_hash,
                task_evaluator_digest=tree_or_blob_digest(
                    adapter.manifest.task_evaluator.to_json_bytes()
                ),
                task_adapter_runtime_digest=tree_or_blob_digest(
                    adapter.manifest.runtime.to_json_bytes()
                ),
                task_adapter_context_binding_digest=tree_or_blob_digest(
                    adapter.manifest.context_binding.to_json_bytes()
                ),
                compute_binding_id=compute_binding.compute_binding_id,
            ),
            control_leg=control_leg,
            candidate_leg=candidate_leg,
            exact_dependency_ids=tuple(sorted(dependencies)),
        )
        return MaterializedExpertSourceReplayCase(
            request_case=request_case,
            selection_case=selected_case,
            bundle_lineage=lineage,
            episode=episode,
            task_adapter=adapter,
            task_context=verified_context,
        )

    def _validate_context(
        self,
        context,
        environment,
        verified_context: VerifiedSourceReplayContext,
    ) -> None:
        receipt = verified_context.receipt
        artifact_content_ids = {
            artifact.starting_artifact_ref: artifact.starting_artifact_content_id
            for artifact in receipt.starting_artifacts
        }
        if (
            receipt.task_context_binding_id != context.task_context_binding_id
            or receipt.input_contract_fingerprint != context.input_contract_fingerprint
            or receipt.target_contract_fingerprint
            != context.target_contract_fingerprint
            or set(artifact_content_ids) != set(context.starting_artifact_refs)
            or artifact_content_ids != dict(environment.starting_artifact_content_ids)
            or receipt.materializer_id
            != self.settings.policy.source_replay_context_materializer_id
            or receipt.materializer_version
            != self.settings.policy.source_replay_context_materializer_version
        ):
            raise ExpertSourceReplayRequestError(
                "source replay artifact materialization differs from captured authority"
            )

    def _materialization_limits(self) -> TaskEvaluationMaterializationLimits:
        return TaskEvaluationMaterializationLimits(
            maximum_entries=(
                self.settings.policy.task_evaluation_materialization_entry_limit
            ),
            maximum_bytes=(
                self.settings.policy.task_evaluation_materialization_byte_limit
            ),
            timeout_seconds=(
                self.settings.policy.task_evaluation_materialization_timeout_seconds
            ),
        )

    def _check_materialization_totals(
        self,
        *,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent | None,
        adapters: tuple[VerifiedTaskAdapter, ...],
        lineages: tuple[VerifiedRunBundleLineage, ...],
        contexts: tuple[VerifiedSourceReplayContext, ...],
    ) -> None:
        limits = self._materialization_limits()
        entry_count, byte_count = self._materialization_usage(
            candidate=candidate,
            parent=parent,
            adapters=adapters,
            lineages=lineages,
            contexts=contexts,
        )
        if entry_count > limits.maximum_entries or byte_count > limits.maximum_bytes:
            raise ExpertSourceReplayRequestError(
                "source replay byte closure exceeds aggregate materialization limits"
            )

    def _remaining_materialization_limits(
        self,
        *,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent | None,
        adapters: tuple[VerifiedTaskAdapter, ...],
        lineages: tuple[VerifiedRunBundleLineage, ...],
        contexts: tuple[VerifiedSourceReplayContext, ...],
        deadline: float,
    ) -> TaskEvaluationMaterializationLimits:
        limits = self._materialization_limits()
        entry_count, byte_count = self._materialization_usage(
            candidate=candidate,
            parent=parent,
            adapters=adapters,
            lineages=lineages,
            contexts=contexts,
        )
        remaining_entries = limits.maximum_entries - entry_count
        remaining_bytes = limits.maximum_bytes - byte_count
        remaining_seconds = int(deadline - self.monotonic_clock())
        if remaining_entries <= 0 or remaining_bytes <= 0 or remaining_seconds <= 0:
            raise ExpertSourceReplayRequestError(
                "source replay byte closure exhausted materialization budget"
            )
        return TaskEvaluationMaterializationLimits(
            maximum_entries=remaining_entries,
            maximum_bytes=remaining_bytes,
            timeout_seconds=remaining_seconds,
        )

    def _require_deadline(self, deadline: float) -> None:
        if self.monotonic_clock() >= deadline:
            raise ExpertSourceReplayRequestError(
                "source replay materialization deadline expired"
            )

    @staticmethod
    def _materialization_usage(
        *,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent | None,
        adapters: tuple[VerifiedTaskAdapter, ...],
        lineages: tuple[VerifiedRunBundleLineage, ...],
        contexts: tuple[VerifiedSourceReplayContext, ...],
    ) -> tuple[int, int]:
        return _source_replay_materialization_usage(
            candidate=candidate,
            parent=parent,
            adapters=adapters,
            lineages=lineages,
            contexts=contexts,
        )


def _replay_bundle_lineage_from_bytes(
    expected_tip_bundle_id: str,
    supplied: VerifiedRunBundleLineage,
    projector: RunBundleProjector,
) -> VerifiedRunBundleLineage:
    if (
        not isinstance(supplied, VerifiedRunBundleLineage)
        or not supplied.bundles
        or any(not isinstance(bundle, StoredRunBundle) for bundle in supplied.bundles)
        or supplied.bundle_ids[-1] != expected_tip_bundle_id
    ):
        raise ExpertSourceReplayRequestError(
            "bundle provider omitted the exact root-to-tip byte closure"
        )
    projection: ProjectionResult | None = None
    for bundle in supplied.bundles:
        projection = projector.project(bundle, previous=projection)
    if projection is None:
        raise ExpertSourceReplayRequestError("source replay bundle projection is empty")
    return VerifiedRunBundleLineage(
        bundles=supplied.bundles,
        tip_projection=projection,
    )


def _deduplicated_materialized_authorities(
    cases: tuple[MaterializedExpertSourceReplayCase, ...],
) -> tuple[
    tuple[VerifiedTaskAdapter, ...],
    tuple[VerifiedRunBundleLineage, ...],
    tuple[VerifiedSourceReplayContext, ...],
]:
    adapters: dict[str, VerifiedTaskAdapter] = {}
    lineages: dict[str, VerifiedRunBundleLineage] = {}
    contexts: dict[str, VerifiedSourceReplayContext] = {}
    for item in cases:
        authorities = (
            (
                adapters,
                item.task_adapter.verification_receipt.verification_receipt_id,
                item.task_adapter,
            ),
            (
                lineages,
                item.bundle_lineage.bundle_ids[-1],
                item.bundle_lineage,
            ),
            (
                contexts,
                item.task_context.receipt.context_materialization_receipt_id,
                item.task_context,
            ),
        )
        for authority_map, authority_id, authority in authorities:
            existing = authority_map.get(authority_id)
            if existing is not None and existing != authority:
                raise ExpertSourceReplayRequestError(
                    "materialized source replay identity has conflicting closures"
                )
            authority_map[authority_id] = authority
    return (
        tuple(adapters[key] for key in sorted(adapters)),
        tuple(lineages[key] for key in sorted(lineages)),
        tuple(contexts[key] for key in sorted(contexts)),
    )


def _source_replay_materialization_usage(
    *,
    candidate: VerifiedTaskEvaluationCandidate,
    parent: VerifiedTaskEvaluationParent | None,
    adapters: tuple[VerifiedTaskAdapter, ...],
    lineages: tuple[VerifiedRunBundleLineage, ...],
    contexts: tuple[VerifiedSourceReplayContext, ...],
) -> tuple[int, int]:
    adapter_usages = tuple(
        task_adapter_materialization_usage(
            source_file_sizes=tuple(
                descriptor.size
                for descriptor in adapter.source_extraction_receipt.source_tree_files
            ),
            source_archive_sizes=(len(adapter.source_archive),),
            proof_object_sizes=tuple(
                len(payload) for payload in adapter.proof_objects.values()
            ),
            publisher_verification_sizes=(len(adapter.publisher_verification),),
        )
        for adapter in adapters
    )
    adapter_entry_count = sum(usage[0] for usage in adapter_usages)
    adapter_byte_count = sum(usage[1] for usage in adapter_usages)
    bundles = _retained_bundle_closures(lineages)
    bundle_entry_count = sum(len(bundle.artifacts) for bundle in bundles.values())
    bundle_byte_count = sum(
        len(payload)
        for bundle in bundles.values()
        for payload in bundle.artifacts.values()
    )
    starting_artifacts = {}
    for context in contexts:
        for starting_artifact in context.starting_artifacts:
            artifact_id = starting_artifact.artifact.starting_artifact_content_id
            existing = starting_artifacts.get(artifact_id)
            if existing is not None and existing != starting_artifact:
                raise ExpertSourceReplayRequestError(
                    "source replay artifact identity has conflicting byte closures"
                )
            starting_artifacts[artifact_id] = starting_artifact
    context_entry_count = sum(
        len(item.artifact.source_files) for item in starting_artifacts.values()
    )
    context_byte_count = sum(
        descriptor.size
        for item in starting_artifacts.values()
        for descriptor in item.artifact.source_files
    )
    return (
        candidate.entry_count
        + (0 if parent is None else parent.entry_count)
        + adapter_entry_count
        + bundle_entry_count
        + context_entry_count,
        candidate.byte_count
        + (0 if parent is None else parent.byte_count)
        + adapter_byte_count
        + bundle_byte_count
        + context_byte_count,
    )


def _retained_bundle_closures(
    lineages: tuple[VerifiedRunBundleLineage, ...],
) -> dict[str, StoredRunBundle]:
    bundles: dict[str, StoredRunBundle] = {}
    for lineage in lineages:
        for bundle in lineage.bundles:
            if not isinstance(bundle, StoredRunBundle):
                raise ExpertSourceReplayRequestError(
                    "source replay lineage lacks a stored bundle byte closure"
                )
            bundle_id = bundle.manifest.bundle_id
            existing = bundles.get(bundle_id)
            if existing is not None and existing != bundle:
                raise ExpertSourceReplayRequestError(
                    "source replay bundle identity has conflicting byte closures"
                )
            bundles[bundle_id] = bundle
    return bundles


def _source_replay_evaluator(settings: ExpertValidationSettings):
    matches = tuple(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    if len(matches) != 1:
        raise ExpertSourceReplayRequestError(
            "source replay requires one configured evaluator"
        )
    return matches[0]


def _control_leg(
    parent: VerifiedTaskEvaluationParent,
) -> ExpertSourceReplayExecutionLeg:
    receipt = parent.parent_tree_receipt
    return ExpertSourceReplayExecutionLeg.mint(
        kind=ExpertSourceReplayExecutionLegKind.CONTROL_PARENT,
        expert_artifact_id=parent.release_manifest.release_id,
        expert_source_receipt_id=receipt.parent_tree_receipt_id,
        expert_tree_hash=receipt.parent_tree_hash,
        exact_dependency_ids=tuple(
            sorted(
                {
                    parent.release_manifest.release_id,
                    receipt.parent_tree_receipt_id,
                }
            )
        ),
    )


def _candidate_leg(
    candidate: VerifiedTaskEvaluationCandidate,
) -> ExpertSourceReplayExecutionLeg:
    return ExpertSourceReplayExecutionLeg.mint(
        kind=ExpertSourceReplayExecutionLegKind.CANDIDATE,
        expert_artifact_id=candidate.manifest.candidate_id,
        expert_source_receipt_id=candidate.commit_record.commit_record_id,
        expert_tree_hash=candidate.source_tree.tree_hash,
        exact_dependency_ids=tuple(
            sorted(
                {
                    candidate.manifest.candidate_id,
                    candidate.commit_record.commit_record_id,
                }
            )
        ),
    )
