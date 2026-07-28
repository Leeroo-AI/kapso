"""Thin operational composition over the sealed cross-run services."""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import canonical_json_bytes, parse_json_bytes
from kapso.cross_run.capture.bundle import RunBundleStore
from kapso.cross_run.capture.exporter import RunCaptureRequest
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    CandidateChangeKind,
    CompletionState,
    EvaluationFingerprint,
    ExpertEvaluatorResultRecord,
    ExpertScopeContract,
    ExpertValidationStage,
    PublicationArtifactKind,
    TaskContextBinding,
)
from kapso.cross_run.expert.architect import ExpertRepositoryArchitect
from kapso.cross_run.expert.attestation import ConfiguredExpertAttestationVerifier
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.generalizer import ExpertCapabilityGeneralizer
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.publisher import ExpertReleasePublisher
from kapso.cross_run.expert.release import ExpertReleaseAssembler
from kapso.cross_run.expert.release_authority import (
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.expert.release_use_policy import (
    GitHubExpertReleaseUsePolicyAuthority,
)
from kapso.cross_run.expert.revocation import ExpertReleaseRevocationCoordinator
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityCoordinator,
)
from kapso.cross_run.expert.promotion_plan import derive_expert_release_matrix_plan
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
)
from kapso.cross_run.expert.proposal import ExpertCandidateProposalEngine
from kapso.cross_run.expert.store import ExpertCandidateStore
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_docker_bootstrap import (
    build_task_evaluation_docker_provider_registry,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    task_evaluation_execution_schedule,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    TaskEvaluationPreflightCoordinator,
)
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
)
from kapso.cross_run.expert.workspace import ExpertCandidateWorkspaceManager
from kapso.cross_run.expert.review import ExpertAutomatedReviewCoordinator
from kapso.cross_run.expert.review_stage import ExpertAutomatedReviewStageOrchestrator
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertValidationError,
    ExpertValidationReducer,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from kapso.cross_run.expert.validation_snapshots import ExpertValidationSnapshot
from kapso.cross_run.github.command import GitHubCommandClient, SubprocessCommandRunner
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import AutonomousGitHubPublisher
from kapso.cross_run.github.resolver import GitHubArtifactResolver
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.handoff import prepare_fresh_run_handoff
from kapso.cross_run.launch.production import (
    build_production_launch_preparation,
    build_production_launch_services,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
)
from kapso.cross_run.task_adapter_authority import CanonicalTaskAdapterAuthority
from kapso.cross_run.task_adapter_store import (
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentRunnerSettings,
    SubprocessCodingAgentCallRunner,
)


class CrossRunOperationError(ValueError):
    """An operational request is incomplete or crosses configured authority."""


_RETRIEVAL_POLICY_VERSION = "kapso.retrieval.v1"


@dataclass(frozen=True)
class GitHubOperationServices:
    """The existing GitHub resolver, materializer, and publication authority."""

    resolver: GitHubArtifactResolver
    materializer: GitHubArtifactMaterializer
    publisher: AutonomousGitHubPublisher


@dataclass(frozen=True)
class ExpertValidationOperationServices:
    """One local candidate/adapter/validation authority composition."""

    candidate_store: ExpertCandidateStore
    validation_store: ExpertValidationStore
    task_adapter_store: TaskAdapterPackageStore


@dataclass(frozen=True)
class CrossRunPolicyOperationServices:
    """Fresh external authorities shared by promotion and publication."""

    security_authority: AuthenticatedSecurityDenylistAuthority
    release_use_authority: GitHubExpertReleaseUsePolicyAuthority


def inspect_cross_run(
    *,
    config_path: str,
    mode: str,
    scope_id: str,
    state_root: Path,
) -> Mapping[str, Any]:
    """Resolve every configured current release without materializing it."""

    settings = _settings(config_path, mode)
    services = _github_services(settings, state_root)
    artifacts = {
        kind.value: _resolved_artifact_summary(
            services.resolver.resolve_current(scope_id, kind)
        )
        for kind in _artifact_kinds()
    }
    repositories = settings.scopes.resolve(scope_id)
    return {
        "operation": "inspect",
        "scope_id": scope_id,
        "repository_binding_fingerprint": repositories.binding_fingerprint,
        "repositories": repositories.to_dict(),
        "artifacts": artifacts,
        "next_action": "verify",
    }


def verify_cross_run(
    *,
    config_path: str,
    mode: str,
    scope_id: str,
    state_root: Path,
) -> Mapping[str, Any]:
    """Resolve and fully materialize each configured current release."""

    settings = _settings(config_path, mode)
    services = _github_services(settings, state_root)
    artifacts = {}
    for kind in _artifact_kinds():
        resolved = services.resolver.resolve_current(scope_id, kind)
        materialized = services.materializer.materialize(resolved)
        artifacts[kind.value] = {
            **_resolved_artifact_summary(resolved),
            "cache_tree_digest": materialized.receipt.cache_tree_digest,
            "materialized_tree_digest": (materialized.receipt.materialized_tree_digest),
            "manifest_digest": materialized.receipt.manifest_digest,
            "asset_digests": dict(materialized.receipt.asset_digests),
            "cache_reused": materialized.reused,
        }
    return {
        "operation": "verify",
        "scope_id": scope_id,
        "artifacts": artifacts,
        "next_action": "resolve-launch",
    }


def capture_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Capture one complete stopped frontier from an explicit typed request."""

    settings = _settings(config_path, mode)
    request = _capture_request(request_path)
    stored = RunCapturePipeline(
        RunCaptureContext(request),
        settings,
    ).capture_if_due(CompletionState.STOPPED, force=True)
    if stored is None:
        raise CrossRunOperationError("forced capture returned no bundle")
    root = _private_state_root(state_root)
    RunBundleStore.initialize(
        root / settings.capture.state_path,
        settings.capture,
        settings.sanitation,
    ).import_exact(stored)
    return {
        "operation": "capture",
        "scope_id": stored.manifest.scope_id,
        "run_id": stored.manifest.run_id,
        "campaign_id": stored.manifest.campaign_id,
        "bundle_id": stored.manifest.bundle_id,
        "capture_generation": stored.manifest.capture_generation,
        "completion_state": stored.manifest.completion_state.value,
        "artifact_digests": dict(stored.manifest.checksums),
        "next_action": "publish-knowledge",
    }


def publish_knowledge_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Build and publish S(n+1) from one explicit catalog and parent."""

    settings = _settings(config_path, mode)
    request = _object_request(
        request_path,
        {
            "catalog_root",
            "scope_contract",
            "expected_parent_sha",
            "expected_current_snapshot_id",
            "committed_at",
            "validation_closure_ids",
        },
    )
    scope_contract = ExpertScopeContract.from_dict(request["scope_contract"])
    catalog_root = _request_path(request_path, request["catalog_root"])
    catalog = CrossRunCatalog(catalog_root, scope_contract, settings.catalog)
    generation = catalog.store.read_current()
    parent_id = request["expected_current_snapshot_id"]
    if parent_id is not None and not isinstance(parent_id, str):
        raise CrossRunOperationError("expected_current_snapshot_id is invalid")
    validation_ids = request["validation_closure_ids"]
    if not isinstance(validation_ids, list) or any(
        not isinstance(item, str) for item in validation_ids
    ):
        raise CrossRunOperationError("validation_closure_ids are invalid")
    services = _github_services(settings, state_root)
    publisher = KnowledgeSnapshotPublisher(
        services.publisher,
        settings.github,
        settings.knowledge,
    )
    parent_ids = () if parent_id is None else (parent_id,)
    built = publisher.build(
        scope_contract,
        generation,
        catalog.store.read_object_bytes,
        parent_snapshot_ids=parent_ids,
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version=_RETRIEVAL_POLICY_VERSION,
        published_at=_required_text(request["committed_at"], "committed_at"),
        publisher_attestation={"issuer": settings.github.publisher_login},
    )
    publication = publisher.publish(
        built.package,
        expected_parent_sha=_required_text(
            request["expected_parent_sha"], "expected_parent_sha"
        ),
        expected_current_snapshot_id=parent_id,
        committed_at=_required_text(request["committed_at"], "committed_at"),
        validation_closure_ids=tuple(sorted(validation_ids)),
    )
    record = publication.telemetry.publication_record
    telemetry = built.embedding_telemetry
    return {
        "operation": "publish-knowledge",
        "scope_id": scope_contract.scope_id,
        "snapshot_id": publication.package.manifest.snapshot_id,
        "catalog_generation_id": generation.catalog_generation_id,
        "catalog_generation": generation.generation_number,
        "commit_sha": record.commit_sha,
        "release_tag": record.tag,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        "embedding": (
            None
            if telemetry is None
            else {
                "provider": telemetry.provider,
                "model": telemetry.model,
                "call_count": telemetry.call_count,
                "input_tokens": telemetry.input_tokens,
            }
        ),
        "next_action": "propose-expert",
    }


def propose_expert_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Run the configured coding-agent proposer and seal one candidate."""

    settings = _settings(config_path, mode)
    request = _object_request(request_path, {"evidence_packet"})
    packet = ExpertTriggerEvidencePacket.from_dict(request["evidence_packet"])
    decision = ExpertTriggerEvaluator(settings.expert.triggers).evaluate(packet)
    if not decision.candidate_required or decision.change_kind is None:
        raise CrossRunOperationError("expert trigger does not require a candidate")
    root = _private_state_root(state_root)
    github = _github_services(settings, root)
    materialized_source = None
    if packet.source_base_release_id is not None:
        resolved = github.resolver.resolve_artifact(
            packet.scope_contract.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            packet.source_base_release_id,
        )
        materialized_source = github.materializer.materialize(resolved)
    expert_state_root = _expert_state_root(settings, root)
    validator = ExpertCandidateValidator(settings.expert, settings.sanitation)
    candidate_store = ExpertCandidateStore(
        expert_state_root / Path(settings.expert.candidate_path).name,
        expert_state_root,
        validator,
    )
    workspace_manager = ExpertCandidateWorkspaceManager(
        expert_state_root / Path(settings.expert.workspace_path).name,
        expert_state_root,
        settings.expert,
        github.materializer,
    )
    engine = ExpertCandidateProposalEngine(
        settings=settings.expert,
        runner=_coding_agent_runner(settings, root),
        workspace_manager=workspace_manager,
        candidate_store=candidate_store,
    )
    if decision.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE:
        proposal = ExpertRepositoryArchitect(engine).propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized_source,
        )
    else:
        if materialized_source is None:
            raise CrossRunOperationError(
                "capability proposal requires a materialized source release"
            )
        proposal = ExpertCapabilityGeneralizer(engine).propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized_source,
        )
    closure = proposal.stored_candidate.closure
    operation = closure.derivation.operation
    return {
        "operation": "propose-expert",
        "scope_id": closure.validation_context.scope_contract.scope_id,
        "candidate_id": closure.manifest.candidate_id,
        "candidate_tree_hash": closure.manifest.candidate_tree_hash,
        "source_base_release_id": closure.manifest.source_base_release_id,
        "trigger_decision_id": decision.trigger_decision_id,
        "proposal_operation_id": operation.operation_receipt.operation_id,
        "change_kind": decision.change_kind.value,
        "next_action": "validate-expert",
    }


def resolve_launch_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
    run_root: Path,
) -> Mapping[str, Any]:
    """Resolve, materialize, and pin one fresh launch without paid work."""

    effective = load_effective_config(config_path, mode)
    settings = effective.cross_run
    if type(settings) is not CrossRunSettings:
        raise CrossRunOperationError("selected configuration has no cross-run settings")
    request = _object_request(
        request_path,
        {
            "goal",
            "additional_context",
            "task_context_request",
            "starting_artifacts",
            "dependency_runtime_contract",
            "budget_fidelity_envelope",
            "scope_id",
            "task_family_id",
            "task_adapter_id",
            "requested_coding_agent",
            "objective_direction",
            "empty_scope_bootstrap_authorization_id",
        },
    )
    starting_artifacts = request["starting_artifacts"]
    if not isinstance(starting_artifacts, Mapping):
        raise CrossRunOperationError("starting_artifacts must be an object")
    sources = {}
    for artifact_ref, artifact in starting_artifacts.items():
        if (
            not isinstance(artifact_ref, str)
            or not isinstance(artifact, Mapping)
            or set(artifact) != {"source", "mount_path"}
        ):
            raise CrossRunOperationError("starting artifact input is invalid")
        sources[artifact_ref] = (
            _request_path(request_path, artifact["source"]),
            _required_text(artifact["mount_path"], "starting artifact mount_path"),
        )
    dependency_runtime = request["dependency_runtime_contract"]
    budget_fidelity = request["budget_fidelity_envelope"]
    if not isinstance(dependency_runtime, Mapping) or not dependency_runtime:
        raise CrossRunOperationError("dependency_runtime_contract is invalid")
    if not isinstance(budget_fidelity, Mapping) or not budget_fidelity:
        raise CrossRunOperationError("budget_fidelity_envelope is invalid")
    preparation = build_production_launch_preparation(
        effective_config=effective,
        goal=_required_text(request["goal"], "goal"),
        additional_context=_required_string(
            request["additional_context"], "additional_context"
        ),
        task_context_request=LaunchTaskContextRequest.from_dict(
            request["task_context_request"]
        ),
        starting_artifact_sources=sources,
        dependency_runtime_contract=dependency_runtime,
        budget_fidelity_envelope=budget_fidelity,
        scope_id=_optional_text(request["scope_id"], "scope_id"),
        task_family_id=_optional_text(request["task_family_id"], "task_family_id"),
        task_adapter_id=_optional_text(request["task_adapter_id"], "task_adapter_id"),
        requested_coding_agent=_optional_text(
            request["requested_coding_agent"], "requested_coding_agent"
        ),
        empty_scope_bootstrap_authorization_id=_optional_text(
            request["empty_scope_bootstrap_authorization_id"],
            "empty_scope_bootstrap_authorization_id",
        ),
    )
    root = _private_state_root(state_root)
    services = build_production_launch_services(
        settings=settings,
        binding=preparation.binding,
        experiment_embedding_space=preparation.experiment_embedding_space,
        starting_artifacts=preparation.starting_artifacts,
        state_root=root,
    )
    handoff = prepare_fresh_run_handoff(
        coordinator=services.coordinator,
        settings=settings,
        security_authority=services.security_authority,
        request=preparation.request,
        run_root=Path(os.path.abspath(run_root)),
        objective_direction=_required_text(
            request["objective_direction"], "objective_direction"
        ),
    )
    replay_context = RunBundleStore.initialize(
        root / settings.capture.state_path,
        settings.capture,
        settings.sanitation,
    ).publish_starting_artifacts(
        task_context_binding=(
            handoff.active_workspace.bootstrap_pin.launch_manifest.task_context_binding
        ),
        launch_artifacts=preparation.starting_artifacts.artifacts,
        validation_settings=settings.expert.validation,
    )
    identity = handoff.identity
    baseline_commit = (
        handoff.active_workspace.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha
    )
    handoff.close()
    return {
        "operation": "resolve-launch",
        "run_id": identity.run_id,
        "campaign_id": identity.campaign_id,
        "scope_id": identity.scope_id,
        "task_family_id": identity.task_family_id,
        "task_adapter_id": identity.task_adapter_id,
        "launch_manifest_id": identity.launch_manifest_id,
        "bootstrap_pin_id": identity.bootstrap_pin_id,
        "expert_release_id": identity.expert_release_id,
        "knowledge_snapshot_id": identity.knowledge_snapshot_id,
        "task_adapter_manifest_id": identity.task_adapter_manifest_id,
        "source_replay_starting_artifact_content_ids": [
            item.artifact.starting_artifact_content_id
            for item in replay_context.starting_artifacts
        ],
        "workspace_baseline_commit_sha": baseline_commit,
        "next_action": "evolve --resume",
    }


def validate_expert_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Enroll or advance exactly one restart-safe expert validation stage."""

    settings = _settings(config_path, mode)
    request = _object_request(
        request_path,
        {"candidate_id", "expected_transition_id", "evaluator_result"},
    )
    candidate_id = _required_text(request["candidate_id"], "candidate_id")
    expected_transition_id = _optional_text(
        request["expected_transition_id"], "expected_transition_id"
    )
    root = _private_state_root(state_root)
    github = _github_services(settings, root)
    services = _expert_validation_services(settings, root, github)
    snapshot = services.validation_store.snapshot(candidate_id)
    if snapshot is None:
        if (
            expected_transition_id is not None
            or request["evaluator_result"] is not None
        ):
            raise CrossRunOperationError(
                "validation enrollment cannot accept a prior transition or result"
            )
        eligibility = ExpertCandidateEligibilityEvaluator(
            settings.expert.validation,
            services.candidate_store,
            services.task_adapter_store,
            GitHubExpertCurrentReleaseProvider(github.resolver),
        ).decide(candidate_id=candidate_id)
        snapshot = services.validation_store.publish_start(
            expected_transition_id=None,
            eligibility=eligibility,
        ).snapshot
    else:
        if expected_transition_id != snapshot.transition.transition_id:
            raise CrossRunOperationError(
                "expected validation transition is not current"
            )
        stage = snapshot.state.next_stage
        if stage is ExpertValidationStage.AUTOMATED_REVIEW:
            if request["evaluator_result"] is not None:
                raise CrossRunOperationError(
                    "automated review cannot consume a generic evaluator result"
                )
            coordinator = ExpertAutomatedReviewCoordinator(settings.expert, root)
            snapshot = ExpertAutomatedReviewStageOrchestrator(
                coordinator=coordinator,
                candidate_store=services.candidate_store,
                validation_store=services.validation_store,
            ).run(snapshot.latest_attempt)
        elif stage is ExpertValidationStage.PUBLICATION_ELIGIBILITY:
            if request["evaluator_result"] is not None:
                raise CrossRunOperationError(
                    "publication eligibility cannot consume a generic evaluator result"
                )
            policies = _policy_services(settings, root, github)
            coordinator = _publication_eligibility_coordinator(services, policies)
            matrix_result_id = _accepted_stage_result_id(
                snapshot,
                ExpertValidationStage.RELEASE_MATRIX,
            )
            snapshot = coordinator.publish(
                candidate_id=candidate_id,
                release_matrix_stage_result_id=matrix_result_id,
            ).snapshot
        elif stage is ExpertValidationStage.RELEASE_MATRIX:
            if request["evaluator_result"] is not None:
                raise CrossRunOperationError(
                    "typed validation stage cannot consume a generic evaluator result"
                )
            snapshot = _execute_release_matrix_stage(
                settings=settings,
                state_root=root,
                github=github,
                services=services,
                snapshot=snapshot,
            )
        elif stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            if request["evaluator_result"] is not None:
                raise CrossRunOperationError(
                    "typed validation stage cannot consume a generic evaluator result"
                )
        else:
            evaluator_result = request["evaluator_result"]
            if not isinstance(evaluator_result, Mapping):
                raise CrossRunOperationError(
                    "current validation stage requires a signed evaluator result"
                )
            snapshot = services.validation_store.publish_evaluator_result(
                candidate_id=candidate_id,
                expected_transition_id=expected_transition_id,
                result=ExpertEvaluatorResultRecord.from_dict(evaluator_result),
            ).snapshot
    return _validation_summary(snapshot)


def _execute_release_matrix_stage(
    *,
    settings: CrossRunSettings,
    state_root: Path,
    github: GitHubOperationServices,
    services: ExpertValidationOperationServices,
    snapshot: ExpertValidationSnapshot,
) -> ExpertValidationSnapshot:
    attempt = snapshot.latest_attempt
    if (
        snapshot.state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
        or attempt is None
    ):
        raise CrossRunOperationError(
            "release matrix execution requires the active typed stage"
        )
    stored_candidate = services.candidate_store.read(attempt.candidate_id)
    verified_adapters = tuple(
        services.task_adapter_store.resolve_exact(
            task_adapter_manifest_id=pin.task_adapter_manifest_id,
            verification_receipt_id=pin.verification_receipt_id,
        )
        for pin in attempt.task_adapter_pins
    )
    prepared_plan = derive_expert_release_matrix_plan(
        state=snapshot.state,
        attempt=attempt,
        accepted_stage_results=snapshot.accepted_stage_results,
        source_replay_request=None,
        stored_candidate=stored_candidate,
        verified_adapters=verified_adapters,
        validation_policy=settings.expert.validation.policy.validation_policy(),
        validation_settings=settings.expert.validation,
    )
    plan_reservation = services.validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    current_release = services.validation_store.reducer.current_release_provider
    if type(current_release) is not GitHubExpertCurrentReleaseProvider:
        raise CrossRunOperationError(
            "release matrix requires the configured GitHub CURRENT authority"
        )
    prepared_request = TaskEvaluationPreflightCoordinator(
        settings=settings.expert.validation,
        plan_reservation_authority=services.validation_store,
        candidate_reader=services.candidate_store,
        source_base_provider=None,
        adapter_provider=services.task_adapter_store,
        current_release_authority=current_release,
        monotonic_clock=time.monotonic,
    ).build(plan_reservation)
    task_reservation = services.validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared_request,
    ).reservation
    execution_store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(
            services.validation_store.root
        ).resolve(),
        services.validation_store.root,
        settings.expert.validation.policy,
    )
    provider_registry = build_task_evaluation_docker_provider_registry(
        prepared_request=prepared_request,
        workspace_root=state_root,
    )
    authority = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=services.validation_store,
        execution_store=execution_store,
        current_release_authority=current_release,
        task_adapter_authority=services.task_adapter_store,
        security_denylist_authority=(
            _policy_services(settings, state_root, github).security_authority
        ),
    )
    schedule = task_evaluation_execution_schedule(
        task_reservation,
        prepared_request,
    )
    with execution_store.reservation_session(
        reservation_snapshot=task_reservation,
        prepared_request=prepared_request,
    ) as session:
        while len(session.events) < 4 * len(schedule):
            phase = len(session.events) % 4
            if phase == 2:
                handle = session.cleanup_interrupted_spawn(provider_registry)
                raise CrossRunOperationError(
                    "task evaluation invocation is permanently interrupted after "
                    f"spawn commit: {handle.provider_handle_id}"
                )
            if phase == 3:
                session.accept_received_result()
                continue
            allocation = session.allocate_expected_leg()
            spawn = authority.commit_spawn(
                prepared_request=prepared_request,
                reservation_id=(task_reservation.reservation.reservation_id),
                invocation_permit=allocation,
                provider_registry=provider_registry,
            )
            session.record_result_received(spawn.execute())
            session.accept_received_result()
        completed = session.completed_execution()
    return (
        ExpertReleaseMatrixStageCoordinator(
            validation_store=services.validation_store,
            execution_store=execution_store,
        )
        .publish_completed(
            completed_execution=completed,
            reservation_snapshot=task_reservation,
            prepared_request=prepared_request,
        )
        .snapshot
    )


def publish_expert_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Terminalize eligibility when needed and publish one approved release."""

    settings = _settings(config_path, mode)
    request = _object_request(request_path, {"candidate_id", "committed_at"})
    candidate_id = _required_text(request["candidate_id"], "candidate_id")
    committed_at = _required_text(request["committed_at"], "committed_at")
    root = _private_state_root(state_root)
    github = _github_services(settings, root)
    services = _expert_validation_services(settings, root, github)
    policies = _policy_services(settings, root, github)
    coordinator = _publication_eligibility_coordinator(services, policies)
    snapshot = services.validation_store.snapshot(candidate_id)
    if snapshot is None:
        raise CrossRunOperationError("expert candidate has no validation state")
    if snapshot.state.next_stage is ExpertValidationStage.PUBLICATION_ELIGIBILITY:
        snapshot = coordinator.publish(
            candidate_id=candidate_id,
            release_matrix_stage_result_id=_accepted_stage_result_id(
                snapshot,
                ExpertValidationStage.RELEASE_MATRIX,
            ),
        ).snapshot
    publisher = ExpertReleasePublisher(
        assembler=ExpertReleaseAssembler(
            candidate_store=services.candidate_store,
            validation_store=services.validation_store,
            expert_settings=settings.expert,
            github_settings=settings.github,
        ),
        validation_store=services.validation_store,
        github_publisher=github.publisher,
        resolver=github.resolver,
        current_release_authority=(
            services.validation_store.reducer.current_release_provider
        ),
        task_adapter_authority=(
            services.validation_store.reducer.task_adapter_provider
        ),
        security_denylist_authority=policies.security_authority,
        release_use_policy_authority=policies.release_use_authority,
    )
    publication = publisher.publish(
        candidate_id=candidate_id,
        committed_at=committed_at,
    )
    receipt = publication.activation.receipt
    record = receipt.github_publication_pointer.publication_record
    return {
        "operation": "publish-expert",
        "scope_id": receipt.github_publication_intent.scope_id,
        "candidate_id": receipt.candidate_id,
        "release_id": receipt.release_id,
        "activation_receipt_id": receipt.activation_receipt_id,
        "publication_id": record.publication_id,
        "commit_sha": record.commit_sha,
        "release_tag": record.tag,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        "replayed": publication.activation.replayed,
        "next_action": "resolve-launch",
    }


def revoke_expert_cross_run(
    *,
    config_path: str,
    mode: str,
    request_path: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Record a release revocation already authorized by security CURRENT."""

    settings = _settings(config_path, mode)
    request = _object_request(request_path, {"candidate_id", "revoked_at"})
    candidate_id = _required_text(request["candidate_id"], "candidate_id")
    root = _private_state_root(state_root)
    github = _github_services(settings, root)
    services = _expert_validation_services(settings, root, github)
    policies = _policy_services(settings, root, github)
    revoked = ExpertReleaseRevocationCoordinator(
        validation_store=services.validation_store,
        security_denylist_authority=policies.security_authority,
    ).revoke(
        candidate_id=candidate_id,
        revoked_at=_required_text(request["revoked_at"], "revoked_at"),
    )
    receipt = revoked.receipt
    observation = receipt.security_denylist_observation
    return {
        "operation": "revoke",
        "scope_id": observation.scope_id,
        "candidate_id": receipt.candidate_id,
        "release_id": receipt.release_id,
        "revocation_receipt_id": receipt.revocation_receipt_id,
        "security_snapshot_id": observation.snapshot_id,
        "security_publication_id": observation.publication_id,
        "matched_revocation_ids": tuple(
            item.revocation_id for item in observation.matched_revocations
        ),
        "replayed": revoked.replayed,
        "next_action": "resolve-launch",
    }


def operation_json(result: Mapping[str, Any]) -> bytes:
    """Render one canonical non-secret operational result."""

    if not isinstance(result, Mapping):
        raise CrossRunOperationError("operation result must be an object")
    return canonical_json_bytes(dict(result)) + b"\n"


def _settings(config_path: str, mode: str) -> CrossRunSettings:
    effective = load_effective_config(config_path, mode)
    if type(effective.cross_run) is not CrossRunSettings:
        raise CrossRunOperationError("selected configuration has no cross-run settings")
    return effective.cross_run


def _github_services(
    settings: CrossRunSettings,
    state_root: Path,
) -> GitHubOperationServices:
    root = _private_state_root(state_root)
    client = GitHubCommandClient(
        SubprocessCommandRunner(),
        working_directory=root,
        timeout_seconds=settings.github.command_timeout_seconds,
        api_version=settings.github.api_version,
        minimum_cli_version=settings.github.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.github.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.github.control_blob_size_bytes,
    )
    resolver = GitHubArtifactResolver(client, settings.github, settings.scopes)
    materializer = GitHubArtifactMaterializer(client, settings.github, root)
    return GitHubOperationServices(
        resolver=resolver,
        materializer=materializer,
        publisher=AutonomousGitHubPublisher(
            client,
            resolver,
            materializer,
            settings.github,
        ),
    )


def _coding_agent_runner(
    settings: CrossRunSettings,
    state_root: Path,
) -> SubprocessCodingAgentCallRunner:
    return SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str(state_root / settings.expert.agent_artifact_path),
            termination_grace_seconds=settings.expert.termination_grace_seconds,
            sensitive_file_glob_scan_max_depth=(
                settings.expert.sensitive_file_glob_scan_max_depth
            ),
        )
    )


class _BoundValidationStateProvider:
    def __init__(self) -> None:
        self.store: ExpertValidationStore | None = None

    def bind(self, store: ExpertValidationStore) -> None:
        if self.store is not None or type(store) is not ExpertValidationStore:
            raise ExpertValidationError("validation state provider is already bound")
        self.store = store

    def current(self, candidate_id: str):
        if self.store is None:
            raise ExpertValidationError("validation state provider is unbound")
        return self.store.current(candidate_id)


def _expert_validation_services(
    settings: CrossRunSettings,
    state_root: Path,
    github: GitHubOperationServices,
) -> ExpertValidationOperationServices:
    expert_root = _expert_state_root(settings, state_root)
    candidate_store = ExpertCandidateStore(
        expert_root / Path(settings.expert.candidate_path).name,
        expert_root,
        ExpertCandidateValidator(settings.expert, settings.sanitation),
    )
    adapter_settings = settings.expert.task_adapters
    task_adapter_store = TaskAdapterPackageStore(
        state_root / adapter_settings.state_path,
        state_root,
        adapter_settings,
        TaskAdapterAuthorityRegistry(
            adapter_settings,
            tuple(
                CanonicalTaskAdapterAuthority(authority)
                for authority in adapter_settings.trusted_authorities
            ),
        ),
    )
    current_release = GitHubExpertCurrentReleaseProvider(github.resolver)
    state_provider = _BoundValidationStateProvider()
    reducer = ExpertValidationReducer(
        settings.expert.validation,
        candidate_store,
        ConfiguredExpertAttestationVerifier(settings.expert.validation),
        task_adapter_store,
        current_release,
        state_provider,
    )
    validation_store = ExpertValidationStore(
        expert_root / Path(settings.expert.validation.state_path).name,
        expert_root,
        settings.expert.validation,
        reducer,
    )
    state_provider.bind(validation_store)
    return ExpertValidationOperationServices(
        candidate_store=candidate_store,
        validation_store=validation_store,
        task_adapter_store=task_adapter_store,
    )


def _policy_services(
    settings: CrossRunSettings,
    state_root: Path,
    github: GitHubOperationServices,
) -> CrossRunPolicyOperationServices:
    security_state_path = state_root / settings.launch.security_denylist_state_path
    security_trusted_root = _private_state_root(security_state_path.parent)
    security_authority = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        settings.launch,
        GitHubSecurityDenylistSnapshotProvider(
            github.resolver,
            github.materializer,
        ),
        SecurityDenylistCheckpointStore(
            security_state_path,
            security_trusted_root,
            settings.launch.security_denylist_checkpoint_size_bytes,
        ),
    )
    activation_provider = GitHubExpertReleaseActivationProvider(
        github.resolver,
        github.materializer,
    )
    return CrossRunPolicyOperationServices(
        security_authority=security_authority,
        release_use_authority=GitHubExpertReleaseUsePolicyAuthority(
            github.resolver,
            github.materializer,
            activation_provider,
        ),
    )


def _publication_eligibility_coordinator(
    services: ExpertValidationOperationServices,
    policies: CrossRunPolicyOperationServices,
) -> ExpertPublicationEligibilityCoordinator:
    current_release = services.validation_store.reducer.current_release_provider
    if type(current_release) is not GitHubExpertCurrentReleaseProvider:
        raise CrossRunOperationError(
            "publication eligibility requires the configured GitHub CURRENT authority"
        )
    return ExpertPublicationEligibilityCoordinator(
        validation_store=services.validation_store,
        current_release_authority=current_release,
        task_adapter_authority=(
            services.validation_store.reducer.task_adapter_provider
        ),
        security_denylist_authority=policies.security_authority,
        release_use_policy_authority=policies.release_use_authority,
    )


def _accepted_stage_result_id(snapshot, stage: ExpertValidationStage) -> str:
    matches = tuple(
        reference.stage_result_record_id
        for reference in snapshot.state.accepted_stage_results
        if reference.stage is stage
    )
    if len(matches) != 1:
        raise CrossRunOperationError(
            f"validation state has no unique accepted {stage.value} result"
        )
    return matches[0]


def _validation_summary(snapshot) -> Mapping[str, Any]:
    next_stage = snapshot.state.next_stage
    return {
        "operation": "validate-expert",
        "candidate_id": snapshot.state.candidate_id,
        "validation_attempt_id": snapshot.state.validation_attempt_id,
        "transition_id": snapshot.transition.transition_id,
        "validation_state_id": snapshot.state.validation_state_id,
        "promotion_state": snapshot.state.promotion_state.value,
        "accepted_stage_result_ids": tuple(
            reference.stage_result_record_id
            for reference in snapshot.state.accepted_stage_results
        ),
        "next_stage": None if next_stage is None else next_stage.value,
        "next_action": (
            "publish-expert"
            if snapshot.state.promotion_state.value == "approved"
            else "validate-expert"
        ),
    }


def _expert_state_root(settings: CrossRunSettings, state_root: Path) -> Path:
    candidate_parent = Path(settings.expert.candidate_path).parent
    if candidate_parent != Path(settings.expert.workspace_path).parent:
        raise CrossRunOperationError("expert state paths do not share one parent")
    return _private_state_root(state_root / candidate_parent)


def _private_state_root(state_root: Path) -> Path:
    if not isinstance(state_root, Path):
        raise CrossRunOperationError("state_root must be a path")
    root = Path(os.path.abspath(state_root))
    if root in {Path("/"), Path.home()}:
        raise CrossRunOperationError("state_root is unsafe")
    if os.path.lexists(root):
        metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise CrossRunOperationError("state_root must be a real directory")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CrossRunOperationError("state_root must be owner-private")
    else:
        root.mkdir(parents=True, mode=0o700)
    return root


def _artifact_kinds() -> tuple[PublicationArtifactKind, ...]:
    return (
        PublicationArtifactKind.EXPERT_BASE_RELEASE,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        PublicationArtifactKind.SECURITY_DENYLIST,
    )


def _resolved_artifact_summary(resolved) -> Mapping[str, Any]:
    record = resolved.pointer.publication_record
    return {
        "artifact_id": record.artifact_id,
        "repository": record.repository_full_name,
        "repository_node_id": record.repository_node_id,
        "pointer_commit_sha": resolved.pointer_commit_sha,
        "source_commit_sha": record.commit_sha,
        "release_tag": record.tag,
        "release_id": record.immutable_release_id,
        "release_attestation_ref": record.release_attestation_ref,
        "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
    }


def _object_request(path: Path, expected_fields: set[str]) -> Mapping[str, Any]:
    normalized = path.expanduser().resolve(strict=True)
    parsed = parse_json_bytes(normalized.read_bytes())
    if not isinstance(parsed, Mapping) or set(parsed) != expected_fields:
        raise CrossRunOperationError("operation request fields are invalid")
    return parsed


def _capture_request(path: Path) -> RunCaptureRequest:
    request = _object_request(
        path,
        {
            "workspace_dir",
            "idea_archive_path",
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "configuration_fingerprint",
            "completion_state",
            "started_at",
            "kapso_commit",
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
            "task_context_binding",
            "artifact_environment",
            "evaluation_fingerprints",
            "run_log_paths",
        },
    )
    fingerprints = request["evaluation_fingerprints"]
    logs = request["run_log_paths"]
    if not isinstance(fingerprints, list) or not isinstance(logs, list):
        raise CrossRunOperationError("capture request arrays are invalid")
    completion = request["completion_state"]
    if completion != CompletionState.STOPPED.value:
        raise CrossRunOperationError("operational capture requires stopped state")
    return RunCaptureRequest(
        workspace_dir=_request_path(path, request["workspace_dir"]),
        idea_archive_path=_request_path(path, request["idea_archive_path"]),
        scope_contract_id=_required_text(
            request["scope_contract_id"], "scope_contract_id"
        ),
        scope_id=_required_text(request["scope_id"], "scope_id"),
        run_id=_required_text(request["run_id"], "run_id"),
        campaign_id=_required_text(request["campaign_id"], "campaign_id"),
        configuration_fingerprint=_required_text(
            request["configuration_fingerprint"], "configuration_fingerprint"
        ),
        completion_state=CompletionState(completion),
        started_at=_required_text(request["started_at"], "started_at"),
        kapso_commit=_required_text(request["kapso_commit"], "kapso_commit"),
        launch_manifest_id=_required_text(
            request["launch_manifest_id"], "launch_manifest_id"
        ),
        knowledge_snapshot_id=_required_text(
            request["knowledge_snapshot_id"], "knowledge_snapshot_id"
        ),
        expert_base_release_id=_required_text(
            request["expert_base_release_id"], "expert_base_release_id"
        ),
        task_context_binding=TaskContextBinding.from_dict(
            request["task_context_binding"]
        ),
        artifact_environment=ArtifactEnvironment.from_dict(
            request["artifact_environment"]
        ),
        evaluation_fingerprints=tuple(
            EvaluationFingerprint.from_dict(item) for item in fingerprints
        ),
        run_log_paths=tuple(_required_text(item, "run_log_path") for item in logs),
    )


def _request_path(request_path: Path, value: object) -> Path:
    text = _required_text(value, "request path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = request_path.expanduser().resolve(strict=True).parent / candidate
    return candidate.resolve(strict=True)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CrossRunOperationError(f"{name} must be non-empty text")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CrossRunOperationError(f"{name} must be text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)
