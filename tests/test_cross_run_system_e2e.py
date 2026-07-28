"""One deterministic cross-module scenario for both supported task shapes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.capture.bundle import (
    RunBundleStore,
    StoredSourceReplayContextProvider,
)
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.lineage import RunBundleLineageProvider
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    CompletionState,
    CrossRunTaskBindingSettings,
)
from kapso.cross_run.expert.release import ExpertReleaseAssembler
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.triggers import (
    ExpertSourceBaseTreeReceipt,
    ExpertTriggerEvidencePacket,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.knowledge.retrieval import (
    CrossRunRetriever,
    PriorKnowledgeQuery,
)
from kapso.cross_run.launch.bootstrap import LaunchBootstrapCoordinator
from kapso.cross_run.launch.handoff import (
    prepare_fresh_run_handoff,
    prepare_resumed_run_handoff,
)
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from kapso.cross_run.settings import CrossRunSettings
from cross_run_capture_fixtures import make_capture_fixture
from test_expert_release_assembly import _approved_normal
from test_knowledge_snapshot_publisher import (
    DeterministicEmbeddingProvider,
    RecordingPublicationAuthority,
)
from test_launch_bootstrap import _fresh_coordinator
from test_launch_handoff import _DescendantSecurityAuthority
from test_launch_resolver import build_resolver_case
from test_expert_triggers import (
    configuration_fingerprint,
    inspection_operation,
    trigger_packet,
    trigger_settings,
)
from test_run_resume_coordinator import _coordinator

_CONFIG_PATH = "src/kapso/config.yaml"
_PUBLISHED_AT = "2026-07-27T12:00:00Z"


@pytest.mark.parametrize(
    "binding",
    (
        CrossRunTaskBindingSettings(
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            task_adapter_id="posttrain",
        ),
        CrossRunTaskBindingSettings(
            scope_id="ml_ai",
            task_family_id="relational_tabular_prediction",
            task_adapter_id="relbench",
        ),
    ),
    ids=("posttrain", "relbench"),
)
def test_empty_launch_to_s1_e1_later_task_and_old_resume(
    binding,
    tmp_path,
    monkeypatch,
) -> None:
    """Join M3-M9 receipts without re-testing each already-proven boundary."""

    settings = CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])
    repositories = settings.scopes.resolve(binding.scope_id)
    launch_fixture_root = tmp_path / "launch-fixture"
    launch_fixture_root.mkdir()
    resolver_case = build_resolver_case(
        launch_fixture_root,
        monkeypatch,
        binding,
    )
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    first_run_root = (tmp_path / "first-run").absolute()
    first = prepare_fresh_run_handoff(
        coordinator=_fresh_coordinator(resolver_case),
        settings=resolver_case["resolver"].settings,
        security_authority=_DescendantSecurityAuthority(),
        request=resolver_case["request"],
        run_root=first_run_root,
        objective_direction="maximize",
    )
    first_pin = first.active_workspace.bootstrap_pin
    first_identity = first.identity
    first.close()

    assert (
        first_identity.expert_release_id == resolved.manifest.expert_manifest.release_id
    )
    assert first_identity.knowledge_snapshot_id == (
        resolver_case["knowledge_package"].manifest.snapshot_id
    )
    assert resolver_case["knowledge_package"].manifest.catalog_generation == 0

    capture_root = tmp_path / "captured-experiment"
    capture_root.mkdir()
    capture = make_capture_fixture(capture_root)
    task_context = resolved.manifest.task_context_binding
    evidence_workspace = tmp_path / "replay-evidence"
    evidence_workspace.mkdir()
    bundle_store = RunBundleStore.initialize(
        evidence_workspace / settings.capture.state_path,
        settings.capture,
        settings.sanitation,
    )
    replay_context = bundle_store.publish_starting_artifacts(
        task_context_binding=task_context,
        launch_artifacts=(
            resolver_case["starting_artifacts"].verified.starting_artifacts
        ),
        validation_settings=settings.expert.validation,
    )
    environment = _artifact_environment(
        capture.request.artifact_environment,
        expert_release_id=first_identity.expert_release_id,
        task_adapter_manifest_id=resolved.manifest.task_adapter.manifest.task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            resolved.manifest.task_adapter.verification_receipt.verification_receipt_id
        ),
        starting_artifact_content_ids={
            item.artifact.starting_artifact_ref: (
                item.artifact.starting_artifact_content_id
            )
            for item in replay_context.starting_artifacts
        },
    )
    capture.request = replace(
        capture.request,
        scope_contract_id=resolved.manifest.scope_contract.scope_contract_id,
        scope_id=binding.scope_id,
        launch_manifest_id=first_identity.launch_manifest_id,
        knowledge_snapshot_id=first_identity.knowledge_snapshot_id,
        expert_base_release_id=first_identity.expert_release_id,
        task_context_binding=task_context,
        artifact_environment=environment,
    )
    stored_bundle = RunCapturePipeline(
        RunCaptureContext(capture.request),
        capture.settings,
    ).capture_if_due(CompletionState.STOPPED, force=True)
    assert stored_bundle is not None
    bundle_store.import_exact(stored_bundle)
    projection = RunBundleProjector(
        capture.settings.capture.score_comparison_tolerance
    ).project(stored_bundle)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        resolved.manifest.scope_contract,
        capture.settings.catalog,
    )
    catalog_generation = catalog.publish_projection(
        catalog.store.read_current(),
        projection,
    ).generation
    embedding_provider = DeterministicEmbeddingProvider(settings.knowledge.embeddings)
    knowledge_publisher = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        settings.github,
        settings.knowledge,
        embedding_provider,
    )
    snapshot_build = knowledge_publisher.build(
        resolved.manifest.scope_contract,
        catalog_generation,
        catalog.store.read_object_bytes,
        parent_snapshot_ids=(first_identity.knowledge_snapshot_id,),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version="kapso.retrieval.v1",
        published_at=_PUBLISHED_AT,
        publisher_attestation={"issuer": "deterministic-system-scenario"},
    )
    snapshot_publication = knowledge_publisher.publish(
        snapshot_build.package,
        expected_parent_sha="a" * 40,
        expected_current_snapshot_id=first_identity.knowledge_snapshot_id,
        committed_at=_PUBLISHED_AT,
        validation_closure_ids=(),
    )
    s1 = snapshot_publication.package
    assert catalog_generation.generation_number == 1
    assert s1.manifest.parent_snapshot_ids == (first_identity.knowledge_snapshot_id,)

    expert_root = tmp_path / "expert-successor"
    expert_root.mkdir()
    source_fixture, released_source = _successor_inputs(
        binding=binding,
        resolved=resolved,
        projection=projection,
        bundle_store=bundle_store,
        settings=settings,
    )
    validation_store, approval, authority, _predecessor = _approved_normal(
        expert_root,
        monkeypatch,
        source_fixture=source_fixture,
        released_source=released_source,
        source_adapter=resolver_case["task_adapters"].binding.verified_adapter,
    )
    candidate_store = validation_store.reducer.candidate_store
    stored_candidate = candidate_store.read(approval.snapshot.state.candidate_id)
    e1 = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    ).build(candidate_id=approval.snapshot.state.candidate_id)
    matrix_result = next(
        result
        for result in approval.snapshot.accepted_stage_results
        if type(result) is ExpertReleaseMatrixStageResultRecord
    )
    second_fixture_root = tmp_path / "second-launch-fixture"
    second_fixture_root.mkdir()
    second_case = build_resolver_case(
        second_fixture_root,
        monkeypatch,
        binding,
        expert_case=SimpleNamespace(
            stored_candidate=stored_candidate,
            expert_package=e1,
            release_matrix_stage_result=matrix_result,
            verified_adapter=authority.adapters.adapter,
        ),
        knowledge_package=s1,
    )
    second = prepare_fresh_run_handoff(
        coordinator=_fresh_coordinator(second_case),
        settings=second_case["resolver"].settings,
        security_authority=_DescendantSecurityAuthority(),
        request=second_case["request"],
        run_root=(tmp_path / "second-run").absolute(),
        objective_direction="maximize",
    )
    second_identity = second.identity
    second_pin = second.active_workspace.bootstrap_pin
    index_files = {
        path: payload
        for path, payload in s1.files.items()
        if PurePosixPath(path).parts[0] == "index"
    }
    retrieval = CrossRunRetriever(
        s1,
        SnapshotSearchIndex.open(s1.prepared, index_files),
        settings.knowledge.retrieval,
    ).retrieve(
        PriorKnowledgeQuery(
            task_context_binding=second_pin.launch_manifest.task_context_binding,
            problem=projection.episodes[0].proposal,
            current_gaps=("Select one evidence-grounded next experiment.",),
            directive="Retrieve transferable interventions and failures.",
        )
    )
    second.close()

    assert (
        e1.manifest.lineage.source_base_release_id == first_identity.expert_release_id
    )
    assert e1.manifest.candidate_id == approval.snapshot.state.candidate_id
    assert second_identity.expert_release_id == e1.manifest.release_id
    assert second_identity.knowledge_snapshot_id == s1.manifest.snapshot_id
    assert second_pin.launch_manifest.launch_request.binding == binding
    assert second_pin.launch_manifest.scope_repositories == repositories
    assert retrieval.prior_knowledge_snapshot.source_snapshot_id == (
        s1.manifest.snapshot_id
    )
    assert projection.episodes[0].episode_id in (
        retrieval.prior_knowledge_snapshot.selected_record_ids
    )

    resume_coordinator, _release_use, _security = _coordinator(
        settings=resolver_case["resolver"].settings,
        pin=first_pin,
        release_use_observation=first_pin.launch_manifest.release_use_observation,
    )
    resumed = prepare_resumed_run_handoff(
        coordinator=LaunchBootstrapCoordinator(
            settings=resolver_case["resolver"].settings,
            binding=resolver_case["request"].binding,
            resolver=resolver_case["resolver"],
            resume_coordinator=resume_coordinator,
        ),
        settings=resolver_case["resolver"].settings,
        run_root=first_run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert resumed.identity.expert_release_id == first_identity.expert_release_id
    assert (
        resumed.identity.knowledge_snapshot_id == first_identity.knowledge_snapshot_id
    )
    assert resumed.identity.expert_release_id != second_identity.expert_release_id
    assert (
        resumed.identity.knowledge_snapshot_id != second_identity.knowledge_snapshot_id
    )
    resumed.close()


def _artifact_environment(
    source: ArtifactEnvironment,
    *,
    expert_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
    starting_artifact_content_ids,
) -> ArtifactEnvironment:
    return ArtifactEnvironment.mint(
        kapso_commit=source.kapso_commit,
        expert_base_release_id=expert_release_id,
        task_adapter_manifest_id=task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(task_adapter_verification_receipt_id),
        starting_artifact_content_ids=starting_artifact_content_ids,
        dependency_lock_hash=source.dependency_lock_hash,
    )


def _successor_inputs(*, binding, resolved, projection, bundle_store, settings):
    trigger_policy = trigger_settings()
    episode = projection.episodes[0]
    expert_manifest = resolved.manifest.expert_manifest
    repository_map = resolved.manifest.expert_repository_map
    module_contracts = resolved.manifest.expert_module_contracts
    packet_without_observation = trigger_packet(
        settings=trigger_policy,
        episodes=(episode,),
        source_base_repository_map=repository_map,
        source_base_module_contracts=module_contracts,
        source_base_release=expert_manifest,
        current_scope_contract=resolved.manifest.scope_contract,
        source_base_scope_contract=resolved.manifest.scope_contract,
        active_task_bindings=(binding,),
        knowledge_source_bundle=projection.source_bundle,
        knowledge_sanitation_report=projection.sanitation_report,
        knowledge_extra_facts=projection.derivation_objects,
        knowledge_projection_derivation_ids=tuple(
            event.event_id for event in projection.derivation_objects
        ),
    )
    extraction_receipt = resolved.expert_source.source_extraction_receipt
    source_cache_receipt = replace(
        resolved.expert_artifact.receipt,
        asset_digests={
            extraction_receipt.source_archive_ref: (
                extraction_receipt.source_archive_digest
            )
        },
    )
    source_receipt = ExpertSourceBaseTreeReceipt.mint(
        release_id=expert_manifest.release_id,
        cache_verification_receipt=source_cache_receipt,
        source_extraction_receipt=extraction_receipt,
        source_base_tree_hash=expert_manifest.candidate_tree_hash,
        repository_map_id=repository_map.repository_map_id,
        module_contract_ids=tuple(
            module.module_contract_id for module in module_contracts
        ),
        materializer_version=(
            packet_without_observation.source_base_tree_receipt.materializer_version
        ),
    )
    module = module_contracts[0]
    description = "The captured experiment supports one reusable provenance change."
    observation_payload = {
        "affected_capability_ids": [module.module_id],
        "affected_paths": [module.entrypoint_refs[0]],
        "configuration_fingerprint": configuration_fingerprint(trigger_policy),
        "description": description,
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "exact_evidence_ids": [repository_map.repository_map_id],
        "independent_lineage_ids": [],
        "inspection_policy_version": trigger_policy.inspection_policy_version,
        "kind": ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX.value,
        "occurrence_count": 1,
        "source_base_tree_hash": expert_manifest.candidate_tree_hash,
        "task_context_binding_ids": [],
    }
    inspection_final_output = json.dumps(observation_payload, indent=2) + "\n"
    observation = ExpertTriggerObservation.mint(
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        source_base_tree_hash=expert_manifest.candidate_tree_hash,
        inspection_policy_version=trigger_policy.inspection_policy_version,
        configuration_fingerprint=configuration_fingerprint(trigger_policy),
        inspection_operation=inspection_operation(
            trigger_policy,
            inspection_final_output,
        ),
        inspection_final_output=inspection_final_output,
        difficulty_signature=None,
        difficulty_evidence_signatures={},
        description=description,
        affected_capability_ids=(module.module_id,),
        affected_paths=(module.entrypoint_refs[0],),
        exact_evidence_ids=(repository_map.repository_map_id,),
        independent_lineage_ids=(),
        task_context_binding_ids=(),
        occurrence_count=1,
    )
    packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=(
            packet_without_observation.knowledge_snapshot_manifest
        ),
        knowledge_record_closure_digest=(
            packet_without_observation.knowledge_record_closure_digest
        ),
        configuration_fingerprint=(
            packet_without_observation.configuration_fingerprint
        ),
        scope_contract=packet_without_observation.scope_contract,
        source_base_scope_contract=packet_without_observation.source_base_scope_contract,
        source_base_release=expert_manifest,
        source_base_tree_receipt=source_receipt,
        source_base_tree_hash=expert_manifest.candidate_tree_hash,
        source_base_repository_map=repository_map,
        source_base_module_contracts=module_contracts,
        episodes=packet_without_observation.episodes,
        claims=packet_without_observation.claims,
        trigger_observations=(observation,),
        active_task_bindings=(binding,),
        proof_reference_ids=packet_without_observation.proof_reference_ids,
        recovery_barrier_basis_packet_id=None,
    )
    source_fixture = SimpleNamespace(
        packet=packet,
        bundle_provider=RunBundleLineageProvider(
            bundle_store,
            RunBundleProjector(settings.capture.score_comparison_tolerance),
            settings.capture.bundle_lineage_limit,
        ),
        context_provider=StoredSourceReplayContextProvider(
            bundle_store,
            settings.expert.validation,
        ),
    )
    source_contents = dict(resolved.expert_source.source_contents)
    materialized_source = replace(
        resolved.expert_artifact,
        receipt=source_cache_receipt,
    )
    return source_fixture, (packet, materialized_source, source_contents)
