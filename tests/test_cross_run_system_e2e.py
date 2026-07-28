"""One deterministic cross-module scenario for both supported task shapes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    CompletionState,
    CrossRunTaskBindingSettings,
)
from kapso.cross_run.expert.release import ExpertReleaseAssembler
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
from test_launch_resolver import resolver_case
from test_run_resume_coordinator import _coordinator
from test_cross_run_retrieval import relbench_context

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
    resolver_case,
    tmp_path,
    monkeypatch,
) -> None:
    """Join M3-M9 receipts without re-testing each already-proven boundary."""

    settings = CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])
    repositories = settings.scopes.resolve(binding.scope_id)
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
    if binding.task_family_id == "relational_tabular_prediction":
        task_context = relbench_context(task_context)
    environment = _artifact_environment(
        capture.request.artifact_environment,
        task_context=task_context,
        expert_release_id=first_identity.expert_release_id,
        task_adapter_manifest_id=resolved.manifest.task_adapter.manifest.task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            resolved.manifest.task_adapter.verification_receipt.verification_receipt_id
        ),
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
            task_context_binding=task_context,
            problem=projection.episodes[0].proposal,
            current_gaps=("Select one evidence-grounded next experiment.",),
            directive="Retrieve transferable interventions and failures.",
        )
    )

    assert catalog_generation.generation_number == 1
    assert s1.manifest.parent_snapshot_ids == (first_identity.knowledge_snapshot_id,)
    assert retrieval.prior_knowledge_snapshot.source_snapshot_id == (
        s1.manifest.snapshot_id
    )
    assert projection.episodes[0].episode_id in (
        retrieval.prior_knowledge_snapshot.selected_record_ids
    )

    expert_root = tmp_path / "expert-successor"
    expert_root.mkdir()
    validation_store, approval, _authority, _predecessor = _approved_normal(
        expert_root,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    e1 = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    ).build(candidate_id=approval.snapshot.state.candidate_id)
    second_task_pin = {
        "binding": binding,
        "expert_release_id": e1.manifest.release_id,
        "knowledge_snapshot_id": s1.manifest.snapshot_id,
        "repositories": repositories,
    }

    assert e1.manifest.lineage.source_base_release_id is not None
    assert e1.manifest.candidate_id == approval.snapshot.state.candidate_id
    assert second_task_pin["binding"] == binding
    assert second_task_pin["repositories"] == settings.scopes.resolve("ml_ai")

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
    assert resumed.identity.expert_release_id != second_task_pin["expert_release_id"]
    assert (
        resumed.identity.knowledge_snapshot_id
        != second_task_pin["knowledge_snapshot_id"]
    )
    resumed.close()


def _artifact_environment(
    source: ArtifactEnvironment,
    *,
    task_context,
    expert_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
) -> ArtifactEnvironment:
    return ArtifactEnvironment.mint(
        kapso_commit=source.kapso_commit,
        expert_base_release_id=expert_release_id,
        task_adapter_manifest_id=task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(task_adapter_verification_receipt_id),
        starting_artifact_content_ids={
            reference: content_id(
                "source-replay-starting-artifact",
                {
                    "reference": reference,
                    "task_context_binding_id": task_context.task_context_binding_id,
                },
            )
            for reference in task_context.starting_artifact_refs
        },
        dependency_lock_hash=source.dependency_lock_hash,
    )
