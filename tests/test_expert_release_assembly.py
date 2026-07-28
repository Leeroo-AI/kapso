from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

import test_expert_release_matrix_reservation as reservation_fixture_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertReleaseLineage,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH,
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
    ExpertReleaseAssemblyError,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleasePublicationPlan,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.publisher import (
    ExpertReleasePublicationError,
    ExpertReleasePublisher,
)
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationCompareAndSwapError,
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactActivationWitness,
    GitHubArtifactResolver,
    PublicationAssetIntent,
    PublicationSourceFile,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.github.publisher import AutonomousGitHubPublisher
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.source_archives import (
    SourceArchiveError,
    build_deterministic_tar_zst,
)
from test_expert_promotion_decision import _settings
from test_expert_promotion_evidence import _bootstrap_prepared_with_store
from test_expert_promotion_stage import _completed_runtime
from test_expert_publication_eligibility import _coordinator, _publish_matrix
from kapso.cross_run.expert.promotion_stage import ExpertReleaseMatrixStageCoordinator

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _approved_bootstrap(
    tmp_path,
    monkeypatch,
    *,
    candidate_closure=None,
    source_adapter=None,
):
    settings = _settings(minimum_replicates=1, minimum_pairs=1)
    monkeypatch.setattr(
        reservation_fixture_module,
        "_quality_only_validation_settings",
        lambda: settings,
    )
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
        candidate_closure=candidate_closure,
        source_adapter=source_adapter,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    matrix = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    case = type(
        "ReleaseCase",
        (),
        {
            "validation_store": validation_store,
            "matrix_commit": matrix,
            "prepared": prepared,
        },
    )()
    authority = _coordinator(case)
    approval = authority.coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )
    return validation_store, matrix, approval, authority


def _approved_normal(
    tmp_path,
    monkeypatch,
    *,
    source_fixture=None,
    released_source=None,
    source_adapter=None,
):
    validation_settings = _settings(minimum_replicates=1, minimum_pairs=2)
    case = _publish_matrix(
        tmp_path,
        monkeypatch,
        bootstrap=False,
        settings=validation_settings,
        source_fixture=source_fixture,
        released_source=released_source,
        source_adapter=source_adapter,
    )
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    release_id = case.prepared.current_release_observation.release_id
    assert release_id is not None
    predecessor_pointer = _activation_predecessor_pointer(release_id, settings)
    predecessor_record = predecessor_pointer.publication_record
    observation = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=predecessor_pointer.scope_id,
        release_id=release_id,
        publication_id=predecessor_record.publication_id,
        repository_full_name=predecessor_record.repository_full_name,
        repository_node_id=predecessor_record.repository_node_id,
        default_branch_head_commit_sha="c" * 40,
        current_pointer_digest=tree_or_blob_digest(predecessor_pointer.to_json_bytes()),
        validation_closure_ids=predecessor_pointer.validation_closure_ids,
    )
    authority = _coordinator(
        case,
        current_observations=(observation, observation),
    )
    approval = authority.coordinator.publish(
        candidate_id=case.matrix_commit.snapshot.state.candidate_id,
        release_matrix_stage_result_id=(
            case.matrix_commit.stage_result.stage_result_record_id
        ),
    )
    return case.validation_store, approval, authority, predecessor_pointer


def _activation_predecessor_pointer(release_id, settings):
    asset = GitHubReleaseAsset(
        asset_id="predecessor-asset",
        name="expert-source.tar.zst",
        media_type="application/zstd",
        size=1,
        sha256=tree_or_blob_digest(b"predecessor"),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release_id,
        repository_node_id="expert_repo_node",
        repository_full_name="Leeroo-AI/kapso-expert",
        commit_sha="a" * 40,
        immutable_release_id="predecessor-release",
        tag=(
            f"{settings.github.expert_tag_prefix}E000000-"
            f"{release_id.rsplit(':sha256:', 1)[1]}"
        ),
        assets=(asset,),
        release_attestation_ref="attestation-predecessor",
        published_at="2026-07-21T11:00:00Z",
        publisher_identity="leeroo-coder",
    )
    return CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=publication,
        publication_intent_digest=tree_or_blob_digest(b"predecessor-intent"),
        source_tree_digest=tree_or_blob_digest(b"predecessor-source"),
        source_git_tree_sha="b" * 40,
        materialized_tree_digest=tree_or_blob_digest(b"predecessor-materialized"),
        manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
        manifest_digest=tree_or_blob_digest(b"predecessor-manifest"),
        validation_closure_ids=(release_id,),
    )


def _publication_publisher(
    *,
    validation_store,
    assembler,
    authority,
    settings,
    tmp_path,
    current_observation,
    current_pointer,
):
    github_client = object()
    resolver = GitHubArtifactResolver(
        github_client,
        settings.github,
        settings.scopes,
    )
    materializer = GitHubArtifactMaterializer(
        github_client,
        settings.github,
        tmp_path / "reservation-publisher-state",
    )
    generic_publisher = AutonomousGitHubPublisher(
        github_client,
        resolver,
        materializer,
        settings.github,
    )
    current_authority = GitHubExpertCurrentReleaseProvider(resolver)
    validation_store.reducer.current_release_provider = current_authority
    publisher = ExpertReleasePublisher(
        assembler=assembler,
        validation_store=validation_store,
        github_publisher=generic_publisher,
        resolver=resolver,
        current_release_authority=current_authority,
        task_adapter_authority=validation_store.reducer.task_adapter_provider,
        security_denylist_authority=authority.denylist,
        release_use_policy_authority=authority.release_use_policy,
    )
    current_state = CurrentPointerState(
        pointer=current_pointer,
        head_commit_sha=current_observation.default_branch_head_commit_sha,
    )
    resolver.read_current_pointer_state = (
        lambda scope_id, artifact_kind, allow_missing: current_state
    )
    current_authority.observe_task_evaluation_current = (
        lambda scope_id: current_observation
    )
    return publisher


def _published_pointer(plan, release_id, commit_sha):
    asset = GitHubReleaseAsset(
        asset_id="asset-1",
        name="expert.tar.zst",
        media_type="application/zstd",
        size=1,
        sha256=tree_or_blob_digest(b"x"),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release_id,
        repository_node_id=(plan.current_release_observation.repository_node_id),
        repository_full_name=(plan.current_release_observation.repository_full_name),
        commit_sha=commit_sha,
        immutable_release_id="release-1",
        tag="expert/E999999",
        assets=(asset,),
        release_attestation_ref="attestation-1",
        published_at="2026-07-21T12:01:30Z",
        publisher_identity="leeroo-coder",
    )
    return CurrentArtifactPointer(
        scope_id=plan.scope_id,
        publication_record=publication,
        publication_intent_digest=tree_or_blob_digest(b"intent"),
        source_tree_digest=tree_or_blob_digest(b"source"),
        source_git_tree_sha="a" * 40,
        materialized_tree_digest=tree_or_blob_digest(b"materialized"),
        manifest_relative_path="release-manifest.json",
        manifest_digest=tree_or_blob_digest(b"manifest"),
        validation_closure_ids=(release_id,),
    )


def _publication_intent(package, plan, committed_at, commit_sha):
    source_files = tuple(
        PublicationSourceFile(
            relative_path=path,
            mode=mode,
            size=len(payload),
            sha256=tree_or_blob_digest(payload),
            git_blob_sha=git_object_sha("blob", payload),
        )
        for path, (payload, mode) in sorted(package.publication_files.items())
    )
    source_git_tree_sha = git_tree_shas(
        {
            source.relative_path: (source.git_blob_sha, source.mode)
            for source in source_files
        }
    )[""]
    return ArtifactPublicationIntent(
        scope_id=plan.scope_id,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_node_id=plan.current_release_observation.repository_node_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        expected_parent_sha=(
            plan.current_release_observation.default_branch_head_commit_sha
        ),
        source_commit_sha=commit_sha,
        source_tree_digest=plan.publication_source_tree_digest,
        source_git_tree_sha=source_git_tree_sha,
        source_files=source_files,
        preserved_current=None,
        materialized_tree_digest=tree_or_blob_digest(b"materialized expert package"),
        manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
        manifest_digest=plan.manifest_digest,
        tag=plan.tag,
        assets=tuple(
            PublicationAssetIntent(
                name=asset.name,
                media_type=asset.media_type,
                size=asset.size,
                sha256=asset.sha256,
            )
            for asset in plan.assets
        ),
        validation_closure_ids=plan.validation_closure_ids,
        publisher_identity="leeroo-coder",
        committed_at=committed_at,
    )


def _publication_pointer(plan, intent):
    release_assets = tuple(
        GitHubReleaseAsset(
            asset_id=str(position),
            name=asset.name,
            media_type=asset.media_type,
            size=asset.size,
            sha256=asset.sha256,
        )
        for position, asset in enumerate(plan.assets, start=1)
    )
    record = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_node_id=intent.repository_node_id,
        repository_full_name=intent.repository_full_name,
        commit_sha=intent.source_commit_sha,
        immutable_release_id="7",
        tag=plan.tag,
        assets=release_assets,
        release_attestation_ref="github-release-attestation:sha256:" + "a" * 64,
        published_at="2026-07-21T12:01:30Z",
        publisher_identity=intent.publisher_identity,
    )
    return CurrentArtifactPointer(
        scope_id=plan.scope_id,
        publication_record=record,
        publication_intent_digest=intent.digest,
        source_tree_digest=intent.source_tree_digest,
        source_git_tree_sha=intent.source_git_tree_sha,
        materialized_tree_digest=intent.materialized_tree_digest,
        manifest_relative_path=intent.manifest_relative_path,
        manifest_digest=intent.manifest_digest,
        validation_closure_ids=intent.validation_closure_ids,
    )


def test_release_assembly_is_exact_deterministic_and_approval_only(
    tmp_path,
    monkeypatch,
):
    validation_store, matrix, approval, _authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    stored_candidate = candidate_store.read(approval.snapshot.state.candidate_id)
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )

    first = assembler.build(
        candidate_id=stored_candidate.closure.manifest.candidate_id,
    )
    second = assembler.build(
        candidate_id=stored_candidate.closure.manifest.candidate_id,
    )

    assert approval.snapshot.state.promotion_state is ExpertPromotionState.APPROVED
    assert first.manifest == second.manifest
    assert first.source_archive == second.source_archive
    assert first.evidence_archive == second.evidence_archive
    assert first.control_archive == second.control_archive
    assert first.manifest.candidate_tree_hash == source_tree_digest(
        {
            path: (tree_or_blob_digest(payload), mode, len(payload))
            for path, (payload, mode) in first.source_files.items()
        }
    )
    assert first.evidence_manifest.evidence_manifest_id in (
        first.manifest.consumed_dependency_ids
    )
    assert first.manifest.candidate_consumed_expert_release_ids == (
        stored_candidate.closure.manifest.consumed_expert_release_ids
    )
    assert set(first.manifest.candidate_consumed_expert_release_ids).issubset(
        first.manifest.consumed_dependency_ids
    )
    assert EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH in first.evidence_files
    assert EXPERT_RELEASE_MANIFEST_PATH in first.publication_files
    assert not any(
        "artifacts/" in path
        or "workspace-delta" in path
        or "operation-receipt" in path
        or "expert-evaluator-result" in path
        for path in first.evidence_files
    )

    changed_source_files = dict(first.source_files)
    changed_path = next(iter(changed_source_files))
    payload, mode = changed_source_files[changed_path]
    changed_source_files[changed_path] = (payload + b"mutated", mode)
    with pytest.raises(ExpertReleaseAssemblyError):
        assembler.verify(replace(first, source_files=changed_source_files))
    unrelated_dependency = "unrelated-release-evidence:sha256:" + "f" * 64
    with pytest.raises(ValueError, match="not exact"):
        replace(
            first.manifest,
            consumed_dependency_ids=tuple(
                sorted(
                    {
                        *first.manifest.consumed_dependency_ids,
                        unrelated_dependency,
                    }
                )
            ),
        )


def test_normal_release_binds_source_base_and_activation_predecessor(
    tmp_path,
    monkeypatch,
):
    validation_store, approval, authority, predecessor_pointer = _approved_normal(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    package = assembler.build(candidate_id=approval.snapshot.state.candidate_id)
    current_observation = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=current_observation,
        current_pointer=predecessor_pointer,
    )
    committed = publisher.reserve(
        candidate_id=approval.snapshot.state.candidate_id,
        committed_at="2026-07-21T12:00:00Z",
    )
    plan = committed.reservation.plan
    source_base_id = package.manifest.lineage.source_base_release_id

    assert source_base_id is not None
    assert package.manifest.lineage.activation_predecessor_release_id == (
        source_base_id
    )
    assert plan.lineage == package.manifest.lineage
    assert plan.current_release_observation.release_id == source_base_id
    assert plan.activation_predecessor_pointer == predecessor_pointer
    assert plan.generation == 1
    assert plan.tag.endswith(f"/E000001-{plan.release_id.rsplit(':sha256:', 1)[1]}")
    assert source_base_id in package.manifest.consumed_dependency_ids
    assert source_base_id in package.manifest.candidate_consumed_expert_release_ids
    assert package.manifest.control_dependency_ids == ()

    reopened = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    ).reopen_release_publication(plan.candidate_id)

    assert committed.reservation.plan == plan
    assert reopened == committed.reservation


@pytest.mark.parametrize(
    "files",
    (
        {"unsafe\\path": (b"payload", "100644")},
        {"collision": (b"file", "100644"), "collision/child": (b"nested", "100644")},
        {".git/config": (b"metadata", "100644")},
        {".gitmodules": (b"metadata", "100644")},
    ),
)
def test_deterministic_archive_writer_rejects_reader_incompatible_paths(files):
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    with pytest.raises(SourceArchiveError, match="path closure"):
        build_deterministic_tar_zst(
            files,
            compression_level=settings.expert.release_archive_compression_level,
            zstd_window_size_bytes=settings.github.zstd_window_size_bytes,
        )


def test_release_publication_reservation_is_durable_idempotent_and_freezes_head(
    tmp_path,
    monkeypatch,
):
    validation_store, _matrix, approval, authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    package = assembler.build(candidate_id=approval.snapshot.state.candidate_id)
    current_observation = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=current_observation,
        current_pointer=None,
    )
    assert not hasattr(assembler, "authorize_publication_plan")
    assert not hasattr(validation_store, "reserve_release_publication")
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_results = tuple(
            executor.map(
                lambda _position: publisher.reserve(
                    candidate_id=approval.snapshot.state.candidate_id,
                    committed_at="2026-07-21T12:00:00Z",
                ),
                range(2),
            )
        )
    committed = next(result for result in concurrent_results if not result.replayed)
    concurrent_replay = next(result for result in concurrent_results if result.replayed)
    publisher.current_release_authority.observe_task_evaluation_current = (
        lambda scope_id: pytest.fail(
            "durable reservation replay must not read remote CURRENT"
        )
    )
    publisher.resolver.read_current_pointer_state = (
        lambda scope_id, artifact_kind, allow_missing: pytest.fail(
            "durable reservation replay must not read the remote pointer"
        )
    )
    replayed = publisher.reserve(
        candidate_id=approval.snapshot.state.candidate_id,
        committed_at="2026-07-21T12:01:00Z",
    )
    plan = committed.reservation.plan
    reopened_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )

    assert committed.replayed is False
    assert concurrent_replay.reservation == committed.reservation
    assert replayed.replayed is True
    assert replayed.reservation == committed.reservation
    assert replayed.reservation.intent.committed_at == "2026-07-21T12:00:00Z"
    assert (
        reopened_store.reopen_release_publication(plan.candidate_id)
        == committed.reservation
    )
    with validation_store._lock(exclusive=False):
        journal = validation_store._read_journal_unlocked(plan.candidate_id)
    with pytest.raises(
        ExpertValidationCompareAndSwapError,
        match="frozen for release publication",
    ):
        validation_store._append_transition(
            journal,
            approval.snapshot.transition,
        )

    reopened_assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=reopened_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    github_client = object()
    resolver = GitHubArtifactResolver(
        github_client,
        settings.github,
        settings.scopes,
    )
    generic_publisher = AutonomousGitHubPublisher(
        github_client,
        resolver,
        GitHubArtifactMaterializer(
            github_client,
            settings.github,
            tmp_path / "publisher-state",
        ),
        settings.github,
    )
    current_authority = GitHubExpertCurrentReleaseProvider(resolver)
    reopened_store.reducer.current_release_provider = current_authority
    ExpertPublicationEligibilityCoordinator(
        validation_store=reopened_store,
        current_release_authority=current_authority,
        task_adapter_authority=reopened_store.reducer.task_adapter_provider,
        security_denylist_authority=authority.denylist,
        release_use_policy_authority=authority.release_use_policy,
    )
    publisher = ExpertReleasePublisher(
        assembler=reopened_assembler,
        validation_store=reopened_store,
        github_publisher=generic_publisher,
        resolver=resolver,
        current_release_authority=current_authority,
        task_adapter_authority=reopened_store.reducer.task_adapter_provider,
        security_denylist_authority=authority.denylist,
        release_use_policy_authority=authority.release_use_policy,
    )
    resolver.read_artifact_intent = lambda scope_id, kind, artifact_id: None
    resolver.read_artifact_pointer = lambda scope_id, kind, artifact_id: None
    remote = {
        "state": CurrentPointerState(
            pointer=None,
            head_commit_sha=plan.current_release_observation.default_branch_head_commit_sha,
        ),
        "observation": plan.current_release_observation,
    }
    monkeypatch.setattr(
        GitHubArtifactResolver,
        "read_current_pointer_state",
        lambda self, scope_id, artifact_kind, allow_missing: remote["state"],
    )
    monkeypatch.setattr(
        GitHubExpertCurrentReleaseProvider,
        "observe_task_evaluation_current",
        lambda self, scope_id: remote["observation"],
    )

    with pytest.raises(
        ExpertReleasePublicationError,
        match="absent CURRENT",
    ):
        publisher.resolve_stale(
            candidate_id=plan.candidate_id,
            resolved_at="2026-07-21T12:02:00Z",
        )

    own_pointer = _published_pointer(plan, plan.release_id, "e" * 40)
    remote["state"] = CurrentPointerState(
        pointer=own_pointer,
        head_commit_sha="e" * 40,
    )
    with pytest.raises(
        ExpertReleasePublicationError,
        match="RELEASED recovery",
    ):
        publisher.resolve_stale(
            candidate_id=plan.candidate_id,
            resolved_at="2026-07-21T12:02:10Z",
        )

    other_release_id = content_id("expert-base-release", {"winner": 1})
    other_pointer = _published_pointer(plan, other_release_id, "f" * 40)
    remote["state"] = CurrentPointerState(
        pointer=other_pointer,
        head_commit_sha="f" * 40,
    )
    remote["observation"] = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=plan.scope_id,
        release_id=other_release_id,
        publication_id=other_pointer.publication_record.publication_id,
        repository_full_name=other_pointer.publication_record.repository_full_name,
        repository_node_id=other_pointer.publication_record.repository_node_id,
        default_branch_head_commit_sha="f" * 40,
        current_pointer_digest=tree_or_blob_digest(other_pointer.to_json_bytes()),
        validation_closure_ids=other_pointer.validation_closure_ids,
    )
    own_intent = _publication_intent(
        package,
        plan,
        committed.reservation.intent.committed_at,
        "d" * 40,
    )
    own_identity = _publication_pointer(plan, own_intent)
    winner_intent = replace(
        own_intent,
        artifact_id=other_release_id,
        source_commit_sha=other_pointer.publication_record.commit_sha,
        tag=other_pointer.publication_record.tag,
        validation_closure_ids=other_pointer.validation_closure_ids,
    )
    winner_identity = replace(
        other_pointer,
        publication_intent_digest=winner_intent.digest,
        source_tree_digest=winner_intent.source_tree_digest,
        source_git_tree_sha=winner_intent.source_git_tree_sha,
        materialized_tree_digest=winner_intent.materialized_tree_digest,
        manifest_relative_path=winner_intent.manifest_relative_path,
        manifest_digest=winner_intent.manifest_digest,
        validation_closure_ids=winner_intent.validation_closure_ids,
    )
    remote["state"] = CurrentPointerState(
        pointer=winner_identity,
        head_commit_sha="f" * 40,
    )
    remote["observation"] = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=plan.scope_id,
        release_id=other_release_id,
        publication_id=winner_identity.publication_record.publication_id,
        repository_full_name=(winner_identity.publication_record.repository_full_name),
        repository_node_id=winner_identity.publication_record.repository_node_id,
        default_branch_head_commit_sha="f" * 40,
        current_pointer_digest=tree_or_blob_digest(winner_identity.to_json_bytes()),
        validation_closure_ids=winner_identity.validation_closure_ids,
    )
    resolver.read_artifact_intent = lambda scope_id, kind, artifact_id: (
        winner_intent if artifact_id == other_release_id else own_intent
    )
    resolver.read_artifact_pointer = lambda scope_id, kind, artifact_id: (
        winner_identity if artifact_id == other_release_id else own_identity
    )
    resolver.diagnose_repository = lambda scope_id, kind: object()
    resolver.resolve_artifact = lambda scope_id, kind, artifact_id: SimpleNamespace(
        pointer=own_identity if artifact_id == plan.release_id else winner_identity
    )
    activation_preparation_commit_sha = "c" * 40
    resolver.resolve_artifact_activation_preparation = (
        lambda scope_id, kind, artifact_id, intent, pointer, allow_missing=False: (
            activation_preparation_commit_sha
        )
    )
    repository_binding = settings.scopes.resolve(plan.scope_id).binding_fingerprint
    winner_witness = GitHubArtifactActivationWitness.mint(
        scope_id=plan.scope_id,
        scope_repository_binding_hash=repository_binding,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=other_release_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        activation_commit_sha="f" * 40,
        publication_intent_digest=winner_intent.digest,
        current_pointer_digest=tree_or_blob_digest(winner_identity.to_json_bytes()),
    )
    own_witness = GitHubArtifactActivationWitness.mint(
        scope_id=plan.scope_id,
        scope_repository_binding_hash=repository_binding,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        activation_commit_sha=activation_preparation_commit_sha,
        publication_intent_digest=own_intent.digest,
        current_pointer_digest=tree_or_blob_digest(own_identity.to_json_bytes()),
    )
    witness_state = {"own": own_witness}

    def resolve_activation_witness(
        scope_id,
        kind,
        artifact_id,
        intent,
        pointer,
        allow_missing=False,
    ):
        if artifact_id == other_release_id:
            return winner_witness
        return witness_state["own"]

    resolver.resolve_artifact_activation_witness = resolve_activation_witness
    generic_publisher.finalize_artifact_activation_witness = (
        lambda scope_id, kind, artifact_id, intent, pointer: winner_witness
    )
    with pytest.raises(
        ExpertReleasePublicationError,
        match="historically active",
    ):
        publisher.resolve_stale(
            candidate_id=plan.candidate_id,
            resolved_at="2026-07-21T12:02:15Z",
        )
    witness_state["own"] = None
    resolution = publisher.resolve_stale(
        candidate_id=plan.candidate_id,
        resolved_at="2026-07-21T12:02:20Z",
    )
    replayed_resolution = publisher.resolve_stale(
        candidate_id=plan.candidate_id,
        resolved_at="2026-07-21T12:03:00Z",
    )

    assert resolution == replayed_resolution
    assert resolution.publication_intent_id == (
        committed.reservation.intent.publication_intent_id
    )
    assert resolution.own_github_publication_intent == own_intent
    assert resolution.own_github_publication_pointer == own_identity
    assert resolution.own_github_activation_preparation_commit_sha == (
        activation_preparation_commit_sha
    )
    assert resolution.observed_current_activation_witness == winner_witness
    assert own_identity.publication_record.publication_id in (
        resolution.exact_dependency_ids
    )
    assert reopened_store.reopen_release_publication(plan.candidate_id) is None
    with reopened_store._lock(exclusive=False):
        resolved_journal = reopened_store._read_journal_unlocked(plan.candidate_id)
    assert resolved_journal.release_publication_intent_id is None
    assert resolved_journal.release_publication_stale_resolution_id == (
        resolution.stale_resolution_id
    )
    transition_values = approval.snapshot.transition.to_dict()
    transition_values.pop("transition_id")
    transition_values.update(
        {
            "transition_number": (approval.snapshot.transition.transition_number + 1),
            "predecessor_transition_id": (approval.snapshot.transition.transition_id),
            "predecessor_state_id": approval.snapshot.state.validation_state_id,
            "operation_id": content_id(
                "expert-validation-operation",
                {"after": "stale-publication"},
            ),
        }
    )
    successor_transition = type(approval.snapshot.transition).mint(**transition_values)
    unfrozen = reopened_store._append_transition(
        resolved_journal,
        successor_transition,
    )
    assert unfrozen.release_publication_stale_resolution_id == (
        resolution.stale_resolution_id
    )


def test_publication_planning_rejects_stale_or_substituted_current(
    tmp_path,
    monkeypatch,
):
    validation_store, _matrix, approval, authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    changed_values = frozen_current.to_dict()
    changed_values.pop("observation_id")
    changed_values["default_branch_head_commit_sha"] = "d" * 40
    moved_current = TaskEvaluationCurrentReleaseObservation.mint(**changed_values)
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=moved_current,
        current_pointer=None,
    )

    with pytest.raises(
        ExpertReleasePublicationError,
        match="CURRENT changed after approval",
    ):
        publisher.reserve(
            candidate_id=approval.snapshot.state.candidate_id,
            committed_at="2026-07-21T12:00:00Z",
        )
    assert (
        validation_store.reopen_release_publication(
            approval.snapshot.state.candidate_id
        )
        is None
    )

    substituted_pointer = _activation_predecessor_pointer(
        content_id("expert-base-release", {"release": "substituted"}),
        settings,
    )
    publisher.current_release_authority.observe_task_evaluation_current = (
        lambda scope_id: frozen_current
    )
    publisher.resolver.read_current_pointer_state = (
        lambda scope_id, artifact_kind, allow_missing: CurrentPointerState(
            pointer=substituted_pointer,
            head_commit_sha=frozen_current.default_branch_head_commit_sha,
        )
    )
    with pytest.raises(
        ExpertReleasePublicationError,
        match="CURRENT changed after approval",
    ):
        publisher.reserve(
            candidate_id=approval.snapshot.state.candidate_id,
            committed_at="2026-07-21T12:00:00Z",
        )
    assert (
        validation_store.reopen_release_publication(
            approval.snapshot.state.candidate_id
        )
        is None
    )


def test_release_publication_reservation_rejects_caller_plan_shapes(
    tmp_path,
    monkeypatch,
):
    validation_store, _matrix, approval, authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    package = assembler.build(candidate_id=approval.snapshot.state.candidate_id)
    current_observation = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=current_observation,
        current_pointer=None,
    )
    plan = publisher.reserve(
        candidate_id=approval.snapshot.state.candidate_id,
        committed_at="2026-07-21T12:00:00Z",
    ).reservation.plan
    legacy_plan = plan.to_dict()
    legacy_plan["parent_release_id"] = None
    legacy_plan["parent_pointer"] = None
    legacy_plan.pop("lineage")
    legacy_plan.pop("activation_predecessor_pointer")
    with pytest.raises(ValueError, match="fields mismatch"):
        ExpertReleasePublicationPlan.from_dict(legacy_plan)
    unequal_values = plan.to_dict()
    unequal_values.pop("publication_plan_id")
    unequal_values["lineage"] = ExpertReleaseLineage(
        source_base_release_id=None,
        activation_predecessor_release_id=content_id(
            "expert-base-release",
            {"release": "unexpected-current"},
        ),
    )
    with pytest.raises(ValueError, match="CURRENT observation differs"):
        ExpertReleasePublicationPlan.mint(**unequal_values)
    changed_values = plan.to_dict()
    changed_values.pop("publication_plan_id")
    changed_values["publication_source_tree_digest"] = tree_or_blob_digest(
        b"different publication source"
    )
    conflicting_plan = ExpertReleasePublicationPlan.mint(**changed_values)

    with pytest.raises(ExpertValidationStoreError, match="bound publisher"):
        validation_store._reserve_release_publication(
            object(),
            plan=conflicting_plan,
            package=package,
            committed_at="2026-07-21T12:01:00Z",
        )
    with pytest.raises(ExpertValidationStoreError, match="deterministic package"):
        validation_store._reserve_release_publication(
            publisher,
            plan=conflicting_plan,
            package=package,
            committed_at="2026-07-21T12:01:00Z",
        )
