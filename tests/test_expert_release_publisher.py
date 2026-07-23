from __future__ import annotations

from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.publisher import (
    ExpertReleasePublicationGate,
    ExpertReleasePublicationError,
    ExpertReleasePublisher,
)
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    PublicationTelemetry,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactResolver,
)
from kapso.cross_run.settings import CrossRunSettings
from test_expert_release_assembly import _approved_bootstrap, _publication_plan

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _release_pointer(plan, materialized_tree_digest):
    assets = tuple(
        GitHubReleaseAsset(
            asset_id=f"asset-{position}",
            name=asset.name,
            media_type=asset.media_type,
            size=asset.size,
            sha256=asset.sha256,
        )
        for position, asset in enumerate(plan.assets, start=1)
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_node_id=plan.current_release_observation.repository_node_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        commit_sha="b" * 40,
        immutable_release_id="7",
        tag=plan.tag,
        assets=assets,
        release_attestation_ref="expert-release-attestation",
        published_at="2026-07-21T12:00:00Z",
        publisher_identity="leeroo-coder",
    )
    return CurrentArtifactPointer(
        scope_id=plan.scope_id,
        publication_record=publication,
        publication_intent_digest=tree_or_blob_digest(b"publication-intent"),
        source_tree_digest=plan.publication_source_tree_digest,
        source_git_tree_sha="d" * 40,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
        manifest_digest=plan.manifest_digest,
        validation_closure_ids=plan.validation_closure_ids,
    )


def _source_file_projection(source_tree: Path):
    return {
        path.relative_to(source_tree).as_posix(): (
            tree_or_blob_digest(path.read_bytes()),
            "100755" if path.stat().st_mode & 0o111 else "100644",
        )
        for path in sorted(source_tree.rglob("*"))
        if path.is_file()
    }


def test_expert_publisher_derives_package_and_refreshes_activation_authority(
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
    plan = _publication_plan(package, approval, settings)
    reservation = validation_store.reserve_release_publication(
        assembler.authorize_publication_plan(package=package, plan=plan),
        committed_at="2026-07-21T12:00:00Z",
    ).reservation
    github_client = object()
    resolver = GitHubArtifactResolver(
        github_client,
        settings.github,
        settings.scopes,
    )
    materializer = GitHubArtifactMaterializer(
        github_client,
        settings.github,
        tmp_path / "publisher-state",
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
    )
    current_state = CurrentPointerState(
        pointer=None,
        head_commit_sha=plan.current_release_observation.default_branch_head_commit_sha,
    )
    resolver.read_current_pointer_state = (
        lambda scope_id, artifact_kind, allow_missing: current_state
    )
    monkeypatch.setattr(
        GitHubExpertCurrentReleaseProvider,
        "observe_task_evaluation_current",
        lambda self, scope_id: plan.current_release_observation,
    )
    captured = {}

    def publish(generic, envelope, *, activation_authorization=None):
        manifest_payload = (
            envelope.source_tree / envelope.manifest_relative_path
        ).read_bytes()
        manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_payload)
        source_files = _source_file_projection(envelope.source_tree)
        materialized_digest = generic.package_validator.validate_local_package(
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=tree_or_blob_digest(manifest_payload),
            assets=envelope.assets,
            source_files=source_files,
        )
        gate = activation_authorization.verifier_for(generic, envelope)
        gate.validate_before_publication(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            current_state=current_state,
            manifest=manifest,
            source_tree_digest=source_tree_digest(
                {
                    relative_path: (
                        digest,
                        mode,
                        (envelope.source_tree / relative_path).stat().st_size,
                    )
                    for relative_path, (digest, mode) in source_files.items()
                }
            ),
            manifest_digest=tree_or_blob_digest(manifest_payload),
        )
        pointer = _release_pointer(plan, materialized_digest)
        gate.revalidate_before_activation(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            pointer=pointer,
            manifest=manifest,
        )
        captured.update(
            {
                "envelope": envelope,
                "gate": gate,
                "manifest": manifest,
                "pointer": pointer,
                "source_digest": plan.publication_source_tree_digest,
            }
        )
        return PublicationTelemetry(
            publication_record=pointer.publication_record,
            expected_parent_sha=envelope.expected_parent_sha,
            source_commit_sha=pointer.publication_record.commit_sha,
            pointer_commit_sha="c" * 40,
            source_tree_digest=plan.publication_source_tree_digest,
            validation_closure_ids=plan.validation_closure_ids,
            idempotent_replay=False,
        )

    monkeypatch.setattr(AutonomousGitHubPublisher, "publish", publish)

    result = publisher.publish(candidate_id=plan.candidate_id)

    assert result.reservation == reservation
    assert result.package == package
    assert result.telemetry.publication_record.artifact_id == plan.release_id
    assert captured["gate"].preflight_mode == "parent"
    assert tuple(asset.name for asset in captured["envelope"].assets) == tuple(
        asset.name for asset in plan.assets
    )
    assert authority.calls.count("denylist") == 3
    assert captured["pointer"].publication_record.publication_id in (
        authority.denylist.checked_subject_ids
    )

    active_observation = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=plan.scope_id,
        release_id=plan.release_id,
        publication_id=captured["pointer"].publication_record.publication_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        repository_node_id=plan.current_release_observation.repository_node_id,
        default_branch_head_commit_sha="c" * 40,
        current_pointer_digest=tree_or_blob_digest(captured["pointer"].to_json_bytes()),
        validation_closure_ids=plan.validation_closure_ids,
    )
    monkeypatch.setattr(
        GitHubExpertCurrentReleaseProvider,
        "observe_task_evaluation_current",
        lambda self, scope_id: active_observation,
    )
    active_gate = ExpertReleasePublicationGate(publisher, reservation)
    active_gate.validate_before_publication(
        envelope=captured["envelope"],
        repositories=settings.scopes.resolve(plan.scope_id),
        current_state=CurrentPointerState(
            pointer=captured["pointer"],
            head_commit_sha="c" * 40,
        ),
        manifest=captured["manifest"],
        source_tree_digest=captured["source_digest"],
        manifest_digest=plan.manifest_digest,
    )

    assert active_gate.preflight_mode == "active-release"
    assert authority.calls.count("denylist") == 3

    with pytest.raises(ExpertReleasePublicationError, match="immutable"):
        publisher.security_denylist_authority = object()
    generic_publisher.resolver = object()
    with pytest.raises(ExpertReleasePublicationError, match="binding changed"):
        publisher.publish(candidate_id=plan.candidate_id)
