import io
import os
import shutil
import struct
import tarfile
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
import zstandard

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    ScopeRepositorySettings,
    SourceFileDescriptor,
)
from kapso.cross_run.expert import ExpertParentTreeReceipt
from kapso.cross_run.github.materializer import (
    CacheCorruptionError,
    GitHubArtifactMaterializer,
    MaterializationError,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.github.publisher import ReleaseAssetInput
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
    release_attestation_reference,
)
from kapso.cross_run.settings import CrossRunSettings
from tests.cross_run_github_fixtures import release_attestation

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
REPOSITORY = "Leeroo-AI/kapso-knowledge"


def extract_verified_source_archive(
    materializer,
    *,
    materialized,
    expected,
    destination,
):
    with ExitStack() as descriptors:
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, parent_descriptor)
        return materializer.extract_verified_source_archive(
            materialized=materialized,
            expected=expected,
            destination=destination,
            destination_parent_descriptor=parent_descriptor,
        )


def test_source_extraction_receipt_rejects_file_directory_collision():
    files = (
        SourceFileDescriptor(
            relative_path="src/node",
            digest=tree_or_blob_digest(b"node"),
            mode="100644",
            size=4,
        ),
        SourceFileDescriptor(
            relative_path="src/node/child.py",
            digest=tree_or_blob_digest(b"child"),
            mode="100644",
            size=5,
        ),
    )
    tree_hash = tree_or_blob_digest(b"intentionally-colliding-source-tree")

    with pytest.raises(MaterializationError, match="file/directory collision"):
        SourceArchiveExtractionReceipt.mint(
            artifact_id=content_id("fixture", {"artifact": "release"}),
            source_archive_ref="expert-source.tar.zst",
            source_archive_digest=tree_or_blob_digest(b"archive"),
            source_tree_hash=tree_hash,
            source_tree_files=files,
            extractor_version="kapso-source-extractor-v1",
        )


def github_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).github


def snapshot_manifest(data, generation=1, additional_checksums=None):
    checksums = {"data.txt": tree_or_blob_digest(data)}
    checksums.update(additional_checksums or {})
    return KnowledgeSnapshotManifest.mint(
        scope_contract_id=content_id("fixture", {"scope": "ml_ai"}),
        scope_id="ml_ai",
        parent_snapshot_ids=(),
        included_bundle_ids=(),
        admitted_episode_ids=(),
        admitted_prior_idea_ids=(),
        active_claim_revision_ids=(),
        catalog_generation=generation,
        configuration_fingerprint=tree_or_blob_digest(b"config"),
        entry_state_refs=(),
        included_assertion_ids=(),
        included_revocation_ids=(),
        proof_dependency_closure_ids=(),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        embedding_sidecars=(),
        prompt_budget_policy={"maximum_records": 1},
        checksums=checksums,
        published_at="2026-07-20T15:00:00Z",
        publisher_attestation={"issuer": "fixture"},
    )


def tar_payload(files, symlink=None, executable_files=()):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as package:
        for name, payload in files:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mtime = 0
            member.mode = 0o755 if name in executable_files else 0o644
            package.addfile(member, io.BytesIO(payload))
        if symlink is not None:
            member = tarfile.TarInfo(symlink)
            member.type = tarfile.SYMTYPE
            member.linkname = "data.txt"
            package.addfile(member)
    return buffer.getvalue()


def resolved_fixture(
    data=b"scientific evidence",
    generation=1,
    archive_name="snapshot.tar",
    extra_files=(),
    declare_extra_files=True,
):
    manifest = snapshot_manifest(
        data,
        generation,
        (
            {name: tree_or_blob_digest(payload) for name, payload in extra_files}
            if declare_extra_files
            else {}
        ),
    )
    source_payloads = (
        ("data.txt", data),
        ("snapshot.json", manifest.to_json_bytes()),
        *extra_files,
    )
    archive = tar_payload(source_payloads)
    if archive_name.endswith(".zst"):
        archive = zstandard.ZstdCompressor().compress(archive)
    commit_sha = f"{generation:x}" * 40
    tag = f"knowledge/S{generation:06d}"
    asset = GitHubReleaseAsset(
        asset_id=str(generation),
        name=archive_name,
        media_type=(
            "application/zstd" if archive_name.endswith(".zst") else "application/x-tar"
        ),
        size=len(archive),
        sha256=tree_or_blob_digest(archive),
    )
    attestation = release_attestation(
        REPOSITORY,
        tag,
        commit_sha,
        {asset.name: asset.sha256},
    )
    record = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=manifest.snapshot_id,
        repository_node_id="repository-node",
        repository_full_name=REPOSITORY,
        commit_sha=commit_sha,
        immutable_release_id=str(generation),
        tag=tag,
        assets=(asset,),
        release_attestation_ref=release_attestation_reference(attestation),
        published_at="2026-07-20T15:00:00Z",
        publisher_identity="leeroo-coder",
    )
    package_tree_digest = source_tree_digest(
        {
            name: (tree_or_blob_digest(payload), "100644", len(payload))
            for name, payload in source_payloads
        }
    )
    pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=record,
        publication_intent_digest=tree_or_blob_digest(b"intent"),
        source_tree_digest=package_tree_digest,
        source_git_tree_sha="e" * 40,
        materialized_tree_digest=package_tree_digest,
        manifest_relative_path="snapshot.json",
        manifest_digest=tree_or_blob_digest(manifest.to_json_bytes()),
        validation_closure_ids=(content_id("fixture", {"review": generation}),),
    )
    policy = RepositoryPolicyReport(
        repository_full_name=REPOSITORY,
        repository_node_id="repository-node",
        private=True,
        default_branch="main",
        authenticated_actor="leeroo-coder",
        write_access=True,
        immutable_releases=True,
    )
    resolved = ResolvedGitHubArtifact(
        repositories=ScopeRepositorySettings(
            scope_id="ml_ai",
            expert_repository="Leeroo-AI/kapso-expert",
            knowledge_repository=REPOSITORY,
            security_repository="Leeroo-AI/kapso-security",
        ),
        pointer=pointer,
        policy=policy,
        pointer_commit_sha="f" * 40,
    )
    return resolved, archive


class DownloadClient:
    def __init__(self, payloads, repository=REPOSITORY):
        self.payloads = dict(payloads)
        self.repository = repository
        self.downloads = []

    def download_release_asset(self, repository, asset_id, destination, maximum_bytes):
        assert repository == self.repository
        self.downloads.append(asset_id)
        payload = self.payloads[asset_id]
        assert len(payload) <= maximum_bytes
        destination.write_bytes(payload)
        return destination


def test_materializer_atomically_commits_read_only_cache_and_reuses_it(tmp_path):
    resolved, archive = resolved_fixture(archive_name="snapshot.tar.zst")
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(
        client, github_settings(), tmp_path / "cache"
    )

    first = materializer.materialize(resolved)
    second = materializer.materialize(resolved)

    assert not first.reused
    assert second.reused
    assert client.downloads == ["1"]
    assert (first.content / "data.txt").read_bytes() == b"scientific evidence"
    assert first.receipt.artifact_id == resolved.pointer.publication_record.artifact_id
    assert tuple(receipt.artifact_id for receipt in materializer.inspect()) == (
        first.receipt.artifact_id,
    )
    for path in (first.root, *tuple(first.root.rglob("*"))):
        assert path.stat().st_mode & 0o222 == 0


def test_local_package_validation_hashes_the_staged_asset(monkeypatch, tmp_path):
    resolved, archive = resolved_fixture()
    asset_path = tmp_path / "snapshot.tar"
    asset_path.write_bytes(archive)
    release_asset = ReleaseAssetInput(
        path=asset_path,
        name=asset_path.name,
        media_type="application/x-tar",
        size=len(archive),
        sha256=tree_or_blob_digest(archive),
    )

    def replace_during_copy(source, destination):
        assert source == asset_path
        Path(destination).write_bytes(b"x" * len(archive))
        return destination

    monkeypatch.setattr(shutil, "copyfile", replace_during_copy)

    with pytest.raises(MaterializationError, match="local release asset digest"):
        GitHubArtifactMaterializer(
            DownloadClient({}), github_settings(), tmp_path / "cache"
        ).validate_local_package(
            artifact_kind=resolved.pointer.publication_record.artifact_kind,
            artifact_id=resolved.pointer.publication_record.artifact_id,
            manifest_relative_path=resolved.pointer.manifest_relative_path,
            manifest_digest=resolved.pointer.manifest_digest,
            assets=(release_asset,),
            source_files={},
        )


def test_materializer_accepts_manifest_bound_asset_only_search_content(tmp_path):
    resolved, archive = resolved_fixture(
        extra_files=(("search/index.bin", b"rebuildable semantic index"),)
    )
    resolved = replace(
        resolved,
        pointer=replace(
            resolved.pointer,
            source_tree_digest=tree_or_blob_digest(b"small Git metadata tree"),
        ),
    )

    materialized = GitHubArtifactMaterializer(
        DownloadClient({"1": archive}), github_settings(), tmp_path / "cache"
    ).materialize(resolved)

    assert (materialized.content / "search/index.bin").read_bytes() == (
        b"rebuildable semantic index"
    )
    assert materialized.receipt.materialized_tree_digest == (
        resolved.pointer.materialized_tree_digest
    )


def test_materializer_accepts_split_expert_source_and_release_assets(
    tmp_path,
    monkeypatch,
):
    expert_repository = "Leeroo-AI/kapso-expert"
    source_payload = b"def train():\n    return 'validated'\n"
    source_archive = tar_payload((("main.py", source_payload),))
    evidence_payload = b'{"evidence":"approved"}'
    evidence_archive = tar_payload(
        (("release-evidence/manifest.json", evidence_payload),)
    )
    test_summary = b'{"fresh_task":"passed"}'
    repository_map_id = content_id("expert-repository-map", {"repository_map": 1})
    approval_id = content_id("fixture", {"approval": 1})
    scope_contract_id = content_id("expert-scope-contract", {"scope": "ml_ai"})
    release_ids = {
        namespace: content_id(namespace, {"fixture": "expert-release"})
        for namespace in (
            "expert-candidate",
            "expert-candidate-commit",
            "expert-source-tree",
            "expert-agent-proposal-derivation",
            "expert-candidate-validation-context",
            "expert-candidate-patch",
            "expert-candidate-sanitation",
            "expert-validation-attempt",
            "expert-validation-transition",
            "expert-candidate-validation-state",
            "expert-publication-eligibility-stage-result",
            "expert-release-matrix-stage-result",
            "expert-release-matrix-report",
            "expert-release-matrix-promotion-decision",
            "expert-validation-policy",
            "expert-release-evidence-manifest",
            "expert-release-matrix-summary",
            "expert-module-contract",
        )
    }
    dependencies = tuple(
        sorted(
            {
                scope_contract_id,
                repository_map_id,
                approval_id,
                *release_ids.values(),
            }
        )
    )
    manifest = ExpertBaseReleaseManifest.mint(
        scope_contract_id=scope_contract_id,
        scope_id="ml_ai",
        parent_release_id=None,
        candidate_id=release_ids["expert-candidate"],
        candidate_commit_record_id=release_ids["expert-candidate-commit"],
        candidate_tree_ref=release_ids["expert-source-tree"],
        candidate_tree_hash=tree_or_blob_digest(b"candidate-tree"),
        candidate_derivation_ref=release_ids["expert-agent-proposal-derivation"],
        candidate_validation_context_ref=release_ids[
            "expert-candidate-validation-context"
        ],
        candidate_patch_ref=release_ids["expert-candidate-patch"],
        candidate_sanitation_report_id=release_ids["expert-candidate-sanitation"],
        candidate_ancestor_ids=(),
        candidate_source_dependency_ids=(scope_contract_id,),
        repository_map_ref=repository_map_id,
        module_contract_refs=(release_ids["expert-module-contract"],),
        module_versions={"shared.runner": "v1"},
        semantic_book_digest=tree_or_blob_digest(b"EXPERT_REPO.md"),
        validation_attempt_id=release_ids["expert-validation-attempt"],
        approval_transition_id=release_ids["expert-validation-transition"],
        approval_state_id=release_ids["expert-candidate-validation-state"],
        publication_eligibility_result_id=release_ids[
            "expert-publication-eligibility-stage-result"
        ],
        release_matrix_stage_result_id=release_ids[
            "expert-release-matrix-stage-result"
        ],
        release_matrix_report_id=release_ids["expert-release-matrix-report"],
        promotion_decision_id=release_ids["expert-release-matrix-promotion-decision"],
        approval_assertion_ids=(approval_id,),
        validation_policy_id=release_ids["expert-validation-policy"],
        configuration_fingerprint=tree_or_blob_digest(b"expert-config"),
        source_archive_ref="expert-source.tar",
        evidence_archive_ref="expert-evidence.tar",
        evidence_manifest_ref=release_ids["expert-release-evidence-manifest"],
        test_matrix_summary_ref=release_ids["expert-release-matrix-summary"],
        evidence_dependency_ids=dependencies,
        dependency_closure_ids=dependencies,
        checksums={
            "expert-source.tar": tree_or_blob_digest(source_archive),
            "expert-evidence.tar": tree_or_blob_digest(evidence_archive),
            "main.py": tree_or_blob_digest(source_payload),
            "release-evidence/manifest.json": tree_or_blob_digest(evidence_payload),
            "test-summary.json": tree_or_blob_digest(test_summary),
        },
    )
    manifest_payload = manifest.to_json_bytes()
    control_archive = tar_payload((("expert-release.json", manifest_payload),))
    asset_payloads = {
        "control.tar": control_archive,
        "expert-source.tar": source_archive,
        "expert-evidence.tar": evidence_archive,
        "test-summary.json": test_summary,
    }
    assets = tuple(
        GitHubReleaseAsset(
            asset_id=str(position),
            name=name,
            media_type=(
                "application/x-tar" if name.endswith(".tar") else "application/json"
            ),
            size=len(payload),
            sha256=tree_or_blob_digest(payload),
        )
        for position, (name, payload) in enumerate(
            sorted(asset_payloads.items()), start=1
        )
    )
    tag = "expert/E000001"
    commit_sha = "7" * 40
    attestation = release_attestation(
        expert_repository,
        tag,
        commit_sha,
        {asset.name: asset.sha256 for asset in assets},
    )
    record = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=manifest.release_id,
        repository_node_id="expert-repository-node",
        repository_full_name=expert_repository,
        commit_sha=commit_sha,
        immutable_release_id="7",
        tag=tag,
        assets=assets,
        release_attestation_ref=release_attestation_reference(attestation),
        published_at="2026-07-20T15:00:00Z",
        publisher_identity="leeroo-coder",
    )
    materialized_digest = source_tree_digest(
        {
            "expert-release.json": (
                tree_or_blob_digest(manifest_payload),
                "100644",
                len(manifest_payload),
            ),
            "main.py": (
                tree_or_blob_digest(source_payload),
                "100644",
                len(source_payload),
            ),
            "release-evidence/manifest.json": (
                tree_or_blob_digest(evidence_payload),
                "100644",
                len(evidence_payload),
            ),
        }
    )
    pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=record,
        publication_intent_digest=tree_or_blob_digest(b"intent"),
        source_tree_digest=tree_or_blob_digest(b"Git metadata tree"),
        source_git_tree_sha="8" * 40,
        materialized_tree_digest=materialized_digest,
        manifest_relative_path="expert-release.json",
        manifest_digest=tree_or_blob_digest(manifest_payload),
        validation_closure_ids=(approval_id,),
    )
    resolved = ResolvedGitHubArtifact(
        repositories=ScopeRepositorySettings(
            scope_id="ml_ai",
            expert_repository=expert_repository,
            knowledge_repository=REPOSITORY,
            security_repository="Leeroo-AI/kapso-security",
        ),
        pointer=pointer,
        policy=RepositoryPolicyReport(
            repository_full_name=expert_repository,
            repository_node_id="expert-repository-node",
            private=True,
            default_branch="main",
            authenticated_actor="leeroo-coder",
            write_access=True,
            immutable_releases=True,
        ),
        pointer_commit_sha="9" * 40,
    )

    materializer = GitHubArtifactMaterializer(
        DownloadClient(
            {asset.asset_id: asset_payloads[asset.name] for asset in assets},
            expert_repository,
        ),
        github_settings(),
        tmp_path / "expert-cache",
    )
    materialized = materializer.materialize(resolved)
    source_receipt = materializer.inspect_source_archive(
        materialized,
        manifest.source_archive_ref,
    )
    expert_source_snapshot = materializer.inspect_expert_release_source(
        materialized,
        maximum_entries=10,
        maximum_bytes=len(source_payload),
    )
    assert expert_source_snapshot.release_manifest == manifest
    assert expert_source_snapshot.source_extraction_receipt == source_receipt
    assert dict(expert_source_snapshot.source_contents) == {"main.py": source_payload}
    with pytest.raises(MaterializationError, match="configured size limit"):
        materializer.inspect_expert_release_source(
            materialized,
            maximum_entries=10,
            maximum_bytes=len(source_payload) - 1,
        )
    extracted_source = (tmp_path / "extracted-source").resolve()
    assert (
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=extracted_source,
        )
        == source_receipt
    )

    assert (materialized.content / "main.py").read_bytes() == source_payload
    assert (extracted_source / "main.py").read_bytes() == source_payload
    assert (materialized.assets / "test-summary.json").read_bytes() == test_summary
    assert source_receipt.source_tree_hash == source_tree_digest(
        {
            "main.py": (
                tree_or_blob_digest(source_payload),
                "100644",
                len(source_payload),
            )
        }
    )
    assert source_receipt.source_tree_hash != (
        materialized.receipt.materialized_tree_digest
    )
    assert (
        source_receipt.source_archive_digest
        == manifest.checksums[manifest.source_archive_ref]
    )
    assert tuple(file.relative_path for file in source_receipt.source_tree_files) == (
        "main.py",
    )
    pinned_parent = (tmp_path / "pinned-extraction-parent").resolve()
    pinned_parent.mkdir(mode=0o700)
    pinned_parent.chmod(0o700)
    pinned_destination = pinned_parent / "source"
    moved_parent = (tmp_path / "moved-extraction-parent").resolve()
    with ExitStack() as descriptors:
        pinned_parent_descriptor = os.open(
            pinned_parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, pinned_parent_descriptor)
        pinned_parent.rename(moved_parent)
        pinned_parent.mkdir(mode=0o700)
        with pytest.raises(MaterializationError, match="pinned descriptor"):
            materializer.extract_verified_source_archive(
                materialized=materialized,
                expected=source_receipt,
                destination=pinned_destination,
                destination_parent_descriptor=pinned_parent_descriptor,
            )
    assert tuple(pinned_parent.iterdir()) == ()
    assert tuple(moved_parent.iterdir()) == ()
    with pytest.raises(MaterializationError, match="reference is invalid"):
        replace(source_receipt, source_archive_ref="source.bin")
    with pytest.raises(MaterializationError, match="must be absent"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=extracted_source,
        )
    symlink_destination = (tmp_path / "linked-extraction").resolve()
    symlink_destination.symlink_to(extracted_source, target_is_directory=True)
    with pytest.raises(MaterializationError, match="must be absent"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=symlink_destination,
        )
    public_parent = (tmp_path / "public-extraction-parent").resolve()
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    with pytest.raises(MaterializationError, match="parent must be private"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=public_parent / "source",
        )
    wrong_destination = (tmp_path / "wrong-extraction").resolve()
    wrong_source_receipt = SourceArchiveExtractionReceipt.mint(
        artifact_id=source_receipt.artifact_id,
        source_archive_ref=source_receipt.source_archive_ref,
        source_archive_digest=tree_or_blob_digest(b"another archive"),
        source_tree_hash=source_receipt.source_tree_hash,
        source_tree_files=source_receipt.source_tree_files,
        extractor_version=source_receipt.extractor_version,
    )
    with pytest.raises(MaterializationError, match="differs from verified asset"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=wrong_source_receipt,
            destination=wrong_destination,
        )
    assert not wrong_destination.exists()
    wrong_file = SourceFileDescriptor(
        relative_path="main.py",
        digest=tree_or_blob_digest(b"different source"),
        mode="100644",
        size=len(b"different source"),
    )
    wrong_tree_receipt = SourceArchiveExtractionReceipt.mint(
        artifact_id=source_receipt.artifact_id,
        source_archive_ref=source_receipt.source_archive_ref,
        source_archive_digest=source_receipt.source_archive_digest,
        source_tree_hash=source_tree_digest(
            {
                wrong_file.relative_path: (
                    wrong_file.digest,
                    wrong_file.mode,
                    wrong_file.size,
                )
            }
        ),
        source_tree_files=(wrong_file,),
        extractor_version=source_receipt.extractor_version,
    )
    wrong_tree_destination = (tmp_path / "wrong-tree-extraction").resolve()
    with pytest.raises(MaterializationError, match="differs from expected receipt"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=wrong_tree_receipt,
            destination=wrong_tree_destination,
        )
    assert not wrong_tree_destination.exists()
    original_publish = materializer._publish_source_tree

    def substitute_staged_source(
        staged_source,
        destination,
        destination_parent_descriptor,
        expected_parent_identity,
        expected,
    ):
        (staged_source / "main.py").write_bytes(b"attacker substitution")
        return original_publish(
            staged_source,
            destination,
            destination_parent_descriptor,
            expected_parent_identity,
            expected,
        )

    monkeypatch.setattr(
        materializer,
        "_publish_source_tree",
        substitute_staged_source,
    )
    substituted_destination = (tmp_path / "substituted-extraction").resolve()
    with pytest.raises(MaterializationError, match="at publication"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=substituted_destination,
        )
    assert not substituted_destination.exists()

    def inject_empty_git_directory(
        staged_source,
        destination,
        destination_parent_descriptor,
        expected_parent_identity,
        expected,
    ):
        (staged_source / ".git").mkdir()
        return original_publish(
            staged_source,
            destination,
            destination_parent_descriptor,
            expected_parent_identity,
            expected,
        )

    monkeypatch.setattr(
        materializer,
        "_publish_source_tree",
        inject_empty_git_directory,
    )
    git_injected_destination = (tmp_path / "git-injected-extraction").resolve()
    with pytest.raises(MaterializationError, match="unsafe path"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=git_injected_destination,
        )
    assert not git_injected_destination.exists()
    outside_source = tmp_path / "outside-source.py"
    outside_source.write_bytes(source_payload)

    def inject_hardlinked_source(
        staged_source,
        destination,
        destination_parent_descriptor,
        expected_parent_identity,
        expected,
    ):
        (staged_source / "main.py").unlink()
        os.link(outside_source, staged_source / "main.py")
        return original_publish(
            staged_source,
            destination,
            destination_parent_descriptor,
            expected_parent_identity,
            expected,
        )

    monkeypatch.setattr(
        materializer,
        "_publish_source_tree",
        inject_hardlinked_source,
    )
    hardlinked_destination = (tmp_path / "hardlinked-extraction").resolve()
    with pytest.raises(MaterializationError, match="must be independent"):
        extract_verified_source_archive(
            materializer,
            materialized=materialized,
            expected=source_receipt,
            destination=hardlinked_destination,
        )
    assert not hardlinked_destination.exists()
    parent_receipt = ExpertParentTreeReceipt.mint(
        release_id=manifest.release_id,
        cache_verification_receipt=materialized.receipt,
        source_extraction_receipt=source_receipt,
        parent_tree_hash=source_receipt.source_tree_hash,
        repository_map_id=repository_map_id,
        module_contract_ids=(content_id("fixture", {"module": 1}),),
        materializer_version="kapso.expert_materializer.v1",
    )
    assert parent_receipt.parent_tree_hash != (
        parent_receipt.cache_verification_receipt.materialized_tree_digest
    )


def test_content_cache_reuses_authorized_relocation_of_same_artifact(tmp_path):
    resolved, archive = resolved_fixture()
    state_root = tmp_path / "cache"
    first = GitHubArtifactMaterializer(
        DownloadClient({"1": archive}), github_settings(), state_root
    ).materialize(resolved)
    original = resolved.pointer.publication_record
    relocated_asset = replace(original.assets[0], asset_id="relocated-asset")
    relocated_repository = "Leeroo-AI/kapso-knowledge-relocated"
    relocated_record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in original.to_dict().items()
            if key
            not in {
                "publication_id",
                "repository_node_id",
                "repository_full_name",
                "commit_sha",
                "immutable_release_id",
                "tag",
                "assets",
                "release_attestation_ref",
            }
        },
        repository_node_id="relocated-repository-node",
        repository_full_name=relocated_repository,
        commit_sha="a" * 40,
        immutable_release_id="relocated-release",
        tag="knowledge/S000001-relocated",
        assets=(relocated_asset,),
        release_attestation_ref=release_attestation_reference(
            release_attestation(
                relocated_repository,
                "knowledge/S000001-relocated",
                "a" * 40,
                {relocated_asset.name: relocated_asset.sha256},
            )
        ),
    )
    relocated = replace(
        resolved,
        repositories=replace(
            resolved.repositories, knowledge_repository=relocated_repository
        ),
        pointer=replace(
            resolved.pointer,
            publication_record=relocated_record,
        ),
        policy=replace(
            resolved.policy,
            repository_full_name=relocated_repository,
            repository_node_id="relocated-repository-node",
        ),
    )
    relocation_client = DownloadClient({})

    reused = GitHubArtifactMaterializer(
        relocation_client, github_settings(), state_root
    ).materialize(relocated)

    assert reused.reused
    assert reused.root == first.root
    assert reused.receipt == first.receipt
    assert relocation_client.downloads == []


def test_materializer_rejects_package_with_different_source_file_mode(tmp_path):
    resolved, _ = resolved_fixture()
    manifest = snapshot_manifest(b"scientific evidence")
    archive = tar_payload(
        (
            ("data.txt", b"scientific evidence"),
            ("snapshot.json", manifest.to_json_bytes()),
        ),
        executable_files=("data.txt",),
    )
    asset = replace(
        resolved.pointer.publication_record.assets[0],
        size=len(archive),
        sha256=tree_or_blob_digest(archive),
    )
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=(asset,),
    )
    mismatched = replace(
        resolved,
        pointer=replace(resolved.pointer, publication_record=record),
    )

    with pytest.raises(MaterializationError, match="package descriptor"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive}), github_settings(), tmp_path / "cache"
        ).materialize(mismatched)


def test_manifest_control_bound_applies_to_download_and_cache_reuse(tmp_path):
    resolved, archive = resolved_fixture()
    state_root = tmp_path / "cache"
    materialized = GitHubArtifactMaterializer(
        DownloadClient({"1": archive}), github_settings(), state_root
    ).materialize(resolved)
    manifest_size = (materialized.content / "snapshot.json").stat().st_size
    constrained = replace(github_settings(), control_blob_size_bytes=manifest_size - 1)

    with pytest.raises(CacheCorruptionError, match="control bound"):
        GitHubArtifactMaterializer(
            DownloadClient({}), constrained, state_root
        ).materialize(resolved)

    fresh_client = DownloadClient({"1": archive})
    with pytest.raises(MaterializationError, match="control bound"):
        GitHubArtifactMaterializer(
            fresh_client, constrained, tmp_path / "fresh-cache"
        ).materialize(resolved)
    assert fresh_client.downloads == ["1"]


def test_corrupt_cache_raises_without_unrecorded_redownload(tmp_path):
    resolved, archive = resolved_fixture()
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(
        client, github_settings(), tmp_path / "cache"
    )
    materialized = materializer.materialize(resolved)
    data_path = materialized.content / "data.txt"
    data_path.chmod(0o644)
    data_path.write_bytes(b"tampered")

    with pytest.raises(CacheCorruptionError):
        materializer.materialize(resolved)

    assert client.downloads == ["1"]


def test_sparse_corrupt_cache_fails_size_bound_before_hashing(tmp_path):
    resolved, archive = resolved_fixture()
    settings = github_settings()
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(client, settings, tmp_path / "cache")
    materialized = materializer.materialize(resolved)
    data_path = materialized.content / "data.txt"
    materialized.root.chmod(0o755)
    materialized.content.chmod(0o755)
    data_path.chmod(0o644)
    with data_path.open("r+b") as file_handle:
        file_handle.truncate(settings.materialized_asset_size_bytes + 1)

    with pytest.raises(CacheCorruptionError, match="size bound"):
        materializer.materialize(resolved)

    assert client.downloads == ["1"]


def test_forged_receipt_cannot_authenticate_content_not_reproduced_by_assets(tmp_path):
    resolved, archive = resolved_fixture()
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(
        client, github_settings(), tmp_path / "cache"
    )
    materialized = materializer.materialize(resolved)
    extra = materialized.content / "forged.txt"
    materialized.content.chmod(0o755)
    extra.write_bytes(b"not present in the immutable release asset")
    extra.chmod(0o444)
    materialized.content.chmod(0o555)
    receipt_path = materialized.root / "VERIFIED.json"
    forged_receipt = replace(
        materialized.receipt,
        cache_tree_digest=materializer._tree_digest(
            materialized.root, ignore_root_receipt=True
        ),
    )
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(forged_receipt.to_json_bytes())
    receipt_path.chmod(0o444)

    with pytest.raises(CacheCorruptionError):
        materializer.materialize(resolved)

    assert client.downloads == ["1"]


def test_nested_receipt_filename_is_authenticated_as_release_content(tmp_path):
    resolved, archive = resolved_fixture(
        extra_files=(("nested/VERIFIED.json", b"authentic"),)
    )
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(
        client, github_settings(), tmp_path / "cache"
    )
    materialized = materializer.materialize(resolved)
    nested = materialized.content / "nested" / "VERIFIED.json"
    nested.chmod(0o644)
    nested.write_bytes(b"tampered")
    nested.chmod(0o444)

    with pytest.raises(CacheCorruptionError):
        materializer.materialize(resolved)

    assert client.downloads == ["1"]


def test_materializer_rejects_unchecksummed_archive_payload(tmp_path):
    resolved, archive = resolved_fixture(
        extra_files=(("untracked.txt", b"not in the manifest closure"),),
        declare_extra_files=False,
    )

    with pytest.raises(MaterializationError, match="not closed"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive}), github_settings(), tmp_path / "cache"
        ).materialize(resolved)


def test_materializer_rejects_opaque_asset_outside_manifest_closure(tmp_path):
    resolved, archive = resolved_fixture()
    opaque_payload = b"not bound by the artifact manifest"
    opaque = GitHubReleaseAsset(
        asset_id="2",
        name="credentials.bin",
        media_type="application/octet-stream",
        size=len(opaque_payload),
        sha256=tree_or_blob_digest(opaque_payload),
    )
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=tuple(
            sorted(
                (*resolved.pointer.publication_record.assets, opaque),
                key=lambda asset: asset.name,
            )
        ),
    )
    pointer = CurrentArtifactPointer(
        **{
            **resolved.pointer.to_dict(),
            "publication_record": record.to_dict(),
        }
    )

    with pytest.raises(MaterializationError, match="outside manifest closure"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive, "2": opaque_payload}),
            github_settings(),
            tmp_path / "cache",
        ).materialize(replace(resolved, pointer=pointer))


def test_partial_download_never_becomes_visible_cache_entry(tmp_path):
    resolved, archive = resolved_fixture()
    client = DownloadClient({"1": archive[:-1]})
    cache_root = tmp_path / "cache"
    materializer = GitHubArtifactMaterializer(client, github_settings(), cache_root)

    with pytest.raises(MaterializationError):
        materializer.materialize(resolved)

    committed = [
        path
        for path in cache_root.rglob("*")
        if path.is_dir() and not path.name.startswith(".staging-")
    ]
    assert all(not (path / "VERIFIED.json").exists() for path in committed)
    assert not tuple(cache_root.rglob(".staging-*"))


@pytest.mark.parametrize(
    "unsafe_kind",
    ["traversal", "symlink", "checksum", "git_directory", "gitmodules"],
)
def test_materializer_rejects_unsafe_archive_and_checksum_shapes(tmp_path, unsafe_kind):
    resolved, _ = resolved_fixture()
    manifest = snapshot_manifest(b"scientific evidence")
    if unsafe_kind == "traversal":
        archive = tar_payload(
            (("../escape", b"bad"), ("snapshot.json", manifest.to_json_bytes()))
        )
    elif unsafe_kind == "symlink":
        archive = tar_payload(
            (
                ("data.txt", b"scientific evidence"),
                ("snapshot.json", manifest.to_json_bytes()),
            ),
            symlink="linked",
        )
    elif unsafe_kind == "git_directory":
        archive = tar_payload(
            (
                (".git/config", b"forbidden history"),
                ("snapshot.json", manifest.to_json_bytes()),
            )
        )
    elif unsafe_kind == "gitmodules":
        archive = tar_payload(
            (
                (".gitmodules", b"forbidden submodule"),
                ("snapshot.json", manifest.to_json_bytes()),
            )
        )
    else:
        archive = tar_payload(
            (("data.txt", b"wrong"), ("snapshot.json", manifest.to_json_bytes()))
        )
    asset = replace(
        resolved.pointer.publication_record.assets[0],
        size=len(archive),
        sha256=tree_or_blob_digest(archive),
    )
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=(asset,),
    )
    pointer = CurrentArtifactPointer(
        **{
            **resolved.pointer.to_dict(),
            "publication_record": record.to_dict(),
        }
    )
    corrupted = replace(resolved, pointer=pointer)

    with pytest.raises(MaterializationError):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive}), github_settings(), tmp_path / "cache"
        ).materialize(corrupted)


def test_archive_limits_fail_before_extraction_becomes_visible(tmp_path):
    resolved, archive = resolved_fixture()
    constrained = replace(github_settings(), archive_entry_limit=1)

    with pytest.raises(MaterializationError):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive}), constrained, tmp_path / "cache"
        ).materialize(resolved)


def test_materializer_rejects_aggregate_asset_bytes_before_download(tmp_path):
    resolved, archive = resolved_fixture()
    first = resolved.pointer.publication_record.assets[0]
    second = replace(first, asset_id="2", name="sidecar.bin")
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=(first, second),
    )
    pointer = CurrentArtifactPointer(
        **{
            **resolved.pointer.to_dict(),
            "publication_record": record.to_dict(),
        }
    )
    client = DownloadClient({"1": archive, "2": archive})
    constrained = replace(
        github_settings(), materialized_asset_size_bytes=first.size + second.size - 1
    )

    with pytest.raises(MaterializationError, match="closure"):
        GitHubArtifactMaterializer(client, constrained, tmp_path / "cache").materialize(
            replace(resolved, pointer=pointer)
        )

    assert client.downloads == []


def test_archive_entry_limit_counts_directories_and_zstd_has_a_byte_bound(tmp_path):
    resolved, _ = resolved_fixture()
    directory_archive = io.BytesIO()
    with tarfile.open(fileobj=directory_archive, mode="w") as package:
        for name in ("one", "two"):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            package.addfile(member)
    payload = directory_archive.getvalue()
    asset = replace(
        resolved.pointer.publication_record.assets[0],
        size=len(payload),
        sha256=tree_or_blob_digest(payload),
    )
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=(asset,),
    )
    pointer = CurrentArtifactPointer(
        **{
            **resolved.pointer.to_dict(),
            "publication_record": record.to_dict(),
        }
    )
    constrained = replace(github_settings(), archive_entry_limit=1)

    with pytest.raises(MaterializationError, match="entry limit"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": payload}), constrained, tmp_path / "directory-cache"
        ).materialize(replace(resolved, pointer=pointer))

    compressed_resolved, compressed = resolved_fixture(archive_name="snapshot.tar.zst")
    byte_constrained = replace(github_settings(), materialized_asset_size_bytes=1024)
    with pytest.raises(MaterializationError, match="decompressed archive"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": compressed}),
            byte_constrained,
            tmp_path / "compressed-cache",
        ).materialize(compressed_resolved)


def test_archive_entry_limit_counts_implicit_parent_directories(tmp_path):
    resolved, _ = resolved_fixture()
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as package:
        payload = b"x"
        member = tarfile.TarInfo("a/b/c/d/payload.bin")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    archive = archive_buffer.getvalue()
    asset = replace(
        resolved.pointer.publication_record.assets[0],
        size=len(archive),
        sha256=tree_or_blob_digest(archive),
    )
    record = GitHubPublicationRecord.mint(
        **{
            key: value
            for key, value in resolved.pointer.publication_record.to_dict().items()
            if key not in {"publication_id", "assets"}
        },
        assets=(asset,),
    )
    pointer = replace(resolved.pointer, publication_record=record)
    constrained = replace(github_settings(), archive_entry_limit=1)

    with pytest.raises(MaterializationError, match="entry limit"):
        GitHubArtifactMaterializer(
            DownloadClient({"1": archive}),
            constrained,
            tmp_path / "implicit-directory-cache",
        ).materialize(replace(resolved, pointer=pointer))


def test_archive_entry_limit_combines_headers_and_implicit_directories(tmp_path):
    archive = tmp_path / "combined-entry-work.tar"
    with tarfile.open(archive, mode="w") as package:
        payload = b"x"
        file_member = tarfile.TarInfo("parent/payload.bin")
        file_member.size = len(payload)
        package.addfile(file_member, io.BytesIO(payload))
        directory_member = tarfile.TarInfo("parent")
        directory_member.type = tarfile.DIRTYPE
        package.addfile(directory_member)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )
    destination = tmp_path / "content"
    destination.mkdir()

    with pytest.raises(MaterializationError, match="entry limit"):
        materializer._extract_archive(
            archive,
            destination,
            {},
            github_settings().materialized_asset_size_bytes,
            2,
        )


def test_archive_entry_limit_counts_ignored_root_headers(tmp_path):
    archive = tmp_path / "root-headers.tar"
    with tarfile.open(archive, mode="w") as package:
        for _ in range(3):
            member = tarfile.TarInfo(".")
            member.type = tarfile.DIRTYPE
            package.addfile(member)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )
    destination = tmp_path / "content"
    destination.mkdir()

    with pytest.raises(MaterializationError, match="entry limit"):
        materializer._extract_archive(
            archive,
            destination,
            {},
            github_settings().materialized_asset_size_bytes,
            1,
        )


def test_tar_extensions_are_rejected_before_hidden_payload_processing(tmp_path):
    pax_archive = tmp_path / "pax.tar"
    with tarfile.open(
        pax_archive,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": "x" * 4096},
    ) as package:
        package.addfile(tarfile.TarInfo("payload"))
    gnu_archive = tmp_path / "gnu.tar"
    with tarfile.open(gnu_archive, mode="w", format=tarfile.GNU_FORMAT) as package:
        package.addfile(tarfile.TarInfo("nested/" + "x" * 101))
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )

    for position, archive in enumerate((pax_archive, gnu_archive)):
        destination = tmp_path / f"content-{position}"
        destination.mkdir()
        with pytest.raises(MaterializationError, match="extension headers"):
            materializer._extract_archive(
                archive,
                destination,
                {},
                github_settings().materialized_asset_size_bytes,
                github_settings().archive_entry_limit,
            )
        assert tuple(destination.iterdir()) == ()


@pytest.mark.parametrize("member_name", ["", "."])
def test_tar_rejects_hidden_regular_file_members(tmp_path, member_name):
    archive = tmp_path / "hidden-member.tar"
    with tarfile.open(archive, mode="w") as package:
        payload = b"unmanifested payload"
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )
    destination = tmp_path / "content"
    destination.mkdir()

    with pytest.raises(MaterializationError, match="hidden regular-file"):
        materializer._extract_archive(
            archive,
            destination,
            {},
            github_settings().materialized_asset_size_bytes,
            github_settings().archive_entry_limit,
        )

    assert tuple(destination.iterdir()) == ()


def test_zstd_archive_rejects_trailing_and_additional_frames(tmp_path):
    tar_path = tmp_path / "payload.tar"
    with tarfile.open(tar_path, mode="w") as package:
        payload = b"bounded payload"
        member = tarfile.TarInfo("payload.txt")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    compressor = zstandard.ZstdCompressor()
    frame = compressor.compress(tar_path.read_bytes())
    hidden = b"skippable metadata"
    variants = (
        frame + b"trailing bytes",
        frame + compressor.compress(b"second frame"),
        frame + struct.pack("<II", 0x184D2A50, len(hidden)) + hidden,
    )
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )

    for position, encoded in enumerate(variants):
        archive = tmp_path / f"payload-{position}.tar.zst"
        archive.write_bytes(encoded)
        destination = tmp_path / f"content-{position}"
        destination.mkdir()
        with pytest.raises(MaterializationError, match="exactly one canonical frame"):
            materializer._extract_archive(
                archive,
                destination,
                {},
                github_settings().materialized_asset_size_bytes,
                github_settings().archive_entry_limit,
            )
        assert tuple(destination.iterdir()) == ()


def test_zstd_decoder_enforces_configured_window_in_bytes(tmp_path):
    tar_path = tmp_path / "large-window.tar"
    payload = bytes(range(256)) * 12288
    with tarfile.open(tar_path, mode="w") as package:
        member = tarfile.TarInfo("payload.bin")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    constrained = replace(github_settings(), zstd_window_size_bytes=1048576)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}),
        constrained,
        tmp_path / "state",
    )
    accepted_parameters = zstandard.ZstdCompressionParameters.from_level(
        3, window_log=20, write_content_size=0
    )
    accepted_archive = tmp_path / "accepted-window.tar.zst"
    accepted_archive.write_bytes(
        zstandard.ZstdCompressor(
            compression_params=accepted_parameters,
        ).compress(tar_path.read_bytes())
    )
    assert (
        zstandard.get_frame_parameters(accepted_archive.read_bytes()).window_size
        == constrained.zstd_window_size_bytes
    )
    accepted_destination = tmp_path / "accepted-content"
    accepted_destination.mkdir()

    materializer._extract_archive(
        accepted_archive,
        accepted_destination,
        {},
        constrained.materialized_asset_size_bytes,
        constrained.archive_entry_limit,
    )

    rejected_parameters = zstandard.ZstdCompressionParameters.from_level(
        3, window_log=21, write_content_size=0
    )
    rejected_archive = tmp_path / "rejected-window.tar.zst"
    rejected_archive.write_bytes(
        zstandard.ZstdCompressor(
            compression_params=rejected_parameters,
        ).compress(tar_path.read_bytes())
    )
    rejected_destination = tmp_path / "rejected-content"
    rejected_destination.mkdir()

    with pytest.raises(MaterializationError, match="frame window"):
        materializer._extract_archive(
            rejected_archive,
            rejected_destination,
            {},
            constrained.materialized_asset_size_bytes,
            constrained.archive_entry_limit,
        )


def test_plain_tar_transport_rejects_disguised_xz_codec(tmp_path):
    archive = tmp_path / "disguised.tar"
    with tarfile.open(archive, mode="w:xz") as package:
        payload = b"bounded transport"
        member = tarfile.TarInfo("payload.txt")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )
    destination = tmp_path / "content"
    destination.mkdir()

    with pytest.raises(MaterializationError, match="incomplete block"):
        materializer._extract_archive(
            archive,
            destination,
            {},
            github_settings().materialized_asset_size_bytes,
            github_settings().archive_entry_limit,
        )

    assert tuple(destination.iterdir()) == ()


def test_pruning_preserves_pins_and_configured_recent_entries(tmp_path):
    cache_root = tmp_path / "cache"
    settings = replace(github_settings(), cache_retention_releases=1)
    materialized = []
    materializer = None
    for generation in (1, 2, 3):
        resolved, archive = resolved_fixture(
            data=f"evidence-{generation}".encode("utf-8"),
            generation=generation,
        )
        materializer = GitHubArtifactMaterializer(
            DownloadClient({str(generation): archive}), settings, cache_root
        )
        entry = materializer.materialize(resolved)
        os.utime(entry.root, (generation, generation))
        materialized.append(entry)
    assert materializer is not None

    removed = materializer.prune((materialized[0].receipt.artifact_id,))

    assert removed == (materialized[1].receipt.artifact_id,)
    assert materialized[0].root.exists()
    assert not materialized[1].root.exists()
    assert materialized[2].root.exists()


def test_cache_rejects_symlinked_ancestor_and_kind_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_state = tmp_path / "linked-state"
    linked_state.symlink_to(outside, target_is_directory=True)

    with pytest.raises(CacheCorruptionError, match="symlinked ancestor"):
        GitHubArtifactMaterializer(DownloadClient({}), github_settings(), linked_state)

    state = tmp_path / "state"
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), state
    )
    materializer.cache_root.mkdir(parents=True)
    kind_link = materializer.cache_root / "knowledge_snapshot"
    kind_link.symlink_to(outside, target_is_directory=True)

    resolved, archive = resolved_fixture()
    materializer.client = DownloadClient({"1": archive})
    with pytest.raises(CacheCorruptionError, match="kind directory"):
        materializer.materialize(resolved)
    with pytest.raises(CacheCorruptionError):
        materializer.inspect()
    with pytest.raises(CacheCorruptionError):
        materializer.prune(())
    assert outside.is_dir()
    assert tuple(outside.iterdir()) == ()


def test_cache_kind_swap_is_rejected_before_external_download(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    resolved, archive = resolved_fixture()
    client = DownloadClient({"1": archive})
    materializer = GitHubArtifactMaterializer(
        client,
        github_settings(),
        tmp_path / "state",
    )
    original_validation = materializer._validate_cache_kind_directory
    validations = 0

    def swap_after_validation(kind_directory, artifact_kind):
        nonlocal validations
        original_validation(kind_directory, artifact_kind)
        validations += 1
        if validations == 1:
            kind_directory.rmdir()
            kind_directory.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        materializer,
        "_validate_cache_kind_directory",
        swap_after_validation,
    )

    with pytest.raises(CacheCorruptionError, match="kind directory"):
        materializer.materialize(resolved)

    assert client.downloads == []
    assert tuple(outside.iterdir()) == ()


def test_cache_lease_serializes_cooperating_operations(tmp_path):
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), github_settings(), tmp_path / "state"
    )
    started = Event()

    def inspect_after_signal():
        started.set()
        return materializer.inspect()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with materializer._cache_lease():
            pending = executor.submit(inspect_after_signal)
            assert started.wait(timeout=1)
            assert not pending.done()
        assert pending.result(timeout=1) == ()


def test_cache_inspection_rejects_misnamed_content_address(tmp_path):
    resolved, archive = resolved_fixture()
    materializer = GitHubArtifactMaterializer(
        DownloadClient({"1": archive}),
        github_settings(),
        tmp_path / "state",
    )
    materialized = materializer.materialize(resolved)
    wrong = materialized.root.with_name("wrong-content-address")
    materialized.root.rename(wrong)

    with pytest.raises(CacheCorruptionError, match="placement"):
        materializer.inspect()


def test_cache_operations_reclaim_hard_crash_staging_before_entry_bound(tmp_path):
    settings = replace(github_settings(), cache_entry_limit=1)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({}), settings, tmp_path / "state"
    )
    kind = materializer.cache_root / "knowledge_snapshot"
    kind.mkdir(parents=True)
    (kind / ".staging-one").mkdir()
    (kind / ".staging-two").mkdir()
    (kind / ".validation-three").mkdir()
    (kind / ".pruning-four").mkdir()

    assert materializer.inspect() == ()
    assert tuple(kind.iterdir()) == ()


def test_materialization_refuses_to_overfill_cache_before_download(tmp_path):
    settings = replace(github_settings(), cache_entry_limit=1)
    first_resolved, first_archive = resolved_fixture(data=b"first", generation=1)
    second_resolved, second_archive = resolved_fixture(data=b"second", generation=2)
    client = DownloadClient({"1": first_archive, "2": second_archive})
    materializer = GitHubArtifactMaterializer(
        client,
        settings,
        tmp_path / "state",
    )
    first = materializer.materialize(first_resolved)

    with pytest.raises(MaterializationError, match="capacity"):
        materializer.materialize(second_resolved)

    assert client.downloads == ["1"]
    assert tuple(receipt.artifact_id for receipt in materializer.inspect()) == (
        first.receipt.artifact_id,
    )


def test_pruning_crash_hides_partial_deletion_and_allows_rebuild(tmp_path, monkeypatch):
    settings = replace(github_settings(), cache_retention_releases=1)
    cache_state = tmp_path / "state"
    first_resolved, first_archive = resolved_fixture(data=b"first", generation=1)
    second_resolved, second_archive = resolved_fixture(data=b"second", generation=2)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({"1": first_archive, "2": second_archive}),
        settings,
        cache_state,
    )
    first = materializer.materialize(first_resolved)
    second = materializer.materialize(second_resolved)
    os.utime(first.root, (1, 1))
    os.utime(second.root, (2, 2))

    def fail_removal(path):
        raise RuntimeError(f"injected pruning crash: {path.name}")

    monkeypatch.setattr(materializer, "_delete_transient_directory", fail_removal)
    with pytest.raises(RuntimeError, match="injected pruning crash"):
        materializer.prune(())

    tombstones = tuple(first.root.parent.glob(".pruning-*"))
    assert not first.root.exists()
    assert len(tombstones) == 1
    assert second.root.exists()

    monkeypatch.undo()
    rebuilt = GitHubArtifactMaterializer(
        DownloadClient({"1": first_archive}), settings, cache_state
    ).materialize(first_resolved)
    assert rebuilt.root == first.root
    assert rebuilt.root.exists()

    materializer.prune(
        (
            first.receipt.artifact_id,
            second.receipt.artifact_id,
        )
    )
    assert not tuple(first.root.parent.glob(".pruning-*"))


def test_pruning_revalidates_placement_before_recursive_delete(tmp_path, monkeypatch):
    settings = replace(github_settings(), cache_retention_releases=1)
    first_resolved, first_archive = resolved_fixture(data=b"first", generation=1)
    second_resolved, second_archive = resolved_fixture(data=b"second", generation=2)
    materializer = GitHubArtifactMaterializer(
        DownloadClient({"1": first_archive, "2": second_archive}),
        settings,
        tmp_path / "state",
    )
    first = materializer.materialize(first_resolved)
    second = materializer.materialize(second_resolved)
    os.utime(first.root, (1, 1))
    os.utime(second.root, (2, 2))
    outside = tmp_path / "outside"
    outside.mkdir()
    moved_kind = outside / first.root.parent.name
    redirect = outside / "redirect"
    redirect.mkdir()
    original_validation = materializer._validate_open_cache_kind_directory
    validations = 0

    def relocate_after_validation(descriptor, kind_directory, artifact_kind):
        nonlocal validations
        original_validation(descriptor, kind_directory, artifact_kind)
        validations += 1
        if validations == 3:
            kind_directory.rename(moved_kind)
            kind_directory.symlink_to(redirect, target_is_directory=True)

    monkeypatch.setattr(
        materializer,
        "_validate_open_cache_kind_directory",
        relocate_after_validation,
    )

    with pytest.raises(CacheCorruptionError, match="kind directory"):
        materializer.prune(())

    assert tuple(moved_kind.glob(".pruning-*"))
    assert second.root.name in {path.name for path in moved_kind.iterdir()}
