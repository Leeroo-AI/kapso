from __future__ import annotations

import io
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ContractValidationError,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    MissingReferenceError,
    PublicationArtifactKind,
    SECURITY_DENYLIST_EVIDENCE_FILENAME,
    SECURITY_DENYLIST_POLICY_VERSION,
    SECURITY_DENYLIST_SCHEMA_VERSION,
    SecurityDenylistEvidence,
    SecurityDenylistEvidenceBundle,
    SecurityDenylistKind,
    SecurityDenylistRevocation,
    SecurityDenylistSnapshot,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    GitHubPublicationError,
    PublicationEnvelope,
    ReleaseAssetInput,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    CurrentPointerState,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
    release_attestation_reference,
    repository_for_artifact,
    security_denylist_tag,
    tag_prefix_for_artifact,
)
from kapso.cross_run.security_denylist import (
    AuthenticatedSecurityDenylistAuthority,
    AuthenticatedSecurityDenylistSnapshot,
    GitHubSecurityDenylistSnapshotProvider,
    SecurityDenylistCheckpointStore,
    SecurityDenylistError,
    SecurityDenylistPublicationGate,
    SecurityDenylistPublisher,
)
from kapso.cross_run.settings import CrossRunSettings
from tests.cross_run_github_fixtures import release_attestation
from tests.test_cross_run_github_publisher import (
    EXPECTED_PARENT,
    SOURCE_COMMIT,
    FakePublisherClient,
    FakeResolver,
    build_envelope as build_knowledge_envelope,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
SCOPE_ID = "ml_ai"
SCOPE_CONTRACT_ID = content_id("scope-contract", {"scope": SCOPE_ID})


class _SnapshotProvider:
    def __init__(
        self,
        current: AuthenticatedSecurityDenylistSnapshot,
        historical: tuple[AuthenticatedSecurityDenylistSnapshot, ...] = (),
    ) -> None:
        self.current = current
        self.historical = {item.snapshot.snapshot_id: item for item in historical}
        self.calls: list[tuple[str, str]] = []
        self.current_failure: Exception | None = None

    def resolve_current(self, scope_id):
        self.calls.append(("current", scope_id))
        if self.current_failure is not None:
            raise self.current_failure
        return self.current

    def resolve_exact(self, scope_id, snapshot_id):
        self.calls.append(("exact", snapshot_id))
        return self.historical[snapshot_id]


class _ResolvedArtifactProvider:
    def __init__(self, resolved, settings):
        self.resolved = resolved
        self.settings = settings
        self.calls = []

    def resolve_current(self, scope_id, artifact_kind):
        self.calls.append(("current", scope_id, artifact_kind))
        return self.resolved

    def resolve_artifact(self, scope_id, artifact_kind, artifact_id):
        self.calls.append(("exact", scope_id, artifact_kind, artifact_id))
        return self.resolved


class _MaterializedArtifactProvider:
    def __init__(self, content):
        self.content = content

    def materialize(self, resolved):
        return SimpleNamespace(content=self.content)


class _LiveCurrentGitHubProvider(GitHubSecurityDenylistSnapshotProvider):
    def __init__(self, resolver, materializer, current_snapshot, current_pointer):
        super().__init__(resolver, materializer)
        self.current_snapshot = current_snapshot
        self.current_pointer = current_pointer
        self.current_calls = 0

    def resolve_current(self, scope_id):
        self.current_calls += 1
        record = self.current_pointer.publication_record
        return AuthenticatedSecurityDenylistSnapshot.mint(
            snapshot=self.current_snapshot,
            publication_id=record.publication_id,
            repository_full_name=record.repository_full_name,
            repository_node_id=record.repository_node_id,
            authority_commit_sha=self.resolver.current_head,
            pointer_digest=tree_or_blob_digest(self.current_pointer.to_json_bytes()),
            release_attestation_ref=record.release_attestation_ref,
            validation_closure_ids=self.current_pointer.validation_closure_ids,
        )


class _NoOpPublicationGate:
    def validate_before_publication(self, **_arguments):
        return None

    def revalidate_before_activation(self, **_arguments):
        return None


def _settings():
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def _evidence(label: str) -> SecurityDenylistEvidence:
    return SecurityDenylistEvidence.mint(
        evidence_kind="authenticated_review",
        summary=f"Verified security evidence for {label}",
        source_ids=(content_id("security-source", {"label": label}),),
        recorded_at="2026-07-21T12:00:00Z",
    )


def _revocation(
    label: str,
    evidence: SecurityDenylistEvidence,
) -> SecurityDenylistRevocation:
    return SecurityDenylistRevocation.mint(
        subject_id=content_id("security-subject", {"label": label}),
        kind=SecurityDenylistKind.SECURITY,
        reason_code="verified_compromise",
        evidence_ids=(evidence.evidence_id,),
        recorded_at="2026-07-21T12:00:00Z",
    )


def _evidence_bundle(
    evidence: tuple[SecurityDenylistEvidence, ...],
) -> SecurityDenylistEvidenceBundle:
    return SecurityDenylistEvidenceBundle.mint(
        evidence=tuple(sorted(evidence, key=lambda item: item.evidence_id))
    )


def _snapshot(
    generation: int,
    predecessor: SecurityDenylistSnapshot | None,
    revocations: tuple[SecurityDenylistRevocation, ...],
    evidence: tuple[SecurityDenylistEvidence, ...] = (),
    *,
    binding_hash: str | None = None,
) -> SecurityDenylistSnapshot:
    repositories = _settings().scopes.resolve(SCOPE_ID)
    evidence_bundle = _evidence_bundle(evidence)
    exact_dependency_ids = {
        SCOPE_CONTRACT_ID,
        evidence_bundle.evidence_bundle_id,
        *evidence_bundle.source_ids,
        *(item.revocation_id for item in revocations),
        *(item.subject_id for item in revocations),
        *(evidence_id for item in revocations for evidence_id in item.evidence_ids),
    }
    if predecessor is not None:
        exact_dependency_ids.add(predecessor.snapshot_id)
    return SecurityDenylistSnapshot.mint(
        schema_version=SECURITY_DENYLIST_SCHEMA_VERSION,
        policy_version=SECURITY_DENYLIST_POLICY_VERSION,
        scope_id=SCOPE_ID,
        scope_contract_id=SCOPE_CONTRACT_ID,
        scope_repository_binding_hash=(
            binding_hash or repositories.binding_fingerprint
        ),
        generation=generation,
        predecessor_snapshot_id=(
            predecessor.snapshot_id if predecessor is not None else None
        ),
        evidence_bundle_id=evidence_bundle.evidence_bundle_id,
        evidence_source_ids=evidence_bundle.source_ids,
        revocations=tuple(sorted(revocations, key=lambda item: item.revocation_id)),
        exact_dependency_ids=tuple(sorted(exact_dependency_ids)),
        checksums={
            SECURITY_DENYLIST_EVIDENCE_FILENAME: tree_or_blob_digest(
                evidence_bundle.to_json_bytes()
            )
        },
    )


def _authenticated(
    snapshot: SecurityDenylistSnapshot,
    *,
    repository_full_name: str | None = None,
) -> AuthenticatedSecurityDenylistSnapshot:
    repositories = _settings().scopes.resolve(SCOPE_ID)
    return AuthenticatedSecurityDenylistSnapshot.mint(
        snapshot=snapshot,
        publication_id=content_id(
            "github-publication",
            {"snapshot_id": snapshot.snapshot_id},
        ),
        repository_full_name=(repository_full_name or repositories.security_repository),
        repository_node_id="security_repo_node",
        authority_commit_sha=(f"{snapshot.generation + 1:x}" * 40)[:40],
        pointer_digest=tree_or_blob_digest(snapshot.to_json_bytes()),
        release_attestation_ref=(
            f"attestations/security-denylist/{snapshot.generation}"
        ),
        validation_closure_ids=tuple(
            sorted({snapshot.snapshot_id, *snapshot.exact_dependency_ids})
        ),
    )


def _resolved(snapshot: SecurityDenylistSnapshot) -> ResolvedGitHubArtifact:
    settings = _settings()
    repositories = settings.scopes.resolve(SCOPE_ID)
    repository = repositories.security_repository
    commit_sha = "a" * 40
    tag = f"security-denylist/D{snapshot.generation:06d}"
    asset = GitHubReleaseAsset(
        asset_id="71",
        name="security-denylist.tar",
        media_type="application/x-tar",
        size=7,
        sha256=tree_or_blob_digest(b"archive"),
    )
    attestation = release_attestation(
        repository,
        tag,
        commit_sha,
        {asset.name: asset.sha256},
    )
    record = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        artifact_id=snapshot.snapshot_id,
        repository_node_id="security_repo_node",
        repository_full_name=repository,
        commit_sha=commit_sha,
        immutable_release_id="17",
        tag=tag,
        assets=(asset,),
        release_attestation_ref=release_attestation_reference(attestation),
        published_at="2026-07-21T12:00:00Z",
        publisher_identity="leeroo-coder",
    )
    closure = tuple(sorted({snapshot.snapshot_id, *snapshot.exact_dependency_ids}))
    pointer = CurrentArtifactPointer(
        scope_id=SCOPE_ID,
        publication_record=record,
        publication_intent_digest=tree_or_blob_digest(b"publication intent"),
        source_tree_digest=tree_or_blob_digest(b"source tree"),
        source_git_tree_sha="b" * 40,
        materialized_tree_digest=tree_or_blob_digest(b"materialized tree"),
        manifest_relative_path="security-denylist.json",
        manifest_digest=tree_or_blob_digest(snapshot.to_json_bytes()),
        validation_closure_ids=closure,
    )
    policy = RepositoryPolicyReport(
        repository_full_name=repository,
        repository_node_id="security_repo_node",
        private=True,
        default_branch="main",
        authenticated_actor="leeroo-coder",
        write_access=True,
        immutable_releases=True,
    )
    return ResolvedGitHubArtifact(
        repositories=repositories,
        pointer=pointer,
        policy=policy,
        pointer_commit_sha="c" * 40,
    )


def _publication_envelope(
    tmp_path: Path,
    snapshot: SecurityDenylistSnapshot,
    evidence: tuple[SecurityDenylistEvidence, ...] = (),
) -> PublicationEnvelope:
    settings = _settings()
    source = tmp_path / "security-source"
    source.mkdir()
    manifest_path = source / "security-denylist.json"
    evidence_path = source / SECURITY_DENYLIST_EVIDENCE_FILENAME
    manifest_path.write_bytes(snapshot.to_json_bytes())
    evidence_path.write_bytes(_evidence_bundle(evidence).to_json_bytes())
    archive_path = tmp_path / "security-denylist.tar"
    with tarfile.open(archive_path, "w") as package:
        for path in (manifest_path, evidence_path):
            payload = path.read_bytes()
            member = tarfile.TarInfo(path.name)
            member.size = len(payload)
            member.mtime = 0
            package.addfile(member, io.BytesIO(payload))
    archive_payload = archive_path.read_bytes()
    return PublicationEnvelope(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        artifact_id=snapshot.snapshot_id,
        scope_id=SCOPE_ID,
        expected_parent_sha=EXPECTED_PARENT,
        source_tree=source,
        manifest_relative_path=manifest_path.name,
        assets=(
            ReleaseAssetInput(
                path=archive_path,
                name=archive_path.name,
                media_type="application/x-tar",
                size=len(archive_payload),
                sha256=tree_or_blob_digest(archive_payload),
            ),
        ),
        tag=security_denylist_tag(settings.github, snapshot.generation),
        committed_at="2026-07-21T12:00:00Z",
        validation_closure_ids=tuple(
            sorted({snapshot.snapshot_id, *snapshot.exact_dependency_ids})
        ),
    )


def _store(tmp_path: Path) -> SecurityDenylistCheckpointStore:
    settings = _settings()
    return SecurityDenylistCheckpointStore(
        (tmp_path / "security-state").resolve(),
        tmp_path.resolve(),
        settings.launch.security_denylist_checkpoint_size_bytes,
    )


def _authority(
    provider: _SnapshotProvider,
    store: SecurityDenylistCheckpointStore,
) -> AuthenticatedSecurityDenylistAuthority:
    settings = _settings()
    return AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        settings.launch,
        provider,
        store,
    )


def _observe(
    authority: AuthenticatedSecurityDenylistAuthority,
    subjects: tuple[str, ...],
):
    return authority.observe_exact(
        scope_id=SCOPE_ID,
        scope_contract_id=SCOPE_CONTRACT_ID,
        checked_subject_ids=tuple(sorted(subjects)),
    )


def test_generation_zero_establishes_a_durable_floor_and_live_resolves_again(
    tmp_path,
):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    store = _store(tmp_path)
    authority = _authority(provider, store)
    subject_id = content_id("security-subject", {"label": "safe"})

    first = _observe(authority, (subject_id,))
    restarted = _authority(provider, _store(tmp_path))
    second = _observe(restarted, (subject_id,))

    assert first.denied_subject_ids == ()
    assert second.snapshot_id == generation_zero.snapshot_id
    assert provider.calls == [
        ("current", SCOPE_ID),
        ("current", SCOPE_ID),
    ]
    assert store.checkpoint(SCOPE_ID).snapshot_id == generation_zero.snapshot_id


def test_multi_generation_lineage_is_authenticated_and_denials_intersect_exactly(
    tmp_path,
):
    first_evidence = _evidence("first")
    second_evidence = _evidence("second")
    first_revocation = _revocation("first", first_evidence)
    second_revocation = _revocation("second", second_evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (first_revocation,),
        (first_evidence,),
    )
    generation_two = _snapshot(
        2,
        generation_one,
        (first_revocation, second_revocation),
        (first_evidence, second_evidence),
    )
    provider = _SnapshotProvider(
        _authenticated(generation_two),
        (_authenticated(generation_zero), _authenticated(generation_one)),
    )
    authority = _authority(provider, _store(tmp_path))
    safe_subject = content_id("security-subject", {"label": "safe"})

    observation = _observe(
        authority,
        (safe_subject, first_revocation.subject_id),
    )

    assert observation.denied_subject_ids == (first_revocation.subject_id,)
    assert provider.calls == [
        ("current", SCOPE_ID),
        ("exact", generation_one.snapshot_id),
        ("exact", generation_zero.snapshot_id),
    ]


def test_local_floor_rejects_rollback_and_equal_generation_fork(tmp_path):
    first_evidence = _evidence("first")
    fork_evidence = _evidence("fork")
    first_revocation = _revocation("first", first_evidence)
    fork_revocation = _revocation("fork", fork_evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (first_revocation,),
        (first_evidence,),
    )
    provider = _SnapshotProvider(
        _authenticated(generation_one),
        (_authenticated(generation_zero),),
    )
    authority = _authority(provider, _store(tmp_path))
    _observe(authority, (first_revocation.subject_id,))

    provider.current = _authenticated(generation_zero)
    with pytest.raises(SecurityDenylistError, match="below the local floor"):
        _observe(authority, (first_revocation.subject_id,))

    provider.current = _authenticated(
        _snapshot(
            1,
            generation_zero,
            (fork_revocation,),
            (fork_evidence,),
        )
    )
    with pytest.raises(SecurityDenylistError, match="equal-generation fork"):
        _observe(authority, (first_revocation.subject_id,))


def test_lineage_rejects_missing_history_and_removed_revocations(tmp_path):
    evidence = _evidence("revoked")
    revoked = _revocation("revoked", evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (revoked,),
        (evidence,),
    )
    generation_two = _snapshot(2, generation_one, ())
    provider = _SnapshotProvider(_authenticated(generation_two))
    authority = _authority(provider, _store(tmp_path))

    with pytest.raises(KeyError):
        _observe(authority, (revoked.subject_id,))

    provider.historical = {
        generation_one.snapshot_id: _authenticated(generation_one),
        generation_zero.snapshot_id: _authenticated(generation_zero),
    }
    with pytest.raises(SecurityDenylistError, match="removes a revocation"):
        _observe(authority, (revoked.subject_id,))


def test_scope_binding_and_repository_substitution_fail_closed(tmp_path):
    wrong_binding = tree_or_blob_digest(b"wrong scope repository binding")
    substituted = _snapshot(0, None, (), binding_hash=wrong_binding)
    provider = _SnapshotProvider(_authenticated(substituted))
    authority = _authority(provider, _store(tmp_path))
    subject_id = content_id("security-subject", {"label": "safe"})

    with pytest.raises(SecurityDenylistError, match="another scope authority"):
        _observe(authority, (subject_id,))

    valid = _snapshot(0, None, ())
    provider.current = _authenticated(
        valid,
        repository_full_name="Leeroo-AI/substituted-security",
    )
    with pytest.raises(SecurityDenylistError, match="another scope authority"):
        _observe(authority, (subject_id,))


def test_provider_failure_never_falls_back_to_the_checkpoint(tmp_path):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    authority = _authority(provider, _store(tmp_path))
    subject_id = content_id("security-subject", {"label": "safe"})
    _observe(authority, (subject_id,))
    provider.current_failure = RuntimeError("GitHub unavailable")

    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        _observe(authority, (subject_id,))

    assert [call[0] for call in provider.calls] == ["current", "current"]


@pytest.mark.parametrize("bound", ("count", "bytes"))
def test_checked_subject_bounds_fail_before_github_work(tmp_path, bound):
    settings = _settings()
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    launch_settings = replace(
        settings.launch,
        security_denylist_checked_subject_limit=(1 if bound == "count" else 100),
        security_denylist_checked_subject_size_bytes=(1 if bound == "bytes" else 10000),
    )
    authority = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        launch_settings,
        provider,
        _store(tmp_path),
    )
    subjects = tuple(
        sorted(
            (
                content_id("security-subject", {"position": 1}),
                content_id("security-subject", {"position": 2}),
            )
        )
    )

    with pytest.raises(SecurityDenylistError, match=f"{bound}.*bound"):
        _observe(authority, subjects)

    assert provider.calls == []


@pytest.mark.parametrize(
    ("checked_subjects", "expected_error"),
    (
        ((object(), object()), "count"),
        (("x" * 1000,), "bytes"),
    ),
)
def test_checked_subject_limits_precede_element_validation_and_ordering(
    tmp_path,
    checked_subjects,
    expected_error,
):
    settings = _settings()
    provider = _SnapshotProvider(_authenticated(_snapshot(0, None, ())))
    authority = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        replace(
            settings.launch,
            security_denylist_checked_subject_limit=1,
            security_denylist_checked_subject_size_bytes=16,
        ),
        provider,
        _store(tmp_path),
    )

    with pytest.raises(SecurityDenylistError, match=expected_error):
        authority.observe_exact(
            scope_id=SCOPE_ID,
            scope_contract_id=SCOPE_CONTRACT_ID,
            checked_subject_ids=checked_subjects,
        )

    assert provider.calls == []


def test_checked_subject_ordering_runs_only_after_bounded_content_validation(tmp_path):
    settings = _settings()
    provider = _SnapshotProvider(_authenticated(_snapshot(0, None, ())))
    authority = _authority(provider, _store(tmp_path))
    subjects = tuple(
        reversed(
            sorted(
                (
                    content_id("security-subject", {"position": 1}),
                    content_id("security-subject", {"position": 2}),
                )
            )
        )
    )

    with pytest.raises(SecurityDenylistError, match="sorted and unique"):
        authority.observe_exact(
            scope_id=SCOPE_ID,
            scope_contract_id=SCOPE_CONTRACT_ID,
            checked_subject_ids=subjects,
        )

    assert provider.calls == []


def test_oversized_checkpoint_is_rejected_before_publication(tmp_path):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    store = SecurityDenylistCheckpointStore(
        (tmp_path / "bounded-security-state").resolve(),
        tmp_path.resolve(),
        1,
    )
    authority = _authority(provider, store)
    subject_id = content_id("security-subject", {"label": "safe"})

    with pytest.raises(SecurityDenylistError, match="configured bound"):
        _observe(authority, (subject_id,))

    assert store.checkpoint(SCOPE_ID) is None


def test_checkpoint_size_is_independent_of_cumulative_revocation_content(tmp_path):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    store = _store(tmp_path)
    authority = _authority(provider, store)
    subject_id = content_id("security-subject", {"label": "safe"})

    _observe(authority, (subject_id,))
    checkpoint = store.checkpoint(SCOPE_ID)

    assert checkpoint is not None
    assert "revocation_ids" not in checkpoint.to_dict()
    assert "denied_subject_ids" not in checkpoint.to_dict()


def test_checkpoint_store_rejects_a_world_writable_trusted_root(tmp_path):
    os.chmod(tmp_path, 0o777)

    with pytest.raises(SecurityDenylistError, match="owner-private"):
        SecurityDenylistCheckpointStore(
            (tmp_path / "security-state").resolve(),
            tmp_path.resolve(),
            _settings().launch.security_denylist_checkpoint_size_bytes,
        )


def test_concurrent_first_observers_converge_on_one_checkpoint(tmp_path):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    store = _store(tmp_path)
    authority = _authority(provider, store)
    subject_id = content_id("security-subject", {"label": "safe"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        observations = tuple(
            executor.map(
                lambda _position: _observe(authority, (subject_id,)),
                range(2),
            )
        )

    assert {item.snapshot_id for item in observations} == {generation_zero.snapshot_id}
    assert store.checkpoint(SCOPE_ID).snapshot_id == generation_zero.snapshot_id


@pytest.mark.parametrize("corruption", ("mode", "hardlink", "oversized"))
def test_checkpoint_filesystem_corruption_fails_loud(tmp_path, corruption):
    generation_zero = _snapshot(0, None, ())
    provider = _SnapshotProvider(_authenticated(generation_zero))
    store = _store(tmp_path)
    authority = _authority(provider, store)
    subject_id = content_id("security-subject", {"label": "safe"})
    _observe(authority, (subject_id,))
    checkpoint_path = store._checkpoint_path(SCOPE_ID)

    if corruption == "mode":
        os.chmod(checkpoint_path, 0o600)
    elif corruption == "hardlink":
        source = tmp_path / "checkpoint-hardlink-source"
        checkpoint_path.rename(source)
        os.link(source, checkpoint_path)
    else:
        os.chmod(checkpoint_path, 0o600)
        checkpoint_path.write_bytes(b"x" * (store.maximum_checkpoint_size_bytes + 1))
        os.chmod(checkpoint_path, 0o400)

    with pytest.raises(SecurityDenylistError):
        store.checkpoint(SCOPE_ID)


def test_security_snapshot_generation_zero_cannot_start_with_revocations():
    evidence = _evidence("premature")
    revoked = _revocation("premature", evidence)

    with pytest.raises(ContractValidationError, match="generation zero"):
        _snapshot(0, None, (revoked,), (evidence,))


def test_security_snapshot_rejects_substituted_or_extra_evidence():
    evidence = _evidence("expected")
    revocation = _revocation("expected", evidence)
    generation_zero = _snapshot(0, None, ())
    snapshot = _snapshot(
        1,
        generation_zero,
        (revocation,),
        (evidence,),
    )

    with pytest.raises(ContractValidationError, match="evidence bundle"):
        snapshot.validate_evidence_bundle(_evidence_bundle((_evidence("other"),)))

    with pytest.raises(MissingReferenceError, match="not exact"):
        replace(
            snapshot,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        *snapshot.exact_dependency_ids,
                        content_id("security-extra", {"unexpected": True}),
                    }
                )
            ),
        )


def test_security_artifact_routes_to_its_dedicated_repository_and_tag_prefix():
    settings = _settings()
    repositories = settings.scopes.resolve(SCOPE_ID)

    assert (
        repository_for_artifact(
            repositories,
            PublicationArtifactKind.SECURITY_DENYLIST,
        )
        == "Leeroo-AI/kapso-security"
    )
    assert (
        tag_prefix_for_artifact(
            settings.github,
            PublicationArtifactKind.SECURITY_DENYLIST,
        )
        == "security-denylist/"
    )


def test_security_release_package_recreates_the_exact_source_closure(tmp_path):
    settings = _settings()
    snapshot = _snapshot(0, None, ())
    manifest_name = "security-denylist.json"
    closure_name = SECURITY_DENYLIST_EVIDENCE_FILENAME
    closure_payload = _evidence_bundle(()).to_json_bytes()
    manifest_payload = snapshot.to_json_bytes()
    archive_path = tmp_path / "security-denylist.tar"
    with tarfile.open(archive_path, "w") as package:
        for name, payload in (
            (manifest_name, manifest_payload),
            (closure_name, closure_payload),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mtime = 0
            package.addfile(member, io.BytesIO(payload))
    archive_payload = archive_path.read_bytes()
    asset = ReleaseAssetInput(
        path=archive_path,
        name=archive_path.name,
        media_type="application/x-tar",
        size=len(archive_payload),
        sha256=tree_or_blob_digest(archive_payload),
    )
    materializer = GitHubArtifactMaterializer(
        object(),
        settings.github,
        tmp_path / "state",
    )

    materialized_digest = materializer.validate_local_package(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        artifact_id=snapshot.snapshot_id,
        manifest_relative_path=manifest_name,
        manifest_digest=tree_or_blob_digest(manifest_payload),
        assets=(asset,),
        source_files={
            manifest_name: (tree_or_blob_digest(manifest_payload), "100644"),
            closure_name: (tree_or_blob_digest(closure_payload), "100644"),
        },
    )

    assert materialized_digest.startswith("sha256:")


def test_github_provider_resolves_current_and_exact_authenticated_manifests(tmp_path):
    settings = _settings()
    snapshot = _snapshot(0, None, ())
    resolved = _resolved(snapshot)
    content = tmp_path / "materialized"
    content.mkdir()
    (content / resolved.pointer.manifest_relative_path).write_bytes(
        snapshot.to_json_bytes()
    )
    (content / SECURITY_DENYLIST_EVIDENCE_FILENAME).write_bytes(
        _evidence_bundle(()).to_json_bytes()
    )
    resolver = _ResolvedArtifactProvider(resolved, settings.github)
    provider = GitHubSecurityDenylistSnapshotProvider(
        resolver,
        _MaterializedArtifactProvider(content),
    )

    current = provider.resolve_current(SCOPE_ID)
    exact = provider.resolve_exact(SCOPE_ID, snapshot.snapshot_id)

    assert current == exact
    assert current.snapshot == snapshot
    assert current.repository_full_name == "Leeroo-AI/kapso-security"
    assert resolver.calls == [
        ("current", SCOPE_ID, PublicationArtifactKind.SECURITY_DENYLIST),
        (
            "exact",
            SCOPE_ID,
            PublicationArtifactKind.SECURITY_DENYLIST,
            snapshot.snapshot_id,
        ),
    ]


def test_github_provider_rejects_noncanonical_manifest_bytes(tmp_path):
    settings = _settings()
    snapshot = _snapshot(0, None, ())
    resolved = _resolved(snapshot)
    content = tmp_path / "materialized"
    content.mkdir()
    (content / resolved.pointer.manifest_relative_path).write_bytes(
        b" " + snapshot.to_json_bytes()
    )
    (content / SECURITY_DENYLIST_EVIDENCE_FILENAME).write_bytes(
        _evidence_bundle(()).to_json_bytes()
    )
    provider = GitHubSecurityDenylistSnapshotProvider(
        _ResolvedArtifactProvider(resolved, settings.github),
        _MaterializedArtifactProvider(content),
    )

    with pytest.raises(SecurityDenylistError, match="not canonical"):
        provider.resolve_current(SCOPE_ID)


def test_security_publication_runs_the_full_immutable_transaction(tmp_path):
    settings = _settings()
    snapshot = _snapshot(0, None, ())
    envelope = _publication_envelope(tmp_path, snapshot)
    repository = settings.scopes.resolve(SCOPE_ID).security_repository
    resolver = FakeResolver(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        repository=repository,
        repository_node_id="security_repo_node",
    )
    client = FakePublisherClient(
        envelope.assets[0],
        repository=repository,
        repository_node_id="security_repo_node",
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        tag=envelope.tag,
        head_observer=lambda commit_sha: setattr(
            resolver,
            "current_head",
            commit_sha,
        ),
    )
    resolver.current_state_observer = lambda: client.events.append("current_gate")
    materializer = GitHubArtifactMaterializer(
        client,
        settings.github,
        tmp_path / "state",
    )
    generic_publisher = AutonomousGitHubPublisher(
        client,
        resolver,
        materializer,
        settings.github,
    )
    provider = GitHubSecurityDenylistSnapshotProvider(resolver, materializer)
    publisher = SecurityDenylistPublisher(
        generic_publisher,
        resolver,
        provider,
        settings.launch,
    )

    telemetry = publisher.publish(envelope)

    assert telemetry.publication_record.repository_full_name == repository
    assert telemetry.publication_record.artifact_id == snapshot.snapshot_id
    assert telemetry.source_commit_sha == SOURCE_COMMIT
    assert client.events[-3:] == ["pointer_commit", "current_gate", "pointer_ref"]


def test_security_successor_transaction_reauthenticates_the_live_predecessor(
    tmp_path,
):
    settings = _settings()
    evidence = _evidence("successor")
    revocation = _revocation("successor", evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (revocation,),
        (evidence,),
    )
    current_pointer = _resolved(generation_zero).pointer
    envelope = _publication_envelope(tmp_path, generation_one, (evidence,))
    repository = settings.scopes.resolve(SCOPE_ID).security_repository
    resolver = FakeResolver(
        existing=current_pointer,
        current_head=EXPECTED_PARENT,
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        repository=repository,
        repository_node_id="security_repo_node",
    )
    client = FakePublisherClient(
        envelope.assets[0],
        repository=repository,
        repository_node_id="security_repo_node",
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        tag=envelope.tag,
        head_observer=lambda commit_sha: setattr(
            resolver,
            "current_head",
            commit_sha,
        ),
    )
    materializer = GitHubArtifactMaterializer(
        client,
        settings.github,
        tmp_path / "state",
    )
    generic_publisher = AutonomousGitHubPublisher(
        client,
        resolver,
        materializer,
        settings.github,
    )
    provider = _LiveCurrentGitHubProvider(
        resolver,
        materializer,
        generation_zero,
        current_pointer,
    )
    publisher = SecurityDenylistPublisher(
        generic_publisher,
        resolver,
        provider,
        settings.launch,
    )

    telemetry = publisher.publish(envelope)

    assert telemetry.publication_record.artifact_id == generation_one.snapshot_id
    assert provider.current_calls == 2
    assert client.events[-1] == "pointer_ref"


def test_generic_publisher_cannot_activate_security_without_the_lineage_gate(
    tmp_path,
):
    settings = _settings()
    snapshot = _snapshot(0, None, ())
    envelope = _publication_envelope(tmp_path, snapshot)
    resolver = FakeResolver(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        repository=settings.scopes.resolve(SCOPE_ID).security_repository,
        repository_node_id="security_repo_node",
    )
    publisher = AutonomousGitHubPublisher(
        object(),
        resolver,
        object(),
        settings.github,
    )

    with pytest.raises(GitHubPublicationError, match="sealed authorization"):
        publisher.publish(envelope)

    with pytest.raises(GitHubPublicationError, match="sealed authorization"):
        publisher.publish(
            envelope,
            activation_authorization=_NoOpPublicationGate(),
        )

    publisher._bind_activation_verifier(
        PublicationArtifactKind.SECURITY_DENYLIST,
        SecurityDenylistPublicationGate,
    )
    with pytest.raises(GitHubPublicationError, match="registered concrete verifier"):
        publisher._authorize_publication(
            envelope,
            _NoOpPublicationGate(),
        )

    other_publisher = AutonomousGitHubPublisher(
        object(),
        resolver,
        object(),
        settings.github,
    )
    authorization = publisher._authorize_publication(
        envelope,
        SecurityDenylistPublicationGate(
            resolver,
            object(),
            settings.launch,
        ),
    )
    with pytest.raises(GitHubPublicationError, match="another artifact kind"):
        authorization.verifier_for(
            publisher,
            replace(
                envelope,
                artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            ),
        )
    with pytest.raises(GitHubPublicationError, match="another artifact"):
        authorization.verifier_for(
            publisher,
            replace(
                envelope,
                artifact_id=content_id(
                    "security-denylist-snapshot",
                    {"different": "artifact"},
                ),
            ),
        )

    knowledge_envelope, _manifest = build_knowledge_envelope(tmp_path)
    with pytest.raises(GitHubPublicationError, match="cannot guard"):
        publisher.publish(
            knowledge_envelope,
            activation_authorization=authorization,
        )

    with pytest.raises(GitHubPublicationError, match="another publisher"):
        other_publisher.publish(
            envelope,
            activation_authorization=authorization,
        )


@pytest.mark.parametrize("failure", ("skipped_generation", "removed_revocation"))
def test_publication_gate_rejects_nonadjacent_or_nonmonotonic_successors(
    tmp_path,
    failure,
):
    evidence = _evidence("revoked")
    revocation = _revocation("revoked", evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (revocation,),
        (evidence,),
    )
    candidate = (
        _snapshot(3, generation_one, (revocation,), (evidence,))
        if failure == "skipped_generation"
        else _snapshot(2, generation_one, ())
    )
    resolved = _resolved(generation_one)
    content = tmp_path / "current-security"
    content.mkdir()
    (content / resolved.pointer.manifest_relative_path).write_bytes(
        generation_one.to_json_bytes()
    )
    (content / SECURITY_DENYLIST_EVIDENCE_FILENAME).write_bytes(
        _evidence_bundle((evidence,)).to_json_bytes()
    )
    resolver = _ResolvedArtifactProvider(resolved, _settings().github)
    provider = GitHubSecurityDenylistSnapshotProvider(
        resolver,
        _MaterializedArtifactProvider(content),
    )
    gate = SecurityDenylistPublicationGate(
        resolver,
        provider,
        _settings().launch,
    )

    with pytest.raises(
        SecurityDenylistError,
        match=(
            "discontinuous"
            if failure == "skipped_generation"
            else "removes a revocation"
        ),
    ):
        gate.validate_before_publication(
            envelope=_publication_envelope(tmp_path, candidate),
            repositories=_settings().scopes.resolve(SCOPE_ID),
            current_state=CurrentPointerState(
                pointer=resolved.pointer,
                head_commit_sha=resolved.pointer_commit_sha,
            ),
            manifest=candidate,
        )


def test_finite_lineage_horizon_rejects_publication_and_fresh_read(tmp_path):
    evidence = _evidence("horizon")
    revocation = _revocation("horizon", evidence)
    generation_zero = _snapshot(0, None, ())
    generation_one = _snapshot(
        1,
        generation_zero,
        (revocation,),
        (evidence,),
    )
    settings = _settings()
    bounded_launch = replace(
        settings.launch,
        security_denylist_lineage_limit=1,
    )
    resolved = _resolved(generation_zero)
    content = tmp_path / "bounded-current-security"
    content.mkdir()
    (content / resolved.pointer.manifest_relative_path).write_bytes(
        generation_zero.to_json_bytes()
    )
    (content / SECURITY_DENYLIST_EVIDENCE_FILENAME).write_bytes(
        _evidence_bundle(()).to_json_bytes()
    )
    resolver = _ResolvedArtifactProvider(resolved, settings.github)
    provider = GitHubSecurityDenylistSnapshotProvider(
        resolver,
        _MaterializedArtifactProvider(content),
    )
    gate = SecurityDenylistPublicationGate(
        resolver,
        provider,
        bounded_launch,
    )

    with pytest.raises(SecurityDenylistError, match="finite lineage horizon"):
        gate.validate_before_publication(
            envelope=_publication_envelope(tmp_path, generation_one, (evidence,)),
            repositories=settings.scopes.resolve(SCOPE_ID),
            current_state=CurrentPointerState(
                pointer=resolved.pointer,
                head_commit_sha=resolved.pointer_commit_sha,
            ),
            manifest=generation_one,
        )

    live_provider = _SnapshotProvider(
        _authenticated(generation_one),
        (_authenticated(generation_zero),),
    )
    authority = AuthenticatedSecurityDenylistAuthority(
        settings.scopes,
        bounded_launch,
        live_provider,
        _store(tmp_path),
    )
    with pytest.raises(SecurityDenylistError, match="finite lineage horizon"):
        _observe(authority, (revocation.subject_id,))


@pytest.mark.parametrize("failure", ("binding", "canonical", "tag"))
def test_invalid_security_publication_fails_before_remote_mutation(tmp_path, failure):
    settings = _settings()
    snapshot = _snapshot(
        0,
        None,
        (),
        binding_hash=(
            tree_or_blob_digest(b"wrong binding") if failure == "binding" else None
        ),
    )
    envelope = _publication_envelope(tmp_path, snapshot)
    if failure == "canonical":
        manifest_path = envelope.source_tree / envelope.manifest_relative_path
        manifest_path.write_bytes(b" " + manifest_path.read_bytes())
    elif failure == "tag":
        envelope = replace(envelope, tag="security-denylist/not-a-generation")
    repository = settings.scopes.resolve(SCOPE_ID).security_repository
    resolver = FakeResolver(
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        repository=repository,
        repository_node_id="security_repo_node",
    )
    client = FakePublisherClient(
        envelope.assets[0],
        repository=repository,
        repository_node_id="security_repo_node",
        artifact_kind=PublicationArtifactKind.SECURITY_DENYLIST,
        tag=envelope.tag,
    )
    publisher = AutonomousGitHubPublisher(
        client,
        resolver,
        object(),
        settings.github,
    )

    with pytest.raises((GitHubPublicationError, ContractValidationError)):
        publisher.publish(envelope)

    assert client.events == []
