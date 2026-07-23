"""Crash-safe direct publication to immutable GitHub releases."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass
from io import DEFAULT_BUFFER_SIZE
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    require_content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    SECURITY_DENYLIST_EVIDENCE_FILENAME,
    ScopeRepositorySettings,
    SecurityDenylistEvidenceBundle,
    SecurityDenylistSnapshot,
)
from kapso.cross_run.github.command import (
    GitHubCommandClient,
    GitHubCompareAndSwapError,
    validate_release_attestation,
)
from kapso.cross_run.git_refs import (
    git_object_sha,
    git_tree_shas,
    require_git_ref_name,
)
from kapso.cross_run.github.resolver import (
    ARTIFACT_POINTER_FILENAME,
    ARTIFACT_PUBLICATION_INTENT_FILENAME,
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactResolver,
    GitHubArtifactActivationWitness,
    PublicationAssetIntent,
    PublicationSourceFile,
    RepositoryPolicyReport,
    artifact_activation_ref,
    artifact_activation_preparation_ref,
    artifact_identity_ref,
    artifact_publication_intent_ref,
    release_attestation_reference,
    repository_for_artifact,
    security_denylist_tag,
    tag_prefix_for_artifact,
)
from kapso.cross_run.settings import GitHubSettings

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ASSET_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+$"
)


class GitHubPublicationError(RuntimeError):
    """A publication input or GitHub response violates the release protocol."""


@dataclass(frozen=True)
class ReleaseAssetInput:
    path: Path
    name: str
    media_type: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PublicationEnvelope:
    artifact_kind: PublicationArtifactKind
    artifact_id: str
    scope_id: str
    expected_parent_sha: str
    source_tree: Path
    manifest_relative_path: str
    assets: tuple[ReleaseAssetInput, ...]
    tag: str
    committed_at: str
    validation_closure_ids: tuple[str, ...]


@dataclass(frozen=True)
class PublicationTelemetry:
    publication_record: GitHubPublicationRecord
    expected_parent_sha: str
    source_commit_sha: str
    pointer_commit_sha: str | None
    source_tree_digest: str
    validation_closure_ids: tuple[str, ...]
    idempotent_replay: bool


@dataclass(frozen=True)
class _SourceFile:
    relative_path: str
    content: bytes
    mode: str
    digest: str


class PublicationPackageValidator(Protocol):
    def validate_local_package(
        self,
        *,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
        manifest_relative_path: str,
        manifest_digest: str,
        assets: tuple[ReleaseAssetInput, ...],
        source_files: Mapping[str, tuple[str, str]],
    ) -> str:
        """Verify release assets before any remote publication write."""


_GuardedPublicationManifest = ExpertBaseReleaseManifest | SecurityDenylistSnapshot


class _PublicationActivationVerifier(Protocol):
    """Domain authorization that must remain true through pointer activation."""

    def validate_before_publication(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        current_state: CurrentPointerState,
        manifest: _GuardedPublicationManifest,
        source_tree_digest: str,
        manifest_digest: str,
    ) -> None: ...

    def revalidate_before_activation(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        pointer: CurrentArtifactPointer,
        manifest: _GuardedPublicationManifest,
    ) -> None: ...


_PUBLICATION_AUTHORIZATION_SEAL = object()
_ACTIVATION_VERIFIER_AUTHORITIES = {
    PublicationArtifactKind.EXPERT_BASE_RELEASE: (
        "kapso.cross_run.expert.publisher",
        "ExpertReleasePublicationGate",
    ),
    PublicationArtifactKind.SECURITY_DENYLIST: (
        "kapso.cross_run.security_denylist",
        "SecurityDenylistPublicationGate",
    ),
}


class _ArtifactPublicationAuthorization:
    """Owner- and artifact-bound capability created by a domain authority."""

    __slots__ = (
        "_artifact_id",
        "_artifact_kind",
        "_owner",
        "_scope_id",
        "_verifier",
    )

    def __init__(
        self,
        seal: object,
        owner: object,
        envelope: PublicationEnvelope,
        verifier: _PublicationActivationVerifier,
    ) -> None:
        if seal is not _PUBLICATION_AUTHORIZATION_SEAL:
            raise GitHubPublicationError("publication authorization is not sealed")
        self._owner = owner
        self._artifact_kind = envelope.artifact_kind
        self._artifact_id = envelope.artifact_id
        self._scope_id = envelope.scope_id
        self._verifier = verifier

    def verifier_for(
        self,
        owner: object,
        envelope: PublicationEnvelope,
    ) -> _PublicationActivationVerifier:
        if self._owner is not owner:
            raise GitHubPublicationError(
                "publication authorization belongs to another publisher"
            )
        if self._artifact_kind is not envelope.artifact_kind:
            raise GitHubPublicationError(
                "publication authorization belongs to another artifact kind"
            )
        if (
            self._artifact_id != envelope.artifact_id
            or self._scope_id != envelope.scope_id
        ):
            raise GitHubPublicationError(
                "publication authorization belongs to another artifact"
            )
        return self._verifier


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GitHubPublicationError(f"{name} must be an object")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise GitHubPublicationError(f"{name} must be 40 lowercase hex")
    return value


class AutonomousGitHubPublisher:
    """Publish a validated tree with expected-parent and immutable-release gates."""

    def __init__(
        self,
        client: GitHubCommandClient,
        resolver: GitHubArtifactResolver,
        package_validator: PublicationPackageValidator,
        settings: GitHubSettings,
    ) -> None:
        self.client = client
        self.resolver = resolver
        self.package_validator = package_validator
        self.settings = settings
        self._activation_verifier_types: dict[PublicationArtifactKind, type[object]] = (
            {}
        )

    def publish(
        self,
        envelope: PublicationEnvelope,
        *,
        activation_authorization: _ArtifactPublicationAuthorization | None = None,
    ) -> PublicationTelemetry:
        repositories = self.resolver.repositories_for_scope(envelope.scope_id)
        (
            source_files,
            source_tree_digest,
            manifest_digest,
            manifest,
        ) = self._validate_envelope(envelope, repositories)
        requires_activation_authorization = (
            envelope.artifact_kind in _ACTIVATION_VERIFIER_AUTHORITIES
        )
        if requires_activation_authorization:
            if type(activation_authorization) is not _ArtifactPublicationAuthorization:
                raise GitHubPublicationError(
                    f"{envelope.artifact_kind.value} publication requires "
                    "sealed authorization"
                )
            activation_gate = activation_authorization.verifier_for(
                self,
                envelope,
            )
        elif activation_authorization is not None:
            raise GitHubPublicationError(
                "publication authorization cannot guard this artifact kind"
            )
        else:
            activation_gate = None
        materialized_tree_digest = self.package_validator.validate_local_package(
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=manifest_digest,
            assets=envelope.assets,
            source_files={
                source.relative_path: (source.digest, source.mode)
                for source in source_files
            },
        )
        repository = repository_for_artifact(repositories, envelope.artifact_kind)
        policy = self.resolver.diagnose_repository(
            envelope.scope_id, envelope.artifact_kind
        )
        publication_intent = self.resolver.read_artifact_intent(
            envelope.scope_id,
            envelope.artifact_kind,
            envelope.artifact_id,
        )
        published_identity = self.resolver.read_artifact_pointer(
            envelope.scope_id,
            envelope.artifact_kind,
            envelope.artifact_id,
        )
        current_state = self.resolver.read_current_pointer_state(
            envelope.scope_id,
            envelope.artifact_kind,
            allow_missing=True,
        )
        existing = current_state.pointer
        observed_head = current_state.head_commit_sha
        preserved_current = (
            publication_intent.preserved_current
            if publication_intent is not None
            else self._preserved_current(existing)
        )
        self._validate_publication_closure_bounds(
            source_files,
            envelope.assets,
            preserved_current,
        )
        if published_identity is not None:
            if publication_intent is None or not publication_intent.binds(
                published_identity
            ):
                raise GitHubPublicationError(
                    "artifact identity has no matching pre-release publication intent"
                )
            self._validate_publication_intent(
                envelope,
                policy,
                source_files,
                source_tree_digest,
                materialized_tree_digest,
                manifest_digest,
                publication_intent,
            )
            if (
                existing is not None
                and existing.publication_record.artifact_id == envelope.artifact_id
                and existing != published_identity
            ):
                raise GitHubPublicationError(
                    "active artifact conflicts with its write-once publication identity"
                )
            self._validate_idempotent_replay(
                envelope,
                source_tree_digest,
                materialized_tree_digest,
                manifest_digest,
                published_identity,
            )
            self.resolver.verify_pointer(
                envelope.scope_id,
                envelope.artifact_kind,
                policy,
                published_identity,
                publication_intent,
            )
            activation_preparation = (
                self.resolver.resolve_artifact_activation_preparation(
                    envelope.scope_id,
                    envelope.artifact_kind,
                    envelope.artifact_id,
                    publication_intent,
                    published_identity,
                    allow_missing=True,
                )
            )
            if activation_preparation is not None:
                activation_witness = self.resolver.resolve_artifact_activation_witness(
                    envelope.scope_id,
                    envelope.artifact_kind,
                    envelope.artifact_id,
                    publication_intent,
                    published_identity,
                    allow_missing=True,
                )
                if activation_witness is not None:
                    return PublicationTelemetry(
                        publication_record=published_identity.publication_record,
                        expected_parent_sha=envelope.expected_parent_sha,
                        source_commit_sha=(
                            published_identity.publication_record.commit_sha
                        ),
                        pointer_commit_sha=(activation_witness.activation_commit_sha),
                        source_tree_digest=source_tree_digest,
                        validation_closure_ids=envelope.validation_closure_ids,
                        idempotent_replay=True,
                    )
            if activation_gate is not None:
                activation_gate.validate_before_publication(
                    envelope=envelope,
                    repositories=repositories,
                    current_state=current_state,
                    manifest=manifest,
                    source_tree_digest=source_tree_digest,
                    manifest_digest=manifest_digest,
                )
            if existing == published_identity:
                pointer_commit_sha = observed_head
                activation_commit_sha = (
                    self.resolver.resolve_artifact_activation_preparation(
                        envelope.scope_id,
                        envelope.artifact_kind,
                        envelope.artifact_id,
                        publication_intent,
                        published_identity,
                    )
                )
                if activation_commit_sha != pointer_commit_sha:
                    raise GitHubPublicationError(
                        "active CURRENT differs from its activation preparation"
                    )
                self.finalize_artifact_activation_witness(
                    envelope.scope_id,
                    envelope.artifact_kind,
                    envelope.artifact_id,
                    publication_intent,
                    published_identity,
                )
            elif observed_head == envelope.expected_parent_sha:
                pointer_commit_sha = self._prepare_current_pointer_commit(
                    repository,
                    published_identity.publication_record.commit_sha,
                    published_identity,
                    publication_intent,
                    envelope.committed_at,
                )
                self._commit_artifact_activation_preparation(
                    repository,
                    envelope,
                    publication_intent,
                    published_identity,
                    pointer_commit_sha,
                )
                self._finalize_expected_parent_witness(envelope)
                if activation_gate is not None:
                    self._validate_activation_pointer(
                        envelope,
                        published_identity,
                        manifest_digest,
                    )
                    activation_gate.revalidate_before_activation(
                        envelope=envelope,
                        repositories=repositories,
                        pointer=published_identity,
                        manifest=manifest,
                    )
                self.resolver.require_artifact_pointer(
                    envelope.scope_id,
                    envelope.artifact_kind,
                    envelope.artifact_id,
                    published_identity,
                )
                self._activate_current_pointer(
                    repository,
                    policy.repository_node_id,
                    envelope.expected_parent_sha,
                    pointer_commit_sha,
                )
                self.finalize_artifact_activation_witness(
                    envelope.scope_id,
                    envelope.artifact_kind,
                    envelope.artifact_id,
                    publication_intent,
                    published_identity,
                )
            else:
                raise GitHubCompareAndSwapError(
                    "published artifact is immutable but is not the active CURRENT"
                )
            return PublicationTelemetry(
                publication_record=published_identity.publication_record,
                expected_parent_sha=envelope.expected_parent_sha,
                source_commit_sha=published_identity.publication_record.commit_sha,
                pointer_commit_sha=pointer_commit_sha,
                source_tree_digest=source_tree_digest,
                validation_closure_ids=envelope.validation_closure_ids,
                idempotent_replay=True,
            )
        if activation_gate is not None:
            activation_gate.validate_before_publication(
                envelope=envelope,
                repositories=repositories,
                current_state=current_state,
                manifest=manifest,
                source_tree_digest=source_tree_digest,
                manifest_digest=manifest_digest,
            )
        if existing is not None and (
            existing.publication_record.artifact_id == envelope.artifact_id
        ):
            raise GitHubPublicationError(
                "active artifact has no write-once publication identity ref"
            )
        if publication_intent is not None:
            self._validate_publication_intent(
                envelope,
                policy,
                source_files,
                source_tree_digest,
                materialized_tree_digest,
                manifest_digest,
                publication_intent,
            )
            self._validate_intent_source_commit(repository, publication_intent)
            source_tree_sha = publication_intent.source_git_tree_sha
            source_commit_sha = publication_intent.source_commit_sha
        else:
            commit_source_files = source_files
            if preserved_current is not None:
                if not isinstance(preserved_current, _SourceFile):
                    raise GitHubPublicationError(
                        "new publication has a remote-only preserved pointer"
                    )
                commit_source_files = (
                    *source_files,
                    preserved_current,
                )
            expected_source_tree_sha = self._git_tree_sha(commit_source_files)
            if observed_head != envelope.expected_parent_sha:
                raise GitHubCompareAndSwapError(
                    "default branch has a stale expected parent"
                )
            source_tree_sha = self._create_git_tree(
                repository, commit_source_files, expected_source_tree_sha
            )
            source_commit_sha = self._create_commit(
                repository,
                tree_sha=source_tree_sha,
                parent_sha=envelope.expected_parent_sha,
                message=f"Publish {envelope.artifact_id}",
                committed_at=envelope.committed_at,
            )
            self._validate_source_commit(
                repository,
                source_commit_sha,
                expected_source_tree_sha,
                envelope.expected_parent_sha,
            )
            publication_intent = self._build_publication_intent(
                envelope,
                policy,
                source_commit_sha,
                source_tree_sha,
                source_files,
                preserved_current,
                source_tree_digest,
                materialized_tree_digest,
                manifest_digest,
            )
            self._validate_intent_source_commit(repository, publication_intent)
            self._commit_publication_intent(
                repository,
                source_commit_sha,
                publication_intent,
                envelope.committed_at,
            )
            self.resolver.require_artifact_intent(
                envelope.scope_id,
                envelope.artifact_kind,
                envelope.artifact_id,
                publication_intent,
            )
        self.client.create_ref_if_absent(
            repository,
            f"refs/tags/{envelope.tag}",
            source_commit_sha,
        )
        release_id = self.resolver.find_release_id(repository, envelope.tag)
        published = self._complete_release(
            repository,
            envelope,
            source_commit_sha,
            release_id,
            policy.authenticated_actor,
        )
        release_id = published.get("id")
        if (
            published.get("draft") is not False
            or published.get("immutable") is not True
            or type(release_id) is not int
            or release_id < 1
        ):
            raise GitHubPublicationError("published release is not immutable")
        if published.get("tag_name") != envelope.tag:
            raise GitHubPublicationError("published release tag mismatch")
        author = _require_mapping(published.get("author"), "published release author")
        if author.get("login") != policy.authenticated_actor:
            raise GitHubPublicationError("published release author mismatch")
        self._verify_release_assets(published, envelope.assets)
        self._validate_tag_commit(repository, envelope.tag, source_commit_sha)
        release_assets = self._publication_assets(published)
        asset_digests = {asset.name: asset.sha256 for asset in release_assets}
        attestation = validate_release_attestation(
            self.client.verify_release(
                repository,
                envelope.tag,
                source_commit_sha,
                asset_digests,
            ),
            repository=repository,
            tag=envelope.tag,
            commit_sha=source_commit_sha,
            asset_digests=asset_digests,
            error_type=GitHubPublicationError,
        )
        published_at = published.get("published_at")
        if not isinstance(published_at, str):
            raise GitHubPublicationError("published release has no timestamp")
        record = GitHubPublicationRecord.mint(
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            repository_node_id=policy.repository_node_id,
            repository_full_name=repository,
            commit_sha=source_commit_sha,
            immutable_release_id=str(release_id),
            tag=envelope.tag,
            assets=release_assets,
            release_attestation_ref=release_attestation_reference(attestation),
            published_at=published_at,
            publisher_identity=policy.authenticated_actor,
        )
        pointer = CurrentArtifactPointer(
            scope_id=envelope.scope_id,
            publication_record=record,
            publication_intent_digest=publication_intent.digest,
            source_tree_digest=source_tree_digest,
            source_git_tree_sha=source_tree_sha,
            materialized_tree_digest=materialized_tree_digest,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=manifest_digest,
            validation_closure_ids=envelope.validation_closure_ids,
        )
        self._commit_artifact_identity(
            repository,
            source_commit_sha,
            pointer,
            envelope.committed_at,
        )
        pointer_commit_sha = self._prepare_current_pointer_commit(
            repository,
            source_commit_sha,
            pointer,
            publication_intent,
            envelope.committed_at,
        )
        self._commit_artifact_activation_preparation(
            repository,
            envelope,
            publication_intent,
            pointer,
            pointer_commit_sha,
        )
        self._finalize_expected_parent_witness(envelope)
        if activation_gate is not None:
            self._validate_activation_pointer(envelope, pointer, manifest_digest)
            activation_gate.revalidate_before_activation(
                envelope=envelope,
                repositories=repositories,
                pointer=pointer,
                manifest=manifest,
            )
        self.resolver.require_artifact_pointer(
            envelope.scope_id,
            envelope.artifact_kind,
            envelope.artifact_id,
            pointer,
        )
        self._activate_current_pointer(
            repository,
            policy.repository_node_id,
            envelope.expected_parent_sha,
            pointer_commit_sha,
        )
        self.finalize_artifact_activation_witness(
            envelope.scope_id,
            envelope.artifact_kind,
            envelope.artifact_id,
            publication_intent,
            pointer,
        )
        return PublicationTelemetry(
            publication_record=record,
            expected_parent_sha=envelope.expected_parent_sha,
            source_commit_sha=source_commit_sha,
            pointer_commit_sha=pointer_commit_sha,
            source_tree_digest=source_tree_digest,
            validation_closure_ids=envelope.validation_closure_ids,
            idempotent_replay=False,
        )

    def _authorize_publication(
        self,
        envelope: PublicationEnvelope,
        verifier: _PublicationActivationVerifier,
    ) -> _ArtifactPublicationAuthorization:
        artifact_kind = envelope.artifact_kind
        verifier_type = self._activation_verifier_types.get(artifact_kind)
        if type(verifier) is not verifier_type:
            raise GitHubPublicationError(
                f"{artifact_kind.value} publication requires the registered "
                "concrete verifier"
            )
        return _ArtifactPublicationAuthorization(
            _PUBLICATION_AUTHORIZATION_SEAL,
            self,
            envelope,
            verifier,
        )

    def _validate_activation_pointer(
        self,
        envelope: PublicationEnvelope,
        pointer: CurrentArtifactPointer,
        manifest_digest: str,
    ) -> None:
        record = pointer.publication_record
        if (
            pointer.scope_id != envelope.scope_id
            or record.artifact_kind is not envelope.artifact_kind
            or record.artifact_id != envelope.artifact_id
            or pointer.manifest_relative_path != envelope.manifest_relative_path
            or pointer.manifest_digest != manifest_digest
            or pointer.validation_closure_ids != envelope.validation_closure_ids
        ):
            raise GitHubPublicationError(
                "publication activation pointer does not match its envelope"
            )

    def _bind_activation_verifier(
        self,
        artifact_kind: PublicationArtifactKind,
        verifier_type: type[object],
    ) -> None:
        authority = _ACTIVATION_VERIFIER_AUTHORITIES.get(artifact_kind)
        if authority is None:
            raise GitHubPublicationError(
                f"{artifact_kind.value} has no publication activation authority"
            )
        module_name, class_name = authority
        authority_module = sys.modules.get(module_name)
        expected_type = (
            None if authority_module is None else vars(authority_module).get(class_name)
        )
        if verifier_type is not expected_type:
            raise GitHubPublicationError(
                f"{artifact_kind.value} publication verifier type is not the "
                "concrete authority"
            )
        bound_type = self._activation_verifier_types.get(artifact_kind)
        if bound_type is not None and bound_type is not verifier_type:
            raise GitHubPublicationError(
                f"{artifact_kind.value} publication verifier authority is "
                "already bound"
            )
        self._activation_verifier_types[artifact_kind] = verifier_type

    def _validate_envelope(
        self,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
    ) -> tuple[
        tuple[_SourceFile, ...],
        str,
        str,
        KnowledgeSnapshotManifest
        | ExpertBaseReleaseManifest
        | SecurityDenylistSnapshot,
    ]:
        require_content_id(envelope.artifact_id, "artifact_id")
        _require_sha(envelope.expected_parent_sha, "expected_parent_sha")
        normalize_utc_timestamp(envelope.committed_at, "committed_at")
        if (
            envelope.validation_closure_ids
            != tuple(sorted(set(envelope.validation_closure_ids)))
            or not envelope.validation_closure_ids
        ):
            raise GitHubPublicationError(
                "validation closure must be non-empty, sorted, and unique"
            )
        for reference in envelope.validation_closure_ids:
            require_content_id(reference, "validation_closure_ids")
        expected_prefix = tag_prefix_for_artifact(
            self.settings,
            envelope.artifact_kind,
        )
        if (
            not envelope.tag.startswith(expected_prefix)
            or envelope.tag == expected_prefix
        ):
            raise GitHubPublicationError("publication tag uses an invalid prefix")
        require_git_ref_name(
            f"refs/tags/{envelope.tag}",
            "publication tag",
            qualified=True,
            error_type=GitHubPublicationError,
        )
        source_files = self._source_files(
            envelope.source_tree,
            self.settings.source_tree_size_bytes,
            self.settings.source_entry_limit,
        )
        manifest_path = PurePosixPath(envelope.manifest_relative_path)
        if (
            manifest_path.is_absolute()
            or ".." in manifest_path.parts
            or manifest_path.as_posix() != envelope.manifest_relative_path
        ):
            raise GitHubPublicationError(
                "manifest path must be normalized and relative"
            )
        file_by_path = {source.relative_path: source for source in source_files}
        if envelope.manifest_relative_path not in file_by_path:
            raise GitHubPublicationError("manifest is absent from source tree")
        manifest_bytes = file_by_path[envelope.manifest_relative_path].content
        if len(manifest_bytes) > self.settings.control_blob_size_bytes:
            raise GitHubPublicationError(
                "publication manifest exceeds configured control bound"
            )
        if envelope.artifact_kind is PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
            manifest = KnowledgeSnapshotManifest.from_json_bytes(manifest_bytes)
            manifest_id = manifest.snapshot_id
            manifest_scope = manifest.scope_id
        elif envelope.artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE:
            manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_bytes)
            manifest_id = manifest.release_id
            manifest_scope = manifest.scope_id
            required_validation_closure = tuple(
                sorted(
                    {
                        manifest.release_id,
                        *manifest.consumed_dependency_ids,
                        *manifest.control_dependency_ids,
                    }
                )
            )
            if envelope.validation_closure_ids != required_validation_closure:
                raise GitHubPublicationError(
                    "expert release publication dependency closure is not exact"
                )
        elif envelope.artifact_kind is PublicationArtifactKind.SECURITY_DENYLIST:
            manifest = SecurityDenylistSnapshot.from_json_bytes(manifest_bytes)
            manifest_id = manifest.snapshot_id
            manifest_scope = manifest.scope_id
            if manifest.scope_repository_binding_hash != (
                repositories.binding_fingerprint
            ):
                raise GitHubPublicationError(
                    "security denylist repository binding mismatch"
                )
            if envelope.tag != security_denylist_tag(
                self.settings,
                manifest.generation,
            ):
                raise GitHubPublicationError(
                    "security denylist publication tag is not its generation"
                )
            evidence_source = file_by_path.get(SECURITY_DENYLIST_EVIDENCE_FILENAME)
            if evidence_source is None:
                raise GitHubPublicationError(
                    "security denylist evidence bundle is absent"
                )
            evidence_bundle = SecurityDenylistEvidenceBundle.from_json_bytes(
                evidence_source.content
            )
            if evidence_source.content != evidence_bundle.to_json_bytes():
                raise GitHubPublicationError(
                    "security denylist evidence bundle is not canonical"
                )
            manifest.validate_evidence_bundle(evidence_bundle)
            required_validation_closure = tuple(
                sorted(
                    {
                        manifest.snapshot_id,
                        *manifest.exact_dependency_ids,
                    }
                )
            )
            if envelope.validation_closure_ids != required_validation_closure:
                raise GitHubPublicationError(
                    "security denylist publication dependency closure is not exact"
                )
        else:
            raise GitHubPublicationError("publication artifact kind is unsupported")
        if manifest_bytes != manifest.to_json_bytes():
            raise GitHubPublicationError("publication manifest is not canonical")
        if manifest_id != envelope.artifact_id:
            raise GitHubPublicationError("manifest artifact identity mismatch")
        if manifest_scope != envelope.scope_id:
            raise GitHubPublicationError("manifest scope mismatch")
        self._validate_assets(envelope.assets)
        return (
            source_files,
            source_tree_digest(
                {
                    source.relative_path: (
                        source.digest,
                        source.mode,
                        len(source.content),
                    )
                    for source in source_files
                }
            ),
            tree_or_blob_digest(manifest_bytes),
            manifest,
        )

    def _preserved_current(
        self, pointer: CurrentArtifactPointer | None
    ) -> _SourceFile | None:
        if pointer is None:
            return None
        payload = pointer.to_json_bytes()
        return _SourceFile(
            relative_path="CURRENT.json",
            content=payload,
            mode="100644",
            digest=tree_or_blob_digest(payload),
        )

    def _validate_publication_closure_bounds(
        self,
        source_files: tuple[_SourceFile, ...],
        assets: tuple[ReleaseAssetInput, ...],
        preserved_current: _SourceFile | PublicationSourceFile | None,
    ) -> None:
        source_bytes = sum(len(source.content) for source in source_files)
        if preserved_current is not None:
            source_bytes += (
                len(preserved_current.content)
                if isinstance(preserved_current, _SourceFile)
                else preserved_current.size
            )
        if source_bytes > self.settings.source_tree_size_bytes:
            raise GitHubPublicationError(
                "complete publication source exceeds configured size limit"
            )
        source_paths = [source.relative_path for source in source_files]
        if preserved_current is not None:
            source_paths.append(preserved_current.relative_path)
        source_directories = {
            parent.as_posix()
            for relative_path in source_paths
            for parent in PurePosixPath(relative_path).parents
            if parent.as_posix() != "."
        }
        if (
            len(source_paths) + len(source_directories)
            > self.settings.source_entry_limit
        ):
            raise GitHubPublicationError(
                "complete publication source exceeds configured entry limit"
            )
        publication_bytes = source_bytes + sum(asset.size for asset in assets)
        if publication_bytes > self.settings.materialized_asset_size_bytes:
            raise GitHubPublicationError("publication exceeds configured size limit")

    def _source_files(
        self,
        source_tree: Path,
        maximum_bytes: int,
        maximum_entries: int,
    ) -> tuple[_SourceFile, ...]:
        if source_tree.is_symlink():
            raise GitHubPublicationError("source tree cannot be a symlink")
        root = source_tree.resolve()
        if not root.is_dir():
            raise GitHubPublicationError("source tree must be a directory")
        files: list[_SourceFile] = []
        pending = [root]
        total_bytes = 0
        visited_entries = 0
        while pending:
            directory = pending.pop()
            children: list[Path] = []
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if visited_entries >= maximum_entries:
                        raise GitHubPublicationError(
                            "publication source exceeds configured entry limit"
                        )
                    visited_entries += 1
                    children.append(Path(entry.path))
            for path in reversed(sorted(children)):
                if path.is_symlink():
                    raise GitHubPublicationError(
                        "publication source cannot contain symlinks"
                    )
                relative = path.relative_to(root).as_posix()
                if ".git" in PurePosixPath(relative).parts or relative == ".gitmodules":
                    raise GitHubPublicationError(
                        "publication source cannot contain Git metadata"
                    )
                if relative == "CURRENT.json":
                    raise GitHubPublicationError("CURRENT.json is publisher-owned")
                if path.is_dir():
                    pending.append(path)
                    continue
                descriptor = os.open(
                    path,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                )
                with os.fdopen(descriptor, "rb") as source_handle:
                    metadata = os.fstat(source_handle.fileno())
                    if not stat.S_ISREG(metadata.st_mode):
                        raise GitHubPublicationError(
                            "publication source must contain regular files"
                        )
                    remaining_bytes = maximum_bytes - total_bytes
                    content = source_handle.read(remaining_bytes + 1)
                total_bytes += len(content)
                if total_bytes > maximum_bytes:
                    raise GitHubPublicationError(
                        "publication source exceeds configured size limit"
                    )
                mode = "100755" if metadata.st_mode & 0o111 else "100644"
                files.append(
                    _SourceFile(
                        relative_path=relative,
                        content=content,
                        mode=mode,
                        digest=tree_or_blob_digest(content),
                    )
                )
        if not files:
            raise GitHubPublicationError("publication source tree must not be empty")
        return tuple(sorted(files, key=lambda source: source.relative_path))

    def _validate_assets(self, assets: tuple[ReleaseAssetInput, ...]) -> None:
        if not assets or tuple(asset.name for asset in assets) != tuple(
            sorted({asset.name for asset in assets})
        ):
            raise GitHubPublicationError("assets must be non-empty, sorted, and unique")
        if len(assets) > self.settings.release_asset_count_limit:
            raise GitHubPublicationError("release asset count exceeds configured limit")
        total_size = 0
        for asset in assets:
            asset_name = PurePosixPath(asset.name)
            if (
                asset_name.is_absolute()
                or len(asset_name.parts) != 1
                or asset_name.as_posix() != asset.name
                or _RELEASE_ASSET_NAME_PATTERN.fullmatch(asset.name) is None
            ):
                raise GitHubPublicationError(
                    "release asset names must be stable GitHub basenames"
                )
            if asset.path.is_symlink() or not asset.path.is_file():
                raise GitHubPublicationError("release asset must be a regular file")
            if type(asset.size) is not int or asset.size < 1:
                raise GitHubPublicationError("release asset size must be positive")
            if asset.size > self.settings.release_asset_size_bytes:
                raise GitHubPublicationError(
                    "release asset exceeds configured size limit"
                )
            total_size += asset.size
            if total_size > self.settings.materialized_asset_size_bytes:
                raise GitHubPublicationError(
                    "release asset closure exceeds configured size limit"
                )
            if asset.size != asset.path.stat().st_size:
                raise GitHubPublicationError("release asset size mismatch")
            if asset.sha256 != self._file_digest(asset.path):
                raise GitHubPublicationError("release asset digest mismatch")
            if _MEDIA_TYPE_PATTERN.fullmatch(asset.media_type) is None:
                raise GitHubPublicationError("release asset media type is invalid")

    def _file_digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _build_publication_intent(
        self,
        envelope: PublicationEnvelope,
        policy: RepositoryPolicyReport,
        source_commit_sha: str,
        source_git_tree_sha: str,
        source_files: tuple[_SourceFile, ...],
        preserved_current: _SourceFile | PublicationSourceFile | None,
        source_tree_digest: str,
        materialized_tree_digest: str,
        manifest_digest: str,
    ) -> ArtifactPublicationIntent:
        return ArtifactPublicationIntent(
            scope_id=envelope.scope_id,
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            repository_node_id=policy.repository_node_id,
            repository_full_name=policy.repository_full_name,
            expected_parent_sha=envelope.expected_parent_sha,
            source_commit_sha=source_commit_sha,
            source_tree_digest=source_tree_digest,
            source_git_tree_sha=source_git_tree_sha,
            source_files=tuple(
                PublicationSourceFile(
                    relative_path=source.relative_path,
                    mode=source.mode,
                    size=len(source.content),
                    sha256=source.digest,
                    git_blob_sha=git_object_sha("blob", source.content),
                )
                for source in source_files
            ),
            preserved_current=(
                None
                if preserved_current is None
                else (
                    preserved_current
                    if isinstance(preserved_current, PublicationSourceFile)
                    else PublicationSourceFile(
                        relative_path=preserved_current.relative_path,
                        mode=preserved_current.mode,
                        size=len(preserved_current.content),
                        sha256=preserved_current.digest,
                        git_blob_sha=git_object_sha("blob", preserved_current.content),
                    )
                )
            ),
            materialized_tree_digest=materialized_tree_digest,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=manifest_digest,
            tag=envelope.tag,
            assets=tuple(
                PublicationAssetIntent(
                    name=asset.name,
                    media_type=asset.media_type,
                    size=asset.size,
                    sha256=asset.sha256,
                )
                for asset in envelope.assets
            ),
            validation_closure_ids=envelope.validation_closure_ids,
            publisher_identity=policy.authenticated_actor,
            committed_at=envelope.committed_at,
        )

    def _validate_publication_intent(
        self,
        envelope: PublicationEnvelope,
        policy: RepositoryPolicyReport,
        source_files: tuple[_SourceFile, ...],
        source_tree_digest: str,
        materialized_tree_digest: str,
        manifest_digest: str,
        intent: ArtifactPublicationIntent,
    ) -> None:
        expected = self._build_publication_intent(
            envelope,
            policy,
            intent.source_commit_sha,
            intent.source_git_tree_sha,
            source_files,
            intent.preserved_current,
            source_tree_digest,
            materialized_tree_digest,
            manifest_digest,
        )
        if expected != intent:
            raise GitHubPublicationError(
                "artifact publication intent conflicts with replayed bytes"
            )

    def _validate_intent_source_commit(
        self, repository: str, intent: ArtifactPublicationIntent
    ) -> None:
        self.resolver.verify_publication_intent_source(repository, intent)

    def _validate_source_commit(
        self,
        repository: str,
        source_commit_sha: str,
        expected_tree_sha: str,
        expected_parent_sha: str,
    ) -> None:
        commit = _require_mapping(
            self.client.api_json(
                "GET", f"repos/{repository}/git/commits/{source_commit_sha}"
            ),
            "publication intent source commit",
        )
        tree = _require_mapping(commit.get("tree"), "publication intent source tree")
        parents = commit.get("parents")
        if (
            tree.get("sha") != expected_tree_sha
            or not isinstance(parents, list)
            or len(parents) != 1
        ):
            raise GitHubPublicationError("publication intent source commit mismatch")
        parent = _require_mapping(parents[0], "publication intent source parent")
        if parent.get("sha") != expected_parent_sha:
            raise GitHubPublicationError("publication intent source parent mismatch")

    def _validate_idempotent_replay(
        self,
        envelope: PublicationEnvelope,
        source_tree_digest: str,
        materialized_tree_digest: str,
        manifest_digest: str,
        existing: CurrentArtifactPointer,
    ) -> None:
        expected_assets = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in envelope.assets
        )
        existing_assets = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in existing.publication_record.assets
        )
        if (
            existing.scope_id != envelope.scope_id
            or existing.source_tree_digest != source_tree_digest
            or existing.materialized_tree_digest != materialized_tree_digest
            or existing.manifest_relative_path != envelope.manifest_relative_path
            or existing.manifest_digest != manifest_digest
            or existing.validation_closure_ids != envelope.validation_closure_ids
            or existing.publication_record.tag != envelope.tag
            or existing_assets != expected_assets
        ):
            raise GitHubPublicationError(
                "artifact identity replay has conflicting publication bytes"
            )

    def _complete_release(
        self,
        repository: str,
        envelope: PublicationEnvelope,
        source_commit_sha: str,
        release_id: int | None,
        expected_author: str,
    ) -> Mapping[str, Any]:
        self._validate_tag_commit(repository, envelope.tag, source_commit_sha)
        if release_id is None:
            draft = _require_mapping(
                self.client.api_json(
                    "POST",
                    f"repos/{repository}/releases",
                    {
                        "draft": True,
                        "name": envelope.tag,
                        "prerelease": False,
                        "tag_name": envelope.tag,
                        "target_commitish": source_commit_sha,
                    },
                ),
                "draft release",
            )
            release_id = draft.get("id")
            if type(release_id) is not int or release_id < 1:
                raise GitHubPublicationError("draft release identity is invalid")
            self._validate_release_target(
                draft,
                release_id,
                envelope.tag,
            )
            self._validate_release_author(draft, expected_author)
            if draft.get("draft") is not True:
                raise GitHubPublicationError("new release is not a draft")
            release = draft
        else:
            release = self._get_release(repository, release_id)
            self._validate_release_target(
                release,
                release_id,
                envelope.tag,
            )
            self._validate_release_author(release, expected_author)
        if release.get("draft") is True:
            release = self._remove_failed_upload_assets(
                repository,
                release_id,
                envelope,
                release,
                expected_author,
            )
            present_assets = self._publication_assets(release, allow_empty=True)
            missing_assets = self._missing_release_assets(
                present_assets, envelope.assets
            )
            for asset in missing_assets:
                self.client.upload_release_asset(
                    repository,
                    release_id,
                    asset.path,
                    asset.name,
                    asset.media_type,
                    asset.size,
                )
            draft_with_assets = self._get_release(repository, release_id)
            self._validate_release_target(
                draft_with_assets,
                release_id,
                envelope.tag,
            )
            self._validate_release_author(draft_with_assets, expected_author)
            if draft_with_assets.get("draft") is not True:
                raise GitHubPublicationError(
                    "release became public before verification"
                )
            self._verify_release_assets(draft_with_assets, envelope.assets)
            self._validate_tag_commit(repository, envelope.tag, source_commit_sha)
            return _require_mapping(
                self.client.api_json(
                    "PATCH",
                    f"repos/{repository}/releases/{release_id}",
                    {"draft": False},
                ),
                "published release",
            )
        if release.get("draft") is False and release.get("immutable") is True:
            self._verify_release_assets(release, envelope.assets)
            return release
        raise GitHubPublicationError(
            "existing release is neither a draft nor immutable publication"
        )

    def _remove_failed_upload_assets(
        self,
        repository: str,
        release_id: int,
        envelope: PublicationEnvelope,
        release: Mapping[str, Any],
        expected_author: str,
    ) -> Mapping[str, Any]:
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise GitHubPublicationError("draft release asset list is invalid")
        if len(assets) > self.settings.release_asset_count_limit:
            raise GitHubPublicationError("release asset count exceeds configured limit")
        expected_names = {asset.name for asset in envelope.assets}
        deleted = False
        for asset_value in assets:
            asset = _require_mapping(asset_value, "draft release asset")
            if asset.get("state") != "starter":
                continue
            asset_id = asset.get("id")
            if (
                type(asset_id) is not int
                or asset_id < 1
                or asset.get("name") not in expected_names
                or asset.get("size") != 0
                or asset.get("digest") not in (None, "")
            ):
                raise GitHubPublicationError(
                    "failed-upload starter asset is not safely reclaimable"
                )
            self.client.delete_release_asset(repository, asset_id)
            deleted = True
        if not deleted:
            return release
        refreshed = self._get_release(repository, release_id)
        self._validate_release_target(refreshed, release_id, envelope.tag)
        self._validate_release_author(refreshed, expected_author)
        if refreshed.get("draft") is not True:
            raise GitHubPublicationError("release became public during asset recovery")
        return refreshed

    def _validate_release_author(
        self, release: Mapping[str, Any], expected_author: str
    ) -> None:
        author = _require_mapping(release.get("author"), "release author")
        if author.get("login") != expected_author:
            raise GitHubPublicationError("release author mismatch")

    def _validate_release_target(
        self,
        release: Mapping[str, Any],
        release_id: int,
        tag: str,
    ) -> None:
        if release.get("id") != release_id or release.get("tag_name") != tag:
            raise GitHubPublicationError("release target mismatch")

    def _validate_tag_commit(
        self,
        repository: str,
        tag: str,
        source_commit_sha: str,
    ) -> None:
        tag_reference = _require_mapping(
            self.client.api_json("GET", f"repos/{repository}/git/ref/tags/{tag}"),
            "publication tag ref",
        )
        if tag_reference.get("ref") != f"refs/tags/{tag}":
            raise GitHubPublicationError("publication tag ref identity mismatch")
        target = _require_mapping(tag_reference.get("object"), "publication tag target")
        if target.get("type") != "commit" or target.get("sha") != source_commit_sha:
            raise GitHubPublicationError(
                "publication tag must directly target the source commit"
            )

    def _missing_release_assets(
        self,
        present: tuple[GitHubReleaseAsset, ...],
        expected: tuple[ReleaseAssetInput, ...],
    ) -> tuple[ReleaseAssetInput, ...]:
        expected_by_name = {asset.name: asset for asset in expected}
        for asset in present:
            wanted = expected_by_name.get(asset.name)
            if wanted is None or (
                asset.media_type,
                asset.size,
                asset.sha256,
            ) != (wanted.media_type, wanted.size, wanted.sha256):
                raise GitHubPublicationError(
                    "draft release contains a conflicting asset"
                )
        present_names = {asset.name for asset in present}
        return tuple(asset for asset in expected if asset.name not in present_names)

    def _git_tree_sha(self, source_files: tuple[_SourceFile, ...]) -> str:
        files = {
            source.relative_path: (git_object_sha("blob", source.content), source.mode)
            for source in source_files
        }
        return git_tree_shas(files)[""]

    def _create_git_tree(
        self,
        repository: str,
        source_files: tuple[_SourceFile, ...],
        expected_tree_sha: str,
    ) -> str:
        entries = []
        for source in source_files:
            content = source.content.decode("utf-8")
            entries.append(
                {
                    "content": content,
                    "mode": source.mode,
                    "path": source.relative_path,
                    "type": "blob",
                }
            )
        body = {"tree": entries}
        if len(canonical_json_bytes(body)) > self.settings.git_tree_request_size_bytes:
            raise GitHubPublicationError(
                "Git source tree request exceeds configured size limit"
            )
        response = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/trees",
                body,
            ),
            "created tree",
        )
        tree_sha = _require_sha(response.get("sha"), "tree sha")
        if tree_sha != expected_tree_sha:
            raise GitHubPublicationError("GitHub created an unexpected source tree")
        return tree_sha

    def _create_commit(
        self,
        repository: str,
        *,
        tree_sha: str,
        parent_sha: str,
        message: str,
        committed_at: str,
    ) -> str:
        identity = {
            "date": committed_at,
            "email": self.settings.commit_author_email,
            "name": self.settings.commit_author_name,
        }
        response = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/commits",
                {
                    "author": identity,
                    "committer": identity,
                    "message": message,
                    "parents": [parent_sha],
                    "tree": tree_sha,
                },
            ),
            "created commit",
        )
        return _require_sha(response.get("sha"), "commit sha")

    def _get_release(self, repository: str, release_id: int) -> Mapping[str, Any]:
        return _require_mapping(
            self.client.api_json("GET", f"repos/{repository}/releases/{release_id}"),
            "release",
        )

    def _verify_release_assets(
        self,
        release: Mapping[str, Any],
        expected: tuple[ReleaseAssetInput, ...],
    ) -> None:
        actual = self._publication_assets(release)
        expected_values = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256)
            for asset in expected
        )
        actual_values = tuple(
            (asset.name, asset.media_type, asset.size, asset.sha256) for asset in actual
        )
        if actual_values != expected_values:
            raise GitHubPublicationError("GitHub release asset digest closure mismatch")

    def _publication_assets(
        self, release: Mapping[str, Any], *, allow_empty: bool = False
    ) -> tuple[GitHubReleaseAsset, ...]:
        assets = release.get("assets")
        if not isinstance(assets, list) or (not assets and not allow_empty):
            raise GitHubPublicationError("release has no assets")
        if len(assets) > self.settings.release_asset_count_limit:
            raise GitHubPublicationError("release asset count exceeds configured limit")
        result: list[GitHubReleaseAsset] = []
        for asset in assets:
            value = _require_mapping(asset, "release asset")
            asset_id = value.get("id")
            size = value.get("size")
            if type(asset_id) is not int or asset_id < 1:
                raise GitHubPublicationError("release asset ID must be positive")
            if type(size) is not int or size < 0:
                raise GitHubPublicationError("release asset size must be non-negative")
            name = value.get("name")
            media_type = value.get("content_type")
            digest = value.get("digest")
            if value.get("state") != "uploaded":
                raise GitHubPublicationError("release asset is not fully uploaded")
            if not isinstance(name, str) or not name:
                raise GitHubPublicationError("release asset name is invalid")
            if not isinstance(media_type, str) or not media_type:
                raise GitHubPublicationError("release asset content type is invalid")
            if not isinstance(digest, str):
                raise GitHubPublicationError("release asset digest is invalid")
            result.append(
                GitHubReleaseAsset(
                    asset_id=str(asset_id),
                    name=name,
                    media_type=media_type,
                    size=size,
                    sha256=digest,
                )
            )
        return tuple(sorted(result, key=lambda asset: asset.name))

    def _commit_artifact_identity(
        self,
        repository: str,
        source_commit_sha: str,
        pointer: CurrentArtifactPointer,
        committed_at: str,
    ) -> str:
        pointer_payload = pointer.to_json_bytes()
        if len(pointer_payload) > self.settings.control_blob_size_bytes:
            raise GitHubPublicationError(
                "artifact publication pointer exceeds configured control bound"
            )
        pointer_blob = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/blobs",
                {
                    "content": base64.b64encode(pointer_payload).decode("ascii"),
                    "encoding": "base64",
                },
            ),
            "artifact identity pointer blob",
        )
        identity_tree = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/trees",
                {
                    "tree": [
                        {
                            "mode": "100644",
                            "path": ARTIFACT_POINTER_FILENAME,
                            "sha": _require_sha(
                                pointer_blob.get("sha"),
                                "artifact identity pointer blob sha",
                            ),
                            "type": "blob",
                        }
                    ]
                },
            ),
            "artifact identity tree",
        )
        identity_commit_sha = self._create_commit(
            repository,
            tree_sha=_require_sha(
                identity_tree.get("sha"), "artifact identity tree sha"
            ),
            parent_sha=source_commit_sha,
            message=f"Record {pointer.publication_record.artifact_id}",
            committed_at=committed_at,
        )
        self.client.create_ref_if_absent(
            repository,
            artifact_identity_ref(
                pointer.publication_record.artifact_kind,
                pointer.publication_record.artifact_id,
            ),
            identity_commit_sha,
        )
        return identity_commit_sha

    def _commit_publication_intent(
        self,
        repository: str,
        source_commit_sha: str,
        intent: ArtifactPublicationIntent,
        committed_at: str,
    ) -> str:
        payload = intent.to_json_bytes()
        if len(payload) > self.settings.control_blob_size_bytes:
            raise GitHubPublicationError(
                "artifact publication intent exceeds configured control bound"
            )
        blob = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/blobs",
                {
                    "content": base64.b64encode(payload).decode("ascii"),
                    "encoding": "base64",
                },
            ),
            "artifact publication intent blob",
        )
        tree = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/trees",
                {
                    "tree": [
                        {
                            "mode": "100644",
                            "path": ARTIFACT_PUBLICATION_INTENT_FILENAME,
                            "sha": _require_sha(
                                blob.get("sha"), "publication intent blob sha"
                            ),
                            "type": "blob",
                        }
                    ]
                },
            ),
            "artifact publication intent tree",
        )
        commit_sha = self._create_commit(
            repository,
            tree_sha=_require_sha(tree.get("sha"), "publication intent tree sha"),
            parent_sha=source_commit_sha,
            message=f"Claim {intent.artifact_id}",
            committed_at=committed_at,
        )
        self.client.create_ref_if_absent(
            repository,
            artifact_publication_intent_ref(
                intent.artifact_kind,
                intent.artifact_id,
            ),
            commit_sha,
        )
        return commit_sha

    def _prepare_current_pointer_commit(
        self,
        repository: str,
        source_commit_sha: str,
        pointer: CurrentArtifactPointer,
        publication_intent: ArtifactPublicationIntent,
        committed_at: str,
    ) -> str:
        pointer_payload = pointer.to_json_bytes()
        if len(pointer_payload) > self.settings.control_blob_size_bytes:
            raise GitHubPublicationError(
                "CURRENT pointer exceeds configured control bound"
            )
        source_commit = _require_mapping(
            self.client.api_json(
                "GET", f"repos/{repository}/git/commits/{source_commit_sha}"
            ),
            "source commit",
        )
        source_tree = _require_mapping(source_commit.get("tree"), "source tree")
        source_tree_sha = _require_sha(source_tree.get("sha"), "source tree sha")
        pointer_blob = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/blobs",
                {
                    "content": base64.b64encode(pointer_payload).decode("ascii"),
                    "encoding": "base64",
                },
            ),
            "CURRENT blob",
        )
        pointer_blob_sha = _require_sha(
            pointer_blob.get("sha"),
            "CURRENT blob sha",
        )
        if pointer_blob_sha != git_object_sha("blob", pointer_payload):
            raise GitHubPublicationError("GitHub created an unexpected CURRENT blob")
        pointer_tree = _require_mapping(
            self.client.api_json(
                "POST",
                f"repos/{repository}/git/trees",
                {
                    "base_tree": source_tree_sha,
                    "tree": [
                        {
                            "mode": "100644",
                            "path": "CURRENT.json",
                            "sha": pointer_blob_sha,
                            "type": "blob",
                        }
                    ],
                },
            ),
            "CURRENT tree",
        )
        expected_files = {
            source.relative_path: (source.git_blob_sha, source.mode)
            for source in publication_intent.source_files
        }
        expected_files["CURRENT.json"] = (pointer_blob_sha, "100644")
        expected_pointer_tree_sha = git_tree_shas(expected_files)[""]
        pointer_tree_sha = _require_sha(
            pointer_tree.get("sha"),
            "CURRENT tree sha",
        )
        if pointer_tree_sha != expected_pointer_tree_sha:
            raise GitHubPublicationError("GitHub created an unexpected CURRENT tree")
        pointer_commit_sha = self._create_commit(
            repository,
            tree_sha=pointer_tree_sha,
            parent_sha=source_commit_sha,
            message=f"Activate {pointer.publication_record.artifact_id}",
            committed_at=committed_at,
        )
        self._validate_activation_commit(
            repository,
            pointer_commit_sha,
            expected_pointer_tree_sha,
            source_commit_sha,
        )
        return pointer_commit_sha

    def _validate_activation_commit(
        self,
        repository: str,
        activation_commit_sha: str,
        expected_tree_sha: str,
        source_commit_sha: str,
    ) -> None:
        commit = _require_mapping(
            self.client.api_json(
                "GET", f"repos/{repository}/git/commits/{activation_commit_sha}"
            ),
            "CURRENT activation commit",
        )
        tree = _require_mapping(commit.get("tree"), "CURRENT activation tree")
        parents = commit.get("parents")
        if (
            commit.get("sha") != activation_commit_sha
            or tree.get("sha") != expected_tree_sha
            or not isinstance(parents, list)
            or len(parents) != 1
        ):
            raise GitHubPublicationError("CURRENT activation commit mismatch")
        parent = _require_mapping(parents[0], "CURRENT activation parent")
        if parent.get("sha") != source_commit_sha:
            raise GitHubPublicationError("CURRENT activation parent mismatch")

    def _activate_current_pointer(
        self,
        repository: str,
        repository_node_id: str,
        expected_parent_sha: str,
        pointer_commit_sha: str,
    ) -> None:
        self.client.update_ref_compare_and_swap(
            repository,
            repository_node_id,
            self.settings.default_branch,
            expected_parent_sha,
            pointer_commit_sha,
        )

    def finalize_artifact_activation_witness(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
        intent: ArtifactPublicationIntent,
        pointer: CurrentArtifactPointer,
    ) -> GitHubArtifactActivationWitness:
        """Seal a success ref only for the exact prepared CURRENT commit."""

        activation_commit_sha = self.resolver.resolve_artifact_activation_preparation(
            scope_id,
            artifact_kind,
            artifact_id,
            intent,
            pointer,
        )
        existing = self.resolver.resolve_artifact_activation_witness(
            scope_id,
            artifact_kind,
            artifact_id,
            intent,
            pointer,
            allow_missing=True,
        )
        if existing is not None:
            return existing
        current = self.resolver.read_current_pointer_state(
            scope_id,
            artifact_kind,
            allow_missing=True,
        )
        if (
            current.pointer != pointer
            or current.head_commit_sha != activation_commit_sha
        ):
            raced = self.resolver.resolve_artifact_activation_witness(
                scope_id,
                artifact_kind,
                artifact_id,
                intent,
                pointer,
                allow_missing=True,
            )
            if raced is not None:
                return raced
            raise GitHubPublicationError(
                "activation witness requires the exact prepared CURRENT head"
            )
        repositories = self.resolver.repositories_for_scope(scope_id)
        repository = repository_for_artifact(repositories, artifact_kind)
        self.resolver.require_artifact_intent(
            scope_id,
            artifact_kind,
            artifact_id,
            intent,
        )
        self.resolver.require_artifact_pointer(
            scope_id,
            artifact_kind,
            artifact_id,
            pointer,
        )
        self.client.create_ref_if_absent(
            repository,
            artifact_activation_ref(artifact_kind, artifact_id),
            activation_commit_sha,
        )
        witnessed = self.resolver.resolve_artifact_activation_witness(
            scope_id,
            artifact_kind,
            artifact_id,
            intent,
            pointer,
        )
        if witnessed is None:
            raise GitHubPublicationError("artifact activation witness is missing")
        return witnessed

    def _finalize_expected_parent_witness(
        self,
        envelope: PublicationEnvelope,
    ) -> None:
        current = self.resolver.read_current_pointer_state(
            envelope.scope_id,
            envelope.artifact_kind,
            allow_missing=True,
        )
        if current.head_commit_sha != envelope.expected_parent_sha:
            raise GitHubCompareAndSwapError(
                "default branch changed before predecessor witness finalization"
            )
        pointer = current.pointer
        if pointer is None:
            return
        artifact_id = pointer.publication_record.artifact_id
        intent = self.resolver.read_artifact_intent(
            envelope.scope_id,
            envelope.artifact_kind,
            artifact_id,
        )
        identity = self.resolver.read_artifact_pointer(
            envelope.scope_id,
            envelope.artifact_kind,
            artifact_id,
        )
        if intent is None or identity != pointer:
            raise GitHubPublicationError(
                "current predecessor lacks its exact publication identity"
            )
        witness = self.finalize_artifact_activation_witness(
            envelope.scope_id,
            envelope.artifact_kind,
            artifact_id,
            intent,
            pointer,
        )
        if witness.activation_commit_sha != current.head_commit_sha:
            raise GitHubPublicationError(
                "current predecessor head differs from its activation witness"
            )

    def _commit_artifact_activation_preparation(
        self,
        repository: str,
        envelope: PublicationEnvelope,
        intent: ArtifactPublicationIntent,
        pointer: CurrentArtifactPointer,
        activation_commit_sha: str,
    ) -> None:
        self.client.create_ref_if_absent(
            repository,
            artifact_activation_preparation_ref(
                envelope.artifact_kind,
                envelope.artifact_id,
            ),
            activation_commit_sha,
        )
        observed = self.resolver.resolve_artifact_activation_preparation(
            envelope.scope_id,
            envelope.artifact_kind,
            envelope.artifact_id,
            intent,
            pointer,
        )
        if observed != activation_commit_sha:
            raise GitHubPublicationError(
                "artifact activation preparation differs from its commit"
            )
