"""Authenticated GitHub security-denylist resolution and local anti-rollback state."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    PublicationArtifactKind,
    SECURITY_DENYLIST_EVIDENCE_FILENAME,
    ScopeRepositorySettings,
    SecurityDenylistEvidenceBundle,
    SecurityDenylistSnapshot,
    StrictContract,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    PublicationEnvelope,
    PublicationTelemetry,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactResolver,
    ResolvedGitHubArtifact,
    security_denylist_tag,
)
from kapso.cross_run.settings import LaunchSettings, ScopeRegistrySettings

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_STAGING_PATTERN = re.compile(r"^(?P<scope>[0-9a-f]{64})-(?P<token>[0-9a-f]{32})\.tmp$")


def _read_independent_regular_file(
    path: Path,
    maximum_size_bytes: int,
    description: str,
) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SecurityDenylistError(
                f"{description} is not an independent regular file"
            )
        payload = handle.read(maximum_size_bytes + 1)
    if len(payload) > maximum_size_bytes:
        raise SecurityDenylistError(f"{description} exceeds its bound")
    return payload


class SecurityDenylistError(ValueError):
    """Security denylist authority is invalid, stale, corrupt, or unavailable."""


@dataclass(frozen=True)
class AuthenticatedSecurityDenylistSnapshot(StrictContract):
    observation_id: str
    snapshot: SecurityDenylistSnapshot
    publication_id: str
    repository_full_name: str
    repository_node_id: str
    authority_commit_sha: str
    pointer_digest: str
    release_attestation_ref: str
    validation_closure_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "authenticated-security-denylist-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_content_id(self.publication_id, "security denylist publication_id")
        if self.publication_id.split(":sha256:", 1)[0] != "github-publication":
            raise SecurityDenylistError(
                "security denylist publication uses the wrong namespace"
            )
        if _REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise SecurityDenylistError("security denylist repository is invalid")
        require_identifier(
            self.repository_node_id,
            "security denylist repository_node_id",
        )
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise SecurityDenylistError("security denylist authority commit is invalid")
        if _DIGEST_PATTERN.fullmatch(self.pointer_digest) is None:
            raise SecurityDenylistError("security denylist pointer digest is invalid")
        if not self.release_attestation_ref.strip():
            raise SecurityDenylistError(
                "security denylist release attestation is required"
            )
        if self.validation_closure_ids != tuple(
            sorted(set(self.validation_closure_ids))
        ):
            raise SecurityDenylistError(
                "security denylist validation closure must be sorted and unique"
            )
        for dependency_id in self.validation_closure_ids:
            require_content_id(dependency_id, "security denylist validation closure")
        required = {
            self.snapshot.snapshot_id,
            *self.snapshot.exact_dependency_ids,
        }
        if required != set(self.validation_closure_ids):
            raise SecurityDenylistError(
                "security denylist publication dependency closure is not exact"
            )


class SecurityDenylistSnapshotProvider(Protocol):
    """Resolve fully authenticated current and immutable historical snapshots."""

    def resolve_current(
        self,
        scope_id: str,
    ) -> AuthenticatedSecurityDenylistSnapshot: ...

    def resolve_exact(
        self,
        scope_id: str,
        snapshot_id: str,
    ) -> AuthenticatedSecurityDenylistSnapshot: ...


class GitHubSecurityDenylistSnapshotProvider:
    """Resolve denylist manifests through verified immutable GitHub releases."""

    def __init__(
        self,
        resolver: GitHubArtifactResolver,
        materializer: GitHubArtifactMaterializer,
    ) -> None:
        self.resolver = resolver
        self.materializer = materializer

    def resolve_current(
        self,
        scope_id: str,
    ) -> AuthenticatedSecurityDenylistSnapshot:
        return self._materialize(
            self.resolver.resolve_current(
                scope_id,
                PublicationArtifactKind.SECURITY_DENYLIST,
            )
        )

    def resolve_exact(
        self,
        scope_id: str,
        snapshot_id: str,
    ) -> AuthenticatedSecurityDenylistSnapshot:
        require_content_id(snapshot_id, "security denylist snapshot_id")
        if snapshot_id.split(":sha256:", 1)[0] != "security-denylist-snapshot":
            raise SecurityDenylistError(
                "security denylist exact snapshot uses the wrong namespace"
            )
        return self._materialize(
            self.resolver.resolve_artifact(
                scope_id,
                PublicationArtifactKind.SECURITY_DENYLIST,
                snapshot_id,
            )
        )

    def _materialize(
        self,
        resolved: ResolvedGitHubArtifact,
    ) -> AuthenticatedSecurityDenylistSnapshot:
        pointer = resolved.pointer
        record = pointer.publication_record
        repositories = resolved.repositories
        if (
            record.artifact_kind is not PublicationArtifactKind.SECURITY_DENYLIST
            or record.repository_full_name != repositories.security_repository
            or record.repository_node_id != resolved.policy.repository_node_id
        ):
            raise SecurityDenylistError(
                "security denylist resolved another repository authority"
            )
        materialized = self.materializer.materialize(resolved)
        manifest_path = materialized.content / pointer.manifest_relative_path
        payload = _read_independent_regular_file(
            manifest_path,
            self.resolver.settings.control_blob_size_bytes,
            "security denylist manifest",
        )
        snapshot = SecurityDenylistSnapshot.from_json_bytes(payload)
        if payload != snapshot.to_json_bytes():
            raise SecurityDenylistError("security denylist manifest is not canonical")
        if (
            snapshot.snapshot_id != record.artifact_id
            or snapshot.scope_id != repositories.scope_id
            or snapshot.scope_repository_binding_hash
            != repositories.binding_fingerprint
            or record.tag
            != security_denylist_tag(
                self.resolver.settings,
                snapshot.generation,
            )
        ):
            raise SecurityDenylistError(
                "security denylist manifest differs from its scope publication"
            )
        evidence_payload = _read_independent_regular_file(
            materialized.content / SECURITY_DENYLIST_EVIDENCE_FILENAME,
            self.resolver.settings.source_tree_size_bytes,
            "security denylist evidence bundle",
        )
        evidence_bundle = SecurityDenylistEvidenceBundle.from_json_bytes(
            evidence_payload
        )
        if evidence_payload != evidence_bundle.to_json_bytes():
            raise SecurityDenylistError(
                "security denylist evidence bundle is not canonical"
            )
        snapshot.validate_evidence_bundle(evidence_bundle)
        return AuthenticatedSecurityDenylistSnapshot.mint(
            snapshot=snapshot,
            publication_id=record.publication_id,
            repository_full_name=record.repository_full_name,
            repository_node_id=record.repository_node_id,
            authority_commit_sha=resolved.pointer_commit_sha,
            pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
            release_attestation_ref=record.release_attestation_ref,
            validation_closure_ids=pointer.validation_closure_ids,
        )


def _validate_snapshot_successor(
    child: SecurityDenylistSnapshot,
    parent: SecurityDenylistSnapshot,
) -> None:
    if (
        child.predecessor_snapshot_id != parent.snapshot_id
        or child.generation != parent.generation + 1
        or child.scope_id != parent.scope_id
        or child.scope_contract_id != parent.scope_contract_id
        or child.scope_repository_binding_hash != parent.scope_repository_binding_hash
        or child.schema_version != parent.schema_version
        or child.policy_version != parent.policy_version
    ):
        raise SecurityDenylistError(
            "security denylist predecessor authority is discontinuous"
        )
    parent_revocations = {item.revocation_id for item in parent.revocations}
    child_revocations = {item.revocation_id for item in child.revocations}
    if not parent_revocations.issubset(child_revocations):
        raise SecurityDenylistError("security denylist successor removes a revocation")


class SecurityDenylistPublicationGate:
    """Keep one authenticated predecessor stable through CURRENT activation."""

    def __init__(
        self,
        resolver: GitHubArtifactResolver,
        provider: GitHubSecurityDenylistSnapshotProvider,
        launch_settings: LaunchSettings,
    ) -> None:
        self.resolver = resolver
        self.provider = provider
        self.launch_settings = launch_settings
        self.expected_current_pointer: CurrentArtifactPointer | None = None
        self.expected_current_snapshot_id: str | None = None
        self.validated = False

    def validate_before_publication(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        current_state: CurrentPointerState,
        manifest: SecurityDenylistSnapshot,
    ) -> None:
        if (
            self.validated
            or envelope.artifact_kind is not PublicationArtifactKind.SECURITY_DENYLIST
            or manifest.scope_id != envelope.scope_id
            or repositories.scope_id != envelope.scope_id
        ):
            raise SecurityDenylistError(
                "security denylist publication gate input is invalid"
            )
        if len(manifest.revocations) > (
            self.launch_settings.security_denylist_revocation_limit
        ):
            raise SecurityDenylistError(
                "security denylist publication revocations exceed their bound"
            )
        if manifest.generation >= (
            self.launch_settings.security_denylist_lineage_limit
        ):
            raise SecurityDenylistError(
                "security denylist publication exceeds the finite lineage horizon"
            )
        current_pointer = current_state.pointer
        if current_pointer is None:
            if manifest.generation != 0:
                raise SecurityDenylistError(
                    "security denylist first publication must be generation zero"
                )
        else:
            current = self.provider.resolve_current(envelope.scope_id)
            if (
                current.authority_commit_sha != current_state.head_commit_sha
                or current.snapshot.snapshot_id
                != current_pointer.publication_record.artifact_id
                or current.publication_id
                != current_pointer.publication_record.publication_id
                or current.pointer_digest
                != tree_or_blob_digest(current_pointer.to_json_bytes())
            ):
                raise SecurityDenylistError(
                    "security denylist current changed during publication preflight"
                )
            if len(current.snapshot.revocations) > (
                self.launch_settings.security_denylist_revocation_limit
            ):
                raise SecurityDenylistError(
                    "security denylist current revocations exceed their bound"
                )
            if manifest.snapshot_id != current.snapshot.snapshot_id:
                _validate_snapshot_successor(manifest, current.snapshot)
            self.expected_current_snapshot_id = current.snapshot.snapshot_id
        self.expected_current_pointer = current_pointer
        self.validated = True

    def revalidate_before_activation(
        self,
        *,
        envelope: PublicationEnvelope,
        repositories: ScopeRepositorySettings,
        pointer: CurrentArtifactPointer,
        manifest: SecurityDenylistSnapshot,
    ) -> None:
        if (
            not self.validated
            or envelope.artifact_kind is not PublicationArtifactKind.SECURITY_DENYLIST
            or manifest.snapshot_id != envelope.artifact_id
            or repositories.scope_id != envelope.scope_id
        ):
            raise SecurityDenylistError(
                "security denylist activation was not preauthorized"
            )
        current_state = self.resolver.read_current_pointer_state(
            envelope.scope_id,
            PublicationArtifactKind.SECURITY_DENYLIST,
            allow_missing=True,
        )
        source_commit_sha = pointer.publication_record.commit_sha
        if (
            current_state.head_commit_sha != source_commit_sha
            or current_state.pointer != self.expected_current_pointer
        ):
            raise SecurityDenylistError(
                "security denylist current changed before activation"
            )
        if self.expected_current_pointer is not None:
            current = self.provider.resolve_current(envelope.scope_id)
            if (
                current.authority_commit_sha != source_commit_sha
                or current.snapshot.snapshot_id != self.expected_current_snapshot_id
                or current.pointer_digest
                != tree_or_blob_digest(self.expected_current_pointer.to_json_bytes())
            ):
                raise SecurityDenylistError(
                    "security denylist predecessor failed final authentication"
                )


class SecurityDenylistPublisher:
    """Publish only an adjacent cumulative denylist through the generic transport."""

    def __init__(
        self,
        publisher: AutonomousGitHubPublisher,
        resolver: GitHubArtifactResolver,
        provider: GitHubSecurityDenylistSnapshotProvider,
        launch_settings: LaunchSettings,
    ) -> None:
        if (
            publisher.resolver is not resolver
            or provider.resolver is not resolver
            or publisher.package_validator is not provider.materializer
        ):
            raise SecurityDenylistError(
                "security denylist publisher must share one resolver authority"
            )
        self.publisher = publisher
        self.resolver = resolver
        self.provider = provider
        self.launch_settings = launch_settings
        self.publisher._bind_activation_verifier(
            PublicationArtifactKind.SECURITY_DENYLIST, SecurityDenylistPublicationGate
        )

    def publish(self, envelope: PublicationEnvelope) -> PublicationTelemetry:
        return self.publisher.publish(
            envelope,
            activation_authorization=self.publisher._authorize_publication(
                envelope,
                SecurityDenylistPublicationGate(
                    self.resolver,
                    self.provider,
                    self.launch_settings,
                ),
            ),
        )


@dataclass(frozen=True)
class SecurityDenylistCheckpoint(StrictContract):
    checkpoint_id: str
    scope_id: str
    scope_contract_id: str
    scope_repository_binding_hash: str
    repository_full_name: str
    repository_node_id: str
    snapshot_id: str
    generation: int
    publication_id: str
    pointer_digest: str
    authority_commit_sha: str

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-checkpoint"
    IDENTITY_FIELD: ClassVar[str] = "checkpoint_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "security checkpoint scope_id")
        require_content_id(
            self.scope_contract_id,
            "security checkpoint scope_contract_id",
        )
        if _DIGEST_PATTERN.fullmatch(self.scope_repository_binding_hash) is None:
            raise SecurityDenylistError(
                "security checkpoint repository binding is invalid"
            )
        if _REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise SecurityDenylistError("security checkpoint repository is invalid")
        require_identifier(
            self.repository_node_id,
            "security checkpoint repository_node_id",
        )
        for value, namespace, name in (
            (self.snapshot_id, "security-denylist-snapshot", "snapshot_id"),
            (self.publication_id, "github-publication", "publication_id"),
        ):
            require_content_id(value, f"security checkpoint {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise SecurityDenylistError(
                    f"security checkpoint {name} uses the wrong namespace"
                )
        if type(self.generation) is not int or self.generation < 0:
            raise SecurityDenylistError(
                "security checkpoint generation must be non-negative"
            )
        if _DIGEST_PATTERN.fullmatch(self.pointer_digest) is None:
            raise SecurityDenylistError("security checkpoint pointer is invalid")
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise SecurityDenylistError(
                "security checkpoint authority commit is invalid"
            )


class SecurityDenylistCheckpointStore:
    """Persist one private crash-atomic anti-rollback floor per scope."""

    def __init__(
        self,
        root: Path,
        trusted_root: Path,
        maximum_checkpoint_size_bytes: int,
    ) -> None:
        if not isinstance(trusted_root, Path) or not trusted_root.is_absolute():
            raise SecurityDenylistError(
                "security checkpoint trusted root must be resolved"
            )
        trusted_root_metadata = os.stat(trusted_root, follow_symlinks=False)
        if (
            trusted_root.resolve() != trusted_root
            or not stat.S_ISDIR(trusted_root_metadata.st_mode)
            or stat.S_IMODE(trusted_root_metadata.st_mode) != 0o700
            or trusted_root_metadata.st_uid != os.geteuid()
        ):
            raise SecurityDenylistError(
                "security checkpoint trusted root must be owner-private"
            )
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != trusted_root
        ):
            raise SecurityDenylistError(
                "security checkpoint root must be a direct normalized child"
            )
        if (
            type(maximum_checkpoint_size_bytes) is not int
            or maximum_checkpoint_size_bytes <= 0
        ):
            raise SecurityDenylistError(
                "security checkpoint size bound must be positive"
            )
        self.root = root
        self.trusted_root = trusted_root
        self.maximum_checkpoint_size_bytes = maximum_checkpoint_size_bytes
        self.lock_root = root / "locks"
        self.checkpoint_root = root / "checkpoints"
        self.staging_root = root / "staging"
        self.initialization_lock_path = trusted_root / f".{root.name}.lock"
        with _CheckpointLock(self.initialization_lock_path):
            self._ensure_private_directory(self.root, self.trusted_root)
            self._ensure_private_directory(self.lock_root, self.root)
            self._ensure_private_directory(self.checkpoint_root, self.root)
            self._ensure_private_directory(self.staging_root, self.root)

    def checkpoint(self, scope_id: str) -> SecurityDenylistCheckpoint | None:
        require_identifier(scope_id, "security checkpoint scope_id")
        with _CheckpointLock(self._lock_path(scope_id)):
            self._clean_staging(scope_id)
            return self._read_checkpoint(scope_id)

    def accept(
        self,
        observation: AuthenticatedSecurityDenylistSnapshot,
        lineage: tuple[AuthenticatedSecurityDenylistSnapshot, ...],
    ) -> SecurityDenylistCheckpoint:
        if not lineage or lineage[0] != observation:
            raise SecurityDenylistError(
                "security checkpoint lineage omits its current observation"
            )
        snapshot = observation.snapshot
        candidate = SecurityDenylistCheckpoint.mint(
            scope_id=snapshot.scope_id,
            scope_contract_id=snapshot.scope_contract_id,
            scope_repository_binding_hash=snapshot.scope_repository_binding_hash,
            repository_full_name=observation.repository_full_name,
            repository_node_id=observation.repository_node_id,
            snapshot_id=snapshot.snapshot_id,
            generation=snapshot.generation,
            publication_id=observation.publication_id,
            pointer_digest=observation.pointer_digest,
            authority_commit_sha=observation.authority_commit_sha,
        )
        scope_id = snapshot.scope_id
        with _CheckpointLock(self._lock_path(scope_id)):
            self._clean_staging(scope_id)
            current = self._read_checkpoint(scope_id)
            if current is not None:
                self._require_same_authority(current, candidate)
                if candidate.generation < current.generation:
                    raise SecurityDenylistError(
                        "security denylist current observation rolled back"
                    )
                if candidate.generation == current.generation:
                    if (
                        candidate.snapshot_id != current.snapshot_id
                        or candidate.publication_id != current.publication_id
                        or candidate.pointer_digest != current.pointer_digest
                    ):
                        raise SecurityDenylistError(
                            "security denylist has an equal-generation fork"
                        )
                    return current
                lineage_ids = {item.snapshot.snapshot_id for item in lineage}
                if current.snapshot_id not in lineage_ids:
                    raise SecurityDenylistError(
                        "security denylist advance does not preserve its local floor"
                    )
            elif lineage[-1].snapshot.generation != 0:
                raise SecurityDenylistError(
                    "security denylist initial observation omits generation zero"
                )
            self._write_checkpoint(scope_id, candidate)
            return candidate

    @staticmethod
    def _require_same_authority(
        current: SecurityDenylistCheckpoint,
        candidate: SecurityDenylistCheckpoint,
    ) -> None:
        if (
            current.scope_id != candidate.scope_id
            or current.scope_contract_id != candidate.scope_contract_id
            or current.scope_repository_binding_hash
            != candidate.scope_repository_binding_hash
            or current.repository_full_name != candidate.repository_full_name
            or current.repository_node_id != candidate.repository_node_id
        ):
            raise SecurityDenylistError(
                "security denylist authority changed across checkpoints"
            )

    def _read_checkpoint(
        self,
        scope_id: str,
    ) -> SecurityDenylistCheckpoint | None:
        path = self._checkpoint_path(scope_id)
        if not os.path.lexists(path):
            return None
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o400
            ):
                raise SecurityDenylistError(
                    "security checkpoint must be a private independent file"
                )
            payload = handle.read(self.maximum_checkpoint_size_bytes + 1)
        if len(payload) > self.maximum_checkpoint_size_bytes:
            raise SecurityDenylistError(
                "security checkpoint exceeds its configured bound"
            )
        checkpoint = SecurityDenylistCheckpoint.from_json_bytes(payload)
        if payload != checkpoint.to_json_bytes() or checkpoint.scope_id != scope_id:
            raise SecurityDenylistError(
                "security checkpoint bytes or scope are invalid"
            )
        return checkpoint

    def _write_checkpoint(
        self,
        scope_id: str,
        checkpoint: SecurityDenylistCheckpoint,
    ) -> None:
        payload = checkpoint.to_json_bytes()
        if len(payload) > self.maximum_checkpoint_size_bytes:
            raise SecurityDenylistError(
                "security checkpoint exceeds its configured bound"
            )
        temporary_path = self.staging_root / (
            f"{self._scope_digest(scope_id)}-{secrets.token_hex(16)}.tmp"
        )
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        os.replace(temporary_path, self._checkpoint_path(scope_id))
        self._fsync_directory(self.checkpoint_root)
        self._fsync_directory(self.staging_root)

    def _clean_staging(self, scope_id: str) -> None:
        scope_digest = self._scope_digest(scope_id)
        with os.scandir(self.staging_root) as iterator:
            entries = tuple(iterator)
        for entry in entries:
            match = _STAGING_PATTERN.fullmatch(entry.name)
            if match is None:
                raise SecurityDenylistError(
                    "security checkpoint staging entry is unexpected"
                )
            if match.group("scope") != scope_digest:
                continue
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
            ):
                raise SecurityDenylistError(
                    "security checkpoint staging entry is unsafe"
                )
            os.unlink(entry.path)
        if entries:
            self._fsync_directory(self.staging_root)

    def _lock_path(self, scope_id: str) -> Path:
        return self.lock_root / f"{self._scope_digest(scope_id)}.lock"

    def _checkpoint_path(self, scope_id: str) -> Path:
        return self.checkpoint_root / f"{self._scope_digest(scope_id)}.json"

    @staticmethod
    def _scope_digest(scope_id: str) -> str:
        require_identifier(scope_id, "security checkpoint scope_id")
        return hashlib.sha256(scope_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _ensure_private_directory(path: Path, parent: Path) -> None:
        if not os.path.lexists(path):
            os.mkdir(path, mode=0o700)
            SecurityDenylistCheckpointStore._fsync_directory(parent)
        metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SecurityDenylistError(
                "security checkpoint directory must be private and real"
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        with ExitStack() as descriptors:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, descriptor)
            os.fsync(descriptor)


class AuthenticatedSecurityDenylistAuthority:
    """Live-resolve one scope denylist and advance a durable monotonic floor."""

    def __init__(
        self,
        scopes: ScopeRegistrySettings,
        launch_settings: LaunchSettings,
        provider: SecurityDenylistSnapshotProvider,
        checkpoint_store: SecurityDenylistCheckpointStore,
    ) -> None:
        self.scopes = scopes
        self.launch_settings = launch_settings
        self.provider = provider
        self.checkpoint_store = checkpoint_store

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation:
        require_identifier(scope_id, "security denylist scope_id")
        require_content_id(scope_contract_id, "security denylist scope_contract_id")
        if not isinstance(checked_subject_ids, tuple) or not checked_subject_ids:
            raise SecurityDenylistError(
                "security denylist checked subjects must be a non-empty tuple"
            )
        if len(checked_subject_ids) > (
            self.launch_settings.security_denylist_checked_subject_limit
        ):
            raise SecurityDenylistError(
                "security denylist checked-subject count exceeds its bound"
            )
        checked_subject_size = 2
        checked_subject_size_bound = (
            self.launch_settings.security_denylist_checked_subject_size_bytes
        )
        for position, subject_id in enumerate(checked_subject_ids):
            if not isinstance(subject_id, str):
                raise SecurityDenylistError(
                    "security denylist checked subject must be text"
                )
            checked_subject_size += len(subject_id) + 2 + (position > 0)
            if checked_subject_size > checked_subject_size_bound:
                raise SecurityDenylistError(
                    "security denylist checked-subject bytes exceed their bound"
                )
            require_content_id(subject_id, "security denylist checked subject")
        if checked_subject_ids != tuple(sorted(set(checked_subject_ids))):
            raise SecurityDenylistError(
                "security denylist checked subjects must be sorted and unique"
            )
        repositories = self.scopes.resolve(scope_id)
        current = self.provider.resolve_current(scope_id)
        snapshot = current.snapshot
        if (
            snapshot.scope_id != scope_id
            or snapshot.scope_contract_id != scope_contract_id
            or snapshot.scope_repository_binding_hash
            != repositories.binding_fingerprint
            or current.repository_full_name != repositories.security_repository
        ):
            raise SecurityDenylistError(
                "security denylist current snapshot uses another scope authority"
            )
        if len(snapshot.revocations) > (
            self.launch_settings.security_denylist_revocation_limit
        ):
            raise SecurityDenylistError(
                "security denylist revocation count exceeds its configured bound"
            )
        if snapshot.generation >= (
            self.launch_settings.security_denylist_lineage_limit
        ):
            raise SecurityDenylistError(
                "security denylist current exceeds the finite lineage horizon"
            )
        floor = self.checkpoint_store.checkpoint(scope_id)
        lineage = self._lineage_to_floor(current, floor)
        self.checkpoint_store.accept(current, lineage)
        denied_subjects = set(snapshot.denied_subject_ids)
        denied_subject_ids = tuple(
            subject_id
            for subject_id in checked_subject_ids
            if subject_id in denied_subjects
        )
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=repositories.binding_fingerprint,
            snapshot_id=snapshot.snapshot_id,
            generation=snapshot.generation,
            publication_id=current.publication_id,
            repository_full_name=current.repository_full_name,
            repository_node_id=current.repository_node_id,
            pointer_digest=current.pointer_digest,
            authority_commit_sha=current.authority_commit_sha,
            release_attestation_ref=current.release_attestation_ref,
            checked_subject_ids=checked_subject_ids,
            denied_subject_ids=denied_subject_ids,
        )

    def _lineage_to_floor(
        self,
        current: AuthenticatedSecurityDenylistSnapshot,
        floor: SecurityDenylistCheckpoint | None,
    ) -> tuple[AuthenticatedSecurityDenylistSnapshot, ...]:
        snapshot = current.snapshot
        if floor is not None:
            SecurityDenylistCheckpointStore._require_same_authority(
                floor,
                SecurityDenylistCheckpoint.mint(
                    scope_id=snapshot.scope_id,
                    scope_contract_id=snapshot.scope_contract_id,
                    scope_repository_binding_hash=(
                        snapshot.scope_repository_binding_hash
                    ),
                    repository_full_name=current.repository_full_name,
                    repository_node_id=current.repository_node_id,
                    snapshot_id=snapshot.snapshot_id,
                    generation=snapshot.generation,
                    publication_id=current.publication_id,
                    pointer_digest=current.pointer_digest,
                    authority_commit_sha=current.authority_commit_sha,
                ),
            )
            if snapshot.generation < floor.generation:
                raise SecurityDenylistError(
                    "security denylist remote generation is below the local floor"
                )
            if snapshot.generation == floor.generation:
                if (
                    snapshot.snapshot_id != floor.snapshot_id
                    or current.publication_id != floor.publication_id
                    or current.pointer_digest != floor.pointer_digest
                ):
                    raise SecurityDenylistError(
                        "security denylist remote has an equal-generation fork"
                    )
                return (current,)
        lineage = [current]
        while snapshot.generation > 0 and (
            floor is None or snapshot.snapshot_id != floor.snapshot_id
        ):
            if len(lineage) >= self.launch_settings.security_denylist_lineage_limit:
                raise SecurityDenylistError(
                    "security denylist lineage exceeds its configured bound"
                )
            predecessor_id = snapshot.predecessor_snapshot_id
            if predecessor_id is None:
                raise SecurityDenylistError(
                    "security denylist successor omits its predecessor"
                )
            predecessor = self.provider.resolve_exact(snapshot.scope_id, predecessor_id)
            self._validate_successor(lineage[-1], predecessor)
            if len(predecessor.snapshot.revocations) > (
                self.launch_settings.security_denylist_revocation_limit
            ):
                raise SecurityDenylistError(
                    "security denylist historical revocations exceed their bound"
                )
            lineage.append(predecessor)
            snapshot = predecessor.snapshot
        if floor is not None and snapshot.snapshot_id != floor.snapshot_id:
            raise SecurityDenylistError(
                "security denylist lineage does not reach the local floor"
            )
        if floor is None and snapshot.generation != 0:
            raise SecurityDenylistError(
                "security denylist lineage does not reach generation zero"
            )
        return tuple(lineage)

    @staticmethod
    def _validate_successor(
        successor: AuthenticatedSecurityDenylistSnapshot,
        predecessor: AuthenticatedSecurityDenylistSnapshot,
    ) -> None:
        child = successor.snapshot
        parent = predecessor.snapshot
        if (
            successor.repository_full_name != predecessor.repository_full_name
            or successor.repository_node_id != predecessor.repository_node_id
        ):
            raise SecurityDenylistError(
                "security denylist predecessor authority is discontinuous"
            )
        _validate_snapshot_successor(child, parent)


class _CheckpointLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        self.handle = os.fdopen(descriptor, "r+b")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            self.handle.close()
            self.handle = None
            raise SecurityDenylistError(
                "security checkpoint lock must be a private independent file"
            )
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exception_type, exception, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        return False
