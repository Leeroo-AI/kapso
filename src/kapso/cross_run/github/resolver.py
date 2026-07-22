"""Strict immutable GitHub artifact resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
    ScopeRepositorySettings,
    StrictContract,
)
from kapso.cross_run.git_refs import (
    git_object_sha,
    git_tree_shas,
    require_git_ref_name,
)
from kapso.cross_run.github.command import (
    BoundedJsonResponse,
    GitHubCommandClient,
    validate_release_attestation,
)
from kapso.cross_run.settings import GitHubSettings, ScopeRegistrySettings

_DEFAULT_BRANCH_HEAD_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef {
      name
      target { ... on Commit { oid } }
    }
  }
}
"""

_POINTER_AT_COMMIT_QUERY = """
query($owner: String!, $name: String!, $expression: String!) {
  repository(owner: $owner, name: $name) {
    object(expression: $expression) {
      ... on Blob { byteSize isBinary isTruncated text }
    }
  }
}
"""

_RELEASE_BY_TAG_QUERY = """
query($owner: String!, $name: String!, $tag: String!) {
  repository(owner: $owner, name: $name) {
    release(tagName: $tag) { databaseId }
  }
}
"""

_ARTIFACT_REF_QUERY = """
query($owner: String!, $name: String!, $qualifiedName: String!) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: $qualifiedName) {
      target { ... on Commit { oid } }
    }
  }
}
"""

ARTIFACT_POINTER_FILENAME = "PUBLICATION.json"
ARTIFACT_PUBLICATION_INTENT_FILENAME = "PUBLICATION_INTENT.json"


class GitHubResolutionError(RuntimeError):
    """Remote GitHub state is absent, malformed, or incompatible."""


@dataclass(frozen=True)
class PublicationAssetIntent(StrictContract):
    name: str
    media_type: str
    size: int
    sha256: str

    def _validate(self) -> None:
        if PurePosixPath(self.name).name != self.name or not self.name:
            raise GitHubResolutionError("publication intent asset name is invalid")
        if not self.media_type:
            raise GitHubResolutionError("publication intent media type is required")
        if self.size < 1:
            raise GitHubResolutionError(
                "publication intent asset size must be positive"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256):
            raise GitHubResolutionError("publication intent asset digest is invalid")


@dataclass(frozen=True)
class PublicationSourceFile(StrictContract):
    relative_path: str
    mode: str
    size: int
    sha256: str
    git_blob_sha: str

    def _validate(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
        ):
            raise GitHubResolutionError("publication source path is invalid")
        if self.mode not in {"100644", "100755"}:
            raise GitHubResolutionError("publication source mode is invalid")
        if type(self.size) is not int or self.size < 0:
            raise GitHubResolutionError("publication source size is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256):
            raise GitHubResolutionError("publication source digest is invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_blob_sha):
            raise GitHubResolutionError("publication source Git blob is invalid")


@dataclass(frozen=True)
class ArtifactPublicationIntent(StrictContract):
    scope_id: str
    artifact_kind: PublicationArtifactKind
    artifact_id: str
    repository_node_id: str
    repository_full_name: str
    expected_parent_sha: str
    source_commit_sha: str
    source_tree_digest: str
    source_git_tree_sha: str
    source_files: tuple[PublicationSourceFile, ...]
    preserved_current: PublicationSourceFile | None
    materialized_tree_digest: str
    manifest_relative_path: str
    manifest_digest: str
    tag: str
    assets: tuple[PublicationAssetIntent, ...]
    validation_closure_ids: tuple[str, ...]
    publisher_identity: str
    committed_at: str

    def _validate(self) -> None:
        require_identifier(self.scope_id, "scope_id")
        require_content_id(self.artifact_id, "artifact_id")
        if not self.repository_node_id or self.repository_full_name.count("/") != 1:
            raise GitHubResolutionError("publication intent repository is invalid")
        for name in ("expected_parent_sha", "source_commit_sha", "source_git_tree_sha"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                raise GitHubResolutionError(f"publication intent {name} is invalid")
        for name in (
            "source_tree_digest",
            "materialized_tree_digest",
            "manifest_digest",
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)):
                raise GitHubResolutionError(f"publication intent {name} is invalid")
        source_paths = tuple(source.relative_path for source in self.source_files)
        if (
            not source_paths
            or "CURRENT.json" in source_paths
            or source_paths != tuple(sorted(set(source_paths)))
        ):
            raise GitHubResolutionError(
                "publication intent source files must be non-empty, sorted, and unique"
            )
        expected_source_digest = source_tree_digest(
            {
                source.relative_path: (source.sha256, source.mode, source.size)
                for source in self.source_files
            }
        )
        if expected_source_digest != self.source_tree_digest:
            raise GitHubResolutionError(
                "publication intent source descriptor digest mismatch"
            )
        if self.preserved_current is not None:
            if self.preserved_current.relative_path != "CURRENT.json":
                raise GitHubResolutionError(
                    "publication intent preserved pointer path is invalid"
                )
        manifest_path = PurePosixPath(self.manifest_relative_path)
        if (
            manifest_path.is_absolute()
            or ".." in manifest_path.parts
            or manifest_path.as_posix() != self.manifest_relative_path
        ):
            raise GitHubResolutionError("publication intent manifest path is invalid")
        require_git_ref_name(
            f"refs/tags/{self.tag}",
            "publication intent tag",
            qualified=True,
            error_type=GitHubResolutionError,
        )
        if not self.assets or tuple(asset.name for asset in self.assets) != tuple(
            sorted({asset.name for asset in self.assets})
        ):
            raise GitHubResolutionError(
                "publication intent assets must be non-empty, sorted, and unique"
            )
        if not self.validation_closure_ids or self.validation_closure_ids != tuple(
            sorted(set(self.validation_closure_ids))
        ):
            raise GitHubResolutionError(
                "publication intent validation closure is invalid"
            )
        for reference in self.validation_closure_ids:
            require_content_id(reference, "validation_closure_ids")
        if not self.publisher_identity:
            raise GitHubResolutionError("publication intent publisher is required")
        normalize_utc_timestamp(self.committed_at, "committed_at")

    def binds(self, pointer: CurrentArtifactPointer) -> bool:
        record = pointer.publication_record
        return (
            pointer.publication_intent_digest == self.digest
            and self.scope_id == pointer.scope_id
            and self.artifact_kind is record.artifact_kind
            and self.artifact_id == record.artifact_id
            and self.repository_node_id == record.repository_node_id
            and self.repository_full_name == record.repository_full_name
            and self.source_commit_sha == record.commit_sha
            and self.source_tree_digest == pointer.source_tree_digest
            and self.source_git_tree_sha == pointer.source_git_tree_sha
            and self.materialized_tree_digest == pointer.materialized_tree_digest
            and self.manifest_relative_path == pointer.manifest_relative_path
            and self.manifest_digest == pointer.manifest_digest
            and self.tag == record.tag
            and tuple(
                (asset.name, asset.media_type, asset.size, asset.sha256)
                for asset in self.assets
            )
            == tuple(
                (asset.name, asset.media_type, asset.size, asset.sha256)
                for asset in record.assets
            )
            and self.validation_closure_ids == pointer.validation_closure_ids
            and self.publisher_identity == record.publisher_identity
        )

    @property
    def digest(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())


@dataclass(frozen=True)
class CurrentArtifactPointer(StrictContract):
    scope_id: str
    publication_record: GitHubPublicationRecord
    publication_intent_digest: str
    source_tree_digest: str
    source_git_tree_sha: str
    materialized_tree_digest: str
    manifest_relative_path: str
    manifest_digest: str
    validation_closure_ids: tuple[str, ...]

    def _validate(self) -> None:
        require_identifier(self.scope_id, "scope_id")
        for name in (
            "publication_intent_digest",
            "source_tree_digest",
            "materialized_tree_digest",
            "manifest_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value
            ):
                raise GitHubResolutionError(f"{name} must be a sha256 digest")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_git_tree_sha):
            raise GitHubResolutionError("source_git_tree_sha must be 40 lowercase hex")
        manifest_path = PurePosixPath(self.manifest_relative_path)
        if (
            manifest_path.is_absolute()
            or ".." in manifest_path.parts
            or manifest_path.as_posix() != self.manifest_relative_path
        ):
            raise GitHubResolutionError(
                "manifest_relative_path must be normalized and relative"
            )
        if self.validation_closure_ids != tuple(
            sorted(set(self.validation_closure_ids))
        ):
            raise GitHubResolutionError(
                "validation_closure_ids must be sorted and unique"
            )
        for reference in self.validation_closure_ids:
            require_content_id(reference, "validation_closure_ids")


@dataclass(frozen=True)
class RepositoryPolicyReport:
    repository_full_name: str
    repository_node_id: str
    private: bool
    default_branch: str
    authenticated_actor: str
    write_access: bool
    immutable_releases: bool


@dataclass(frozen=True)
class ResolvedGitHubArtifact:
    repositories: ScopeRepositorySettings
    pointer: CurrentArtifactPointer
    policy: RepositoryPolicyReport
    pointer_commit_sha: str


@dataclass(frozen=True)
class CurrentPointerState:
    pointer: CurrentArtifactPointer | None
    head_commit_sha: str


@dataclass(frozen=True)
class _ArtifactPointerState:
    pointer: CurrentArtifactPointer | None
    identity_commit_sha: str | None


def repository_for_artifact(
    repositories: ScopeRepositorySettings,
    artifact_kind: PublicationArtifactKind,
) -> str:
    if artifact_kind is PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
        return repositories.knowledge_repository
    if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE:
        return repositories.expert_repository
    if artifact_kind is PublicationArtifactKind.SECURITY_DENYLIST:
        return repositories.security_repository
    raise GitHubResolutionError(f"unsupported artifact kind: {artifact_kind}")


def tag_prefix_for_artifact(
    settings: GitHubSettings,
    artifact_kind: PublicationArtifactKind,
) -> str:
    if artifact_kind is PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
        return settings.knowledge_tag_prefix
    if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE:
        return settings.expert_tag_prefix
    if artifact_kind is PublicationArtifactKind.SECURITY_DENYLIST:
        return settings.security_denylist_tag_prefix
    raise GitHubResolutionError(f"unsupported artifact kind: {artifact_kind}")


def security_denylist_tag(settings: GitHubSettings, generation: int) -> str:
    if type(generation) is not int or generation < 0:
        raise GitHubResolutionError("security denylist generation must be non-negative")
    return f"{settings.security_denylist_tag_prefix}D{generation:06d}"


def artifact_identity_ref(
    artifact_kind: PublicationArtifactKind, artifact_id: str
) -> str:
    require_content_id(artifact_id, "artifact_id")
    digest = artifact_id.rsplit(":", 1)[1]
    return f"refs/kapso-artifacts/{artifact_kind.value}/{digest}"


def artifact_publication_intent_ref(
    artifact_kind: PublicationArtifactKind, artifact_id: str
) -> str:
    require_content_id(artifact_id, "artifact_id")
    digest = artifact_id.rsplit(":", 1)[1]
    return f"refs/kapso-publication-intents/{artifact_kind.value}/{digest}"


def release_attestation_reference(attestation: Any) -> str:
    result = _require_mapping(attestation, "release verification result")
    attestation_record = _require_mapping(
        result.get("attestation"), "release attestation"
    )
    bundle = _require_mapping(
        attestation_record.get("bundle"), "release attestation bundle"
    )
    digest = tree_or_blob_digest(canonical_json_bytes(bundle))
    return f"github-release-attestation:{digest}"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GitHubResolutionError(f"{name} must be an object")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GitHubResolutionError(f"{name} must be non-empty text")
    return value


def _require_graphql_data(value: Any, name: str) -> Mapping[str, Any]:
    response = _require_mapping(value, f"{name} response")
    if response.get("errors") not in (None, []):
        raise GitHubResolutionError(f"{name} returned GraphQL errors")
    return _require_mapping(response.get("data"), f"{name} data")


class GitHubArtifactResolver:
    """Resolve configured CURRENT pointers into verified immutable records."""

    def __init__(
        self,
        client: GitHubCommandClient,
        settings: GitHubSettings,
        scope_registry: ScopeRegistrySettings,
    ) -> None:
        self.client = client
        self.settings = settings
        self.scope_registry = scope_registry

    def repositories_for_scope(self, scope_id: str) -> ScopeRepositorySettings:
        """Return the sole canonical repository binding for a registered scope."""
        return self.scope_registry.resolve(scope_id)

    def diagnose_repository(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
    ) -> RepositoryPolicyReport:
        repositories = self.repositories_for_scope(scope_id)
        repository = repository_for_artifact(repositories, artifact_kind)
        metadata = _require_mapping(
            self.client.api_json("GET", f"repos/{repository}"),
            "repository metadata",
        )
        immutable = _require_mapping(
            self.client.api_json("GET", f"repos/{repository}/immutable-releases"),
            "immutable-release policy",
        )
        actor = _require_mapping(
            self.client.api_json("GET", "user"), "authenticated actor"
        )
        full_name = _require_text(metadata.get("full_name"), "repository full_name")
        node_id = _require_text(metadata.get("node_id"), "repository node_id")
        private = metadata.get("private")
        default_branch = _require_text(
            metadata.get("default_branch"), "repository default_branch"
        )
        permissions = _require_mapping(
            metadata.get("permissions"), "repository permissions"
        )
        write_access = permissions.get("push")
        immutable_enabled = immutable.get("enabled")
        actor_login = _require_text(actor.get("login"), "authenticated actor login")
        if full_name != repository:
            raise GitHubResolutionError("GitHub repository identity mismatch")
        if type(private) is not bool or not private:
            raise GitHubResolutionError("cross-run repository must be private")
        if default_branch != self.settings.default_branch:
            raise GitHubResolutionError("GitHub default branch mismatch")
        if type(write_access) is not bool or not write_access:
            raise GitHubResolutionError("authenticated actor lacks write access")
        if type(immutable_enabled) is not bool or not immutable_enabled:
            raise GitHubResolutionError("immutable releases are not enabled")
        if actor_login != self.settings.publisher_login:
            raise GitHubResolutionError(
                "authenticated actor is not configured publisher"
            )
        return RepositoryPolicyReport(
            repository_full_name=full_name,
            repository_node_id=node_id,
            private=private,
            default_branch=default_branch,
            authenticated_actor=actor_login,
            write_access=write_access,
            immutable_releases=immutable_enabled,
        )

    def read_current_pointer(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        *,
        allow_missing: bool = False,
    ) -> CurrentArtifactPointer | None:
        return self._read_current_pointer_state(
            scope_id,
            artifact_kind,
            allow_missing=allow_missing,
        ).pointer

    def read_current_pointer_state(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        *,
        allow_missing: bool = False,
    ) -> CurrentPointerState:
        """Read CURRENT and the branch commit containing it as one pinned state."""
        return self._read_current_pointer_state(
            scope_id,
            artifact_kind,
            allow_missing=allow_missing,
        )

    def find_release_id(self, repository: str, tag: str) -> int | None:
        owner, name = repository.split("/", 1)
        data = _require_graphql_data(
            self.client.graphql(
                _RELEASE_BY_TAG_QUERY,
                {"owner": owner, "name": name, "tag": tag},
            ),
            "release query",
        )
        repository_data = _require_mapping(
            data.get("repository"), "release query repository"
        )
        release = repository_data.get("release")
        if release is None:
            return None
        release_data = _require_mapping(release, "release query result")
        release_id = release_data.get("databaseId")
        if type(release_id) is not int or release_id < 1:
            raise GitHubResolutionError("release database ID is invalid")
        return release_id

    def read_artifact_pointer(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
    ) -> CurrentArtifactPointer | None:
        """Resolve the write-once global identity ref for an artifact, if present."""
        return self._read_artifact_pointer_state(
            scope_id, artifact_kind, artifact_id
        ).pointer

    def _read_artifact_pointer_state(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
    ) -> _ArtifactPointerState:
        repositories = self.repositories_for_scope(scope_id)
        repository = repository_for_artifact(repositories, artifact_kind)
        owner, name = repository.split("/", 1)
        data = _require_graphql_data(
            self.client.graphql(
                _ARTIFACT_REF_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "qualifiedName": artifact_identity_ref(artifact_kind, artifact_id),
                },
            ),
            "artifact identity ref query",
        )
        repository_data = _require_mapping(
            data.get("repository"), "artifact identity ref repository"
        )
        reference = repository_data.get("ref")
        if reference is None:
            return _ArtifactPointerState(pointer=None, identity_commit_sha=None)
        reference_data = _require_mapping(reference, "artifact identity ref")
        target = _require_mapping(
            reference_data.get("target"), "artifact identity ref target"
        )
        commit_sha = _require_text(target.get("oid"), "artifact identity commit")
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise GitHubResolutionError("artifact identity commit is invalid")
        payload = self._read_blob_at_commit(
            repository, commit_sha, ARTIFACT_POINTER_FILENAME
        )
        if payload is None:
            raise GitHubResolutionError("artifact publication pointer is missing")
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, Mapping):
            raise GitHubResolutionError(
                "artifact publication pointer must be an object"
            )
        pointer = CurrentArtifactPointer.from_dict(parsed)
        if payload != pointer.to_json_bytes():
            raise GitHubResolutionError("artifact publication pointer is not canonical")
        record = pointer.publication_record
        if (
            pointer.scope_id != scope_id
            or record.artifact_kind is not artifact_kind
            or record.artifact_id != artifact_id
            or record.repository_full_name != repository
        ):
            raise GitHubResolutionError("artifact identity ref target mismatch")
        return _ArtifactPointerState(
            pointer=pointer,
            identity_commit_sha=commit_sha,
        )

    def read_artifact_intent(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
    ) -> ArtifactPublicationIntent | None:
        """Resolve the pre-release write-once publication claim, if present."""
        repositories = self.repositories_for_scope(scope_id)
        repository = repository_for_artifact(repositories, artifact_kind)
        owner, name = repository.split("/", 1)
        data = _require_graphql_data(
            self.client.graphql(
                _ARTIFACT_REF_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "qualifiedName": artifact_publication_intent_ref(
                        artifact_kind, artifact_id
                    ),
                },
            ),
            "artifact publication intent ref query",
        )
        repository_data = _require_mapping(
            data.get("repository"), "artifact publication intent repository"
        )
        reference = repository_data.get("ref")
        if reference is None:
            return None
        reference_data = _require_mapping(reference, "artifact publication intent ref")
        target = _require_mapping(
            reference_data.get("target"), "artifact publication intent target"
        )
        commit_sha = _require_text(target.get("oid"), "publication intent commit")
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise GitHubResolutionError("publication intent commit is invalid")
        payload = self._read_blob_at_commit(
            repository,
            commit_sha,
            ARTIFACT_PUBLICATION_INTENT_FILENAME,
        )
        if payload is None:
            raise GitHubResolutionError("artifact publication intent is missing")
        intent = ArtifactPublicationIntent.from_json_bytes(payload)
        if payload != intent.to_json_bytes():
            raise GitHubResolutionError("artifact publication intent is not canonical")
        if (
            intent.scope_id != scope_id
            or intent.artifact_kind is not artifact_kind
            or intent.artifact_id != artifact_id
            or intent.repository_full_name != repository
        ):
            raise GitHubResolutionError("artifact publication intent target mismatch")
        return intent

    def _read_current_pointer_state(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        *,
        allow_missing: bool,
    ) -> CurrentPointerState:
        repositories = self.repositories_for_scope(scope_id)
        repository = repository_for_artifact(repositories, artifact_kind)
        owner, name = repository.split("/", 1)
        head_data = _require_graphql_data(
            self.client.graphql(
                _DEFAULT_BRANCH_HEAD_QUERY,
                {"owner": owner, "name": name},
            ),
            "default branch query",
        )
        repository_data = _require_mapping(
            head_data.get("repository"), "default branch query repository"
        )
        default_branch = _require_mapping(
            repository_data.get("defaultBranchRef"), "CURRENT default branch"
        )
        if default_branch.get("name") != self.settings.default_branch:
            raise GitHubResolutionError("CURRENT query default branch mismatch")
        target = _require_mapping(
            default_branch.get("target"), "CURRENT default branch target"
        )
        head_commit_sha = _require_text(
            target.get("oid"), "CURRENT default branch commit"
        )
        if not re.fullmatch(r"[0-9a-f]{40}", head_commit_sha):
            raise GitHubResolutionError("CURRENT default branch commit is invalid")
        pointer_payload = self._read_blob_at_commit(
            repository,
            head_commit_sha,
            "CURRENT.json",
            allow_missing=allow_missing,
        )
        if pointer_payload is None:
            return CurrentPointerState(pointer=None, head_commit_sha=head_commit_sha)
        parsed = parse_json_bytes(pointer_payload)
        if not isinstance(parsed, Mapping):
            raise GitHubResolutionError("CURRENT.json must contain an object")
        pointer = CurrentArtifactPointer.from_dict(parsed)
        if pointer_payload != pointer.to_json_bytes():
            raise GitHubResolutionError("CURRENT.json is not canonical")
        return CurrentPointerState(pointer=pointer, head_commit_sha=head_commit_sha)

    def resolve_current(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
    ) -> ResolvedGitHubArtifact:
        repositories = self.repositories_for_scope(scope_id)
        policy = self.diagnose_repository(scope_id, artifact_kind)
        state = self._read_current_pointer_state(
            scope_id, artifact_kind, allow_missing=False
        )
        pointer = state.pointer
        if pointer is None:
            raise GitHubResolutionError("CURRENT.json is missing")
        identity_pointer = self.read_artifact_pointer(
            scope_id,
            artifact_kind,
            pointer.publication_record.artifact_id,
        )
        if identity_pointer is None or identity_pointer != pointer:
            raise GitHubResolutionError(
                "CURRENT pointer has no matching write-once artifact identity"
            )
        intent = self.read_artifact_intent(
            scope_id,
            artifact_kind,
            pointer.publication_record.artifact_id,
        )
        if intent is None or not intent.binds(pointer):
            raise GitHubResolutionError(
                "CURRENT pointer has no matching pre-release publication intent"
            )
        self.verify_pointer(scope_id, artifact_kind, policy, pointer, intent)
        return ResolvedGitHubArtifact(
            repositories=repositories,
            pointer=pointer,
            policy=policy,
            pointer_commit_sha=state.head_commit_sha,
        )

    def resolve_artifact(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
    ) -> ResolvedGitHubArtifact:
        """Resolve one immutable artifact identity without following CURRENT."""
        repositories = self.repositories_for_scope(scope_id)
        policy = self.diagnose_repository(scope_id, artifact_kind)
        state = self._read_artifact_pointer_state(scope_id, artifact_kind, artifact_id)
        if state.pointer is None or state.identity_commit_sha is None:
            raise GitHubResolutionError("artifact publication identity is missing")
        intent = self.read_artifact_intent(scope_id, artifact_kind, artifact_id)
        if intent is None or not intent.binds(state.pointer):
            raise GitHubResolutionError(
                "artifact identity has no matching pre-release publication intent"
            )
        self.verify_pointer(scope_id, artifact_kind, policy, state.pointer, intent)
        return ResolvedGitHubArtifact(
            repositories=repositories,
            pointer=state.pointer,
            policy=policy,
            pointer_commit_sha=state.identity_commit_sha,
        )

    def verify_pointer(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        policy: RepositoryPolicyReport,
        pointer: CurrentArtifactPointer,
        intent: ArtifactPublicationIntent,
    ) -> None:
        repositories = self.repositories_for_scope(scope_id)
        record = pointer.publication_record
        repository = repository_for_artifact(repositories, artifact_kind)
        if pointer.scope_id != repositories.scope_id:
            raise GitHubResolutionError("CURRENT pointer uses another scope")
        if record.artifact_kind is not artifact_kind:
            raise GitHubResolutionError("CURRENT pointer uses another artifact kind")
        if record.repository_full_name != repository:
            raise GitHubResolutionError("publication repository mismatch")
        if record.repository_node_id != policy.repository_node_id:
            raise GitHubResolutionError("publication repository node mismatch")
        if record.publisher_identity != policy.authenticated_actor:
            raise GitHubResolutionError("publication actor is not authenticated actor")
        expected_prefix = tag_prefix_for_artifact(self.settings, artifact_kind)
        if not record.tag.startswith(expected_prefix):
            raise GitHubResolutionError("publication tag uses another artifact prefix")
        require_git_ref_name(
            f"refs/tags/{record.tag}",
            "publication tag",
            qualified=True,
            error_type=GitHubResolutionError,
        )
        if not record.immutable_release_id.isdigit():
            raise GitHubResolutionError("immutable release ID must be numeric")
        release = _require_mapping(
            self.client.api_json(
                "GET",
                f"repos/{repository}/releases/{record.immutable_release_id}",
            ),
            "release",
        )
        if str(release.get("id")) != record.immutable_release_id:
            raise GitHubResolutionError("release identity mismatch")
        if release.get("draft") is not False or release.get("immutable") is not True:
            raise GitHubResolutionError("release is not published and immutable")
        if release.get("tag_name") != record.tag:
            raise GitHubResolutionError("release tag mismatch")
        author = _require_mapping(release.get("author"), "release author")
        if author.get("login") != record.publisher_identity:
            raise GitHubResolutionError("release author mismatch")
        if release.get("published_at") != record.published_at:
            raise GitHubResolutionError("release publication timestamp mismatch")
        remote_assets = self._release_assets(release)
        if tuple(asset.to_dict() for asset in remote_assets) != tuple(
            asset.to_dict() for asset in record.assets
        ):
            raise GitHubResolutionError("release asset closure mismatch")
        self._verify_exact_tag_ref(repository, record.tag, record.commit_sha)
        if not intent.binds(pointer):
            raise GitHubResolutionError("publication intent does not bind pointer")
        self.verify_publication_intent_source(repository, intent)
        manifest_bytes = self._read_blob_at_commit(
            repository, record.commit_sha, pointer.manifest_relative_path
        )
        if manifest_bytes is None:
            raise GitHubResolutionError("source commit manifest is missing")
        if tree_or_blob_digest(manifest_bytes) != pointer.manifest_digest:
            raise GitHubResolutionError("source commit manifest digest mismatch")
        asset_digests = {asset.name: asset.sha256 for asset in remote_assets}
        attestation = validate_release_attestation(
            self.client.verify_release(
                repository,
                record.tag,
                record.commit_sha,
                asset_digests,
            ),
            repository=repository,
            tag=record.tag,
            commit_sha=record.commit_sha,
            asset_digests=asset_digests,
            error_type=GitHubResolutionError,
        )
        if release_attestation_reference(attestation) != record.release_attestation_ref:
            raise GitHubResolutionError("release attestation mismatch")

    def verify_publication_intent_source(
        self,
        repository: str,
        intent: ArtifactPublicationIntent,
    ) -> None:
        """Recompute the complete remote Git source closure bound by an intent."""
        if repository != intent.repository_full_name:
            raise GitHubResolutionError("publication intent repository mismatch")
        source_commit = _require_mapping(
            self.client.api_json(
                "GET", f"repos/{repository}/git/commits/{intent.source_commit_sha}"
            ),
            "publication intent source commit",
        )
        source_tree = _require_mapping(
            source_commit.get("tree"), "publication intent source tree"
        )
        if source_tree.get("sha") != intent.source_git_tree_sha:
            raise GitHubResolutionError("publication intent source tree mismatch")
        self._verify_source_commit(repository, source_commit, intent)

    def _verify_source_commit(
        self,
        repository: str,
        source_commit: Mapping[str, Any],
        intent: ArtifactPublicationIntent,
    ) -> None:
        parents = source_commit.get("parents")
        if not isinstance(parents, list) or len(parents) != 1:
            raise GitHubResolutionError("source commit parent closure is invalid")
        parent = _require_mapping(parents[0], "source commit parent")
        if parent.get("sha") != intent.expected_parent_sha:
            raise GitHubResolutionError("source commit parent mismatch")
        blobs, directories = self._read_source_tree(
            repository,
            intent.source_git_tree_sha,
        )
        expected_files = {
            source.relative_path: source for source in intent.source_files
        }
        if intent.preserved_current is not None:
            expected_files["CURRENT.json"] = intent.preserved_current
        if set(blobs) != set(expected_files):
            raise GitHubResolutionError("source commit file closure mismatch")
        expected_directories: set[str] = set()
        for relative_path in expected_files:
            for parent_path in PurePosixPath(relative_path).parents:
                if parent_path.as_posix() != ".":
                    expected_directories.add(parent_path.as_posix())
        if set(directories) != expected_directories:
            raise GitHubResolutionError("source commit directory closure mismatch")
        total_source_bytes = 0
        git_files: dict[str, tuple[str, str]] = {}
        for relative_path, source in expected_files.items():
            entry = blobs[relative_path]
            if (
                entry.get("mode") != source.mode
                or entry.get("sha") != source.git_blob_sha
                or entry.get("size") != source.size
            ):
                raise GitHubResolutionError("source commit descriptor mismatch")
            total_source_bytes += source.size
            if total_source_bytes > self.settings.source_tree_size_bytes:
                raise GitHubResolutionError(
                    "source commit exceeds configured size limit"
                )
            payload = self.client.read_git_blob(
                repository,
                source.git_blob_sha,
                max(source.size, 1),
            )
            if (
                len(payload) != source.size
                or tree_or_blob_digest(payload) != source.sha256
            ):
                raise GitHubResolutionError("source commit blob digest mismatch")
            payload.decode("utf-8")
            actual_blob_sha = git_object_sha("blob", payload)
            if actual_blob_sha != source.git_blob_sha:
                raise GitHubResolutionError("source commit Git blob identity mismatch")
            git_files[relative_path] = (actual_blob_sha, source.mode)
        tree_shas = git_tree_shas(git_files)
        if tree_shas[""] != intent.source_git_tree_sha:
            raise GitHubResolutionError("source commit Git tree identity mismatch")
        for relative_path, entry in directories.items():
            if entry.get("sha") != tree_shas[relative_path]:
                raise GitHubResolutionError("source commit subtree identity mismatch")

    def _read_source_tree(
        self,
        repository: str,
        root_tree_sha: str,
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
        """Traverse exact Git trees without GitHub's recursive-response truncation."""
        blobs: dict[str, Mapping[str, Any]] = {}
        directories: dict[str, Mapping[str, Any]] = {}
        pending = [("", root_tree_sha)]
        remaining_metadata_bytes = self.settings.git_tree_metadata_size_bytes
        observed_entries = 0
        while pending:
            if remaining_metadata_bytes <= 0:
                raise GitHubResolutionError(
                    "source tree metadata exceeds configured limit"
                )
            directory_path, tree_sha = pending.pop()
            bounded = self.client.api_json_bounded(
                "GET",
                f"repos/{repository}/git/trees/{tree_sha}",
                remaining_metadata_bytes,
            )
            if (
                not isinstance(bounded, BoundedJsonResponse)
                or type(bounded.size_bytes) is not int
                or bounded.size_bytes <= 0
                or bounded.size_bytes > remaining_metadata_bytes
            ):
                raise GitHubResolutionError("source tree metadata response is invalid")
            remaining_metadata_bytes -= bounded.size_bytes
            tree_response = _require_mapping(
                bounded.value,
                "source tree",
            )
            if (
                tree_response.get("sha") != tree_sha
                or tree_response.get("truncated") is not False
            ):
                raise GitHubResolutionError("source tree response is incomplete")
            entries = tree_response.get("tree")
            if not isinstance(entries, list):
                raise GitHubResolutionError("source tree entries are invalid")
            for entry_value in entries:
                observed_entries += 1
                if observed_entries > self.settings.source_entry_limit:
                    raise GitHubResolutionError(
                        "source tree exceeds configured entry limit"
                    )
                entry = _require_mapping(entry_value, "source tree entry")
                direct_name = entry.get("path")
                entry_type = entry.get("type")
                mode = entry.get("mode")
                entry_sha = entry.get("sha")
                if (
                    not isinstance(direct_name, str)
                    or not direct_name
                    or direct_name in {".", ".."}
                    or len(PurePosixPath(direct_name).parts) != 1
                    or PurePosixPath(direct_name).as_posix() != direct_name
                    or re.fullmatch(r"[0-9a-f]{40}", str(entry_sha)) is None
                ):
                    raise GitHubResolutionError("source tree entry is invalid")
                relative_path = (
                    f"{directory_path}/{direct_name}" if directory_path else direct_name
                )
                if relative_path in directories or relative_path in blobs:
                    raise GitHubResolutionError("source tree contains a duplicate path")
                if entry_type == "tree":
                    if mode != "040000":
                        raise GitHubResolutionError(
                            "source tree directory mode is invalid"
                        )
                    directories[relative_path] = entry
                    pending.append((relative_path, str(entry_sha)))
                elif entry_type == "blob" and mode in {"100644", "100755"}:
                    blobs[relative_path] = entry
                else:
                    raise GitHubResolutionError(
                        "source tree contains an unsupported entry"
                    )
        return blobs, directories

    def _verify_exact_tag_ref(
        self,
        repository: str,
        tag: str,
        expected_commit_sha: str,
    ) -> None:
        tag_reference = _require_mapping(
            self.client.api_json("GET", f"repos/{repository}/git/ref/tags/{tag}"),
            "release tag ref",
        )
        if tag_reference.get("ref") != f"refs/tags/{tag}":
            raise GitHubResolutionError("release tag ref identity mismatch")
        target = _require_mapping(tag_reference.get("object"), "release tag target")
        if target.get("type") != "commit" or target.get("sha") != expected_commit_sha:
            raise GitHubResolutionError(
                "release tag must directly target the source commit"
            )

    def _read_blob_at_commit(
        self,
        repository: str,
        commit_sha: str,
        relative_path: str,
        *,
        allow_missing: bool = False,
    ) -> bytes | None:
        owner, name = repository.split("/", 1)
        data = _require_graphql_data(
            self.client.graphql(
                _POINTER_AT_COMMIT_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "expression": f"{commit_sha}:{relative_path}",
                },
            ),
            "source blob query",
        )
        repository_data = _require_mapping(
            data.get("repository"), "source blob query repository"
        )
        blob_value = repository_data.get("object")
        if blob_value is None and allow_missing:
            return None
        blob = _require_mapping(blob_value, "source blob")
        byte_size = blob.get("byteSize")
        if type(byte_size) is not int or byte_size < 0:
            raise GitHubResolutionError("source blob byte size is invalid")
        if byte_size > self.settings.control_blob_size_bytes:
            raise GitHubResolutionError("source blob exceeds configured control bound")
        if blob.get("isBinary") is not False:
            raise GitHubResolutionError("source blob is not text")
        if blob.get("isTruncated") is not False:
            raise GitHubResolutionError("source blob text is truncated")
        payload = _require_text(blob.get("text"), "source blob text").encode("utf-8")
        if len(payload) != byte_size:
            raise GitHubResolutionError("source blob text byte size mismatch")
        return payload

    def _release_assets(
        self, release: Mapping[str, Any]
    ) -> tuple[GitHubReleaseAsset, ...]:
        assets = release.get("assets")
        if not isinstance(assets, list) or not assets:
            raise GitHubResolutionError("release asset list is missing")
        if len(assets) > self.settings.release_asset_count_limit:
            raise GitHubResolutionError("release asset count exceeds configured limit")
        parsed: list[GitHubReleaseAsset] = []
        total_size = 0
        for asset in assets:
            value = _require_mapping(asset, "release asset")
            if value.get("state") != "uploaded":
                raise GitHubResolutionError("release asset is not fully uploaded")
            asset_id = value.get("id")
            size = value.get("size")
            if type(asset_id) is not int or asset_id < 1:
                raise GitHubResolutionError("release asset ID must be positive")
            if type(size) is not int or size < 0:
                raise GitHubResolutionError("release asset size must be non-negative")
            if size > self.settings.release_asset_size_bytes:
                raise GitHubResolutionError(
                    "release asset exceeds configured size limit"
                )
            total_size += size
            if total_size > self.settings.materialized_asset_size_bytes:
                raise GitHubResolutionError(
                    "release asset closure exceeds configured size limit"
                )
            name = _require_text(value.get("name"), "release asset name")
            if PurePosixPath(name).name != name:
                raise GitHubResolutionError("release asset name must be a basename")
            parsed.append(
                GitHubReleaseAsset(
                    asset_id=str(asset_id),
                    name=name,
                    media_type=_require_text(
                        value.get("content_type"), "release asset content type"
                    ),
                    size=size,
                    sha256=_require_text(value.get("digest"), "release asset digest"),
                )
            )
        return tuple(sorted(parsed, key=lambda asset: asset.name))
