"""Verified external readers used by expert validation enrollment."""

from __future__ import annotations

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.replay_authority import (
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation import ExpertValidationError
from kapso.cross_run.github.resolver import GitHubArtifactResolver


class GitHubExpertCurrentReleaseProvider:
    """Resolve one optional expert CURRENT through its immutable GitHub identity."""

    def __init__(self, resolver: GitHubArtifactResolver) -> None:
        self.resolver = resolver

    def current_release_id(self, scope_id: str) -> str | None:
        artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
        self.resolver.diagnose_repository(scope_id, artifact_kind)
        current = self.resolver.read_current_pointer_state(
            scope_id,
            artifact_kind,
            allow_missing=True,
        )
        if current.pointer is None:
            return None
        release_id = current.pointer.publication_record.artifact_id
        require_content_id(release_id, "current expert release ID")
        resolved = self.resolver.resolve_artifact(
            scope_id,
            artifact_kind,
            release_id,
        )
        if resolved.pointer != current.pointer:
            raise ExpertValidationError(
                "resolved expert release differs from the observed CURRENT pointer"
            )
        return release_id

    def current_release_observation(
        self,
        scope_id: str,
    ) -> SourceReplayCurrentReleaseObservation:
        artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
        resolved = self.resolver.resolve_current(scope_id, artifact_kind)
        pointer = resolved.pointer
        publication = pointer.publication_record
        if (
            pointer.scope_id != scope_id
            or publication.artifact_kind is not artifact_kind
        ):
            raise ExpertValidationError(
                "resolved current expert release has another scope or artifact kind"
            )
        return SourceReplayCurrentReleaseObservation.mint(
            scope_id=scope_id,
            release_id=publication.artifact_id,
            publication_id=publication.publication_id,
            repository_full_name=resolved.policy.repository_full_name,
            repository_node_id=resolved.policy.repository_node_id,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
            current_pointer_commit_sha=resolved.pointer_commit_sha,
            validation_closure_ids=pointer.validation_closure_ids,
        )
