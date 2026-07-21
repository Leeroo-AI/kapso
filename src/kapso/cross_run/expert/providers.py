"""Verified external readers used by expert validation enrollment."""

from __future__ import annotations

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import PublicationArtifactKind
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
