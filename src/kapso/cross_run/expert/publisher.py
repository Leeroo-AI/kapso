"""Domain authority for crash-safe expert release publication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.release import ExpertReleaseAssembler
from kapso.cross_run.expert.release_contracts import (
    ExpertReleasePublicationStaleResolution,
)
from kapso.cross_run.github.publisher import AutonomousGitHubPublisher
from kapso.cross_run.github.resolver import GitHubArtifactResolver

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertReleasePublicationStalePermit,
        ExpertValidationStore,
    )


class ExpertReleasePublicationError(ValueError):
    """Expert publication authority is missing, stale, or contradictory."""


class ExpertReleasePublisher:
    """Own the exact local and GitHub authorities for expert publication."""

    def __init__(
        self,
        *,
        assembler: ExpertReleaseAssembler,
        validation_store: ExpertValidationStore,
        github_publisher: AutonomousGitHubPublisher,
        resolver: GitHubArtifactResolver,
        current_release_authority: GitHubExpertCurrentReleaseProvider,
    ) -> None:
        if (
            type(assembler) is not ExpertReleaseAssembler
            or assembler.validation_store is not validation_store
            or type(github_publisher) is not AutonomousGitHubPublisher
            or type(resolver) is not GitHubArtifactResolver
            or github_publisher.resolver is not resolver
            or github_publisher.client is not resolver.client
            or github_publisher.settings != assembler.github_settings
            or resolver.settings != assembler.github_settings
            or type(current_release_authority) is not GitHubExpertCurrentReleaseProvider
            or current_release_authority.resolver is not resolver
        ):
            raise ExpertReleasePublicationError(
                "expert publisher authorities are not one concrete trust boundary"
            )
        self.assembler = assembler
        self.validation_store = validation_store
        self.github_publisher = github_publisher
        self.resolver = resolver
        self.current_release_authority = current_release_authority
        self.validation_store._bind_release_publisher_authority(self)

    def resolve_stale(
        self,
        *,
        candidate_id: str,
        resolved_at: str,
    ) -> ExpertReleasePublicationStaleResolution:
        """Resolve only an intent displaced by another active expert release."""

        require_content_id(candidate_id, "stale expert publication candidate_id")
        reservation = self.validation_store.reopen_release_publication(candidate_id)
        if reservation is None:
            return self.validation_store.reopen_stale_release_publication(candidate_id)
        plan = reservation.plan
        state = self.resolver.read_current_pointer_state(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            allow_missing=True,
        )
        pointer = state.pointer
        if pointer is None:
            raise ExpertReleasePublicationError(
                "absent CURRENT cannot displace a reserved expert release"
            )
        active_release_id = pointer.publication_record.artifact_id
        if active_release_id == plan.release_id:
            raise ExpertReleasePublicationError(
                "active reserved release requires RELEASED recovery"
            )
        if active_release_id == plan.parent_release_id:
            raise ExpertReleasePublicationError(
                "pending parent-preserving publication must resume before resolution"
            )
        observed = self.current_release_authority.observe_task_evaluation_current(
            plan.scope_id
        )
        if (
            observed.release_id != active_release_id
            or observed.default_branch_head_commit_sha != state.head_commit_sha
        ):
            raise ExpertReleasePublicationError(
                "expert CURRENT changed during stale publication classification"
            )
        permit = self.validation_store._seal_stale_release_publication(
            publisher=self,
            reservation=reservation,
            observed_current=observed,
            resolved_at=resolved_at,
        )
        return self.validation_store.resolve_stale_release_publication(permit)


__all__ = [
    "ExpertReleasePublicationError",
    "ExpertReleasePublisher",
]
