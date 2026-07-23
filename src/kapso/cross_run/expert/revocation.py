"""Fresh emergency-denylist authority for expert release lifecycle revocation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertReleaseRevocationCommitResult,
        ExpertValidationStore,
    )


class ExpertReleaseRevocationError(ValueError):
    """A release cannot be revoked from the current authenticated authority."""


class ExpertReleaseRevocationDenylistAuthority(Protocol):
    """Authenticate current emergency revocations for one exact release closure."""

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


class ExpertReleaseRevocationCoordinator:
    """Append REVOKED only from one fresh authenticated emergency observation."""

    def __init__(
        self,
        *,
        validation_store: ExpertValidationStore,
        security_denylist_authority: ExpertReleaseRevocationDenylistAuthority,
    ) -> None:
        self.validation_store = validation_store
        self.security_denylist_authority = security_denylist_authority
        self.validation_store._bind_release_revocation_authority(self)

    def revoke(
        self,
        *,
        candidate_id: str,
        revoked_at: str,
    ) -> ExpertReleaseRevocationCommitResult:
        require_content_id(candidate_id, "revoked release candidate_id")
        replay = self.validation_store.reopen_release_revocation(candidate_id)
        if replay is not None:
            return replay
        target = self.validation_store.reopen_release_revocation_target(candidate_id)
        observation = self.security_denylist_authority.observe_exact(
            scope_id=target.manifest.scope_id,
            scope_contract_id=target.manifest.scope_contract_id,
            checked_subject_ids=target.security_subject_ids,
        )
        if (
            type(observation) is not SecurityDenylistObservation
            or observation.scope_id != target.manifest.scope_id
            or observation.scope_contract_id != target.manifest.scope_contract_id
            or observation.scope_repository_binding_hash
            != target.activation.receipt.activation_witness.scope_repository_binding_hash
            or observation.checked_subject_ids != target.security_subject_ids
            or not observation.matched_revocations
        ):
            raise ExpertReleaseRevocationError(
                "release revocation lacks an exact current emergency match"
            )
        permit = self.validation_store._seal_release_revocation(
            coordinator=self,
            target=target,
            security_denylist_observation=observation,
            revoked_at=revoked_at,
        )
        return self.validation_store.commit_release_revocation(permit)
