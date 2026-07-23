"""Authenticated author for non-emergency expert release-use revocations."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.canonical import (
    normalize_utc_timestamp,
    require_content_id,
    require_identifier,
)
from kapso.cross_run.catalog.release_use_authority import (
    CatalogReleaseUseRevocationAuthority,
    _seal_catalog_release_use_revocation_authority,
)
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.catalog.store import (
    CatalogCommitResult,
    CatalogGenerationManifest,
)
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.release_authority import (
    AuthenticatedExpertReleaseActivation,
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.record_contracts import (
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)


class ExpertReleaseUseRevocationAuthorError(ValueError):
    """A release-use revocation request lacks exact authenticated authority."""


@dataclass(frozen=True)
class PendingExpertReleaseUseRevocation:
    """A catalog fact pending projection into a successor knowledge snapshot."""

    event: ExpertReleaseUseRevocation
    catalog_commit: CatalogCommitResult

    def __post_init__(self) -> None:
        if (
            type(self.event) is not ExpertReleaseUseRevocation
            or type(self.catalog_commit) is not CatalogCommitResult
            or self.event.revocation_id
            not in self.catalog_commit.generation.fact_object_ids
            or self.event.scope_contract_id
            != self.catalog_commit.generation.scope_contract_id
        ):
            raise ExpertReleaseUseRevocationAuthorError(
                "pending release-use revocation does not join its catalog commit"
            )


class ExpertReleaseUseRevocationAuthor:
    """Bind historical GitHub activation proof to one exact catalog event."""

    __slots__ = ("_authority", "_catalog", "_provider")

    def __init__(
        self,
        catalog: CrossRunCatalog,
        provider: GitHubExpertReleaseActivationProvider,
    ) -> None:
        if type(catalog) is not CrossRunCatalog:
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use author requires one exact catalog"
            )
        if type(provider) is not GitHubExpertReleaseActivationProvider:
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use author requires the GitHub activation provider"
            )
        object.__setattr__(self, "_catalog", catalog)
        object.__setattr__(self, "_provider", provider)
        authority = _seal_catalog_release_use_revocation_authority(
            author=self,
            catalog=catalog,
        )
        object.__setattr__(self, "_authority", authority)
        catalog._bind_release_use_revocation_authority(authority)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertReleaseUseRevocationAuthorError(
            "release-use revocation author is immutable"
        )

    def publish(
        self,
        *,
        expected_generation: CatalogGenerationManifest,
        release_id: str,
        kind: ExpertReleaseUseRevocationKind,
        reason_code: str,
        rationale: str,
        exact_evidence_refs: tuple[str, ...],
        recorded_at: str,
    ) -> PendingExpertReleaseUseRevocation:
        self._authority._require_bound(catalog=self._catalog)
        self._validate_request_before_remote_resolution(
            expected_generation=expected_generation,
            release_id=release_id,
            kind=kind,
            reason_code=reason_code,
            rationale=rationale,
            exact_evidence_refs=exact_evidence_refs,
            recorded_at=recorded_at,
        )
        historical_activation = self._provider.resolve_exact(
            self._catalog.scope_contract,
            release_id,
        )
        event = ExpertReleaseUseRevocation.mint(
            scope_contract_id=self._catalog.scope_contract.scope_contract_id,
            scope_id=self._catalog.scope_contract.scope_id,
            release_id=historical_activation.manifest.release_id,
            release_publication_id=(historical_activation.publication.publication_id),
            release_activation_witness_id=historical_activation.witness.witness_id,
            kind=kind,
            reason_code=reason_code,
            rationale=rationale,
            exact_evidence_refs=exact_evidence_refs,
            recorded_at=recorded_at,
        )
        commit = self._catalog._publish_authenticated_release_use_revocation(
            authority=self._authority,
            historical_activation=historical_activation,
            expected_generation=expected_generation,
            event=event,
        )
        return PendingExpertReleaseUseRevocation(
            event=event,
            catalog_commit=commit,
        )

    def _validate_request_before_remote_resolution(
        self,
        *,
        expected_generation: CatalogGenerationManifest,
        release_id: str,
        kind: ExpertReleaseUseRevocationKind,
        reason_code: str,
        rationale: str,
        exact_evidence_refs: tuple[str, ...],
        recorded_at: str,
    ) -> None:
        require_content_id(release_id, "release-use revocation release_id")
        if release_id.split(":sha256:", 1)[0] != "expert-base-release":
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation release_id uses the wrong namespace"
            )
        if type(kind) is not ExpertReleaseUseRevocationKind:
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation kind is invalid"
            )
        require_identifier(reason_code, "release-use revocation reason_code")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation rationale must not be empty"
            )
        if (
            type(exact_evidence_refs) is not tuple
            or not exact_evidence_refs
            or exact_evidence_refs != tuple(sorted(set(exact_evidence_refs)))
        ):
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation evidence must be non-empty, sorted, and unique"
            )
        for evidence_id in exact_evidence_refs:
            require_content_id(
                evidence_id,
                "release-use revocation exact_evidence_refs",
            )
        normalize_utc_timestamp(
            recorded_at,
            "release-use revocation recorded_at",
        )
        self._catalog._require_release_use_evidence_generation(
            expected_generation=expected_generation,
            exact_evidence_refs=exact_evidence_refs,
        )

    def _require_authenticated_event(
        self,
        *,
        historical_activation: object,
        event: object,
    ) -> None:
        if (
            type(historical_activation) is not AuthenticatedExpertReleaseActivation
            or type(event) is not ExpertReleaseUseRevocation
        ):
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation lacks exact historical activation"
            )
        activation = self._provider.require_exact(historical_activation)
        manifest = activation.manifest
        publication = activation.publication
        witness = activation.witness
        scope = self._catalog.scope_contract
        artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
        if (
            event.scope_contract_id != scope.scope_contract_id
            or event.scope_id != scope.scope_id
            or manifest.scope_contract_id != scope.scope_contract_id
            or manifest.scope_id != scope.scope_id
            or event.release_id != manifest.release_id
            or publication.artifact_kind is not artifact_kind
            or publication.artifact_id != manifest.release_id
            or event.release_publication_id != publication.publication_id
            or witness.scope_id != scope.scope_id
            or witness.artifact_kind is not artifact_kind
            or witness.artifact_id != manifest.release_id
            or event.release_activation_witness_id != witness.witness_id
        ):
            raise ExpertReleaseUseRevocationAuthorError(
                "release-use revocation does not join historical activation exactly"
            )


__all__ = [
    "ExpertReleaseUseRevocationAuthor",
    "ExpertReleaseUseRevocationAuthorError",
    "PendingExpertReleaseUseRevocation",
]
