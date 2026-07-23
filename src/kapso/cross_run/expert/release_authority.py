"""Authenticated historical activation authority for expert releases."""

from __future__ import annotations

import os
from dataclasses import dataclass

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertScopeContract,
    GitHubPublicationRecord,
    PublicationArtifactKind,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    GitHubArtifactMaterializer,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
    GitHubArtifactResolver,
    ResolvedGitHubArtifact,
)


class ExpertReleaseActivationAuthorityError(ValueError):
    """Historical GitHub state does not prove one exact expert activation."""


@dataclass(frozen=True)
class _ResolvedExpertReleaseActivation:
    resolved: ResolvedGitHubArtifact
    intent: ArtifactPublicationIntent
    witness: GitHubArtifactActivationWitness


_AUTHENTICATED_EXPERT_RELEASE_ACTIVATION_SEAL = object()


class AuthenticatedExpertReleaseActivation:
    """Process-local proof that an immutable expert release won CURRENT."""

    __slots__ = (
        "_cache_receipt",
        "_manifest",
        "_materialized",
        "_owner_process_id",
        "_provider",
        "_publication",
        "_remote",
        "_scope_contract",
        "_witness",
    )

    def __init__(
        self,
        seal: object,
        provider: GitHubExpertReleaseActivationProvider,
        *,
        scope_contract: ExpertScopeContract,
        remote: _ResolvedExpertReleaseActivation,
        manifest: ExpertBaseReleaseManifest,
        materialized: MaterializedArtifact,
    ) -> None:
        if seal is not _AUTHENTICATED_EXPERT_RELEASE_ACTIVATION_SEAL:
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation capability is not provider sealed"
            )
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_scope_contract", scope_contract)
        object.__setattr__(self, "_remote", remote)
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(
            self, "_publication", remote.resolved.pointer.publication_record
        )
        object.__setattr__(self, "_witness", remote.witness)
        object.__setattr__(self, "_materialized", materialized)
        object.__setattr__(self, "_cache_receipt", materialized.receipt)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertReleaseActivationAuthorityError(
            "expert release activation capability is immutable"
        )

    def __reduce__(self) -> object:
        raise ExpertReleaseActivationAuthorityError(
            "expert release activation capability cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        raise ExpertReleaseActivationAuthorityError(
            "expert release activation capability cannot be serialized"
        )

    @property
    def manifest(self) -> ExpertBaseReleaseManifest:
        self._require_owner_process()
        return self._manifest

    @property
    def publication(self) -> GitHubPublicationRecord:
        self._require_owner_process()
        return self._publication

    @property
    def pointer(self) -> CurrentArtifactPointer:
        self._require_owner_process()
        return self._remote.resolved.pointer

    @property
    def witness(self) -> GitHubArtifactActivationWitness:
        self._require_owner_process()
        return self._witness

    @property
    def cache_receipt(self) -> CacheVerificationReceipt:
        self._require_owner_process()
        return self._cache_receipt

    @property
    def materialized(self) -> MaterializedArtifact:
        self._require_owner_process()
        return self._materialized

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation capability is foreign"
            )

    def _require_bound(
        self,
        provider: GitHubExpertReleaseActivationProvider,
    ) -> None:
        self._require_owner_process()
        if self._provider is not provider:
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation capability belongs to another provider"
            )


class GitHubExpertReleaseActivationProvider:
    """Prove one exact historical expert publication became CURRENT."""

    __slots__ = ("_materializer", "_resolver")

    def __init__(
        self,
        resolver: GitHubArtifactResolver,
        materializer: GitHubArtifactMaterializer,
    ) -> None:
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_materializer", materializer)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertReleaseActivationAuthorityError(
            "expert release activation provider is immutable"
        )

    def resolve_exact(
        self,
        scope_contract: ExpertScopeContract,
        release_id: str,
    ) -> AuthenticatedExpertReleaseActivation:
        if type(scope_contract) is not ExpertScopeContract:
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation requires one exact scope contract"
            )
        require_content_id(release_id, "historical expert release_id")
        if release_id.split(":sha256:", 1)[0] != "expert-base-release":
            raise ExpertReleaseActivationAuthorityError(
                "historical expert release_id uses the wrong namespace"
            )
        first = self._resolve_remote(scope_contract.scope_id, release_id)
        materialized = self._materializer.materialize(first.resolved)
        if type(materialized) is not MaterializedArtifact:
            raise ExpertReleaseActivationAuthorityError(
                "expert release materializer returned an invalid artifact"
            )
        manifest = self._materializer.inspect_expert_release_manifest(materialized)
        if type(manifest) is not ExpertBaseReleaseManifest:
            raise ExpertReleaseActivationAuthorityError(
                "expert release materializer returned an invalid manifest"
            )
        self._validate_materialized(
            scope_contract=scope_contract,
            remote=first,
            materialized=materialized,
            manifest=manifest,
        )
        second = self._resolve_remote(scope_contract.scope_id, release_id)
        second_manifest = self._materializer.inspect_expert_release_manifest(
            materialized
        )
        if (
            second != first
            or second_manifest != manifest
            or materialized.receipt
            != self._validate_materialized(
                scope_contract=scope_contract,
                remote=second,
                materialized=materialized,
                manifest=second_manifest,
            )
        ):
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation authority changed during resolution"
            )
        return AuthenticatedExpertReleaseActivation(
            _AUTHENTICATED_EXPERT_RELEASE_ACTIVATION_SEAL,
            self,
            scope_contract=scope_contract,
            remote=second,
            manifest=second_manifest,
            materialized=materialized,
        )

    def require_exact(
        self,
        capability: AuthenticatedExpertReleaseActivation,
    ) -> AuthenticatedExpertReleaseActivation:
        """Reauthenticate one provider-owned capability without accepting its data."""

        if type(capability) is not AuthenticatedExpertReleaseActivation:
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation freshness requires its live capability"
            )
        capability._require_bound(self)
        refreshed = self.resolve_exact(
            capability._scope_contract,
            capability._manifest.release_id,
        )
        if (
            refreshed._remote != capability._remote
            or refreshed._manifest != capability._manifest
            or refreshed._publication != capability._publication
            or refreshed._witness != capability._witness
            or refreshed._cache_receipt != capability._cache_receipt
            or refreshed._materialized != capability._materialized
        ):
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation authority changed after authentication"
            )
        return capability

    def _resolve_remote(
        self,
        scope_id: str,
        release_id: str,
    ) -> _ResolvedExpertReleaseActivation:
        artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
        resolved = self._resolver.resolve_artifact(
            scope_id,
            artifact_kind,
            release_id,
        )
        if type(resolved) is not ResolvedGitHubArtifact:
            raise ExpertReleaseActivationAuthorityError(
                "GitHub resolver returned an invalid expert release"
            )
        intent = self._resolver.read_artifact_intent(
            scope_id,
            artifact_kind,
            release_id,
        )
        if type(intent) is not ArtifactPublicationIntent:
            raise ExpertReleaseActivationAuthorityError(
                "expert release publication intent is missing"
            )
        witness = self._resolver.resolve_artifact_activation_witness(
            scope_id,
            artifact_kind,
            release_id,
            intent,
            resolved.pointer,
        )
        if type(witness) is not GitHubArtifactActivationWitness:
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation witness is missing"
            )
        pointer = resolved.pointer
        publication = pointer.publication_record
        repositories = resolved.repositories
        if (
            pointer.scope_id != scope_id
            or repositories.scope_id != scope_id
            or publication.artifact_kind is not artifact_kind
            or publication.artifact_id != release_id
            or publication.repository_full_name != repositories.expert_repository
            or publication.repository_full_name != resolved.policy.repository_full_name
            or publication.repository_node_id != resolved.policy.repository_node_id
            or not intent.binds(pointer)
            or witness.scope_id != scope_id
            or witness.scope_repository_binding_hash != repositories.binding_fingerprint
            or witness.artifact_kind is not artifact_kind
            or witness.artifact_id != release_id
            or witness.repository_full_name != publication.repository_full_name
            or witness.publication_intent_digest != intent.digest
            or witness.current_pointer_digest
            != tree_or_blob_digest(pointer.to_json_bytes())
        ):
            raise ExpertReleaseActivationAuthorityError(
                "expert release activation authority does not join exactly"
            )
        return _ResolvedExpertReleaseActivation(
            resolved=resolved,
            intent=intent,
            witness=witness,
        )

    @staticmethod
    def _validate_materialized(
        *,
        scope_contract: ExpertScopeContract,
        remote: _ResolvedExpertReleaseActivation,
        materialized: MaterializedArtifact,
        manifest: ExpertBaseReleaseManifest,
    ) -> CacheVerificationReceipt:
        pointer = remote.resolved.pointer
        publication = pointer.publication_record
        receipt = materialized.receipt
        expected_assets = {asset.name: asset.sha256 for asset in publication.assets}
        if (
            receipt.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or receipt.artifact_id != publication.artifact_id
            or receipt.materialized_tree_digest != pointer.materialized_tree_digest
            or receipt.manifest_relative_path != pointer.manifest_relative_path
            or receipt.manifest_digest != pointer.manifest_digest
            or dict(receipt.asset_digests) != expected_assets
            or manifest.release_id != publication.artifact_id
            or manifest.scope_id != scope_contract.scope_id
            or manifest.scope_contract_id != scope_contract.scope_contract_id
        ):
            raise ExpertReleaseActivationAuthorityError(
                "materialized expert release differs from activation authority"
            )
        return receipt


__all__ = [
    "AuthenticatedExpertReleaseActivation",
    "ExpertReleaseActivationAuthorityError",
    "GitHubExpertReleaseActivationProvider",
]
