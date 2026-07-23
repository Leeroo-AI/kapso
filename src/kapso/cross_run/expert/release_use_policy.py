"""Authenticated current KnowledgeSnapshot reader for expert release-use policy."""

from __future__ import annotations

from collections import defaultdict

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertScopeContract,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.release_authority import (
    AuthenticatedExpertReleaseActivation,
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    GitHubArtifactMaterializer,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    GitHubArtifactResolver,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.record_contracts import ExpertReleaseUseRevocation


class ExpertReleaseUsePolicyError(ValueError):
    """Current scientific memory cannot authorize a release-use decision."""


class GitHubExpertReleaseUsePolicyAuthority:
    """Read exact release-use matches from authenticated knowledge CURRENT."""

    __slots__ = ("_activation_provider", "_materializer", "_resolver")

    def __init__(
        self,
        resolver: GitHubArtifactResolver,
        materializer: GitHubArtifactMaterializer,
        activation_provider: GitHubExpertReleaseActivationProvider,
    ) -> None:
        if type(activation_provider) is not GitHubExpertReleaseActivationProvider:
            raise ExpertReleaseUsePolicyError(
                "release-use policy requires the GitHub activation provider"
            )
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_materializer", materializer)
        object.__setattr__(self, "_activation_provider", activation_provider)

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertReleaseUsePolicyError("release-use policy authority is immutable")

    def observe_exact(
        self,
        *,
        scope_contract: ExpertScopeContract,
        checked_release_ids: tuple[str, ...],
    ) -> ExpertReleaseUsePolicyObservation:
        if type(scope_contract) is not ExpertScopeContract:
            raise ExpertReleaseUsePolicyError(
                "release-use policy requires one exact scope contract"
            )
        self._validate_checked_release_ids(checked_release_ids)
        first = self._resolve_current(scope_contract.scope_id)
        materialized = self._materializer.materialize(first)
        if type(materialized) is not MaterializedArtifact:
            raise ExpertReleaseUsePolicyError(
                "knowledge materializer returned an invalid artifact"
            )
        package = KnowledgeSnapshotPackage.open(materialized.content)
        self._validate_current_package(
            scope_contract=scope_contract,
            resolved=first,
            materialized=materialized,
            package=package,
        )
        revocations = self._release_use_revocations(package)
        matches = tuple(
            revocation
            for revocation in revocations
            if revocation.release_id in checked_release_ids
        )
        self._authenticate_matches(scope_contract, matches)
        second = self._resolve_current(scope_contract.scope_id)
        if second != first:
            raise ExpertReleaseUsePolicyError(
                "knowledge CURRENT changed during release-use policy resolution"
            )
        self._validate_current_package(
            scope_contract=scope_contract,
            resolved=second,
            materialized=materialized,
            package=package,
        )
        pointer = second.pointer
        publication = pointer.publication_record
        return ExpertReleaseUsePolicyObservation.mint(
            scope_id=scope_contract.scope_id,
            scope_contract_id=scope_contract.scope_contract_id,
            scope_repository_binding_hash=(second.repositories.binding_fingerprint),
            repository_full_name=publication.repository_full_name,
            repository_node_id=publication.repository_node_id,
            knowledge_snapshot_id=package.manifest.snapshot_id,
            catalog_generation=package.manifest.catalog_generation,
            knowledge_publication_id=publication.publication_id,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
            authority_commit_sha=second.pointer_commit_sha,
            release_attestation_ref=publication.release_attestation_ref,
            checked_release_ids=checked_release_ids,
            matched_revocations=matches,
        )

    @staticmethod
    def _validate_checked_release_ids(
        checked_release_ids: tuple[str, ...],
    ) -> None:
        if type(checked_release_ids) is not tuple or checked_release_ids != tuple(
            sorted(set(checked_release_ids))
        ):
            raise ExpertReleaseUsePolicyError(
                "checked release IDs must be a sorted unique tuple"
            )
        for release_id in checked_release_ids:
            require_content_id(release_id, "checked expert release ID")
            if release_id.split(":sha256:", 1)[0] != "expert-base-release":
                raise ExpertReleaseUsePolicyError(
                    "checked expert release ID uses the wrong namespace"
                )

    def _resolve_current(self, scope_id: str) -> ResolvedGitHubArtifact:
        resolved = self._resolver.resolve_current(
            scope_id,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        )
        if type(resolved) is not ResolvedGitHubArtifact:
            raise ExpertReleaseUsePolicyError(
                "GitHub resolver returned an invalid knowledge CURRENT"
            )
        pointer = resolved.pointer
        publication = pointer.publication_record
        if (
            pointer.scope_id != scope_id
            or resolved.repositories.scope_id != scope_id
            or publication.artifact_kind
            is not PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
            or publication.repository_full_name
            != resolved.repositories.knowledge_repository
            or publication.repository_full_name != resolved.policy.repository_full_name
            or publication.repository_node_id != resolved.policy.repository_node_id
        ):
            raise ExpertReleaseUsePolicyError(
                "knowledge CURRENT uses another scope, routing, kind, or repository"
            )
        return resolved

    @staticmethod
    def _validate_current_package(
        *,
        scope_contract: ExpertScopeContract,
        resolved: ResolvedGitHubArtifact,
        materialized: MaterializedArtifact,
        package: KnowledgeSnapshotPackage,
    ) -> None:
        pointer = resolved.pointer
        publication = pointer.publication_record
        receipt = materialized.receipt
        expected_assets = {asset.name: asset.sha256 for asset in publication.assets}
        if (
            receipt.artifact_kind is not PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
            or receipt.artifact_id != publication.artifact_id
            or receipt.materialized_tree_digest != pointer.materialized_tree_digest
            or receipt.manifest_relative_path != pointer.manifest_relative_path
            or receipt.manifest_digest != pointer.manifest_digest
            or dict(receipt.asset_digests) != expected_assets
            or package.manifest.snapshot_id != publication.artifact_id
            or package.manifest.scope_id != scope_contract.scope_id
            or package.manifest.scope_contract_id != scope_contract.scope_contract_id
            or package.prepared.scope_contract != scope_contract
        ):
            raise ExpertReleaseUsePolicyError(
                "knowledge package differs from current publication authority"
            )

    @staticmethod
    def _release_use_revocations(
        package: KnowledgeSnapshotPackage,
    ) -> tuple[ExpertReleaseUseRevocation, ...]:
        revocations = []
        for revocation_id in package.manifest.active_expert_release_use_revocation_ids:
            envelope = package.record_by_id(revocation_id)
            if envelope["record_kind"] != "expert-release-use-revocation":
                raise ExpertReleaseUsePolicyError(
                    "release-use projection contains another record kind"
                )
            revocation = ExpertReleaseUseRevocation.from_dict(envelope["payload"])
            if (
                revocation.revocation_id != revocation_id
                or canonical_json_bytes(envelope["payload"])
                != revocation.to_json_bytes()
            ):
                raise ExpertReleaseUsePolicyError(
                    "release-use projection record is not canonical"
                )
            revocations.append(revocation)
        ordered = tuple(
            sorted(revocations, key=lambda revocation: revocation.revocation_id)
        )
        if tuple(revocations) != ordered:
            raise ExpertReleaseUsePolicyError("release-use projection is not canonical")
        return ordered

    def _authenticate_matches(
        self,
        scope_contract: ExpertScopeContract,
        matches: tuple[ExpertReleaseUseRevocation, ...],
    ) -> None:
        by_release: dict[str, list[ExpertReleaseUseRevocation]] = defaultdict(list)
        for revocation in matches:
            by_release[revocation.release_id].append(revocation)
        for release_id in sorted(by_release):
            activation = self._activation_provider.resolve_exact(
                scope_contract,
                release_id,
            )
            if type(activation) is not AuthenticatedExpertReleaseActivation:
                raise ExpertReleaseUsePolicyError(
                    "release activation provider returned invalid authority"
                )
            for revocation in by_release[release_id]:
                if (
                    revocation.scope_id != scope_contract.scope_id
                    or revocation.scope_contract_id != scope_contract.scope_contract_id
                    or revocation.release_id != activation.manifest.release_id
                    or revocation.release_publication_id
                    != activation.publication.publication_id
                    or revocation.release_activation_witness_id
                    != activation.witness.witness_id
                ):
                    raise ExpertReleaseUsePolicyError(
                        "release-use revocation differs from historical activation"
                    )


__all__ = [
    "ExpertReleaseUsePolicyError",
    "GitHubExpertReleaseUsePolicyAuthority",
]
