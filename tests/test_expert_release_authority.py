"""Authenticated historical expert-release activation boundaries."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

import kapso.cross_run.expert as expert_package
from kapso.cross_run.canonical import (
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import PublicationArtifactKind
from kapso.cross_run.expert.release_authority import (
    AuthenticatedExpertReleaseActivation,
    ExpertReleaseActivationAuthorityError,
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.github.materializer import MaterializedArtifact
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    GitHubArtifactActivationWitness,
    PublicationAssetIntent,
    PublicationSourceFile,
)
from test_expert_composition_base import _case, _remint
from test_expert_composition_base_provider import _resolved


class _Resolver:
    def __init__(self, resolved, intent, witness) -> None:
        self.resolved = list(resolved)
        self.intent = list(intent)
        self.witness = list(witness)
        self.resolve_calls = []
        self.intent_calls = []
        self.witness_calls = []

    @staticmethod
    def _next(values):
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    def resolve_artifact(self, scope_id, artifact_kind, release_id):
        self.resolve_calls.append((scope_id, artifact_kind, release_id))
        return self._next(self.resolved)

    def read_artifact_intent(self, scope_id, artifact_kind, release_id):
        self.intent_calls.append((scope_id, artifact_kind, release_id))
        return self._next(self.intent)

    def resolve_artifact_activation_witness(
        self,
        scope_id,
        artifact_kind,
        release_id,
        intent,
        pointer,
        *,
        allow_missing=False,
    ):
        self.witness_calls.append(
            (
                scope_id,
                artifact_kind,
                release_id,
                intent,
                pointer,
                allow_missing,
            )
        )
        return self._next(self.witness)


class _Materializer:
    def __init__(self, materialized, manifests) -> None:
        self.materialized = materialized
        self.manifests = list(manifests)
        self.materialize_calls = []
        self.inspect_calls = []

    @staticmethod
    def _next(values):
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    def materialize(self, resolved):
        self.materialize_calls.append(resolved)
        return self.materialized

    def inspect_expert_release_manifest(self, materialized):
        self.inspect_calls.append(materialized)
        return self._next(self.manifests)


def _authority_fixture(
    *,
    resolved_overrides=(),
    manifest_overrides=(),
    witness_overrides=(),
):
    case = _case()
    base = _resolved(case)
    manifest_payload = case.release.to_json_bytes()
    source_file = PublicationSourceFile(
        relative_path=base.pointer.manifest_relative_path,
        mode="100644",
        size=len(manifest_payload),
        sha256=tree_or_blob_digest(manifest_payload),
        git_blob_sha=git_object_sha("blob", manifest_payload),
    )
    source_tree_hash = source_tree_digest(
        {
            source_file.relative_path: (
                source_file.sha256,
                source_file.mode,
                source_file.size,
            )
        }
    )
    source_git_tree_sha = git_tree_shas(
        {
            source_file.relative_path: (
                source_file.git_blob_sha,
                source_file.mode,
            )
        }
    )[""]
    publication = base.pointer.publication_record
    intent = ArtifactPublicationIntent(
        scope_id=case.scope.scope_id,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=case.release.release_id,
        repository_node_id=publication.repository_node_id,
        repository_full_name=publication.repository_full_name,
        expected_parent_sha="1" * 40,
        source_commit_sha=publication.commit_sha,
        source_tree_digest=source_tree_hash,
        source_git_tree_sha=source_git_tree_sha,
        source_files=(source_file,),
        preserved_current=None,
        materialized_tree_digest=(
            case.source_base_receipt.cache_verification_receipt.materialized_tree_digest
        ),
        manifest_relative_path=base.pointer.manifest_relative_path,
        manifest_digest=tree_or_blob_digest(manifest_payload),
        tag=publication.tag,
        assets=tuple(
            PublicationAssetIntent(
                name=asset.name,
                media_type=asset.media_type,
                size=asset.size,
                sha256=asset.sha256,
            )
            for asset in publication.assets
        ),
        validation_closure_ids=base.pointer.validation_closure_ids,
        publisher_identity=publication.publisher_identity,
        committed_at=publication.published_at,
    )
    pointer = replace(
        base.pointer,
        publication_intent_digest=intent.digest,
        source_tree_digest=intent.source_tree_digest,
        source_git_tree_sha=intent.source_git_tree_sha,
        materialized_tree_digest=intent.materialized_tree_digest,
        manifest_relative_path=intent.manifest_relative_path,
        manifest_digest=intent.manifest_digest,
    )
    resolved = replace(base, pointer=pointer)
    witness = GitHubArtifactActivationWitness.mint(
        scope_id=case.scope.scope_id,
        scope_repository_binding_hash=resolved.repositories.binding_fingerprint,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=case.release.release_id,
        repository_full_name=publication.repository_full_name,
        activation_commit_sha="2" * 40,
        publication_intent_digest=intent.digest,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
    )
    receipt = case.source_base_receipt.cache_verification_receipt
    materialized = MaterializedArtifact(
        root=Path("/verified/expert"),
        content=Path("/verified/expert/content"),
        assets=Path("/verified/expert/assets"),
        receipt=receipt,
        reused=False,
    )
    resolver = _Resolver(
        resolved_overrides or (resolved,),
        (intent,),
        witness_overrides or (witness,),
    )
    materializer = _Materializer(
        materialized,
        manifest_overrides or (case.release,),
    )
    provider = GitHubExpertReleaseActivationProvider(resolver, materializer)
    return case, resolved, intent, witness, resolver, materializer, provider


def test_fresh_host_resolves_historical_activation_without_validation_store() -> None:
    case, resolved, intent, witness, resolver, materializer, provider = (
        _authority_fixture()
    )

    capability = provider.resolve_exact(case.scope, case.release.release_id)

    assert capability.manifest == case.release
    assert capability.publication == resolved.pointer.publication_record
    assert capability.witness == witness
    assert capability.cache_receipt == materializer.materialized.receipt
    assert resolver.resolve_calls == [
        (
            case.scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            case.release.release_id,
        ),
        (
            case.scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            case.release.release_id,
        ),
    ]
    assert resolver.intent_calls == resolver.resolve_calls
    assert len(resolver.witness_calls) == 2
    assert all(not call[-1] for call in resolver.witness_calls)
    assert materializer.materialize_calls == [resolved]
    assert materializer.inspect_calls == [
        materializer.materialized,
        materializer.materialized,
    ]
    assert intent.binds(resolved.pointer)


def test_historical_activation_provider_is_exported_from_expert_facade() -> None:
    names = {
        "AuthenticatedExpertReleaseActivation",
        "ExpertReleaseActivationAuthorityError",
        "GitHubExpertReleaseActivationProvider",
    }

    assert names.issubset(expert_package.__all__)
    assert all(hasattr(expert_package, name) for name in names)


def test_published_but_never_activated_release_fails() -> None:
    case, _, _, _, _, _, provider = _authority_fixture(
        witness_overrides=(None,),
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="witness is missing",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)


def test_release_activation_rejects_wrong_publication_scope() -> None:
    case, resolved, _, _, _, _, _ = _authority_fixture()
    wrong_pointer = replace(resolved.pointer, scope_id="another_scope")
    wrong_resolved = replace(resolved, pointer=wrong_pointer)
    case, _, _, _, _, _, provider = _authority_fixture(
        resolved_overrides=(wrong_resolved,),
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="does not join exactly",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)


@pytest.mark.parametrize("change_kind", ("scope", "release_id"))
def test_release_activation_rejects_wrong_manifest_scope_or_id(
    change_kind: str,
) -> None:
    case = _case()
    changes = (
        {"scope_id": "another_scope"}
        if change_kind == "scope"
        else {"candidate_tree_hash": tree_or_blob_digest(b"another candidate tree")}
    )
    manifest = _remint(case.release, **changes)
    case, _, _, _, _, _, provider = _authority_fixture(
        manifest_overrides=(manifest,),
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="materialized expert release differs",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)


def test_release_activation_rejects_missing_or_mismatched_witness() -> None:
    case, resolved, intent, witness, _, _, _ = _authority_fixture()
    mismatched_witness = GitHubArtifactActivationWitness.mint(
        scope_id=case.scope.scope_id,
        scope_repository_binding_hash=resolved.repositories.binding_fingerprint,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=case.release.release_id,
        repository_full_name="Leeroo-AI/another-expert",
        activation_commit_sha=witness.activation_commit_sha,
        publication_intent_digest=intent.digest,
        current_pointer_digest=tree_or_blob_digest(resolved.pointer.to_json_bytes()),
    )
    case, _, _, _, _, _, provider = _authority_fixture(
        witness_overrides=(mismatched_witness,),
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="does not join exactly",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)


def test_remote_or_cache_change_during_resolution_fails() -> None:
    case, resolved, _, _, _, _, _ = _authority_fixture()
    changed_resolved = replace(resolved, pointer_commit_sha="3" * 40)
    case, _, _, _, _, _, provider = _authority_fixture(
        resolved_overrides=(resolved, changed_resolved),
    )
    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="changed during resolution",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)

    case, _, _, _, _, _, provider = _authority_fixture()
    changed_manifest = _remint(case.release, scope_id="another_scope")
    provider._materializer.manifests = [case.release, changed_manifest]
    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="changed during resolution",
    ):
        provider.resolve_exact(case.scope, case.release.release_id)


def test_capability_is_unforgeable_unpicklable_and_provider_bound() -> None:
    case, _, _, _, _, materializer, provider = _authority_fixture()
    capability = provider.resolve_exact(case.scope, case.release.release_id)

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="not provider sealed",
    ):
        AuthenticatedExpertReleaseActivation(
            object(),
            provider,
            scope_contract=case.scope,
            remote=capability._remote,
            manifest=capability.manifest,
            cache_receipt=capability.cache_receipt,
        )
    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="cannot be serialized",
    ):
        pickle.dumps(capability)

    foreign_provider = GitHubExpertReleaseActivationProvider(
        _Resolver(
            (capability._remote.resolved,),
            (capability._remote.intent,),
            (capability.witness,),
        ),
        materializer,
    )
    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="another provider",
    ):
        foreign_provider.require_exact(capability)

    assert provider.require_exact(capability) is capability
