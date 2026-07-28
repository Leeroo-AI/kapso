from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
    ScopeRepositorySettings,
)
from kapso.cross_run.expert.composition_base_provider import (
    CurrentExpertCompositionBase,
    ExpertCompositionBaseProviderError,
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
)
from kapso.cross_run.github.materializer import (
    ExpertReleaseSourceSnapshot,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.settings import CrossRunSettings
from test_expert_composition_base import _case, _remint

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class _Resolver:
    def __init__(self, *resolved):
        self.resolved = list(resolved)
        self.calls = []
        self.artifact_calls = []

    def resolve_current(self, scope_id, artifact_kind):
        self.calls.append((scope_id, artifact_kind))
        if len(self.resolved) > 1:
            return self.resolved.pop(0)
        return self.resolved[0]

    def resolve_artifact(self, scope_id, artifact_kind, artifact_id):
        self.artifact_calls.append((scope_id, artifact_kind, artifact_id))
        return self.resolved[0]


class _Materializer:
    def __init__(self, materialized, source_snapshot):
        self.materialized = materialized
        self.source_snapshot = source_snapshot
        self.materialize_calls = []
        self.inspect_calls = []

    def materialize(self, resolved):
        self.materialize_calls.append(resolved)
        return self.materialized

    def inspect_expert_release_source(
        self,
        materialized,
        *,
        maximum_entries,
        maximum_bytes,
    ):
        self.inspect_calls.append((materialized, maximum_entries, maximum_bytes))
        return self.source_snapshot


def _settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert


def _digest(label):
    return tree_or_blob_digest(label.encode("utf-8"))


def _resolved(case, *, head_commit_sha="a" * 40, pointer=None):
    receipt = case.source_base_receipt.cache_verification_receipt
    repository = "Leeroo-AI/kapso-expert"
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=case.release.release_id,
        repository_node_id="expert_repository_node",
        repository_full_name=repository,
        commit_sha="b" * 40,
        immutable_release_id="expert_release_1",
        tag="expert/ml_ai/release-1",
        assets=(
            GitHubReleaseAsset(
                asset_id="expert_source_asset",
                name=case.release.source_archive_ref,
                media_type="application/x-tar",
                size=1,
                sha256=case.release.checksums[case.release.source_archive_ref],
            ),
        ),
        release_attestation_ref="github-release-attestation:test",
        published_at="2026-07-21T00:00:00Z",
        publisher_identity="leeroo-coder",
    )
    current_pointer = pointer or CurrentArtifactPointer(
        scope_id=case.scope.scope_id,
        publication_record=publication,
        publication_intent_digest=_digest("publication intent"),
        source_tree_digest=_digest("published source tree"),
        source_git_tree_sha="c" * 40,
        materialized_tree_digest=receipt.materialized_tree_digest,
        manifest_relative_path=receipt.manifest_relative_path,
        manifest_digest=receipt.manifest_digest,
        validation_closure_ids=(
            content_id("expert-validation-closure", {"release": 1}),
        ),
    )
    repositories = ScopeRepositorySettings(
        scope_id=case.scope.scope_id,
        expert_repository=repository,
        knowledge_repository="Leeroo-AI/kapso-knowledge",
        security_repository="Leeroo-AI/kapso-security",
    )
    policy = RepositoryPolicyReport(
        repository_full_name=repository,
        repository_node_id="expert_repository_node",
        private=True,
        default_branch="main",
        authenticated_actor="leeroo-coder",
        write_access=True,
        immutable_releases=True,
    )
    return ResolvedGitHubArtifact(
        repositories=repositories,
        pointer=current_pointer,
        policy=policy,
        pointer_commit_sha=head_commit_sha,
    )


def _provider_case(*resolved):
    case = _case()
    authority = resolved or (
        _resolved(case),
        _resolved(case, head_commit_sha="d" * 40),
    )
    materialized = MaterializedArtifact(
        root=Path("/verified/expert"),
        content=Path("/verified/expert/content"),
        assets=Path("/verified/expert/assets"),
        receipt=case.source_base_receipt.cache_verification_receipt,
        reused=False,
    )
    source_snapshot = ExpertReleaseSourceSnapshot(
        release_manifest=case.release,
        source_extraction_receipt=(case.source_base_receipt.source_extraction_receipt),
        source_contents=case.source_contents,
    )
    resolver = _Resolver(*authority)
    materializer = _Materializer(materialized, source_snapshot)
    settings = _settings()
    provider = GitHubExpertCompositionBaseProvider(
        resolver,
        materializer,
        settings,
    )
    return case, resolver, materializer, settings, provider


def test_current_base_is_sealed_after_materialization_between_two_observations():
    case, resolver, materializer, settings, provider = _provider_case()

    capability = provider.resolve_current(case.scope)

    assert capability.closure.reference.release_id == case.release.release_id
    assert capability.closure.source_contents == case.source_contents
    assert capability.current_observation.current_pointer_commit_sha == "d" * 40
    assert capability.current_observation.observation_id in (
        capability.security_subject_ids
    )
    assert capability.closure.source_base_tree_receipt.source_base_tree_receipt_id in (
        capability.security_subject_ids
    )
    assert materializer.inspect_calls == [
        (
            materializer.materialized,
            settings.candidate_entry_limit,
            settings.candidate_byte_limit,
        )
    ]
    assert resolver.calls == [
        (
            case.scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        ),
        (
            case.scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        ),
    ]


def test_current_base_freshness_allows_same_pointer_on_newer_head():
    case = _case()
    first = _resolved(case)
    second = _resolved(case, head_commit_sha="d" * 40)
    third = _resolved(case, head_commit_sha="e" * 40)
    case, _, _, _, provider = _provider_case(first, second, third)
    capability = provider.resolve_current(case.scope)

    observation = provider.require_current(capability)

    assert observation.current_pointer_commit_sha == "e" * 40
    assert observation.current_pointer_digest == (
        capability.current_observation.current_pointer_digest
    )


def test_current_base_resolution_rejects_pointer_change_during_materialization():
    case = _case()
    first = _resolved(case)
    changed_pointer = replace(
        first.pointer,
        source_tree_digest=_digest("changed published source tree"),
    )
    second = _resolved(case, pointer=changed_pointer)
    case, _, _, _, provider = _provider_case(first, second)

    with pytest.raises(ExpertCompositionBaseProviderError, match="changed"):
        provider.resolve_current(case.scope)


def test_current_base_rejects_substituted_scope_repository_routing():
    case = _case()
    authority = _resolved(case)
    substituted = replace(
        authority,
        repositories=replace(authority.repositories, scope_id="another_scope"),
    )
    case, _, _, _, provider = _provider_case(substituted)

    with pytest.raises(ExpertCompositionBaseProviderError, match="routing"):
        provider.resolve_current(case.scope)


def test_current_base_capability_is_unforgeable_unpicklable_and_provider_bound():
    case, _, materializer, settings, provider = _provider_case()
    capability = provider.resolve_current(case.scope)

    with pytest.raises(ExpertCompositionBaseProviderError, match="not provider sealed"):
        CurrentExpertCompositionBase(
            object(),
            provider,
            closure=capability.closure,
            current_observation=capability.current_observation,
            resolved_current=_resolved(case),
        )
    with pytest.raises(ExpertCompositionBaseProviderError, match="serialized"):
        pickle.dumps(capability)

    foreign_provider = GitHubExpertCompositionBaseProvider(
        _Resolver(_resolved(case)),
        materializer,
        settings,
    )
    with pytest.raises(ExpertCompositionBaseProviderError, match="another provider"):
        foreign_provider.require_current(capability)


def test_current_base_rejects_scope_contract_substitution():
    case, _, _, _, provider = _provider_case()
    substituted_scope = _remint(case.scope, purpose="Substituted scope purpose.")

    with pytest.raises(
        ExpertCompositionBaseProviderError,
        match="scope authority",
    ):
        provider.resolve_current(substituted_scope)


def test_exact_historical_base_materializes_under_task_evaluation_limits():
    case, resolver, materializer, _settings, provider = _provider_case(
        _resolved(_case())
    )
    limits = TaskEvaluationMaterializationLimits(
        maximum_entries=100,
        maximum_bytes=100_000,
        timeout_seconds=30,
    )

    source = provider.materialize_exact(
        case.release,
        case.source_base_receipt,
        limits,
    )

    assert source.release_manifest == case.release
    assert source.source_base_tree_receipt == case.source_base_receipt
    assert source.source_contents == case.source_contents
    assert resolver.artifact_calls == [
        (
            case.scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            case.release.release_id,
        )
    ]
    assert materializer.inspect_calls == [(materializer.materialized, 100, 100_000)]
