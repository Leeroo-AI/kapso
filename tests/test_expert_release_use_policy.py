"""Authenticated current release-use policy observations."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.catalog.store import (
    CatalogGenerationManifest,
    CatalogInputDelta,
)
from kapso.cross_run.contracts import (
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
    ScopeRepositorySettings,
)
from kapso.cross_run.expert.release_authority import (
    ExpertReleaseActivationAuthorityError,
)
from kapso.cross_run.expert.release_use_policy import (
    ExpertReleaseUsePolicyError,
    GitHubExpertReleaseUsePolicyAuthority,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyContractError,
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.knowledge.package import (
    KnowledgeSnapshotPackageBuilder,
)
from kapso.cross_run.record_contracts import (
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from test_expert_release_authority import _authority_fixture
from test_knowledge_snapshot_package import (
    empty_generation,
    finalize,
    populated_generation,
)


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _package_with_revocations(revocations):
    scope, _, _, generation, objects = populated_generation()
    fact_ids = tuple(
        sorted(
            (
                *generation.fact_object_ids,
                *(revocation.revocation_id for revocation in revocations),
            )
        )
    )
    input_delta = CatalogInputDelta.mint(
        scope_contract_id=scope.scope_contract_id,
        operation_id="release-use-policy-observation-test",
        configuration_fingerprint=generation.configuration_fingerprint,
        added_object_ids=fact_ids,
        dependency_closure_ids=fact_ids,
    )
    current_generation = CatalogGenerationManifest.mint(
        scope_contract_id=generation.scope_contract_id,
        generation_number=generation.generation_number,
        parent_generation_id=generation.parent_generation_id,
        configuration_fingerprint=generation.configuration_fingerprint,
        fact_object_ids=fact_ids,
        derived_object_ids=generation.derived_object_ids,
        applied_input_delta_ids=(input_delta.input_delta_id,),
        bundle_frontier=generation.bundle_frontier,
        active_entry_state_ids=generation.active_entry_state_ids,
    )
    current_objects = {
        object_id: payload
        for object_id, payload in objects.items()
        if object_id not in generation.applied_input_delta_ids
    }
    current_objects[input_delta.input_delta_id] = input_delta.to_json_bytes()
    current_objects.update(
        {
            revocation.revocation_id: revocation.to_json_bytes()
            for revocation in revocations
        }
    )
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        current_generation,
        current_objects.__getitem__,
    )
    return scope, finalize(prepared)


def _empty_package():
    scope, _, _, _, _ = populated_generation()
    prepared = KnowledgeSnapshotPackageBuilder.prepare_empty(
        scope,
        empty_generation(scope),
    )
    return scope, finalize(prepared)


def _run_bundle_evidence_id() -> str:
    _, _, _, generation, _ = populated_generation()
    return next(
        object_id
        for object_id in generation.fact_object_ids
        if object_id.startswith("run-bundle:")
    )


def _revocation(
    *,
    scope,
    release_id,
    publication_id,
    witness_id,
    kind=ExpertReleaseUseRevocationKind.PERFORMANCE,
):
    return ExpertReleaseUseRevocation.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id=scope.scope_id,
        release_id=release_id,
        release_publication_id=publication_id,
        release_activation_witness_id=witness_id,
        kind=kind,
        reason_code=f"{kind.value}_regression",
        rationale=f"Observed a release-wide {kind.value} regression.",
        exact_evidence_refs=(_run_bundle_evidence_id(),),
        recorded_at="2026-07-23T00:00:00Z",
    )


class _CurrentResolver:
    def __init__(self, resolved) -> None:
        self.resolved = list(resolved)
        self.calls = []

    def resolve_current(self, scope_id, artifact_kind):
        self.calls.append((scope_id, artifact_kind))
        if len(self.resolved) > 1:
            return self.resolved.pop(0)
        return self.resolved[0]


class _MissingCurrentResolver:
    def __init__(self) -> None:
        self.calls = []

    def resolve_current(self, scope_id, artifact_kind):
        self.calls.append((scope_id, artifact_kind))
        raise LookupError("knowledge CURRENT is missing")


class _KnowledgeMaterializer:
    def __init__(self, materialized) -> None:
        self.materialized = materialized
        self.calls = []

    def materialize(self, resolved):
        self.calls.append(resolved)
        return self.materialized


def test_policy_authority_requires_sealed_historical_provider() -> None:
    with pytest.raises(
        ExpertReleaseUsePolicyError,
        match="requires the GitHub activation provider",
    ):
        GitHubExpertReleaseUsePolicyAuthority(object(), object(), object())


def _current_policy_authority(
    tmp_path: Path,
    package,
    activation_provider,
):
    content = (tmp_path / "knowledge-content").absolute()
    package.materialize(content)
    asset_payload = b"verified knowledge package"
    asset = GitHubReleaseAsset(
        asset_id="knowledge_asset",
        name="knowledge-snapshot.tar.zst",
        media_type="application/zstd",
        size=len(asset_payload),
        sha256=tree_or_blob_digest(asset_payload),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=package.manifest.snapshot_id,
        repository_node_id="knowledge_repository_node",
        repository_full_name="Leeroo-AI/kapso-knowledge",
        commit_sha="a" * 40,
        immutable_release_id="knowledge_release_1",
        tag="knowledge/S000001",
        assets=(asset,),
        release_attestation_ref="knowledge-release-attestation",
        published_at="2026-07-23T00:00:00Z",
        publisher_identity="leeroo-coder",
    )
    materialized_tree_digest = tree_or_blob_digest(b"materialized knowledge package")
    pointer = CurrentArtifactPointer(
        scope_id=package.manifest.scope_id,
        publication_record=publication,
        publication_intent_digest=tree_or_blob_digest(b"knowledge publication intent"),
        source_tree_digest=tree_or_blob_digest(b"knowledge source tree"),
        source_git_tree_sha="b" * 40,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path="snapshot.json",
        manifest_digest=tree_or_blob_digest(package.manifest.to_json_bytes()),
        validation_closure_ids=(package.manifest.snapshot_id,),
    )
    repositories = ScopeRepositorySettings(
        scope_id=package.manifest.scope_id,
        expert_repository="Leeroo-AI/kapso-expert",
        knowledge_repository=publication.repository_full_name,
        security_repository="Leeroo-AI/kapso-security",
    )
    policy = RepositoryPolicyReport(
        repository_full_name=publication.repository_full_name,
        repository_node_id=publication.repository_node_id,
        private=True,
        default_branch="main",
        authenticated_actor="leeroo-coder",
        write_access=True,
        immutable_releases=True,
    )
    resolved = ResolvedGitHubArtifact(
        repositories=repositories,
        pointer=pointer,
        policy=policy,
        pointer_commit_sha="c" * 40,
    )
    receipt = CacheVerificationReceipt(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=package.manifest.snapshot_id,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=pointer.manifest_relative_path,
        manifest_digest=pointer.manifest_digest,
        cache_tree_digest=tree_or_blob_digest(b"verified knowledge cache"),
        asset_digests={asset.name: asset.sha256},
    )
    materialized = MaterializedArtifact(
        root=tmp_path.absolute(),
        content=content,
        assets=(tmp_path / "knowledge-assets").absolute(),
        receipt=receipt,
        reused=False,
    )
    resolver = _CurrentResolver((resolved,))
    materializer = _KnowledgeMaterializer(materialized)
    authority = GitHubExpertReleaseUsePolicyAuthority(
        resolver,
        materializer,
        activation_provider,
    )
    return authority, resolver, materializer, resolved


def test_empty_checked_release_set_produces_authenticated_e0_observation(
    tmp_path,
) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    scope, package = _empty_package()
    authority, resolver, materializer, resolved = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(),
    )

    assert scope == case.scope
    assert observation.checked_release_ids == ()
    assert observation.matched_revocations == ()
    assert observation.matched_release_ids == ()
    assert observation.catalog_generation == 0
    assert observation.knowledge_snapshot_id == package.manifest.snapshot_id
    assert observation.knowledge_publication_id == (
        resolved.pointer.publication_record.publication_id
    )
    assert ExpertReleaseUsePolicyObservation.from_dict(observation.to_dict()) == (
        observation
    )
    assert activation_resolver.resolve_calls == []
    assert resolver.calls == [
        (scope.scope_id, PublicationArtifactKind.KNOWLEDGE_SNAPSHOT),
        (scope.scope_id, PublicationArtifactKind.KNOWLEDGE_SNAPSHOT),
    ]
    assert materializer.calls == [resolved]


def test_unmatched_checked_release_is_still_authenticated(tmp_path) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    baseline_calls = len(activation_resolver.resolve_calls)

    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(case.release.release_id,),
    )

    assert observation.matched_revocations == ()
    assert len(activation_resolver.resolve_calls) == baseline_calls + 2


def test_predecessor_contract_release_cannot_receive_false_absence(
    tmp_path,
) -> None:
    case, _, _, _, _, _, activation_provider = _authority_fixture()
    successor_scope = _remint(
        case.scope,
        purpose=f"{case.scope.purpose} successor",
        supersedes_scope_contract_id=case.scope.scope_contract_id,
    )
    prepared = KnowledgeSnapshotPackageBuilder.prepare_empty(
        successor_scope,
        empty_generation(successor_scope),
    )
    package = finalize(prepared)
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="materialized expert release differs",
    ):
        authority.observe_exact(
            scope_contract=successor_scope,
            checked_release_ids=(case.release.release_id,),
        )


def test_matching_revocation_is_bound_to_historical_activation(tmp_path) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    activation = activation_provider.resolve_exact(
        case.scope,
        case.release.release_id,
    )
    revocation = _revocation(
        scope=case.scope,
        release_id=activation.manifest.release_id,
        publication_id=activation.publication.publication_id,
        witness_id=activation.witness.witness_id,
    )
    scope, package = _package_with_revocations((revocation,))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    baseline_calls = len(activation_resolver.resolve_calls)

    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(case.release.release_id,),
    )

    assert observation.matched_revocations == (revocation,)
    assert observation.matched_release_ids == (case.release.release_id,)
    assert len(activation_resolver.resolve_calls) == baseline_calls + 2
    with pytest.raises(
        ExpertReleaseUsePolicyContractError,
        match="generation-zero",
    ):
        _remint(observation, catalog_generation=0)


def test_exact_scope_contract_not_only_scope_name_is_required(tmp_path) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    substituted_scope = _remint(
        scope,
        purpose=f"{scope.purpose} with substituted authority",
    )

    with pytest.raises(
        ExpertReleaseUsePolicyError,
        match="package differs",
    ):
        authority.observe_exact(
            scope_contract=substituted_scope,
            checked_release_ids=(case.release.release_id,),
        )

    assert activation_resolver.resolve_calls == []


def test_current_publication_must_name_the_materialized_snapshot(tmp_path) -> None:
    _, _, _, _, _, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    authority, resolver, _, resolved = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    wrong_publication = _remint(
        resolved.pointer.publication_record,
        artifact_id=content_id(
            "knowledge-snapshot",
            {"substituted": "publication"},
        ),
    )
    resolver.resolved = [
        replace(
            resolved,
            pointer=replace(
                resolved.pointer,
                publication_record=wrong_publication,
            ),
        )
    ]

    with pytest.raises(
        ExpertReleaseUsePolicyError,
        match="package differs",
    ):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=(),
        )


@pytest.mark.parametrize("changed_field", ("publication", "witness"))
def test_matching_event_must_name_exact_historical_publication_and_witness(
    tmp_path,
    changed_field,
) -> None:
    case, _, _, _, _, _, activation_provider = _authority_fixture()
    activation = activation_provider.resolve_exact(
        case.scope,
        case.release.release_id,
    )
    changes = (
        {
            "publication_id": content_id(
                "github-publication",
                {"substituted": "expert release"},
            ),
            "witness_id": activation.witness.witness_id,
        }
        if changed_field == "publication"
        else {
            "publication_id": activation.publication.publication_id,
            "witness_id": content_id(
                "github-artifact-activation-witness",
                {"substituted": "expert release"},
            ),
        }
    )
    revocation = _revocation(
        scope=case.scope,
        release_id=case.release.release_id,
        publication_id=changes["publication_id"],
        witness_id=changes["witness_id"],
    )
    scope, package = _package_with_revocations((revocation,))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    with pytest.raises(
        ExpertReleaseUsePolicyError,
        match="differs from historical activation",
    ):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=(case.release.release_id,),
        )


def test_published_but_never_activated_matching_release_fails(tmp_path) -> None:
    (
        case,
        resolved_release,
        _,
        witness,
        _,
        _,
        activation_provider,
    ) = _authority_fixture(witness_overrides=(None,))
    revocation = _revocation(
        scope=case.scope,
        release_id=case.release.release_id,
        publication_id=(resolved_release.pointer.publication_record.publication_id),
        witness_id=witness.witness_id,
    )
    scope, package = _package_with_revocations((revocation,))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    with pytest.raises(
        ExpertReleaseActivationAuthorityError,
        match="witness is missing",
    ):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=(case.release.release_id,),
        )


def test_unrelated_broken_event_is_scanned_but_not_authenticated(tmp_path) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    unrelated = _revocation(
        scope=case.scope,
        release_id=content_id(
            "expert-base-release",
            {"unrelated": "release"},
        ),
        publication_id=content_id(
            "github-publication",
            {"unrelated": "publication"},
        ),
        witness_id=content_id(
            "github-artifact-activation-witness",
            {"unrelated": "witness"},
        ),
    )
    scope, package = _package_with_revocations((unrelated,))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(case.release.release_id,),
    )

    assert observation.matched_revocations == ()
    assert activation_resolver.resolve_calls == [
        (
            scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            case.release.release_id,
        ),
        (
            scope.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            case.release.release_id,
        ),
    ]


def test_multiple_matching_events_authenticate_once_per_release_and_sort(
    tmp_path,
) -> None:
    case, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    activation = activation_provider.resolve_exact(
        case.scope,
        case.release.release_id,
    )
    revocations = tuple(
        _revocation(
            scope=case.scope,
            release_id=case.release.release_id,
            publication_id=activation.publication.publication_id,
            witness_id=activation.witness.witness_id,
            kind=kind,
        )
        for kind in (
            ExpertReleaseUseRevocationKind.PERFORMANCE,
            ExpertReleaseUseRevocationKind.COMPATIBILITY,
        )
    )
    scope, package = _package_with_revocations(tuple(reversed(revocations)))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    baseline_calls = len(activation_resolver.resolve_calls)

    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(case.release.release_id,),
    )

    assert observation.matched_revocations == tuple(
        sorted(revocations, key=lambda event: event.revocation_id)
    )
    assert len(activation_resolver.resolve_calls) == baseline_calls + 2


@pytest.mark.parametrize(
    "checked_release_ids",
    (
        [],
        (
            f"expert-base-release:sha256:{'2' * 64}",
            f"expert-base-release:sha256:{'1' * 64}",
        ),
        (
            content_id("expert-base-release", {"duplicate": 1}),
            content_id("expert-base-release", {"duplicate": 1}),
        ),
        (content_id("knowledge-snapshot", {"wrong": "kind"}),),
    ),
)
def test_checked_release_ids_are_rejected_before_remote_resolution(
    tmp_path,
    checked_release_ids,
) -> None:
    _, _, _, _, activation_resolver, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    authority, resolver, materializer, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )

    with pytest.raises(ValueError):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=checked_release_ids,
        )

    assert resolver.calls == []
    assert materializer.calls == []
    assert activation_resolver.resolve_calls == []


def test_observation_cannot_match_a_release_it_did_not_check(tmp_path) -> None:
    case, _, _, _, _, _, activation_provider = _authority_fixture()
    activation = activation_provider.resolve_exact(
        case.scope,
        case.release.release_id,
    )
    revocation = _revocation(
        scope=case.scope,
        release_id=case.release.release_id,
        publication_id=activation.publication.publication_id,
        witness_id=activation.witness.witness_id,
    )
    scope, package = _package_with_revocations((revocation,))
    authority, _, _, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    observation = authority.observe_exact(
        scope_contract=scope,
        checked_release_ids=(case.release.release_id,),
    )

    with pytest.raises(
        ExpertReleaseUsePolicyContractError,
        match="was not checked",
    ):
        _remint(observation, checked_release_ids=())


def test_current_change_during_observation_fails_closed(tmp_path) -> None:
    _, _, _, _, _, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    authority, resolver, _, resolved = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    resolver.resolved = [
        resolved,
        replace(resolved, pointer_commit_sha="d" * 40),
    ]

    with pytest.raises(
        ExpertReleaseUsePolicyError,
        match="changed during",
    ):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=(),
        )


def test_missing_current_has_no_checkpoint_or_offline_fallback(tmp_path) -> None:
    _, _, _, _, _, _, activation_provider = _authority_fixture()
    scope, package = _package_with_revocations(())
    _, _, materializer, _ = _current_policy_authority(
        tmp_path,
        package,
        activation_provider,
    )
    resolver = _MissingCurrentResolver()
    authority = GitHubExpertReleaseUsePolicyAuthority(
        resolver,
        materializer,
        activation_provider,
    )

    with pytest.raises(LookupError, match="CURRENT is missing"):
        authority.observe_exact(
            scope_contract=scope,
            checked_release_ids=(),
        )

    assert materializer.calls == []
