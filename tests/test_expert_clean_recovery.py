"""Authenticated clean-forward expert recovery source selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertReleaseLineage,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.recovery_base import (
    ExpertRecoveryBaseError,
    ExpertRecoveryBaseSelection,
    ExpertRecoveryBaseSelector,
)
from kapso.cross_run.expert.recovery_contracts import ExpertRecoveryContractError
from kapso.cross_run.expert.release_authority import (
    GitHubExpertReleaseActivationProvider,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
    PublicationAssetIntent,
    PublicationSourceFile,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.record_contracts import (
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings
from security_denylist_fixtures import matched_security_revocations
from test_expert_composition_base import _case, _remint

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _settings() -> CrossRunSettings:
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def _release_chain(length: int):
    case = _case()
    releases = [case.release]
    for position in range(1, length):
        predecessor = releases[-1]
        releases.append(
            _remint(
                case.release,
                lineage=ExpertReleaseLineage(
                    source_base_release_id=predecessor.release_id,
                    activation_predecessor_release_id=predecessor.release_id,
                ),
                candidate_tree_hash=_digest(f"candidate-tree-{position}"),
                candidate_consumed_expert_release_ids=(predecessor.release_id,),
                consumed_dependency_ids=tuple(
                    sorted(
                        {
                            *case.release.consumed_dependency_ids,
                            predecessor.release_id,
                        }
                    )
                ),
            )
        )
    return case, tuple(reversed(releases))


def _remote_activation(case, release, position, settings):
    manifest_payload = release.to_json_bytes()
    source_file = PublicationSourceFile(
        relative_path="expert-release.json",
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
    asset_digest = release.checksums[release.source_archive_ref]
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release.release_id,
        repository_node_id="expert_repository_node",
        repository_full_name="Leeroo-AI/kapso-expert",
        commit_sha=f"{position + 1:040x}",
        immutable_release_id=str(position + 1),
        tag=f"{settings.github.expert_tag_prefix}E{position:06d}",
        assets=(
            GitHubReleaseAsset(
                asset_id=f"asset-{position}",
                name=release.source_archive_ref,
                media_type="application/zstd",
                size=1,
                sha256=asset_digest,
            ),
        ),
        release_attestation_ref=f"attestation-{position}",
        published_at=f"2026-07-2{position}T00:00:00Z",
        publisher_identity="leeroo-coder",
    )
    intent = ArtifactPublicationIntent(
        scope_id=case.scope.scope_id,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release.release_id,
        repository_node_id=publication.repository_node_id,
        repository_full_name=publication.repository_full_name,
        expected_parent_sha=f"{position + 10:040x}",
        source_commit_sha=publication.commit_sha,
        source_tree_digest=source_tree_hash,
        source_git_tree_sha=source_git_tree_sha,
        source_files=(source_file,),
        preserved_current=None,
        materialized_tree_digest=_digest(f"materialized-{position}"),
        manifest_relative_path=source_file.relative_path,
        manifest_digest=source_file.sha256,
        tag=publication.tag,
        assets=(
            PublicationAssetIntent(
                name=release.source_archive_ref,
                media_type="application/zstd",
                size=1,
                sha256=asset_digest,
            ),
        ),
        validation_closure_ids=(release.release_id,),
        publisher_identity=publication.publisher_identity,
        committed_at=publication.published_at,
    )
    pointer = CurrentArtifactPointer(
        scope_id=case.scope.scope_id,
        publication_record=publication,
        publication_intent_digest=intent.digest,
        source_tree_digest=source_tree_hash,
        source_git_tree_sha=source_git_tree_sha,
        materialized_tree_digest=intent.materialized_tree_digest,
        manifest_relative_path=intent.manifest_relative_path,
        manifest_digest=intent.manifest_digest,
        validation_closure_ids=intent.validation_closure_ids,
    )
    repositories = settings.scopes.resolve(case.scope.scope_id)
    resolved = ResolvedGitHubArtifact(
        repositories=repositories,
        pointer=pointer,
        policy=RepositoryPolicyReport(
            repository_full_name=repositories.expert_repository,
            repository_node_id=publication.repository_node_id,
            private=True,
            default_branch="main",
            authenticated_actor="leeroo-coder",
            write_access=True,
            immutable_releases=True,
        ),
        pointer_commit_sha=f"{position + 20:040x}",
    )
    witness = GitHubArtifactActivationWitness.mint(
        scope_id=case.scope.scope_id,
        scope_repository_binding_hash=repositories.binding_fingerprint,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release.release_id,
        repository_full_name=publication.repository_full_name,
        activation_commit_sha=resolved.pointer_commit_sha,
        publication_intent_digest=intent.digest,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
    )
    receipt = CacheVerificationReceipt(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release.release_id,
        materialized_tree_digest=pointer.materialized_tree_digest,
        manifest_relative_path=pointer.manifest_relative_path,
        manifest_digest=pointer.manifest_digest,
        cache_tree_digest=_digest(f"cache-{position}"),
        asset_digests={release.source_archive_ref: asset_digest},
    )
    materialized = MaterializedArtifact(
        root=Path(f"/verified/{position}"),
        content=Path(f"/verified/{position}/content"),
        assets=Path(f"/verified/{position}/assets"),
        receipt=receipt,
        reused=False,
    )
    return SimpleNamespace(
        release=release,
        resolved=resolved,
        intent=intent,
        pointer=pointer,
        witness=witness,
        materialized=materialized,
    )


class _Resolver:
    def __init__(self, remotes):
        self.remotes = {remote.release.release_id: remote for remote in remotes}
        self.calls = []

    def resolve_artifact(self, scope_id, artifact_kind, release_id):
        self.calls.append(("resolve", release_id))
        return self.remotes[release_id].resolved

    def read_artifact_intent(self, scope_id, artifact_kind, release_id):
        self.calls.append(("intent", release_id))
        return self.remotes[release_id].intent

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
        self.calls.append(("witness", release_id))
        return self.remotes[release_id].witness


class _Materializer:
    def __init__(self, remotes):
        self.remotes = {remote.release.release_id: remote for remote in remotes}

    def materialize(self, resolved):
        release_id = resolved.pointer.publication_record.artifact_id
        return self.remotes[release_id].materialized

    def inspect_expert_release_manifest(self, materialized):
        return self.remotes[materialized.receipt.artifact_id].release


class _CurrentAuthority:
    def __init__(self, observation, *, move=False):
        self.observation = observation
        self.move = move
        self.calls = 0

    def observe_task_evaluation_current(self, scope_id):
        self.calls += 1
        if self.move and self.calls > 1:
            return TaskEvaluationCurrentReleaseObservation.mint(
                scope_id=self.observation.scope_id,
                release_id=self.observation.release_id,
                publication_id=self.observation.publication_id,
                repository_full_name=self.observation.repository_full_name,
                repository_node_id=self.observation.repository_node_id,
                default_branch_head_commit_sha="f" * 40,
                current_pointer_digest=self.observation.current_pointer_digest,
                validation_closure_ids=self.observation.validation_closure_ids,
            )
        return self.observation


class _SecurityAuthority:
    def __init__(self, settings, blocked_release_ids):
        self.settings = settings
        self.blocked_release_ids = set(blocked_release_ids)
        self.calls = []

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append(checked_subject_ids)
        matched = tuple(
            release_id
            for release_id in sorted(self.blocked_release_ids)
            if release_id in checked_subject_ids
        )
        repositories = self.settings.scopes.resolve(scope_id)
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=repositories.binding_fingerprint,
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"blocked": tuple(sorted(self.blocked_release_ids))},
            ),
            generation=7,
            publication_id=content_id(
                "github-publication",
                {"security": tuple(sorted(self.blocked_release_ids))},
            ),
            repository_full_name=repositories.security_repository,
            repository_node_id="security_repository_node",
            pointer_digest=_digest("security-current"),
            authority_commit_sha="d" * 40,
            release_attestation_ref="security-attestation",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=matched_security_revocations(matched),
        )


class _ReleaseUseAuthority:
    def __init__(self, settings, remotes, blocked_release_ids):
        self.settings = settings
        self.remotes = {remote.release.release_id: remote for remote in remotes}
        self.blocked_release_ids = set(blocked_release_ids)
        self.calls = []

    def observe_exact(self, *, scope_contract, checked_release_ids):
        self.calls.append(checked_release_ids)
        matched = ()
        release_id = checked_release_ids[0]
        if release_id in self.blocked_release_ids:
            remote = self.remotes[release_id]
            matched = (
                ExpertReleaseUseRevocation.mint(
                    scope_contract_id=scope_contract.scope_contract_id,
                    scope_id=scope_contract.scope_id,
                    release_id=release_id,
                    release_publication_id=(
                        remote.pointer.publication_record.publication_id
                    ),
                    release_activation_witness_id=remote.witness.witness_id,
                    kind=ExpertReleaseUseRevocationKind.PERFORMANCE,
                    reason_code="release_wide_regression",
                    rationale="Authenticated release-wide regression.",
                    exact_evidence_refs=(
                        content_id("run-bundle", {"release": release_id}),
                    ),
                    recorded_at="2026-07-23T00:00:00Z",
                ),
            )
        repositories = self.settings.scopes.resolve(scope_contract.scope_id)
        return ExpertReleaseUsePolicyObservation.mint(
            scope_id=scope_contract.scope_id,
            scope_contract_id=scope_contract.scope_contract_id,
            scope_repository_binding_hash=repositories.binding_fingerprint,
            repository_full_name=repositories.knowledge_repository,
            repository_node_id="knowledge_repository_node",
            knowledge_snapshot_id=content_id(
                "knowledge-snapshot",
                {"blocked": tuple(sorted(self.blocked_release_ids))},
            ),
            catalog_generation=7,
            knowledge_publication_id=content_id(
                "github-publication",
                {"knowledge": tuple(sorted(self.blocked_release_ids))},
            ),
            current_pointer_digest=_digest("knowledge-current"),
            authority_commit_sha="e" * 40,
            release_attestation_ref="knowledge-attestation",
            checked_release_ids=checked_release_ids,
            matched_revocations=matched,
        )


def _fixture(
    *,
    length=4,
    security_blocked=(),
    release_use_blocked=(),
    lineage_limit=None,
    move_current=False,
):
    case, releases = _release_chain(length)
    settings = _settings()
    if lineage_limit is not None:
        settings = replace(
            settings,
            expert=replace(
                settings.expert,
                recovery_lineage_limit=lineage_limit,
            ),
        )
    remotes = tuple(
        _remote_activation(case, release, position, settings)
        for position, release in enumerate(releases)
    )
    resolver = _Resolver(remotes)
    provider = GitHubExpertReleaseActivationProvider(
        resolver,
        _Materializer(remotes),
    )
    barrier = remotes[0]
    current = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=case.scope.scope_id,
        release_id=barrier.release.release_id,
        publication_id=barrier.pointer.publication_record.publication_id,
        repository_full_name=barrier.pointer.publication_record.repository_full_name,
        repository_node_id=barrier.pointer.publication_record.repository_node_id,
        default_branch_head_commit_sha=barrier.witness.activation_commit_sha,
        current_pointer_digest=tree_or_blob_digest(barrier.pointer.to_json_bytes()),
        validation_closure_ids=barrier.pointer.validation_closure_ids,
    )
    current_authority = _CurrentAuthority(current, move=move_current)
    security = _SecurityAuthority(settings, security_blocked)
    release_use = _ReleaseUseAuthority(
        settings,
        remotes,
        release_use_blocked,
    )
    selector = ExpertRecoveryBaseSelector(
        settings=settings,
        activation_provider=provider,
        current_authority=current_authority,
        security_authority=security,
        release_use_authority=release_use,
    )
    return SimpleNamespace(
        case=case,
        releases=releases,
        remotes=remotes,
        resolver=resolver,
        current=current_authority,
        security=security,
        release_use=release_use,
        selector=selector,
    )


def test_recovery_selects_newest_clear_historical_release():
    fixture = _fixture(
        release_use_blocked=(),
    )
    barrier, selected = fixture.releases[:2]
    fixture.release_use.blocked_release_ids.add(barrier.release_id)

    result = fixture.selector.select(fixture.case.scope)

    assert result.plan.activation_predecessor_release_id == barrier.release_id
    assert result.plan.source_base_release_id == selected.release_id
    assert (
        result.plan.configuration_fingerprint == _settings().configuration_fingerprint
    )
    assert result.selected_activation.manifest == selected
    assert tuple(assessment.release_id for assessment in result.plan.assessments) == (
        barrier.release_id,
        selected.release_id,
    )
    assert all(
        release_id != fixture.releases[2].release_id
        for _kind, release_id in fixture.resolver.calls
    )
    assert fixture.current.calls == 2


def test_recovery_skips_emergency_and_release_use_blocked_predecessors():
    fixture = _fixture()
    barrier, emergency_blocked, release_use_blocked, selected = fixture.releases
    fixture.release_use.blocked_release_ids.update(
        {barrier.release_id, release_use_blocked.release_id}
    )
    fixture.security.blocked_release_ids.add(emergency_blocked.release_id)

    result = fixture.selector.select(fixture.case.scope)

    assert result.plan.source_base_release_id == selected.release_id
    assert tuple(
        assessment.release_id for assessment in result.plan.assessments
    ) == tuple(release.release_id for release in fixture.releases)
    assert result.plan.assessments[1].security_observation.matched_revocations
    assert result.plan.assessments[2].release_use_observation.matched_revocations
    assert all(
        assessment.security_subject_ids
        == tuple(
            sorted(
                {
                    assessment.release_id,
                    assessment.publication_pointer.publication_record.publication_id,
                    assessment.activation_witness.witness_id,
                    *assessment.manifest.consumed_dependency_ids,
                }
            )
        )
        for assessment in result.plan.assessments
    )


def test_recovery_selects_empty_only_after_authenticated_exhaustion():
    fixture = _fixture()
    fixture.release_use.blocked_release_ids.update(
        release.release_id for release in fixture.releases
    )

    result = fixture.selector.select(fixture.case.scope)

    assert result.plan.source_base_release_id is None
    assert result.plan.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST
    assert result.selected_activation is None
    assert result.plan.assessments[-1].manifest.lineage == ExpertReleaseLineage(
        source_base_release_id=None,
        activation_predecessor_release_id=None,
    )
    assert len(result.plan.assessments) == len(fixture.releases)


def test_recovery_does_not_treat_missing_historical_activation_as_exhaustion():
    fixture = _fixture()
    barrier, missing_predecessor = fixture.releases[:2]
    fixture.release_use.blocked_release_ids.add(barrier.release_id)
    fixture.resolver.remotes.pop(missing_predecessor.release_id)

    with pytest.raises(KeyError):
        fixture.selector.select(fixture.case.scope)


def test_recovery_plan_rejects_duplicate_historical_assessments():
    fixture = _fixture()
    barrier = fixture.releases[0]
    fixture.release_use.blocked_release_ids.add(barrier.release_id)
    plan = fixture.selector.select(fixture.case.scope).plan

    with pytest.raises(
        ExpertRecoveryContractError,
        match="CURRENT or ordered assessment identity",
    ):
        replace(plan, assessments=(plan.assessments[0], plan.assessments[0]))


def test_recovery_assessment_rejects_cache_asset_substitution():
    fixture = _fixture()
    barrier = fixture.releases[0]
    fixture.release_use.blocked_release_ids.add(barrier.release_id)
    assessment = fixture.selector.select(fixture.case.scope).plan.assessments[0]
    substituted_receipt = replace(
        assessment.cache_receipt,
        asset_digests={"substituted.tar.zst": _digest("substituted")},
    )

    with pytest.raises(
        ExpertRecoveryContractError,
        match="authorities do not join",
    ):
        replace(assessment, cache_receipt=substituted_receipt)


def test_recovery_selection_cannot_be_reconstructed_from_its_durable_plan():
    fixture = _fixture()
    barrier = fixture.releases[0]
    fixture.release_use.blocked_release_ids.add(barrier.release_id)
    selection = fixture.selector.select(fixture.case.scope)

    with pytest.raises(ExpertRecoveryBaseError, match="not selector sealed"):
        ExpertRecoveryBaseSelection(
            object(),
            fixture.selector,
            plan=selection.plan,
            selected_activation=selection.selected_activation,
        )


@pytest.mark.parametrize("failure", ("depth", "current_movement", "clear_current"))
def test_recovery_fails_loud_without_complete_stable_barrier_authority(failure):
    fixture = _fixture(
        lineage_limit=2 if failure == "depth" else None,
        move_current=failure == "current_movement",
    )
    if failure == "clear_current":
        expected = "CURRENT is clear"
    else:
        fixture.release_use.blocked_release_ids.update(
            release.release_id for release in fixture.releases
        )
        expected = "lineage limit" if failure == "depth" else "CURRENT changed"

    with pytest.raises(ExpertRecoveryBaseError, match=expected):
        fixture.selector.select(fixture.case.scope)
