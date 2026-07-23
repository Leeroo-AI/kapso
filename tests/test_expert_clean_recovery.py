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
    ExpertCandidateDerivationKind,
    ExpertCandidateOperationKind,
    ExpertSourceTreeManifest,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.recovery_base import (
    ExpertRecoveryBaseError,
    ExpertRecoveryBaseSelection,
    ExpertRecoveryBaseSelector,
)
from kapso.cross_run.expert.candidates import (
    ExpertCandidateValidationError,
)
from kapso.cross_run.expert.composition_base_provider import (
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.recovery_candidate_coordinator import (
    ExpertCleanForwardRecoveryCandidateCoordinator,
    ExpertRecoveryCandidateCoordinatorError,
)
from kapso.cross_run.expert.proposal_contract import ExpertProposalContractError
from kapso.cross_run.expert.recovery_candidate import (
    project_canonical_empty_recovery_packet,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStoreError,
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
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvaluator,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    ExpertReleaseSourceSnapshot,
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
from test_expert_composition_base import _case, _parent_receipt, _remint
from test_cross_run_retrieval import source_fixture
from test_expert_proposal import proposal_system
from test_expert_proposal import released_observation_packet
from test_expert_triggers import trigger_packet

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _settings() -> CrossRunSettings:
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def _release_chain(length: int):
    case = _case()
    source_tree = ExpertSourceTreeManifest.mint(
        tree_hash=case.source_base_receipt.source_base_tree_hash,
        files=case.source_base_receipt.source_extraction_receipt.source_tree_files,
    )
    release = _remint(
        case.release,
        candidate_tree_ref=source_tree.source_tree_manifest_id,
        candidate_tree_hash=source_tree.tree_hash,
        evidence_dependency_ids=tuple(
            sorted(
                {
                    *case.release.evidence_dependency_ids,
                    source_tree.source_tree_manifest_id,
                }
                - {case.release.candidate_tree_ref}
            )
        ),
        consumed_dependency_ids=tuple(
            sorted(
                {
                    *case.release.consumed_dependency_ids,
                    source_tree.source_tree_manifest_id,
                }
                - {case.release.candidate_tree_ref}
            )
        ),
    )
    case = SimpleNamespace(
        **{
            **vars(case),
            "release": release,
            "source_base_receipt": _parent_receipt(
                release,
                case.repository_map,
                case.modules,
                case.source_contents,
            ),
        }
    )
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
    def __init__(self, case, remotes):
        self.case = case
        self.remotes = {remote.release.release_id: remote for remote in remotes}
        self.source_contents_by_release_id = {
            remote.release.release_id: case.source_contents for remote in remotes
        }

    def materialize(self, resolved):
        release_id = resolved.pointer.publication_record.artifact_id
        return self.remotes[release_id].materialized

    def inspect_expert_release_manifest(self, materialized):
        return self.remotes[materialized.receipt.artifact_id].release

    def inspect_expert_release_source(
        self,
        materialized,
        *,
        maximum_entries,
        maximum_bytes,
    ):
        release = self.remotes[materialized.receipt.artifact_id].release
        source_contents = self.source_contents_by_release_id[release.release_id]
        receipt = _parent_receipt(
            release,
            self.case.repository_map,
            self.case.modules,
            source_contents,
        )
        return ExpertReleaseSourceSnapshot(
            release_manifest=release,
            source_extraction_receipt=receipt.source_extraction_receipt,
            source_contents=source_contents,
        )


class _CurrentAuthority:
    def __init__(self, observation, *, move_after=None):
        self.observation = observation
        self.move_after = move_after
        self.calls = 0

    def observe_task_evaluation_current(self, scope_id):
        self.calls += 1
        if self.move_after is not None and self.calls > self.move_after:
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
        self.revision = 7
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
                {
                    "blocked": tuple(sorted(self.blocked_release_ids)),
                    "revision": self.revision,
                },
            ),
            generation=self.revision,
            publication_id=content_id(
                "github-publication",
                {
                    "security": tuple(sorted(self.blocked_release_ids)),
                    "revision": self.revision,
                },
            ),
            repository_full_name=repositories.security_repository,
            repository_node_id="security_repository_node",
            pointer_digest=_digest(f"security-current-{self.revision}"),
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
    current_move_after=None,
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
    materializer = _Materializer(case, remotes)
    provider = GitHubExpertReleaseActivationProvider(
        resolver,
        materializer,
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
    current_authority = _CurrentAuthority(
        current,
        move_after=current_move_after,
    )
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
        materializer=materializer,
        current=current_authority,
        security=security,
        release_use=release_use,
        selector=selector,
    )


def _historical_candidate_system(
    tmp_path,
    *,
    current_move_after=None,
    empty_selection=False,
    episodes=(),
    sanitation_mismatch=False,
    taint_episodes_with_barrier=False,
):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    fixture = _fixture(
        length=2,
        current_move_after=current_move_after,
    )
    settings = _settings()
    barrier, selected = fixture.releases
    if taint_episodes_with_barrier:
        episodes = tuple(
            _remint(
                episode,
                artifact_environment=_remint(
                    episode.artifact_environment,
                    expert_base_release_id=barrier.release_id,
                ),
            )
            for episode in episodes
        )
    fixture.release_use.blocked_release_ids.update(
        release.release_id
        for release in (fixture.releases if empty_selection else (barrier,))
    )
    barrier_contents = dict(fixture.case.source_contents)
    barrier_contents["src/reproducible_execution/__init__.py"] = (
        b"def resume():\n    raise RuntimeError('blocked current')\n"
    )
    fixture.materializer.source_contents_by_release_id[barrier.release_id] = (
        barrier_contents
    )
    barrier_receipt = _parent_receipt(
        barrier,
        fixture.case.repository_map,
        fixture.case.modules,
        barrier_contents,
    )
    selected_receipt = _parent_receipt(
        selected,
        fixture.case.repository_map,
        fixture.case.modules,
        fixture.case.source_contents,
    )
    replay_basis = trigger_packet(
        settings=settings.expert.triggers,
        source_base_repository_map=fixture.case.repository_map,
        source_base_module_contracts=fixture.case.modules,
        source_base_release=barrier,
        current_scope_contract=fixture.case.scope,
        source_base_scope_contract=fixture.case.scope,
        episodes=episodes,
    )
    replay_basis = _remint(
        replay_basis,
        source_base_tree_receipt=barrier_receipt,
        source_base_tree_hash=barrier_receipt.source_base_tree_hash,
    )
    architect, candidate_store, runner, _source = proposal_system(
        tmp_path,
        settings=settings.expert,
    )
    validator = candidate_store.validator
    if sanitation_mismatch:
        validator.sanitizer.settings = replace(
            settings.sanitation,
            policy_version=f"{settings.sanitation.policy_version}.mismatch",
        )
    base_provider = GitHubExpertCompositionBaseProvider(
        fixture.resolver,
        fixture.materializer,
        settings.expert,
    )
    coordinator = ExpertCleanForwardRecoveryCandidateCoordinator(
        selector=fixture.selector,
        base_provider=base_provider,
        candidate_store=candidate_store,
        proposal_engine=architect.engine,
    )
    return SimpleNamespace(
        fixture=fixture,
        settings=settings,
        barrier=barrier,
        selected=selected,
        barrier_receipt=barrier_receipt,
        selected_receipt=selected_receipt,
        barrier_contents=barrier_contents,
        replay_basis=replay_basis,
        validator=validator,
        candidate_store=candidate_store,
        coordinator=coordinator,
        architect=architect,
        runner=runner,
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


def test_historical_recovery_candidate_is_exact_admitted_restore(tmp_path):
    system = _historical_candidate_system(tmp_path)

    stored = system.coordinator.restore_historical(
        scope_contract=system.fixture.case.scope,
        replay_basis_packet=system.replay_basis,
    )

    assert stored.closure.manifest.source_base_release_id == system.selected.release_id
    assert stored.closure.manifest.candidate_tree_hash == (
        system.selected_receipt.source_base_tree_hash
    )
    assert stored.closure.patch.changes == ()
    assert stored.closure.candidate_contents == system.fixture.case.source_contents
    assert stored.closure.candidate_contents != system.barrier_contents
    assert stored.recovery_admission.recovery_plan.source_base_release_id == (
        system.selected.release_id
    )
    assert (
        stored.recovery_admission.recovery_plan.activation_predecessor_release_id
        == (system.barrier.release_id)
    )
    assert system.candidate_store.read(stored.closure.manifest.candidate_id) == stored
    with pytest.raises(
        ExpertCandidateStoreError,
        match="sealed admission authority",
    ):
        system.candidate_store.persist(stored.closure)
    system.fixture.security.revision += 1
    with pytest.raises(
        ExpertCandidateStoreError,
        match="identity conflicts",
    ):
        system.coordinator.restore_historical(
            scope_contract=system.fixture.case.scope,
            replay_basis_packet=system.replay_basis,
        )
    contents = dict(stored.closure.candidate_contents)
    changed_path = sorted(contents)[0]
    contents[changed_path] += b"\nsubstitution"
    substituted = replace(stored.closure, candidate_contents=contents)
    with pytest.raises(
        ExpertCandidateValidationError,
        match="candidate bytes differ",
    ):
        system.validator.validate_persisted(substituted)
    admission_path = stored.root / "RECOVERY_ADMISSION.json"
    admission_path.write_bytes(stored.recovery_admission.to_json_bytes() + b"\n")
    with pytest.raises(
        ExpertCandidateStoreError,
        match="admission is not canonical",
    ):
        system.candidate_store.read(stored.closure.manifest.candidate_id)


def test_empty_recovery_is_agent_authored_admitted_and_reopenable(tmp_path):
    _, _, episode, _, _, _, _ = source_fixture()
    system = _historical_candidate_system(
        tmp_path,
        empty_selection=True,
        episodes=(episode,),
    )

    result = system.coordinator.bootstrap_empty(
        scope_contract=system.fixture.case.scope,
        replay_basis_packet=system.replay_basis,
    )
    stored = result.stored_candidate
    closure = stored.closure

    assert closure.manifest.derivation_kind is (
        ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP
    )
    assert closure.derivation.operation.operation_kind is (
        ExpertCandidateOperationKind.RECOVERY_BOOTSTRAP
    )
    assert closure.manifest.source_base_release_id is None
    assert closure.manifest.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST
    assert closure.validation_context.replay_evidence.causal_episode_ids == tuple(
        episode.episode_id for episode in system.replay_basis.episodes
    )
    assert stored.recovery_admission.recovery_plan.source_base_release_id is None
    assert stored.recovery_admission.barrier_replay_basis == system.replay_basis
    assert system.candidate_store.read(closure.manifest.candidate_id) == stored
    with pytest.raises(
        ExpertCandidateStoreError,
        match="sealed admission authority",
    ):
        system.candidate_store.persist(closure)
    empty_packet = project_canonical_empty_recovery_packet(system.replay_basis)
    empty_decision = ExpertTriggerEvaluator(system.settings.expert.triggers).evaluate(
        empty_packet
    )
    with pytest.raises(
        ExpertProposalContractError,
        match="cannot cross generic persistence",
    ):
        system.architect.propose(
            packet=empty_packet,
            decision=empty_decision,
            materialized_source_base=None,
        )
    ordinary_empty_packet = _remint(
        empty_packet,
        recovery_barrier_basis_packet_id=None,
    )
    ordinary_empty_decision = ExpertTriggerEvaluator(
        system.settings.expert.triggers
    ).evaluate(ordinary_empty_packet)
    with pytest.raises(
        ExpertProposalContractError,
        match="cannot be used as an ordinary ancestor",
    ):
        system.architect.propose(
            packet=ordinary_empty_packet,
            decision=ordinary_empty_decision,
            materialized_source_base=None,
            ancestor_candidate_ids=(closure.manifest.candidate_id,),
        )
    with pytest.raises(
        ExpertProposalContractError,
        match="exact coordinator authority",
    ):
        system.architect.engine._propose_recovery_bootstrap(
            authority=object(),
            packet=empty_packet,
            decision=empty_decision,
            prior_knowledge=None,
        )


def test_empty_recovery_packet_removes_blocked_source_observations():
    barrier_packet, _materialized, _contents = released_observation_packet(
        ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        "Blocked topology must not enter an empty recovery prompt.",
    )

    projected = project_canonical_empty_recovery_packet(barrier_packet)

    assert projected.trigger_observations == ()
    assert projected.source_base_release is None
    assert projected.source_base_repository_map is None
    assert projected.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST
    assert projected.episodes == barrier_packet.episodes
    assert projected.claims == barrier_packet.claims
    assert projected.proof_reference_ids == barrier_packet.proof_reference_ids


def test_empty_recovery_rejects_historical_selection_and_final_current_movement(
    tmp_path,
):
    historical = _historical_candidate_system(tmp_path / "historical")
    with pytest.raises(
        ExpertRecoveryCandidateCoordinatorError,
        match="historical exhaustion",
    ):
        historical.coordinator.bootstrap_empty(
            scope_contract=historical.fixture.case.scope,
            replay_basis_packet=historical.replay_basis,
        )

    moving = _historical_candidate_system(
        tmp_path / "moving",
        empty_selection=True,
        current_move_after=2,
    )
    with pytest.raises(
        ExpertRecoveryBaseError,
        match="selection became stale",
    ):
        moving.coordinator.bootstrap_empty(
            scope_contract=moving.fixture.case.scope,
            replay_basis_packet=moving.replay_basis,
        )
    assert not tuple(moving.candidate_store.object_root.iterdir())


def test_recovery_coordinator_rejects_mismatched_sanitation_policy(tmp_path):
    with pytest.raises(
        ExpertRecoveryCandidateCoordinatorError,
        match="exact production components",
    ):
        _historical_candidate_system(
            tmp_path,
            sanitation_mismatch=True,
        )


def test_empty_recovery_rejects_scientifically_consumed_barrier_before_agent_call(
    tmp_path,
):
    _, _, episode, _, _, _, _ = source_fixture()
    system = _historical_candidate_system(
        tmp_path,
        empty_selection=True,
        episodes=(episode,),
        taint_episodes_with_barrier=True,
    )

    with pytest.raises(
        ExpertRecoveryCandidateCoordinatorError,
        match="scientifically consumed",
    ):
        system.coordinator.bootstrap_empty(
            scope_contract=system.fixture.case.scope,
            replay_basis_packet=system.replay_basis,
        )
    assert system.runner.calls == []
    assert not tuple(system.candidate_store.object_root.iterdir())


def test_recovery_candidate_final_admission_rejects_current_movement(tmp_path):
    system = _historical_candidate_system(
        tmp_path,
        current_move_after=2,
    )

    with pytest.raises(
        ExpertRecoveryBaseError,
        match="selection became stale",
    ):
        system.coordinator.restore_historical(
            scope_contract=system.fixture.case.scope,
            replay_basis_packet=system.replay_basis,
        )

    assert not tuple(system.candidate_store.object_root.iterdir())


@pytest.mark.parametrize("failure", ("depth", "current_movement", "clear_current"))
def test_recovery_fails_loud_without_complete_stable_barrier_authority(failure):
    fixture = _fixture(
        lineage_limit=2 if failure == "depth" else None,
        current_move_after=1 if failure == "current_movement" else None,
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
