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
    ExpertBaseReleaseManifest,
    ExpertPromotionState,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.publisher import (
    ExpertReleasePublicationGate,
    ExpertReleasePublicationError,
    ExpertReleasePublisher,
)
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleaseActivationReceipt,
    ExpertReleaseContractError,
)
from kapso.cross_run.expert.revocation import (
    ExpertReleaseRevocationCoordinator,
    ExpertReleaseRevocationError,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStoreError
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.git_refs import git_object_sha
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    PublicationTelemetry,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    CurrentPointerState,
    GitHubArtifactActivationWitness,
    GitHubArtifactResolver,
    PublicationSourceFile,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings
from test_expert_release_assembly import (
    _approved_bootstrap,
    _approved_normal,
    _publication_intent,
    _publication_publisher,
    _publication_pointer,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def test_late_release_use_match_blocks_before_first_reservation(
    tmp_path,
    monkeypatch,
):
    validation_store, approval, authority, predecessor_pointer = _approved_normal(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=frozen_current,
        current_pointer=predecessor_pointer,
    )
    authority.release_use_policy.denied = True

    with pytest.raises(
        ExpertReleasePublicationError,
        match="blocked by current release-use policy",
    ):
        publisher.reserve(
            candidate_id=approval.snapshot.state.candidate_id,
            committed_at="2026-07-23T12:00:00Z",
        )

    blocked = validation_store.reopen_release_use_block(
        approval.snapshot.state.candidate_id
    )
    assert blocked is not None
    assert blocked.snapshot.state.promotion_state is (
        ExpertPromotionState.RELEASE_USE_BLOCKED
    )
    assert (
        validation_store.reopen_release_publication(
            approval.snapshot.state.candidate_id
        )
        is None
    )


def test_late_release_use_block_retains_intent_for_recovery_only(
    tmp_path,
    monkeypatch,
):
    validation_store, approval, authority, predecessor_pointer = _approved_normal(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=frozen_current,
        current_pointer=predecessor_pointer,
    )
    candidate_id = approval.snapshot.state.candidate_id
    reserved = publisher.reserve(
        candidate_id=candidate_id,
        committed_at="2026-07-23T12:00:00Z",
    ).reservation
    authority.release_use_policy.denied = True

    with pytest.raises(
        ExpertReleasePublicationError,
        match="blocked by current release-use policy",
    ):
        publisher.reserve(
            candidate_id=candidate_id,
            committed_at="2026-07-23T12:00:00Z",
        )

    retained = validation_store.reopen_release_publication(candidate_id)
    assert retained == reserved
    assert validation_store.snapshot(candidate_id).state.promotion_state is (
        ExpertPromotionState.RELEASE_USE_BLOCKED
    )
    publisher.resolver.read_artifact_intent = (
        lambda scope_id, artifact_kind, artifact_id: None
    )
    publisher.resolver.read_artifact_pointer = (
        lambda scope_id, artifact_kind, artifact_id: None
    )
    with pytest.raises(
        ExpertReleasePublicationError,
        match="blocked and has no witnessed activation",
    ):
        publisher.publish(
            candidate_id=candidate_id,
            committed_at="2026-07-23T12:00:00Z",
        )
    assert validation_store.reopen_release_publication(candidate_id) == reserved


def test_witnessed_remote_activation_recovers_from_late_block(
    tmp_path,
    monkeypatch,
):
    validation_store, approval, authority, predecessor_pointer = _approved_normal(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=frozen_current,
        current_pointer=predecessor_pointer,
    )
    candidate_id = approval.snapshot.state.candidate_id
    package = assembler.build(candidate_id=candidate_id)
    reservation = publisher.reserve(
        candidate_id=candidate_id,
        committed_at="2026-07-23T12:00:00Z",
    ).reservation
    plan = reservation.plan
    intent = _publication_intent(
        package,
        plan,
        reservation.intent.committed_at,
        "b" * 40,
    )
    predecessor_payload = predecessor_pointer.to_json_bytes()
    intent = replace(
        intent,
        preserved_current=PublicationSourceFile(
            relative_path="CURRENT.json",
            mode="100644",
            size=len(predecessor_payload),
            sha256=tree_or_blob_digest(predecessor_payload),
            git_blob_sha=git_object_sha("blob", predecessor_payload),
        ),
    )
    pointer = _publication_pointer(plan, intent)
    witness = GitHubArtifactActivationWitness.mint(
        scope_id=plan.scope_id,
        scope_repository_binding_hash=(
            settings.scopes.resolve(plan.scope_id).binding_fingerprint
        ),
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        activation_commit_sha="c" * 40,
        publication_intent_digest=intent.digest,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
    )
    observed = TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=plan.scope_id,
        release_id=plan.release_id,
        publication_id=pointer.publication_record.publication_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        repository_node_id=plan.current_release_observation.repository_node_id,
        default_branch_head_commit_sha=witness.activation_commit_sha,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
        validation_closure_ids=plan.validation_closure_ids,
    )
    authority.release_use_policy.denied = True
    with pytest.raises(ExpertReleasePublicationError, match="blocked"):
        publisher.reserve(
            candidate_id=candidate_id,
            committed_at=reservation.intent.committed_at,
        )
    blocked = validation_store.reopen_release_use_block(candidate_id)
    assert blocked is not None

    publisher.resolver.read_artifact_intent = lambda *_arguments: intent
    publisher.resolver.read_artifact_pointer = lambda *_arguments: pointer
    publisher.resolver.resolve_artifact = lambda *_arguments: SimpleNamespace(
        pointer=pointer
    )
    publisher.resolver.resolve_artifact_activation_preparation = (
        lambda *_arguments, **_keywords: witness.activation_commit_sha
    )
    publisher.resolver.resolve_artifact_activation_witness = (
        lambda *_arguments, **_keywords: witness
    )
    publisher.resolver.require_artifact_intent = lambda *_arguments: None
    publisher.resolver.require_artifact_pointer = lambda *_arguments: None
    publisher.current_release_authority.observe_task_evaluation_current = (
        lambda _scope_id: observed
    )
    release_use_reads = authority.calls.count("release-use")
    generic_publication_calls = []
    monkeypatch.setattr(
        AutonomousGitHubPublisher,
        "publish",
        lambda *_arguments, **_keywords: generic_publication_calls.append(True),
    )
    publication = publisher.publish(
        candidate_id=candidate_id,
        committed_at=reservation.intent.committed_at,
    )
    activation = publication.activation

    assert publication.telemetry is None
    assert generic_publication_calls == []
    assert authority.calls.count("release-use") == release_use_reads
    assert activation.snapshot.state.promotion_state is ExpertPromotionState.RELEASED
    assert blocked.decision.release_use_decision_id in (
        activation.snapshot.state.terminal_evidence_ids
    )
    assert validation_store.reopen_release_publication(candidate_id) is None


def _release_pointer(plan, materialized_tree_digest):
    assets = tuple(
        GitHubReleaseAsset(
            asset_id=f"asset-{position}",
            name=asset.name,
            media_type=asset.media_type,
            size=asset.size,
            sha256=asset.sha256,
        )
        for position, asset in enumerate(plan.assets, start=1)
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=plan.release_id,
        repository_node_id=plan.current_release_observation.repository_node_id,
        repository_full_name=plan.current_release_observation.repository_full_name,
        commit_sha="b" * 40,
        immutable_release_id="7",
        tag=plan.tag,
        assets=assets,
        release_attestation_ref="expert-release-attestation",
        published_at="2026-07-21T12:00:00Z",
        publisher_identity="leeroo-coder",
    )
    return CurrentArtifactPointer(
        scope_id=plan.scope_id,
        publication_record=publication,
        publication_intent_digest=tree_or_blob_digest(b"publication-intent"),
        source_tree_digest=plan.publication_source_tree_digest,
        source_git_tree_sha="d" * 40,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
        manifest_digest=plan.manifest_digest,
        validation_closure_ids=plan.validation_closure_ids,
    )


def _source_file_projection(source_tree: Path):
    return {
        path.relative_to(source_tree).as_posix(): (
            tree_or_blob_digest(path.read_bytes()),
            "100755" if path.stat().st_mode & 0o111 else "100644",
        )
        for path in sorted(source_tree.rglob("*"))
        if path.is_file()
    }


def test_publish_derives_and_freezes_a_missing_reservation(
    tmp_path,
    monkeypatch,
):
    validation_store, _matrix, approval, authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=frozen_current,
        current_pointer=None,
    )
    publisher.resolver.read_artifact_intent = (
        lambda scope_id, artifact_kind, artifact_id: None
    )
    publisher.resolver.read_artifact_pointer = (
        lambda scope_id, artifact_kind, artifact_id: None
    )

    def stop_before_remote_write(*_arguments, **_keywords):
        assert (
            validation_store.reopen_release_publication(
                approval.snapshot.state.candidate_id
            )
            is not None
        )
        raise ExpertReleasePublicationError("stop before remote write")

    monkeypatch.setattr(
        AutonomousGitHubPublisher,
        "publish",
        stop_before_remote_write,
    )
    with pytest.raises(
        ExpertReleasePublicationError,
        match="stop before remote write",
    ):
        publisher.publish(
            candidate_id=approval.snapshot.state.candidate_id,
            committed_at="2026-07-21T12:00:00Z",
        )

    reservation = validation_store.reopen_release_publication(
        approval.snapshot.state.candidate_id
    )
    assert reservation is not None
    assert reservation.plan.lineage == reservation.manifest.lineage
    assert reservation.intent.committed_at == "2026-07-21T12:00:00Z"


@pytest.mark.parametrize(
    ("crash_before_local_commit", "successor_wins_after_activation"),
    ((False, False), (True, True)),
)
def test_expert_publisher_derives_package_and_recovers_activation(
    tmp_path,
    monkeypatch,
    crash_before_local_commit,
    successor_wins_after_activation,
):
    validation_store, _matrix, approval, authority = _approved_bootstrap(
        tmp_path,
        monkeypatch,
    )
    candidate_store = validation_store.reducer.candidate_store
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=candidate_store,
        validation_store=validation_store,
        expert_settings=candidate_store.validator.settings,
        github_settings=settings.github,
    )
    package = assembler.build(candidate_id=approval.snapshot.state.candidate_id)
    frozen_current = (
        approval.stage_result.publication_authority_fence.current_release_observation
    )
    github_client = object()
    resolver = GitHubArtifactResolver(
        github_client,
        settings.github,
        settings.scopes,
    )
    materializer = GitHubArtifactMaterializer(
        github_client,
        settings.github,
        tmp_path / "publisher-state",
    )
    generic_publisher = AutonomousGitHubPublisher(
        github_client,
        resolver,
        materializer,
        settings.github,
    )
    current_authority = GitHubExpertCurrentReleaseProvider(resolver)
    validation_store.reducer.current_release_provider = current_authority
    publisher = ExpertReleasePublisher(
        assembler=assembler,
        validation_store=validation_store,
        github_publisher=generic_publisher,
        resolver=resolver,
        current_release_authority=current_authority,
        task_adapter_authority=validation_store.reducer.task_adapter_provider,
        security_denylist_authority=authority.denylist,
        release_use_policy_authority=authority.release_use_policy,
    )
    current_state = CurrentPointerState(
        pointer=None,
        head_commit_sha=frozen_current.default_branch_head_commit_sha,
    )
    resolver.read_current_pointer_state = (
        lambda scope_id, artifact_kind, allow_missing: current_state
    )
    remote = {"observation": frozen_current}
    monkeypatch.setattr(
        GitHubExpertCurrentReleaseProvider,
        "observe_task_evaluation_current",
        lambda self, scope_id: remote["observation"],
    )
    reservation = publisher.reserve(
        candidate_id=approval.snapshot.state.candidate_id,
        committed_at="2026-07-21T12:00:00Z",
    ).reservation
    plan = reservation.plan
    captured = {}
    activation_trace = {"active": False, "events": []}
    observe_current = (
        publisher.current_release_authority.observe_task_evaluation_current
    )
    observe_denylist = publisher.security_denylist_authority.observe_exact
    observe_release_use = publisher.release_use_policy_authority.observe_exact
    reopen_publication = validation_store.reopen_release_publication

    def traced_current(scope_id):
        if activation_trace["active"]:
            activation_trace["events"].append("current")
        return observe_current(scope_id)

    def traced_denylist(**keywords):
        if activation_trace["active"]:
            activation_trace["events"].append("denylist")
        return observe_denylist(**keywords)

    def traced_release_use(**keywords):
        if activation_trace["active"]:
            activation_trace["events"].append("release-use")
        return observe_release_use(**keywords)

    def traced_reopen_publication(candidate_id):
        if activation_trace["active"]:
            activation_trace["events"].append("reservation")
        return reopen_publication(candidate_id)

    monkeypatch.setattr(
        publisher.current_release_authority,
        "observe_task_evaluation_current",
        traced_current,
    )
    monkeypatch.setattr(
        publisher.security_denylist_authority,
        "observe_exact",
        traced_denylist,
    )
    monkeypatch.setattr(
        publisher.release_use_policy_authority,
        "observe_exact",
        traced_release_use,
    )
    monkeypatch.setattr(
        validation_store,
        "reopen_release_publication",
        traced_reopen_publication,
    )

    def publish(generic, envelope, *, activation_authorization=None):
        captured["publish_calls"] = captured.get("publish_calls", 0) + 1
        manifest_payload = (
            envelope.source_tree / envelope.manifest_relative_path
        ).read_bytes()
        manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_payload)
        source_files = _source_file_projection(envelope.source_tree)
        materialized_digest = generic.package_validator.validate_local_package(
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=tree_or_blob_digest(manifest_payload),
            assets=envelope.assets,
            source_files=source_files,
        )
        gate = activation_authorization.verifier_for(generic, envelope)
        gate.validate_before_publication(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            current_state=current_state,
            manifest=manifest,
            source_tree_digest=source_tree_digest(
                {
                    relative_path: (
                        digest,
                        mode,
                        (envelope.source_tree / relative_path).stat().st_size,
                    )
                    for relative_path, (digest, mode) in source_files.items()
                }
            ),
            manifest_digest=tree_or_blob_digest(manifest_payload),
        )
        intent = _publication_intent(
            package,
            plan,
            reservation.intent.committed_at,
            "b" * 40,
        )
        pointer = _publication_pointer(plan, intent)
        activation_trace["active"] = True
        activation_trace["events"].append("artifact-pointer")
        generic.resolver.require_artifact_pointer(
            plan.scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
            plan.release_id,
            pointer,
        )
        gate.revalidate_before_activation(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            pointer=pointer,
            manifest=manifest,
        )
        activation_trace["events"].append("cas")
        activation_trace["active"] = False
        captured.update(
            {
                "envelope": envelope,
                "gate": gate,
                "manifest": manifest,
                "pointer": pointer,
                "intent": intent,
                "source_digest": plan.publication_source_tree_digest,
            }
        )
        observed_release_id = plan.release_id
        observed_publication_id = pointer.publication_record.publication_id
        observed_head = "c" * 40
        observed_digest = tree_or_blob_digest(pointer.to_json_bytes())
        if successor_wins_after_activation:
            observed_release_id = content_id(
                "expert-base-release", {"successor": plan.release_id}
            )
            observed_publication_id = content_id(
                "github-publication", {"successor": plan.release_id}
            )
            observed_head = "d" * 40
            observed_digest = tree_or_blob_digest(b"successor pointer")
        remote["observation"] = TaskEvaluationCurrentReleaseObservation.mint(
            scope_id=plan.scope_id,
            release_id=observed_release_id,
            publication_id=observed_publication_id,
            repository_full_name=plan.current_release_observation.repository_full_name,
            repository_node_id=plan.current_release_observation.repository_node_id,
            default_branch_head_commit_sha=observed_head,
            current_pointer_digest=observed_digest,
            validation_closure_ids=plan.validation_closure_ids,
        )
        return PublicationTelemetry(
            publication_record=pointer.publication_record,
            expected_parent_sha=envelope.expected_parent_sha,
            source_commit_sha=pointer.publication_record.commit_sha,
            pointer_commit_sha="c" * 40,
            source_tree_digest=plan.publication_source_tree_digest,
            validation_closure_ids=plan.validation_closure_ids,
            idempotent_replay=False,
        )

    monkeypatch.setattr(AutonomousGitHubPublisher, "publish", publish)
    resolver.read_artifact_intent = (
        lambda scope_id, artifact_kind, artifact_id: captured.get("intent")
    )
    resolver.read_artifact_pointer = (
        lambda scope_id, artifact_kind, artifact_id: captured.get("pointer")
    )
    resolver.resolve_artifact = (
        lambda scope_id, artifact_kind, artifact_id: SimpleNamespace(
            pointer=captured["pointer"]
        )
    )
    resolver.resolve_artifact_activation_preparation = (
        lambda scope_id, artifact_kind, artifact_id, intent, pointer, allow_missing=False: "c"
        * 40
    )

    def resolve_activation_witness(
        scope_id,
        artifact_kind,
        artifact_id,
        intent,
        pointer,
        allow_missing=False,
    ):
        return GitHubArtifactActivationWitness.mint(
            scope_id=plan.scope_id,
            scope_repository_binding_hash=(
                settings.scopes.resolve(plan.scope_id).binding_fingerprint
            ),
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            artifact_id=plan.release_id,
            repository_full_name=(
                plan.current_release_observation.repository_full_name
            ),
            activation_commit_sha="c" * 40,
            publication_intent_digest=intent.digest,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
        )

    resolver.resolve_artifact_activation_witness = resolve_activation_witness
    resolver.require_artifact_intent = lambda *args: None
    resolver.require_artifact_pointer = lambda *args: None

    if crash_before_local_commit:
        commit_release_activation = validation_store.commit_release_activation
        crashed_permits = []

        def simulate_crash_after_remote_activation(activation_permit):
            crashed_permits.append(activation_permit)
            raise RuntimeError("simulated crash after remote activation")

        monkeypatch.setattr(
            validation_store,
            "commit_release_activation",
            simulate_crash_after_remote_activation,
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            publisher.publish(
                candidate_id=plan.candidate_id,
                committed_at=reservation.intent.committed_at,
            )
        monkeypatch.setattr(
            validation_store,
            "commit_release_activation",
            commit_release_activation,
        )
        recovered_publication = publisher.publish(
            candidate_id=plan.candidate_id,
            committed_at=reservation.intent.committed_at,
        )
        assert recovered_publication.telemetry is None
        activation = recovered_publication.activation
        competing_replay = validation_store.commit_release_activation(
            crashed_permits[0]
        )
        assert competing_replay.replayed is True
        assert competing_replay.receipt == activation.receipt
        result = None
    else:
        result = publisher.publish(
            candidate_id=plan.candidate_id,
            committed_at=reservation.intent.committed_at,
        )
        activation = result.activation

    assert activation.receipt.publication_intent_id == (
        reservation.intent.publication_intent_id
    )
    assert activation.snapshot.state.promotion_state.value == "released"
    assert activation.receipt.observed_current_release == remote["observation"]
    assert activation.receipt.release_id in (activation.receipt.consumed_dependency_ids)
    assert activation.receipt.observed_current_release.observation_id in (
        activation.receipt.control_dependency_ids
    )
    reclassified_dependency = activation.receipt.control_dependency_ids[0]
    with pytest.raises(
        ExpertReleaseContractError,
        match="control dependencies|categorized",
    ):
        ExpertReleaseActivationReceipt.mint(
            publication_intent_id=activation.receipt.publication_intent_id,
            publication_plan_id=activation.receipt.publication_plan_id,
            release_id=activation.receipt.release_id,
            candidate_id=activation.receipt.candidate_id,
            approval_transition_id=activation.receipt.approval_transition_id,
            approval_state_id=activation.receipt.approval_state_id,
            planned_current_observation_id=(
                activation.receipt.planned_current_observation_id
            ),
            github_publication_intent=activation.receipt.github_publication_intent,
            github_publication_pointer=activation.receipt.github_publication_pointer,
            activation_witness=activation.receipt.activation_witness,
            observed_current_release=activation.receipt.observed_current_release,
            consumed_dependency_ids=tuple(
                sorted(
                    {
                        *activation.receipt.consumed_dependency_ids,
                        reclassified_dependency,
                    }
                )
            ),
            control_dependency_ids=tuple(
                dependency_id
                for dependency_id in activation.receipt.control_dependency_ids
                if dependency_id != reclassified_dependency
            ),
        )
    assert captured["publish_calls"] == 1
    if result is not None:
        assert result.telemetry.publication_record.artifact_id == plan.release_id
    assert captured["gate"].preflight_mode == "activation-predecessor"
    assert tuple(asset.name for asset in captured["envelope"].assets) == tuple(
        asset.name for asset in plan.assets
    )
    assert authority.calls.count("denylist") == 3
    assert activation_trace["events"][0] == "artifact-pointer"
    assert activation_trace["events"][-4:] == [
        "current",
        "release-use",
        "reservation",
        "cas",
    ]
    assert (
        activation_trace["events"].index("denylist")
        < len(activation_trace["events"]) - 4
    )
    assert captured["pointer"].publication_record.publication_id in (
        authority.denylist.checked_subject_ids
    )
    replayed = publisher.publish(
        candidate_id=plan.candidate_id,
        committed_at=reservation.intent.committed_at,
    )
    assert replayed.telemetry is None
    assert replayed.activation.replayed is True
    assert replayed.activation.receipt == activation.receipt
    assert validation_store.reopen_release_publication(plan.candidate_id) is None

    assert authority.calls.count("denylist") == 3
    revocation_coordinator = ExpertReleaseRevocationCoordinator(
        validation_store=validation_store,
        security_denylist_authority=authority.denylist,
    )
    with pytest.raises(ExpertReleaseRevocationError, match="emergency match"):
        revocation_coordinator.revoke(
            candidate_id=plan.candidate_id,
            revoked_at="2026-07-21T12:30:00Z",
        )
    assert validation_store.snapshot(plan.candidate_id).state.promotion_state.value == (
        "released"
    )
    authority.denylist.denied = True
    revocation_target = validation_store.reopen_release_revocation_target(
        plan.candidate_id
    )
    valid_observation = authority.denylist.observe_exact(
        scope_id=plan.scope_id,
        scope_contract_id=plan.scope_contract_id,
        checked_subject_ids=revocation_target.security_subject_ids,
    )
    wrong_repository_observation = SecurityDenylistObservation.mint(
        scope_id=valid_observation.scope_id,
        scope_contract_id=valid_observation.scope_contract_id,
        scope_repository_binding_hash=tree_or_blob_digest(b"wrong repository binding"),
        snapshot_id=valid_observation.snapshot_id,
        generation=valid_observation.generation,
        publication_id=valid_observation.publication_id,
        repository_full_name=valid_observation.repository_full_name,
        repository_node_id=valid_observation.repository_node_id,
        pointer_digest=valid_observation.pointer_digest,
        authority_commit_sha=valid_observation.authority_commit_sha,
        release_attestation_ref=valid_observation.release_attestation_ref,
        checked_subject_ids=valid_observation.checked_subject_ids,
        matched_revocations=valid_observation.matched_revocations,
    )
    with pytest.raises(ExpertValidationStoreError, match="exact emergency match"):
        validation_store._seal_release_revocation(
            coordinator=revocation_coordinator,
            target=revocation_target,
            security_denylist_observation=wrong_repository_observation,
            revoked_at="2026-07-21T12:45:00Z",
        )

    commit_release_revocation = validation_store.commit_release_revocation
    pending_revocation_permits = []

    def simulate_crash_before_revocation_commit(revocation_permit):
        pending_revocation_permits.append(revocation_permit)
        raise RuntimeError("simulated crash before revocation commit")

    monkeypatch.setattr(
        validation_store,
        "commit_release_revocation",
        simulate_crash_before_revocation_commit,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        revocation_coordinator.revoke(
            candidate_id=plan.candidate_id,
            revoked_at="2026-07-21T13:00:00Z",
        )
    monkeypatch.setattr(
        validation_store,
        "commit_release_revocation",
        commit_release_revocation,
    )
    revoked = revocation_coordinator.revoke(
        candidate_id=plan.candidate_id,
        revoked_at="2026-07-21T14:00:00Z",
    )
    competing_revocation = validation_store.commit_release_revocation(
        pending_revocation_permits[0]
    )

    assert revoked.replayed is False
    assert competing_revocation.replayed is True
    assert competing_revocation.receipt == revoked.receipt
    assert revoked.snapshot.state.promotion_state.value == "revoked"
    assert revoked.receipt.release_id == plan.release_id
    assert revoked.receipt.security_denylist_observation.matched_revocations
    checked_revocation_subjects = set(
        revoked.receipt.security_denylist_observation.checked_subject_ids
    )
    assert set(package.manifest.consumed_dependency_ids).issubset(
        checked_revocation_subjects
    )
    assert set(activation.receipt.consumed_dependency_ids).issubset(
        checked_revocation_subjects
    )
    assert set(activation.receipt.control_dependency_ids).isdisjoint(
        checked_revocation_subjects
    )
    assert activation.receipt.activation_receipt_id in (
        revoked.snapshot.state.terminal_evidence_ids
    )
    historical_activation = validation_store.reopen_release_activation(
        plan.candidate_id
    )
    assert historical_activation.snapshot.state.promotion_state.value == "released"
    denylist_call_count = authority.calls.count("denylist")
    offline_replay = revocation_coordinator.revoke(
        candidate_id=plan.candidate_id,
        revoked_at="2026-07-21T15:00:00Z",
    )
    assert offline_replay.replayed is True
    assert offline_replay.receipt == revoked.receipt
    assert authority.calls.count("denylist") == denylist_call_count
    with pytest.raises(ExpertReleasePublicationError, match="revoked"):
        publisher.publish(
            candidate_id=plan.candidate_id,
            committed_at=reservation.intent.committed_at,
        )

    with pytest.raises(ExpertReleasePublicationError, match="immutable"):
        publisher.security_denylist_authority = object()
    generic_publisher.resolver = object()
    with pytest.raises(ExpertReleasePublicationError, match="binding changed"):
        publisher.publish(
            candidate_id=plan.candidate_id,
            committed_at=reservation.intent.committed_at,
        )
    revocation_path = validation_store._object_path(
        revoked.receipt.revocation_receipt_id,
        create_namespace=False,
    )
    revocation_path.unlink()
    with pytest.raises(ExpertValidationStoreError, match="regular file"):
        validation_store.reopen_release_revocation(plan.candidate_id)
