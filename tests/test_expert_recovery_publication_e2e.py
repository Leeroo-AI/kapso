"""End-to-end clean-forward recovery publication and activation."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertAcceptedStageResultRef,
    ExpertBaseReleaseManifest,
    ExpertPromotionState,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityError,
)
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH,
    EXPERT_RELEASE_MATRIX_SUMMARY_PATH,
    ExpertReleaseAssembler,
    ExpertReleaseAssemblyError,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleaseEvidenceManifest,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from kapso.cross_run.git_refs import git_object_sha
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    PublicationTelemetry,
)
from kapso.cross_run.github.resolver import (
    CurrentPointerState,
    GitHubArtifactActivationWitness,
    PublicationSourceFile,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings
from security_denylist_fixtures import matched_security_revocations
from test_expert_publication_eligibility import _coordinator
from test_expert_recovery_release_matrix_e2e import (
    _canonical_empty_recovery_matrix_case,
    _historical_recovery_matrix_case,
)
from test_expert_release_assembly import (
    _publication_intent,
    _publication_pointer,
    _publication_publisher,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
CANONICAL_CROSS_RUN_SETTINGS = CrossRunSettings.from_dict(
    load_config(CANONICAL_CONFIG_PATH)["cross_run"]
)


class _RecoveryPublicationDenylistAuthority:
    def __init__(self, matched_subject_ids):
        self.matched_subject_ids = tuple(sorted(matched_subject_ids))
        self.checked_subject_ids = ()

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.checked_subject_ids = checked_subject_ids
        repositories = CANONICAL_CROSS_RUN_SETTINGS.scopes.resolve(scope_id)
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=repositories.binding_fingerprint,
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"matched": self.matched_subject_ids},
            ),
            generation=1,
            publication_id=content_id(
                "github-publication",
                {"security": self.matched_subject_ids},
            ),
            repository_full_name=repositories.security_repository,
            repository_node_id="security_repo_node",
            pointer_digest=tree_or_blob_digest(b"security CURRENT"),
            authority_commit_sha="b" * 40,
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=matched_security_revocations(self.matched_subject_ids),
        )


def _remint(record, **changes):
    values = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    values.update(changes)
    return type(record).mint(**values)


def _record_path(record_id):
    namespace, digest = record_id.split(":sha256:", 1)
    return f".kapso/expert/release-evidence/records/{namespace}/{digest}.json"


def _replace_ids(values, replacements):
    return tuple(sorted(replacements.get(value, value) for value in values))


def _approve_recovery(case, matched_control_subject_ids):
    authority = _coordinator(
        SimpleNamespace(
            validation_store=case.validation_store,
            matrix_commit=case.committed_stage,
            prepared=case.prepared_task,
        ),
        denylist=_RecoveryPublicationDenylistAuthority(matched_control_subject_ids),
    )
    approved = authority.coordinator.publish(
        candidate_id=case.committed_stage.snapshot.state.candidate_id,
        release_matrix_stage_result_id=(
            case.committed_stage.stage_result.stage_result_record_id
        ),
    )
    assert approved.snapshot.state.promotion_state is ExpertPromotionState.APPROVED
    return approved, authority


def _reserve_recovery(case, approved, authority, tmp_path):
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    assembler = ExpertReleaseAssembler(
        candidate_store=case.candidate_store,
        validation_store=case.validation_store,
        expert_settings=case.configured_expert_settings,
        github_settings=settings.github,
    )
    package = assembler.build(candidate_id=approved.snapshot.state.candidate_id)
    barrier_pointer = case.recovery.fixture.remotes[0].pointer
    frozen_current = (
        approved.stage_result.publication_authority_fence.current_release_observation
    )
    publisher = _publication_publisher(
        validation_store=case.validation_store,
        assembler=assembler,
        authority=authority,
        settings=settings,
        tmp_path=tmp_path,
        current_observation=frozen_current,
        current_pointer=barrier_pointer,
    )
    reserved = publisher.reserve(
        candidate_id=approved.snapshot.state.candidate_id,
        committed_at="2026-07-24T12:00:00Z",
    )
    return settings, assembler, package, publisher, reserved


def _tamper_nested_recovery_waiver(
    *,
    assembler,
    package,
    approved,
    extra_control_subject_id,
):
    publication = approved.stage_result
    fence = publication.publication_authority_fence
    assert fence is not None
    tampered_allowed_subject_ids = tuple(
        sorted(
            {
                *publication.allowed_control_security_subject_ids,
                extra_control_subject_id,
            }
        )
    )
    tampered_fence = _remint(
        fence,
        allowed_control_security_subject_ids=tampered_allowed_subject_ids,
    )
    tampered_publication = _remint(
        publication,
        allowed_control_security_subject_ids=tampered_allowed_subject_ids,
        publication_authority_fence=tampered_fence,
        exact_dependency_ids=_replace_ids(
            publication.exact_dependency_ids,
            {fence.fence_id: tampered_fence.fence_id},
        ),
    )
    old_state = approved.snapshot.state
    accepted_stage_results = tuple(
        (
            ExpertAcceptedStageResultRef(
                stage=result.stage,
                stage_result_record_id=tampered_publication.stage_result_record_id,
            )
            if result.stage_result_record_id == publication.stage_result_record_id
            else result
        )
        for result in old_state.accepted_stage_results
    )
    state_replacements = {
        publication.stage_result_record_id: (
            tampered_publication.stage_result_record_id
        ),
        fence.fence_id: tampered_fence.fence_id,
    }
    tampered_state = _remint(
        old_state,
        accepted_stage_results=accepted_stage_results,
        transition_evidence_id=state_replacements.get(
            old_state.transition_evidence_id,
            old_state.transition_evidence_id,
        ),
        terminal_evidence_ids=_replace_ids(
            old_state.terminal_evidence_ids,
            state_replacements,
        ),
    )
    old_transition = approved.snapshot.transition
    tampered_transition = _remint(
        old_transition,
        target_state_id=tampered_state.validation_state_id,
        accepted_stage_result_record_ids=_replace_ids(
            old_transition.accepted_stage_result_record_ids,
            state_replacements,
        ),
        transition_stage_result_record_id=state_replacements.get(
            old_transition.transition_stage_result_record_id,
            old_transition.transition_stage_result_record_id,
        ),
    )
    evidence_files = dict(package.evidence_files)
    for old_record, new_record in (
        (publication, tampered_publication),
        (old_state, tampered_state),
        (old_transition, tampered_transition),
    ):
        _payload, mode = evidence_files.pop(
            _record_path(getattr(old_record, old_record.IDENTITY_FIELD))
        )
        evidence_files[_record_path(getattr(new_record, new_record.IDENTITY_FIELD))] = (
            new_record.to_json_bytes(),
            mode,
        )
    evidence_files.pop(EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH)
    record_ids = tuple(
        sorted(
            f"{path.split('/')[-2]}:sha256:{path.split('/')[-1].removesuffix('.json')}"
            for path in evidence_files
            if path != EXPERT_RELEASE_MATRIX_SUMMARY_PATH
        )
    )
    record_checksums = {
        path: tree_or_blob_digest(payload)
        for path, (payload, _mode) in evidence_files.items()
    }
    old_evidence = package.evidence_manifest
    evidence_id_replacements = {
        publication.stage_result_record_id: tampered_publication.stage_result_record_id,
        old_state.validation_state_id: tampered_state.validation_state_id,
        old_transition.transition_id: tampered_transition.transition_id,
    }
    tampered_evidence = _remint(
        old_evidence,
        approval_transition_id=tampered_transition.transition_id,
        approval_state_id=tampered_state.validation_state_id,
        publication_eligibility_result_id=(tampered_publication.stage_result_record_id),
        record_ids=record_ids,
        record_checksums=record_checksums,
        exact_dependency_ids=_replace_ids(
            old_evidence.exact_dependency_ids,
            evidence_id_replacements,
        ),
    )
    evidence_files[EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH] = (
        tampered_evidence.to_json_bytes(),
        package.evidence_files[EXPERT_RELEASE_EVIDENCE_MANIFEST_PATH][1],
    )
    evidence_files = dict(sorted(evidence_files.items()))
    evidence_archive = assembler._archive(evidence_files)
    evidence_dependency_ids = assembler._evidence_dependency_ids(evidence_files)
    old_manifest = package.manifest
    direct_dependencies = {
        old_manifest.scope_contract_id,
        old_manifest.candidate_id,
        old_manifest.candidate_commit_record_id,
        old_manifest.candidate_tree_ref,
        old_manifest.candidate_derivation_ref,
        old_manifest.candidate_validation_context_ref,
        old_manifest.candidate_patch_ref,
        old_manifest.candidate_sanitation_report_id,
        *old_manifest.candidate_ancestor_ids,
        *old_manifest.candidate_source_dependency_ids,
        *old_manifest.candidate_consumed_expert_release_ids,
        old_manifest.repository_map_ref,
        *old_manifest.module_contract_refs,
        old_manifest.validation_attempt_id,
        tampered_transition.transition_id,
        tampered_state.validation_state_id,
        tampered_publication.stage_result_record_id,
        old_manifest.release_matrix_stage_result_id,
        old_manifest.release_matrix_report_id,
        old_manifest.promotion_decision_id,
        *old_manifest.approval_assertion_ids,
        old_manifest.validation_policy_id,
        tampered_evidence.evidence_manifest_id,
        old_manifest.test_matrix_summary_ref,
    }
    if old_manifest.lineage.source_base_release_id is not None:
        direct_dependencies.add(old_manifest.lineage.source_base_release_id)
    if old_manifest.lineage.activation_predecessor_release_id is not None:
        direct_dependencies.add(old_manifest.lineage.activation_predecessor_release_id)
    dependency_universe = {*direct_dependencies, *evidence_dependency_ids}
    tampered_manifest = _remint(
        old_manifest,
        approval_transition_id=tampered_transition.transition_id,
        approval_state_id=tampered_state.validation_state_id,
        publication_eligibility_result_id=(tampered_publication.stage_result_record_id),
        evidence_manifest_ref=tampered_evidence.evidence_manifest_id,
        evidence_dependency_ids=evidence_dependency_ids,
        consumed_dependency_ids=tuple(
            sorted(dependency_universe - set(old_manifest.control_dependency_ids))
        ),
        checksums={
            **{
                path: tree_or_blob_digest(payload)
                for path, (payload, _mode) in package.source_files.items()
            },
            **{
                path: tree_or_blob_digest(payload)
                for path, (payload, _mode) in evidence_files.items()
            },
            old_manifest.source_archive_ref: tree_or_blob_digest(
                package.source_archive
            ),
            old_manifest.evidence_archive_ref: tree_or_blob_digest(evidence_archive),
        },
    )
    return replace(
        package,
        manifest=tampered_manifest,
        evidence_manifest=tampered_evidence,
        evidence_files=evidence_files,
        evidence_archive=evidence_archive,
        control_archive=assembler._archive(
            {
                ".kapso/expert/release.json": (
                    tampered_manifest.to_json_bytes(),
                    "100644",
                )
            }
        ),
    )


def _activate_recovery(
    *,
    case,
    settings,
    package,
    publisher,
    reserved,
    monkeypatch,
):
    reservation = reserved.reservation
    plan = reservation.plan
    barrier_pointer = case.recovery.fixture.remotes[0].pointer
    current_state = CurrentPointerState(
        pointer=barrier_pointer,
        head_commit_sha=plan.current_release_observation.default_branch_head_commit_sha,
    )
    remote = {"observation": plan.current_release_observation}
    captured = {}
    publisher.resolver.read_current_pointer_state = (
        lambda _scope_id, _artifact_kind, allow_missing: current_state
    )
    publisher.current_release_authority.observe_task_evaluation_current = (
        lambda _scope_id: remote["observation"]
    )

    def publish(generic, envelope, *, activation_authorization=None):
        manifest_payload = (
            envelope.source_tree / envelope.manifest_relative_path
        ).read_bytes()
        manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_payload)
        gate = activation_authorization.verifier_for(generic, envelope)
        gate.validate_before_publication(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            current_state=current_state,
            manifest=manifest,
            source_tree_digest=source_tree_digest(
                {
                    path: (
                        tree_or_blob_digest(payload),
                        mode,
                        len(payload),
                    )
                    for path, (payload, mode) in package.publication_files.items()
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
        predecessor_payload = barrier_pointer.to_json_bytes()
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
        gate.revalidate_before_activation(
            envelope=envelope,
            repositories=settings.scopes.resolve(plan.scope_id),
            pointer=pointer,
            manifest=manifest,
        )
        captured["intent"] = intent
        captured["pointer"] = pointer
        remote["observation"] = TaskEvaluationCurrentReleaseObservation.mint(
            scope_id=plan.scope_id,
            release_id=plan.release_id,
            publication_id=pointer.publication_record.publication_id,
            repository_full_name=(
                plan.current_release_observation.repository_full_name
            ),
            repository_node_id=plan.current_release_observation.repository_node_id,
            default_branch_head_commit_sha="c" * 40,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
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
    publisher.resolver.read_artifact_intent = (
        lambda _scope_id, _artifact_kind, _artifact_id: captured.get("intent")
    )
    publisher.resolver.read_artifact_pointer = (
        lambda _scope_id, _artifact_kind, _artifact_id: captured.get("pointer")
    )
    publisher.resolver.resolve_artifact = (
        lambda _scope_id, _artifact_kind, _artifact_id: SimpleNamespace(
            pointer=captured["pointer"]
        )
    )
    publisher.resolver.resolve_artifact_activation_preparation = (
        lambda _scope_id, _artifact_kind, _artifact_id, _intent, _pointer, allow_missing=False: "c"
        * 40
    )

    def activation_witness(
        _scope_id,
        _artifact_kind,
        _artifact_id,
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

    publisher.resolver.resolve_artifact_activation_witness = activation_witness
    publisher.resolver.require_artifact_intent = lambda *_arguments: None
    publisher.resolver.require_artifact_pointer = lambda *_arguments: None
    return publisher.publish(
        candidate_id=plan.candidate_id,
        committed_at=reservation.intent.committed_at,
    )


def test_historical_recovery_preserves_scientific_source_and_control_barrier(
    tmp_path,
    monkeypatch,
):
    case = _historical_recovery_matrix_case(tmp_path, monkeypatch)
    barrier_id = case.recovery.barrier.release_id
    selected_id = case.recovery.selected.release_id
    approved, authority = _approve_recovery(case, (barrier_id,))
    settings, assembler, package, publisher, reserved = _reserve_recovery(
        case,
        approved,
        authority,
        tmp_path,
    )
    manifest = package.manifest
    plan = reserved.reservation.plan

    assert manifest.lineage.source_base_release_id == selected_id
    assert manifest.lineage.activation_predecessor_release_id == barrier_id
    assert manifest.control_dependency_ids == case.admission.control_dependency_ids
    assert set(case.admission.scientific_dependency_ids).issubset(
        manifest.consumed_dependency_ids
    )
    assert selected_id in manifest.consumed_dependency_ids
    assert selected_id not in manifest.control_dependency_ids
    assert barrier_id in manifest.control_dependency_ids
    assert barrier_id not in manifest.consumed_dependency_ids
    assert plan.lineage == manifest.lineage
    assert plan.current_release_observation.release_id == barrier_id
    assert (
        plan.activation_predecessor_pointer == case.recovery.fixture.remotes[0].pointer
    )
    assert plan.generation == 1
    assert plan.manifest_control_dependency_ids == (
        case.admission.control_dependency_ids
    )
    reopened = ExpertValidationStore(
        case.validation_store.root,
        case.validation_store.state_root,
        case.validation_store.settings,
        case.validation_store.reducer,
    ).reopen_release_publication(plan.candidate_id)
    assert reopened == reserved.reservation

    tampered = _tamper_nested_recovery_waiver(
        assembler=assembler,
        package=package,
        approved=approved,
        extra_control_subject_id=case.admission.recovery_plan.recovery_plan_id,
    )
    with pytest.raises(
        ExpertReleaseAssemblyError,
        match="differs from deterministic assembly",
    ):
        assembler.verify(tampered)

    publication = _activate_recovery(
        case=case,
        settings=settings,
        package=package,
        publisher=publisher,
        reserved=reserved,
        monkeypatch=monkeypatch,
    )
    receipt = publication.activation.receipt
    assert publication.activation.snapshot.state.promotion_state is (
        ExpertPromotionState.RELEASED
    )
    assert set(case.admission.control_dependency_ids).issubset(
        receipt.control_dependency_ids
    )
    assert selected_id in receipt.consumed_dependency_ids
    assert barrier_id in receipt.control_dependency_ids
    assert barrier_id not in receipt.consumed_dependency_ids
    reopened_activation = case.validation_store.reopen_release_activation(
        plan.candidate_id
    )
    assert reopened_activation.receipt == receipt
    assert reopened_activation.snapshot == publication.activation.snapshot
    assert reopened_activation.replayed is True
    assert case.validation_store.reopen_release_publication(plan.candidate_id) is None
    replayed = publisher.publish(
        candidate_id=plan.candidate_id,
        committed_at=reserved.reservation.intent.committed_at,
    )
    assert replayed.telemetry is None
    assert replayed.activation.replayed is True
    assert replayed.activation.receipt == receipt


def test_canonical_empty_recovery_publishes_successor_without_scientific_source(
    tmp_path,
    monkeypatch,
):
    case = _canonical_empty_recovery_matrix_case(tmp_path, monkeypatch)
    barrier_id = case.recovery.barrier.release_id
    approved, authority = _approve_recovery(case, (barrier_id,))
    settings, _assembler, package, publisher, reserved = _reserve_recovery(
        case,
        approved,
        authority,
        tmp_path,
    )
    manifest = package.manifest
    plan = reserved.reservation.plan

    assert manifest.lineage.source_base_release_id is None
    assert manifest.lineage.activation_predecessor_release_id == barrier_id
    assert manifest.control_dependency_ids == case.admission.control_dependency_ids
    assert set(case.admission.scientific_dependency_ids).issubset(
        manifest.consumed_dependency_ids
    )
    assert barrier_id in manifest.control_dependency_ids
    assert barrier_id not in manifest.consumed_dependency_ids
    assert plan.lineage == manifest.lineage
    assert plan.current_release_observation.release_id == barrier_id
    assert (
        plan.activation_predecessor_pointer == case.recovery.fixture.remotes[0].pointer
    )
    assert plan.generation == 1
    assert plan.manifest_control_dependency_ids == (
        case.admission.control_dependency_ids
    )

    publication = _activate_recovery(
        case=case,
        settings=settings,
        package=package,
        publisher=publisher,
        reserved=reserved,
        monkeypatch=monkeypatch,
    )
    receipt = publication.activation.receipt
    assert publication.activation.snapshot.state.promotion_state is (
        ExpertPromotionState.RELEASED
    )
    assert set(case.admission.control_dependency_ids).issubset(
        receipt.control_dependency_ids
    )
    assert barrier_id in receipt.control_dependency_ids
    assert barrier_id not in receipt.consumed_dependency_ids
    reopened_activation = case.validation_store.reopen_release_activation(
        plan.candidate_id
    )
    assert reopened_activation.receipt == receipt
    assert reopened_activation.snapshot == publication.activation.snapshot
    assert reopened_activation.replayed is True
    assert case.validation_store.reopen_release_publication(plan.candidate_id) is None


def test_recovery_publication_allows_only_exact_barrier_security_match(
    tmp_path,
    monkeypatch,
):
    case = _historical_recovery_matrix_case(
        tmp_path,
        monkeypatch,
        chain_length=3,
    )
    barrier_id = case.recovery.barrier.release_id
    intermediate_id = case.recovery.fixture.releases[1].release_id
    assert intermediate_id in case.admission.control_dependency_ids
    assert case.admission.allowed_control_security_subject_ids == (barrier_id,)
    authority = _coordinator(
        SimpleNamespace(
            validation_store=case.validation_store,
            matrix_commit=case.committed_stage,
            prepared=case.prepared_task,
        ),
        denylist=_RecoveryPublicationDenylistAuthority((intermediate_id,)),
    )

    with pytest.raises(
        ExpertPublicationEligibilityError,
        match="denylist differs from exact authority",
    ):
        authority.coordinator.publish(
            candidate_id=case.committed_stage.snapshot.state.candidate_id,
            release_matrix_stage_result_id=(
                case.committed_stage.stage_result.stage_result_record_id
            ),
        )

    assert intermediate_id in authority.denylist.checked_subject_ids
