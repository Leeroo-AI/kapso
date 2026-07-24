from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    SecurityDenylistKind,
    SecurityDenylistRevocation,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.launch.resume_contracts import (
    resume_security_subject_ids,
    ResumeContractError,
    RunBranchAdvance,
    RunDerivativeEvidence,
    RunDerivativeFrontier,
    RunDerivativeKind,
    RunDerivativeRecord,
    RunEligibilityDisposition,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.record_contracts import (
    ExpertReleaseUseRevocation,
    ExpertReleaseUseRevocationKind,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_launch_resolver import resolver_case

RECORDED_AT = "2026-07-23T16:00:00Z"


def _bootstrap_pin(resolver_case, tmp_path):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(resolver_case["resolver"]._settings).build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-resume-contracts",
        campaign_id="campaign-resume-contracts",
    )
    return prepared.bootstrap_pin


def _launch_subjects(pin):
    return tuple(
        sorted(
            {
                pin.bootstrap_pin_id,
                pin.installation_receipt.workspace_installation_receipt_id,
                pin.launch_manifest.launch_manifest_id,
                *pin.launch_manifest.exact_dependency_ids,
            }
        )
    )


def _derivative_evidence(pin, derivatives, revision):
    return RunDerivativeEvidence.mint(
        state_authority_digests={
            "idea_archive": tree_or_blob_digest(f"ideas-{revision}".encode("utf-8")),
            "experiment_history": tree_or_blob_digest(
                f"experiments-{revision}".encode("utf-8")
            ),
            "execution_journal": tree_or_blob_digest(
                f"journal-{revision}".encode("utf-8")
            ),
        },
        state_authority_revisions={
            "idea_archive": revision,
            "experiment_history": revision,
            "execution_journal": revision,
        },
        branch_origin_heads={
            pin.installation_receipt.workspace_git_branch: (
                pin.installation_receipt.workspace_baseline_commit_sha
            )
        },
        branch_advances=(),
        branch_heads={
            pin.installation_receipt.workspace_git_branch: (
                pin.installation_receipt.workspace_baseline_commit_sha
            )
        },
        artifact_digests={
            derivative.local_locator: derivative.payload_digest
            for derivative in derivatives
            if derivative.kind is RunDerivativeKind.ARTIFACT
        },
        derivative_ids=tuple(
            sorted(derivative.derivative_id for derivative in derivatives)
        ),
    )


def _remint_evidence(evidence, **changes):
    values = {
        "state_authority_digests": evidence.state_authority_digests,
        "state_authority_revisions": evidence.state_authority_revisions,
        "branch_origin_heads": evidence.branch_origin_heads,
        "branch_advances": evidence.branch_advances,
        "branch_heads": evidence.branch_heads,
        "artifact_digests": evidence.artifact_digests,
        "derivative_ids": evidence.derivative_ids,
    }
    values.update(changes)
    return RunDerivativeEvidence.mint(**values)


def _empty_frontier(pin):
    return RunDerivativeFrontier.build(
        launch_subject_ids=_launch_subjects(pin),
        evidence=_derivative_evidence(pin, (), 0),
        derivatives=(),
    )


def _derivative_frontier(pin, authorization_safety_state_id):
    source_id = pin.launch_manifest.task_context_binding.task_context_binding_id
    idea = RunDerivativeRecord.mint(
        kind=RunDerivativeKind.IDEA,
        local_locator="idea/0",
        payload_digest=tree_or_blob_digest(b"idea"),
        authorization_safety_state_id=authorization_safety_state_id,
        direct_source_ids=tuple(sorted((source_id, authorization_safety_state_id))),
    )
    experiment = RunDerivativeRecord.mint(
        kind=RunDerivativeKind.EXPERIMENT,
        local_locator="experiment/0",
        payload_digest=tree_or_blob_digest(b"experiment"),
        authorization_safety_state_id=authorization_safety_state_id,
        direct_source_ids=tuple(
            sorted((idea.derivative_id, authorization_safety_state_id))
        ),
    )
    artifact = RunDerivativeRecord.mint(
        kind=RunDerivativeKind.ARTIFACT,
        local_locator="artifact/0",
        payload_digest=tree_or_blob_digest(b"artifact"),
        authorization_safety_state_id=authorization_safety_state_id,
        direct_source_ids=tuple(
            sorted((experiment.derivative_id, authorization_safety_state_id))
        ),
    )
    derivatives = (artifact, experiment, idea)
    return RunDerivativeFrontier.build(
        launch_subject_ids=_launch_subjects(pin),
        evidence=_derivative_evidence(pin, derivatives, 1),
        derivatives=derivatives,
    )


def _release_use_with_revocation(pin):
    manifest = pin.launch_manifest
    base = manifest.release_use_observation
    revocation = ExpertReleaseUseRevocation.mint(
        scope_contract_id=manifest.scope_contract.scope_contract_id,
        scope_id=manifest.scope_contract.scope_id,
        release_id=manifest.expert_manifest.release_id,
        release_publication_id=(manifest.expert_component.publication.publication_id),
        release_activation_witness_id=(
            manifest.expert_component.activation_witness.witness_id
        ),
        kind=ExpertReleaseUseRevocationKind.PERFORMANCE,
        reason_code="performance_regression",
        rationale="A later audit found a reproducible performance regression.",
        exact_evidence_refs=(content_id("release-use-evidence", {"case": "resume"}),),
        recorded_at=RECORDED_AT,
    )
    return ExpertReleaseUsePolicyObservation.mint(
        scope_id=base.scope_id,
        scope_contract_id=base.scope_contract_id,
        scope_repository_binding_hash=base.scope_repository_binding_hash,
        repository_full_name=base.repository_full_name,
        repository_node_id=base.repository_node_id,
        knowledge_snapshot_id=content_id(
            "knowledge-snapshot",
            {"catalog_generation": base.catalog_generation + 1},
        ),
        catalog_generation=base.catalog_generation + 1,
        knowledge_publication_id=content_id(
            "github-publication",
            {"catalog_generation": base.catalog_generation + 1},
        ),
        current_pointer_digest=tree_or_blob_digest(b"new knowledge current"),
        authority_commit_sha="8" * 40,
        release_attestation_ref="attestations/knowledge/current",
        checked_release_ids=(manifest.expert_manifest.release_id,),
        matched_revocations=(revocation,),
    )


def _security_observation(
    pin,
    subjects,
    revocations=(),
    generation_offset=1,
):
    base = pin.launch_manifest.security_observation
    generation = base.generation + generation_offset
    return SecurityDenylistObservation.mint(
        scope_id=base.scope_id,
        scope_contract_id=base.scope_contract_id,
        scope_repository_binding_hash=base.scope_repository_binding_hash,
        snapshot_id=content_id(
            "security-denylist-snapshot",
            {
                "generation": generation,
                "revocations": tuple(
                    revocation.revocation_id for revocation in revocations
                ),
            },
        ),
        generation=generation,
        publication_id=content_id(
            "github-publication",
            {
                "security_generation": generation,
                "revocations": tuple(
                    revocation.revocation_id for revocation in revocations
                ),
            },
        ),
        repository_full_name=base.repository_full_name,
        repository_node_id=base.repository_node_id,
        pointer_digest=tree_or_blob_digest(
            b"current security pointer"
            + b"".join(
                revocation.revocation_id.encode("utf-8") for revocation in revocations
            )
        ),
        authority_commit_sha="7" * 40,
        release_attestation_ref="attestations/security/current",
        checked_subject_ids=subjects,
        matched_revocations=tuple(
            sorted(revocations, key=lambda item: item.revocation_id)
        ),
    )


def _subjects(pin, release_use, frontier, predecessor=None):
    return resume_security_subject_ids(
        bootstrap_pin=pin,
        release_use_observation=release_use,
        derivative_frontier=frontier,
        predecessor_safety_state_id=(
            None if predecessor is None else predecessor.safety_state_id
        ),
        inherited_security_subject_ids=(
            ()
            if predecessor is None
            else predecessor.security_observation.checked_subject_ids
        ),
    )


def test_safety_state_derives_eligibility_and_transitive_taint(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    initial_frontier = _empty_frontier(pin)
    release_use = pin.launch_manifest.release_use_observation
    eligible = RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=initial_frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, initial_frontier),
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    frontier = _derivative_frontier(pin, eligible.safety_state_id)
    revoked_release_use = _release_use_with_revocation(pin)
    reproducibility_only = RunSafetyState.build(
        predecessor=eligible,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, revoked_release_use, frontier, eligible),
        ),
        release_use_observation=revoked_release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    with pytest.raises(ResumeContractError, match="rolled back"):
        RunSafetyState.build(
            predecessor=reproducibility_only,
            bootstrap_pin=pin,
            boundary=RunSafetyBoundary.IMPLEMENTATION,
            derivative_frontier=frontier,
            security_observation=_security_observation(
                pin,
                _subjects(
                    pin,
                    release_use,
                    frontier,
                    reproducibility_only,
                ),
            ),
            release_use_observation=release_use,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )

    idea_id = next(
        item.derivative_id
        for item in frontier.derivatives
        if item.kind is RunDerivativeKind.IDEA
    )
    security_revocation = SecurityDenylistRevocation.mint(
        subject_id=idea_id,
        kind=SecurityDenylistKind.CONTAMINATION,
        reason_code="contaminated_idea",
        evidence_ids=(content_id("security-evidence", {"case": "idea"}),),
        recorded_at=RECORDED_AT,
    )
    blocked = RunSafetyState.build(
        predecessor=reproducibility_only,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.EVALUATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(
                pin,
                revoked_release_use,
                frontier,
                reproducibility_only,
            ),
            (security_revocation,),
            generation_offset=2,
        ),
        release_use_observation=revoked_release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert eligible.disposition is RunEligibilityDisposition.ELIGIBLE
    assert (
        reproducibility_only.disposition
        is RunEligibilityDisposition.REPRODUCIBILITY_ONLY
    )
    assert blocked.disposition is RunEligibilityDisposition.SECURITY_BLOCKED
    assert {taint.derivative_id for taint in blocked.derivative_taints} == set(
        frontier.derivative_ids
    )
    assert any(taint.predecessor_taint_ids for taint in blocked.derivative_taints)
    with pytest.raises(ResumeContractError, match="fixed point"):
        replace(blocked, derivative_taints=blocked.derivative_taints[:-1])

    launch_revocation = SecurityDenylistRevocation.mint(
        subject_id=pin.launch_manifest.expert_manifest.release_id,
        kind=SecurityDenylistKind.SECURITY,
        reason_code="compromised_launch",
        evidence_ids=(content_id("security-evidence", {"case": "launch-root"}),),
        recorded_at=RECORDED_AT,
    )
    launch_blocked = RunSafetyState.build(
        predecessor=reproducibility_only,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.EVALUATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(
                pin,
                revoked_release_use,
                frontier,
                reproducibility_only,
            ),
            (launch_revocation,),
            generation_offset=2,
        ),
        release_use_observation=revoked_release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    assert {taint.derivative_id for taint in launch_blocked.derivative_taints} == set(
        frontier.derivative_ids
    )

    artifact = next(
        item for item in frontier.derivatives if item.kind is RunDerivativeKind.ARTIFACT
    )
    with pytest.raises(ResumeContractError, match="artifacts differ"):
        RunDerivativeFrontier.build(
            launch_subject_ids=frontier.launch_subject_ids,
            evidence=_remint_evidence(
                frontier.evidence,
                artifact_digests={
                    artifact.local_locator: tree_or_blob_digest(b"rewritten")
                },
            ),
            derivatives=frontier.derivatives,
        )

    branch = pin.installation_receipt.workspace_git_branch
    first_advance = RunBranchAdvance.build(
        branch=branch,
        predecessor_commit_sha=frontier.evidence.branch_heads[branch],
        commit_sha="5" * 40,
        predecessor_branch_advance_id=None,
        authorization_safety_state_id=reproducibility_only.safety_state_id,
    )
    advanced_evidence = _remint_evidence(
        frontier.evidence,
        branch_advances=(first_advance,),
        branch_heads={branch: first_advance.commit_sha},
    )
    advanced_frontier = RunDerivativeFrontier.build(
        launch_subject_ids=frontier.launch_subject_ids,
        evidence=advanced_evidence,
        derivatives=frontier.derivatives,
    )
    assert first_advance.branch_advance_id in advanced_frontier.exact_dependency_ids
    assert first_advance.branch_advance_id in _subjects(
        pin,
        revoked_release_use,
        advanced_frontier,
        reproducibility_only,
    )
    advanced = RunSafetyState.build(
        predecessor=reproducibility_only,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.EVALUATION,
        derivative_frontier=advanced_frontier,
        security_observation=_security_observation(
            pin,
            _subjects(
                pin,
                revoked_release_use,
                advanced_frontier,
                reproducibility_only,
            ),
            generation_offset=2,
        ),
        release_use_observation=revoked_release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    with pytest.raises(ResumeContractError, match="rolled back"):
        RunSafetyState.build(
            predecessor=advanced,
            bootstrap_pin=pin,
            boundary=RunSafetyBoundary.PUBLICATION,
            derivative_frontier=frontier,
            security_observation=_security_observation(
                pin,
                _subjects(pin, revoked_release_use, frontier, advanced),
                generation_offset=3,
            ),
            release_use_observation=revoked_release_use,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )


def test_offline_release_use_is_always_reproducibility_only(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    frontier = _empty_frontier(pin)
    release_use = pin.launch_manifest.release_use_observation
    state = RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, frontier),
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.PINNED_OFFLINE,
    )

    assert state.disposition is RunEligibilityDisposition.REPRODUCIBILITY_ONLY
    assert RunSafetyState.from_json_bytes(state.to_json_bytes()) == state

    forked_release_use = ExpertReleaseUsePolicyObservation.mint(
        scope_id=release_use.scope_id,
        scope_contract_id=release_use.scope_contract_id,
        scope_repository_binding_hash=release_use.scope_repository_binding_hash,
        repository_full_name=release_use.repository_full_name,
        repository_node_id=release_use.repository_node_id,
        knowledge_snapshot_id=release_use.knowledge_snapshot_id,
        catalog_generation=release_use.catalog_generation,
        knowledge_publication_id=release_use.knowledge_publication_id,
        current_pointer_digest=tree_or_blob_digest(b"forked knowledge current"),
        authority_commit_sha="6" * 40,
        release_attestation_ref=release_use.release_attestation_ref,
        checked_release_ids=release_use.checked_release_ids,
        matched_revocations=release_use.matched_revocations,
    )
    with pytest.raises(ResumeContractError, match="bootstrap authority"):
        RunSafetyState.build(
            predecessor=None,
            bootstrap_pin=pin,
            boundary=RunSafetyBoundary.INITIALIZATION,
            derivative_frontier=frontier,
            security_observation=_security_observation(
                pin,
                _subjects(pin, forked_release_use, frontier),
            ),
            release_use_observation=forked_release_use,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )


def test_frontier_rejects_unknown_sources_and_safety_rejects_splicing(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    unknown_source = content_id("unknown-source", {"case": "resume"})
    authorization_id = content_id("run-safety-state", {"case": "authorization"})
    derivative = RunDerivativeRecord.mint(
        kind=RunDerivativeKind.IDEA,
        local_locator="idea/unknown",
        payload_digest=tree_or_blob_digest(b"unknown"),
        authorization_safety_state_id=authorization_id,
        direct_source_ids=tuple(sorted((authorization_id, unknown_source))),
    )
    with pytest.raises(ResumeContractError, match="unknown source"):
        RunDerivativeFrontier.build(
            launch_subject_ids=_launch_subjects(pin),
            evidence=_derivative_evidence(pin, (derivative,), 1),
            derivatives=(derivative,),
        )

    frontier = _empty_frontier(pin)
    release_use = pin.launch_manifest.release_use_observation
    state = RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, frontier),
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    with pytest.raises(ResumeContractError, match="predecessor"):
        state.require_predecessor(state)

    nonempty_initial_frontier = RunDerivativeFrontier.build(
        launch_subject_ids=_launch_subjects(pin),
        evidence=_derivative_evidence(pin, (), 1),
        derivatives=(),
    )
    with pytest.raises(ResumeContractError, match="empty predecessor frontier"):
        RunSafetyState.build(
            predecessor=None,
            bootstrap_pin=pin,
            boundary=RunSafetyBoundary.INITIALIZATION,
            derivative_frontier=nonempty_initial_frontier,
            security_observation=_security_observation(
                pin,
                _subjects(pin, release_use, nonempty_initial_frontier),
            ),
            release_use_observation=release_use,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )
