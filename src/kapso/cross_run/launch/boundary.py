"""Publish one action-reconciled safety boundary for an active run."""

from __future__ import annotations

from dataclasses import replace

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
)
from kapso.cross_run.launch.resume import (
    RunResumeReleaseUseAuthority,
    RunResumeSecurityAuthority,
)
from kapso.cross_run.launch.resume_contracts import (
    resume_security_subject_ids,
    RunBranchAdvance,
    RunDerivativeEvidence,
    RunDerivativeFrontier,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
)


class RunBoundaryPublicationError(RuntimeError):
    """An active run cannot publish the requested dangerous boundary."""


def publish_run_boundary(
    *,
    publisher: RunStatePublisher,
    frontier: ReconciledRunFrontier,
    security_authority: RunResumeSecurityAuthority,
    release_use_authority: RunResumeReleaseUseAuthority,
    boundary: RunSafetyBoundary,
) -> ReconciledRunFrontier:
    """Reconcile terminal actions and publish one freshly authorized boundary."""

    if (
        type(publisher) is not RunStatePublisher
        or type(frontier) is not ReconciledRunFrontier
        or not hasattr(security_authority, "observe_exact_descendant_of")
        or not hasattr(release_use_authority, "observe_exact")
        or type(boundary) is not RunSafetyBoundary
        or boundary
        not in {
            RunSafetyBoundary.IDEATION,
            RunSafetyBoundary.IMPLEMENTATION,
            RunSafetyBoundary.EVALUATION,
            RunSafetyBoundary.PUBLICATION,
        }
    ):
        raise RunBoundaryPublicationError(
            "run boundary publication requires exact active authorities"
        )
    predecessor = frontier.require_current(publisher)
    if (
        predecessor.status is not RunCheckpointStatus.ACTIVE
        or predecessor.last_stop is not None
    ):
        raise RunBoundaryPublicationError(
            "only an active non-yielded run can publish an action boundary"
        )
    pin = predecessor.safety_state.bootstrap_pin
    projection = replace(
        frontier.projection,
        action_ledger=publisher.action_ledger_snapshot(),
    )
    derivative_frontier = _reconciled_derivative_frontier(
        publisher=publisher,
        frontier=frontier,
        projection=projection,
    )
    manifest = pin.launch_manifest
    release_use = release_use_authority.observe_exact(
        scope_contract=manifest.scope_contract,
        checked_release_ids=(manifest.expert_manifest.release_id,),
    )
    predecessor_safety = predecessor.safety_state
    checked_subject_ids = resume_security_subject_ids(
        bootstrap_pin=pin,
        release_use_observation=release_use,
        derivative_frontier=derivative_frontier,
        predecessor_safety_state_id=predecessor_safety.safety_state_id,
        inherited_security_subject_ids=(
            predecessor_safety.security_observation.checked_subject_ids
        ),
    )
    security = security_authority.observe_exact_descendant_of(
        scope_id=manifest.scope_contract.scope_id,
        scope_contract_id=manifest.scope_contract.scope_contract_id,
        checked_subject_ids=checked_subject_ids,
        required_ancestor=predecessor_safety.security_observation,
    )
    safety = RunSafetyState.build(
        predecessor=predecessor_safety,
        bootstrap_pin=pin,
        boundary=boundary,
        derivative_frontier=derivative_frontier,
        security_observation=security,
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    bundle = projection.build_bundle(
        bootstrap_pin=pin,
        run_state_layout=predecessor.derived_state_generation.run_state_layout,
        predecessor_checkpoint_head_id=frontier.journal_head_id,
        predecessor_checkpoint_id=predecessor.run_checkpoint_id,
        predecessor_evidence_id=predecessor_safety.derivative_frontier.evidence.evidence_id,
        target_evidence_id=derivative_frontier.evidence.evidence_id,
        predecessor_bundle=frontier.bundle,
        predecessor_strategy_state=predecessor.strategy_state,
    )
    candidate = RunCheckpoint.build(
        predecessor=predecessor,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=None,
        completed_iterations=predecessor.completed_iterations,
        cumulative_cost=predecessor.cumulative_cost,
        elapsed_seconds=predecessor.elapsed_seconds,
        cost_by_component=predecessor.cost_by_component,
        feedback_source=predecessor.feedback_source,
        current_feedback=predecessor.current_feedback,
        termination_decision=None,
        strategy_state=predecessor.strategy_state,
        safety_state=safety,
        derived_state_generation=bundle.generation,
    )
    permit = publisher.issue_publication_permit(frontier, candidate, bundle)
    return publisher.publish(permit, candidate, bundle)


def _reconciled_derivative_frontier(
    *,
    publisher: RunStatePublisher,
    frontier: ReconciledRunFrontier,
    projection,
) -> RunDerivativeFrontier:
    predecessor = frontier.checkpoint.safety_state
    prior_frontier = predecessor.derivative_frontier
    prior_evidence = prior_frontier.evidence
    inspection = publisher._action_store.inspect()
    new_operations = inspection.operations_since(frontier.projection.action_ledger)
    workspace_changes = tuple(
        pair
        for pair in inspection.workspace_chain(new_operations)
        if pair[0] != pair[1]
    )
    if len(workspace_changes) > 1:
        raise RunBoundaryPublicationError(
            "one safety boundary cannot reconcile multiple workspace edits"
        )
    branch_heads = dict(prior_evidence.branch_heads)
    branch_advances = prior_evidence.branch_advances
    if workspace_changes:
        before, after = workspace_changes[0]
        branch = publisher._settings.workspace_git_branch
        if (
            before is None
            or after is None
            or before.branch != branch
            or after.branch != branch
            or before.commit_sha != prior_evidence.branch_heads.get(branch)
        ):
            raise RunBoundaryPublicationError(
                "terminal workspace edit differs from the current run frontier"
            )
        prior_branch_advances = tuple(
            advance
            for advance in prior_evidence.branch_advances
            if advance.branch == branch and advance.commit_sha == before.commit_sha
        )
        if before.commit_sha == prior_evidence.branch_origin_heads[branch]:
            predecessor_advance_id = None
        elif len(prior_branch_advances) == 1:
            predecessor_advance_id = prior_branch_advances[0].branch_advance_id
        else:
            raise RunBoundaryPublicationError(
                "current workspace head lacks one exact branch advance"
            )
        advance = RunBranchAdvance.build(
            branch=branch,
            predecessor_commit_sha=before.commit_sha,
            commit_sha=after.commit_sha,
            predecessor_branch_advance_id=predecessor_advance_id,
            authorization_safety_state_id=predecessor.safety_state_id,
        )
        branch_heads[branch] = after.commit_sha
        branch_advances = tuple(
            sorted(
                (*prior_evidence.branch_advances, advance),
                key=lambda item: item.branch_advance_id,
            )
        )
    evidence = RunDerivativeEvidence.mint(
        state_authority_digests={
            authority.value: tree_or_blob_digest(payload)
            for authority, payload in projection.payload_by_authority.items()
        },
        state_authority_revisions={
            authority.value: revision
            for authority, revision in projection.revision_by_authority.items()
        },
        branch_origin_heads=prior_evidence.branch_origin_heads,
        branch_advances=branch_advances,
        branch_heads=branch_heads,
        artifact_digests=prior_evidence.artifact_digests,
        derivative_ids=prior_evidence.derivative_ids,
    )
    return RunDerivativeFrontier.build(
        launch_subject_ids=prior_frontier.launch_subject_ids,
        evidence=evidence,
        derivatives=prior_frontier.derivatives,
    )


__all__ = [
    "publish_run_boundary",
    "RunBoundaryPublicationError",
]
