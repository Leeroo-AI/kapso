"""Genesis publication for one freshly bootstrapped run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.capture.revision_projection import ExecutionRevisionProjection
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
    RunCheckpointStatus,
    RunStrategyKind,
    RunStrategyState,
)
from kapso.cross_run.launch.derived_state_contracts import (
    RunStateAuthority,
    RunStateLayout,
)
from kapso.cross_run.launch.resume_contracts import (
    resume_security_subject_ids,
    RunDerivativeEvidence,
    RunDerivativeFrontier,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_action_ledger import RunActionLedgerSnapshot
from kapso.cross_run.launch.run_state_projection import ReconciledRunStateProjection
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.security_authority_contracts import SecurityDenylistObservation
from kapso.cross_run.settings import LaunchSettings
from kapso.execution.evaluation_integrity import AGENT_GENERATED
from kapso.execution.memories.experiment_memory.projection import (
    build_experiment_history_genesis,
)
from kapso.execution.search_strategies.generic.ideation.archive_projection import (
    build_archive_genesis,
)


class RunStateInitializationError(RuntimeError):
    """A fresh BootstrapPin cannot become an active run-state frontier."""


class RunInitializationSecurityAuthority(Protocol):
    """Narrow authenticated security reader needed at genesis publication."""

    def observe_exact_descendant_of(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
        required_ancestor: SecurityDenylistObservation,
    ) -> SecurityDenylistObservation: ...


@dataclass(frozen=True)
class InitializedRunState:
    """Live fresh-run authority after its atomic genesis publication."""

    active_workspace: ActiveLaunchWorkspace
    publisher: RunStatePublisher
    frontier: ReconciledRunFrontier

    def __post_init__(self) -> None:
        if (
            type(self.active_workspace) is not ActiveLaunchWorkspace
            or type(self.publisher) is not RunStatePublisher
            or type(self.frontier) is not ReconciledRunFrontier
            or self.publisher._authority is not self.active_workspace
        ):
            raise RunStateInitializationError(
                "initialized run state has mixed live authority"
            )
        self.active_workspace.require_control_authority()
        self.frontier.require_current(self.publisher)


def initialize_run_state(
    *,
    active_workspace: ActiveLaunchWorkspace,
    launch_settings: LaunchSettings,
    security_authority: RunInitializationSecurityAuthority,
    objective_direction: str,
) -> InitializedRunState:
    """Publish the sole empty checkpoint immediately after BootstrapPin handoff."""

    if (
        type(active_workspace) is not ActiveLaunchWorkspace
        or type(launch_settings) is not LaunchSettings
        or not hasattr(security_authority, "observe_exact_descendant_of")
        or objective_direction not in {"maximize", "minimize"}
    ):
        raise RunStateInitializationError(
            "run-state initialization requires exact configured authorities"
        )
    active_workspace.require_control_authority()
    pin = active_workspace.bootstrap_pin
    manifest = pin.launch_manifest
    installation = pin.installation_receipt
    strategy_kind = RunStrategyKind(manifest.launch_request.search_mode)
    projection = _genesis_projection(
        run_id=installation.run_id,
        campaign_id=installation.campaign_id,
        strategy_kind=strategy_kind,
        objective_direction=objective_direction,
        embedding_space=manifest.experiment_embedding_space,
        created_at=manifest.compatibility_receipt.resolved_at,
    )
    layout = _run_state_layout(active_workspace, strategy_kind)
    evidence = RunDerivativeEvidence.mint(
        state_authority_digests={
            authority.value: tree_or_blob_digest(payload)
            for authority, payload in projection.payload_by_authority.items()
        },
        state_authority_revisions={
            authority.value: revision
            for authority, revision in projection.revision_by_authority.items()
        },
        branch_origin_heads={
            installation.workspace_git_branch: (
                installation.workspace_baseline_commit_sha
            )
        },
        branch_advances=(),
        branch_heads={
            installation.workspace_git_branch: (
                installation.workspace_baseline_commit_sha
            )
        },
        artifact_digests={},
        derivative_ids=(),
    )
    frontier = RunDerivativeFrontier.build(
        launch_subject_ids=tuple(
            sorted(
                {
                    pin.bootstrap_pin_id,
                    installation.workspace_installation_receipt_id,
                    manifest.launch_manifest_id,
                    *manifest.exact_dependency_ids,
                }
            )
        ),
        evidence=evidence,
        derivatives=(),
    )
    release_use = manifest.release_use_observation
    security = security_authority.observe_exact_descendant_of(
        scope_id=manifest.scope_contract.scope_id,
        scope_contract_id=manifest.scope_contract.scope_contract_id,
        checked_subject_ids=resume_security_subject_ids(
            bootstrap_pin=pin,
            release_use_observation=release_use,
            derivative_frontier=frontier,
            predecessor_safety_state_id=None,
            inherited_security_subject_ids=(),
        ),
        required_ancestor=manifest.security_observation,
    )
    safety = RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=frontier,
        security_observation=security,
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    bundle = projection.build_bundle(
        bootstrap_pin=pin,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=(
            RunCheckpointHead.initial(pin).run_checkpoint_head_id
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=frontier.evidence.evidence_id,
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )
    checkpoint = RunCheckpoint.build(
        predecessor=None,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=None,
        completed_iterations=0,
        cumulative_cost=0.0,
        elapsed_seconds=0.0,
        cost_by_component={},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=projection.strategy_state,
        safety_state=safety,
        derived_state_generation=bundle.generation,
    )
    publisher = RunStatePublisher(active_workspace, launch_settings)
    if publisher.load_reconciled() is not None:
        raise RunStateInitializationError(
            "fresh run-state initialization found a published frontier"
        )
    permit = publisher.issue_publication_permit(None, checkpoint, bundle)
    published = publisher.publish(permit, checkpoint, bundle)
    return InitializedRunState(
        active_workspace=active_workspace,
        publisher=publisher,
        frontier=published,
    )


def _genesis_projection(
    *,
    run_id: str,
    campaign_id: str,
    strategy_kind: RunStrategyKind,
    objective_direction: str,
    embedding_space,
    created_at: str,
) -> ReconciledRunStateProjection:
    generic = strategy_kind is RunStrategyKind.GENERIC
    archive = (
        build_archive_genesis(campaign_id=campaign_id, created_at=created_at)
        if generic
        else None
    )
    evaluation_integrity = {
        "provenance": AGENT_GENERATED,
        "manifest": {},
        "fingerprint": None,
    }
    if generic:
        strategy_payload = {
            "idea_archive_snapshot": archive.to_dict(),
            "node_history": [],
            "iteration_count": 0,
            "previous_errors": [],
            "evaluation_integrity": evaluation_integrity,
            "scores_evaluator_id": "",
            "evaluator_transition": None,
        }
    else:
        strategy_payload = {
            "nodes": [],
            "node_history_ids": [],
            "experimentation_count": 0,
            "previous_errors": [],
            "evaluation_integrity": evaluation_integrity,
        }
    return ReconciledRunStateProjection(
        strategy_state=RunStrategyState.build(
            strategy_kind=strategy_kind,
            campaign_id=campaign_id,
            state=strategy_payload,
        ),
        experiment_history=build_experiment_history_genesis(
            run_id=run_id,
            campaign_id=campaign_id,
            embedding_space_id=embedding_space.embedding_space_id,
            embedding_provider=embedding_space.provider,
            embedding_model=embedding_space.model,
            embedding_dimensions=embedding_space.dimensions,
            embedding_canonicalizer_version=embedding_space.canonicalizer_version,
            objective_direction=objective_direction,
            require_idea_links=generic,
        ),
        execution_journal=ExecutionRevisionProjection(
            run_id=run_id,
            campaign_id=campaign_id,
            require_contiguous_node_ids=generic,
        ),
        idea_archive=archive,
        action_ledger=RunActionLedgerSnapshot.empty(),
    )


def _run_state_layout(
    active_workspace: ActiveLaunchWorkspace,
    strategy_kind: RunStrategyKind,
) -> RunStateLayout:
    installed = active_workspace.bootstrap_pin.installation_receipt.layout
    authority_paths = {
        RunStateAuthority.ACTION_LEDGER: installed.run_action_ledger_relative_path,
        RunStateAuthority.EXPERIMENT_HISTORY: (
            installed.run_experiment_history_relative_path
        ),
        RunStateAuthority.EXECUTION_JOURNAL: (
            installed.run_execution_journal_relative_path
        ),
    }
    if strategy_kind is RunStrategyKind.GENERIC:
        authority_paths[RunStateAuthority.IDEA_ARCHIVE] = (
            installed.run_idea_archive_relative_path
        )
    return RunStateLayout.build(
        strategy_kind=strategy_kind.value,
        authority_paths=authority_paths,
    )


__all__ = [
    "InitializedRunState",
    "initialize_run_state",
    "RunInitializationSecurityAuthority",
    "RunStateInitializationError",
]
