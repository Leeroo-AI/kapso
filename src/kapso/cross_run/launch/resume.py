"""Policy-refreshed admission of one locally pinned run."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kapso.cross_run.contracts import CrossRunTaskBindingSettings
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
)
from kapso.cross_run.launch.resume_contracts import (
    resume_security_subject_ids,
    RunEligibilityDisposition,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
)
from kapso.cross_run.launch.workspace import (
    ActiveLaunchWorkspace,
    StarterWorkspaceBuilder,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings


class RunResumeError(RuntimeError):
    """A local run cannot be admitted against its durable and live authorities."""


class RunResumeReleaseUseAuthority(Protocol):
    """Narrow current scientific-policy reader used without replacing pinned bytes."""

    def observe_exact(
        self,
        *,
        scope_contract,
        checked_release_ids: tuple[str, ...],
    ) -> ExpertReleaseUsePolicyObservation: ...


class RunResumeSecurityAuthority(Protocol):
    """Authenticated current denylist reader with local and run ancestry proof."""

    def observe_exact_descendant_of(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
        required_ancestor: SecurityDenylistObservation,
    ) -> SecurityDenylistObservation: ...


@dataclass(frozen=True)
class AdmittedRunResume:
    """Live authority returned only for a non-security-blocked successor."""

    active_workspace: ActiveLaunchWorkspace
    publisher: RunStatePublisher
    frontier: ReconciledRunFrontier

    def __post_init__(self) -> None:
        if (
            type(self.active_workspace) is not ActiveLaunchWorkspace
            or type(self.publisher) is not RunStatePublisher
            or type(self.frontier) is not ReconciledRunFrontier
            or self.publisher._authority is not self.active_workspace
            or self.frontier.checkpoint.safety_state.disposition
            is RunEligibilityDisposition.SECURITY_BLOCKED
        ):
            raise RunResumeError("admitted run resume lacks exact live authority")
        self.active_workspace.require_control_authority()
        self.frontier.require_current(self.publisher)

    def close(self) -> None:
        self.active_workspace.close()


@dataclass(frozen=True)
class BlockedRunResume:
    """Durable blocked evidence carrying no filesystem or execution capability."""

    checkpoint: RunCheckpoint

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint) is not RunCheckpoint
            or self.checkpoint.safety_state.disposition
            is not RunEligibilityDisposition.SECURITY_BLOCKED
        ):
            raise RunResumeError(
                "blocked run resume must contain one security-blocked checkpoint"
            )


class RunResumeCoordinator:
    """Join a local pin, durable frontier, workspace, and refreshed live policy."""

    def __init__(
        self,
        *,
        settings: CrossRunSettings,
        binding: CrossRunTaskBindingSettings,
        security_authority: RunResumeSecurityAuthority,
        release_use_authority: RunResumeReleaseUseAuthority | None,
    ) -> None:
        if (
            type(settings) is not CrossRunSettings
            or type(binding) is not CrossRunTaskBindingSettings
        ):
            raise RunResumeError("run resume requires exact settings and task binding")
        settings.scopes.resolve(binding.scope_id)
        if not hasattr(security_authority, "observe_exact_descendant_of"):
            raise RunResumeError(
                "run resume requires a descendant-checking security authority"
            )
        if release_use_authority is not None and not hasattr(
            release_use_authority,
            "observe_exact",
        ):
            raise RunResumeError(
                "run resume release-use authority has no exact observation method"
            )
        self._settings = settings
        self._binding = binding
        self._security_authority = security_authority
        self._release_use_authority = release_use_authority

    @property
    def settings(self) -> CrossRunSettings:
        """Return the exact configuration object bound to resumed runs."""

        return self._settings

    @property
    def binding(self) -> CrossRunTaskBindingSettings:
        """Return the exact task binding required from the local pin."""

        return self._binding

    def resume(
        self,
        run_root: Path,
        *,
        release_use_mode: RunReleaseUseMode,
    ) -> AdmittedRunResume | BlockedRunResume:
        """Publish one RESUME safety successor before returning live authority."""

        if type(release_use_mode) is not RunReleaseUseMode:
            raise RunResumeError("run resume requires one exact release-use mode")
        with ExitStack() as resources:
            active = StarterWorkspaceBuilder(self._settings).reopen(run_root)
            resources.callback(active.close)
            pin = active.bootstrap_pin
            if pin.launch_manifest.launch_request.binding != self._binding:
                raise RunResumeError(
                    "local run pin differs from its configured task binding"
                )
            publisher = RunStatePublisher(active, self._settings.launch)
            current = publisher.load_reconciled()
            if current is None:
                raise RunResumeError(
                    "run resume requires one published checkpoint frontier"
                )
            predecessor = current.checkpoint
            if (
                predecessor.safety_state.disposition
                is RunEligibilityDisposition.SECURITY_BLOCKED
            ):
                return BlockedRunResume(checkpoint=predecessor)
            if predecessor.status is not RunCheckpointStatus.ACTIVE:
                raise RunResumeError("completed run checkpoint cannot resume")
            if publisher.action_ledger_snapshot() != current.projection.action_ledger:
                raise RunResumeError(
                    "run action ledger must be reconciled before run resume"
                )
            expected_commit = (
                predecessor.safety_state.derivative_frontier.evidence.branch_heads.get(
                    self._settings.launch.workspace_git_branch
                )
            )
            if expected_commit is None:
                raise RunResumeError(
                    "run checkpoint omits its configured workspace branch"
                )
            with ExitStack() as descriptors:
                workspace_descriptor, _identity = active._open_execution_workspace(
                    descriptors
                )
                inspect_run_workspace_frontier(
                    workspace_descriptor,
                    settings=self._settings.launch,
                    expected_commit_sha=expected_commit,
                )

            manifest = pin.launch_manifest
            release_use = self._release_use_observation(
                predecessor,
                release_use_mode,
            )
            checked_subject_ids = resume_security_subject_ids(
                bootstrap_pin=pin,
                release_use_observation=release_use,
                derivative_frontier=(predecessor.safety_state.derivative_frontier),
                predecessor_safety_state_id=(predecessor.safety_state.safety_state_id),
                inherited_security_subject_ids=(
                    predecessor.safety_state.security_observation.checked_subject_ids
                ),
            )
            security = self._security_authority.observe_exact_descendant_of(
                scope_id=manifest.scope_contract.scope_id,
                scope_contract_id=manifest.scope_contract.scope_contract_id,
                checked_subject_ids=checked_subject_ids,
                required_ancestor=(predecessor.safety_state.security_observation),
            )
            safety = RunSafetyState.build(
                predecessor=predecessor.safety_state,
                bootstrap_pin=pin,
                boundary=RunSafetyBoundary.RESUME,
                derivative_frontier=(predecessor.safety_state.derivative_frontier),
                security_observation=security,
                release_use_observation=release_use,
                release_use_mode=release_use_mode,
            )
            bundle = current.projection.build_bundle(
                bootstrap_pin=pin,
                run_state_layout=(
                    predecessor.derived_state_generation.run_state_layout
                ),
                predecessor_checkpoint_head_id=current.journal_head_id,
                predecessor_checkpoint_id=predecessor.run_checkpoint_id,
                predecessor_evidence_id=(
                    predecessor.safety_state.derivative_frontier.evidence.evidence_id
                ),
                target_evidence_id=(safety.derivative_frontier.evidence.evidence_id),
                predecessor_bundle=current.bundle,
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
            permit = publisher.issue_publication_permit(
                current,
                candidate,
                bundle,
            )
            published = publisher.publish(permit, candidate, bundle)
            if (
                published.checkpoint.safety_state.disposition
                is RunEligibilityDisposition.SECURITY_BLOCKED
            ):
                return BlockedRunResume(checkpoint=published.checkpoint)
            admitted = AdmittedRunResume(
                active_workspace=active,
                publisher=publisher,
                frontier=published,
            )
            resources.pop_all()
            return admitted

    def _release_use_observation(
        self,
        predecessor: RunCheckpoint,
        mode: RunReleaseUseMode,
    ) -> ExpertReleaseUsePolicyObservation:
        if mode is RunReleaseUseMode.PINNED_OFFLINE:
            return (
                predecessor.safety_state.bootstrap_pin.launch_manifest.release_use_observation
            )
        authority = self._release_use_authority
        if authority is None:
            raise RunResumeError(
                "online run resume requires a release-use policy authority"
            )
        manifest = predecessor.safety_state.bootstrap_pin.launch_manifest
        return authority.observe_exact(
            scope_contract=manifest.scope_contract,
            checked_release_ids=(manifest.expert_manifest.release_id,),
        )


__all__ = [
    "AdmittedRunResume",
    "BlockedRunResume",
    "RunResumeCoordinator",
    "RunResumeError",
    "RunResumeReleaseUseAuthority",
    "RunResumeSecurityAuthority",
]
