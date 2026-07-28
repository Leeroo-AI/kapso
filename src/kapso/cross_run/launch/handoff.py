"""Prepared orchestration handoff after verified fresh launch or strict resume."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from kapso.cross_run.launch.bootstrap import (
    LaunchBootstrapCoordinator,
    LaunchBootstrapIdentity,
)
from kapso.cross_run.launch.contracts import LaunchRequest
from kapso.cross_run.launch.initialization import initialize_run_state
from kapso.cross_run.launch.repository_memory import (
    build_repository_memory,
    RepositoryMemory,
)
from kapso.cross_run.launch.resume import AdmittedRunResume, BlockedRunResume
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.settings import CrossRunSettings


class LaunchHandoffError(RuntimeError):
    """A launch cannot expose mixed or incompletely prepared run authority."""


@dataclass(frozen=True)
class PreparedRunHandoff:
    """The sole live authority that may be passed to paid orchestration."""

    active_workspace: ActiveLaunchWorkspace
    publisher: RunStatePublisher
    frontier: ReconciledRunFrontier
    identity: LaunchBootstrapIdentity
    repository_memory: RepositoryMemory
    resumed: bool

    def __post_init__(self) -> None:
        if (
            type(self.active_workspace) is not ActiveLaunchWorkspace
            or type(self.publisher) is not RunStatePublisher
            or type(self.frontier) is not ReconciledRunFrontier
            or type(self.identity) is not LaunchBootstrapIdentity
            or type(self.repository_memory) is not RepositoryMemory
            or type(self.resumed) is not bool
            or self.publisher._authority is not self.active_workspace
            or self.identity
            != LaunchBootstrapIdentity.from_bootstrap_pin(
                self.active_workspace.bootstrap_pin
            )
            or self.repository_memory.source_commit_sha
            != self.frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads.get(
                self.active_workspace.bootstrap_pin.installation_receipt.workspace_git_branch
            )
        ):
            raise LaunchHandoffError(
                "prepared run handoff contains mixed launch, state, or repository authority"
            )
        self.active_workspace.require_control_authority()
        self.frontier.require_current(self.publisher)

    def close(self) -> None:
        self.active_workspace.close()


def prepare_fresh_run_handoff(
    *,
    coordinator: LaunchBootstrapCoordinator,
    settings: CrossRunSettings,
    security_authority,
    request: LaunchRequest,
    run_root: Path,
    objective_direction: str,
) -> PreparedRunHandoff:
    """Resolve, activate, map, initialize, then expose one fresh run."""

    if (
        type(coordinator) is not LaunchBootstrapCoordinator
        or type(settings) is not CrossRunSettings
        or coordinator._settings is not settings
        or not hasattr(security_authority, "observe_exact_descendant_of")
        or type(request) is not LaunchRequest
        or not isinstance(run_root, Path)
    ):
        raise LaunchHandoffError("fresh handoff requires exact configured authority")
    with ExitStack() as resources:
        bootstrapped = coordinator.fresh(request=request, run_root=run_root)
        resources.callback(bootstrapped.close)
        baseline_commit = (
            bootstrapped.active_workspace.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha
        )
        repository_memory = build_repository_memory(
            active_workspace=bootstrapped.active_workspace,
            settings=settings,
            expected_commit_sha=baseline_commit,
        )
        initialized = initialize_run_state(
            active_workspace=bootstrapped.active_workspace,
            launch_settings=settings.launch,
            security_authority=security_authority,
            objective_direction=objective_direction,
        )
        handoff = PreparedRunHandoff(
            active_workspace=initialized.active_workspace,
            publisher=initialized.publisher,
            frontier=initialized.frontier,
            identity=bootstrapped.identity,
            repository_memory=repository_memory,
            resumed=False,
        )
        resources.pop_all()
        return handoff


def prepare_resumed_run_handoff(
    *,
    coordinator: LaunchBootstrapCoordinator,
    settings: CrossRunSettings,
    run_root: Path,
    release_use_mode: RunReleaseUseMode,
) -> PreparedRunHandoff | BlockedRunResume:
    """Refresh policy, map the current branch head, then expose one resumed run."""

    if (
        type(coordinator) is not LaunchBootstrapCoordinator
        or type(settings) is not CrossRunSettings
        or coordinator._settings is not settings
        or not isinstance(run_root, Path)
        or type(release_use_mode) is not RunReleaseUseMode
    ):
        raise LaunchHandoffError("resume handoff requires exact configured authority")
    with ExitStack() as resources:
        resumed = coordinator.resume(run_root, release_use_mode=release_use_mode)
        if type(resumed) is BlockedRunResume:
            return resumed
        if type(resumed) is not AdmittedRunResume:
            raise LaunchHandoffError("resume returned an unknown authority")
        resources.callback(resumed.close)
        branch = (
            resumed.active_workspace.bootstrap_pin.installation_receipt.workspace_git_branch
        )
        expected_commit = resumed.frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads.get(
            branch
        )
        if expected_commit is None:
            raise LaunchHandoffError("resumed frontier omits its workspace branch")
        repository_memory = build_repository_memory(
            active_workspace=resumed.active_workspace,
            settings=settings,
            expected_commit_sha=expected_commit,
        )
        handoff = PreparedRunHandoff(
            active_workspace=resumed.active_workspace,
            publisher=resumed.publisher,
            frontier=resumed.frontier,
            identity=LaunchBootstrapIdentity.from_bootstrap_pin(
                resumed.active_workspace.bootstrap_pin
            ),
            repository_memory=repository_memory,
            resumed=True,
        )
        resources.pop_all()
        return handoff


__all__ = [
    "LaunchHandoffError",
    "PreparedRunHandoff",
    "prepare_fresh_run_handoff",
    "prepare_resumed_run_handoff",
]
