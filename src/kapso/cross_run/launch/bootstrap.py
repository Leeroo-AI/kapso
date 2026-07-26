"""Dormant composition seam for fresh launch bootstrap and local-pin resume."""

from __future__ import annotations

import secrets
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import CrossRunTaskBindingSettings
from kapso.cross_run.launch.contracts import BootstrapPin, LaunchRequest
from kapso.cross_run.launch.resolver import LaunchResolver
from kapso.cross_run.launch.resume import (
    AdmittedRunResume,
    BlockedRunResume,
    RunResumeCoordinator,
)
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from kapso.cross_run.launch.workspace import (
    ActiveLaunchWorkspace,
    StarterWorkspaceBuilder,
)
from kapso.cross_run.settings import CrossRunSettings


class LaunchBootstrapError(RuntimeError):
    """A typed launch cannot safely cross the workspace-bootstrap boundary."""


@dataclass(frozen=True)
class LaunchBootstrapIdentity:
    """Stable identity projection for result metadata and later composition."""

    run_id: str
    campaign_id: str
    launch_manifest_id: str
    bootstrap_pin_id: str
    scope_id: str
    scope_contract_id: str
    task_family_id: str
    task_adapter_id: str
    task_context_binding_id: str
    expert_release_id: str
    knowledge_snapshot_id: str
    task_adapter_manifest_id: str
    task_adapter_activation_id: str
    security_observation_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "launch bootstrap run"),
            (self.campaign_id, "launch bootstrap campaign"),
            (self.scope_id, "launch bootstrap scope"),
            (self.task_family_id, "launch bootstrap task family"),
            (self.task_adapter_id, "launch bootstrap task adapter"),
        ):
            require_identifier(value, name)
        for value, name in (
            (self.launch_manifest_id, "launch bootstrap manifest"),
            (self.bootstrap_pin_id, "launch bootstrap pin"),
            (self.scope_contract_id, "launch bootstrap scope contract"),
            (self.task_context_binding_id, "launch bootstrap task context"),
            (self.expert_release_id, "launch bootstrap expert release"),
            (self.knowledge_snapshot_id, "launch bootstrap knowledge snapshot"),
            (
                self.task_adapter_manifest_id,
                "launch bootstrap task adapter manifest",
            ),
            (
                self.task_adapter_activation_id,
                "launch bootstrap task adapter activation",
            ),
            (
                self.security_observation_id,
                "launch bootstrap security observation",
            ),
        ):
            require_content_id(value, name)

    @classmethod
    def from_bootstrap_pin(cls, pin: BootstrapPin) -> "LaunchBootstrapIdentity":
        if type(pin) is not BootstrapPin:
            raise LaunchBootstrapError(
                "launch bootstrap identity requires one exact bootstrap pin"
            )
        manifest = pin.launch_manifest
        installation = pin.installation_receipt
        binding = manifest.launch_request.binding
        return cls(
            run_id=installation.run_id,
            campaign_id=installation.campaign_id,
            launch_manifest_id=manifest.launch_manifest_id,
            bootstrap_pin_id=pin.bootstrap_pin_id,
            scope_id=binding.scope_id,
            scope_contract_id=manifest.scope_contract.scope_contract_id,
            task_family_id=binding.task_family_id,
            task_adapter_id=binding.task_adapter_id,
            task_context_binding_id=(
                manifest.task_context_binding.task_context_binding_id
            ),
            expert_release_id=manifest.expert_manifest.release_id,
            knowledge_snapshot_id=manifest.knowledge_manifest.snapshot_id,
            task_adapter_manifest_id=(
                manifest.task_adapter.manifest.task_adapter_manifest_id
            ),
            task_adapter_activation_id=(manifest.task_adapter.activation.activation_id),
            security_observation_id=manifest.security_observation.observation_id,
        )


@dataclass(frozen=True)
class BootstrappedLaunchWorkspace:
    """Live fresh workspace authority issued only after atomic activation."""

    active_workspace: ActiveLaunchWorkspace
    identity: LaunchBootstrapIdentity

    def __post_init__(self) -> None:
        if (
            type(self.active_workspace) is not ActiveLaunchWorkspace
            or type(self.identity) is not LaunchBootstrapIdentity
            or self.identity
            != LaunchBootstrapIdentity.from_bootstrap_pin(
                self.active_workspace.bootstrap_pin
            )
        ):
            raise LaunchBootstrapError(
                "bootstrapped launch workspace has mixed authority"
            )
        self.active_workspace.require_control_authority()

    def close(self) -> None:
        self.active_workspace.close()


class LaunchBootstrapCoordinator:
    """Compose verified fresh bootstrap or delegate strict local-pin resume."""

    def __init__(
        self,
        *,
        settings: CrossRunSettings,
        binding: CrossRunTaskBindingSettings,
        resolver: LaunchResolver,
        resume_coordinator: RunResumeCoordinator,
    ) -> None:
        if (
            type(settings) is not CrossRunSettings
            or type(binding) is not CrossRunTaskBindingSettings
            or type(resolver) is not LaunchResolver
            or type(resume_coordinator) is not RunResumeCoordinator
            or resolver.settings is not settings
            or resume_coordinator.settings is not settings
            or resume_coordinator.binding != binding
        ):
            raise LaunchBootstrapError(
                "launch bootstrap composition has mixed configuration authority"
            )
        settings.scopes.resolve(binding.scope_id)
        self._settings = settings
        self._binding = binding
        self._resolver = resolver
        self._resume_coordinator = resume_coordinator

    def fresh(
        self,
        *,
        request: LaunchRequest,
        run_root: Path,
    ) -> BootstrappedLaunchWorkspace:
        """Resolve and atomically activate one repository-free launch request."""

        if type(request) is not LaunchRequest or request.binding != self._binding:
            raise LaunchBootstrapError(
                "fresh launch request differs from its configured task binding"
            )
        resolved = self._resolver.resolve(request)
        prepared = StarterWorkspaceBuilder(self._settings).build(
            resolved,
            run_root,
            run_id=self._new_identifier("run"),
            campaign_id=self._new_identifier("campaign"),
        )
        with ExitStack() as resources:
            active = prepared.activate()
            resources.callback(active.close)
            bootstrapped = BootstrappedLaunchWorkspace(
                active_workspace=active,
                identity=LaunchBootstrapIdentity.from_bootstrap_pin(
                    active.bootstrap_pin
                ),
            )
            resources.pop_all()
            return bootstrapped

    def resume(
        self,
        run_root: Path,
        *,
        release_use_mode: RunReleaseUseMode,
    ) -> AdmittedRunResume | BlockedRunResume:
        """Resume locally without invoking the fresh resolver."""

        with ExitStack() as resources:
            result = self._resume_coordinator.resume(
                run_root,
                release_use_mode=release_use_mode,
            )
            if type(result) is AdmittedRunResume:
                resources.callback(result.close)
                pin = result.active_workspace.bootstrap_pin
            elif type(result) is BlockedRunResume:
                pin = result.checkpoint.safety_state.bootstrap_pin
            else:
                raise LaunchBootstrapError("run resume returned an unknown authority")
            if pin.launch_manifest.launch_request.binding != self._binding:
                raise LaunchBootstrapError(
                    "resumed launch differs from its configured task binding"
                )
            resources.pop_all()
            return result

    @staticmethod
    def _new_identifier(namespace: str) -> str:
        require_identifier(namespace, "launch bootstrap identifier namespace")
        return f"{namespace}_{secrets.token_hex(16)}"


__all__ = [
    "BootstrappedLaunchWorkspace",
    "LaunchBootstrapCoordinator",
    "LaunchBootstrapError",
    "LaunchBootstrapIdentity",
]
