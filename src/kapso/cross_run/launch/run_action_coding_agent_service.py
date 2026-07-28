"""Production composition and synchronous use of one coding-agent action path."""

from __future__ import annotations

import os
import stat
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from kapso.cross_run.docker.runtime import DockerImageAuthority, PinnedDockerRuntime
from kapso.cross_run.launch.boundary import publish_run_boundary
from kapso.cross_run.launch.handoff import PreparedRunHandoff
from kapso.cross_run.launch.production import ProductionLaunchServices
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentInterpretationPolicy,
    CodingAgentProviderEgressMode,
    CodingAgentRunActionRequest,
    CodingAgentRunActionResultEnvelope,
    read_canonical_coding_agent_result,
)
from kapso.cross_run.launch.run_action_coding_agent_credential import (
    NativeCodexCredentialBroker,
)
from kapso.cross_run.launch.run_action_coding_agent_egress import (
    NativeCodingAgentEgressBroker,
)
from kapso.cross_run.launch.run_action_coding_agent_interpreter import (
    CodingAgentRunActionResultInterpreter,
)
from kapso.cross_run.launch.run_action_coding_agent_production import (
    build_coding_agent_boundary_identity,
    build_coding_agent_execution_policy,
    build_coding_agent_interpretation_policy,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialBrokerRegistry,
)
from kapso.cross_run.launch.run_action_docker_adapter import (
    DockerRunActionExecutionAdapter,
)
from kapso.cross_run.launch.run_action_docker_cleanup import (
    DockerRunActionCleanupManager,
    issue_docker_run_action_resource_finalization_authority,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_gate import RunFrontierActionGate
from kapso.cross_run.launch.run_action_recovery import (
    RunActionRecoveryCoordinator,
    RunActionRecoveryImplementation,
    RunActionRecoveryImplementationRegistry,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
)
from kapso.cross_run.launch.run_state_publisher import ReconciledRunFrontier
from kapso.cross_run.settings import CodingAgentSettings, CrossRunSettings


class ProductionCodingAgentActionError(RuntimeError):
    """The production action cannot preserve one exact run authority."""


@dataclass(frozen=True)
class ProductionCodingAgentActionResult:
    """One accepted provider result and its reconciled successor frontier."""

    result: CodingAgentRunActionResultEnvelope
    frontier: ReconciledRunFrontier

    def __post_init__(self) -> None:
        if (
            type(self.result) is not CodingAgentRunActionResultEnvelope
            or type(self.frontier) is not ReconciledRunFrontier
        ):
            raise ProductionCodingAgentActionError(
                "coding-agent result contains substituted authority"
            )


class ProductionCodingAgentAction:
    """Sole synchronous caller of the durable eight-event coding-agent path."""

    def __init__(
        self,
        *,
        handoff: PreparedRunHandoff,
        services: ProductionLaunchServices,
        gate: RunFrontierActionGate,
        coordinator: RunActionRecoveryCoordinator,
        interpretation_policy: CodingAgentInterpretationPolicy,
        boundary_identity: RunActionBoundaryIdentity,
        egress_broker: NativeCodingAgentEgressBroker | None,
        recovery_poll_interval_seconds: int,
    ) -> None:
        if (
            type(handoff) is not PreparedRunHandoff
            or type(services) is not ProductionLaunchServices
            or type(gate) is not RunFrontierActionGate
            or type(coordinator) is not RunActionRecoveryCoordinator
            or coordinator._active_workspace is not handoff.active_workspace
            or coordinator._publisher is not handoff.publisher
            or type(interpretation_policy) is not CodingAgentInterpretationPolicy
            or type(boundary_identity) is not RunActionBoundaryIdentity
            or type(recovery_poll_interval_seconds) is not int
            or recovery_poll_interval_seconds <= 0
        ):
            raise ProductionCodingAgentActionError(
                "coding-agent action contains mixed run authority"
            )
        self._handoff = handoff
        self._services = services
        self._gate = gate
        self._coordinator = coordinator
        self.interpretation_policy = interpretation_policy
        self.boundary_identity = boundary_identity
        self._egress_broker = egress_broker
        self._recovery_poll_interval_seconds = recovery_poll_interval_seconds
        self._owner_process_id = os.getpid()
        self._closed = False

    def execute(
        self,
        *,
        frontier: ReconciledRunFrontier,
        request: CodingAgentRunActionRequest,
    ) -> ProductionCodingAgentActionResult:
        """Reserve, recover to terminal, and reconcile one complete action."""

        self._require_current()
        if (
            type(frontier) is not ReconciledRunFrontier
            or type(request) is not CodingAgentRunActionRequest
            or request.interpretation_policy != self.interpretation_policy
            or request.interpretation_policy.workspace_access
            is not self.interpretation_policy.workspace_access
        ):
            raise ProductionCodingAgentActionError(
                "coding-agent request differs from its production action"
            )
        request.require_policy(self.interpretation_policy)
        boundary = _safety_boundary(self.interpretation_policy.workspace_access)
        action_frontier = publish_run_boundary(
            publisher=self._handoff.publisher,
            frontier=frontier,
            security_authority=self._services.security_authority,
            release_use_authority=self._services.release_use_authority,
            boundary=boundary,
        )
        self._gate.reserve(
            action_frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=boundary,
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            workspace_access=self.interpretation_policy.workspace_access,
            boundary_identity=self.boundary_identity,
        )
        while True:
            self._require_current()
            report = self._coordinator.recover(action_frontier)
            if report.is_complete:
                break
            time.sleep(self._recovery_poll_interval_seconds)
        matching = tuple(
            recovered
            for recovered in report.recovered_operations
            if recovered.events[0].reservation.intent.operation_id
            == request.operation_id
        )
        if (
            len(matching) != 1
            or matching[0].events[-1].event_kind
            is not RunActionExecutionEventKind.RESULT_ACCEPTED
            or matching[0].accepted_result_payload is None
        ):
            raise ProductionCodingAgentActionError(
                "coding-agent action did not yield one accepted terminal result"
            )
        result = read_canonical_coding_agent_result(matching[0].accepted_result_payload)
        result.validate_against(
            policy=self.interpretation_policy,
            request=request,
        )
        reconciled = publish_run_boundary(
            publisher=self._handoff.publisher,
            frontier=action_frontier,
            security_authority=self._services.security_authority,
            release_use_authority=self._services.release_use_authority,
            boundary=boundary,
        )
        return ProductionCodingAgentActionResult(
            result=result,
            frontier=reconciled,
        )

    def close(self) -> None:
        """Release the host broker without closing the caller-owned handoff."""

        if self._closed:
            return
        if self._owner_process_id != os.getpid():
            raise ProductionCodingAgentActionError(
                "coding-agent action cannot close from another process"
            )
        if self._egress_broker is not None:
            self._egress_broker.close()
        self._closed = True

    def _require_current(self) -> None:
        if self._closed or self._owner_process_id != os.getpid():
            raise ProductionCodingAgentActionError(
                "coding-agent action is closed, cloned, or foreign"
            )
        self._handoff.active_workspace.require_control_authority()
        if self._egress_broker is not None:
            self._egress_broker.require_current()

    def __enter__(self) -> ProductionCodingAgentAction:
        self._require_current()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def build_production_coding_agent_action(
    *,
    handoff: PreparedRunHandoff,
    services: ProductionLaunchServices,
    settings: CrossRunSettings,
    image_authority: DockerImageAuthority,
    agent: CodingAgentSettings,
    state_root: Path,
    principal_id: str,
    role: str,
    workspace_access: RunFrontierWorkspaceAccess,
    web_search_enabled: bool,
    provider_network_enabled: bool,
    native_credential_enabled: bool,
) -> ProductionCodingAgentAction:
    """Compose the existing gate, adapter, interpreter, cleanup, and brokers."""

    if (
        type(handoff) is not PreparedRunHandoff
        or type(services) is not ProductionLaunchServices
        or type(settings) is not CrossRunSettings
        or type(image_authority) is not DockerImageAuthority
        or type(agent) is not CodingAgentSettings
        or not isinstance(state_root, Path)
        or handoff.publisher._settings is not settings.launch
        or native_credential_enabled
        and agent.cli != "codex"
        or provider_network_enabled
        and not native_credential_enabled
    ):
        raise ProductionCodingAgentActionError(
            "coding-agent production composition requires exact launch authority"
        )
    _require_private_directory(state_root)
    with ExitStack() as resources:
        egress_broker = (
            NativeCodingAgentEgressBroker(
                settings=settings.launch,
                state_root=state_root,
            )
            if provider_network_enabled
            else None
        )
        if egress_broker is not None:
            resources.callback(egress_broker.close)
        interpretation_policy = build_coding_agent_interpretation_policy(
            settings=settings,
            agent=agent,
            principal_id=principal_id,
            role=role,
            workspace_access=workspace_access,
            web_search_enabled=web_search_enabled,
            provider_network_enabled=provider_network_enabled,
        )
        credential_mode = (
            RunActionCredentialMode.SUPERVISOR_FILE
            if native_credential_enabled
            else RunActionCredentialMode.NONE
        )
        execution_policy, command = build_coding_agent_execution_policy(
            settings=settings,
            image_authority=image_authority,
            interpretation_policy=interpretation_policy,
            credential_mode=credential_mode,
            egress_broker_socket_source_path=(
                None if egress_broker is None else egress_broker.socket_path.as_posix()
            ),
        )
        boundary_identity = build_coding_agent_boundary_identity(
            execution_policy,
            interpretation_policy,
        )
        runtime_root = state_root / "docker-runtime"
        _require_private_directory(runtime_root)
        runtime = PinnedDockerRuntime.create(
            trusted_root=runtime_root,
            settings=settings.docker,
        )
        resource_manager = DockerRunActionResourceManager(runtime)
        cleanup_manager = DockerRunActionCleanupManager(runtime)
        finalization_authority = (
            issue_docker_run_action_resource_finalization_authority(
                action_store=handoff.publisher._action_store,
                launch_settings=settings.launch,
                resource_manager=resource_manager,
                cleanup_manager=cleanup_manager,
            )
        )
        credential_backends = (
            (
                NativeCodexCredentialBroker(
                    settings=settings.launch,
                    state_root=state_root,
                ),
            )
            if native_credential_enabled
            else ()
        )
        credential_registry = RunActionCredentialBrokerRegistry(credential_backends)
        credential_registry.require_policy(execution_policy.credential_policy)
        gate = RunFrontierActionGate(
            active_workspace=handoff.active_workspace,
            publisher=handoff.publisher,
            security_authority=services.security_authority,
            credential_broker_registry=credential_registry,
            resource_finalization_authority=finalization_authority,
        )
        adapter = DockerRunActionExecutionAdapter(
            execution_lifecycle_identity=(
                boundary_identity.execution_lifecycle_identity
            ),
            execution_policy=execution_policy,
            command=command,
            runtime=runtime,
            launch_settings=settings.launch,
        )
        interpreter = CodingAgentRunActionResultInterpreter(
            result_interpreter_identity=(boundary_identity.result_interpreter_identity),
            interpretation_policy=interpretation_policy,
        )
        coordinator = gate.recovery_coordinator(
            RunActionRecoveryImplementationRegistry(
                (
                    RunActionRecoveryImplementation(
                        boundary_identity=boundary_identity,
                        execution_adapter=adapter,
                        result_interpreter=interpreter,
                    ),
                )
            )
        )
        action = ProductionCodingAgentAction(
            handoff=handoff,
            services=services,
            gate=gate,
            coordinator=coordinator,
            interpretation_policy=interpretation_policy,
            boundary_identity=boundary_identity,
            egress_broker=egress_broker,
            recovery_poll_interval_seconds=(
                settings.docker.run_action_barrier_poll_interval_seconds
            ),
        )
        resources.pop_all()
        return action


def _safety_boundary(
    workspace_access: RunFrontierWorkspaceAccess,
) -> RunSafetyBoundary:
    if workspace_access is RunFrontierWorkspaceAccess.READ_ONLY:
        return RunSafetyBoundary.IDEATION
    if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
        return RunSafetyBoundary.IMPLEMENTATION
    raise ProductionCodingAgentActionError(
        "coding-agent action requires read-only or edit workspace authority"
    )


def _require_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    metadata = os.stat(path, follow_symlinks=False)
    if (
        path.resolve() != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ProductionCodingAgentActionError(
            "coding-agent action state directory is unsafe"
        )


__all__ = [
    "build_production_coding_agent_action",
    "ProductionCodingAgentAction",
    "ProductionCodingAgentActionError",
    "ProductionCodingAgentActionResult",
]
