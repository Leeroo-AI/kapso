"""Explicit real-Docker validation for issued run-action create projections.

Run directly:

    pytest -q tests/live_run_action_docker_projection.py -s
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

import kapso.cross_run.launch.run_action_docker_adapter as docker_adapter_module
from expert_live_docker_support import (
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.core.config import load_config
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerCleanupAuthority,
    PinnedDockerContainmentAuthority,
    PinnedDockerPreparationAuthority,
    PinnedDockerRuntime,
    PinnedDockerStartAuthority,
    read_verified_root_executable,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
    keeper_create_arguments,
    main_create_arguments,
    require_run_action_image,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_docker_adapter import (
    DockerRunActionExecutionAdapter,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_REQUEST_PROTOCOL_VERSION,
    CODING_AGENT_RESULT_PROTOCOL_VERSION,
    CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
    CodingAgentInterpretationPolicy,
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_coding_agent_interpreter import (
    CodingAgentRunActionResultInterpreter,
    FixedOfflineCodingAgentConsumer,
    coding_agent_result_interpreter_identity,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialBrokerBackend,
    RunActionCredentialBrokerRegistry,
    RunActionCredentialIssueResponse,
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_credential_retirement import (
    DockerRunActionCredentialRetirementManager,
    retire_run_action_expired_credential_once,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionResolvedMountKind,
)
from kapso.cross_run.launch.run_action_activation_envelope import (
    activation_execution_event_size_bound,
)
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PREPARATION_AUTHORITY,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionInterpretedResult,
    RunActionPreparationCapability,
    RunActionPreparationMode,
    RunActionPreparationOrigin,
    RunActionPreparationState,
    RunActionRecoveryImplementation,
    RunActionRecoveryImplementationRegistry,
)
from kapso.cross_run.launch.run_action_natural_terminal import (
    resolve_run_action_natural_terminal_once,
)
from kapso.cross_run.launch.run_action_main_start import (
    DockerRunActionStartManager,
    inspect_run_action_inert_activation,
    RunActionMainStartError,
    start_run_action_barrier_once,
)
from kapso.cross_run.launch.run_action_pre_release_main_loss import (
    capture_run_action_pre_release_main_loss_termination,
    inspect_run_action_pre_release_main_loss,
)
from kapso.cross_run.launch.run_action_pre_release_main_terminal import (
    capture_run_action_pre_release_main_terminal_termination,
    inspect_run_action_pre_release_main_terminal,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    open_run_action_blocked_workload,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_release_envelope import (
    workload_release_receipt_size_bound,
)
from kapso.cross_run.launch.run_action_release_publisher import (
    publish_run_action_workload_release_once,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    inspect_run_action_terminal,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_loss_observation_token,
    run_action_pre_release_main_terminal_observation_token,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionExecutionEventKind,
    RunActionResultDisposition,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_docker_preparation import (
    DockerRunActionPreparationManager,
)
from kapso.cross_run.launch.run_action_docker_cleanup import (
    DockerRunActionCleanupManager,
    issue_docker_run_action_resource_finalization_authority,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionInspectionError,
    observe_inert_keeper,
    observe_inert_main_container,
    observe_running_barrier_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionSupervisorHelperError,
    observe_docker_init_source,
    observe_supervisor_helper,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionRuntimeVolumeError,
    adopt_prepared_runtime_volume_layout,
    deliver_and_reobserve_runtime_volume_activation,
    materialize_runtime_volume_layout,
    observe_empty_runtime_volume,
    reobserve_runtime_volume_layout,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionStaticEnvironmentVariable,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
    run_action_credential_lease_authority_id,
    run_action_credential_lease_request,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.process import BoundedProcessOutcome
from live_expert_replay_docker import _start_local_oci_registry
from test_launch_resolver import resolver_case
from test_run_action_docker_projection import (
    _GENERATION_NONCE,
    _policy,
)
from test_run_action_recovery import _recovery_coordinator
from test_run_frontier_action_gate import _action_case, _boundary_identity
from test_run_action_supervisor_contracts import (
    _claim,
    _remint_contract,
    _remint_policy,
    _volume_authority,
)
from test_run_state_publisher import publisher_case

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ORIGINAL_SUBPROCESS_RUN = subprocess.run
_LIVE_RESULT_PAYLOAD = b'{"live":"captured"}'


class _UnusedLiveResultInterpreter:
    def __init__(self, result_interpreter_identity) -> None:
        self.result_interpreter_identity = result_interpreter_identity

    def interpret(self, *, operation_id, request_payload, result_payload):
        raise AssertionError("blocked-workload proof must not interpret a result")


class _LiveAcceptedResultInterpreter:
    def __init__(self, result_interpreter_identity) -> None:
        self.result_interpreter_identity = result_interpreter_identity

    def interpret(self, *, operation_id, request_payload, result_payload):
        if request_payload != b"complete request":
            raise AssertionError("live result interpreter received another request")
        if result_payload != _LIVE_RESULT_PAYLOAD:
            raise AssertionError("live result interpreter received another result")
        return RunActionInterpretedResult(
            disposition=RunActionResultDisposition.SUCCEEDED,
            operation_id=operation_id,
            accepted_result_payload=result_payload,
        )


def _production_recovery_coordinator(
    gate,
    boundary_identity,
    execution_adapter,
    result_interpreter,
):
    return gate.recovery_coordinator(
        RunActionRecoveryImplementationRegistry(
            (
                RunActionRecoveryImplementation(
                    boundary_identity=boundary_identity,
                    execution_adapter=execution_adapter,
                    result_interpreter=result_interpreter,
                ),
            )
        )
    )


class _LiveCredentialBrokerBackend(RunActionCredentialBrokerBackend):
    def __init__(self, maximum_lease_seconds) -> None:
        super().__init__(
            broker_id="test.credential.broker",
            broker_protocol_version="test.credential.broker.v1",
        )
        self._maximum_lease_seconds = maximum_lease_seconds
        self.issue_calls = []
        self.status_calls = []
        self._lease_expiries = {}

    def _valid_until(self, request):
        existing = self._lease_expiries.get(request.credential_lease_request_id)
        if existing is not None:
            return existing
        valid_until = time.time_ns() + (self._maximum_lease_seconds - 1) * 1_000_000_000
        self._lease_expiries[request.credential_lease_request_id] = valid_until
        return valid_until

    def issue_or_replay_exact(self, request):
        self.issue_calls.append(request)
        return RunActionCredentialIssueResponse(
            credential_lease_request_id=request.credential_lease_request_id,
            payload=b"credential bytes",
            valid_until_realtime_nanoseconds=self._valid_until(request),
        )

    def observe_exact(self, request):
        self.status_calls.append(request)
        return RunActionCredentialLeaseStatus.mint(
            credential_lease_request_id=request.credential_lease_request_id,
            valid_until_realtime_nanoseconds=self._valid_until(request),
        )


class _LiveInertStartAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        start_manager,
        preparation_allocation,
        command,
        volume_observation,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        launch_settings,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _UnusedLiveResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self._resource_manager = resource_manager
        self._start_manager = start_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._volume_observation = volume_observation
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self.running_observation = None

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        return inspect_run_action_inert_activation(
            query=query,
            resource_manager=self._resource_manager,
            launch_settings=self._launch_settings,
        )

    def continue_committed_once(self, capability):
        self.running_observation = start_run_action_barrier_once(
            capability=capability,
            resource_manager=self._resource_manager,
            start_manager=self._start_manager,
            command=self._command,
            volume_observation=self._volume_observation,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _LiveExpiredCredentialRetirementAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        retirement_manager,
        preparation_allocation,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        launch_settings,
        running,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _UnusedLiveResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self._resource_manager = resource_manager
        self._retirement_manager = retirement_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self._running = running
        self.retirement_attempted = False

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        if not self._running:
            return inspect_run_action_inert_activation(
                query=query,
                resource_manager=self._resource_manager,
                launch_settings=self._launch_settings,
            )
        inventory = self._resource_manager.observe(self._preparation_allocation)
        volume = observe_runtime_volume(
            self._resource_manager.inspect_volume(inventory),
            self._preparation_allocation.preparation_claim,
            self._preparation_allocation.runtime_volume_authority,
            self._docker_settings,
        )
        running = observe_running_barrier_main_container(
            self._resource_manager.inspect_main(inventory),
            self._preparation_allocation.preparation_claim,
            self._preparation_allocation.runtime_volume_authority,
            volume,
            self._command,
            self._helper_evidence,
            self._init_source_evidence,
            self._docker_settings,
        )
        if running.container_id != query.spawn_commit.provider_execution_id:
            raise AssertionError(
                "expired credential query differs from running container"
            )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=running.complete_inspection_digest,
        )

    def continue_committed_once(self, capability):
        retire_run_action_expired_credential_once(
            capability=capability,
            resource_manager=self._resource_manager,
            retirement_manager=self._retirement_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        self.retirement_attempted = True
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _LiveBlockedWorkloadAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        preparation_allocation,
        command,
        volume_observation,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        launch_settings,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _UnusedLiveResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self._resource_manager = resource_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._volume_observation = volume_observation
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self._committed_running_observation = None
        self.lease = None
        self.resolved_workload_observation = None
        self.release_receipt = None

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        if query.preparation_allocation != self._preparation_allocation:
            raise AssertionError("committed live query differs from exact allocation")
        inventory = self._resource_manager.observe(self._preparation_allocation)
        running = observe_running_barrier_main_container(
            self._resource_manager.inspect_main(inventory),
            self._preparation_allocation.preparation_claim,
            self._preparation_allocation.runtime_volume_authority,
            self._volume_observation,
            self._command,
            self._helper_evidence,
            self._init_source_evidence,
            self._docker_settings,
        )
        if running.container_id != query.spawn_commit.provider_execution_id:
            raise AssertionError("committed live query differs from running container")
        self._committed_running_observation = running
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=running.complete_inspection_digest,
        )

    def continue_committed_once(self, capability):
        if self._committed_running_observation is None:
            raise AssertionError("committed running observation was not sealed")
        self.lease = open_run_action_blocked_workload(
            capability,
            committed_running_observation=self._committed_running_observation,
            resource_manager=self._resource_manager,
            preparation_allocation=self._preparation_allocation,
            command=self._command,
            volume_observation=self._volume_observation,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        with self.lease:
            self.resolved_workload_observation = (
                self.lease.resolved_workload_observation
            )
            self.release_receipt = publish_run_action_workload_release_once(
                capability=capability,
                blocked_workload_lease=self.lease,
            )
        if self.release_receipt is None:
            raise AssertionError("live blocked workload lost release authorization")
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _LiveNaturalTerminalWorkloadAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        preparation_allocation,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        launch_settings,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _LiveAcceptedResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self._resource_manager = resource_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self.terminal_observation = None
        self.reinspected_terminal_observation = None
        self.captured_result = None
        self.provider_termination_receipt = None

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        if query.preparation_allocation != self._preparation_allocation:
            raise AssertionError("terminal live query differs from exact allocation")
        self.terminal_observation = inspect_run_action_terminal(
            query=query,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=(self.terminal_observation.complete_inspection_digest),
        )

    def continue_committed_once(self, capability):
        if self.terminal_observation is None:
            raise AssertionError("terminal observation was not sealed")
        outcome = resolve_run_action_natural_terminal_once(
            capability=capability,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        if outcome.state is RunActionContinuationState.RESULT_CAPTURED:
            self.reinspected_terminal_observation = outcome.result.terminal_observation
            self.captured_result = outcome.result
        elif outcome.state is RunActionContinuationState.PROVIDER_TERMINATED:
            self.provider_termination_receipt = outcome.provider_termination_receipt
            self.reinspected_terminal_observation = (
                self.provider_termination_receipt.terminal_observation
            )
        else:
            raise AssertionError("live natural terminal remained unresolved")
        if self.reinspected_terminal_observation != self.terminal_observation:
            raise AssertionError("terminal reinspection changed its occurrence")
        return outcome


class _LivePreReleaseMainLossAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        preparation_allocation,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _UnusedLiveResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self._resource_manager = resource_manager
        self._preparation_allocation = preparation_allocation
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self.loss_observation = None
        self.termination_receipt = None

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        if query.preparation_allocation != self._preparation_allocation:
            raise AssertionError("main-loss live query differs from exact allocation")
        self.loss_observation = inspect_run_action_pre_release_main_loss(
            query=query,
            resource_manager=self._resource_manager,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
            observation_token=run_action_pre_release_main_loss_observation_token(
                self.loss_observation
            ),
        )

    def continue_committed_once(self, capability):
        if self.loss_observation is None:
            raise AssertionError("pre-release main loss was not classified")
        outcome = capture_run_action_pre_release_main_loss_termination(
            capability=capability,
            resource_manager=self._resource_manager,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
        )
        self.termination_receipt = outcome.provider_termination_receipt
        return outcome


class _LivePreReleaseMainTerminalAdapter(_LivePreReleaseMainLossAdapter):
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        preparation_allocation,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        launch_settings,
    ) -> None:
        super().__init__(
            boundary_identity=boundary_identity,
            execution_policy=execution_policy,
            resource_manager=resource_manager,
            preparation_allocation=preparation_allocation,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
        )
        self._command = command
        self._launch_settings = launch_settings
        self.terminal_observation = None

    def inspect_committed(self, query):
        if query.preparation_allocation != self._preparation_allocation:
            raise AssertionError(
                "pre-release terminal query differs from exact allocation"
            )
        self.terminal_observation = inspect_run_action_pre_release_main_terminal(
            query=query,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        return RunActionCommittedSpawnObservation(
            state=(RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE),
            observation_token=(
                run_action_pre_release_main_terminal_observation_token(
                    self.terminal_observation
                )
            ),
        )

    def continue_committed_once(self, capability):
        if self.terminal_observation is None:
            raise AssertionError("pre-release terminal was not classified")
        outcome = capture_run_action_pre_release_main_terminal_termination(
            capability=capability,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        self.termination_receipt = outcome.provider_termination_receipt
        return outcome


def _remove_owned_container(
    settings,
    docker_config_root: Path,
    container_name: str,
    labels,
) -> None:
    arguments = [
        "container",
        "ls",
        "--all",
        "--no-trunc",
        "--quiet",
        "--filter",
        f"name=^/{container_name}$",
    ]
    for label in labels:
        arguments.extend(("--filter", f"label={label.key}={label.value}"))
    observation = run_setup_docker(
        settings,
        docker_config_root,
        tuple(arguments),
    )
    require_setup_docker_success(observation, "run-action container cleanup lookup")
    if observation.stdout == b"":
        return
    container_ids = observation.stdout.decode("ascii").splitlines()
    if (
        len(container_ids) != 1
        or _CONTAINER_ID_PATTERN.fullmatch(container_ids[0]) is None
    ):
        raise AssertionError("run-action cleanup container lookup was ambiguous")
    container_id = container_ids[0]
    label_observation = run_setup_docker(
        settings,
        docker_config_root,
        (
            "container",
            "inspect",
            "--format",
            "{{json .Config.Labels}}",
            container_id,
        ),
    )
    require_setup_docker_success(
        label_observation,
        "run-action container cleanup label inspection",
    )
    if json.loads(label_observation.stdout) != {
        label.key: label.value for label in labels
    }:
        raise AssertionError("run-action cleanup container labels differ")
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("container", "rm", "--force", "--volumes", container_id),
    )
    require_setup_docker_success(removal, "run-action container cleanup")


def _remove_owned_volume(
    settings,
    docker_config_root: Path,
    volume_name: str,
    labels,
) -> None:
    arguments = [
        "volume",
        "ls",
        "--quiet",
        "--filter",
        f"name=^{volume_name}$",
    ]
    for label in labels:
        arguments.extend(("--filter", f"label={label.key}={label.value}"))
    result = run_setup_docker(settings, docker_config_root, tuple(arguments))
    require_setup_docker_success(result, "run-action volume cleanup lookup")
    if result.stdout == b"":
        return
    if result.stdout != f"{volume_name}\n".encode("ascii"):
        raise AssertionError("run-action cleanup volume lookup was ambiguous")
    label_observation = run_setup_docker(
        settings,
        docker_config_root,
        (
            "volume",
            "inspect",
            "--format",
            "{{json .Labels}}",
            volume_name,
        ),
    )
    require_setup_docker_success(
        label_observation,
        "run-action volume cleanup label inspection",
    )
    if json.loads(label_observation.stdout) != {
        label.key: label.value for label in labels
    }:
        raise AssertionError("run-action cleanup volume labels differ")
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("volume", "rm", "--force", volume_name),
    )
    require_setup_docker_success(removal, "run-action volume cleanup")


def _listed_exact(
    settings,
    docker_config_root: Path,
    arguments: tuple[str, ...],
) -> tuple[str, ...]:
    result = run_setup_docker(settings, docker_config_root, arguments)
    require_setup_docker_success(result, "run-action projection inventory")
    return tuple(line.decode("ascii") for line in result.stdout.splitlines())


@pytest.mark.parametrize(
    "terminal_path",
    (
        "result",
        "ambiguous_start",
        "timeout",
        "empty",
        "nonzero",
        "oom",
        "pre_release_main_loss",
        "pre_release_main_terminal",
        "expired_inert_credential",
        "expired_running_credential",
        "frontier_invalidated",
        "allocation_volume",
        "allocation_created_keeper",
        "allocation_running_keeper",
        "allocation_inert_main",
        "production_adapter_result",
        "production_preparation",
        "production_preparation_ambiguous",
    ),
)
def test_real_docker_accepts_only_the_issued_run_action_projection(
    tmp_path: Path,
    publisher_case,
    monkeypatch,
    terminal_path,
) -> None:
    monkeypatch.setattr(subprocess, "run", _ORIGINAL_SUBPROCESS_RUN)
    tmp_path.chmod(0o700)
    cross_run_settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    settings = cross_run_settings.docker
    busybox_bytes = read_verified_root_executable(
        Path(settings.helper_executable_path),
        settings.helper_executable_digest,
    )
    coding_agent_policy = None
    coding_agent_request = None
    coding_agent_result_payload = None
    if terminal_path == "production_adapter_result":
        coding_agent_policy = CodingAgentInterpretationPolicy.mint(
            request_protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
            result_protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
            schema_protocol_version=CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
            consumer_id="kapso.offline_coding_agent_consumer",
            consumer_version="v1",
            principal_id="kapso.live_validation",
            role="live_coding_agent_validation",
            cli="codex",
            model="gpt-5.6-sol",
            effort="xhigh",
            allowed_tools=("Read",),
            timeout_nanoseconds=(
                _policy(settings).supervisor_limits.execution_timeout_seconds
                * 1_000_000_000
            ),
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            maximum_raw_result_bytes=(
                cross_run_settings.launch.run_action_result_size_bytes
            ),
        )
        coding_agent_request = CodingAgentRunActionRequest(
            protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
            interpretation_policy_id=coding_agent_policy.interpretation_policy_id,
            operation_id="agent_call_" + "a" * 32,
            prompt="Return the fixed live-validation result object.",
            response_schema={
                "type": "object",
                "properties": {"live": {"type": "string", "enum": ["captured"]}},
                "required": ["live"],
                "additionalProperties": False,
            },
            prior_knowledge=None,
            edit_predecessor_source_tree_digest=None,
        )
        coding_agent_result_payload = FixedOfflineCodingAgentConsumer(
            interpretation_policy=coding_agent_policy,
            structured_output={"live": "captured"},
            duration_nanoseconds=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=None,
            prior_knowledge_accesses=(),
            edited_source_tree_digest=None,
        ).consume(coding_agent_request.to_json_bytes())

    with ExitStack() as cleanup:
        local_registry = _start_local_oci_registry(cleanup, busybox_bytes)
        docker_config_root = tmp_path / "setup-docker-config"
        docker_config_root.mkdir(mode=0o700)
        docker_config_path = docker_config_root / "config.json"
        docker_config_path.write_bytes(b'{"auths":{}}\n')
        docker_config_path.chmod(0o400)
        cleanup.callback(
            remove_exact_image,
            settings,
            docker_config_root,
            local_registry.image_reference,
        )
        pull_result = run_setup_docker(
            settings,
            docker_config_root,
            (
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                local_registry.image_reference,
            ),
        )
        require_setup_docker_success(pull_result, "run-action projection image")
        assert local_registry.server.observed_violations == ()

        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir(mode=0o700)
        runtime = PinnedDockerRuntime.create(
            trusted_root=runtime_root.resolve(),
            settings=settings,
        )
        resource_manager = DockerRunActionResourceManager(runtime)
        cleanup_manager = DockerRunActionCleanupManager(runtime)
        image_authority = DockerImageAuthority.mint(
            image_reference=local_registry.image_reference,
            image_config_digest=local_registry.config_digest,
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
        )
        target_command = (
            "printf started > /kapso/tmp/target-started"
            " && grep -Fqx 'complete request' /kapso/input/request.blob"
            " && grep -Fqx 'credential bytes'"
            " /kapso/credentials/credentials"
            " && test -d /kapso/workspace/.git"
        )
        if terminal_path in {
            "result",
            "ambiguous_start",
            "timeout",
        }:
            target_command = (
                'printf \'{"live":"captured"}\''
                " > /kapso/result/result.blob"
                f" && {target_command}"
            )
        if terminal_path == "production_adapter_result":
            request_digest = coding_agent_request.request_digest.removeprefix("sha256:")
            encoded_result = base64.b64encode(coding_agent_result_payload).decode(
                "ascii"
            )
            target_command = (
                f"printf '%s' '{encoded_result}'"
                " | /bin/busybox base64 -d"
                " > /kapso/result/result.blob"
                f" && printf '%s  %s\\n' '{request_digest}'"
                " /kapso/input/request.blob"
                " | /bin/busybox sha256sum -c -"
                " && test -d /kapso/workspace/.git"
            )
        if terminal_path == "timeout":
            target_command += (
                " && exec /bin/busybox sleep " f"{2 * settings.command_timeout_seconds}"
            )
        elif terminal_path == "nonzero":
            target_command += " && exit 23"
        elif terminal_path == "oom":
            target_command += (
                " && exec /bin/busybox awk"
                " 'BEGIN { value=\"x\"; while (1) value=value value }'"
            )
        command = DockerRunActionCommand.build(
            entrypoint="/bin/busybox",
            arguments=(
                "sh",
                "-c",
                target_command,
                "target; printf injected > /kapso/tmp/barrier-injection",
            ),
        )
        policy = _remint_policy(
            _policy(settings),
            image_authority=image_authority,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(key="PATH", value="/bin"),
            ),
        )
        helper_evidence = observe_supervisor_helper(policy)
        init_source_evidence = observe_docker_init_source(policy)
        claim = _claim(policy=policy)
        authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
        allocation = RunActionPreparationAllocation.mint(
            preparation_claim=claim,
            runtime_volume_authority=authority,
        )
        claim = allocation.preparation_claim
        authority = allocation.runtime_volume_authority
        main_name = preparation_container_name(claim)
        main_labels = preparation_container_labels(claim)
        keeper_name = preparation_keeper_container_name(claim)
        keeper_labels = preparation_keeper_container_labels(claim)
        volume_name = preparation_volume_name(claim)
        volume_labels = preparation_volume_labels(
            claim,
            authority.generation_nonce,
        )
        for name in (main_name, keeper_name):
            assert (
                _listed_exact(
                    settings,
                    docker_config_root,
                    (
                        "container",
                        "ls",
                        "--all",
                        "--quiet",
                        "--filter",
                        f"name=^/{name}$",
                    ),
                )
                == ()
            )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{volume_name}$",
                ),
            )
            == ()
        )
        assert resource_manager.observe(allocation).is_absent

        image = runtime.inspect_exact_image(image_authority)
        require_run_action_image(image, policy, settings)

        cleanup.callback(
            _remove_owned_volume,
            settings,
            docker_config_root,
            volume_name,
            volume_labels,
        )
        volume_result = runtime.run_control(
            volume_create_arguments(claim, authority, settings)
        )
        assert volume_result.stdout == f"{volume_name}\n".encode("ascii")
        volume_inventory = resource_manager.observe(allocation)
        assert volume_inventory.volume_present is True
        assert volume_inventory.keeper_container_id is None
        assert volume_inventory.main_container_id is None
        volume_observation = observe_runtime_volume(
            resource_manager.inspect_volume(volume_inventory),
            claim,
            authority,
            settings,
        )
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            keeper_name,
            keeper_labels,
        )
        keeper_result = runtime.run_control(
            keeper_create_arguments(
                claim,
                authority,
                image,
                settings,
            )
        )
        keeper_id = keeper_result.stdout.decode("ascii").strip()
        assert _CONTAINER_ID_PATTERN.fullmatch(keeper_id) is not None
        inert_keeper_inventory = resource_manager.observe(allocation)
        assert inert_keeper_inventory.volume_present is True
        assert inert_keeper_inventory.keeper_container_id == keeper_id
        assert inert_keeper_inventory.main_container_id is None
        inert_keeper = observe_inert_keeper(
            resource_manager.inspect_keeper(inert_keeper_inventory),
            claim,
            authority,
            volume_observation,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        assert inert_keeper.container_id == keeper_id
        started_keeper = runtime.run_control(("container", "start", keeper_id))
        assert started_keeper.stdout == f"{keeper_id}\n".encode("ascii")
        empty_volume_inventory = resource_manager.observe(allocation)
        assert empty_volume_inventory.volume_present is True
        assert empty_volume_inventory.keeper_container_id == keeper_id
        assert empty_volume_inventory.main_container_id is None
        empty_volume_keeper = observe_running_keeper(
            resource_manager.inspect_keeper(empty_volume_inventory),
            claim,
            authority,
            volume_observation,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        empty_volume = observe_empty_runtime_volume(
            authority,
            volume_observation,
            empty_volume_keeper,
        )
        assert empty_volume.keeper_container_id == keeper_id
        assert empty_volume.filesystem_type == "tmpfs"
        assert empty_volume.observed_mount_flags == (
            "nodev",
            "nosuid",
            "noswap",
        )
        assert empty_volume.empty_entry_count == 0
        assert empty_volume.empty_size_bytes == 0
        assert (
            empty_volume.used_size_bytes + empty_volume.available_size_bytes
            == empty_volume.effective_size_bytes
        )
        assert (
            empty_volume.used_inode_count + empty_volume.available_inode_count
            == empty_volume.effective_inode_limit
        )
        runtime.run_control(
            (
                "container",
                "exec",
                keeper_id,
                "/kapso-supervisor/busybox",
                "mkdir",
                "-p",
                "/kapso/runtime-volume/credential",
                "/kapso/runtime-volume/control",
                "/kapso/runtime-volume/input",
                "/kapso/runtime-volume/result",
                "/kapso/runtime-volume/temporary",
                "/kapso/runtime-volume/workspace",
            )
        )
        keeper_inventory = resource_manager.observe(allocation)
        assert keeper_inventory.volume_present is True
        assert keeper_inventory.keeper_container_id == keeper_id
        assert keeper_inventory.main_container_id is None

        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            main_name,
            main_labels,
        )
        main_result = runtime.run_control(
            main_create_arguments(
                claim,
                authority,
                command,
                image,
                settings,
            )
        )
        main_id = main_result.stdout.decode("ascii").strip()
        assert _CONTAINER_ID_PATTERN.fullmatch(main_id) is not None

        complete_inventory = resource_manager.observe(allocation)
        assert complete_inventory.volume_present is True
        assert complete_inventory.keeper_container_id == keeper_id
        assert complete_inventory.main_container_id == main_id
        keeper = resource_manager.inspect_keeper(complete_inventory)
        main = resource_manager.inspect_main(complete_inventory)
        main_evidence = observe_inert_main_container(
            main,
            claim,
            authority,
            volume_observation,
            command,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        assert main_evidence.container_id == main_id
        assert (
            main_evidence.observed_inspect_projection
            == main_evidence.issued_create_projection
        )
        keeper_evidence = observe_running_keeper(
            keeper,
            claim,
            authority,
            volume_observation,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        assert keeper_evidence.container_id == keeper_id
        assert (
            keeper_evidence.observed_inspect_projection
            == keeper_evidence.issued_create_projection
        )
        substituted_helper_evidence = _remint_contract(
            helper_evidence,
            mount_id=helper_evidence.mount_id + 1,
            device=helper_evidence.device + 1,
            inode=helper_evidence.inode + 1,
        )
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="differs from its issued source inode",
        ):
            observe_running_keeper(
                keeper,
                claim,
                authority,
                volume_observation,
                substituted_helper_evidence,
                init_source_evidence,
                settings,
            )
        assert keeper["State"]["Status"] == "running"
        assert keeper["State"]["Pid"] > 0
        assert keeper["HostConfig"]["NetworkMode"] == "none"
        assert len(keeper["Mounts"]) == 2
        assert main["State"]["Status"] == "created"
        assert main["State"]["Pid"] == 0
        assert main["RestartCount"] == 0
        assert main["Path"] == (
            main_evidence.issued_create_projection.command_executable
        )
        assert tuple(main["Args"]) == (
            main_evidence.issued_create_projection.command_arguments
        )
        host_mounts = main["HostConfig"]["Mounts"]
        assert len(host_mounts) == (
            main_evidence.issued_create_projection.exact_mount_count
        )
        helper_host_mounts = tuple(
            mount for mount in host_mounts if mount["Type"] == "bind"
        )
        assert helper_host_mounts == (
            {
                "Type": "bind",
                "Source": policy.supervisor_helper_source_path,
                "Target": (main_evidence.issued_create_projection.command_executable),
                "ReadOnly": True,
                "BindOptions": {
                    "Propagation": "rprivate",
                    "NonRecursive": True,
                },
            },
        )
        volume_host_mounts = tuple(
            mount for mount in host_mounts if mount["Type"] == "volume"
        )
        assert {mount["VolumeOptions"]["Subpath"] for mount in volume_host_mounts} == {
            "control",
            "credential",
            "input",
            "result",
            "temporary",
            "workspace",
        }
        assert all("Subpath" not in mount for mount in main["Mounts"])
        assert len(main["Mounts"]) == (
            main_evidence.issued_create_projection.exact_mount_count
        )

        runtime.run_control(("container", "rm", "--force", "--volumes", main_id))
        runtime.run_control(("container", "rm", "--force", "--volumes", keeper_id))
        runtime.run_control(("volume", "rm", volume_name))

        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"name=^/{main_name}$",
                ),
            )
            == ()
        )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"name=^/{keeper_name}$",
                ),
            )
            == ()
        )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{volume_name}$",
                ),
            )
            == ()
        )

        workspace_descriptor, _workspace_identity = publisher_case[
            "active"
        ]._open_execution_workspace(cleanup)
        expected_workspace_commit = publisher_case[
            "checkpoint"
        ].safety_state.derivative_frontier.evidence.branch_heads[
            publisher_case["settings"].workspace_git_branch
        ]
        source_frontier = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=expected_workspace_commit,
        )
        workspace_binding = RunActionWorkspaceBinding.from_identity(source_frontier)
        layout_policy = _remint_policy(
            _policy(
                settings,
                credential_mode=(
                    RunActionCredentialMode.NONE
                    if terminal_path == "production_adapter_result"
                    else RunActionCredentialMode.SUPERVISOR_FILE
                ),
            ),
            image_authority=image_authority,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(key="PATH", value="/bin"),
            ),
        )
        credential_backend = _LiveCredentialBrokerBackend(
            layout_policy.credential_policy.maximum_lease_seconds,
        )
        credential_broker_registry = RunActionCredentialBrokerRegistry(
            (credential_backend,)
        )
        (
            _action_publisher,
            action_frontier,
            _security_authority,
            action_gate,
        ) = _action_case(
            publisher_case,
            credential_broker_registry=credential_broker_registry,
            resource_finalization_authority_factory=(
                lambda publisher: issue_docker_run_action_resource_finalization_authority(
                    action_store=publisher._action_store,
                    launch_settings=publisher._settings,
                    resource_manager=resource_manager,
                    cleanup_manager=cleanup_manager,
                )
            ),
        )
        base_boundary_identity = _boundary_identity(
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.READ_ONLY,
        )
        result_interpreter_identity = (
            coding_agent_result_interpreter_identity(coding_agent_policy)
            if terminal_path == "production_adapter_result"
            else base_boundary_identity.result_interpreter_identity
        )
        boundary_identity = RunActionBoundaryIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            execution_lifecycle_identity=_remint_contract(
                base_boundary_identity.execution_lifecycle_identity,
                execution_policy_id=layout_policy.docker_execution_policy_id,
            ),
            result_interpreter_identity=result_interpreter_identity,
        )
        request_payload = (
            coding_agent_request.to_json_bytes()
            if terminal_path == "production_adapter_result"
            else b"complete request"
        )
        operation_id = (
            coding_agent_request.operation_id
            if terminal_path == "production_adapter_result"
            else "live_blocked_workload_0123456789abcdef"
        )
        layout_reservation = action_gate.reserve(
            action_frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id=operation_id,
            request_payload=request_payload,
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=boundary_identity,
        )
        assert layout_reservation.frontier.workspace_before == workspace_binding
        with action_gate._action_store._recovery_session(
            layout_reservation,
            _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
        ) as session:
            layout_allocation = session.allocate_preparation(layout_policy)

        def recover_allocation_invalidation(
            prepared_execution: RunActionPreparedExecution | None = None,
        ) -> None:
            with action_gate._action_store._recovery_session(
                layout_reservation,
                _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
            ) as session:
                if prepared_execution is not None:
                    session.commit_prepared_execution(prepared_execution)
                session.invalidate_frontier()
            inert_adapter = _LiveNaturalTerminalWorkloadAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                preparation_allocation=layout_allocation,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
            )
            invalidated_report = _recovery_coordinator(
                action_gate,
                inert_adapter,
            ).recover(action_frontier)
            assert invalidated_report.is_complete
            assert resource_manager.observe(layout_allocation).is_absent
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )

        layout_claim = layout_allocation.preparation_claim
        layout_authority = layout_allocation.runtime_volume_authority
        layout_volume_name = preparation_volume_name(layout_claim)
        layout_volume_labels = preparation_volume_labels(
            layout_claim,
            layout_authority.generation_nonce,
        )
        layout_keeper_name = preparation_keeper_container_name(layout_claim)
        layout_keeper_labels = preparation_keeper_container_labels(layout_claim)
        layout_main_name = preparation_container_name(layout_claim)
        layout_main_labels = preparation_container_labels(layout_claim)
        if terminal_path in {
            "production_adapter_result",
            "production_preparation",
            "production_preparation_ambiguous",
        }:
            cleanup.callback(
                _remove_owned_volume,
                settings,
                docker_config_root,
                layout_volume_name,
                layout_volume_labels,
            )
            cleanup.callback(
                _remove_owned_container,
                settings,
                docker_config_root,
                layout_keeper_name,
                layout_keeper_labels,
            )
            cleanup.callback(
                _remove_owned_container,
                settings,
                docker_config_root,
                layout_main_name,
                layout_main_labels,
            )
            if terminal_path == "production_adapter_result":
                production_adapter = DockerRunActionExecutionAdapter(
                    execution_lifecycle_identity=(
                        boundary_identity.execution_lifecycle_identity
                    ),
                    execution_policy=layout_policy,
                    command=command,
                    runtime=runtime,
                    launch_settings=cross_run_settings.launch,
                )
                production_interpreter = CodingAgentRunActionResultInterpreter(
                    result_interpreter_identity=(
                        boundary_identity.result_interpreter_identity
                    ),
                    interpretation_policy=coding_agent_policy,
                )
                coordinator = _production_recovery_coordinator(
                    action_gate,
                    boundary_identity,
                    production_adapter,
                    production_interpreter,
                )
                original_delivery = (
                    docker_adapter_module.deliver_and_reobserve_runtime_volume_activation
                )

                def fail_before_activation_delivery(*_arguments, **_keywords):
                    raise RuntimeError("injected death before activation delivery")

                monkeypatch.setattr(
                    docker_adapter_module,
                    "deliver_and_reobserve_runtime_volume_activation",
                    fail_before_activation_delivery,
                )
                with pytest.raises(
                    RuntimeError,
                    match="injected death before activation delivery",
                ):
                    coordinator.recover(action_frontier)
                crash_events = action_gate._action_store.inspect().events_for(
                    layout_reservation.intent.operation_id
                )
                assert len(crash_events) == 4
                assert crash_events[-1].event_kind is (
                    RunActionExecutionEventKind.SPAWN_COMMITTED
                )
                crash_inventory = resource_manager.observe(layout_allocation)
                crash_main = resource_manager.inspect_main(crash_inventory)
                assert crash_main["HostConfig"]["NetworkMode"] == "none"
                assert all(
                    mount.get("VolumeOptions", {}).get("Subpath") != "credential"
                    for mount in crash_main["HostConfig"]["Mounts"]
                )
                assert credential_backend.issue_calls == []
                monkeypatch.setattr(
                    docker_adapter_module,
                    "deliver_and_reobserve_runtime_volume_activation",
                    original_delivery,
                )

                started_report = coordinator.recover(action_frontier)
                assert not started_report.is_complete
                released_report = coordinator.recover(action_frontier)
                assert not released_report.is_complete
                main_id = crash_events[-1].spawn_commit.provider_execution_id
                wait_result = runtime.run_control(("container", "wait", main_id))
                assert wait_result.stdout == b"0\n"
                terminal_report = coordinator.recover(action_frontier)
                assert terminal_report.is_complete
                terminal_events = action_gate._action_store.inspect().events_for(
                    layout_reservation.intent.operation_id
                )
                assert len(terminal_events) == 8
                assert terminal_events[5].event_kind is (
                    RunActionExecutionEventKind.RESULT_RECEIVED
                )
                assert terminal_events[-1].event_kind is (
                    RunActionExecutionEventKind.RESULT_ACCEPTED
                )
                assert terminal_report.recovered_operations[
                    -1
                ].accepted_result_payload == canonical_json_bytes({"live": "captured"})
                assert terminal_events[5].result_receipt.result_blob.digest == (
                    tree_or_blob_digest(coding_agent_result_payload)
                )
                assert terminal_events[5].result_receipt.result_blob.size_bytes == len(
                    coding_agent_result_payload
                )
                assert terminal_events[
                    6
                ].result_decision.accepted_result_blob.digest == (
                    tree_or_blob_digest(canonical_json_bytes({"live": "captured"}))
                )
                assert terminal_events[
                    6
                ].result_decision.accepted_result_blob.size_bytes == len(
                    canonical_json_bytes({"live": "captured"})
                )
                assert resource_manager.observe(layout_allocation).is_absent
                action_gate._resource_finalization_authority.require_terminal_absence(
                    layout_reservation.intent.operation_id
                )
                return
            ambiguous_dispatches = []
            if terminal_path == "production_preparation_ambiguous":
                original_volume_create = (
                    PinnedDockerPreparationAuthority._create_volume_once
                )
                original_container_create = (
                    PinnedDockerPreparationAuthority._create_container_once
                )
                original_keeper_start = (
                    PinnedDockerPreparationAuthority._start_created_container_once
                )

                def lose_volume_create_response(
                    preparation_authority,
                    *,
                    arguments,
                    exclusion_lease,
                    _authority,
                ):
                    result = original_volume_create(
                        preparation_authority,
                        arguments=arguments,
                        exclusion_lease=exclusion_lease,
                        _authority=_authority,
                    )
                    ambiguous_dispatches.append(arguments[:2])
                    return replace(
                        result,
                        outcome=BoundedProcessOutcome.TIMED_OUT,
                        returncode=-1,
                    )

                def lose_container_create_response(
                    preparation_authority,
                    *,
                    arguments,
                    exclusion_lease,
                    _authority,
                ):
                    result = original_container_create(
                        preparation_authority,
                        arguments=arguments,
                        exclusion_lease=exclusion_lease,
                        _authority=_authority,
                    )
                    ambiguous_dispatches.append(arguments[:2])
                    return replace(
                        result,
                        outcome=BoundedProcessOutcome.TIMED_OUT,
                        returncode=-1,
                    )

                def lose_keeper_start_response(
                    preparation_authority,
                    *,
                    container_id,
                    exclusion_lease,
                    _authority,
                ):
                    result = original_keeper_start(
                        preparation_authority,
                        container_id=container_id,
                        exclusion_lease=exclusion_lease,
                        _authority=_authority,
                    )
                    ambiguous_dispatches.append(("container", "start"))
                    return replace(
                        result,
                        outcome=BoundedProcessOutcome.TIMED_OUT,
                        returncode=-1,
                    )

                monkeypatch.setattr(
                    PinnedDockerPreparationAuthority,
                    "_create_volume_once",
                    lose_volume_create_response,
                )
                monkeypatch.setattr(
                    PinnedDockerPreparationAuthority,
                    "_create_container_once",
                    lose_container_create_response,
                )
                monkeypatch.setattr(
                    PinnedDockerPreparationAuthority,
                    "_start_created_container_once",
                    lose_keeper_start_response,
                )
            preparation_manager = DockerRunActionPreparationManager(
                runtime=runtime,
                resource_manager=resource_manager,
                launch_settings=cross_run_settings.launch,
            )
            active_workspace = publisher_case["active"]
            workspace_source_path = active_workspace.run_root / (
                active_workspace.bootstrap_pin.installation_receipt.layout.workspace_relative_path
            )

            def reconcile_preparation(
                mode: RunActionPreparationMode,
                durable: RunActionPreparedExecution | None,
            ):
                capability = RunActionPreparationCapability(
                    preparation_allocation=layout_allocation,
                    mode=mode,
                    durable_prepared_execution=durable,
                    workspace_descriptor=workspace_descriptor,
                    workspace_source_path=workspace_source_path,
                    _authority=_RUN_ACTION_PREPARATION_AUTHORITY,
                )
                with capability._begin_invocation():
                    return preparation_manager.reconcile(capability, command)

            created = reconcile_preparation(
                RunActionPreparationMode.CREATE_ALLOCATED,
                None,
            )
            assert created.state is RunActionPreparationState.EXACT_PREPARED
            assert created.origin is RunActionPreparationOrigin.NEWLY_MATERIALIZED
            prepared_execution = created.prepared_execution
            complete_inventory = resource_manager.observe(layout_allocation)
            assert complete_inventory.volume_present
            assert complete_inventory.keeper_container_id is not None
            assert complete_inventory.main_container_id is not None
            if terminal_path == "production_preparation_ambiguous":
                assert ambiguous_dispatches == [
                    ("volume", "create"),
                    ("container", "create"),
                    ("container", "start"),
                    ("container", "create"),
                ]

            reopened = reconcile_preparation(
                RunActionPreparationMode.REOPEN_ALLOCATED,
                None,
            )
            assert reopened.state is RunActionPreparationState.EXACT_PREPARED
            assert reopened.origin is RunActionPreparationOrigin.REOPENED_ALLOCATION
            assert reopened.prepared_execution == prepared_execution

            revalidated = reconcile_preparation(
                RunActionPreparationMode.REVALIDATE_PREPARED,
                prepared_execution,
            )
            assert revalidated.state is RunActionPreparationState.EXACT_PREPARED
            assert revalidated.origin is RunActionPreparationOrigin.REVALIDATED_PREPARED
            assert revalidated.prepared_execution == prepared_execution
            recover_allocation_invalidation(prepared_execution)
            return
        cleanup.callback(
            _remove_owned_volume,
            settings,
            docker_config_root,
            layout_volume_name,
            layout_volume_labels,
        )
        layout_volume_result = runtime.run_control(
            volume_create_arguments(
                layout_claim,
                layout_authority,
                settings,
            )
        )
        assert layout_volume_result.stdout == (
            f"{layout_volume_name}\n".encode("ascii")
        )
        layout_volume_inventory = resource_manager.observe(layout_allocation)
        layout_volume_observation = observe_runtime_volume(
            resource_manager.inspect_volume(layout_volume_inventory),
            layout_claim,
            layout_authority,
            settings,
        )
        if terminal_path == "allocation_volume":
            recover_allocation_invalidation()
            return
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            layout_keeper_name,
            layout_keeper_labels,
        )
        layout_keeper_result = runtime.run_control(
            keeper_create_arguments(
                layout_claim,
                layout_authority,
                image,
                settings,
            )
        )
        layout_keeper_id = layout_keeper_result.stdout.decode("ascii").strip()
        if terminal_path == "allocation_created_keeper":
            recover_allocation_invalidation()
            return
        runtime.run_control(("container", "start", layout_keeper_id))
        layout_keeper_inventory = resource_manager.observe(layout_allocation)
        layout_keeper_evidence = observe_running_keeper(
            resource_manager.inspect_keeper(layout_keeper_inventory),
            layout_claim,
            layout_authority,
            layout_volume_observation,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        if terminal_path == "allocation_running_keeper":
            recover_allocation_invalidation()
            return
        layout_empty_volume = observe_empty_runtime_volume(
            layout_authority,
            layout_volume_observation,
            layout_keeper_evidence,
        )
        prepared_volume = materialize_runtime_volume_layout(
            layout_claim,
            layout_empty_volume,
            layout_keeper_evidence,
            workspace_descriptor=workspace_descriptor,
            settings=cross_run_settings.launch,
        )
        adopted_prepared_volume = adopt_prepared_runtime_volume_layout(
            layout_allocation,
            resource_manager,
            layout_keeper_evidence,
            settings=cross_run_settings.launch,
        )
        assert adopted_prepared_volume == prepared_volume
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            layout_main_name,
            layout_main_labels,
        )
        layout_main_result = runtime.run_control(
            main_create_arguments(
                layout_claim,
                layout_authority,
                command,
                image,
                settings,
            )
        )
        layout_main_id = layout_main_result.stdout.decode("ascii").strip()
        layout_complete_inventory = resource_manager.observe(layout_allocation)
        layout_main_evidence = observe_inert_main_container(
            resource_manager.inspect_main(layout_complete_inventory),
            layout_claim,
            layout_authority,
            layout_volume_observation,
            command,
            helper_evidence,
            init_source_evidence,
            settings,
        )
        if terminal_path == "allocation_inert_main":
            recover_allocation_invalidation()
            return
        prepared_execution = RunActionPreparedExecution.mint(
            preparation_claim=layout_claim,
            runtime_volume_authority=layout_authority,
            runtime_volume_evidence=prepared_volume.runtime_volume_evidence,
            volume_keeper_evidence=layout_keeper_evidence,
            input_delivery_slot=prepared_volume.input_delivery_slot,
            result_directory=prepared_volume.result_directory,
            control_directory=prepared_volume.control_directory,
            result_file=prepared_volume.result_file,
            temporary_directory=prepared_volume.temporary_directory,
            credential_delivery_slot=prepared_volume.credential_delivery_slot,
            workspace_proof=prepared_volume.workspace_proof,
            layout_proof=prepared_volume.layout_proof,
            inert_container_evidence=layout_main_evidence,
        )
        reopened_volume = reobserve_runtime_volume_layout(
            prepared_execution,
            layout_volume_observation,
            layout_keeper_evidence,
            settings=cross_run_settings.launch,
        )
        assert reopened_volume == prepared_volume
        assert prepared_volume.credential_delivery_slot is not None
        assert prepared_volume.workspace_proof is not None
        assert prepared_volume.runtime_volume_evidence.root_inode == (
            layout_empty_volume.root_inode
        )
        assert prepared_volume.runtime_volume_evidence.sentinel_evidence.inode != (
            layout_empty_volume.root_inode
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "touch",
                "/kapso/runtime-volume/unexpected",
            )
        )
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="root topology is incomplete",
        ):
            reobserve_runtime_volume_layout(
                prepared_execution,
                layout_volume_observation,
                layout_keeper_evidence,
                settings=cross_run_settings.launch,
            )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "rm",
                "/kapso/runtime-volume/unexpected",
            )
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "chmod",
                "600",
                "/kapso/runtime-volume/.kapso-generation",
            )
        )
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="file is unsafe or substituted",
        ):
            reobserve_runtime_volume_layout(
                prepared_execution,
                layout_volume_observation,
                layout_keeper_evidence,
                settings=cross_run_settings.launch,
            )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "chmod",
                "400",
                "/kapso/runtime-volume/.kapso-generation",
            )
        )
        with action_gate._action_store._recovery_session(
            layout_reservation,
            _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
        ) as session:
            session.commit_prepared_execution(prepared_execution)
            if terminal_path == "frontier_invalidated":
                session.invalidate_frontier()
                spawn_commit = None
            else:
                spawn_commit = session.commit_spawn(
                    security_observation_id=(
                        layout_reservation.frontier.security_observation_id
                    ),
                    boundary_identity=boundary_identity,
                )
        if terminal_path == "frontier_invalidated":
            inert_adapter = _LiveNaturalTerminalWorkloadAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                preparation_allocation=layout_allocation,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
            )
            invalidated_report = _recovery_coordinator(
                action_gate,
                inert_adapter,
            ).recover(action_frontier)
            assert invalidated_report.is_complete
            assert resource_manager.observe(layout_allocation).is_absent
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return
        assert spawn_commit is not None
        credential_issue_response = credential_broker_registry.issue_or_replay_exact(
            prepared_execution,
            spawn_commit,
        )
        credential_materialization = credential_broker_registry.materialize_exact(
            credential_issue_response,
            prepared_execution,
            spawn_commit,
        )
        activated_volume = deliver_and_reobserve_runtime_volume_activation(
            prepared_execution,
            spawn_commit,
            layout_volume_observation,
            layout_keeper_evidence,
            request_payload=b"complete request",
            credential_materialization=credential_materialization,
            workspace_descriptor=workspace_descriptor,
            settings=cross_run_settings.launch,
        )
        assert activated_volume.spawn_commit == spawn_commit
        assert (
            activated_volume.input_file_observation.content_digest
            == prepared_execution.preparation_claim.reservation.request_blob.digest
        )
        assert (
            activated_volume.input_file_observation.prepared_parent_authority_id
            == prepared_execution.input_delivery_slot.prepared_delivery_slot_id
        )
        assert (
            activated_volume.result_file_observation.prepared_parent_authority_id
            == prepared_execution.result_directory.prepared_runtime_directory_id
        )
        assert activated_volume.credential_file_observation is not None
        assert activated_volume.credential_file_observation.content_digest is None
        assert (
            activated_volume.credential_file_observation.content_authority_id
            == run_action_credential_lease_authority_id(
                prepared_execution,
                spawn_commit,
            )
        )
        assert activated_volume.activated_workspace_observation is not None
        assert (
            activated_volume.activated_workspace_observation.inode
            == prepared_execution.workspace_proof.inode
        )
        assert tuple(
            observation.inode
            for observation in (
                activated_volume.activated_runtime_directory_observations
            )
        ) == (
            prepared_execution.control_directory.inode,
            prepared_execution.temporary_directory.inode,
        )
        activation_receipt = RunActionActivationRevalidationReceipt.mint(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            reobserved_volume_evidence=(activated_volume.reobserved_volume_evidence),
            reobserved_keeper_evidence=layout_keeper_evidence,
            reobserved_container_evidence=layout_main_evidence,
            activated_workspace_observation=(
                activated_volume.activated_workspace_observation
            ),
            activated_runtime_directory_observations=(
                activated_volume.activated_runtime_directory_observations
            ),
            activated_sentinel_observation=(
                activated_volume.activated_sentinel_observation
            ),
            input_file_observation=activated_volume.input_file_observation,
            result_file_observation=activated_volume.result_file_observation,
            credential_file_observation=(activated_volume.credential_file_observation),
        )
        with action_gate._action_store._recovery_session(
            layout_reservation,
            _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
        ) as session:
            activation_bound = activation_execution_event_size_bound(
                prepared_execution=prepared_execution,
                spawn_commit=spawn_commit,
                predecessor_event_id=session.events[-1].event_id,
            )
            assert (
                session.activation_event_size_bytes(activation_receipt)
                <= activation_bound
            )
            activation_event = session.commit_activation(activation_receipt)
        if terminal_path == "expired_inert_credential":
            credential_request = run_action_credential_lease_request(
                prepared_execution,
                spawn_commit,
            )
            credential_backend._lease_expiries[
                credential_request.credential_lease_request_id
            ] = 1
            retirement_adapter = DockerRunActionExecutionAdapter(
                execution_lifecycle_identity=(
                    boundary_identity.execution_lifecycle_identity
                ),
                execution_policy=layout_policy,
                command=command,
                runtime=runtime,
                launch_settings=cross_run_settings.launch,
            )
            retirement_coordinator = _production_recovery_coordinator(
                action_gate,
                boundary_identity,
                retirement_adapter,
                _UnusedLiveResultInterpreter(
                    boundary_identity.result_interpreter_identity
                ),
            )
            intent_report = retirement_coordinator.recover(action_frontier)
            assert not intent_report.is_complete
            intent_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(intent_events) == 6
            assert intent_events[-1].event_kind is (
                RunActionExecutionEventKind.CREDENTIAL_RETIREMENT_REQUESTED
            )
            status_call_count = len(credential_backend.status_calls)
            credential_backend._lease_expiries[
                credential_request.credential_lease_request_id
            ] = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            remove_dispatches = []
            original_remove = (
                PinnedDockerCleanupAuthority._remove_stopped_container_once
            )

            def lose_remove_response(
                cleanup_authority,
                *,
                container_id,
                exclusion_lease,
                _authority,
            ):
                result = original_remove(
                    cleanup_authority,
                    container_id=container_id,
                    exclusion_lease=exclusion_lease,
                    _authority=_authority,
                )
                remove_dispatches.append(container_id)
                return replace(
                    result,
                    outcome=BoundedProcessOutcome.TIMED_OUT,
                    returncode=-1,
                )

            monkeypatch.setattr(
                PinnedDockerCleanupAuthority,
                "_remove_stopped_container_once",
                lose_remove_response,
            )
            retirement_report = retirement_coordinator.recover(action_frontier)
            assert not retirement_report.is_complete
            assert remove_dispatches == [layout_main_id]
            assert len(credential_backend.status_calls) == status_call_count
            retirement_inventory = resource_manager.observe(layout_allocation)
            assert retirement_inventory.volume_present
            assert retirement_inventory.keeper_container_id == layout_keeper_id
            assert retirement_inventory.main_container_id is None
            loss_report = retirement_coordinator.recover(action_frontier)
            assert loss_report.is_complete
            loss_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(loss_events) == 7
            termination_receipt = loss_events[-1].provider_termination_receipt
            assert termination_receipt is not None
            assert termination_receipt.reason is (
                RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
            )
            assert (
                termination_receipt.credential_retirement_intent
                == loss_events[5].credential_retirement_intent
            )
            assert loss_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert resource_manager.observe(layout_allocation).is_absent
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return
        start_manager = DockerRunActionStartManager(runtime)
        start_dispatches = []
        if terminal_path == "ambiguous_start":
            original_start = PinnedDockerStartAuthority._start_created_container_once

            def lose_start_response(
                start_authority,
                *,
                container_id,
                exclusion_lease,
                _authority,
            ):
                result = original_start(
                    start_authority,
                    container_id=container_id,
                    exclusion_lease=exclusion_lease,
                    _authority=_authority,
                )
                start_dispatches.append(container_id)
                return replace(
                    result,
                    outcome=BoundedProcessOutcome.TIMED_OUT,
                    returncode=-1,
                )

            monkeypatch.setattr(
                PinnedDockerStartAuthority,
                "_start_created_container_once",
                lose_start_response,
            )
        start_adapter = _LiveInertStartAdapter(
            boundary_identity=boundary_identity,
            execution_policy=layout_policy,
            resource_manager=resource_manager,
            start_manager=start_manager,
            preparation_allocation=layout_allocation,
            command=command,
            volume_observation=layout_volume_observation,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=settings,
            launch_settings=cross_run_settings.launch,
        )
        if terminal_path == "ambiguous_start":
            with pytest.raises(
                RunActionMainStartError,
                match="failed or ambiguous",
            ):
                _recovery_coordinator(
                    action_gate,
                    start_adapter,
                ).recover(action_frontier)
            assert start_dispatches == [layout_main_id]
            with pytest.raises(DockerRunActionInspectionError):
                _recovery_coordinator(
                    action_gate,
                    start_adapter,
                ).recover(action_frontier)
            assert start_dispatches == [layout_main_id]
        else:
            start_report = _recovery_coordinator(
                action_gate,
                start_adapter,
            ).recover(action_frontier)
            assert not start_report.is_complete
            assert (
                start_report.unresolved_operation_id
                == layout_reservation.intent.operation_id
            )
            assert start_adapter.running_observation is not None
        assert (
            len(
                action_gate._action_store.inspect().events_for(
                    layout_reservation.intent.operation_id
                )
            )
            == 5
        )
        running_layout_main = resource_manager.inspect_main(
            resource_manager.observe(layout_allocation)
        )
        running_main_observation = (
            observe_running_barrier_main_container(
                running_layout_main,
                layout_claim,
                layout_authority,
                layout_volume_observation,
                command,
                helper_evidence,
                init_source_evidence,
                settings,
            )
            if terminal_path == "ambiguous_start"
            else start_adapter.running_observation
        )
        assert running_layout_main["State"]["Running"] is True
        assert running_layout_main["State"]["Pid"] > 0
        assert running_main_observation.container_id == layout_main_id
        assert running_main_observation.init_process_id == (
            running_layout_main["State"]["Pid"]
        )
        assert running_layout_main["Path"] == (
            layout_main_evidence.issued_create_projection.command_executable
        )
        assert tuple(running_layout_main["Args"]) == (
            layout_main_evidence.issued_create_projection.command_arguments
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "test",
                "!",
                "-e",
                "/kapso/runtime-volume/control/release",
            )
        )
        for forbidden_path in (
            "/kapso/runtime-volume/temporary/target-started",
            "/kapso/runtime-volume/temporary/barrier-injection",
        ):
            runtime.run_control(
                (
                    "container",
                    "exec",
                    layout_keeper_id,
                    "/kapso-supervisor/busybox",
                    "test",
                    "!",
                    "-e",
                    forbidden_path,
                )
            )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "test",
                "!",
                "-s",
                "/kapso/runtime-volume/result/result.blob",
            )
        )
        if terminal_path == "expired_running_credential":
            credential_request = run_action_credential_lease_request(
                prepared_execution,
                spawn_commit,
            )
            credential_backend._lease_expiries[
                credential_request.credential_lease_request_id
            ] = 1
            retirement_adapter = _LiveExpiredCredentialRetirementAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                retirement_manager=(
                    DockerRunActionCredentialRetirementManager(runtime)
                ),
                preparation_allocation=layout_allocation,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
                running=True,
            )
            intent_report = _recovery_coordinator(
                action_gate,
                retirement_adapter,
            ).recover(action_frontier)
            assert not intent_report.is_complete
            assert not retirement_adapter.retirement_attempted
            intent_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(intent_events) == 6
            assert intent_events[-1].event_kind is (
                RunActionExecutionEventKind.CREDENTIAL_RETIREMENT_REQUESTED
            )
            status_call_count = len(credential_backend.status_calls)
            credential_backend._lease_expiries[
                credential_request.credential_lease_request_id
            ] = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            signal_dispatches = []
            original_signal = PinnedDockerContainmentAuthority._signal_container_once

            def lose_signal_response(
                containment_authority,
                *,
                container_id,
                signal_name,
                _authority,
            ):
                result = original_signal(
                    containment_authority,
                    container_id=container_id,
                    signal_name=signal_name,
                    _authority=_authority,
                )
                signal_dispatches.append((container_id, signal_name))
                return replace(
                    result,
                    outcome=BoundedProcessOutcome.TIMED_OUT,
                    returncode=-1,
                )

            monkeypatch.setattr(
                PinnedDockerContainmentAuthority,
                "_signal_container_once",
                lose_signal_response,
            )
            retirement_report = _recovery_coordinator(
                action_gate,
                retirement_adapter,
            ).recover(action_frontier)
            assert not retirement_report.is_complete
            assert retirement_adapter.retirement_attempted
            assert signal_dispatches == [(layout_main_id, "SIGKILL")]
            assert len(credential_backend.status_calls) == status_call_count
            inspection_deadline = time.monotonic() + settings.command_timeout_seconds
            terminal_main = resource_manager.inspect_main(
                resource_manager.observe(layout_allocation)
            )
            while (
                terminal_main["State"]["Running"] is True
                and time.monotonic() < inspection_deadline
            ):
                time.sleep(settings.run_action_barrier_poll_interval_seconds)
                terminal_main = resource_manager.inspect_main(
                    resource_manager.observe(layout_allocation)
                )
            assert terminal_main["State"]["Running"] is False
            assert terminal_main["State"]["Status"] == "exited"
            terminal_adapter = _LivePreReleaseMainTerminalAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                preparation_allocation=layout_allocation,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
            )
            terminal_report = _recovery_coordinator(
                action_gate,
                terminal_adapter,
            ).recover(action_frontier)
            assert terminal_report.is_complete
            assert terminal_adapter.termination_receipt is not None
            assert terminal_adapter.termination_receipt.reason is (
                RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
            )
            terminal_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(terminal_events) == 7
            assert terminal_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert (
                terminal_adapter.termination_receipt.credential_retirement_intent
                == terminal_events[5].credential_retirement_intent
            )
            assert resource_manager.observe(layout_allocation).is_absent
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return
        if terminal_path == "pre_release_main_loss":
            runtime.run_control(
                ("container", "rm", "--force", "--volumes", layout_main_id)
            )
            loss_inventory = resource_manager.observe(layout_allocation)
            assert loss_inventory.volume_present
            assert loss_inventory.keeper_container_id == layout_keeper_id
            assert loss_inventory.main_container_id is None
            with open_run_action_release_inspection(
                activation_event=activation_event,
                launch_settings=cross_run_settings.launch,
            ) as absent_release_inspection:
                assert absent_release_inspection.topology is (
                    RunActionControlDirectoryTopology.EMPTY
                )
            loss_adapter = _LivePreReleaseMainLossAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                preparation_allocation=layout_allocation,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
            )
            loss_report = _recovery_coordinator(
                action_gate,
                loss_adapter,
            ).recover(action_frontier)
            assert loss_report.is_complete
            assert loss_adapter.termination_receipt is not None
            assert loss_adapter.termination_receipt.disposition is (
                RunActionProviderTerminationDisposition.FAILED
            )
            assert loss_adapter.termination_receipt.reason is (
                RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
            )
            loss_observation = (
                loss_adapter.termination_receipt.pre_release_main_loss_observation
            )
            assert loss_observation.missing_provider_execution_id == layout_main_id
            assert loss_observation.observed_main_container_ids == ()
            assert loss_observation.observed_keeper_container_ids == (layout_keeper_id,)
            assert loss_observation.first_complete_inventory_digest == (
                loss_observation.second_complete_inventory_digest
            )
            loss_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(loss_events) == 6
            assert loss_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert (
                loss_events[-1].provider_termination_receipt
                == loss_adapter.termination_receipt
            )
            assert tree_or_blob_digest(busybox_bytes) == (
                settings.helper_executable_digest
            )
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return
        if terminal_path == "pre_release_main_terminal":
            runtime.run_control(
                ("container", "kill", "--signal", "KILL", layout_main_id)
            )
            inspection_deadline = time.monotonic() + settings.command_timeout_seconds
            terminal_main = resource_manager.inspect_main(
                resource_manager.observe(layout_allocation)
            )
            while (
                terminal_main["State"]["Running"] is True
                and time.monotonic() < inspection_deadline
            ):
                time.sleep(settings.run_action_barrier_poll_interval_seconds)
                terminal_main = resource_manager.inspect_main(
                    resource_manager.observe(layout_allocation)
                )
            assert terminal_main["Id"] == layout_main_id
            assert terminal_main["State"]["Running"] is False
            assert terminal_main["State"]["Status"] == "exited"
            terminal_inventory = resource_manager.observe(layout_allocation)
            assert terminal_inventory.main_container_id == layout_main_id
            with open_run_action_release_inspection(
                activation_event=activation_event,
                launch_settings=cross_run_settings.launch,
            ) as absent_release_inspection:
                assert absent_release_inspection.topology is (
                    RunActionControlDirectoryTopology.EMPTY
                )
            terminal_adapter = _LivePreReleaseMainTerminalAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                preparation_allocation=layout_allocation,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
            )
            terminal_report = _recovery_coordinator(
                action_gate,
                terminal_adapter,
            ).recover(action_frontier)
            assert terminal_report.is_complete
            assert terminal_adapter.termination_receipt is not None
            assert terminal_adapter.termination_receipt.disposition is (
                RunActionProviderTerminationDisposition.FAILED
            )
            assert terminal_adapter.termination_receipt.reason is (
                RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
            )
            observation = terminal_adapter.termination_receipt.terminal_observation
            assert observation.observed_main_container_ids == (layout_main_id,)
            assert (
                observation.terminal_container_observation.provider_execution_id
                == layout_main_id
            )
            assert observation.first_complete_inventory_digest == (
                observation.second_complete_inventory_digest
            )
            terminal_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(terminal_events) == 6
            assert terminal_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert (
                terminal_events[-1].provider_termination_receipt
                == terminal_adapter.termination_receipt
            )
            assert resource_manager.observe(layout_allocation).is_absent
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return
        adapter = _LiveBlockedWorkloadAdapter(
            boundary_identity=boundary_identity,
            execution_policy=layout_policy,
            resource_manager=resource_manager,
            preparation_allocation=layout_allocation,
            command=command,
            volume_observation=layout_volume_observation,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=settings,
            launch_settings=cross_run_settings.launch,
        )
        report = _recovery_coordinator(action_gate, adapter).recover(action_frontier)
        assert not report.is_complete
        assert report.unresolved_operation_id == layout_reservation.intent.operation_id
        if terminal_path == "ambiguous_start":
            assert start_dispatches == [layout_main_id]
        assert adapter.lease is not None
        assert adapter.release_receipt is not None
        required_security_observation = (
            action_frontier.checkpoint.safety_state.security_observation
        )
        first_release_bound = workload_release_receipt_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            required_security_observation=required_security_observation,
        )
        second_release_bound = workload_release_receipt_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            required_security_observation=required_security_observation,
        )
        assert first_release_bound == second_release_bound
        assert len(adapter.release_receipt.to_json_bytes()) <= first_release_bound
        assert first_release_bound <= (
            layout_policy.supervisor_limits.release_receipt_size_bytes
        )
        assert (
            adapter.release_receipt.release_authorization_observation.security_observation.observation_id
            == layout_reservation.frontier.security_observation_id
        )
        resolved = adapter.resolved_workload_observation
        assert resolved.activation_revalidation_receipt == activation_receipt
        assert resolved.running_container_observation.container_id == layout_main_id
        assert resolved.init_process_observation.process_id == (
            running_main_observation.init_process_id
        )
        assert resolved.wrapper_process_observation.parent_process_id == (
            resolved.init_process_observation.process_id
        )
        assert {
            observation.kind
            for observation in resolved.resolved_mount_root_observations
        } == {
            RunActionResolvedMountKind.DOCKER_INIT,
            RunActionResolvedMountKind.SUPERVISOR_HELPER,
            *(
                RunActionResolvedMountKind(mount.kind.value)
                for mount in (
                    prepared_execution.inert_container_evidence.issued_create_projection.mounts
                )
            ),
        }
        assert resolved.control_entry_count == 0
        assert resolved.temporary_entry_count == 0
        assert resolved.mount_info_snapshot.raw_byte_length == len(
            resolved.mount_info_snapshot.raw_payload
        )
        assert (
            resolved.control_directory_topology
            is RunActionControlDirectoryTopology.EMPTY
        )
        credential_request = run_action_credential_lease_request(
            prepared_execution,
            spawn_commit,
        )
        assert credential_backend.issue_calls == [credential_request]
        assert credential_backend.status_calls == [credential_request] * 5
        with open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=cross_run_settings.launch,
        ) as release_inspection:
            assert (
                release_inspection.topology
                is RunActionControlDirectoryTopology.RELEASED
            )
            assert (
                release_inspection.adoption.workload_release_receipt
                == adapter.release_receipt
            )
            release_adoption_id = (
                release_inspection.adoption.workload_release_adoption_id
            )
        assert (
            len(
                action_gate._action_store.inspect().events_for(
                    layout_reservation.intent.operation_id
                )
            )
            == 5
        )
        release_payload = runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "cat",
                "/kapso/runtime-volume/control/release",
            )
        )
        assert release_payload.stdout == adapter.release_receipt.to_json_bytes()
        if terminal_path == "timeout":
            timeout_adapter = DockerRunActionExecutionAdapter(
                execution_lifecycle_identity=(
                    boundary_identity.execution_lifecycle_identity
                ),
                execution_policy=layout_policy,
                command=command,
                runtime=runtime,
                launch_settings=cross_run_settings.launch,
            )
            publication_coordinator = _production_recovery_coordinator(
                action_gate,
                boundary_identity,
                timeout_adapter,
                _UnusedLiveResultInterpreter(
                    boundary_identity.result_interpreter_identity
                ),
            )
            assert type(publication_coordinator._release_clock) is (
                _SystemRunActionClock
            )
            publication_coordinator._release_clock.boottime_nanoseconds = (
                lambda: adapter.release_receipt.execution_deadline_boottime_nanoseconds
            )
            publication_report = publication_coordinator.recover(action_frontier)
            assert not publication_report.is_complete
            with open_run_action_timeout_inspection(
                activation_event=activation_event,
                launch_settings=cross_run_settings.launch,
            ) as timeout_inspection:
                timeout_publication = timeout_inspection.timeout_directive_publication
                assert timeout_publication is not None
                assert (
                    timeout_publication.timeout_directive.execution_deadline_boottime_nanoseconds
                    == adapter.release_receipt.execution_deadline_boottime_nanoseconds
                )
            publication_coordinator._release_clock.boottime_nanoseconds = (
                lambda: adapter.release_receipt.containment_deadline_boottime_nanoseconds
            )
            containment_report = publication_coordinator.recover(action_frontier)
            assert not containment_report.is_complete
            wait_result = runtime.run_control(("container", "wait", layout_main_id))
            assert wait_result.stdout == b"137\n"

            termination_report = publication_coordinator.recover(action_frontier)
            assert termination_report.is_complete
            timeout_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(timeout_events) == 6
            assert timeout_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            termination_receipt = timeout_events[-1].provider_termination_receipt
            assert termination_receipt is not None
            assert (
                termination_receipt.timeout_directive_publication == timeout_publication
            )
            assert tree_or_blob_digest(busybox_bytes) == (
                settings.helper_executable_digest
            )
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            return

        wait_result = runtime.run_control(("container", "wait", layout_main_id))
        expected_exit_status = {
            "result": b"0\n",
            "ambiguous_start": b"0\n",
            "empty": b"0\n",
            "nonzero": b"23\n",
            "oom": b"137\n",
        }[terminal_path]
        assert wait_result.stdout == expected_exit_status
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "test",
                "-f",
                "/kapso/runtime-volume/temporary/target-started",
            )
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "test",
                "!",
                "-e",
                "/kapso/runtime-volume/temporary/barrier-injection",
            )
        )
        terminal_adapter = _LiveNaturalTerminalWorkloadAdapter(
            boundary_identity=boundary_identity,
            execution_policy=layout_policy,
            resource_manager=resource_manager,
            preparation_allocation=layout_allocation,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=settings,
            launch_settings=cross_run_settings.launch,
        )
        terminal_report = _recovery_coordinator(
            action_gate,
            terminal_adapter,
        ).recover(action_frontier)
        assert terminal_report.is_complete
        assert terminal_adapter.terminal_observation is not None
        assert (
            terminal_adapter.reinspected_terminal_observation
            == terminal_adapter.terminal_observation
        )
        assert terminal_adapter.terminal_observation.provider_execution_id == (
            layout_main_id
        )
        assert terminal_adapter.terminal_observation.started_at == (
            adapter.release_receipt.resolved_workload_observation.running_container_observation.started_at
        )
        assert (
            terminal_adapter.terminal_observation.workload_release_adoption_id
            == release_adoption_id
        )
        terminal_events = action_gate._action_store.inspect().events_for(
            layout_reservation.intent.operation_id
        )
        expected_failure_reasons = {
            "empty": RunActionProviderTerminationReason.EMPTY_RESULT,
            "nonzero": RunActionProviderTerminationReason.NONZERO_EXIT,
            "oom": RunActionProviderTerminationReason.OOM,
        }
        if terminal_path in expected_failure_reasons:
            assert terminal_adapter.captured_result is None
            assert terminal_adapter.provider_termination_receipt is not None
            assert terminal_adapter.provider_termination_receipt.disposition is (
                RunActionProviderTerminationDisposition.FAILED
            )
            assert terminal_adapter.provider_termination_receipt.reason is (
                expected_failure_reasons[terminal_path]
            )
            assert len(terminal_events) == 6
            assert terminal_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert terminal_events[-1].provider_termination_receipt == (
                terminal_adapter.provider_termination_receipt
            )
            action_gate._resource_finalization_authority.require_terminal_absence(
                layout_reservation.intent.operation_id
            )
            assert tree_or_blob_digest(busybox_bytes) == (
                settings.helper_executable_digest
            )
            return

        assert terminal_adapter.captured_result is not None
        assert terminal_adapter.captured_result.result_payload == _LIVE_RESULT_PAYLOAD
        assert len(terminal_events) == 8
        assert terminal_events[5].event_kind is (
            RunActionExecutionEventKind.RESULT_RECEIVED
        )
        assert terminal_events[5].result_receipt.terminal_observation == (
            terminal_adapter.terminal_observation
        )
        assert terminal_events[5].result_receipt.result_capture_receipt == (
            terminal_adapter.captured_result.result_capture_receipt
        )
        assert terminal_events[-1].event_kind is (
            RunActionExecutionEventKind.RESULT_ACCEPTED
        )

        action_gate._resource_finalization_authority.require_terminal_absence(
            layout_reservation.intent.operation_id
        )
        assert tree_or_blob_digest(busybox_bytes) == settings.helper_executable_digest
