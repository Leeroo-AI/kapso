"""Explicit real-Docker validation for issued run-action create projections.

Run directly:

    pytest -q tests/live_run_action_docker_projection.py -s
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from expert_live_docker_support import (
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerRuntime,
    read_verified_root_executable,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
    keeper_create_arguments,
    main_create_arguments,
    require_run_action_image,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionResolvedMountKind,
)
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_containment_contracts import (
    RunActionTimeoutContainmentSignal,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionInterpretedResult,
)
from kapso.cross_run.launch.run_action_result_capture import (
    capture_run_action_terminal_result,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    open_run_action_blocked_workload,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_release_publisher import (
    publish_run_action_workload_release_once,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    inspect_run_action_terminal,
    reinspect_run_action_terminal,
)
from kapso.cross_run.launch.run_action_timeout_containment import (
    contain_run_action_timeout_once,
    DockerRunActionContainmentManager,
)
from kapso.cross_run.launch.run_action_timeout_publisher import (
    publish_run_action_timeout_once,
)
from kapso.cross_run.launch.run_action_timeout_termination import (
    capture_run_action_timeout_termination,
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
from kapso.cross_run.launch.run_action_docker_inspect import (
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
    deliver_and_reobserve_runtime_volume_activation,
    materialize_runtime_volume_layout,
    observe_empty_runtime_volume,
    reobserve_runtime_volume_layout,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionActivationRevalidationReceipt,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionStaticEnvironmentVariable,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings
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

    def interpret(self, *, request_payload, result_payload):
        raise AssertionError("blocked-workload proof must not interpret a result")


class _LiveAcceptedResultInterpreter:
    def __init__(self, result_interpreter_identity) -> None:
        self.result_interpreter_identity = result_interpreter_identity

    def interpret(self, *, request_payload, result_payload):
        if request_payload != b"complete request":
            raise AssertionError("live result interpreter received another request")
        if result_payload != _LIVE_RESULT_PAYLOAD:
            raise AssertionError("live result interpreter received another result")
        return RunActionInterpretedResult(
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=result_payload,
        )


class _LiveCredentialValidityAuthority:
    def __init__(self, maximum_lease_seconds) -> None:
        self._maximum_lease_seconds = maximum_lease_seconds
        self.calls = []

    def observe_exact(
        self,
        *,
        activated_credential_file_observation_id,
        credential_lease_authority_id,
    ):
        self.calls.append(
            (
                activated_credential_file_observation_id,
                credential_lease_authority_id,
            )
        )
        observed_at = time.time_ns()
        return RunActionCredentialValidityObservation.mint(
            activated_credential_file_observation_id=(
                activated_credential_file_observation_id
            ),
            credential_lease_authority_id=credential_lease_authority_id,
            observed_at_realtime_nanoseconds=observed_at,
            valid_until_realtime_nanoseconds=(
                observed_at + self._maximum_lease_seconds * 1_000_000_000
            ),
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

    def release_receipt_size_bound(self, *, reservation):
        if reservation != self._preparation_allocation.preparation_claim.reservation:
            raise AssertionError("release bound differs from durable reservation")
        return self.execution_policy.supervisor_limits.release_receipt_size_bytes

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


class _LiveTerminalWorkloadAdapter:
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

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def release_receipt_size_bound(self, *, reservation):
        if reservation != self._preparation_allocation.preparation_claim.reservation:
            raise AssertionError("terminal bound differs from durable reservation")
        return self.execution_policy.supervisor_limits.release_receipt_size_bytes

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
        self.reinspected_terminal_observation = reinspect_run_action_terminal(
            capability=capability,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        if self.reinspected_terminal_observation != self.terminal_observation:
            raise AssertionError("terminal reinspection changed its occurrence")
        self.captured_result = capture_run_action_terminal_result(
            capability=capability,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.RESULT_CAPTURED,
            result=self.captured_result,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _LiveTimeoutWorkloadAdapter:
    def __init__(
        self,
        *,
        boundary_identity,
        execution_policy,
        resource_manager,
        containment_manager,
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
        self._containment_manager = containment_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._volume_observation = volume_observation
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self._observation_state = None
        self.timeout_publication = None
        self.containment_result = None
        self.termination_receipt = None

    def prepared_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay preparation")

    def activation_event_size_bound(self, **_arguments):
        raise AssertionError("durable event 5 must not replay activation")

    def release_receipt_size_bound(self, *, reservation):
        if reservation != self._preparation_allocation.preparation_claim.reservation:
            raise AssertionError("timeout bound differs from durable reservation")
        return self.execution_policy.supervisor_limits.release_receipt_size_bytes

    def prepare(self, _capability):
        raise AssertionError("durable event 5 must not replay preparation")

    def stage_activation(self, _capability):
        raise AssertionError("durable event 5 must not replay activation")

    def inspect_unactivated(self, _query):
        raise AssertionError("durable event 5 is already activated")

    def inspect_committed(self, query):
        if query.preparation_allocation != self._preparation_allocation:
            raise AssertionError("timeout live query differs from exact allocation")
        inventory = self._resource_manager.observe(self._preparation_allocation)
        raw_main = self._resource_manager.inspect_main(inventory)
        state = raw_main.get("State")
        if not isinstance(state, dict) or type(state.get("Running")) is not bool:
            raise AssertionError("timeout live container state is malformed")
        if state["Running"]:
            running = observe_running_barrier_main_container(
                raw_main,
                self._preparation_allocation.preparation_claim,
                self._preparation_allocation.runtime_volume_authority,
                self._volume_observation,
                self._command,
                self._helper_evidence,
                self._init_source_evidence,
                self._docker_settings,
            )
            self._observation_state = RunActionCommittedSpawnState.RUNNING_CONTINUABLE
            return RunActionCommittedSpawnObservation(
                state=self._observation_state,
                observation_token=running.complete_inspection_digest,
            )
        terminal = inspect_run_action_terminal(
            query=query,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        self._observation_state = RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
        return RunActionCommittedSpawnObservation(
            state=self._observation_state,
            observation_token=terminal.complete_inspection_digest,
        )

    def continue_committed_once(self, capability):
        topology = capability.query.control_directory_topology
        if (
            self._observation_state is RunActionCommittedSpawnState.RUNNING_CONTINUABLE
            and topology is RunActionControlDirectoryTopology.RELEASED
        ):
            self.timeout_publication = publish_run_action_timeout_once(
                capability=capability,
                resource_manager=self._resource_manager,
                command=self._command,
                helper_evidence=self._helper_evidence,
                init_source_evidence=self._init_source_evidence,
                docker_settings=self._docker_settings,
                launch_settings=self._launch_settings,
            )
            if self.timeout_publication is None:
                raise AssertionError("live timeout publication was not due")
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=self.timeout_publication,
            )
        if (
            self._observation_state is RunActionCommittedSpawnState.RUNNING_CONTINUABLE
            and topology is RunActionControlDirectoryTopology.TIMED_OUT
        ):
            self.containment_result = contain_run_action_timeout_once(
                capability=capability,
                resource_manager=self._resource_manager,
                containment_manager=self._containment_manager,
                command=self._command,
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
        if (
            self._observation_state is RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
            and topology is RunActionControlDirectoryTopology.TIMED_OUT
        ):
            self.termination_receipt = capture_run_action_timeout_termination(
                capability=capability,
                resource_manager=self._resource_manager,
                command=self._command,
                helper_evidence=self._helper_evidence,
                init_source_evidence=self._init_source_evidence,
                docker_settings=self._docker_settings,
                launch_settings=self._launch_settings,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=self.termination_receipt,
                timeout_directive_publication=None,
            )
        raise AssertionError("live timeout adapter received another continuation state")


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


@pytest.mark.parametrize("terminal_path", ("result", "timeout"))
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
        image_authority = DockerImageAuthority.mint(
            image_reference=local_registry.image_reference,
            image_config_digest=local_registry.config_digest,
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
        )
        target_command = (
            'printf \'{"live":"captured"}\''
            " > /kapso/result/result.blob"
            " && printf started > /kapso/tmp/target-started"
            " && grep -Fqx 'complete request' /kapso/input/request.blob"
            " && grep -Fqx 'credential bytes'"
            " /kapso/credentials/credentials"
            " && test -d /kapso/workspace/.git"
        )
        if terminal_path == "timeout":
            target_command += (
                " && exec /bin/busybox sleep " f"{2 * settings.command_timeout_seconds}"
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
            _policy(settings),
            image_authority=image_authority,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(key="PATH", value="/bin"),
            ),
        )
        credential_validity_authority = _LiveCredentialValidityAuthority(
            layout_policy.credential_policy.maximum_lease_seconds,
        )
        (
            _action_publisher,
            action_frontier,
            _security_authority,
            action_gate,
        ) = _action_case(
            publisher_case,
            credential_validity_authority=credential_validity_authority,
        )
        base_boundary_identity = _boundary_identity(
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.READ_ONLY,
        )
        boundary_identity = RunActionBoundaryIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            execution_lifecycle_identity=_remint_contract(
                base_boundary_identity.execution_lifecycle_identity,
                execution_policy_id=layout_policy.docker_execution_policy_id,
            ),
            result_interpreter_identity=(
                base_boundary_identity.result_interpreter_identity
            ),
        )
        layout_reservation = action_gate.reserve(
            action_frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id="live_blocked_workload_0123456789abcdef",
            request_payload=b"complete request",
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=boundary_identity,
        )
        assert layout_reservation.frontier.workspace_before == workspace_binding
        with action_gate._action_store._recovery_session(
            layout_reservation,
            _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
        ) as session:
            layout_allocation = session.allocate_preparation(layout_policy)
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
            spawn_commit = session.commit_spawn(
                security_observation_id=(
                    layout_reservation.frontier.security_observation_id
                ),
                boundary_identity=boundary_identity,
            )
        activated_volume = deliver_and_reobserve_runtime_volume_activation(
            prepared_execution,
            spawn_commit,
            layout_volume_observation,
            layout_keeper_evidence,
            request_payload=b"complete request",
            credential_payload=b"credential bytes",
            credential_content_authority_id="test.credential.lease",
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
            == "test.credential.lease"
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
            activation_event = session.commit_activation(activation_receipt)
        runtime.run_control(("container", "start", layout_main_id))
        time.sleep(2 * settings.run_action_barrier_poll_interval_seconds)
        running_layout_main = resource_manager.inspect_main(
            resource_manager.observe(layout_allocation)
        )
        running_main_observation = observe_running_barrier_main_container(
            running_layout_main,
            layout_claim,
            layout_authority,
            layout_volume_observation,
            command,
            helper_evidence,
            init_source_evidence,
            settings,
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
        assert adapter.lease is not None
        assert adapter.release_receipt is not None
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
        assert (
            resolved.control_directory_topology
            is RunActionControlDirectoryTopology.EMPTY
        )
        assert len(credential_validity_authority.calls) == 2
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
            timeout_adapter = _LiveTimeoutWorkloadAdapter(
                boundary_identity=boundary_identity,
                execution_policy=layout_policy,
                resource_manager=resource_manager,
                containment_manager=DockerRunActionContainmentManager(runtime),
                preparation_allocation=layout_allocation,
                command=command,
                volume_observation=layout_volume_observation,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=settings,
                launch_settings=cross_run_settings.launch,
            )
            publication_coordinator = _recovery_coordinator(
                action_gate,
                timeout_adapter,
            )
            assert type(publication_coordinator._release_clock) is (
                _SystemRunActionClock
            )
            publication_coordinator._release_clock.boottime_nanoseconds = (
                lambda: adapter.release_receipt.execution_deadline_boottime_nanoseconds
            )
            publication_report = publication_coordinator.recover(action_frontier)
            assert not publication_report.is_complete
            assert timeout_adapter.timeout_publication is not None
            assert (
                timeout_adapter.timeout_publication.timeout_directive.execution_deadline_boottime_nanoseconds
                == adapter.release_receipt.execution_deadline_boottime_nanoseconds
            )
            containment_coordinator = _recovery_coordinator(
                action_gate,
                timeout_adapter,
            )
            containment_coordinator._release_clock.boottime_nanoseconds = (
                lambda: adapter.release_receipt.containment_deadline_boottime_nanoseconds
            )
            containment_report = containment_coordinator.recover(action_frontier)
            assert not containment_report.is_complete
            assert timeout_adapter.containment_result is not None
            assert (
                timeout_adapter.containment_result.signal
                is RunActionTimeoutContainmentSignal.KILL
            )
            assert timeout_adapter.containment_result.signal_dispatch_confirmed
            wait_result = runtime.run_control(("container", "wait", layout_main_id))
            assert wait_result.stdout == b"137\n"

            termination_report = _recovery_coordinator(
                action_gate,
                timeout_adapter,
            ).recover(action_frontier)
            assert termination_report.is_complete
            assert timeout_adapter.termination_receipt is not None
            assert (
                timeout_adapter.termination_receipt.timeout_directive_publication
                == timeout_adapter.timeout_publication
            )
            timeout_events = action_gate._action_store.inspect().events_for(
                layout_reservation.intent.operation_id
            )
            assert len(timeout_events) == 6
            assert timeout_events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )
            assert (
                timeout_events[-1].provider_termination_receipt
                == timeout_adapter.termination_receipt
            )
            assert tree_or_blob_digest(busybox_bytes) == (
                settings.helper_executable_digest
            )
            return

        wait_result = runtime.run_control(("container", "wait", layout_main_id))
        assert wait_result.stdout == b"0\n"
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
        terminal_adapter = _LiveTerminalWorkloadAdapter(
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
        assert terminal_adapter.captured_result is not None
        assert terminal_adapter.captured_result.result_payload == _LIVE_RESULT_PAYLOAD
        with open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=cross_run_settings.launch,
        ) as terminal_release_inspection:
            assert (
                terminal_adapter.terminal_observation.workload_release_adoption_id
                == terminal_release_inspection.adoption.workload_release_adoption_id
            )
        assert (
            len(
                action_gate._action_store.inspect().events_for(
                    layout_reservation.intent.operation_id
                )
            )
            == 8
        )
        terminal_events = action_gate._action_store.inspect().events_for(
            layout_reservation.intent.operation_id
        )
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

        runtime.run_control(("container", "rm", "--force", "--volumes", layout_main_id))
        runtime.run_control(
            ("container", "rm", "--force", "--volumes", layout_keeper_id)
        )
        runtime.run_control(("volume", "rm", layout_volume_name))
        assert tree_or_blob_digest(busybox_bytes) == settings.helper_executable_digest
