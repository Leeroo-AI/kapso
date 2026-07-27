"""Descriptor-retained proof of one blocked post-start run-action workload."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import PurePosixPath
from threading import get_ident, Lock
from weakref import WeakValueDictionary

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierInitProcessObservation,
    RunActionBarrierRunningContainerObservation,
    RunActionBarrierWrapperProcessObservation,
    RunActionMountInfoObservation,
    RunActionMountInfoSnapshot,
    RunActionResolvedFileObservation,
    RunActionResolvedMountKind,
    RunActionResolvedMountRootObservation,
    RunActionResolvedWorkloadObservation,
    RunActionResolvedWorkspaceObservation,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_running_barrier_main_container,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_recovery import (
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionCommittedContinuationCapability,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
    open_run_action_control_directory,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_DOCKER_INIT_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionActivatedFileObservation,
    RunActionActivatedRuntimeDirectoryObservation,
    RunActionActivationRevalidationReceipt,
    RunActionDockerInitSourceEvidence,
    RunActionPreparationAllocation,
    RunActionPreparedFileKind,
    RunActionPreparedMountAccess,
    RunActionPreparedRuntimeDirectoryKind,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionProcessDescriptorMetadata,
    RunActionProcessStatObservation,
    open_run_action_process_executable_descriptor,
    open_run_action_process_namespace_descriptor,
    open_run_action_process_root_descriptor,
    read_run_action_descriptor_mount_id,
    read_run_action_host_boot_id,
    read_run_action_process_cgroup_path_from_descriptor,
    read_run_action_process_command_line_from_descriptor,
    read_run_action_process_direct_child_from_descriptor,
    read_run_action_process_mount_info_from_descriptor,
    read_run_action_process_stat_from_descriptor,
    verify_run_action_executable_descriptor,
)
from kapso.cross_run.launch.workspace_frontier import inspect_run_workspace_frontier
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_BLOCKED_PROCESS_STATES = {"R", "S"}
_LEASE_AUTHORITY = object()
_RELEASE_PUBLICATION_AUTHORITY = object()
_ISSUED_BLOCKED_WORKLOAD_LEASES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_BLOCKED_WORKLOAD_LEASE_LOCK = Lock()


class RunActionResolvedWorkloadError(RuntimeError):
    """The blocked workload changed, was spliced, or lacks closed evidence."""


@dataclass(frozen=True)
class _SourceAuthority:
    authority_id: str
    mount_id: int
    device: int
    inode: int
    owner_user_id: int
    owner_group_id: int
    mode: int


@dataclass(frozen=True)
class _RetainedProcess:
    process_descriptor: int
    process_metadata: tuple[int, ...]
    stat_observation: RunActionProcessStatObservation
    cgroup_path: str
    command_line: tuple[str, ...]
    root_descriptor: int
    root_metadata: RunActionProcessDescriptorMetadata
    executable_descriptor: int
    executable_metadata: RunActionProcessDescriptorMetadata
    executable_digest: str
    mount_namespace_descriptor: int
    mount_namespace_metadata: RunActionProcessDescriptorMetadata
    process_id_namespace_descriptor: int
    process_id_namespace_metadata: RunActionProcessDescriptorMetadata
    process_snapshot_size_limit_bytes: int


@dataclass(frozen=True)
class _RetainedResolvedRoot:
    destination: str
    descriptor: int
    metadata: tuple[int, ...]
    mount_id: int
    process_snapshot_size_limit_bytes: int


class RunActionBlockedWorkloadLease:
    """Process-and-thread-bound authority for one still-blocked workload."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        activation_event: RunActionExecutionEvent,
        resolved_workload_observation: RunActionResolvedWorkloadObservation,
        control_lease: RunActionControlDirectoryLease,
        resource_manager: DockerRunActionResourceManager,
        preparation_allocation: RunActionPreparationAllocation,
        command: DockerRunActionCommand,
        volume_observation: DockerRunActionVolumeObservation,
        helper_evidence: RunActionSupervisorHelperEvidence,
        init_source_evidence: RunActionDockerInitSourceEvidence,
        docker_settings: DockerRuntimeSettings,
        launch_settings: LaunchSettings,
        proc_root_descriptor: int,
        init_process: _RetainedProcess,
        wrapper_process: _RetainedProcess,
        mount_info_snapshot: RunActionMountInfoSnapshot,
        retained_roots: tuple[_RetainedResolvedRoot, ...],
        host_boot_id: str,
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(activation_event) is not RunActionExecutionEvent
            or activation_event.event_number != 5
            or activation_event.event_kind
            is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
            or type(resolved_workload_observation)
            is not RunActionResolvedWorkloadObservation
            or type(control_lease) is not RunActionControlDirectoryLease
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(preparation_allocation) is not RunActionPreparationAllocation
            or type(command) is not DockerRunActionCommand
            or type(volume_observation) is not DockerRunActionVolumeObservation
            or type(helper_evidence) is not RunActionSupervisorHelperEvidence
            or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
            or type(docker_settings) is not DockerRuntimeSettings
            or type(launch_settings) is not LaunchSettings
            or resource_manager.runtime_settings != docker_settings
            or type(proc_root_descriptor) is not int
            or proc_root_descriptor < 0
            or not stat.S_ISDIR(os.fstat(proc_root_descriptor).st_mode)
            or type(init_process) is not _RetainedProcess
            or type(wrapper_process) is not _RetainedProcess
            or type(mount_info_snapshot) is not RunActionMountInfoSnapshot
            or type(retained_roots) is not tuple
            or not retained_roots
            or any(type(root) is not _RetainedResolvedRoot for root in retained_roots)
            or host_boot_id != resolved_workload_observation.host_boot_id
            or _authority is not _LEASE_AUTHORITY
        ):
            raise RunActionResolvedWorkloadError(
                "blocked workload lease lacks exact retained authority"
            )
        self._descriptors = descriptors
        self._activation_event = activation_event
        self._resolved_workload_observation = resolved_workload_observation
        self._control_lease = control_lease
        self._resource_manager = resource_manager
        self._preparation_allocation = preparation_allocation
        self._command = command
        self._volume_observation = volume_observation
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self._process_snapshot_size_limit_bytes = (
            launch_settings.run_action_process_snapshot_size_bytes
        )
        self._proc_root_descriptor = proc_root_descriptor
        self._init_process = init_process
        self._wrapper_process = wrapper_process
        self._mount_info_snapshot = mount_info_snapshot
        self._retained_roots = retained_roots
        self._host_boot_id = host_boot_id
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False

    @property
    def activation_event(self) -> RunActionExecutionEvent:
        self.require_current()
        return self._activation_event

    @property
    def resolved_workload_observation(self) -> RunActionResolvedWorkloadObservation:
        self.require_current()
        return self._resolved_workload_observation

    def require_current(self) -> None:
        """Re-run the reverse sandwich against every retained authority."""

        self._require_issued()
        self._require_current_state()

    def _require_current_state(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionResolvedWorkloadError(
                "blocked workload lease is closed, forked, or on another thread"
            )
        if (
            read_run_action_host_boot_id(self._proc_root_descriptor)
            != self._host_boot_id
        ):
            raise RunActionResolvedWorkloadError(
                "host boot identity changed while workload was blocked"
            )
        self._control_lease.require_current()
        if self._control_lease.topology is not RunActionControlDirectoryTopology.EMPTY:
            raise RunActionResolvedWorkloadError(
                "blocked workload control topology changed"
            )
        expected_container = (
            self._resolved_workload_observation.running_container_observation
        )
        current_container = _observe_running_container(
            self._resource_manager,
            self._preparation_allocation,
            self._command,
            self._volume_observation,
            self._helper_evidence,
            self._init_source_evidence,
            self._docker_settings,
        )
        if not _same_running_container_occurrence(
            current_container,
            expected_container,
        ):
            raise RunActionResolvedWorkloadError(
                "Docker running-container observation changed after resolved proof: "
                + ", ".join(
                    _changed_contract_fields(expected_container, current_container)
                )
            )
        _require_retained_process_current(
            self._init_process,
            self._proc_root_descriptor,
            expected_container.container_id,
            self._process_snapshot_size_limit_bytes,
        )
        _require_retained_process_current(
            self._wrapper_process,
            self._proc_root_descriptor,
            expected_container.container_id,
            self._process_snapshot_size_limit_bytes,
        )
        if (
            read_run_action_process_direct_child_from_descriptor(
                self._init_process.process_descriptor,
                self._init_process.stat_observation.process_id,
                self._process_snapshot_size_limit_bytes,
            )
            != self._wrapper_process.stat_observation.process_id
        ):
            raise RunActionResolvedWorkloadError(
                "Docker init no longer has the exact blocked wrapper child"
            )
        current_mount_info = _read_mount_info_snapshot(
            self._init_process.process_descriptor,
            self._process_snapshot_size_limit_bytes,
        )
        if current_mount_info != self._mount_info_snapshot:
            raise RunActionResolvedWorkloadError(
                "container mount namespace changed after resolved proof"
            )
        for retained_root in self._retained_roots:
            _require_retained_root_current(
                self._init_process.root_descriptor,
                retained_root,
            )
        _require_logical_mounts_current(
            self._resolved_workload_observation,
            self._retained_roots,
            self._launch_settings,
        )
        for retained_root in reversed(self._retained_roots):
            _require_retained_root_current(
                self._init_process.root_descriptor,
                retained_root,
            )
        if (
            _read_mount_info_snapshot(
                self._init_process.process_descriptor,
                self._process_snapshot_size_limit_bytes,
            )
            != self._mount_info_snapshot
        ):
            raise RunActionResolvedWorkloadError(
                "container mount namespace changed after resolved proof"
            )
        if (
            read_run_action_process_direct_child_from_descriptor(
                self._init_process.process_descriptor,
                self._init_process.stat_observation.process_id,
                self._process_snapshot_size_limit_bytes,
            )
            != self._wrapper_process.stat_observation.process_id
        ):
            raise RunActionResolvedWorkloadError(
                "Docker init child changed during resolved-workload revalidation"
            )
        _require_retained_process_current(
            self._wrapper_process,
            self._proc_root_descriptor,
            expected_container.container_id,
            self._process_snapshot_size_limit_bytes,
        )
        _require_retained_process_current(
            self._init_process,
            self._proc_root_descriptor,
            expected_container.container_id,
            self._process_snapshot_size_limit_bytes,
        )
        current_container = _observe_running_container(
            self._resource_manager,
            self._preparation_allocation,
            self._command,
            self._volume_observation,
            self._helper_evidence,
            self._init_source_evidence,
            self._docker_settings,
        )
        if not _same_running_container_occurrence(
            current_container,
            expected_container,
        ):
            raise RunActionResolvedWorkloadError(
                "Docker container changed during resolved-workload revalidation: "
                + ", ".join(
                    _changed_contract_fields(expected_container, current_container)
                )
            )
        self._control_lease.require_current()
        if self._control_lease.topology is not RunActionControlDirectoryTopology.EMPTY:
            raise RunActionResolvedWorkloadError(
                "blocked workload control topology changed"
            )
        if (
            read_run_action_host_boot_id(self._proc_root_descriptor)
            != self._host_boot_id
        ):
            raise RunActionResolvedWorkloadError(
                "host boot identity changed during resolved-workload revalidation"
            )

    def _require_issued(self) -> None:
        with _BLOCKED_WORKLOAD_LEASE_LOCK:
            issued = _ISSUED_BLOCKED_WORKLOAD_LEASES.get(id(self))
        if issued is not self:
            raise RunActionResolvedWorkloadError(
                "blocked workload lease is unissued, closed, or foreign"
            )

    def _duplicate_release_control_descriptor(
        self,
        *,
        _authority: object,
    ) -> int:
        if _authority is not _RELEASE_PUBLICATION_AUTHORITY:
            raise RunActionResolvedWorkloadError(
                "blocked workload release control lacks publication authority"
            )
        self.require_current()
        descriptor = os.dup(self._control_lease._control_descriptor)
        os.set_inheritable(descriptor, False)
        return descriptor

    def __enter__(self) -> RunActionBlockedWorkloadLease:
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        self._require_issued()
        if (
            self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionResolvedWorkloadError(
                "blocked workload lease is already closed, forked, or foreign"
            )
        with _BLOCKED_WORKLOAD_LEASE_LOCK:
            issued = _ISSUED_BLOCKED_WORKLOAD_LEASES.pop(id(self), None)
        if issued is not self:
            raise RunActionResolvedWorkloadError(
                "blocked workload lease issuance changed before close"
            )
        self._closed = True
        self._descriptors.close()


def open_run_action_blocked_workload(
    capability: RunActionCommittedContinuationCapability,
    *,
    committed_running_observation: RunActionBarrierRunningContainerObservation,
    resource_manager: DockerRunActionResourceManager,
    preparation_allocation: RunActionPreparationAllocation,
    command: DockerRunActionCommand,
    volume_observation: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionBlockedWorkloadLease:
    """Resolve and retain one exact event-5 workload before release publication."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(committed_running_observation)
        is not RunActionBarrierRunningContainerObservation
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(preparation_allocation) is not RunActionPreparationAllocation
        or type(command) is not DockerRunActionCommand
        or type(volume_observation) is not DockerRunActionVolumeObservation
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
    ):
        raise RunActionResolvedWorkloadError(
            "blocked workload resolution requires exact positive inputs"
        )
    activation_event = capability.activation_event
    committed_spawn_observation = capability.observation
    activation = activation_event.activation_revalidation_receipt
    prepared = activation.prepared_execution
    spawn = activation.spawn_commit
    policy = prepared.preparation_claim.execution_policy
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        activation_event.event_number != 5
        or activation_event.event_kind
        is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
        or preparation_allocation.preparation_claim != prepared.preparation_claim
        or preparation_allocation.runtime_volume_authority
        != prepared.runtime_volume_authority
        or volume_observation.volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
        or policy.supervisor_limits.process_snapshot_size_bytes
        != launch_settings.run_action_process_snapshot_size_bytes
    ):
        raise RunActionResolvedWorkloadError(
            "blocked workload inputs differ from exact durable event 5"
        )
    with ExitStack() as descriptors:
        process_snapshot_size_limit_bytes = (
            launch_settings.run_action_process_snapshot_size_bytes
        )
        proc_root_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, proc_root_descriptor)
        control_lease = open_run_action_control_directory(prepared)
        descriptors.callback(control_lease.close)
        if control_lease.topology is not RunActionControlDirectoryTopology.EMPTY:
            raise RunActionResolvedWorkloadError(
                "blocked workload control directory is not empty"
            )
        current_running = _observe_running_container(
            resource_manager,
            preparation_allocation,
            command,
            volume_observation,
            helper_evidence,
            init_source_evidence,
            docker_settings,
        )
        _require_committed_running_authority(
            committed_spawn_observation,
            committed_running_observation,
            current_running,
        )
        running = committed_running_observation
        if running.container_id != spawn.provider_execution_id:
            raise RunActionResolvedWorkloadError(
                "running main differs from durable spawn identity"
            )
        host_boot_id = read_run_action_host_boot_id(proc_root_descriptor)
        init_process = _open_retained_process(
            descriptors,
            proc_root_descriptor,
            running.init_process_id,
            running.container_id,
            process_snapshot_size_limit_bytes,
            expected_executable_digest=init_source_evidence.executable_digest,
        )
        child_process_id = read_run_action_process_direct_child_from_descriptor(
            init_process.process_descriptor,
            init_process.stat_observation.process_id,
            process_snapshot_size_limit_bytes,
        )
        wrapper_process = _open_retained_process(
            descriptors,
            proc_root_descriptor,
            child_process_id,
            running.container_id,
            process_snapshot_size_limit_bytes,
            expected_executable_digest=helper_evidence.executable_digest,
        )
        if (
            wrapper_process.stat_observation.parent_process_id
            != init_process.stat_observation.process_id
        ):
            raise RunActionResolvedWorkloadError(
                "blocked wrapper is not Docker init's exact direct child"
            )
        mount_info_snapshot = _read_mount_info_snapshot(
            init_process.process_descriptor,
            process_snapshot_size_limit_bytes,
        )
        container_root_record = _one_mount_record(mount_info_snapshot, "/")
        init_observation = _init_process_observation(
            running,
            init_process,
            container_root_record,
        )
        wrapper_observation = _wrapper_process_observation(
            running,
            init_observation,
            wrapper_process,
            container_root_record,
        )
        retained_roots, root_observations = _open_resolved_roots(
            descriptors,
            activation,
            init_process,
            wrapper_process,
            mount_info_snapshot,
        )
        roots_by_kind = {root.kind: root for root in root_observations}
        retained_roots_by_kind = _retained_roots_by_kind(
            root_observations,
            retained_roots,
        )
        file_observations = _resolved_file_observations(
            activation,
            roots_by_kind,
            retained_roots_by_kind,
        )
        workspace_observation = _resolved_workspace_observation(
            activation,
            roots_by_kind,
            retained_roots_by_kind,
            launch_settings,
        )
        control_root = retained_roots_by_kind[RunActionResolvedMountKind.CONTROL]
        temporary_root = retained_roots_by_kind[RunActionResolvedMountKind.TEMPORARY]
        control_entries = _exact_directory_entries(control_root.descriptor)
        temporary_entries = _exact_directory_entries(temporary_root.descriptor)
        if control_entries or temporary_entries:
            raise RunActionResolvedWorkloadError(
                "control or temporary mount is nonempty before release"
            )
        resolved = RunActionResolvedWorkloadObservation.mint(
            activation_revalidation_receipt=activation,
            host_boot_id=host_boot_id,
            running_container_observation=running,
            init_process_observation=init_observation,
            wrapper_process_observation=wrapper_observation,
            mount_info_snapshot=mount_info_snapshot,
            resolved_mount_root_observations=root_observations,
            resolved_file_observations=file_observations,
            resolved_workspace_observation=workspace_observation,
            control_entry_count=0,
            temporary_entry_count=0,
            control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
        )
        lease = RunActionBlockedWorkloadLease(
            descriptors=descriptors,
            activation_event=activation_event,
            resolved_workload_observation=resolved,
            control_lease=control_lease,
            resource_manager=resource_manager,
            preparation_allocation=preparation_allocation,
            command=command,
            volume_observation=volume_observation,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
            proc_root_descriptor=proc_root_descriptor,
            init_process=init_process,
            wrapper_process=wrapper_process,
            mount_info_snapshot=mount_info_snapshot,
            retained_roots=retained_roots,
            host_boot_id=host_boot_id,
            _authority=_LEASE_AUTHORITY,
        )
        lease._require_current_state()
        with _BLOCKED_WORKLOAD_LEASE_LOCK:
            if _ISSUED_BLOCKED_WORKLOAD_LEASES.get(id(lease)) is not None:
                raise RunActionResolvedWorkloadError(
                    "blocked workload lease identity is already issued"
                )
            _ISSUED_BLOCKED_WORKLOAD_LEASES[id(lease)] = lease
        lease._descriptors = descriptors.pop_all()
    return lease


def _require_committed_running_authority(
    committed_spawn_observation: RunActionCommittedSpawnObservation,
    committed_running_observation: RunActionBarrierRunningContainerObservation,
    current_running_observation: RunActionBarrierRunningContainerObservation,
) -> None:
    if (
        type(committed_spawn_observation) is not RunActionCommittedSpawnObservation
        or committed_spawn_observation.state
        is not RunActionCommittedSpawnState.RUNNING_CONTINUABLE
        or committed_spawn_observation.observation_token
        != committed_running_observation.complete_inspection_digest
        or not _same_running_container_occurrence(
            committed_running_observation,
            current_running_observation,
        )
    ):
        raise RunActionResolvedWorkloadError(
            "blocked workload differs from its committed running observation"
        )


def _observe_running_container(
    resource_manager: DockerRunActionResourceManager,
    preparation_allocation: RunActionPreparationAllocation,
    command: DockerRunActionCommand,
    expected_volume: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> RunActionBarrierRunningContainerObservation:
    inventory = resource_manager.observe(preparation_allocation)
    prepared_claim = preparation_allocation.preparation_claim
    prepared_authority = preparation_allocation.runtime_volume_authority
    if (
        inventory.volume_inspection_digest is None
        or inventory.keeper_container_id is None
        or inventory.main_container_id is None
    ):
        raise RunActionResolvedWorkloadError(
            "blocked workload lacks its exact three Docker resources"
        )
    observed_volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared_claim,
        prepared_authority,
        docker_settings,
    )
    if observed_volume != expected_volume:
        raise RunActionResolvedWorkloadError(
            "runtime volume differs from the prepared Docker observation"
        )
    return observe_running_barrier_main_container(
        resource_manager.inspect_main(inventory),
        prepared_claim,
        prepared_authority,
        observed_volume,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    )


def _open_retained_process(
    descriptors: ExitStack,
    proc_root_descriptor: int,
    process_id: int,
    container_id: str,
    process_snapshot_size_limit_bytes: int,
    *,
    expected_executable_digest: str,
) -> _RetainedProcess:
    process_descriptor = os.open(
        str(process_id),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=proc_root_descriptor,
    )
    descriptors.callback(os.close, process_descriptor)
    process_metadata = _stable_metadata(os.fstat(process_descriptor))
    stat_observation = read_run_action_process_stat_from_descriptor(
        process_descriptor,
        process_id,
        process_snapshot_size_limit_bytes,
    )
    if stat_observation.state not in _BLOCKED_PROCESS_STATES:
        raise RunActionResolvedWorkloadError(
            "barrier process is not running or interruptibly blocked"
        )
    cgroup_path = read_run_action_process_cgroup_path_from_descriptor(
        process_descriptor,
        container_id,
        process_snapshot_size_limit_bytes,
    )
    command_line = _decode_command_line(
        read_run_action_process_command_line_from_descriptor(
            process_descriptor,
            process_snapshot_size_limit_bytes,
        )
    )
    root_descriptor, root_metadata = open_run_action_process_root_descriptor(
        descriptors,
        process_descriptor,
        process_snapshot_size_limit_bytes,
    )
    executable_descriptor, executable_metadata = (
        open_run_action_process_executable_descriptor(
            descriptors,
            process_descriptor,
            process_snapshot_size_limit_bytes,
        )
    )
    executable_observation = verify_run_action_executable_descriptor(
        executable_descriptor,
        expected_executable_digest,
        process_snapshot_size_limit_bytes,
    )
    if (
        executable_observation.mount_id != executable_metadata.mount_id
        or executable_observation.device != executable_metadata.device
        or executable_observation.inode != executable_metadata.inode
        or executable_observation.mode != executable_metadata.mode
        or executable_observation.owner_user_id != executable_metadata.owner_user_id
        or executable_observation.owner_group_id != executable_metadata.owner_group_id
        or executable_observation.link_count != executable_metadata.link_count
        or executable_observation.size != executable_metadata.size
    ):
        raise RunActionResolvedWorkloadError(
            "barrier process executable descriptor changed during verification"
        )
    mount_namespace_descriptor, mount_namespace_metadata = (
        open_run_action_process_namespace_descriptor(
            descriptors,
            process_descriptor,
            "mnt",
            process_snapshot_size_limit_bytes,
        )
    )
    process_id_namespace_descriptor, process_id_namespace_metadata = (
        open_run_action_process_namespace_descriptor(
            descriptors,
            process_descriptor,
            "pid",
            process_snapshot_size_limit_bytes,
        )
    )
    retained = _RetainedProcess(
        process_descriptor=process_descriptor,
        process_metadata=process_metadata,
        stat_observation=stat_observation,
        cgroup_path=cgroup_path,
        command_line=command_line,
        root_descriptor=root_descriptor,
        root_metadata=root_metadata,
        executable_descriptor=executable_descriptor,
        executable_metadata=executable_metadata,
        executable_digest=executable_observation.executable_digest,
        mount_namespace_descriptor=mount_namespace_descriptor,
        mount_namespace_metadata=mount_namespace_metadata,
        process_id_namespace_descriptor=process_id_namespace_descriptor,
        process_id_namespace_metadata=process_id_namespace_metadata,
        process_snapshot_size_limit_bytes=process_snapshot_size_limit_bytes,
    )
    _require_retained_process_current(
        retained,
        proc_root_descriptor,
        container_id,
        process_snapshot_size_limit_bytes,
    )
    return retained


def _require_retained_process_current(
    retained: _RetainedProcess,
    proc_root_descriptor: int,
    container_id: str,
    process_snapshot_size_limit_bytes: int,
) -> None:
    process_id = retained.stat_observation.process_id
    with ExitStack() as current_descriptors:
        current_process_descriptor = os.open(
            str(process_id),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=proc_root_descriptor,
        )
        current_descriptors.callback(os.close, current_process_descriptor)
        current_process_metadata = _stable_metadata(
            os.fstat(current_process_descriptor)
        )
        current_stat = read_run_action_process_stat_from_descriptor(
            current_process_descriptor,
            process_id,
            process_snapshot_size_limit_bytes,
        )
        current_cgroup = read_run_action_process_cgroup_path_from_descriptor(
            current_process_descriptor,
            container_id,
            process_snapshot_size_limit_bytes,
        )
        current_command_line = _decode_command_line(
            read_run_action_process_command_line_from_descriptor(
                current_process_descriptor,
                process_snapshot_size_limit_bytes,
            )
        )
        _, current_root_metadata = open_run_action_process_root_descriptor(
            current_descriptors,
            current_process_descriptor,
            process_snapshot_size_limit_bytes,
        )
        current_executable_descriptor, current_executable_metadata = (
            open_run_action_process_executable_descriptor(
                current_descriptors,
                current_process_descriptor,
                process_snapshot_size_limit_bytes,
            )
        )
        current_executable_observation = verify_run_action_executable_descriptor(
            current_executable_descriptor,
            retained.executable_digest,
            process_snapshot_size_limit_bytes,
        )
        _, current_mount_namespace_metadata = (
            open_run_action_process_namespace_descriptor(
                current_descriptors,
                current_process_descriptor,
                "mnt",
                process_snapshot_size_limit_bytes,
            )
        )
        _, current_process_id_namespace_metadata = (
            open_run_action_process_namespace_descriptor(
                current_descriptors,
                current_process_descriptor,
                "pid",
                process_snapshot_size_limit_bytes,
            )
        )
    if (
        _stable_metadata(os.fstat(retained.process_descriptor))
        != retained.process_metadata
        or _descriptor_metadata(
            retained.root_descriptor,
            "root",
            process_snapshot_size_limit_bytes,
        )
        != retained.root_metadata
        or _descriptor_metadata(
            retained.executable_descriptor,
            "exe",
            process_snapshot_size_limit_bytes,
        )
        != retained.executable_metadata
        or _descriptor_metadata(
            retained.mount_namespace_descriptor,
            "ns/mnt",
            process_snapshot_size_limit_bytes,
        )
        != retained.mount_namespace_metadata
        or _descriptor_metadata(
            retained.process_id_namespace_descriptor,
            "ns/pid",
            process_snapshot_size_limit_bytes,
        )
        != retained.process_id_namespace_metadata
        or current_process_metadata != retained.process_metadata
        or not _same_process_generation(current_stat, retained.stat_observation)
        or current_stat.state not in _BLOCKED_PROCESS_STATES
        or current_cgroup != retained.cgroup_path
        or current_command_line != retained.command_line
        or current_root_metadata != retained.root_metadata
        or current_executable_metadata != retained.executable_metadata
        or current_executable_observation.mount_id
        != retained.executable_metadata.mount_id
        or current_executable_observation.device != retained.executable_metadata.device
        or current_executable_observation.inode != retained.executable_metadata.inode
        or current_executable_observation.executable_digest
        != retained.executable_digest
        or current_mount_namespace_metadata != retained.mount_namespace_metadata
        or current_process_id_namespace_metadata
        != retained.process_id_namespace_metadata
    ):
        raise RunActionResolvedWorkloadError(
            "retained barrier process generation or namespace changed"
        )


def _same_process_generation(
    observed: RunActionProcessStatObservation,
    expected: RunActionProcessStatObservation,
) -> bool:
    return (
        type(observed) is RunActionProcessStatObservation
        and type(expected) is RunActionProcessStatObservation
        and observed.process_id == expected.process_id
        and observed.parent_process_id == expected.parent_process_id
        and observed.start_time_ticks == expected.start_time_ticks
    )


def _changed_contract_fields(
    expected: RunActionBarrierRunningContainerObservation,
    observed: RunActionBarrierRunningContainerObservation,
) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(expected)
        if getattr(expected, field.name) != getattr(observed, field.name)
    )


def _same_running_container_occurrence(
    observed: RunActionBarrierRunningContainerObservation,
    expected: RunActionBarrierRunningContainerObservation,
) -> bool:
    nonauthoritative_fields = {
        "barrier_running_container_observation_id",
        "complete_inspection_digest",
    }
    return (
        type(observed) is RunActionBarrierRunningContainerObservation
        and type(expected) is RunActionBarrierRunningContainerObservation
        and all(
            getattr(observed, field.name) == getattr(expected, field.name)
            for field in fields(expected)
            if field.name not in nonauthoritative_fields
        )
    )


def _descriptor_metadata(
    descriptor: int,
    descriptor_name: str,
    process_snapshot_size_limit_bytes: int,
) -> RunActionProcessDescriptorMetadata:
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(
        descriptor,
        process_snapshot_size_limit_bytes,
    )
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(
        descriptor,
        process_snapshot_size_limit_bytes,
    )
    if (
        _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
        or mount_id_before != mount_id_after
    ):
        raise RunActionResolvedWorkloadError(
            "retained process descriptor changed during reinspection"
        )
    file_type = (
        "directory"
        if stat.S_ISDIR(metadata_before.st_mode)
        else "regular" if stat.S_ISREG(metadata_before.st_mode) else "unsupported"
    )
    return RunActionProcessDescriptorMetadata(
        descriptor_name=descriptor_name,
        file_type=file_type,
        mount_id=mount_id_before,
        device=metadata_before.st_dev,
        inode=metadata_before.st_ino,
        mode=stat.S_IMODE(metadata_before.st_mode),
        owner_user_id=metadata_before.st_uid,
        owner_group_id=metadata_before.st_gid,
        link_count=metadata_before.st_nlink,
        size=metadata_before.st_size,
    )


def _read_mount_info_snapshot(
    process_descriptor: int,
    size_limit_bytes: int,
) -> RunActionMountInfoSnapshot:
    return RunActionMountInfoSnapshot.from_raw_payload(
        read_run_action_process_mount_info_from_descriptor(
            process_descriptor,
            size_limit_bytes,
        )
    )


def _decode_command_line(arguments: tuple[bytes, ...]) -> tuple[str, ...]:
    if not arguments or any(
        not argument or not argument.isascii() for argument in arguments
    ):
        raise RunActionResolvedWorkloadError(
            "barrier process argv is not complete nonempty ASCII"
        )
    return tuple(argument.decode("ascii") for argument in arguments)


def _init_process_observation(
    running: RunActionBarrierRunningContainerObservation,
    retained: _RetainedProcess,
    root_record: RunActionMountInfoObservation,
) -> RunActionBarrierInitProcessObservation:
    return RunActionBarrierInitProcessObservation.mint(
        provider_execution_id=running.container_id,
        process_id=retained.stat_observation.process_id,
        parent_process_id=retained.stat_observation.parent_process_id,
        process_start_time_ticks=retained.stat_observation.start_time_ticks,
        process_state=retained.stat_observation.state,
        process_cgroup_path=retained.cgroup_path,
        mount_namespace_device=retained.mount_namespace_metadata.device,
        mount_namespace_inode=retained.mount_namespace_metadata.inode,
        process_id_namespace_device=retained.process_id_namespace_metadata.device,
        process_id_namespace_inode=retained.process_id_namespace_metadata.inode,
        command_line=retained.command_line,
        root_mount_info_observation_id=root_record.mount_info_observation_id,
        root_mount_id=root_record.mount_id,
        root_device_major=root_record.device_major,
        root_device_minor=root_record.device_minor,
        root_device=retained.root_metadata.device,
        root_inode=retained.root_metadata.inode,
        executable_mount_id=retained.executable_metadata.mount_id,
        executable_device=retained.executable_metadata.device,
        executable_inode=retained.executable_metadata.inode,
        executable_digest=retained.executable_digest,
    )


def _wrapper_process_observation(
    running: RunActionBarrierRunningContainerObservation,
    init_observation: RunActionBarrierInitProcessObservation,
    retained: _RetainedProcess,
    root_record: RunActionMountInfoObservation,
) -> RunActionBarrierWrapperProcessObservation:
    return RunActionBarrierWrapperProcessObservation.mint(
        provider_execution_id=running.container_id,
        init_process_observation_id=(
            init_observation.barrier_init_process_observation_id
        ),
        process_id=retained.stat_observation.process_id,
        parent_process_id=retained.stat_observation.parent_process_id,
        process_start_time_ticks=retained.stat_observation.start_time_ticks,
        process_state=retained.stat_observation.state,
        process_cgroup_path=retained.cgroup_path,
        mount_namespace_device=retained.mount_namespace_metadata.device,
        mount_namespace_inode=retained.mount_namespace_metadata.inode,
        process_id_namespace_device=retained.process_id_namespace_metadata.device,
        process_id_namespace_inode=retained.process_id_namespace_metadata.inode,
        command_line=retained.command_line,
        root_mount_info_observation_id=root_record.mount_info_observation_id,
        root_mount_id=root_record.mount_id,
        root_device_major=root_record.device_major,
        root_device_minor=root_record.device_minor,
        root_device=retained.root_metadata.device,
        root_inode=retained.root_metadata.inode,
        executable_mount_id=retained.executable_metadata.mount_id,
        executable_device=retained.executable_metadata.device,
        executable_inode=retained.executable_metadata.inode,
        executable_digest=retained.executable_digest,
    )


def _one_mount_record(
    snapshot: RunActionMountInfoSnapshot,
    destination: str,
) -> RunActionMountInfoObservation:
    records = tuple(
        record for record in snapshot.records if record.mount_point == destination
    )
    if len(records) != 1:
        raise RunActionResolvedWorkloadError(
            f"container destination {destination} lacks one exact mount root"
        )
    return records[0]


def _open_resolved_roots(
    descriptors: ExitStack,
    activation: RunActionActivationRevalidationReceipt,
    init_process: _RetainedProcess,
    wrapper_process: _RetainedProcess,
    mount_info_snapshot: RunActionMountInfoSnapshot,
) -> tuple[
    tuple[_RetainedResolvedRoot, ...],
    tuple[RunActionResolvedMountRootObservation, ...],
]:
    projection = (
        activation.prepared_execution.inert_container_evidence.issued_create_projection
    )
    init_source = projection.docker_init_source_evidence
    helper_source = projection.supervisor_helper_evidence
    specifications = [
        (
            RunActionResolvedMountKind.DOCKER_INIT,
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            RunActionPreparedMountAccess.READ_ONLY,
            _SourceAuthority(
                init_source.docker_init_source_evidence_id,
                init_source.mount_id,
                init_source.device,
                init_source.inode,
                init_source.owner_user_id,
                init_source.owner_group_id,
                init_source.mode,
            ),
            "regular",
        ),
        (
            RunActionResolvedMountKind.SUPERVISOR_HELPER,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            RunActionPreparedMountAccess.READ_ONLY,
            _SourceAuthority(
                helper_source.supervisor_helper_evidence_id,
                helper_source.mount_id,
                helper_source.device,
                helper_source.inode,
                helper_source.owner_user_id,
                helper_source.owner_group_id,
                helper_source.mode,
            ),
            "regular",
        ),
    ]
    volume_sources = _volume_source_authorities(activation)
    specifications.extend(
        (
            RunActionResolvedMountKind(prepared_mount.kind.value),
            prepared_mount.container_destination,
            prepared_mount.container_access,
            volume_sources[RunActionResolvedMountKind(prepared_mount.kind.value)],
            "directory",
        )
        for prepared_mount in projection.mounts
    )
    retained_roots = []
    observations = []
    for kind, destination, access, source, file_type in sorted(
        specifications,
        key=lambda specification: specification[1],
    ):
        descriptor = _open_nofollow_container_path(
            descriptors,
            init_process.root_descriptor,
            destination,
            directory=file_type == "directory",
        )
        metadata_before = os.fstat(descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(
            descriptor,
            init_process.process_snapshot_size_limit_bytes,
        )
        metadata_after = os.fstat(descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(
            descriptor,
            init_process.process_snapshot_size_limit_bytes,
        )
        mount_record = _one_mount_record(mount_info_snapshot, destination)
        observed_file_type = (
            "directory"
            if stat.S_ISDIR(metadata_before.st_mode)
            else "regular" if stat.S_ISREG(metadata_before.st_mode) else "unsupported"
        )
        access_option = (
            "ro" if access is RunActionPreparedMountAccess.READ_ONLY else "rw"
        )
        if (
            _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
            or mount_id_before != mount_id_after
            or mount_id_before != mount_record.mount_id
            or (os.major(metadata_before.st_dev), os.minor(metadata_before.st_dev))
            != (mount_record.device_major, mount_record.device_minor)
            or access_option not in mount_record.mount_options
            or metadata_before.st_dev != source.device
            or metadata_before.st_ino != source.inode
            or observed_file_type != file_type
            or metadata_before.st_uid != source.owner_user_id
            or metadata_before.st_gid != source.owner_group_id
            or stat.S_IMODE(metadata_before.st_mode) != source.mode
        ):
            raise RunActionResolvedWorkloadError(
                f"resolved mount root {destination} differs from event-5 source"
            )
        if any(
            PurePosixPath(destination) in PurePosixPath(record.mount_point).parents
            for record in mount_info_snapshot.records
        ):
            raise RunActionResolvedWorkloadError(
                f"resolved mount root {destination} contains a nested mount"
            )
        retained_roots.append(
            _RetainedResolvedRoot(
                destination=destination,
                descriptor=descriptor,
                metadata=_stable_metadata(metadata_before),
                mount_id=mount_id_before,
                process_snapshot_size_limit_bytes=(
                    init_process.process_snapshot_size_limit_bytes
                ),
            )
        )
        observations.append(
            RunActionResolvedMountRootObservation.mint(
                kind=kind,
                source_authority_id=source.authority_id,
                container_destination=destination,
                container_access=access,
                mount_info_observation_id=(mount_record.mount_info_observation_id),
                source_mount_id=source.mount_id,
                source_device=source.device,
                source_inode=source.inode,
                resolved_mount_id=mount_id_before,
                resolved_device=metadata_before.st_dev,
                resolved_inode=metadata_before.st_ino,
                mount_namespace_device=(init_process.mount_namespace_metadata.device),
                mount_namespace_inode=(init_process.mount_namespace_metadata.inode),
                file_type=file_type,
                owner_user_id=metadata_before.st_uid,
                owner_group_id=metadata_before.st_gid,
                mode=stat.S_IMODE(metadata_before.st_mode),
            )
        )
    init_retained = next(
        root
        for root in retained_roots
        if root.destination == RUN_ACTION_DOCKER_INIT_DESTINATION
    )
    helper_retained = next(
        root
        for root in retained_roots
        if root.destination == RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
    )
    if (
        init_retained.mount_id != init_process.executable_metadata.mount_id
        or init_retained.metadata[1] != init_process.executable_metadata.inode
        or init_retained.metadata[2] != init_process.executable_metadata.device
        or helper_retained.mount_id != wrapper_process.executable_metadata.mount_id
        or helper_retained.metadata[1] != wrapper_process.executable_metadata.inode
        or helper_retained.metadata[2] != wrapper_process.executable_metadata.device
    ):
        raise RunActionResolvedWorkloadError(
            "process executables do not join their resolved mount roots"
        )
    return tuple(retained_roots), tuple(observations)


def _volume_source_authorities(
    activation: RunActionActivationRevalidationReceipt,
) -> dict[
    RunActionResolvedMountKind,
    _SourceAuthority,
]:
    prepared = activation.prepared_execution
    directories = {
        observation.kind: observation
        for observation in activation.activated_runtime_directory_observations
    }
    sources = {
        RunActionResolvedMountKind.INPUT: _file_parent_source(
            prepared.input_delivery_slot.prepared_delivery_slot_id,
            activation.input_file_observation,
            prepared.input_delivery_slot.owner_user_id,
            prepared.input_delivery_slot.owner_group_id,
            prepared.input_delivery_slot.mode,
        ),
        RunActionResolvedMountKind.RESULT: _file_parent_source(
            prepared.result_directory.prepared_runtime_directory_id,
            activation.result_file_observation,
            prepared.result_directory.owner_user_id,
            prepared.result_directory.owner_group_id,
            prepared.result_directory.mode,
        ),
        RunActionResolvedMountKind.CONTROL: _runtime_directory_source(
            directories[RunActionPreparedRuntimeDirectoryKind.CONTROL]
        ),
        RunActionResolvedMountKind.TEMPORARY: _runtime_directory_source(
            directories[RunActionPreparedRuntimeDirectoryKind.TEMPORARY]
        ),
    }
    credential = activation.credential_file_observation
    if credential is not None:
        slot = prepared.credential_delivery_slot
        sources[RunActionResolvedMountKind.CREDENTIAL] = _file_parent_source(
            slot.prepared_delivery_slot_id,
            credential,
            slot.owner_user_id,
            slot.owner_group_id,
            slot.mode,
        )
    workspace = activation.activated_workspace_observation
    if workspace is not None:
        sources[RunActionResolvedMountKind.WORKSPACE] = _workspace_source(workspace)
    return sources


def _file_parent_source(
    authority_id: str,
    observation: RunActionActivatedFileObservation,
    owner_user_id: int,
    owner_group_id: int,
    mode: int,
) -> _SourceAuthority:
    return _SourceAuthority(
        authority_id=authority_id,
        mount_id=observation.parent_mount_id,
        device=observation.parent_device,
        inode=observation.parent_inode,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
        mode=mode,
    )


def _runtime_directory_source(
    observation: RunActionActivatedRuntimeDirectoryObservation,
) -> _SourceAuthority:
    return _SourceAuthority(
        authority_id=observation.prepared_runtime_directory_id,
        mount_id=observation.mount_id,
        device=observation.device,
        inode=observation.inode,
        owner_user_id=observation.owner_user_id,
        owner_group_id=observation.owner_group_id,
        mode=observation.mode,
    )


def _workspace_source(
    observation: RunActionActivatedWorkspaceObservation,
) -> _SourceAuthority:
    return _SourceAuthority(
        authority_id=observation.prepared_workspace_proof_id,
        mount_id=observation.mount_id,
        device=observation.device,
        inode=observation.inode,
        owner_user_id=observation.owner_user_id,
        owner_group_id=observation.owner_group_id,
        mode=observation.root_mode,
    )


def _open_nofollow_container_path(
    descriptors: ExitStack,
    container_root_descriptor: int,
    path: str,
    *,
    directory: bool,
) -> int:
    parsed = PurePosixPath(path)
    if (
        not parsed.is_absolute()
        or parsed.as_posix() != path
        or path == "/"
        or ".." in parsed.parts
    ):
        raise RunActionResolvedWorkloadError(
            "container path is not canonical and absolute"
        )
    parent_descriptor = container_root_descriptor
    components = parsed.parts[1:]
    for position, component in enumerate(components):
        is_final = position == len(components) - 1
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        if not is_final or directory:
            flags |= os.O_DIRECTORY
        descriptor = os.open(component, flags, dir_fd=parent_descriptor)
        descriptors.callback(os.close, descriptor)
        parent_descriptor = descriptor
    return parent_descriptor


def _require_retained_root_current(
    container_root_descriptor: int,
    retained: _RetainedResolvedRoot,
) -> None:
    retained_before = os.fstat(retained.descriptor)
    retained_mount_id_before = read_run_action_descriptor_mount_id(
        retained.descriptor,
        retained.process_snapshot_size_limit_bytes,
    )
    with ExitStack() as descriptors:
        current_descriptor = _open_nofollow_container_path(
            descriptors,
            container_root_descriptor,
            retained.destination,
            directory=stat.S_ISDIR(retained_before.st_mode),
        )
        current_before = os.fstat(current_descriptor)
        current_mount_id = read_run_action_descriptor_mount_id(
            current_descriptor,
            retained.process_snapshot_size_limit_bytes,
        )
        current_after = os.fstat(current_descriptor)
    retained_after = os.fstat(retained.descriptor)
    retained_mount_id_after = read_run_action_descriptor_mount_id(
        retained.descriptor,
        retained.process_snapshot_size_limit_bytes,
    )
    if (
        _stable_metadata(retained_before) != retained.metadata
        or _stable_metadata(retained_after) != retained.metadata
        or retained_mount_id_before != retained.mount_id
        or retained_mount_id_after != retained.mount_id
        or _stable_metadata(current_before) != retained.metadata
        or _stable_metadata(current_after) != retained.metadata
        or current_mount_id != retained.mount_id
    ):
        raise RunActionResolvedWorkloadError(
            f"resolved root {retained.destination} was replaced or spliced"
        )


def _retained_roots_by_kind(
    observations: tuple[RunActionResolvedMountRootObservation, ...],
    retained_roots: tuple[_RetainedResolvedRoot, ...],
) -> dict[RunActionResolvedMountKind, _RetainedResolvedRoot]:
    observations_by_destination = {
        observation.container_destination: observation for observation in observations
    }
    if len(observations_by_destination) != len(observations) or {
        root.destination for root in retained_roots
    } != set(observations_by_destination):
        raise RunActionResolvedWorkloadError(
            "retained roots differ from resolved mount observations"
        )
    return {
        observations_by_destination[root.destination].kind: root
        for root in retained_roots
    }


def _require_logical_mounts_current(
    resolved: RunActionResolvedWorkloadObservation,
    retained_roots: tuple[_RetainedResolvedRoot, ...],
    launch_settings: LaunchSettings,
) -> None:
    roots = resolved.resolved_mount_root_observations
    roots_by_kind = {root.kind: root for root in roots}
    retained_by_kind = _retained_roots_by_kind(roots, retained_roots)
    current_files = _resolved_file_observations(
        resolved.activation_revalidation_receipt,
        roots_by_kind,
        retained_by_kind,
    )
    current_workspace = _resolved_workspace_observation(
        resolved.activation_revalidation_receipt,
        roots_by_kind,
        retained_by_kind,
        launch_settings,
    )
    control_entries = _exact_directory_entries(
        retained_by_kind[RunActionResolvedMountKind.CONTROL].descriptor
    )
    temporary_entries = _exact_directory_entries(
        retained_by_kind[RunActionResolvedMountKind.TEMPORARY].descriptor
    )
    if (
        current_files != resolved.resolved_file_observations
        or current_workspace != resolved.resolved_workspace_observation
        or control_entries
        or temporary_entries
    ):
        raise RunActionResolvedWorkloadError(
            "logical files or runtime directories changed while workload was blocked"
        )


def _resolved_file_observations(
    activation: RunActionActivationRevalidationReceipt,
    roots_by_kind: dict[
        RunActionResolvedMountKind,
        RunActionResolvedMountRootObservation,
    ],
    retained_roots_by_kind: dict[RunActionResolvedMountKind, _RetainedResolvedRoot],
) -> tuple[RunActionResolvedFileObservation, ...]:
    observations = []
    for activated in (
        activation.input_file_observation,
        activation.result_file_observation,
        activation.credential_file_observation,
    ):
        if activated is None:
            continue
        kind = RunActionResolvedMountKind(activated.kind.value)
        retained_root = retained_roots_by_kind[kind]
        root = roots_by_kind[kind]
        entries = _exact_directory_entries(retained_root.descriptor)
        expected_name = PurePosixPath(activated.relative_path).name
        if entries != (expected_name,):
            raise RunActionResolvedWorkloadError(
                f"{kind.value} mount does not contain its exact logical file"
            )
        descriptor = os.open(
            expected_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=retained_root.descriptor,
        )
        with os.fdopen(descriptor, "rb") as handle:
            metadata_before = os.fstat(handle.fileno())
            mount_id_before = read_run_action_descriptor_mount_id(
                handle.fileno(),
                retained_root.process_snapshot_size_limit_bytes,
            )
            content_digest = None
            if activated.kind is RunActionPreparedFileKind.INPUT:
                payload = handle.read(activated.size_bytes + 1)
                if len(payload) != activated.size_bytes:
                    raise RunActionResolvedWorkloadError(
                        "resolved input differs from its complete event-5 size"
                    )
                content_digest = tree_or_blob_digest(payload)
            metadata_after = os.fstat(handle.fileno())
            mount_id_after = read_run_action_descriptor_mount_id(
                handle.fileno(),
                retained_root.process_snapshot_size_limit_bytes,
            )
        if (
            _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
            or mount_id_before != mount_id_after
            or mount_id_before != retained_root.mount_id
            or metadata_before.st_dev != activated.device
            or metadata_before.st_ino != activated.inode
            or not stat.S_ISREG(metadata_before.st_mode)
            or metadata_before.st_uid != activated.owner_user_id
            or metadata_before.st_gid != activated.owner_group_id
            or stat.S_IMODE(metadata_before.st_mode) != activated.mode
            or metadata_before.st_nlink != activated.link_count
            or metadata_before.st_size != activated.size_bytes
            or content_digest != activated.content_digest
        ):
            raise RunActionResolvedWorkloadError(
                f"resolved {kind.value} file differs from event-5 observation"
            )
        observations.append(
            RunActionResolvedFileObservation.mint(
                kind=activated.kind,
                activated_file_observation_id=(activated.activated_file_observation_id),
                resolved_mount_root_observation_id=(
                    root.resolved_mount_root_observation_id
                ),
                container_path=(
                    PurePosixPath(root.container_destination) / expected_name
                ).as_posix(),
                parent_entry_count=1,
                mount_id=mount_id_before,
                device=metadata_before.st_dev,
                inode=metadata_before.st_ino,
                file_type="regular",
                owner_user_id=metadata_before.st_uid,
                owner_group_id=metadata_before.st_gid,
                mode=stat.S_IMODE(metadata_before.st_mode),
                link_count=metadata_before.st_nlink,
                size_bytes=metadata_before.st_size,
                content_digest=content_digest,
                content_authority_id=activated.content_authority_id,
            )
        )
    return tuple(sorted(observations, key=lambda observation: observation.kind.value))


def _resolved_workspace_observation(
    activation: RunActionActivationRevalidationReceipt,
    roots_by_kind: dict[
        RunActionResolvedMountKind,
        RunActionResolvedMountRootObservation,
    ],
    retained_roots_by_kind: dict[RunActionResolvedMountKind, _RetainedResolvedRoot],
    launch_settings: LaunchSettings,
) -> RunActionResolvedWorkspaceObservation | None:
    activated = activation.activated_workspace_observation
    if activated is None:
        return None
    retained = retained_roots_by_kind[RunActionResolvedMountKind.WORKSPACE]
    prepared_workspace = activation.prepared_execution.workspace_proof
    if prepared_workspace is None:
        raise RunActionResolvedWorkloadError(
            "activated workspace lacks its prepared frontier"
        )
    observed = inspect_run_workspace_frontier(
        retained.descriptor,
        settings=launch_settings,
        expected_commit_sha=prepared_workspace.workspace_binding.commit_sha,
    )
    if (
        observed.source_tree_digest != activated.source_tree_digest
        or observed.git_closure_digest != activated.git_closure_digest
        or observed.source_entry_count != activated.source_entry_count
        or observed.source_size_bytes != activated.source_size_bytes
    ):
        raise RunActionResolvedWorkloadError(
            "resolved workspace differs from event-5 frontier"
        )
    root = roots_by_kind[RunActionResolvedMountKind.WORKSPACE]
    return RunActionResolvedWorkspaceObservation.mint(
        activated_workspace_observation_id=(
            activated.activated_workspace_observation_id
        ),
        resolved_mount_root_observation_id=root.resolved_mount_root_observation_id,
        source_tree_digest=observed.source_tree_digest,
        git_closure_digest=observed.git_closure_digest,
        source_entry_count=observed.source_entry_count,
        source_size_bytes=observed.source_size_bytes,
    )


def _exact_directory_entries(descriptor: int) -> tuple[str, ...]:
    metadata_before = os.fstat(descriptor)
    entries = tuple(sorted(os.listdir(descriptor)))
    metadata_after = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata_before.st_mode)
        or _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
        or any(
            not entry or entry in {".", ".."} or "/" in entry or "\x00" in entry
            for entry in entries
        )
        or len(entries) != len(set(entries))
    ):
        raise RunActionResolvedWorkloadError(
            "resolved directory changed or contains malformed entries"
        )
    return entries


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
    )


__all__ = [
    "RunActionBlockedWorkloadLease",
    "RunActionResolvedWorkloadError",
    "open_run_action_blocked_workload",
]
