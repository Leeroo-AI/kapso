"""Descriptor-bound preparation and observation of one runtime-volume generation."""

from __future__ import annotations

import ctypes
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import get_ident

from kapso.cross_run.canonical import is_content_id, tree_or_blob_digest
from kapso.cross_run.launch.run_action_activation_delivery import (
    RunActionDeliveredFilePhysicalObservation,
    publish_or_adopt_run_action_delivery,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_inert_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    target_command_from_main_projection,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
    DockerRunActionResourceInventory,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
    read_run_action_process_cgroup_path_from_descriptor,
    read_run_action_process_stat_from_descriptor,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RunActionActivatedFileObservation,
    RunActionActivatedSentinelObservation,
    RunActionActivatedRuntimeDirectoryObservation,
    RunActionActivatedWorkspaceObservation,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionInertContainerEvidence,
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    RunActionPreparedDeliverySlot,
    RunActionPreparedExecution,
    RunActionPreparedFile,
    RunActionPreparedFileKind,
    RunActionPreparedRuntimeDirectory,
    RunActionPreparedRuntimeDirectoryKind,
    RunActionPreparedWorkspaceProof,
    RunActionResultCaptureReceipt,
    RunActionRuntimeVolumeAuthority,
    RunActionRuntimeVolumeEvidence,
    RunActionRuntimeVolumeLayoutProof,
    RunActionRuntimeVolumeSentinelEvidence,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
    issue_runtime_volume_authority,
    run_action_activated_volume_evidence_matches,
    run_action_keeper_process_cgroup_path,
)
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceCopyPlan,
    RunWorkspaceFrontierIdentity,
    copy_run_workspace_frontier,
    inspect_run_workspace_frontier,
    plan_run_workspace_frontier_copy,
)
from kapso.cross_run.settings import LaunchSettings

_TMPFS_FILESYSTEM_TYPE = "tmpfs"
_MOUNT_OPTIONS = ("nodev", "nosuid", "relatime", "rw")
_SUPER_OPTION_FLAGS = ("inode64", "noswap", "rw")
_SUPER_OPTION_KEYS = ("gid", "mode", "nr_inodes", "size", "uid")
_OPTIONAL_MOUNT_FIELD_PATTERN = re.compile(r"^master:[1-9][0-9]*$")
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIZE_OPTION_PATTERN = re.compile(r"^([1-9][0-9]*)([kmgt]?)$")
_RENAME_NOREPLACE = 1
_PREPARATION_STAGING_PREFIX = ".kapso-prepare-"
_PENDING_SENTINEL_PREFIX = ".kapso-generation.pending-"
_SENTINEL_NAME = ".kapso-generation"
_PREPARED_DIRECTORY_MODE = 0o700
_PREPARED_FILE_MODE = 0o600
_SENTINEL_MODE = 0o400
_RESULT_WORKSPACE_LEASE_AUTHORITY = object()
_BARRIER_CONTROL_LEASE_AUTHORITY = object()
_ACTIVATION_REVALIDATION_LEASE_AUTHORITY = object()
_SIZE_MULTIPLIERS = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}


class RunActionRuntimeVolumeError(RuntimeError):
    """The mounted runtime volume differs from its issued tmpfs authority."""


class RunActionControlDirectoryLease:
    """Process-bound lease for one exact semantic control topology."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        mounted_volume: "_MountedRuntimeVolumeLease",
        prepared: RunActionPreparedExecution,
        sentinel_observation: "_ExactRegularFileObservation",
        control_descriptor: int,
        control_metadata_identity: tuple[int, ...],
        topology: RunActionControlDirectoryTopology,
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(mounted_volume) is not _MountedRuntimeVolumeLease
            or type(prepared) is not RunActionPreparedExecution
            or type(sentinel_observation) is not _ExactRegularFileObservation
            or type(control_descriptor) is not int
            or control_descriptor < 0
            or type(control_metadata_identity) is not tuple
            or control_metadata_identity
            != _stable_metadata(os.fstat(control_descriptor))
            or type(topology) is not RunActionControlDirectoryTopology
            or _authority is not _BARRIER_CONTROL_LEASE_AUTHORITY
        ):
            raise RunActionRuntimeVolumeError(
                "barrier control lease lacks exact physical authority"
            )
        self._descriptors = descriptors
        self._mounted_volume = mounted_volume
        self._prepared = prepared
        self._sentinel_observation = sentinel_observation
        self._control_descriptor = control_descriptor
        self._control_metadata_identity = control_metadata_identity
        self._topology = topology
        self._owner_process_id = os.getpid()
        self._closed = False
        self.require_current()

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        self.require_current()
        return self._prepared

    @property
    def topology(self) -> RunActionControlDirectoryTopology:
        self.require_current()
        return self._topology

    def require_current(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionRuntimeVolumeError(
                "barrier control lease is closed or belongs to another process"
            )
        prepared_volume = self._prepared.runtime_volume_evidence
        _require_same_mounted_runtime_volume(
            self._mounted_volume,
            self._prepared.volume_keeper_evidence,
        )
        _require_same_exact_regular_file(self._sentinel_observation)
        control_metadata = os.fstat(self._control_descriptor)
        if (
            self._mounted_volume.root_mount_id != prepared_volume.root_mount_id
            or self._mounted_volume.root_device != prepared_volume.root_device
            or self._mounted_volume.root_inode != prepared_volume.root_inode
            or _stable_metadata(control_metadata) != self._control_metadata_identity
        ):
            raise RunActionRuntimeVolumeError(
                "barrier control lease changed physical generation"
            )
        _require_exact_activation_directory(
            self._prepared.control_directory,
            self._mounted_volume.root_descriptor,
            self._control_descriptor,
            expected_entries=self._topology.entries,
        )

    def reobserve_runtime_volume_evidence(
        self,
        volume: DockerRunActionVolumeObservation,
        keeper: RunActionVolumeKeeperEvidence,
    ) -> RunActionRuntimeVolumeEvidence:
        """Sample current usage for the retained prepared volume occurrence."""

        prepared = self._prepared
        authority = prepared.runtime_volume_authority
        if (
            type(volume) is not DockerRunActionVolumeObservation
            or type(keeper) is not RunActionVolumeKeeperEvidence
            or keeper != prepared.volume_keeper_evidence
            or volume.volume_authority_id != authority.runtime_volume_authority_id
            or volume.volume_name != authority.volume_name
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume reobservation lacks exact retained authority"
            )
        self.require_current()
        root_descriptor = self._mounted_volume.root_descriptor
        filesystem_before = os.fstatvfs(root_descriptor)
        root_metadata_before = os.fstat(root_descriptor)
        mount_info_before = _read_mount_info(
            self._mounted_volume.process_descriptor,
            self._mounted_volume.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        _require_mount_authority(mount_info_before, root_metadata_before, authority)
        self.require_current()
        filesystem_after = os.fstatvfs(root_descriptor)
        root_metadata_after = os.fstat(root_descriptor)
        mount_info_after = _read_mount_info(
            self._mounted_volume.process_descriptor,
            self._mounted_volume.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        if (
            _stable_filesystem(filesystem_after)
            != _stable_filesystem(filesystem_before)
            or _root_metadata_identity(root_metadata_after)
            != _root_metadata_identity(root_metadata_before)
            or mount_info_after != mount_info_before
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume changed during retained reobservation"
            )
        evidence = _mint_runtime_volume_evidence(
            authority,
            volume,
            keeper,
            root_mount_id=self._mounted_volume.root_mount_id,
            root_device=self._mounted_volume.root_device,
            root_inode=self._mounted_volume.root_inode,
            sentinel_evidence=prepared.runtime_volume_evidence.sentinel_evidence,
            filesystem=filesystem_after,
        )
        _require_result_volume_occurrence(prepared, evidence)
        self.require_current()
        return evidence

    def __enter__(self) -> "RunActionControlDirectoryLease":
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionRuntimeVolumeError(
                "barrier control lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


class RunActionResultWorkspaceLease:
    """Process-bound descriptor lease for the exact terminal runtime workspace."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        mounted_volume: "_MountedRuntimeVolumeLease",
        keeper: RunActionVolumeKeeperEvidence,
        sentinel_observation: "_ExactRegularFileObservation",
        workspace_descriptor: int,
        workspace_proof: RunActionPreparedWorkspaceProof,
        workspace_metadata_identity: tuple[int, ...],
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(mounted_volume) is not _MountedRuntimeVolumeLease
            or type(keeper) is not RunActionVolumeKeeperEvidence
            or type(sentinel_observation) is not _ExactRegularFileObservation
            or type(workspace_descriptor) is not int
            or workspace_descriptor < 0
            or type(workspace_proof) is not RunActionPreparedWorkspaceProof
            or type(workspace_metadata_identity) is not tuple
            or workspace_metadata_identity
            != _stable_metadata(os.fstat(workspace_descriptor))
            or _authority is not _RESULT_WORKSPACE_LEASE_AUTHORITY
        ):
            raise RunActionRuntimeVolumeError(
                "result workspace lease lacks exact physical authority"
            )
        self._descriptors = descriptors
        self._mounted_volume = mounted_volume
        self._keeper = keeper
        self._sentinel_observation = sentinel_observation
        self._workspace_descriptor = workspace_descriptor
        self._workspace_proof = workspace_proof
        self._workspace_metadata_identity = workspace_metadata_identity
        self._owner_process_id = os.getpid()
        self._closed = False
        self.require_current()

    @property
    def workspace_descriptor(self) -> int:
        self.require_current()
        return self._workspace_descriptor

    def require_current(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionRuntimeVolumeError(
                "result workspace lease is closed or belongs to another process"
            )
        _require_same_mounted_runtime_volume(
            self._mounted_volume,
            self._keeper,
        )
        _require_same_exact_regular_file(self._sentinel_observation)
        opened_before = os.fstat(self._workspace_descriptor)
        opened_mount_id = read_run_action_descriptor_mount_id(
            self._workspace_descriptor
        )
        with ExitStack() as descriptors:
            current_descriptor = os.open(
                self._workspace_proof.volume_subpath,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=self._mounted_volume.root_descriptor,
            )
            descriptors.callback(os.close, current_descriptor)
            current_before = os.fstat(current_descriptor)
            current_mount_id = read_run_action_descriptor_mount_id(current_descriptor)
            path_before = os.stat(
                self._workspace_proof.volume_subpath,
                dir_fd=self._mounted_volume.root_descriptor,
                follow_symlinks=False,
            )
            opened_after = os.fstat(self._workspace_descriptor)
            current_after = os.fstat(current_descriptor)
            path_after = os.stat(
                self._workspace_proof.volume_subpath,
                dir_fd=self._mounted_volume.root_descriptor,
                follow_symlinks=False,
            )
        expected_metadata = self._workspace_metadata_identity
        if (
            not _result_workspace_matches_prepared(
                opened_before,
                opened_mount_id,
                self._workspace_proof,
            )
            or not _result_workspace_matches_prepared(
                current_before,
                current_mount_id,
                self._workspace_proof,
            )
            or _stable_metadata(opened_before) != expected_metadata
            or _stable_metadata(opened_after) != expected_metadata
            or _stable_metadata(current_before) != expected_metadata
            or _stable_metadata(current_after) != expected_metadata
            or _stable_metadata(path_before) != expected_metadata
            or _stable_metadata(path_after) != expected_metadata
        ):
            raise RunActionRuntimeVolumeError(
                "result workspace lease changed physical generation"
            )

    def __enter__(self) -> "RunActionResultWorkspaceLease":
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionRuntimeVolumeError(
                "result workspace lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


@dataclass(frozen=True)
class DockerRunActionPreparedVolumeObservation:
    """Complete prepared layout proven through one keeper process generation."""

    preparation_claim: RunActionPreparationClaim
    runtime_volume_evidence: RunActionRuntimeVolumeEvidence
    input_delivery_slot: RunActionPreparedDeliverySlot
    control_directory: RunActionPreparedRuntimeDirectory
    result_directory: RunActionPreparedRuntimeDirectory
    result_file: RunActionPreparedFile
    temporary_directory: RunActionPreparedRuntimeDirectory
    credential_delivery_slot: RunActionPreparedDeliverySlot | None
    workspace_proof: RunActionPreparedWorkspaceProof | None
    layout_proof: RunActionRuntimeVolumeLayoutProof

    def __post_init__(self) -> None:
        authority = (
            self.runtime_volume_evidence.volume_authority
            if type(self.runtime_volume_evidence) is RunActionRuntimeVolumeEvidence
            else None
        )
        delivery_slots = tuple(
            delivery_slot
            for delivery_slot in (
                self.input_delivery_slot,
                self.credential_delivery_slot,
            )
            if delivery_slot is not None
        )
        expected_delivery_slot_plans = (
            _expected_delivery_slot_plans(self.preparation_claim)
            if type(self.preparation_claim) is RunActionPreparationClaim
            else ()
        )
        expected_result_file_plan = (
            _expected_result_file_plan(self.preparation_claim)
            if type(self.preparation_claim) is RunActionPreparationClaim
            else None
        )
        expected_runtime_directory_plans = (
            _expected_runtime_directory_plans()
            if type(self.preparation_claim) is RunActionPreparationClaim
            else ()
        )
        runtime_directories = (
            self.control_directory,
            self.result_directory,
            self.temporary_directory,
        )
        expected_authority = (
            issue_runtime_volume_authority(
                self.preparation_claim,
                authority.generation_nonce,
            )
            if type(self.preparation_claim) is RunActionPreparationClaim
            and type(authority) is RunActionRuntimeVolumeAuthority
            else None
        )
        workspace_binding = (
            self.preparation_claim.reservation.frontier.workspace_before
            if type(self.preparation_claim) is RunActionPreparationClaim
            else None
        )
        expected_directory_names = (
            _expected_directory_names(self.preparation_claim)
            if type(self.preparation_claim) is RunActionPreparationClaim
            else ()
        )
        workspace_size_bytes = (
            0 if workspace_binding is None else workspace_binding.source_size_bytes
        )
        workspace_entry_count = (
            0 if workspace_binding is None else workspace_binding.source_entry_count
        )
        if (
            type(self.preparation_claim) is not RunActionPreparationClaim
            or type(self.runtime_volume_evidence) is not RunActionRuntimeVolumeEvidence
            or type(self.input_delivery_slot) is not RunActionPreparedDeliverySlot
            or type(self.control_directory) is not RunActionPreparedRuntimeDirectory
            or type(self.result_directory) is not RunActionPreparedRuntimeDirectory
            or type(self.result_file) is not RunActionPreparedFile
            or type(self.temporary_directory) is not RunActionPreparedRuntimeDirectory
            or (
                self.credential_delivery_slot is not None
                and type(self.credential_delivery_slot)
                is not RunActionPreparedDeliverySlot
            )
            or (
                self.workspace_proof is not None
                and type(self.workspace_proof) is not RunActionPreparedWorkspaceProof
            )
            or type(self.layout_proof) is not RunActionRuntimeVolumeLayoutProof
            or authority != expected_authority
            or len(delivery_slots) != len(expected_delivery_slot_plans)
            or any(
                runtime_directory.kind is not directory_plan.kind
                or runtime_directory.directory_relative_path
                != directory_plan.directory_name
                or runtime_directory.observed_entry_count
                != directory_plan.observed_entry_count
                or runtime_directory.preparation_claim_id
                != self.preparation_claim.preparation_claim_id
                or runtime_directory.runtime_volume_authority_id
                != authority.runtime_volume_authority_id
                or runtime_directory.generation_nonce != authority.generation_nonce
                or runtime_directory.owner_user_id != authority.owner_user_id
                or runtime_directory.owner_group_id != authority.owner_group_id
                for runtime_directory, directory_plan in zip(
                    runtime_directories,
                    expected_runtime_directory_plans,
                    strict=True,
                )
            )
            or any(
                (
                    delivery_slot.preparation_claim_id
                    != self.preparation_claim.preparation_claim_id
                    or delivery_slot.runtime_volume_authority_id
                    != authority.runtime_volume_authority_id
                    or delivery_slot.generation_nonce != authority.generation_nonce
                    or delivery_slot.kind is not expected_delivery_slot_plan.kind
                    or delivery_slot.directory_relative_path
                    != expected_delivery_slot_plan.directory_name
                    or delivery_slot.final_file_name
                    != expected_delivery_slot_plan.final_file_name
                    or delivery_slot.owner_user_id != authority.owner_user_id
                    or delivery_slot.owner_group_id != authority.owner_group_id
                    or delivery_slot.payload_size_limit_bytes
                    != expected_delivery_slot_plan.payload_size_limit_bytes
                )
                for delivery_slot, expected_delivery_slot_plan in zip(
                    delivery_slots,
                    expected_delivery_slot_plans,
                    strict=True,
                )
            )
            or expected_result_file_plan is None
            or self.result_file.preparation_claim_id
            != self.preparation_claim.preparation_claim_id
            or self.result_file.prepared_parent_directory_id
            != self.result_directory.prepared_runtime_directory_id
            or self.result_file.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.result_file.generation_nonce != authority.generation_nonce
            or self.result_file.kind is not expected_result_file_plan.kind
            or self.result_file.relative_path != expected_result_file_plan.relative_path
            or self.result_file.owner_user_id != authority.owner_user_id
            or self.result_file.owner_group_id != authority.owner_group_id
            or self.result_file.payload_size_limit_bytes
            != expected_result_file_plan.payload_size_limit_bytes
            or any(
                delivery_slot.mount_id != self.runtime_volume_evidence.root_mount_id
                or delivery_slot.device != self.runtime_volume_evidence.root_device
                for delivery_slot in delivery_slots
            )
            or any(
                runtime_directory.mount_id != self.runtime_volume_evidence.root_mount_id
                or runtime_directory.device != self.runtime_volume_evidence.root_device
                for runtime_directory in runtime_directories
            )
            or self.result_file.mount_id != self.runtime_volume_evidence.root_mount_id
            or self.result_file.device != self.runtime_volume_evidence.root_device
            or (
                self.workspace_proof is not None
                and (
                    self.workspace_proof.mount_id
                    != self.runtime_volume_evidence.root_mount_id
                    or self.workspace_proof.device
                    != self.runtime_volume_evidence.root_device
                )
            )
            or len(
                {
                    *(delivery_slot.inode for delivery_slot in delivery_slots),
                    *(directory.inode for directory in runtime_directories),
                    self.result_file.inode,
                    *(
                        ()
                        if self.workspace_proof is None
                        else (self.workspace_proof.inode,)
                    ),
                }
            )
            != (
                len(delivery_slots)
                + len(runtime_directories)
                + 1
                + (0 if self.workspace_proof is None else 1)
            )
            or {
                *(delivery_slot.inode for delivery_slot in delivery_slots),
                *(directory.inode for directory in runtime_directories),
                self.result_file.inode,
                *(
                    ()
                    if self.workspace_proof is None
                    else (self.workspace_proof.inode,)
                ),
            }
            & {
                self.runtime_volume_evidence.root_inode,
                self.runtime_volume_evidence.sentinel_evidence.inode,
            }
            or self.layout_proof.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.layout_proof.runtime_volume_evidence_id
            != self.runtime_volume_evidence.runtime_volume_evidence_id
            or self.layout_proof.generation_nonce != authority.generation_nonce
            or self.layout_proof.directory_relative_paths != expected_directory_names
            or self.layout_proof.prepared_delivery_slot_ids
            != tuple(sorted(slot.prepared_delivery_slot_id for slot in delivery_slots))
            or self.layout_proof.prepared_result_file_id
            != self.result_file.prepared_file_id
            or self.layout_proof.prepared_runtime_directory_ids
            != tuple(
                sorted(
                    directory.prepared_runtime_directory_id
                    for directory in runtime_directories
                )
            )
            or self.layout_proof.prepared_workspace_proof_id
            != (
                None
                if self.workspace_proof is None
                else self.workspace_proof.prepared_workspace_proof_id
            )
            or self.layout_proof.logical_content_size_bytes
            != len(authority.generation_nonce) + workspace_size_bytes
            or self.layout_proof.logical_entry_count
            != (len(expected_directory_names) + 2 + workspace_entry_count)
            or self.layout_proof.observed_used_size_bytes
            != self.runtime_volume_evidence.used_size_bytes
            or self.layout_proof.observed_used_inode_count
            != self.runtime_volume_evidence.used_inode_count
            or (
                self.workspace_proof is not None
                and (
                    self.workspace_proof.preparation_claim_id
                    != self.preparation_claim.preparation_claim_id
                    or self.workspace_proof.runtime_volume_authority_id
                    != authority.runtime_volume_authority_id
                    or self.workspace_proof.generation_nonce
                    != authority.generation_nonce
                    or self.workspace_proof.workspace_binding != workspace_binding
                    or self.workspace_proof.owner_user_id != authority.owner_user_id
                    or self.workspace_proof.owner_group_id != authority.owner_group_id
                    or self.workspace_proof.root_mode != authority.root_mode
                )
            )
            or (self.workspace_proof is None) != (workspace_binding is None)
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume observation is incomplete"
            )


@dataclass(frozen=True)
class DockerRunActionActivatedVolumeObservation:
    """Exact post-delivery volume evidence ready for final inert inspection."""

    prepared_execution: RunActionPreparedExecution
    spawn_commit: RunActionSpawnCommit
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence
    activated_workspace_observation: RunActionActivatedWorkspaceObservation | None
    activated_runtime_directory_observations: tuple[
        RunActionActivatedRuntimeDirectoryObservation, ...
    ]
    activated_sentinel_observation: RunActionActivatedSentinelObservation
    input_file_observation: RunActionActivatedFileObservation
    result_file_observation: RunActionActivatedFileObservation
    credential_file_observation: RunActionActivatedFileObservation | None

    def __post_init__(self) -> None:
        prepared = self.prepared_execution
        if (
            type(prepared) is not RunActionPreparedExecution
            or type(self.spawn_commit) is not RunActionSpawnCommit
            or type(self.reobserved_volume_evidence)
            is not RunActionRuntimeVolumeEvidence
            or (
                self.activated_workspace_observation is not None
                and type(self.activated_workspace_observation)
                is not RunActionActivatedWorkspaceObservation
            )
            or type(self.activated_runtime_directory_observations) is not tuple
            or any(
                type(observation) is not RunActionActivatedRuntimeDirectoryObservation
                for observation in self.activated_runtime_directory_observations
            )
            or type(self.activated_sentinel_observation)
            is not RunActionActivatedSentinelObservation
            or type(self.input_file_observation)
            is not RunActionActivatedFileObservation
            or type(self.result_file_observation)
            is not RunActionActivatedFileObservation
            or (
                self.credential_file_observation is not None
                and type(self.credential_file_observation)
                is not RunActionActivatedFileObservation
            )
        ):
            raise RunActionRuntimeVolumeError(
                "activated runtime volume observation is incomplete"
            )
        if not run_action_activated_volume_evidence_matches(
            prepared=prepared,
            spawn_commit=self.spawn_commit,
            reobserved_volume_evidence=self.reobserved_volume_evidence,
            activated_workspace_observation=self.activated_workspace_observation,
            activated_runtime_directory_observations=(
                self.activated_runtime_directory_observations
            ),
            activated_sentinel_observation=self.activated_sentinel_observation,
            input_file_observation=self.input_file_observation,
            result_file_observation=self.result_file_observation,
            credential_file_observation=self.credential_file_observation,
        ):
            raise RunActionRuntimeVolumeError(
                "activated runtime volume observation differs from its spawn"
            )


class RunActionActivationRevalidationLease:
    """Retained read-only authority for one durably selected inert activation."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        preparation_allocation: RunActionPreparationAllocation,
        selected_receipt: RunActionActivationRevalidationReceipt,
        resource_manager: DockerRunActionResourceManager,
        inventory: DockerRunActionResourceInventory,
        volume_observation: DockerRunActionVolumeObservation,
        keeper_evidence: RunActionVolumeKeeperEvidence,
        inert_container_evidence: RunActionInertContainerEvidence,
        mounted_volume: "_MountedRuntimeVolumeLease",
        sentinel_observation: "_ExactRegularFileObservation",
        input_file_observation: "_ExactRegularFileObservation",
        result_file_observation: "_ExactRegularFileObservation",
        credential_file_observation: "_ExactRegularFileShapeObservation | None",
        directory_descriptors: tuple[
            tuple[
                RunActionPreparedDeliverySlot
                | RunActionPreparedRuntimeDirectory
                | RunActionPreparedWorkspaceProof,
                int,
                tuple[str, ...] | None,
            ],
            ...,
        ],
        activated_workspace_frontier: RunWorkspaceFrontierIdentity | None,
        filesystem_identity: tuple[int, ...],
        root_metadata_identity: tuple[int, ...],
        mount_info: "_MountInfo",
        settings: LaunchSettings,
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(preparation_allocation) is not RunActionPreparationAllocation
            or type(selected_receipt) is not RunActionActivationRevalidationReceipt
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(inventory) is not DockerRunActionResourceInventory
            or type(volume_observation) is not DockerRunActionVolumeObservation
            or type(keeper_evidence) is not RunActionVolumeKeeperEvidence
            or type(inert_container_evidence) is not RunActionInertContainerEvidence
            or type(mounted_volume) is not _MountedRuntimeVolumeLease
            or type(sentinel_observation) is not _ExactRegularFileObservation
            or type(input_file_observation) is not _ExactRegularFileObservation
            or type(result_file_observation) is not _ExactRegularFileObservation
            or (
                credential_file_observation is not None
                and type(credential_file_observation)
                is not _ExactRegularFileShapeObservation
            )
            or type(directory_descriptors) is not tuple
            or not directory_descriptors
            or any(
                type(directory) is not tuple
                or len(directory) != 3
                or type(directory[1]) is not int
                or directory[1] < 0
                for directory in directory_descriptors
            )
            or (
                activated_workspace_frontier is not None
                and type(activated_workspace_frontier)
                is not RunWorkspaceFrontierIdentity
            )
            or type(filesystem_identity) is not tuple
            or type(root_metadata_identity) is not tuple
            or type(mount_info) is not _MountInfo
            or type(settings) is not LaunchSettings
            or _authority is not _ACTIVATION_REVALIDATION_LEASE_AUTHORITY
        ):
            raise RunActionRuntimeVolumeError(
                "activation revalidation lease lacks exact retained authority"
            )
        self._descriptors = descriptors
        self._preparation_allocation = preparation_allocation
        self._selected_receipt = selected_receipt
        self._resource_manager = resource_manager
        self._inventory = inventory
        self._volume_observation = volume_observation
        self._keeper_evidence = keeper_evidence
        self._inert_container_evidence = inert_container_evidence
        self._mounted_volume = mounted_volume
        self._sentinel_observation = sentinel_observation
        self._input_file_observation = input_file_observation
        self._result_file_observation = result_file_observation
        self._credential_file_observation = credential_file_observation
        self._directory_descriptors = directory_descriptors
        self._activated_workspace_frontier = activated_workspace_frontier
        self._filesystem_identity = filesystem_identity
        self._root_metadata_identity = root_metadata_identity
        self._mount_info = mount_info
        self._settings = settings
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    @property
    def selected_receipt(self) -> RunActionActivationRevalidationReceipt:
        self.require_current()
        return self._selected_receipt

    @property
    def preparation_allocation(self) -> RunActionPreparationAllocation:
        self.require_current()
        return self._preparation_allocation

    @property
    def inventory(self) -> DockerRunActionResourceInventory:
        self.require_current()
        return self._inventory

    def require_current(self) -> None:
        """Reprove the selected activation and its still-inert main."""

        self._require_owner()
        _require_selected_activation_lease_current(
            self,
            require_inert_main=True,
        )

    def require_volume_current(self) -> None:
        """Reprove retained event-5 paths while the main changes lifecycle."""

        self._require_owner()
        _require_selected_activation_lease_current(
            self,
            require_inert_main=False,
        )

    def _require_owner(self) -> None:
        if (
            self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or self._closed
        ):
            raise RunActionRuntimeVolumeError(
                "activation revalidation lease is closed or foreign"
            )

    def __enter__(self) -> "RunActionActivationRevalidationLease":
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        self._require_owner()
        self._closed = True
        self._descriptors.close()


@dataclass(frozen=True)
class DockerRunActionEmptyVolumeObservation:
    """Physical proof that the running keeper mounts one empty bounded tmpfs."""

    runtime_volume_authority: RunActionRuntimeVolumeAuthority
    docker_volume_observation: DockerRunActionVolumeObservation
    keeper_container_id: str
    keeper_process_id: int
    keeper_process_start_time_ticks: int
    process_cgroup_path: str
    mount_id: int
    device: int
    root_inode: int
    filesystem_type: str
    observed_mount_flags: tuple[str, ...]
    owner_user_id: int
    owner_group_id: int
    root_mode: int
    allocation_block_size_bytes: int
    effective_block_count: int
    effective_size_bytes: int
    effective_inode_limit: int
    used_block_count: int
    used_size_bytes: int
    used_inode_count: int
    available_block_count: int
    available_size_bytes: int
    available_inode_count: int
    empty_entry_count: int
    empty_size_bytes: int

    def __post_init__(self) -> None:
        cgroup_path = (
            PurePosixPath(self.process_cgroup_path)
            if type(self.process_cgroup_path) is str
            else None
        )
        if (
            type(self.runtime_volume_authority) is not RunActionRuntimeVolumeAuthority
            or type(self.docker_volume_observation)
            is not DockerRunActionVolumeObservation
            or self.docker_volume_observation.volume_authority_id
            != self.runtime_volume_authority.runtime_volume_authority_id
            or self.docker_volume_observation.volume_name
            != self.runtime_volume_authority.volume_name
            or type(self.keeper_container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(self.keeper_container_id) is None
            or type(self.keeper_process_id) is not int
            or self.keeper_process_id <= 0
            or type(self.keeper_process_start_time_ticks) is not int
            or self.keeper_process_start_time_ticks <= 0
            or cgroup_path is None
            or not self.process_cgroup_path.isascii()
            or "\x00" in self.process_cgroup_path
            or not cgroup_path.is_absolute()
            or cgroup_path.as_posix() != self.process_cgroup_path
            or ".." in cgroup_path.parts
            or not self.process_cgroup_path.endswith(
                f"/docker-{self.keeper_container_id}.scope"
            )
            or any(
                type(value) is not int or value <= 0
                for value in (self.mount_id, self.device, self.root_inode)
            )
            or self.filesystem_type != _TMPFS_FILESYSTEM_TYPE
            or self.observed_mount_flags != ("nodev", "nosuid", "noswap")
            or self.owner_user_id != self.runtime_volume_authority.owner_user_id
            or self.owner_group_id != self.runtime_volume_authority.owner_group_id
            or self.root_mode != self.runtime_volume_authority.root_mode
            or any(
                type(value) is not int
                for value in (
                    self.allocation_block_size_bytes,
                    self.effective_block_count,
                    self.effective_size_bytes,
                    self.effective_inode_limit,
                    self.used_block_count,
                    self.used_size_bytes,
                    self.used_inode_count,
                    self.available_block_count,
                    self.available_size_bytes,
                    self.available_inode_count,
                    self.empty_entry_count,
                    self.empty_size_bytes,
                )
            )
            or self.allocation_block_size_bytes <= 0
            or self.allocation_block_size_bytes & (self.allocation_block_size_bytes - 1)
            != 0
            or type(self.effective_block_count) is not int
            or self.effective_block_count <= 0
            or self.effective_size_bytes
            != self.effective_block_count * self.allocation_block_size_bytes
            or not 0
            < self.effective_size_bytes
            <= self.runtime_volume_authority.size_limit_bytes
            or not 0
            < self.effective_inode_limit
            <= self.runtime_volume_authority.inode_limit
            or self.used_block_count < 0
            or self.used_size_bytes < 0
            or self.used_inode_count < 0
            or self.used_block_count + self.available_block_count
            != self.effective_block_count
            or self.used_size_bytes
            != self.used_block_count * self.allocation_block_size_bytes
            or self.available_size_bytes
            != self.available_block_count * self.allocation_block_size_bytes
            or self.used_size_bytes + self.available_size_bytes
            != self.effective_size_bytes
            or self.used_inode_count + self.available_inode_count
            != self.effective_inode_limit
            or self.available_block_count <= 0
            or self.available_inode_count <= 0
            or self.empty_entry_count != 0
            or self.empty_size_bytes != 0
        ):
            raise RunActionRuntimeVolumeError(
                "empty runtime volume observation is incomplete or unsafe"
            )


@dataclass(frozen=True)
class _MountedRuntimeVolumeLease:
    process_descriptor: int
    root_descriptor: int
    keeper_container_id: str
    keeper_process_id: int
    process_start_time_ticks: int
    process_cgroup_path: str
    root_mount_id: int
    root_device: int
    root_inode: int


@dataclass(frozen=True)
class _PreparedLayoutObservation:
    root_mount_id: int
    root_device: int
    root_inode: int
    sentinel_metadata: os.stat_result
    sentinel_mount_id: int
    delivery_slot_observations: tuple["_ExactDirectoryObservation", ...]
    runtime_directory_observations: tuple["_ExactDirectoryObservation", ...]
    result_file_observation: "_ExactRegularFileObservation"
    workspace_directory_observation: "_ExactDirectoryObservation | None"
    workspace_frontier: RunWorkspaceFrontierIdentity | None
    filesystem: os.statvfs_result


@dataclass(frozen=True)
class _SelectedActivationDescriptorObservation:
    mounted_volume: _MountedRuntimeVolumeLease
    sentinel_observation: "_ExactRegularFileObservation"
    input_file_observation: "_ExactRegularFileObservation"
    result_file_observation: "_ExactRegularFileObservation"
    credential_file_observation: "_ExactRegularFileShapeObservation | None"
    directory_descriptors: tuple[
        tuple[
            RunActionPreparedDeliverySlot
            | RunActionPreparedRuntimeDirectory
            | RunActionPreparedWorkspaceProof,
            int,
            tuple[str, ...] | None,
        ],
        ...,
    ]
    activated_workspace_frontier: RunWorkspaceFrontierIdentity | None
    filesystem_identity: tuple[int, ...]
    root_metadata_identity: tuple[int, ...]
    mount_info: "_MountInfo"
    activated_volume: DockerRunActionActivatedVolumeObservation


@dataclass(frozen=True)
class _ExactRegularFileObservation:
    descriptor: int
    parent_descriptor: int
    name: str
    metadata: os.stat_result
    mount_id: int
    payload: bytes


@dataclass(frozen=True)
class _ExactRegularFileShapeObservation:
    descriptor: int
    parent_descriptor: int
    name: str
    metadata: os.stat_result
    mount_id: int


@dataclass(frozen=True)
class _ExactDirectoryObservation:
    metadata: os.stat_result
    mount_id: int


@dataclass(frozen=True)
class _PreparedDeliverySlotPlan:
    kind: RunActionPreparedFileKind
    directory_name: str
    final_file_name: str
    payload_size_limit_bytes: int


@dataclass(frozen=True)
class _PreparedResultFilePlan:
    kind: RunActionPreparedFileKind
    directory_name: str
    file_name: str
    relative_path: str
    payload_size_limit_bytes: int


@dataclass(frozen=True)
class _PreparedRuntimeDirectoryPlan:
    kind: RunActionPreparedRuntimeDirectoryKind
    directory_name: str
    observed_entry_count: int


@dataclass(frozen=True)
class _RuntimeVolumeLayoutPlan:
    directory_names: tuple[str, ...]
    delivery_slot_plans: tuple[_PreparedDeliverySlotPlan, ...]
    runtime_directory_plans: tuple[_PreparedRuntimeDirectoryPlan, ...]
    result_file_plan: _PreparedResultFilePlan
    workspace_copy_plan: RunWorkspaceCopyPlan | None
    preparation_size_bytes: int
    preparation_inode_count: int


@dataclass(frozen=True)
class _MountInfo:
    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_point: str
    mount_options: tuple[str, ...]
    optional_fields: tuple[str, ...]
    filesystem_type: str
    source: str
    super_options: tuple[str, ...]


def _open_mounted_runtime_volume(
    descriptors: ExitStack,
    keeper: RunActionVolumeKeeperEvidence,
) -> _MountedRuntimeVolumeLease:
    if (
        type(descriptors) is not ExitStack
        or type(keeper) is not RunActionVolumeKeeperEvidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease requires an exact keeper"
        )
    process_descriptor = os.open(
        f"/proc/{keeper.process_id}",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, process_descriptor)
    process_start_time_ticks = read_run_action_process_stat_from_descriptor(
        process_descriptor,
        keeper.process_id,
    ).start_time_ticks
    process_cgroup_path = read_run_action_process_cgroup_path_from_descriptor(
        process_descriptor,
        keeper.container_id,
    )
    process_root_descriptor = os.open(
        "root",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    descriptors.callback(os.close, process_root_descriptor)
    root_descriptor = os.open(
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION.removeprefix("/"),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_root_descriptor,
    )
    descriptors.callback(os.close, root_descriptor)
    root_metadata = os.fstat(root_descriptor)
    lease = _MountedRuntimeVolumeLease(
        process_descriptor=process_descriptor,
        root_descriptor=root_descriptor,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        process_start_time_ticks=process_start_time_ticks,
        process_cgroup_path=process_cgroup_path,
        root_mount_id=read_run_action_descriptor_mount_id(root_descriptor),
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
    )
    _require_same_mounted_runtime_volume(lease, keeper)
    return lease


def _require_same_mounted_runtime_volume(
    lease: _MountedRuntimeVolumeLease,
    keeper: RunActionVolumeKeeperEvidence,
) -> None:
    if (
        type(lease) is not _MountedRuntimeVolumeLease
        or type(keeper) is not RunActionVolumeKeeperEvidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease changed process or physical root"
        )
    authority = keeper.issued_create_projection.volume_authority
    process_stat_before = read_run_action_process_stat_from_descriptor(
        lease.process_descriptor,
        lease.keeper_process_id,
    )
    process_cgroup_before = read_run_action_process_cgroup_path_from_descriptor(
        lease.process_descriptor,
        lease.keeper_container_id,
    )
    retained_root_before = os.fstat(lease.root_descriptor)
    retained_mount_id_before = read_run_action_descriptor_mount_id(
        lease.root_descriptor
    )
    with ExitStack() as descriptors:
        process_root_descriptor = os.open(
            "root",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            dir_fd=lease.process_descriptor,
        )
        descriptors.callback(os.close, process_root_descriptor)
        current_root_descriptor = os.open(
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION.removeprefix("/"),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=process_root_descriptor,
        )
        descriptors.callback(os.close, current_root_descriptor)
        current_root_before = os.fstat(current_root_descriptor)
        current_mount_id_before = read_run_action_descriptor_mount_id(
            current_root_descriptor
        )
        mount_info_before = _read_mount_info(
            lease.process_descriptor,
            current_mount_id_before,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        _require_mount_authority(
            mount_info_before,
            current_root_before,
            authority,
        )
        current_root_after = os.fstat(current_root_descriptor)
        current_mount_id_after = read_run_action_descriptor_mount_id(
            current_root_descriptor
        )
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            current_mount_id_after,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
    retained_root_after = os.fstat(lease.root_descriptor)
    retained_mount_id_after = read_run_action_descriptor_mount_id(lease.root_descriptor)
    process_cgroup_after = read_run_action_process_cgroup_path_from_descriptor(
        lease.process_descriptor,
        lease.keeper_container_id,
    )
    process_stat_after = read_run_action_process_stat_from_descriptor(
        lease.process_descriptor,
        lease.keeper_process_id,
    )
    if (
        keeper.container_id != lease.keeper_container_id
        or keeper.process_id != lease.keeper_process_id
        or keeper.process_start_time_ticks != lease.process_start_time_ticks
        or process_stat_before.start_time_ticks != lease.process_start_time_ticks
        or process_stat_after.process_id != process_stat_before.process_id
        or process_stat_after.parent_process_id != process_stat_before.parent_process_id
        or process_stat_after.start_time_ticks != process_stat_before.start_time_ticks
        or process_cgroup_before != lease.process_cgroup_path
        or process_cgroup_after != process_cgroup_before
        or lease.process_cgroup_path
        != run_action_keeper_process_cgroup_path(
            keeper.issued_create_projection.execution_policy,
            keeper.container_id,
        )
        or retained_mount_id_before != lease.root_mount_id
        or retained_mount_id_after != retained_mount_id_before
        or current_mount_id_before != lease.root_mount_id
        or current_mount_id_after != current_mount_id_before
        or mount_info_after != mount_info_before
        or _stable_metadata(retained_root_before)
        != _stable_metadata(retained_root_after)
        or _stable_metadata(current_root_before) != _stable_metadata(current_root_after)
        or retained_root_before.st_dev != lease.root_device
        or retained_root_before.st_ino != lease.root_inode
        or current_root_before.st_dev != lease.root_device
        or current_root_before.st_ino != lease.root_inode
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease changed process or physical root"
        )


def open_run_action_control_directory(
    prepared: RunActionPreparedExecution,
) -> RunActionControlDirectoryLease:
    """Retain and classify one exact event-5 control directory."""

    if type(prepared) is not RunActionPreparedExecution:
        raise RunActionRuntimeVolumeError(
            "barrier control lease requires exact prepared execution"
        )
    keeper = prepared.volume_keeper_evidence
    prepared_volume = prepared.runtime_volume_evidence
    authority = prepared.runtime_volume_authority
    with ExitStack() as descriptors:
        mounted_volume = _open_mounted_runtime_volume(descriptors, keeper)
        if (
            mounted_volume.root_mount_id != prepared_volume.root_mount_id
            or mounted_volume.root_device != prepared_volume.root_device
            or mounted_volume.root_inode != prepared_volume.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "barrier control runtime volume was substituted"
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            mounted_volume.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=authority.generation_nonce.encode("ascii"),
            expected_mode=_SENTINEL_MODE,
            authority=authority,
            root_mount_id=mounted_volume.root_mount_id,
            root_device=mounted_volume.root_device,
        )
        _require_exact_sentinel_observation(
            sentinel_observation,
            prepared_volume.sentinel_evidence,
        )
        control_descriptor = _open_activation_subpath_directory(
            descriptors,
            mounted_volume,
            authority,
            prepared.control_directory.directory_relative_path,
            prepared.control_directory.mount_id,
            prepared.control_directory.device,
            prepared.control_directory.inode,
        )
        control_metadata_identity = _stable_metadata(os.fstat(control_descriptor))
        _require_exact_activation_directory(
            prepared.control_directory,
            mounted_volume.root_descriptor,
            control_descriptor,
            expected_entries=None,
        )
        entries_before = tuple(sorted(os.listdir(control_descriptor)))
        _require_exact_activation_directory(
            prepared.control_directory,
            mounted_volume.root_descriptor,
            control_descriptor,
            expected_entries=None,
        )
        entries_after = tuple(sorted(os.listdir(control_descriptor)))
        topologies_by_entries = {
            topology.entries: topology for topology in RunActionControlDirectoryTopology
        }
        if (
            entries_after != entries_before
            or entries_before not in topologies_by_entries
        ):
            raise RunActionRuntimeVolumeError(
                "barrier control directory has an invalid semantic topology"
            )
        topology = topologies_by_entries[entries_before]
        _require_exact_activation_directory(
            prepared.control_directory,
            mounted_volume.root_descriptor,
            control_descriptor,
            expected_entries=entries_before,
        )
        lease = RunActionControlDirectoryLease(
            descriptors=descriptors,
            mounted_volume=mounted_volume,
            prepared=prepared,
            sentinel_observation=sentinel_observation,
            control_descriptor=control_descriptor,
            control_metadata_identity=control_metadata_identity,
            topology=topology,
            _authority=_BARRIER_CONTROL_LEASE_AUTHORITY,
        )
        lease._descriptors = descriptors.pop_all()
    return lease


def open_run_action_result_workspace(
    prepared: RunActionPreparedExecution,
    result_capture_receipt: RunActionResultCaptureReceipt,
) -> RunActionResultWorkspaceLease:
    """Retain the exact event-3/event-6 runtime workspace by descriptor."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(result_capture_receipt) is not RunActionResultCaptureReceipt
        or prepared.workspace_proof is None
        or prepared.preparation_claim.reservation.intent.workspace_access
        is not RunFrontierWorkspaceAccess.EDIT_WORKSPACE
        or result_capture_receipt.runtime_volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
    ):
        raise RunActionRuntimeVolumeError(
            "result workspace lease requires exact edit execution evidence"
        )
    result_parent = prepared.result_directory
    result_file = prepared.result_file
    prepared_sentinel = prepared.runtime_volume_evidence.sentinel_evidence
    if (
        result_capture_receipt.prepared_parent_authority_id
        != result_parent.prepared_runtime_directory_id
        or result_capture_receipt.prepared_file_id != result_file.prepared_file_id
        or result_capture_receipt.parent_mount_id != result_parent.mount_id
        or result_capture_receipt.parent_device != result_parent.device
        or result_capture_receipt.parent_inode != result_parent.inode
        or result_capture_receipt.prepared_sentinel_evidence_id
        != prepared_sentinel.runtime_volume_sentinel_evidence_id
        or result_capture_receipt.generation_nonce != result_file.generation_nonce
        or result_capture_receipt.relative_path != result_file.relative_path
        or result_capture_receipt.file_type != result_file.file_type
        or result_capture_receipt.owner_user_id != result_file.owner_user_id
        or result_capture_receipt.owner_group_id != result_file.owner_group_id
        or result_capture_receipt.mode != result_file.mode
        or result_capture_receipt.link_count != result_file.link_count
        or result_capture_receipt.mount_id != result_file.mount_id
        or result_capture_receipt.device != result_file.device
        or result_capture_receipt.inode != result_file.inode
    ):
        raise RunActionRuntimeVolumeError(
            "result workspace lease differs from prepared result capture"
        )
    keeper = prepared.volume_keeper_evidence
    captured_volume = result_capture_receipt.reobserved_volume_evidence
    _require_result_volume_occurrence(prepared, captured_volume)
    with ExitStack() as descriptors:
        mounted_volume = _open_mounted_runtime_volume(descriptors, keeper)
        if (
            mounted_volume.root_mount_id != captured_volume.root_mount_id
            or mounted_volume.root_device != captured_volume.root_device
            or mounted_volume.root_inode != captured_volume.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "captured result workspace physical root was substituted"
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            mounted_volume.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=(
                prepared.runtime_volume_authority.generation_nonce.encode("ascii")
            ),
            expected_mode=_SENTINEL_MODE,
            authority=prepared.runtime_volume_authority,
            root_mount_id=mounted_volume.root_mount_id,
            root_device=mounted_volume.root_device,
        )
        _require_exact_sentinel_observation(
            sentinel_observation,
            captured_volume.sentinel_evidence,
        )
        workspace_descriptor = os.open(
            "workspace",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=mounted_volume.root_descriptor,
        )
        descriptors.callback(os.close, workspace_descriptor)
        workspace_metadata = os.fstat(workspace_descriptor)
        workspace_mount_id = read_run_action_descriptor_mount_id(workspace_descriptor)
        if not _result_workspace_matches_prepared(
            workspace_metadata,
            workspace_mount_id,
            prepared.workspace_proof,
        ):
            raise RunActionRuntimeVolumeError(
                "captured result workspace differs from prepared proof"
            )
        lease = RunActionResultWorkspaceLease(
            descriptors=descriptors,
            mounted_volume=mounted_volume,
            keeper=keeper,
            sentinel_observation=sentinel_observation,
            workspace_descriptor=workspace_descriptor,
            workspace_proof=prepared.workspace_proof,
            workspace_metadata_identity=_stable_metadata(workspace_metadata),
            _authority=_RESULT_WORKSPACE_LEASE_AUTHORITY,
        )
        lease._descriptors = descriptors.pop_all()
    return lease


def capture_run_action_result_file(
    prepared: RunActionPreparedExecution,
    terminal: RunActionTerminalObservation,
    volume: DockerRunActionVolumeObservation,
    *,
    settings: LaunchSettings,
) -> tuple[RunActionResultCaptureReceipt, bytes]:
    """Capture the original bounded result inode through its keeper-mounted root."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(terminal) is not RunActionTerminalObservation
        or type(volume) is not DockerRunActionVolumeObservation
        or type(settings) is not LaunchSettings
    ):
        raise RunActionRuntimeVolumeError(
            "result capture requires exact terminal and runtime authority"
        )
    authority = prepared.runtime_volume_authority
    result_parent = prepared.result_directory
    result_file = prepared.result_file
    policy_limit = (
        prepared.preparation_claim.execution_policy.supervisor_limits.result_size_bytes
    )
    if (
        result_file.payload_size_limit_bytes != policy_limit
        or policy_limit != settings.run_action_result_size_bytes
        or terminal.exit_code != 0
        or terminal.oom_killed is not False
        or terminal.prepared_execution_id != prepared.prepared_execution_id
        or terminal.provider_execution_id
        != prepared.inert_container_evidence.container_id
        or terminal.runtime_volume_authority_id != authority.runtime_volume_authority_id
        or terminal.generation_nonce != authority.generation_nonce
        or terminal.observed_inspect_projection
        != prepared.inert_container_evidence.issued_create_projection
        or volume.volume_authority_id != authority.runtime_volume_authority_id
        or volume.volume_name != authority.volume_name
    ):
        raise RunActionRuntimeVolumeError(
            "result capture differs from prepared terminal authority"
        )
    expected_root_entries = tuple(
        sorted((*_expected_directory_names(prepared.preparation_claim), _SENTINEL_NAME))
    )
    keeper = prepared.volume_keeper_evidence
    prepared_volume = prepared.runtime_volume_evidence
    with ExitStack() as descriptors:
        mounted_volume = _open_mounted_runtime_volume(descriptors, keeper)
        if (
            mounted_volume.root_mount_id != prepared_volume.root_mount_id
            or mounted_volume.root_device != prepared_volume.root_device
            or mounted_volume.root_inode != prepared_volume.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "result capture runtime volume was substituted"
            )
        root_metadata_before = os.fstat(mounted_volume.root_descriptor)
        if tuple(sorted(os.listdir(mounted_volume.root_descriptor))) != (
            expected_root_entries
        ):
            raise RunActionRuntimeVolumeError(
                "result capture runtime volume root topology is incomplete"
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            mounted_volume.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=authority.generation_nonce.encode("ascii"),
            expected_mode=_SENTINEL_MODE,
            authority=authority,
            root_mount_id=mounted_volume.root_mount_id,
            root_device=mounted_volume.root_device,
        )
        _require_exact_sentinel_observation(
            sentinel_observation,
            prepared_volume.sentinel_evidence,
        )
        result_parent_descriptor = _open_activation_subpath_directory(
            descriptors,
            mounted_volume,
            authority,
            result_parent.directory_relative_path,
            result_parent.mount_id,
            result_parent.device,
            result_parent.inode,
        )
        _require_exact_activation_directory(
            result_parent,
            mounted_volume.root_descriptor,
            result_parent_descriptor,
            expected_entries=("result.blob",),
        )
        result_descriptor = os.open(
            "result.blob",
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=result_parent_descriptor,
        )
        descriptors.callback(os.close, result_descriptor)
        result_metadata_before = os.fstat(result_descriptor)
        result_mount_id_before = read_run_action_descriptor_mount_id(result_descriptor)
        if (
            not stat.S_ISREG(result_metadata_before.st_mode)
            or result_metadata_before.st_uid != result_file.owner_user_id
            or result_metadata_before.st_gid != result_file.owner_group_id
            or stat.S_IMODE(result_metadata_before.st_mode) != result_file.mode
            or result_metadata_before.st_nlink != result_file.link_count
            or not 0 <= result_metadata_before.st_size <= policy_limit
            or result_mount_id_before != result_file.mount_id
            or result_metadata_before.st_dev != result_file.device
            or result_metadata_before.st_ino != result_file.inode
        ):
            raise RunActionRuntimeVolumeError(
                "result capture file is oversized or substituted"
            )
        os.fsync(result_descriptor)
        os.fsync(result_parent_descriptor)
        result_metadata_after_sync = os.fstat(result_descriptor)
        result_mount_id_after_sync = read_run_action_descriptor_mount_id(
            result_descriptor
        )
        filesystem_before = os.fstatvfs(mounted_volume.root_descriptor)
        _require_consistent_filesystem(filesystem_before)
        if (
            _stable_metadata(result_metadata_after_sync)
            != _stable_metadata(result_metadata_before)
            or result_mount_id_after_sync != result_mount_id_before
        ):
            raise RunActionRuntimeVolumeError(
                "result capture file changed while it was synchronized"
            )
        payload = _read_bounded_descriptor_payload(
            result_descriptor,
            policy_limit + 1,
        )
        if (
            len(payload) > policy_limit
            or len(payload) != result_metadata_before.st_size
        ):
            raise RunActionRuntimeVolumeError(
                "result capture payload is oversized or unstable"
            )
        captured_file = _ExactRegularFileObservation(
            descriptor=result_descriptor,
            parent_descriptor=result_parent_descriptor,
            name="result.blob",
            metadata=result_metadata_before,
            mount_id=result_mount_id_before,
            payload=payload,
        )
        _require_same_exact_regular_file(captured_file)
        _require_exact_activation_directory(
            result_parent,
            mounted_volume.root_descriptor,
            result_parent_descriptor,
            expected_entries=("result.blob",),
        )
        _require_same_exact_regular_file(sentinel_observation)
        _require_same_mounted_runtime_volume(mounted_volume, keeper)
        filesystem_after = os.fstatvfs(mounted_volume.root_descriptor)
        root_metadata_after = os.fstat(mounted_volume.root_descriptor)
        if (
            _stable_filesystem(filesystem_after)
            != _stable_filesystem(filesystem_before)
            or _root_metadata_identity(root_metadata_after)
            != _root_metadata_identity(root_metadata_before)
            or tuple(sorted(os.listdir(mounted_volume.root_descriptor)))
            != expected_root_entries
        ):
            raise RunActionRuntimeVolumeError(
                "result capture runtime volume changed during observation"
            )
        captured_volume = _mint_runtime_volume_evidence(
            authority,
            volume,
            keeper,
            root_mount_id=mounted_volume.root_mount_id,
            root_device=mounted_volume.root_device,
            root_inode=mounted_volume.root_inode,
            sentinel_evidence=prepared_volume.sentinel_evidence,
            filesystem=filesystem_after,
        )
        _require_result_volume_occurrence(prepared, captured_volume)
        receipt = RunActionResultCaptureReceipt.mint(
            terminal_observation_id=terminal.terminal_observation_id,
            prepared_parent_authority_id=(result_parent.prepared_runtime_directory_id),
            prepared_file_id=result_file.prepared_file_id,
            parent_mount_id=result_parent.mount_id,
            parent_device=result_parent.device,
            parent_inode=result_parent.inode,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            reobserved_volume_evidence=captured_volume,
            prepared_sentinel_evidence_id=(
                prepared_volume.sentinel_evidence.runtime_volume_sentinel_evidence_id
            ),
            generation_nonce=authority.generation_nonce,
            relative_path=result_file.relative_path,
            file_type=result_file.file_type,
            owner_user_id=result_metadata_before.st_uid,
            owner_group_id=result_metadata_before.st_gid,
            mode=stat.S_IMODE(result_metadata_before.st_mode),
            link_count=result_metadata_before.st_nlink,
            size_bytes=len(payload),
            content_digest=tree_or_blob_digest(payload),
            mount_id=result_mount_id_before,
            device=result_metadata_before.st_dev,
            inode=result_metadata_before.st_ino,
        )
        return receipt, payload


def _result_workspace_matches_prepared(
    metadata: os.stat_result,
    mount_id: int,
    prepared: RunActionPreparedWorkspaceProof,
) -> bool:
    return (
        type(metadata) is os.stat_result
        and type(mount_id) is int
        and type(prepared) is RunActionPreparedWorkspaceProof
        and stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == prepared.owner_user_id
        and metadata.st_gid == prepared.owner_group_id
        and stat.S_IMODE(metadata.st_mode) == prepared.root_mode
        and mount_id == prepared.mount_id
        and metadata.st_dev == prepared.device
        and metadata.st_ino == prepared.inode
    )


def _require_exact_sentinel_observation(
    observed: _ExactRegularFileObservation,
    expected: RunActionRuntimeVolumeSentinelEvidence,
) -> None:
    if (
        type(observed) is not _ExactRegularFileObservation
        or type(expected) is not RunActionRuntimeVolumeSentinelEvidence
    ):
        raise RunActionRuntimeVolumeError("runtime volume sentinel was substituted")
    metadata = observed.metadata
    if (
        expected.relative_path,
        expected.file_type,
        expected.owner_user_id,
        expected.owner_group_id,
        expected.mode,
        expected.link_count,
        expected.size_bytes,
        expected.content_digest,
        expected.mount_id,
        expected.device,
        expected.inode,
    ) != (
        observed.name,
        "regular",
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        tree_or_blob_digest(observed.payload),
        observed.mount_id,
        metadata.st_dev,
        metadata.st_ino,
    ):
        raise RunActionRuntimeVolumeError("runtime volume sentinel was substituted")


def _require_result_volume_occurrence(
    prepared: RunActionPreparedExecution,
    captured: RunActionRuntimeVolumeEvidence,
) -> None:
    if type(captured) is not RunActionRuntimeVolumeEvidence:
        raise RunActionRuntimeVolumeError(
            "captured result volume differs from its prepared occurrence"
        )
    original = prepared.runtime_volume_evidence
    keeper = prepared.volume_keeper_evidence
    immutable_original = (
        original.volume_authority,
        original.docker_volume_occurrence_digest,
        original.volume_keeper_evidence_id,
        original.keeper_container_id,
        original.keeper_process_id,
        original.keeper_process_start_time_ticks,
        original.keeper_process_cgroup_path,
        original.root_mount_id,
        original.root_device,
        original.root_inode,
        original.observed_volume_name,
        original.observed_labels,
        original.observed_scope,
        original.observed_driver,
        original.observed_driver_options,
        original.observed_filesystem_type,
        original.observed_mount_flags,
        original.observed_owner_user_id,
        original.observed_owner_group_id,
        original.observed_root_mode,
        original.allocation_block_size_bytes,
        original.effective_block_count,
        original.effective_size_bytes,
        original.effective_inode_limit,
        original.sentinel_evidence,
    )
    immutable_captured = (
        captured.volume_authority,
        captured.docker_volume_occurrence_digest,
        captured.volume_keeper_evidence_id,
        captured.keeper_container_id,
        captured.keeper_process_id,
        captured.keeper_process_start_time_ticks,
        captured.keeper_process_cgroup_path,
        captured.root_mount_id,
        captured.root_device,
        captured.root_inode,
        captured.observed_volume_name,
        captured.observed_labels,
        captured.observed_scope,
        captured.observed_driver,
        captured.observed_driver_options,
        captured.observed_filesystem_type,
        captured.observed_mount_flags,
        captured.observed_owner_user_id,
        captured.observed_owner_group_id,
        captured.observed_root_mode,
        captured.allocation_block_size_bytes,
        captured.effective_block_count,
        captured.effective_size_bytes,
        captured.effective_inode_limit,
        captured.sentinel_evidence,
    )
    if (
        immutable_captured != immutable_original
        or captured.volume_keeper_evidence_id != keeper.volume_keeper_evidence_id
        or captured.keeper_container_id != keeper.container_id
        or captured.keeper_process_id != keeper.process_id
        or captured.keeper_process_start_time_ticks != keeper.process_start_time_ticks
        or captured.keeper_process_cgroup_path
        != keeper.mounted_helper_evidence.process_cgroup_path
    ):
        raise RunActionRuntimeVolumeError(
            "captured result volume differs from its prepared occurrence"
        )


def observe_empty_runtime_volume(
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
) -> DockerRunActionEmptyVolumeObservation:
    """Prove the keeper's mounted generation is empty before first publication."""

    if (
        type(authority) is not RunActionRuntimeVolumeAuthority
        or type(volume) is not DockerRunActionVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or volume.volume_authority_id != authority.runtime_volume_authority_id
        or keeper.issued_create_projection.volume_authority != authority
    ):
        raise RunActionRuntimeVolumeError(
            "empty runtime volume observation requires exact Docker authority"
        )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        metadata_before = os.fstat(lease.root_descriptor)
        mount_info_before = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        filesystem_before = os.fstatvfs(lease.root_descriptor)
        entries = tuple(sorted(os.listdir(lease.root_descriptor)))
        filesystem_after = os.fstatvfs(lease.root_descriptor)
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        metadata_after = os.fstat(lease.root_descriptor)
        _require_same_mounted_runtime_volume(lease, keeper)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_info_after != mount_info_before
        or _stable_filesystem(filesystem_after) != _stable_filesystem(filesystem_before)
        or entries
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume changed during empty-generation observation"
        )
    _require_mount_authority(
        mount_info_before,
        metadata_before,
        authority,
    )
    filesystem = filesystem_before
    if (
        filesystem.f_bsize != filesystem.f_frsize
        or filesystem.f_bfree != filesystem.f_bavail
        or filesystem.f_ffree != filesystem.f_favail
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume statvfs carries reserved or inconsistent capacity"
        )
    allocation_block_size_bytes = filesystem.f_frsize
    effective_block_count = filesystem.f_blocks
    available_block_count = filesystem.f_bfree
    used_block_count = effective_block_count - available_block_count
    effective_inode_limit = filesystem.f_files
    available_inode_count = filesystem.f_ffree
    used_inode_count = effective_inode_limit - available_inode_count
    return DockerRunActionEmptyVolumeObservation(
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
        keeper_container_id=lease.keeper_container_id,
        keeper_process_id=lease.keeper_process_id,
        keeper_process_start_time_ticks=lease.process_start_time_ticks,
        process_cgroup_path=lease.process_cgroup_path,
        mount_id=lease.root_mount_id,
        device=metadata_before.st_dev,
        root_inode=metadata_before.st_ino,
        filesystem_type=mount_info_before.filesystem_type,
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        owner_user_id=metadata_before.st_uid,
        owner_group_id=metadata_before.st_gid,
        root_mode=stat.S_IMODE(metadata_before.st_mode),
        allocation_block_size_bytes=allocation_block_size_bytes,
        effective_block_count=effective_block_count,
        effective_size_bytes=effective_block_count * allocation_block_size_bytes,
        effective_inode_limit=effective_inode_limit,
        used_block_count=used_block_count,
        used_size_bytes=used_block_count * allocation_block_size_bytes,
        used_inode_count=used_inode_count,
        available_block_count=available_block_count,
        available_size_bytes=available_block_count * allocation_block_size_bytes,
        available_inode_count=available_inode_count,
        empty_entry_count=0,
        empty_size_bytes=0,
    )


def materialize_runtime_volume_layout(
    claim: RunActionPreparationClaim,
    empty_volume: DockerRunActionEmptyVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> DockerRunActionPreparedVolumeObservation:
    """Populate one proven-empty generation and publish its sentinel last."""

    if (
        type(claim) is not RunActionPreparationClaim
        or type(empty_volume) is not DockerRunActionEmptyVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
        or empty_volume.runtime_volume_authority.preparation_claim_id
        != claim.preparation_claim_id
        or empty_volume.docker_volume_observation.volume_authority_id
        != empty_volume.runtime_volume_authority.runtime_volume_authority_id
        or keeper.preparation_claim_id != claim.preparation_claim_id
        or keeper.issued_create_projection.execution_policy != claim.execution_policy
        or keeper.issued_create_projection.volume_authority
        != empty_volume.runtime_volume_authority
        or (claim.execution_policy.user_id, claim.execution_policy.group_id)
        != (os.geteuid(), os.getegid())
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume materialization requires exact writable authority"
        )
    plan = _plan_runtime_volume_layout(
        claim,
        empty_volume,
        workspace_descriptor=workspace_descriptor,
        settings=settings,
    )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        _require_empty_volume_lease(empty_volume, keeper, lease)
        workspace_frontier = _materialize_layout_at_descriptor(
            lease.root_descriptor,
            claim=claim,
            authority=empty_volume.runtime_volume_authority,
            plan=plan,
            workspace_descriptor=workspace_descriptor,
            settings=settings,
        )
        observed = _observe_prepared_layout_at_descriptor(
            lease,
            claim=claim,
            authority=empty_volume.runtime_volume_authority,
            keeper=keeper,
            workspace_frontier=workspace_frontier,
            settings=settings,
        )
        _require_same_mounted_runtime_volume(lease, keeper)
    return _mint_prepared_volume_observation(
        claim,
        empty_volume.runtime_volume_authority,
        empty_volume.docker_volume_observation,
        keeper,
        observed,
        empty_entry_count=empty_volume.empty_entry_count,
        empty_size_bytes=empty_volume.empty_size_bytes,
    )


def adopt_prepared_runtime_volume_layout(
    allocation: RunActionPreparationAllocation,
    resource_manager: DockerRunActionResourceManager,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    settings: LaunchSettings,
) -> DockerRunActionPreparedVolumeObservation:
    """Reconstruct event-3 layout evidence from one exact event-2 occurrence."""

    if (
        type(allocation) is not RunActionPreparationAllocation
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume adoption requires one exact durable allocation"
        )
    claim = allocation.preparation_claim
    authority = allocation.runtime_volume_authority
    if (
        keeper.preparation_claim_id != claim.preparation_claim_id
        or keeper.issued_create_projection.execution_policy != claim.execution_policy
        or keeper.issued_create_projection.volume_authority != authority
        or (claim.execution_policy.user_id, claim.execution_policy.group_id)
        != (os.geteuid(), os.getegid())
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume adoption differs from its durable allocation"
        )
    inventory = resource_manager.observe(allocation)
    if (
        not inventory.volume_present
        or inventory.keeper_container_id != keeper.container_id
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume adoption lacks its exact live Docker occurrence"
        )
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        claim,
        authority,
        resource_manager.runtime_settings,
    )
    workspace_binding = claim.reservation.frontier.workspace_before
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        observed = _observe_prepared_layout_at_descriptor(
            lease,
            claim=claim,
            authority=authority,
            keeper=keeper,
            workspace_frontier=(
                None if workspace_binding is None else workspace_binding.to_identity()
            ),
            settings=settings,
        )
        _require_same_mounted_runtime_volume(lease, keeper)
    current_inventory = resource_manager.observe(allocation)
    if current_inventory != inventory:
        raise RunActionRuntimeVolumeError(
            "runtime volume Docker occurrence changed during adoption"
        )
    current_volume = observe_runtime_volume(
        resource_manager.inspect_volume(current_inventory),
        claim,
        authority,
        resource_manager.runtime_settings,
    )
    if current_volume != volume:
        raise RunActionRuntimeVolumeError(
            "runtime volume Docker inspection changed during adoption"
        )
    return _mint_prepared_volume_observation(
        claim,
        authority,
        current_volume,
        keeper,
        observed,
        empty_entry_count=0,
        empty_size_bytes=0,
    )


def reobserve_runtime_volume_layout(
    prepared: RunActionPreparedExecution,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    settings: LaunchSettings,
) -> DockerRunActionPreparedVolumeObservation:
    """Reopen only one durably prepared generation without mutating it."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(volume) is not DockerRunActionVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
        or volume.volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
        or volume.volume_name != prepared.runtime_volume_authority.volume_name
        or keeper != prepared.volume_keeper_evidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume reopen requires exact durable prepared authority"
        )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        evidence = prepared.runtime_volume_evidence
        if (
            lease.root_mount_id != evidence.root_mount_id
            or lease.root_device != evidence.root_device
            or lease.root_inode != evidence.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume physical root was substituted"
            )
        observed = _observe_prepared_layout_at_descriptor(
            lease,
            claim=prepared.preparation_claim,
            authority=prepared.runtime_volume_authority,
            keeper=keeper,
            workspace_frontier=(
                None
                if prepared.workspace_proof is None
                else prepared.workspace_proof.workspace_binding.to_identity()
            ),
            settings=settings,
        )
        _require_same_mounted_runtime_volume(lease, keeper)
    reopened = _mint_prepared_volume_observation(
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        keeper,
        observed,
        empty_entry_count=prepared.layout_proof.empty_entry_count,
        empty_size_bytes=prepared.layout_proof.empty_size_bytes,
    )
    if reopened != DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        control_directory=prepared.control_directory,
        result_directory=prepared.result_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    ):
        raise RunActionRuntimeVolumeError(
            "reopened runtime volume differs from durable prepared layout"
        )
    return reopened


def open_selected_run_action_activation(
    allocation: RunActionPreparationAllocation,
    selected_receipt: RunActionActivationRevalidationReceipt,
    resource_manager: DockerRunActionResourceManager,
    *,
    settings: LaunchSettings,
) -> RunActionActivationRevalidationLease:
    """Reopen event 5 without original payload or workspace-source authority."""

    if (
        type(allocation) is not RunActionPreparationAllocation
        or type(selected_receipt) is not RunActionActivationRevalidationReceipt
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(settings) is not LaunchSettings
    ):
        raise RunActionRuntimeVolumeError(
            "activation reopen requires exact durable and Docker authorities"
        )
    prepared = selected_receipt.prepared_execution
    if (
        allocation.preparation_claim != prepared.preparation_claim
        or allocation.runtime_volume_authority != prepared.runtime_volume_authority
        or selected_receipt.reobserved_keeper_evidence
        != prepared.volume_keeper_evidence
        or selected_receipt.reobserved_container_evidence
        != prepared.inert_container_evidence
        or tree_or_blob_digest(resource_manager.runtime_settings.to_json_bytes())
        != prepared.preparation_claim.execution_policy.docker_runtime_settings_digest
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation differs from its durable allocation"
        )
    inventory = resource_manager.observe(allocation)
    volume, keeper, inert_main = _observe_selected_activation_docker_resources(
        allocation,
        selected_receipt,
        resource_manager,
        inventory,
    )
    with ExitStack() as descriptors:
        descriptor_observation = _open_selected_activation_descriptors(
            descriptors,
            selected_receipt,
            volume,
            keeper,
            settings=settings,
        )
        if descriptor_observation.activated_volume != (
            _selected_activated_volume(selected_receipt)
        ):
            raise RunActionRuntimeVolumeError(
                "reopened activation differs from its durable event-5 volume"
            )
        candidate = RunActionActivationRevalidationReceipt.mint(
            prepared_execution=prepared,
            spawn_commit=selected_receipt.spawn_commit,
            reobserved_volume_evidence=(
                descriptor_observation.activated_volume.reobserved_volume_evidence
            ),
            reobserved_keeper_evidence=keeper,
            reobserved_container_evidence=inert_main,
            activated_workspace_observation=(
                descriptor_observation.activated_volume.activated_workspace_observation
            ),
            activated_runtime_directory_observations=(
                descriptor_observation.activated_volume.activated_runtime_directory_observations
            ),
            activated_sentinel_observation=(
                descriptor_observation.activated_volume.activated_sentinel_observation
            ),
            input_file_observation=(
                descriptor_observation.activated_volume.input_file_observation
            ),
            result_file_observation=(
                descriptor_observation.activated_volume.result_file_observation
            ),
            credential_file_observation=(
                descriptor_observation.activated_volume.credential_file_observation
            ),
        )
        if candidate != selected_receipt:
            raise RunActionRuntimeVolumeError(
                "reopened activation does not reproduce selected event 5"
            )
        current_inventory = resource_manager.observe(allocation)
        if current_inventory != inventory:
            raise RunActionRuntimeVolumeError(
                "activation Docker occurrence changed during reopen"
            )
        current_volume, current_keeper, current_main = (
            _observe_selected_activation_docker_resources(
                allocation,
                selected_receipt,
                resource_manager,
                current_inventory,
            )
        )
        if (
            current_volume != volume
            or current_keeper != keeper
            or current_main != inert_main
        ):
            raise RunActionRuntimeVolumeError(
                "activation Docker evidence changed during reopen"
            )
        lease = RunActionActivationRevalidationLease(
            descriptors=descriptors,
            preparation_allocation=allocation,
            selected_receipt=selected_receipt,
            resource_manager=resource_manager,
            inventory=current_inventory,
            volume_observation=current_volume,
            keeper_evidence=current_keeper,
            inert_container_evidence=current_main,
            mounted_volume=descriptor_observation.mounted_volume,
            sentinel_observation=descriptor_observation.sentinel_observation,
            input_file_observation=(descriptor_observation.input_file_observation),
            result_file_observation=(descriptor_observation.result_file_observation),
            credential_file_observation=(
                descriptor_observation.credential_file_observation
            ),
            directory_descriptors=(descriptor_observation.directory_descriptors),
            activated_workspace_frontier=(
                descriptor_observation.activated_workspace_frontier
            ),
            filesystem_identity=descriptor_observation.filesystem_identity,
            root_metadata_identity=(descriptor_observation.root_metadata_identity),
            mount_info=descriptor_observation.mount_info,
            settings=settings,
            _authority=_ACTIVATION_REVALIDATION_LEASE_AUTHORITY,
        )
        descriptors.pop_all()
        return lease


def deliver_and_reobserve_runtime_volume_activation(
    prepared: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    request_payload: bytes,
    credential_payload: bytes | None,
    credential_content_authority_id: str | None,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> DockerRunActionActivatedVolumeObservation:
    """Atomically deliver payloads and prove every pre-start volume subpath."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
        or type(volume) is not DockerRunActionVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
        or type(request_payload) is not bytes
        or not request_payload
        or len(request_payload)
        != prepared.preparation_claim.reservation.request_blob.size_bytes
        or tree_or_blob_digest(request_payload)
        != prepared.preparation_claim.reservation.request_blob.digest
        or volume.volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
        or volume.volume_name != prepared.runtime_volume_authority.volume_name
        or keeper != prepared.volume_keeper_evidence
        or (prepared.workspace_proof is None) != (workspace_descriptor is None)
        or (
            workspace_descriptor is not None
            and (type(workspace_descriptor) is not int or workspace_descriptor < 0)
        )
    ):
        raise RunActionRuntimeVolumeError(
            "activation delivery requires exact prepared payload and volume authority"
        )
    _require_activation_spawn_join(prepared, spawn_commit)
    credential_required = (
        prepared.preparation_claim.execution_policy.credential_policy.mode
        is RunActionCredentialMode.SUPERVISOR_FILE
    )
    if credential_required:
        if (
            type(credential_payload) is not bytes
            or not credential_payload
            or prepared.credential_delivery_slot is None
            or len(credential_payload)
            > prepared.credential_delivery_slot.payload_size_limit_bytes
            or type(credential_content_authority_id) is not str
        ):
            raise RunActionRuntimeVolumeError(
                "credentialed activation lacks one bounded broker delivery"
            )
        if (
            not is_content_id(credential_content_authority_id)
            or credential_content_authority_id.split(":sha256:", 1)[0]
            != RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE
        ):
            raise RunActionRuntimeVolumeError(
                "credentialed activation authority is not a fixed lease content ID"
            )
    elif (
        credential_payload is not None
        or credential_content_authority_id is not None
        or prepared.credential_delivery_slot is not None
    ):
        raise RunActionRuntimeVolumeError(
            "credential-free activation carries credential delivery authority"
        )
    _require_activation_workspace_source(
        prepared,
        workspace_descriptor,
        settings,
    )
    authority = prepared.runtime_volume_authority
    expected_root_entries = tuple(
        sorted((*_expected_directory_names(prepared.preparation_claim), _SENTINEL_NAME))
    )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        prepared_volume = prepared.runtime_volume_evidence
        if (
            lease.root_mount_id != prepared_volume.root_mount_id
            or lease.root_device != prepared_volume.root_device
            or lease.root_inode != prepared_volume.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "activation runtime volume physical root was substituted"
            )
        root_metadata_before = os.fstat(lease.root_descriptor)
        mount_info_before = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        _require_mount_authority(mount_info_before, root_metadata_before, authority)
        if tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries:
            raise RunActionRuntimeVolumeError(
                "activation runtime volume root topology is incomplete"
            )
        input_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.input_delivery_slot.directory_relative_path,
            prepared.input_delivery_slot.mount_id,
            prepared.input_delivery_slot.device,
            prepared.input_delivery_slot.inode,
        )
        control_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.control_directory.directory_relative_path,
            prepared.control_directory.mount_id,
            prepared.control_directory.device,
            prepared.control_directory.inode,
        )
        result_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.result_directory.directory_relative_path,
            prepared.result_directory.mount_id,
            prepared.result_directory.device,
            prepared.result_directory.inode,
        )
        temporary_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.temporary_directory.directory_relative_path,
            prepared.temporary_directory.mount_id,
            prepared.temporary_directory.device,
            prepared.temporary_directory.inode,
        )
        credential_directory_descriptor = None
        if prepared.credential_delivery_slot is not None:
            credential_directory_descriptor = _open_activation_subpath_directory(
                descriptors,
                lease,
                authority,
                prepared.credential_delivery_slot.directory_relative_path,
                prepared.credential_delivery_slot.mount_id,
                prepared.credential_delivery_slot.device,
                prepared.credential_delivery_slot.inode,
            )
        workspace_directory_descriptor = None
        activated_workspace_frontier = None
        if prepared.workspace_proof is not None:
            workspace_directory_descriptor = _open_activation_subpath_directory(
                descriptors,
                lease,
                authority,
                prepared.workspace_proof.volume_subpath,
                prepared.workspace_proof.mount_id,
                prepared.workspace_proof.device,
                prepared.workspace_proof.inode,
            )
            activated_workspace_frontier = _observe_activation_workspace(
                workspace_directory_descriptor,
                prepared,
                settings,
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            lease.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=authority.generation_nonce.encode("ascii"),
            expected_mode=_SENTINEL_MODE,
            authority=authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
        _require_exact_sentinel_observation(
            sentinel_observation,
            prepared_volume.sentinel_evidence,
        )
        result_file_observation = _open_exact_regular_file(
            descriptors,
            result_directory_descriptor,
            "result.blob",
            expected_payload=b"",
            expected_mode=_PREPARED_FILE_MODE,
            authority=authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
        _require_activation_result_file(
            result_file_observation,
            prepared.result_file,
        )
        _require_exact_activation_directory(
            prepared.control_directory,
            lease.root_descriptor,
            control_directory_descriptor,
            expected_entries=(),
        )
        _require_exact_activation_directory(
            prepared.result_directory,
            lease.root_descriptor,
            result_directory_descriptor,
            expected_entries=("result.blob",),
        )
        _require_exact_activation_directory(
            prepared.temporary_directory,
            lease.root_descriptor,
            temporary_directory_descriptor,
            expected_entries=(),
        )
        input_delivery = descriptors.enter_context(
            publish_or_adopt_run_action_delivery(
                prepared.input_delivery_slot,
                input_directory_descriptor,
                request_payload,
            )
        )
        credential_delivery = None
        if prepared.credential_delivery_slot is not None:
            if (
                type(credential_directory_descriptor) is not int
                or type(credential_payload) is not bytes
            ):
                raise RunActionRuntimeVolumeError(
                    "credential delivery lost its exact directory or payload"
                )
            credential_delivery = descriptors.enter_context(
                publish_or_adopt_run_action_delivery(
                    prepared.credential_delivery_slot,
                    credential_directory_descriptor,
                    credential_payload,
                )
            )
        filesystem_before = os.fstatvfs(lease.root_descriptor)
        _require_consistent_filesystem(filesystem_before)
        _require_same_mounted_runtime_volume(lease, keeper)
        _require_same_exact_regular_file(sentinel_observation)
        _require_same_exact_regular_file(result_file_observation)
        _require_exact_activation_directory(
            prepared.control_directory,
            lease.root_descriptor,
            control_directory_descriptor,
            expected_entries=(),
        )
        _require_exact_activation_directory(
            prepared.input_delivery_slot,
            lease.root_descriptor,
            input_directory_descriptor,
            expected_entries=(prepared.input_delivery_slot.final_file_name,),
        )
        _require_exact_activation_directory(
            prepared.result_directory,
            lease.root_descriptor,
            result_directory_descriptor,
            expected_entries=("result.blob",),
        )
        _require_exact_activation_directory(
            prepared.temporary_directory,
            lease.root_descriptor,
            temporary_directory_descriptor,
            expected_entries=(),
        )
        if prepared.credential_delivery_slot is not None:
            if type(credential_directory_descriptor) is not int:
                raise RunActionRuntimeVolumeError(
                    "credential activation lost its directory descriptor"
                )
            _require_exact_activation_directory(
                prepared.credential_delivery_slot,
                lease.root_descriptor,
                credential_directory_descriptor,
                expected_entries=(prepared.credential_delivery_slot.final_file_name,),
            )
        if prepared.workspace_proof is not None:
            if type(workspace_directory_descriptor) is not int:
                raise RunActionRuntimeVolumeError(
                    "workspace activation lost its directory descriptor"
                )
            reobserved_workspace_frontier = _observe_activation_workspace(
                workspace_directory_descriptor,
                prepared,
                settings,
            )
            _require_exact_activation_directory(
                prepared.workspace_proof,
                lease.root_descriptor,
                workspace_directory_descriptor,
                expected_entries=None,
            )
            if not _same_workspace_semantics(
                reobserved_workspace_frontier,
                activated_workspace_frontier,
            ):
                raise RunActionRuntimeVolumeError(
                    "activation workspace changed during final observation"
                )
        _require_activation_workspace_source(
            prepared,
            workspace_descriptor,
            settings,
        )
        filesystem_after = os.fstatvfs(lease.root_descriptor)
        root_metadata_after = os.fstat(lease.root_descriptor)
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        if (
            _stable_filesystem(filesystem_after)
            != _stable_filesystem(filesystem_before)
            or _root_metadata_identity(root_metadata_after)
            != _root_metadata_identity(root_metadata_before)
            or mount_info_after != mount_info_before
            or tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries
        ):
            raise RunActionRuntimeVolumeError(
                "activated runtime volume changed during final observation"
            )
        reobserved_volume_evidence = _mint_runtime_volume_evidence(
            authority,
            volume,
            keeper,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
            root_inode=lease.root_inode,
            sentinel_evidence=prepared_volume.sentinel_evidence,
            filesystem=filesystem_after,
        )
        activated_volume = _mint_activated_volume_observation(
            prepared,
            spawn_commit,
            reobserved_volume_evidence,
            activated_workspace_frontier,
            input_delivery.observation,
            (None if credential_delivery is None else credential_delivery.observation),
            credential_content_authority_id,
        )
        input_delivery.require_final_path(request_payload)
        if credential_delivery is not None:
            if type(credential_payload) is not bytes:
                raise RunActionRuntimeVolumeError(
                    "credential delivery lost its payload before final validation"
                )
            credential_delivery.require_final_path(credential_payload)
        return activated_volume


def _observe_selected_activation_docker_resources(
    allocation: RunActionPreparationAllocation,
    selected_receipt: RunActionActivationRevalidationReceipt,
    resource_manager: DockerRunActionResourceManager,
    inventory: DockerRunActionResourceInventory,
) -> tuple[
    DockerRunActionVolumeObservation,
    RunActionVolumeKeeperEvidence,
    RunActionInertContainerEvidence,
]:
    prepared = selected_receipt.prepared_execution
    if (
        inventory.preparation_allocation != allocation
        or not inventory.volume_present
        or inventory.keeper_container_id
        != selected_receipt.reobserved_keeper_evidence.container_id
        or inventory.main_container_id
        != selected_receipt.reobserved_container_evidence.container_id
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation lacks its exact three-resource occurrence"
        )
    volume, keeper = _observe_selected_activation_volume_and_keeper(
        allocation,
        selected_receipt,
        resource_manager,
        inventory,
    )
    projection = prepared.inert_container_evidence.issued_create_projection
    command = target_command_from_main_projection(projection)
    inert_main = observe_inert_main_container(
        resource_manager.inspect_main(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        command,
        projection.supervisor_helper_evidence,
        projection.docker_init_source_evidence,
        resource_manager.runtime_settings,
    )
    if inert_main != selected_receipt.reobserved_container_evidence:
        raise RunActionRuntimeVolumeError(
            "selected activation main differs from event 5"
        )
    return volume, keeper, inert_main


def _observe_selected_activation_volume_and_keeper(
    allocation: RunActionPreparationAllocation,
    selected_receipt: RunActionActivationRevalidationReceipt,
    resource_manager: DockerRunActionResourceManager,
    inventory: DockerRunActionResourceInventory,
) -> tuple[DockerRunActionVolumeObservation, RunActionVolumeKeeperEvidence]:
    if (
        inventory.preparation_allocation != allocation
        or not inventory.volume_present
        or inventory.keeper_container_id
        != selected_receipt.reobserved_keeper_evidence.container_id
        or inventory.main_container_id
        != selected_receipt.reobserved_container_evidence.container_id
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation resources differ from event 5"
        )
    prepared = selected_receipt.prepared_execution
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        resource_manager.runtime_settings,
    )
    keeper_projection = prepared.volume_keeper_evidence.issued_create_projection
    keeper = observe_running_keeper(
        resource_manager.inspect_keeper(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        keeper_projection.helper_evidence,
        keeper_projection.docker_init_source_evidence,
        resource_manager.runtime_settings,
    )
    if (
        volume.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
        or keeper != selected_receipt.reobserved_keeper_evidence
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation volume or keeper differs from event 5"
        )
    return volume, keeper


def _open_selected_activation_descriptors(
    descriptors: ExitStack,
    selected_receipt: RunActionActivationRevalidationReceipt,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    settings: LaunchSettings,
) -> _SelectedActivationDescriptorObservation:
    prepared = selected_receipt.prepared_execution
    authority = prepared.runtime_volume_authority
    lease = _open_mounted_runtime_volume(descriptors, keeper)
    prepared_volume = prepared.runtime_volume_evidence
    if (
        lease.root_mount_id != prepared_volume.root_mount_id
        or lease.root_device != prepared_volume.root_device
        or lease.root_inode != prepared_volume.root_inode
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation runtime root was substituted"
        )
    root_metadata_before = os.fstat(lease.root_descriptor)
    mount_info_before = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    _require_mount_authority(mount_info_before, root_metadata_before, authority)
    expected_root_entries = tuple(
        sorted((*_expected_directory_names(prepared.preparation_claim), _SENTINEL_NAME))
    )
    if tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries:
        raise RunActionRuntimeVolumeError(
            "selected activation root topology is incomplete"
        )
    input_directory_descriptor = _open_activation_subpath_directory(
        descriptors,
        lease,
        authority,
        prepared.input_delivery_slot.directory_relative_path,
        prepared.input_delivery_slot.mount_id,
        prepared.input_delivery_slot.device,
        prepared.input_delivery_slot.inode,
    )
    control_directory_descriptor = _open_activation_subpath_directory(
        descriptors,
        lease,
        authority,
        prepared.control_directory.directory_relative_path,
        prepared.control_directory.mount_id,
        prepared.control_directory.device,
        prepared.control_directory.inode,
    )
    result_directory_descriptor = _open_activation_subpath_directory(
        descriptors,
        lease,
        authority,
        prepared.result_directory.directory_relative_path,
        prepared.result_directory.mount_id,
        prepared.result_directory.device,
        prepared.result_directory.inode,
    )
    temporary_directory_descriptor = _open_activation_subpath_directory(
        descriptors,
        lease,
        authority,
        prepared.temporary_directory.directory_relative_path,
        prepared.temporary_directory.mount_id,
        prepared.temporary_directory.device,
        prepared.temporary_directory.inode,
    )
    credential_directory_descriptor = None
    if prepared.credential_delivery_slot is not None:
        credential_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.credential_delivery_slot.directory_relative_path,
            prepared.credential_delivery_slot.mount_id,
            prepared.credential_delivery_slot.device,
            prepared.credential_delivery_slot.inode,
        )
    workspace_directory_descriptor = None
    activated_workspace_frontier = None
    if prepared.workspace_proof is not None:
        workspace_directory_descriptor = _open_activation_subpath_directory(
            descriptors,
            lease,
            authority,
            prepared.workspace_proof.volume_subpath,
            prepared.workspace_proof.mount_id,
            prepared.workspace_proof.device,
            prepared.workspace_proof.inode,
        )
        activated_workspace_frontier = _observe_activation_workspace(
            workspace_directory_descriptor,
            prepared,
            settings,
        )
    sentinel_observation = _open_exact_regular_file(
        descriptors,
        lease.root_descriptor,
        _SENTINEL_NAME,
        expected_payload=authority.generation_nonce.encode("ascii"),
        expected_mode=_SENTINEL_MODE,
        authority=authority,
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
    )
    _require_exact_sentinel_observation(
        sentinel_observation,
        prepared_volume.sentinel_evidence,
    )
    input_file_observation = _open_reobserved_input_file(
        descriptors,
        input_directory_descriptor,
        prepared.input_delivery_slot,
        selected_receipt.input_file_observation,
        authority,
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
    )
    result_file_observation = _open_exact_regular_file(
        descriptors,
        result_directory_descriptor,
        "result.blob",
        expected_payload=b"",
        expected_mode=_PREPARED_FILE_MODE,
        authority=authority,
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
    )
    _require_activation_result_file(
        result_file_observation,
        prepared.result_file,
    )
    credential_file_observation = None
    if prepared.credential_delivery_slot is not None:
        if (
            type(credential_directory_descriptor) is not int
            or selected_receipt.credential_file_observation is None
        ):
            raise RunActionRuntimeVolumeError(
                "selected activation lost its credential authority"
            )
        credential_file_observation = _open_reobserved_credential_file(
            descriptors,
            credential_directory_descriptor,
            prepared.credential_delivery_slot,
            selected_receipt.credential_file_observation,
            authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
    elif selected_receipt.credential_file_observation is not None:
        raise RunActionRuntimeVolumeError(
            "credential-free activation carries a credential observation"
        )
    directory_descriptors = (
        (
            prepared.input_delivery_slot,
            input_directory_descriptor,
            (prepared.input_delivery_slot.final_file_name,),
        ),
        (prepared.control_directory, control_directory_descriptor, ()),
        (
            prepared.result_directory,
            result_directory_descriptor,
            ("result.blob",),
        ),
        (prepared.temporary_directory, temporary_directory_descriptor, ()),
        *(
            ()
            if prepared.credential_delivery_slot is None
            else (
                (
                    prepared.credential_delivery_slot,
                    credential_directory_descriptor,
                    (prepared.credential_delivery_slot.final_file_name,),
                ),
            )
        ),
        *(
            ()
            if prepared.workspace_proof is None
            else ((prepared.workspace_proof, workspace_directory_descriptor, None),)
        ),
    )
    for prepared_directory, descriptor, expected_entries in directory_descriptors:
        if type(descriptor) is not int:
            raise RunActionRuntimeVolumeError(
                "selected activation lost one retained directory"
            )
        _require_exact_activation_directory(
            prepared_directory,
            lease.root_descriptor,
            descriptor,
            expected_entries=expected_entries,
        )
    filesystem_before = os.fstatvfs(lease.root_descriptor)
    _require_consistent_filesystem(filesystem_before)
    _require_same_mounted_runtime_volume(lease, keeper)
    _require_same_exact_regular_file(sentinel_observation)
    _require_same_exact_regular_file(input_file_observation)
    _require_same_exact_regular_file(result_file_observation)
    if credential_file_observation is not None:
        _require_same_exact_regular_file_shape(credential_file_observation)
    if prepared.workspace_proof is not None:
        if type(workspace_directory_descriptor) is not int:
            raise RunActionRuntimeVolumeError(
                "selected activation lost its workspace descriptor"
            )
        reobserved_workspace = _observe_activation_workspace(
            workspace_directory_descriptor,
            prepared,
            settings,
        )
        if not _same_workspace_semantics(
            reobserved_workspace,
            activated_workspace_frontier,
        ):
            raise RunActionRuntimeVolumeError(
                "selected activation workspace changed during reopen"
            )
    filesystem_after = os.fstatvfs(lease.root_descriptor)
    root_metadata_after = os.fstat(lease.root_descriptor)
    mount_info_after = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    if (
        _stable_filesystem(filesystem_after) != _stable_filesystem(filesystem_before)
        or _root_metadata_identity(root_metadata_after)
        != _root_metadata_identity(root_metadata_before)
        or mount_info_after != mount_info_before
        or tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation changed during descriptor reopen"
        )
    reobserved_volume_evidence = _mint_runtime_volume_evidence(
        authority,
        volume,
        keeper,
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
        root_inode=lease.root_inode,
        sentinel_evidence=prepared_volume.sentinel_evidence,
        filesystem=filesystem_after,
    )
    input_delivery = _physical_reobserved_delivery(
        prepared.input_delivery_slot,
        input_file_observation,
        content_digest=tree_or_blob_digest(input_file_observation.payload),
    )
    credential_delivery = (
        None
        if prepared.credential_delivery_slot is None
        else _physical_reobserved_delivery(
            prepared.credential_delivery_slot,
            credential_file_observation,
            content_digest=None,
        )
    )
    activated_volume = _mint_activated_volume_observation(
        prepared,
        selected_receipt.spawn_commit,
        reobserved_volume_evidence,
        activated_workspace_frontier,
        input_delivery,
        credential_delivery,
        (
            None
            if selected_receipt.credential_file_observation is None
            else selected_receipt.credential_file_observation.content_authority_id
        ),
    )
    return _SelectedActivationDescriptorObservation(
        mounted_volume=lease,
        sentinel_observation=sentinel_observation,
        input_file_observation=input_file_observation,
        result_file_observation=result_file_observation,
        credential_file_observation=credential_file_observation,
        directory_descriptors=directory_descriptors,
        activated_workspace_frontier=activated_workspace_frontier,
        filesystem_identity=_stable_filesystem(filesystem_after),
        root_metadata_identity=_root_metadata_identity(root_metadata_after),
        mount_info=mount_info_after,
        activated_volume=activated_volume,
    )


def _selected_activated_volume(
    receipt: RunActionActivationRevalidationReceipt,
) -> DockerRunActionActivatedVolumeObservation:
    return DockerRunActionActivatedVolumeObservation(
        prepared_execution=receipt.prepared_execution,
        spawn_commit=receipt.spawn_commit,
        reobserved_volume_evidence=receipt.reobserved_volume_evidence,
        activated_workspace_observation=receipt.activated_workspace_observation,
        activated_runtime_directory_observations=(
            receipt.activated_runtime_directory_observations
        ),
        activated_sentinel_observation=receipt.activated_sentinel_observation,
        input_file_observation=receipt.input_file_observation,
        result_file_observation=receipt.result_file_observation,
        credential_file_observation=receipt.credential_file_observation,
    )


def _require_selected_activation_lease_current(
    lease: RunActionActivationRevalidationLease,
    *,
    require_inert_main: bool,
) -> None:
    inventory = lease._resource_manager.observe(lease._preparation_allocation)
    if inventory != lease._inventory:
        raise RunActionRuntimeVolumeError(
            "selected activation Docker occurrence changed"
        )
    if require_inert_main:
        volume, keeper, inert_main = _observe_selected_activation_docker_resources(
            lease._preparation_allocation,
            lease._selected_receipt,
            lease._resource_manager,
            inventory,
        )
        if inert_main != lease._inert_container_evidence:
            raise RunActionRuntimeVolumeError("selected activation inert main changed")
    else:
        volume, keeper = _observe_selected_activation_volume_and_keeper(
            lease._preparation_allocation,
            lease._selected_receipt,
            lease._resource_manager,
            inventory,
        )
    if volume != lease._volume_observation or keeper != lease._keeper_evidence:
        raise RunActionRuntimeVolumeError("selected activation Docker evidence changed")
    _require_same_mounted_runtime_volume(
        lease._mounted_volume,
        lease._keeper_evidence,
    )
    _require_same_exact_regular_file(lease._sentinel_observation)
    _require_same_exact_regular_file(lease._input_file_observation)
    _require_same_exact_regular_file(lease._result_file_observation)
    if lease._credential_file_observation is not None:
        _require_same_exact_regular_file_shape(lease._credential_file_observation)
    prepared = lease._selected_receipt.prepared_execution
    for (
        prepared_directory,
        descriptor,
        expected_entries,
    ) in lease._directory_descriptors:
        _require_exact_activation_directory(
            prepared_directory,
            lease._mounted_volume.root_descriptor,
            descriptor,
            expected_entries=expected_entries,
        )
    if prepared.workspace_proof is not None:
        workspace_descriptors = tuple(
            descriptor
            for prepared_directory, descriptor, _entries in lease._directory_descriptors
            if type(prepared_directory) is RunActionPreparedWorkspaceProof
        )
        if len(workspace_descriptors) != 1:
            raise RunActionRuntimeVolumeError(
                "selected activation lost its retained workspace"
            )
        workspace = _observe_activation_workspace(
            workspace_descriptors[0],
            prepared,
            lease._settings,
        )
        if not _same_workspace_semantics(
            workspace,
            lease._activated_workspace_frontier,
        ):
            raise RunActionRuntimeVolumeError(
                "selected activation retained workspace changed"
            )
    filesystem = os.fstatvfs(lease._mounted_volume.root_descriptor)
    root_metadata = os.fstat(lease._mounted_volume.root_descriptor)
    mount_info = _read_mount_info(
        lease._mounted_volume.process_descriptor,
        lease._mounted_volume.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    expected_root_entries = tuple(
        sorted((*_expected_directory_names(prepared.preparation_claim), _SENTINEL_NAME))
    )
    if (
        _stable_filesystem(filesystem) != lease._filesystem_identity
        or _root_metadata_identity(root_metadata) != lease._root_metadata_identity
        or mount_info != lease._mount_info
        or tuple(sorted(os.listdir(lease._mounted_volume.root_descriptor)))
        != expected_root_entries
    ):
        raise RunActionRuntimeVolumeError("selected activation retained volume changed")
    current_inventory = lease._resource_manager.observe(lease._preparation_allocation)
    if current_inventory != inventory:
        raise RunActionRuntimeVolumeError(
            "selected activation Docker occurrence changed during revalidation"
        )
    if require_inert_main:
        current_volume, current_keeper, current_main = (
            _observe_selected_activation_docker_resources(
                lease._preparation_allocation,
                lease._selected_receipt,
                lease._resource_manager,
                current_inventory,
            )
        )
        if current_main != lease._inert_container_evidence:
            raise RunActionRuntimeVolumeError(
                "selected activation inert main changed during revalidation"
            )
    else:
        current_volume, current_keeper = _observe_selected_activation_volume_and_keeper(
            lease._preparation_allocation,
            lease._selected_receipt,
            lease._resource_manager,
            current_inventory,
        )
    if (
        current_volume != lease._volume_observation
        or current_keeper != lease._keeper_evidence
    ):
        raise RunActionRuntimeVolumeError(
            "selected activation Docker evidence changed during revalidation"
        )


def _require_activation_spawn_join(
    prepared: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
) -> None:
    reservation = prepared.preparation_claim.reservation
    if (
        spawn_commit.reservation_id != reservation.reservation_id
        or spawn_commit.prepared_execution_id != prepared.prepared_execution_id
        or spawn_commit.boundary_identity != reservation.intent.boundary_identity
        or spawn_commit.security_observation_id
        != reservation.frontier.security_observation_id
        or spawn_commit.provider_execution_id
        != prepared.inert_container_evidence.container_id
    ):
        raise RunActionRuntimeVolumeError(
            "activation spawn differs from the prepared reservation and provider"
        )


def _mint_activated_volume_observation(
    prepared: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence,
    activated_workspace_frontier: RunWorkspaceFrontierIdentity | None,
    input_delivery: RunActionDeliveredFilePhysicalObservation,
    credential_delivery: RunActionDeliveredFilePhysicalObservation | None,
    credential_content_authority_id: str | None,
) -> DockerRunActionActivatedVolumeObservation:
    input_file_observation = _mint_activated_delivery_file(
        spawn_commit,
        prepared.input_delivery_slot,
        input_delivery,
        content_authority_id=(
            prepared.preparation_claim.reservation.request_blob.request_blob_id
        ),
    )
    credential_file_observation = (
        None
        if credential_delivery is None
        else _mint_activated_delivery_file(
            spawn_commit,
            prepared.credential_delivery_slot,
            credential_delivery,
            content_authority_id=credential_content_authority_id,
        )
    )
    result_file = prepared.result_file
    result_file_observation = RunActionActivatedFileObservation.mint(
        spawn_commit_id=spawn_commit.spawn_commit_id,
        prepared_parent_authority_id=(
            prepared.result_directory.prepared_runtime_directory_id
        ),
        prepared_file_id=result_file.prepared_file_id,
        parent_mount_id=prepared.result_directory.mount_id,
        parent_device=prepared.result_directory.device,
        parent_inode=prepared.result_directory.inode,
        runtime_volume_authority_id=result_file.runtime_volume_authority_id,
        generation_nonce=result_file.generation_nonce,
        kind=result_file.kind,
        relative_path=result_file.relative_path,
        file_type=result_file.file_type,
        owner_user_id=result_file.owner_user_id,
        owner_group_id=result_file.owner_group_id,
        mode=result_file.mode,
        link_count=result_file.link_count,
        size_bytes=0,
        mount_id=result_file.mount_id,
        device=result_file.device,
        inode=result_file.inode,
        content_digest=None,
        content_authority_id=None,
    )
    sentinel = prepared.runtime_volume_evidence.sentinel_evidence
    activated_sentinel_observation = RunActionActivatedSentinelObservation.mint(
        spawn_commit_id=spawn_commit.spawn_commit_id,
        prepared_sentinel_evidence_id=(sentinel.runtime_volume_sentinel_evidence_id),
        runtime_volume_authority_id=sentinel.runtime_volume_authority_id,
        generation_nonce=sentinel.generation_nonce,
        relative_path=sentinel.relative_path,
        file_type=sentinel.file_type,
        owner_user_id=sentinel.owner_user_id,
        owner_group_id=sentinel.owner_group_id,
        mode=sentinel.mode,
        link_count=sentinel.link_count,
        size_bytes=sentinel.size_bytes,
        content_digest=sentinel.content_digest,
        mount_id=sentinel.mount_id,
        device=sentinel.device,
        inode=sentinel.inode,
    )
    activated_runtime_directory_observations = tuple(
        RunActionActivatedRuntimeDirectoryObservation.mint(
            spawn_commit_id=spawn_commit.spawn_commit_id,
            prepared_runtime_directory_id=(
                runtime_directory.prepared_runtime_directory_id
            ),
            runtime_volume_authority_id=(runtime_directory.runtime_volume_authority_id),
            generation_nonce=runtime_directory.generation_nonce,
            kind=runtime_directory.kind,
            directory_relative_path=runtime_directory.directory_relative_path,
            directory_type=runtime_directory.directory_type,
            owner_user_id=runtime_directory.owner_user_id,
            owner_group_id=runtime_directory.owner_group_id,
            mode=runtime_directory.mode,
            observed_entry_count=0,
            mount_id=runtime_directory.mount_id,
            device=runtime_directory.device,
            inode=runtime_directory.inode,
        )
        for runtime_directory in (
            prepared.control_directory,
            prepared.temporary_directory,
        )
    )
    activated_workspace_observation = _mint_activated_workspace_observation(
        prepared,
        spawn_commit,
        activated_workspace_frontier,
    )
    return DockerRunActionActivatedVolumeObservation(
        prepared_execution=prepared,
        spawn_commit=spawn_commit,
        reobserved_volume_evidence=reobserved_volume_evidence,
        activated_workspace_observation=activated_workspace_observation,
        activated_runtime_directory_observations=(
            activated_runtime_directory_observations
        ),
        activated_sentinel_observation=activated_sentinel_observation,
        input_file_observation=input_file_observation,
        result_file_observation=result_file_observation,
        credential_file_observation=credential_file_observation,
    )


def _require_activation_workspace_source(
    prepared: RunActionPreparedExecution,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> None:
    if prepared.workspace_proof is None:
        if workspace_descriptor is not None:
            raise RunActionRuntimeVolumeError(
                "workspace-free activation carries a source descriptor"
            )
        return
    if type(workspace_descriptor) is not int or workspace_descriptor < 0:
        raise RunActionRuntimeVolumeError(
            "workspace activation lacks its exact source descriptor"
        )
    expected = prepared.workspace_proof.workspace_binding.to_identity()
    observed = inspect_run_workspace_frontier(
        workspace_descriptor,
        settings=settings,
        expected_commit_sha=expected.commit_sha,
    )
    if not _same_workspace_semantics(observed, expected):
        raise RunActionRuntimeVolumeError(
            "activation workspace source differs from the prepared frontier"
        )
    plan_run_workspace_frontier_copy(
        workspace_descriptor,
        settings=settings,
        expected=observed,
    )


def _open_activation_subpath_directory(
    descriptors: ExitStack,
    lease: _MountedRuntimeVolumeLease,
    authority: RunActionRuntimeVolumeAuthority,
    relative_path: str,
    expected_mount_id: int,
    expected_device: int,
    expected_inode: int,
) -> int:
    descriptor = os.open(
        relative_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=lease.root_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    mount_id = read_run_action_descriptor_mount_id(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != authority.owner_user_id
        or metadata.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != _PREPARED_DIRECTORY_MODE
        or mount_id != lease.root_mount_id
        or mount_id != expected_mount_id
        or metadata.st_dev != lease.root_device
        or metadata.st_dev != expected_device
        or metadata.st_ino != expected_inode
    ):
        raise RunActionRuntimeVolumeError(
            "activation runtime subpath is unsafe or substituted"
        )
    return descriptor


def _require_exact_activation_directory(
    prepared_directory: (
        RunActionPreparedDeliverySlot
        | RunActionPreparedRuntimeDirectory
        | RunActionPreparedWorkspaceProof
    ),
    root_descriptor: int,
    directory_descriptor: int,
    *,
    expected_entries: tuple[str, ...] | None,
) -> None:
    if type(prepared_directory) is RunActionPreparedWorkspaceProof:
        relative_path = prepared_directory.volume_subpath
    elif type(prepared_directory) in (
        RunActionPreparedDeliverySlot,
        RunActionPreparedRuntimeDirectory,
    ):
        relative_path = prepared_directory.directory_relative_path
    else:
        raise RunActionRuntimeVolumeError(
            "activation directory proof uses an unknown authority"
        )
    metadata_before = os.fstat(directory_descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(directory_descriptor)
    entries = tuple(sorted(os.listdir(directory_descriptor)))
    path_descriptor = os.open(
        relative_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_descriptor,
    )
    with ExitStack() as path_descriptors:
        path_descriptors.callback(os.close, path_descriptor)
        path_metadata_before = os.fstat(path_descriptor)
        path_mount_id_before = read_run_action_descriptor_mount_id(path_descriptor)
        path_entries = tuple(sorted(os.listdir(path_descriptor)))
        path_metadata_after = os.fstat(path_descriptor)
        path_mount_id_after = read_run_action_descriptor_mount_id(path_descriptor)
    metadata_after = os.fstat(directory_descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(directory_descriptor)
    if (
        not stat.S_ISDIR(metadata_before.st_mode)
        or _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or _stable_metadata(path_metadata_before) != _stable_metadata(metadata_before)
        or _stable_metadata(path_metadata_after)
        != _stable_metadata(path_metadata_before)
        or metadata_before.st_uid != prepared_directory.owner_user_id
        or metadata_before.st_gid != prepared_directory.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != _PREPARED_DIRECTORY_MODE
        or mount_id_before != prepared_directory.mount_id
        or mount_id_after != mount_id_before
        or path_mount_id_before != mount_id_before
        or path_mount_id_after != path_mount_id_before
        or metadata_before.st_dev != prepared_directory.device
        or metadata_before.st_ino != prepared_directory.inode
        or path_entries != entries
        or (expected_entries is not None and entries != tuple(sorted(expected_entries)))
    ):
        raise RunActionRuntimeVolumeError(
            "activation runtime directory changed or was substituted"
        )


def _observe_activation_workspace(
    workspace_directory_descriptor: int,
    prepared: RunActionPreparedExecution,
    settings: LaunchSettings,
) -> RunWorkspaceFrontierIdentity:
    workspace_proof = prepared.workspace_proof
    if type(workspace_proof) is not RunActionPreparedWorkspaceProof:
        raise RunActionRuntimeVolumeError(
            "workspace activation lacks its prepared proof"
        )
    expected = workspace_proof.workspace_binding.to_identity()
    observed = inspect_run_workspace_frontier(
        workspace_directory_descriptor,
        settings=settings,
        expected_commit_sha=expected.commit_sha,
    )
    if not _same_workspace_semantics(observed, expected):
        raise RunActionRuntimeVolumeError(
            "activated workspace differs from its prepared frontier"
        )
    plan_run_workspace_frontier_copy(
        workspace_directory_descriptor,
        settings=settings,
        expected=observed,
    )
    return observed


def _require_activation_result_file(
    observed: _ExactRegularFileObservation,
    prepared: RunActionPreparedFile,
) -> None:
    metadata = observed.metadata
    if (
        prepared.prepared_parent_directory_id.split(":sha256:", 1)[0]
        != RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE
        or prepared.kind is not RunActionPreparedFileKind.RESULT
        or observed.mount_id != prepared.mount_id
        or metadata.st_dev != prepared.device
        or metadata.st_ino != prepared.inode
        or metadata.st_uid != prepared.owner_user_id
        or metadata.st_gid != prepared.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != prepared.mode
        or metadata.st_nlink != prepared.link_count
        or metadata.st_size != 0
        or observed.payload
    ):
        raise RunActionRuntimeVolumeError(
            "activation result file differs from its prepared inode"
        )


def _open_reobserved_input_file(
    descriptors: ExitStack,
    directory_descriptor: int,
    slot: RunActionPreparedDeliverySlot,
    selected: RunActionActivatedFileObservation,
    authority: RunActionRuntimeVolumeAuthority,
    *,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileObservation:
    if (
        type(selected) is not RunActionActivatedFileObservation
        or slot.kind is not RunActionPreparedFileKind.INPUT
        or selected.prepared_parent_authority_id != slot.prepared_delivery_slot_id
        or selected.prepared_file_id is not None
        or selected.parent_mount_id != slot.mount_id
        or selected.parent_device != slot.device
        or selected.parent_inode != slot.inode
        or selected.runtime_volume_authority_id != slot.runtime_volume_authority_id
        or selected.generation_nonce != slot.generation_nonce
        or selected.kind is not slot.kind
        or selected.relative_path
        != f"{slot.directory_relative_path}/{slot.final_file_name}"
        or selected.owner_user_id != slot.owner_user_id
        or selected.owner_group_id != slot.owner_group_id
        or selected.size_bytes <= 0
        or selected.size_bytes > slot.payload_size_limit_bytes
        or selected.content_digest is None
    ):
        raise RunActionRuntimeVolumeError(
            "selected delivered file differs from its prepared slot"
        )
    observed = _open_regular_file_by_shape(
        descriptors,
        directory_descriptor,
        slot.final_file_name,
        expected_size_bytes=selected.size_bytes,
        expected_mode=selected.mode,
        authority=authority,
        root_mount_id=root_mount_id,
        root_device=root_device,
    )
    metadata = observed.metadata
    observed_digest = tree_or_blob_digest(observed.payload)
    if (
        metadata.st_ino != selected.inode
        or metadata.st_dev != selected.device
        or observed.mount_id != selected.mount_id
        or metadata.st_uid != selected.owner_user_id
        or metadata.st_gid != selected.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != selected.mode
        or metadata.st_nlink != selected.link_count
        or metadata.st_size != selected.size_bytes
        or observed_digest != selected.content_digest
    ):
        raise RunActionRuntimeVolumeError(
            "reopened delivered file differs from selected event 5"
        )
    return observed


def _open_reobserved_credential_file(
    descriptors: ExitStack,
    directory_descriptor: int,
    slot: RunActionPreparedDeliverySlot,
    selected: RunActionActivatedFileObservation,
    authority: RunActionRuntimeVolumeAuthority,
    *,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileShapeObservation:
    if (
        type(selected) is not RunActionActivatedFileObservation
        or slot.kind is not RunActionPreparedFileKind.CREDENTIAL
        or selected.prepared_parent_authority_id != slot.prepared_delivery_slot_id
        or selected.prepared_file_id is not None
        or selected.parent_mount_id != slot.mount_id
        or selected.parent_device != slot.device
        or selected.parent_inode != slot.inode
        or selected.runtime_volume_authority_id != slot.runtime_volume_authority_id
        or selected.generation_nonce != slot.generation_nonce
        or selected.kind is not slot.kind
        or selected.relative_path
        != f"{slot.directory_relative_path}/{slot.final_file_name}"
        or selected.owner_user_id != slot.owner_user_id
        or selected.owner_group_id != slot.owner_group_id
        or selected.size_bytes <= 0
        or selected.size_bytes > slot.payload_size_limit_bytes
        or selected.content_digest is not None
        or selected.content_authority_id is None
    ):
        raise RunActionRuntimeVolumeError(
            "selected credential differs from its prepared slot"
        )
    observed = _open_regular_file_shape_without_content(
        descriptors,
        directory_descriptor,
        slot.final_file_name,
        expected_size_bytes=selected.size_bytes,
        expected_mode=selected.mode,
        authority=authority,
        root_mount_id=root_mount_id,
        root_device=root_device,
    )
    metadata = observed.metadata
    if (
        metadata.st_ino != selected.inode
        or metadata.st_dev != selected.device
        or observed.mount_id != selected.mount_id
        or metadata.st_uid != selected.owner_user_id
        or metadata.st_gid != selected.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != selected.mode
        or metadata.st_nlink != selected.link_count
        or metadata.st_size != selected.size_bytes
    ):
        raise RunActionRuntimeVolumeError(
            "reopened credential differs from selected event 5"
        )
    return observed


def _physical_reobserved_delivery(
    slot: RunActionPreparedDeliverySlot | None,
    observed: _ExactRegularFileObservation | _ExactRegularFileShapeObservation | None,
    *,
    content_digest: str | None,
) -> RunActionDeliveredFilePhysicalObservation:
    if type(slot) is not RunActionPreparedDeliverySlot or type(observed) not in {
        _ExactRegularFileObservation,
        _ExactRegularFileShapeObservation,
    }:
        raise RunActionRuntimeVolumeError(
            "reopened delivery lacks exact physical authority"
        )
    metadata = observed.metadata
    return RunActionDeliveredFilePhysicalObservation(
        prepared_delivery_slot_id=slot.prepared_delivery_slot_id,
        runtime_volume_authority_id=slot.runtime_volume_authority_id,
        generation_nonce=slot.generation_nonce,
        kind=slot.kind,
        relative_path=f"{slot.directory_relative_path}/{slot.final_file_name}",
        file_type="regular",
        owner_user_id=metadata.st_uid,
        owner_group_id=metadata.st_gid,
        mode=stat.S_IMODE(metadata.st_mode),
        link_count=metadata.st_nlink,
        size_bytes=metadata.st_size,
        mount_id=observed.mount_id,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        content_digest=content_digest,
    )


def _mint_activated_delivery_file(
    spawn_commit: RunActionSpawnCommit,
    slot: RunActionPreparedDeliverySlot | None,
    delivered: RunActionDeliveredFilePhysicalObservation,
    *,
    content_authority_id: str | None,
) -> RunActionActivatedFileObservation:
    if (
        type(slot) is not RunActionPreparedDeliverySlot
        or delivered.prepared_delivery_slot_id != slot.prepared_delivery_slot_id
        or delivered.runtime_volume_authority_id != slot.runtime_volume_authority_id
        or delivered.generation_nonce != slot.generation_nonce
        or delivered.kind is not slot.kind
    ):
        raise RunActionRuntimeVolumeError(
            "delivered file observation differs from its prepared slot"
        )
    return RunActionActivatedFileObservation.mint(
        spawn_commit_id=spawn_commit.spawn_commit_id,
        prepared_parent_authority_id=slot.prepared_delivery_slot_id,
        prepared_file_id=None,
        parent_mount_id=slot.mount_id,
        parent_device=slot.device,
        parent_inode=slot.inode,
        runtime_volume_authority_id=delivered.runtime_volume_authority_id,
        generation_nonce=delivered.generation_nonce,
        kind=delivered.kind,
        relative_path=delivered.relative_path,
        file_type=delivered.file_type,
        owner_user_id=delivered.owner_user_id,
        owner_group_id=delivered.owner_group_id,
        mode=delivered.mode,
        link_count=delivered.link_count,
        size_bytes=delivered.size_bytes,
        mount_id=delivered.mount_id,
        device=delivered.device,
        inode=delivered.inode,
        content_digest=delivered.content_digest,
        content_authority_id=content_authority_id,
    )


def _mint_activated_workspace_observation(
    prepared: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    frontier: RunWorkspaceFrontierIdentity | None,
) -> RunActionActivatedWorkspaceObservation | None:
    workspace_proof = prepared.workspace_proof
    if workspace_proof is None:
        if frontier is not None:
            raise RunActionRuntimeVolumeError(
                "workspace-free activation observed a workspace frontier"
            )
        return None
    if type(
        frontier
    ) is not RunWorkspaceFrontierIdentity or not _same_workspace_semantics(
        frontier,
        workspace_proof.workspace_binding.to_identity(),
    ):
        raise RunActionRuntimeVolumeError(
            "activated workspace frontier differs from its prepared proof"
        )
    return RunActionActivatedWorkspaceObservation.mint(
        spawn_commit_id=spawn_commit.spawn_commit_id,
        prepared_workspace_proof_id=workspace_proof.prepared_workspace_proof_id,
        runtime_volume_authority_id=workspace_proof.runtime_volume_authority_id,
        generation_nonce=workspace_proof.generation_nonce,
        source_tree_digest=frontier.source_tree_digest,
        git_closure_digest=frontier.git_closure_digest,
        source_entry_count=frontier.source_entry_count,
        source_size_bytes=frontier.source_size_bytes,
        owner_user_id=workspace_proof.owner_user_id,
        owner_group_id=workspace_proof.owner_group_id,
        root_mode=workspace_proof.root_mode,
        mount_id=workspace_proof.mount_id,
        device=workspace_proof.device,
        inode=workspace_proof.inode,
    )


def _plan_runtime_volume_layout(
    claim: RunActionPreparationClaim,
    empty_volume: DockerRunActionEmptyVolumeObservation,
    *,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> _RuntimeVolumeLayoutPlan:
    workspace_binding = claim.reservation.frontier.workspace_before
    workspace_access = claim.reservation.intent.workspace_access
    if workspace_access is RunFrontierWorkspaceAccess.NONE:
        if workspace_descriptor is not None or workspace_binding is not None:
            raise RunActionRuntimeVolumeError(
                "workspace-free layout carries workspace authority"
            )
        workspace_copy_plan = None
    else:
        if (
            type(workspace_descriptor) is not int
            or workspace_descriptor < 0
            or workspace_binding is None
        ):
            raise RunActionRuntimeVolumeError(
                "workspace layout lacks its exact source descriptor"
            )
        workspace_copy_plan = plan_run_workspace_frontier_copy(
            workspace_descriptor,
            settings=settings,
            expected=workspace_binding.to_identity(),
        )
    directory_names = _expected_directory_names(claim)
    delivery_slot_plans = _expected_delivery_slot_plans(claim)
    runtime_directory_plans = _expected_runtime_directory_plans()
    result_file_plan = _expected_result_file_plan(claim)
    block_size = empty_volume.allocation_block_size_bytes
    nonworkspace_directory_count = len(directory_names) - (
        1 if workspace_copy_plan is not None else 0
    )
    workspace_size_bytes = (
        0
        if workspace_copy_plan is None
        else workspace_copy_plan.allocated_size_bytes(block_size)
    )
    workspace_inode_count = (
        0 if workspace_copy_plan is None else workspace_copy_plan.physical_entry_count
    )
    current_size_bytes = (
        block_size
        + nonworkspace_directory_count * block_size
        + workspace_size_bytes
        + _allocated_size_bytes(
            len(empty_volume.runtime_volume_authority.generation_nonce),
            block_size,
        )
    )
    limits = claim.execution_policy.docker_resource_limits
    future_size_bytes = _required_execution_headroom_size_bytes(
        claim,
        block_size,
    )
    current_inode_count = 1 + nonworkspace_directory_count + workspace_inode_count + 2
    if (
        current_size_bytes + future_size_bytes >= empty_volume.available_size_bytes
        or current_inode_count
        + len(delivery_slot_plans)
        + limits.runtime_temporary_reservation_inode_count
        + 2
        >= empty_volume.available_inode_count
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lacks peak preparation and execution headroom"
        )
    return _RuntimeVolumeLayoutPlan(
        directory_names=directory_names,
        delivery_slot_plans=delivery_slot_plans,
        runtime_directory_plans=runtime_directory_plans,
        result_file_plan=result_file_plan,
        workspace_copy_plan=workspace_copy_plan,
        preparation_size_bytes=current_size_bytes,
        preparation_inode_count=current_inode_count,
    )


def _materialize_layout_at_descriptor(
    root_descriptor: int,
    *,
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    plan: _RuntimeVolumeLayoutPlan,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> RunWorkspaceFrontierIdentity | None:
    if tuple(sorted(os.listdir(root_descriptor))):
        raise RunActionRuntimeVolumeError(
            "runtime volume is no longer empty before materialization"
        )
    staging_name = _PREPARATION_STAGING_PREFIX + authority.generation_nonce
    pending_sentinel_name = _PENDING_SENTINEL_PREFIX + authority.generation_nonce
    with ExitStack() as descriptors:
        staging_descriptor = _create_private_directory(
            root_descriptor,
            staging_name,
            authority,
        )
        descriptors.callback(os.close, staging_descriptor)
        directory_descriptors = {}
        for directory_name in plan.directory_names:
            directory_descriptor = _create_private_directory(
                staging_descriptor,
                directory_name,
                authority,
            )
            descriptors.callback(os.close, directory_descriptor)
            directory_descriptors[directory_name] = directory_descriptor
        _create_empty_prepared_file(
            directory_descriptors[plan.result_file_plan.directory_name],
            plan.result_file_plan.file_name,
            authority,
        )
        workspace_frontier = None
        if plan.workspace_copy_plan is not None:
            if type(workspace_descriptor) is not int:
                raise RunActionRuntimeVolumeError(
                    "workspace copy lost its source descriptor"
                )
            workspace_frontier = copy_run_workspace_frontier(
                workspace_descriptor,
                directory_descriptors["workspace"],
                settings=settings,
                plan=plan.workspace_copy_plan,
            )
        _create_staged_sentinel(
            staging_descriptor,
            authority,
        )
        for directory_descriptor in directory_descriptors.values():
            os.fsync(directory_descriptor)
        os.fsync(staging_descriptor)
        for directory_name in plan.directory_names:
            _rename_no_replace(
                staging_descriptor,
                directory_name,
                root_descriptor,
                directory_name,
            )
        _rename_no_replace(
            staging_descriptor,
            _SENTINEL_NAME,
            root_descriptor,
            pending_sentinel_name,
        )
        os.fsync(root_descriptor)
        os.rmdir(staging_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        if tuple(sorted(os.listdir(root_descriptor))) != tuple(
            sorted((*plan.directory_names, pending_sentinel_name))
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume publication contains an unexpected path"
            )
        _rename_no_replace(
            root_descriptor,
            pending_sentinel_name,
            root_descriptor,
            _SENTINEL_NAME,
        )
        os.fsync(root_descriptor)
    if tuple(sorted(os.listdir(root_descriptor))) != tuple(
        sorted((*plan.directory_names, _SENTINEL_NAME))
    ):
        raise RunActionRuntimeVolumeError(
            "published runtime volume layout is incomplete"
        )
    return workspace_frontier


def _create_private_directory(
    parent_descriptor: int,
    name: str,
    authority: RunActionRuntimeVolumeAuthority,
) -> int:
    os.mkdir(name, mode=_PREPARED_DIRECTORY_MODE, dir_fd=parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    os.fchmod(descriptor, _PREPARED_DIRECTORY_MODE)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != authority.owner_user_id
        or metadata.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != _PREPARED_DIRECTORY_MODE
        or metadata.st_dev <= 0
        or metadata.st_ino <= 0
    ):
        os.close(descriptor)
        raise RunActionRuntimeVolumeError(
            "runtime volume prepared directory has unsafe metadata"
        )
    return descriptor


def _create_empty_prepared_file(
    parent_descriptor: int,
    name: str,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        _PREPARED_FILE_MODE,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), _PREPARED_FILE_MODE)
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != authority.owner_user_id
            or metadata.st_gid != authority.owner_group_id
            or stat.S_IMODE(metadata.st_mode) != _PREPARED_FILE_MODE
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume prepared file has unsafe metadata"
            )
        os.fsync(handle.fileno())
    os.fsync(parent_descriptor)


def _create_staged_sentinel(
    staging_descriptor: int,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    payload = authority.generation_nonce.encode("ascii")
    descriptor = os.open(
        _SENTINEL_NAME,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        _SENTINEL_MODE,
        dir_fd=staging_descriptor,
    )
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        os.fchmod(handle.fileno(), _SENTINEL_MODE)
        written_size = 0
        while written_size < len(payload):
            written = handle.write(payload[written_size:])
            if written <= 0:
                raise RunActionRuntimeVolumeError(
                    "runtime volume sentinel write made no progress"
                )
            written_size += written
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != authority.owner_user_id
            or metadata.st_gid != authority.owner_group_id
            or stat.S_IMODE(metadata.st_mode) != _SENTINEL_MODE
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume generation sentinel has unsafe metadata"
            )
        os.fsync(handle.fileno())


def _rename_no_replace(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RunActionRuntimeVolumeError(
            "atomic no-replace runtime-volume publication is unavailable"
        )
    rename_at2 = libc.renameat2
    rename_at2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_at2.restype = ctypes.c_int
    result = rename_at2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _allocated_size_bytes(size_bytes: int, block_size_bytes: int) -> int:
    return ((size_bytes + block_size_bytes - 1) // block_size_bytes) * block_size_bytes


def _require_empty_volume_lease(
    empty_volume: DockerRunActionEmptyVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    lease: _MountedRuntimeVolumeLease,
) -> None:
    metadata = os.fstat(lease.root_descriptor)
    mount_info = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    filesystem = os.fstatvfs(lease.root_descriptor)
    _require_mount_authority(
        mount_info,
        metadata,
        empty_volume.runtime_volume_authority,
    )
    _require_consistent_filesystem(filesystem)
    if (
        empty_volume.keeper_container_id != lease.keeper_container_id
        or empty_volume.keeper_process_id != lease.keeper_process_id
        or empty_volume.keeper_process_start_time_ticks
        != lease.process_start_time_ticks
        or empty_volume.process_cgroup_path != lease.process_cgroup_path
        or empty_volume.mount_id != lease.root_mount_id
        or empty_volume.device != lease.root_device
        or empty_volume.root_inode != lease.root_inode
        or empty_volume.keeper_container_id != keeper.container_id
        or _filesystem_capacity(filesystem)
        != (
            empty_volume.allocation_block_size_bytes,
            empty_volume.effective_block_count,
            empty_volume.effective_size_bytes,
            empty_volume.effective_inode_limit,
            empty_volume.used_block_count,
            empty_volume.used_size_bytes,
            empty_volume.used_inode_count,
            empty_volume.available_block_count,
            empty_volume.available_size_bytes,
            empty_volume.available_inode_count,
        )
        or tuple(sorted(os.listdir(lease.root_descriptor)))
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume no longer matches its empty-generation proof"
        )


def _observe_prepared_layout_at_descriptor(
    lease: _MountedRuntimeVolumeLease,
    *,
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    keeper: RunActionVolumeKeeperEvidence,
    workspace_frontier: RunWorkspaceFrontierIdentity | None,
    settings: LaunchSettings,
) -> _PreparedLayoutObservation:
    expected_directories = _expected_directory_names(claim)
    expected_root_entries = tuple(sorted((*expected_directories, _SENTINEL_NAME)))
    metadata_before = os.fstat(lease.root_descriptor)
    mount_info_before = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    filesystem_before = os.fstatvfs(lease.root_descriptor)
    _require_mount_authority(mount_info_before, metadata_before, authority)
    _require_consistent_filesystem(filesystem_before)
    if tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries:
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume root topology is incomplete"
        )
    observed_identities = {
        (metadata_before.st_dev, metadata_before.st_ino),
    }
    directory_descriptors = {}
    directory_metadata_before = {}
    with ExitStack() as descriptors:
        for directory_name in expected_directories:
            directory_descriptor = os.open(
                directory_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=lease.root_descriptor,
            )
            descriptors.callback(os.close, directory_descriptor)
            directory_metadata = os.fstat(directory_descriptor)
            directory_mount_id = read_run_action_descriptor_mount_id(
                directory_descriptor
            )
            identity = (directory_metadata.st_dev, directory_metadata.st_ino)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid != authority.owner_user_id
                or directory_metadata.st_gid != authority.owner_group_id
                or stat.S_IMODE(directory_metadata.st_mode) != _PREPARED_DIRECTORY_MODE
                or directory_mount_id != lease.root_mount_id
                or directory_metadata.st_dev != lease.root_device
                or identity in observed_identities
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume directory is unsafe or substituted"
                )
            observed_identities.add(identity)
            directory_descriptors[directory_name] = directory_descriptor
            directory_metadata_before[directory_name] = _stable_metadata(
                directory_metadata
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            lease.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=authority.generation_nonce.encode("ascii"),
            expected_mode=_SENTINEL_MODE,
            authority=authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
        sentinel_identity = (
            sentinel_observation.metadata.st_dev,
            sentinel_observation.metadata.st_ino,
        )
        if sentinel_identity in observed_identities:
            raise RunActionRuntimeVolumeError(
                "runtime volume sentinel repeats another layout inode"
            )
        observed_identities.add(sentinel_identity)
        expected_children = {
            "control": (),
            "input": (),
            "result": ("result.blob",),
            "temporary": (),
            **(
                {}
                if claim.execution_policy.credential_policy.mode
                is RunActionCredentialMode.NONE
                else {"credential": ()}
            ),
        }
        for directory_name, child_names in expected_children.items():
            if tuple(sorted(os.listdir(directory_descriptors[directory_name]))) != (
                child_names
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume logical directory has extra paths"
                )
        runtime_directory_observations = tuple(
            _ExactDirectoryObservation(
                metadata=os.fstat(directory_descriptors[directory_plan.directory_name]),
                mount_id=read_run_action_descriptor_mount_id(
                    directory_descriptors[directory_plan.directory_name]
                ),
            )
            for directory_plan in _expected_runtime_directory_plans()
        )
        delivery_slot_observations = tuple(
            _ExactDirectoryObservation(
                metadata=os.fstat(directory_descriptors[slot_plan.directory_name]),
                mount_id=read_run_action_descriptor_mount_id(
                    directory_descriptors[slot_plan.directory_name]
                ),
            )
            for slot_plan in _expected_delivery_slot_plans(claim)
        )
        result_file_plan = _expected_result_file_plan(claim)
        result_file_observation = _open_exact_regular_file(
            descriptors,
            directory_descriptors[result_file_plan.directory_name],
            result_file_plan.file_name,
            expected_payload=b"",
            expected_mode=_PREPARED_FILE_MODE,
            authority=authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
        result_identity = (
            result_file_observation.metadata.st_dev,
            result_file_observation.metadata.st_ino,
        )
        if result_identity in observed_identities:
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume result repeats another layout inode"
            )
        observed_identities.add(result_identity)
        for slot_observation in delivery_slot_observations:
            if (
                not stat.S_ISDIR(slot_observation.metadata.st_mode)
                or stat.S_IMODE(slot_observation.metadata.st_mode)
                != _PREPARED_DIRECTORY_MODE
                or slot_observation.mount_id != lease.root_mount_id
                or slot_observation.metadata.st_dev != lease.root_device
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume delivery slot is unsafe or substituted"
                )
        observed_workspace_frontier = None
        workspace_directory_observation = None
        if workspace_frontier is None:
            if "workspace" in directory_descriptors:
                raise RunActionRuntimeVolumeError(
                    "workspace-free runtime volume contains a workspace"
                )
        else:
            workspace_descriptor = directory_descriptors["workspace"]
            workspace_directory_observation = _ExactDirectoryObservation(
                metadata=os.fstat(workspace_descriptor),
                mount_id=read_run_action_descriptor_mount_id(workspace_descriptor),
            )
            observed_workspace_frontier = inspect_run_workspace_frontier(
                workspace_descriptor,
                settings=settings,
                expected_commit_sha=workspace_frontier.commit_sha,
            )
            if not _same_workspace_semantics(
                observed_workspace_frontier,
                workspace_frontier,
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared workspace differs from its durable frontier"
                )
            plan_run_workspace_frontier_copy(
                workspace_descriptor,
                settings=settings,
                expected=observed_workspace_frontier,
            )
        metadata_after = os.fstat(lease.root_descriptor)
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        filesystem_after = os.fstatvfs(lease.root_descriptor)
        _require_same_mounted_runtime_volume(lease, keeper)
        _require_same_exact_regular_file(sentinel_observation)
        _require_same_exact_regular_file(result_file_observation)
        directory_metadata_after = {
            directory_name: _stable_metadata(os.fstat(directory_descriptor))
            for directory_name, directory_descriptor in directory_descriptors.items()
        }
    if (
        _root_metadata_identity(metadata_after)
        != _root_metadata_identity(metadata_before)
        or mount_info_after != mount_info_before
        or _stable_filesystem(filesystem_after) != _stable_filesystem(filesystem_before)
        or directory_metadata_after != directory_metadata_before
        or tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume changed during exact observation"
        )
    return _PreparedLayoutObservation(
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
        root_inode=lease.root_inode,
        sentinel_metadata=sentinel_observation.metadata,
        sentinel_mount_id=sentinel_observation.mount_id,
        delivery_slot_observations=delivery_slot_observations,
        runtime_directory_observations=runtime_directory_observations,
        result_file_observation=result_file_observation,
        workspace_directory_observation=workspace_directory_observation,
        workspace_frontier=observed_workspace_frontier,
        filesystem=filesystem_before,
    )


def _open_exact_regular_file(
    descriptors: ExitStack,
    parent_descriptor: int,
    name: str,
    *,
    expected_payload: bytes,
    expected_mode: int,
    authority: RunActionRuntimeVolumeAuthority,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileObservation:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(descriptor)
    payload = _read_bounded_descriptor_payload(
        descriptor,
        len(expected_payload) + 1,
    )
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(descriptor)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_id_after != mount_id_before
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_uid != authority.owner_user_id
        or metadata_before.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != expected_mode
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != len(expected_payload)
        or payload != expected_payload
        or mount_id_before != root_mount_id
        or metadata_before.st_dev != root_device
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume file is unsafe or substituted"
        )
    return _ExactRegularFileObservation(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        name=name,
        metadata=metadata_before,
        mount_id=mount_id_before,
        payload=payload,
    )


def _open_regular_file_by_shape(
    descriptors: ExitStack,
    parent_descriptor: int,
    name: str,
    *,
    expected_size_bytes: int,
    expected_mode: int,
    authority: RunActionRuntimeVolumeAuthority,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileObservation:
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise RunActionRuntimeVolumeError(
            "selected runtime volume file lacks a positive size"
        )
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(descriptor)
    payload = _read_bounded_descriptor_payload(
        descriptor,
        expected_size_bytes + 1,
    )
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(descriptor)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_id_after != mount_id_before
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_uid != authority.owner_user_id
        or metadata_before.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != expected_mode
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != expected_size_bytes
        or len(payload) != expected_size_bytes
        or mount_id_before != root_mount_id
        or metadata_before.st_dev != root_device
    ):
        raise RunActionRuntimeVolumeError(
            "selected runtime volume file is unsafe or substituted"
        )
    return _ExactRegularFileObservation(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        name=name,
        metadata=metadata_before,
        mount_id=mount_id_before,
        payload=payload,
    )


def _open_regular_file_shape_without_content(
    descriptors: ExitStack,
    parent_descriptor: int,
    name: str,
    *,
    expected_size_bytes: int,
    expected_mode: int,
    authority: RunActionRuntimeVolumeAuthority,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileShapeObservation:
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise RunActionRuntimeVolumeError("selected credential lacks a positive size")
    descriptor = os.open(
        name,
        os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(descriptor)
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(descriptor)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_id_after != mount_id_before
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_uid != authority.owner_user_id
        or metadata_before.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != expected_mode
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != expected_size_bytes
        or mount_id_before != root_mount_id
        or metadata_before.st_dev != root_device
    ):
        raise RunActionRuntimeVolumeError(
            "selected credential file is unsafe or substituted"
        )
    return _ExactRegularFileShapeObservation(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        name=name,
        metadata=metadata_before,
        mount_id=mount_id_before,
    )


def _require_same_exact_regular_file(
    observation: _ExactRegularFileObservation,
) -> None:
    os.lseek(observation.descriptor, 0, os.SEEK_SET)
    payload = _read_bounded_descriptor_payload(
        observation.descriptor,
        len(observation.payload) + 1,
    )
    metadata = os.fstat(observation.descriptor)
    mount_id = read_run_action_descriptor_mount_id(observation.descriptor)
    path_descriptor = os.open(
        observation.name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=observation.parent_descriptor,
    )
    with ExitStack() as path_descriptors:
        path_descriptors.callback(os.close, path_descriptor)
        path_metadata_before = os.fstat(path_descriptor)
        path_mount_id_before = read_run_action_descriptor_mount_id(path_descriptor)
        path_payload = _read_bounded_descriptor_payload(
            path_descriptor,
            len(observation.payload) + 1,
        )
        path_metadata_after = os.fstat(path_descriptor)
        path_mount_id_after = read_run_action_descriptor_mount_id(path_descriptor)
        if (
            payload != observation.payload
            or _stable_metadata(metadata) != _stable_metadata(observation.metadata)
            or mount_id != observation.mount_id
            or path_payload != observation.payload
            or _stable_metadata(path_metadata_before)
            != _stable_metadata(observation.metadata)
            or _stable_metadata(path_metadata_after)
            != _stable_metadata(path_metadata_before)
            or path_mount_id_before != observation.mount_id
            or path_mount_id_after != path_mount_id_before
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume file changed during exact observation"
            )


def _require_same_exact_regular_file_shape(
    observation: _ExactRegularFileShapeObservation,
) -> None:
    metadata = os.fstat(observation.descriptor)
    mount_id = read_run_action_descriptor_mount_id(observation.descriptor)
    path_descriptor = os.open(
        observation.name,
        os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=observation.parent_descriptor,
    )
    with ExitStack() as path_descriptors:
        path_descriptors.callback(os.close, path_descriptor)
        path_metadata_before = os.fstat(path_descriptor)
        path_mount_id_before = read_run_action_descriptor_mount_id(path_descriptor)
        path_metadata_after = os.fstat(path_descriptor)
        path_mount_id_after = read_run_action_descriptor_mount_id(path_descriptor)
    if (
        _stable_metadata(metadata) != _stable_metadata(observation.metadata)
        or mount_id != observation.mount_id
        or _stable_metadata(path_metadata_before)
        != _stable_metadata(observation.metadata)
        or _stable_metadata(path_metadata_after)
        != _stable_metadata(path_metadata_before)
        or path_mount_id_before != observation.mount_id
        or path_mount_id_after != path_mount_id_before
    ):
        raise RunActionRuntimeVolumeError(
            "selected credential shape changed during exact observation"
        )


def _read_bounded_descriptor_payload(descriptor: int, limit: int) -> bytes:
    chunks = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _mint_prepared_volume_observation(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    observed: _PreparedLayoutObservation,
    *,
    empty_entry_count: int,
    empty_size_bytes: int,
) -> DockerRunActionPreparedVolumeObservation:
    delivery_slot_plans = _expected_delivery_slot_plans(claim)
    if len(delivery_slot_plans) != len(observed.delivery_slot_observations):
        raise RunActionRuntimeVolumeError(
            "observed prepared delivery slots differ from the layout plan"
        )
    delivery_slots = tuple(
        RunActionPreparedDeliverySlot.mint(
            preparation_claim_id=claim.preparation_claim_id,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            generation_nonce=authority.generation_nonce,
            kind=slot_plan.kind,
            directory_relative_path=slot_plan.directory_name,
            final_file_name=slot_plan.final_file_name,
            directory_type="directory",
            owner_user_id=authority.owner_user_id,
            owner_group_id=authority.owner_group_id,
            mode=_PREPARED_DIRECTORY_MODE,
            observed_entry_count=0,
            payload_size_limit_bytes=slot_plan.payload_size_limit_bytes,
            mount_id=slot_observation.mount_id,
            device=slot_observation.metadata.st_dev,
            inode=slot_observation.metadata.st_ino,
        )
        for slot_plan, slot_observation in zip(
            delivery_slot_plans,
            observed.delivery_slot_observations,
            strict=True,
        )
    )
    delivery_slot_by_kind = {
        delivery_slot.kind: delivery_slot for delivery_slot in delivery_slots
    }
    runtime_directory_plans = _expected_runtime_directory_plans()
    if len(runtime_directory_plans) != len(observed.runtime_directory_observations):
        raise RunActionRuntimeVolumeError(
            "observed prepared runtime directories differ from the layout plan"
        )
    runtime_directories = tuple(
        RunActionPreparedRuntimeDirectory.mint(
            preparation_claim_id=claim.preparation_claim_id,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            generation_nonce=authority.generation_nonce,
            kind=directory_plan.kind,
            directory_relative_path=directory_plan.directory_name,
            directory_type="directory",
            owner_user_id=authority.owner_user_id,
            owner_group_id=authority.owner_group_id,
            mode=_PREPARED_DIRECTORY_MODE,
            observed_entry_count=directory_plan.observed_entry_count,
            mount_id=directory_observation.mount_id,
            device=directory_observation.metadata.st_dev,
            inode=directory_observation.metadata.st_ino,
        )
        for directory_plan, directory_observation in zip(
            runtime_directory_plans,
            observed.runtime_directory_observations,
            strict=True,
        )
    )
    runtime_directory_by_kind = {
        runtime_directory.kind: runtime_directory
        for runtime_directory in runtime_directories
    }
    result_file_plan = _expected_result_file_plan(claim)
    result_file_observation = observed.result_file_observation
    result_directory = runtime_directory_by_kind[
        RunActionPreparedRuntimeDirectoryKind.RESULT
    ]
    result_file = RunActionPreparedFile.mint(
        preparation_claim_id=claim.preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        prepared_parent_directory_id=(result_directory.prepared_runtime_directory_id),
        kind=result_file_plan.kind,
        relative_path=result_file_plan.relative_path,
        file_type="regular",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=_PREPARED_FILE_MODE,
        link_count=1,
        size_bytes=0,
        payload_size_limit_bytes=result_file_plan.payload_size_limit_bytes,
        mount_id=result_file_observation.mount_id,
        device=result_file_observation.metadata.st_dev,
        inode=result_file_observation.metadata.st_ino,
    )
    workspace_binding = claim.reservation.frontier.workspace_before
    if (
        (workspace_binding is None) != (observed.workspace_frontier is None)
        or (workspace_binding is None)
        != (observed.workspace_directory_observation is None)
        or (
            workspace_binding is not None
            and not _same_workspace_semantics(
                observed.workspace_frontier,
                workspace_binding.to_identity(),
            )
        )
    ):
        raise RunActionRuntimeVolumeError(
            "observed prepared workspace differs from its claim"
        )
    workspace_proof = (
        None
        if workspace_binding is None
        else RunActionPreparedWorkspaceProof.mint(
            preparation_claim_id=claim.preparation_claim_id,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            generation_nonce=authority.generation_nonce,
            volume_subpath="workspace",
            workspace_binding=workspace_binding,
            observed_source_tree_digest=workspace_binding.source_tree_digest,
            observed_git_closure_digest=workspace_binding.git_closure_digest,
            observed_source_entry_count=workspace_binding.source_entry_count,
            observed_source_size_bytes=workspace_binding.source_size_bytes,
            owner_user_id=authority.owner_user_id,
            owner_group_id=authority.owner_group_id,
            root_mode=_PREPARED_DIRECTORY_MODE,
            mount_id=observed.workspace_directory_observation.mount_id,
            device=observed.workspace_directory_observation.metadata.st_dev,
            inode=observed.workspace_directory_observation.metadata.st_ino,
            unexpected_entry_count=0,
        )
    )
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=_SENTINEL_NAME,
        file_type="regular",
        owner_user_id=observed.sentinel_metadata.st_uid,
        owner_group_id=observed.sentinel_metadata.st_gid,
        mode=stat.S_IMODE(observed.sentinel_metadata.st_mode),
        link_count=observed.sentinel_metadata.st_nlink,
        size_bytes=observed.sentinel_metadata.st_size,
        content_digest=tree_or_blob_digest(authority.generation_nonce.encode("ascii")),
        mount_id=observed.sentinel_mount_id,
        device=observed.sentinel_metadata.st_dev,
        inode=observed.sentinel_metadata.st_ino,
    )
    filesystem = observed.filesystem
    _require_consistent_filesystem(filesystem)
    (
        allocation_block_size_bytes,
        effective_block_count,
        effective_size_bytes,
        effective_inode_limit,
        used_block_count,
        used_size_bytes,
        used_inode_count,
        available_block_count,
        available_size_bytes,
        available_inode_count,
    ) = _filesystem_capacity(filesystem)
    limits = claim.execution_policy.docker_resource_limits
    required_available_size_bytes = _required_execution_headroom_size_bytes(
        claim,
        allocation_block_size_bytes,
    )
    required_available_inode_count = (
        len(delivery_slot_plans) + limits.runtime_temporary_reservation_inode_count + 2
    )
    if (
        required_available_size_bytes >= available_size_bytes
        or required_available_inode_count >= available_inode_count
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume lacks positive execution headroom"
        )
    volume_evidence = _mint_runtime_volume_evidence(
        authority,
        volume,
        keeper,
        root_mount_id=observed.root_mount_id,
        root_device=observed.root_device,
        root_inode=observed.root_inode,
        sentinel_evidence=sentinel_evidence,
        filesystem=filesystem,
    )
    expected_directories = _expected_directory_names(claim)
    logical_workspace_size = (
        0 if workspace_binding is None else workspace_binding.source_size_bytes
    )
    logical_workspace_entries = (
        0 if workspace_binding is None else workspace_binding.source_entry_count
    )
    layout_proof = RunActionRuntimeVolumeLayoutProof.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        runtime_volume_evidence_id=volume_evidence.runtime_volume_evidence_id,
        generation_nonce=authority.generation_nonce,
        empty_size_bytes=empty_size_bytes,
        empty_entry_count=empty_entry_count,
        directory_relative_paths=expected_directories,
        prepared_delivery_slot_ids=tuple(
            sorted(
                delivery_slot.prepared_delivery_slot_id
                for delivery_slot in delivery_slots
            )
        ),
        prepared_runtime_directory_ids=tuple(
            sorted(
                directory.prepared_runtime_directory_id
                for directory in runtime_directories
            )
        ),
        prepared_result_file_id=result_file.prepared_file_id,
        prepared_workspace_proof_id=(
            None
            if workspace_proof is None
            else workspace_proof.prepared_workspace_proof_id
        ),
        logical_content_size_bytes=(
            len(authority.generation_nonce) + logical_workspace_size
        ),
        logical_entry_count=(len(expected_directories) + 2 + logical_workspace_entries),
        observed_used_size_bytes=used_size_bytes,
        observed_used_inode_count=used_inode_count,
        unexpected_entry_count=0,
    )
    return DockerRunActionPreparedVolumeObservation(
        preparation_claim=claim,
        runtime_volume_evidence=volume_evidence,
        input_delivery_slot=delivery_slot_by_kind[RunActionPreparedFileKind.INPUT],
        control_directory=runtime_directory_by_kind[
            RunActionPreparedRuntimeDirectoryKind.CONTROL
        ],
        result_directory=result_directory,
        result_file=result_file,
        temporary_directory=runtime_directory_by_kind[
            RunActionPreparedRuntimeDirectoryKind.TEMPORARY
        ],
        credential_delivery_slot=delivery_slot_by_kind.get(
            RunActionPreparedFileKind.CREDENTIAL
        ),
        workspace_proof=workspace_proof,
        layout_proof=layout_proof,
    )


def _mint_runtime_volume_evidence(
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    root_mount_id: int,
    root_device: int,
    root_inode: int,
    sentinel_evidence: RunActionRuntimeVolumeSentinelEvidence,
    filesystem: os.statvfs_result,
) -> RunActionRuntimeVolumeEvidence:
    (
        allocation_block_size_bytes,
        effective_block_count,
        effective_size_bytes,
        effective_inode_limit,
        used_block_count,
        used_size_bytes,
        used_inode_count,
        available_block_count,
        available_size_bytes,
        available_inode_count,
    ) = _filesystem_capacity(filesystem)
    return RunActionRuntimeVolumeEvidence.mint(
        volume_authority=authority,
        docker_volume_occurrence_digest=volume.volume_occurrence_digest,
        volume_keeper_evidence_id=keeper.volume_keeper_evidence_id,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        keeper_process_start_time_ticks=keeper.process_start_time_ticks,
        keeper_process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
        root_mount_id=root_mount_id,
        root_device=root_device,
        root_inode=root_inode,
        observed_volume_name=volume.volume_name,
        observed_labels=authority.labels,
        observed_scope="local",
        observed_driver=authority.driver,
        observed_driver_options=authority.driver_options,
        observed_filesystem_type=_TMPFS_FILESYSTEM_TYPE,
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        observed_owner_user_id=authority.owner_user_id,
        observed_owner_group_id=authority.owner_group_id,
        observed_root_mode=authority.root_mode,
        allocation_block_size_bytes=allocation_block_size_bytes,
        effective_block_count=effective_block_count,
        effective_size_bytes=effective_size_bytes,
        effective_inode_limit=effective_inode_limit,
        used_block_count=used_block_count,
        used_size_bytes=used_size_bytes,
        used_inode_count=used_inode_count,
        available_block_count=available_block_count,
        available_size_bytes=available_size_bytes,
        available_inode_count=available_inode_count,
        sentinel_evidence=sentinel_evidence,
    )


def _expected_directory_names(claim: RunActionPreparationClaim) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                "control",
                "input",
                "result",
                "temporary",
                *(
                    ()
                    if claim.execution_policy.credential_policy.mode
                    is RunActionCredentialMode.NONE
                    else ("credential",)
                ),
                *(
                    ()
                    if claim.reservation.frontier.workspace_before is None
                    else ("workspace",)
                ),
            )
        )
    )


def _expected_delivery_slot_plans(
    claim: RunActionPreparationClaim,
) -> tuple[_PreparedDeliverySlotPlan, ...]:
    credential_policy = claim.execution_policy.credential_policy
    return (
        _PreparedDeliverySlotPlan(
            kind=RunActionPreparedFileKind.INPUT,
            directory_name="input",
            final_file_name="request.blob",
            payload_size_limit_bytes=claim.reservation.request_blob.size_bytes,
        ),
        *(
            ()
            if credential_policy.mode is RunActionCredentialMode.NONE
            else (
                _PreparedDeliverySlotPlan(
                    kind=RunActionPreparedFileKind.CREDENTIAL,
                    directory_name="credential",
                    final_file_name="credentials",
                    payload_size_limit_bytes=(
                        credential_policy.maximum_delivery_size_bytes
                    ),
                ),
            )
        ),
    )


def _expected_result_file_plan(
    claim: RunActionPreparationClaim,
) -> _PreparedResultFilePlan:
    return _PreparedResultFilePlan(
        kind=RunActionPreparedFileKind.RESULT,
        directory_name="result",
        file_name="result.blob",
        relative_path="result/result.blob",
        payload_size_limit_bytes=(
            claim.execution_policy.supervisor_limits.result_size_bytes
        ),
    )


def _required_execution_headroom_size_bytes(
    claim: RunActionPreparationClaim,
    allocation_block_size_bytes: int,
) -> int:
    limits = claim.execution_policy
    return sum(
        _allocated_size_bytes(payload_size_bytes, allocation_block_size_bytes)
        for payload_size_bytes in (
            *(
                slot.payload_size_limit_bytes
                for slot in _expected_delivery_slot_plans(claim)
            ),
            limits.supervisor_limits.result_size_bytes,
            limits.docker_resource_limits.runtime_temporary_reservation_size_bytes,
            limits.supervisor_limits.release_receipt_size_bytes,
            limits.supervisor_limits.timeout_directive_size_bytes,
        )
    )


def _expected_runtime_directory_plans() -> tuple[_PreparedRuntimeDirectoryPlan, ...]:
    return (
        _PreparedRuntimeDirectoryPlan(
            kind=RunActionPreparedRuntimeDirectoryKind.CONTROL,
            directory_name="control",
            observed_entry_count=0,
        ),
        _PreparedRuntimeDirectoryPlan(
            kind=RunActionPreparedRuntimeDirectoryKind.RESULT,
            directory_name="result",
            observed_entry_count=1,
        ),
        _PreparedRuntimeDirectoryPlan(
            kind=RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
            directory_name="temporary",
            observed_entry_count=0,
        ),
    )


def _same_workspace_semantics(
    observed: RunWorkspaceFrontierIdentity,
    expected: RunWorkspaceFrontierIdentity,
) -> bool:
    return (
        observed.branch == expected.branch
        and observed.commit_sha == expected.commit_sha
        and observed.parent_commit_shas == expected.parent_commit_shas
        and observed.git_tree_sha == expected.git_tree_sha
        and observed.source_tree_digest == expected.source_tree_digest
        and observed.git_closure_digest == expected.git_closure_digest
        and observed.source_entry_count == expected.source_entry_count
        and observed.source_size_bytes == expected.source_size_bytes
    )


def _require_consistent_filesystem(filesystem: os.statvfs_result) -> None:
    if (
        filesystem.f_bsize != filesystem.f_frsize
        or filesystem.f_bfree != filesystem.f_bavail
        or filesystem.f_ffree != filesystem.f_favail
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume statvfs carries reserved or inconsistent capacity"
        )


def _filesystem_capacity(
    filesystem: os.statvfs_result,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    allocation_block_size_bytes = filesystem.f_frsize
    effective_block_count = filesystem.f_blocks
    available_block_count = filesystem.f_bfree
    used_block_count = effective_block_count - available_block_count
    effective_inode_limit = filesystem.f_files
    available_inode_count = filesystem.f_ffree
    used_inode_count = effective_inode_limit - available_inode_count
    return (
        allocation_block_size_bytes,
        effective_block_count,
        effective_block_count * allocation_block_size_bytes,
        effective_inode_limit,
        used_block_count,
        used_block_count * allocation_block_size_bytes,
        used_inode_count,
        available_block_count,
        available_block_count * allocation_block_size_bytes,
        available_inode_count,
    )


def _root_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_mount_info(
    process_descriptor: int,
    mount_id: int,
    destination: str,
) -> _MountInfo:
    descriptor = os.open(
        "mountinfo",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_mount_info_payload(payload, mount_id, destination)


def _parse_mount_info_payload(
    payload: bytes,
    mount_id: int,
    destination: str,
) -> _MountInfo:
    if (
        type(payload) is not bytes
        or not payload
        or not payload.endswith(b"\n")
        or b"\x00" in payload
        or type(mount_id) is not int
        or mount_id <= 0
        or type(destination) is not str
        or destination != RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION
    ):
        raise RunActionRuntimeVolumeError("keeper mountinfo is malformed or incomplete")
    encoded_lines = payload.splitlines()
    matching_lines = tuple(
        line
        for line in encoded_lines
        if line.split(b" ", 1)[0] == str(mount_id).encode()
    )
    if len(matching_lines) != 1 or not matching_lines[0].isascii():
        raise RunActionRuntimeVolumeError(
            "keeper mountinfo lacks one runtime-volume mount"
        )
    line = matching_lines[0].decode("ascii")
    sections = line.split(" - ")
    if len(sections) != 2:
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo separator is malformed"
        )
    mount_fields = sections[0].split(" ")
    filesystem_fields = sections[1].split(" ")
    if (
        len(mount_fields) < 6
        or len(filesystem_fields) != 3
        or not mount_fields[0].isdigit()
        or not mount_fields[1].isdigit()
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo fields are malformed"
        )
    device_parts = mount_fields[2].split(":")
    if (
        len(device_parts) != 2
        or any(not part.isdigit() for part in device_parts)
        or mount_fields[3] != "/"
        or mount_fields[4] != destination
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo path or device is malformed"
        )
    return _MountInfo(
        mount_id=int(mount_fields[0]),
        parent_mount_id=int(mount_fields[1]),
        device_major=int(device_parts[0]),
        device_minor=int(device_parts[1]),
        mount_point=mount_fields[4],
        mount_options=_parse_options(mount_fields[5]),
        optional_fields=tuple(mount_fields[6:]),
        filesystem_type=filesystem_fields[0],
        source=filesystem_fields[1],
        super_options=_parse_options(filesystem_fields[2]),
    )


def _parse_options(value: str) -> tuple[str, ...]:
    options = tuple(value.split(","))
    if (
        not options
        or any(
            not option or any(character.isspace() for character in option)
            for option in options
        )
        or len(options) != len(set(options))
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mount options are malformed"
        )
    return tuple(sorted(options))


def _require_mount_authority(
    mount_info: _MountInfo,
    metadata: os.stat_result,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    flag_options = tuple(
        option for option in mount_info.super_options if "=" not in option
    )
    keyed_option_pairs = tuple(
        option.split("=", 1) for option in mount_info.super_options if "=" in option
    )
    keyed_options = dict(keyed_option_pairs)
    if (
        mount_info.mount_options != _MOUNT_OPTIONS
        or any(
            _OPTIONAL_MOUNT_FIELD_PATTERN.fullmatch(field) is None
            for field in mount_info.optional_fields
        )
        or len(mount_info.optional_fields) > 1
        or mount_info.filesystem_type != _TMPFS_FILESYSTEM_TYPE
        or mount_info.source != _TMPFS_FILESYSTEM_TYPE
        or flag_options != _SUPER_OPTION_FLAGS
        or len(keyed_options) != len(keyed_option_pairs)
        or tuple(sorted(keyed_options)) != _SUPER_OPTION_KEYS
        or _parse_size_option(keyed_options["size"]) != authority.size_limit_bytes
        or keyed_options["nr_inodes"] != str(authority.inode_limit)
        or keyed_options["mode"] != f"{authority.root_mode:o}"
        or keyed_options["uid"] != str(authority.owner_user_id)
        or keyed_options["gid"] != str(authority.owner_group_id)
        or mount_info.device_major != os.major(metadata.st_dev)
        or mount_info.device_minor != os.minor(metadata.st_dev)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != authority.owner_user_id
        or metadata.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != authority.root_mode
        or metadata.st_nlink < 2
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mount differs from issued tmpfs authority"
        )


def _parse_size_option(value: str) -> int:
    match = _SIZE_OPTION_PATTERN.fullmatch(value)
    if match is None:
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume size option is malformed"
        )
    return int(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2)]


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_filesystem(filesystem: os.statvfs_result) -> tuple[int, ...]:
    return (
        filesystem.f_bsize,
        filesystem.f_frsize,
        filesystem.f_blocks,
        filesystem.f_bfree,
        filesystem.f_bavail,
        filesystem.f_files,
        filesystem.f_ffree,
        filesystem.f_favail,
        filesystem.f_flag,
        filesystem.f_namemax,
    )


__all__ = [
    "DockerRunActionActivatedVolumeObservation",
    "DockerRunActionEmptyVolumeObservation",
    "DockerRunActionPreparedVolumeObservation",
    "RunActionActivationRevalidationLease",
    "RunActionControlDirectoryLease",
    "RunActionResultWorkspaceLease",
    "RunActionRuntimeVolumeError",
    "adopt_prepared_runtime_volume_layout",
    "capture_run_action_result_file",
    "deliver_and_reobserve_runtime_volume_activation",
    "materialize_runtime_volume_layout",
    "observe_empty_runtime_volume",
    "open_run_action_control_directory",
    "open_run_action_result_workspace",
    "open_selected_run_action_activation",
    "reobserve_runtime_volume_layout",
]
