"""Contracts for deterministic claims and concrete inert Docker occurrences."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

import kapso.cross_run.launch.run_action_supervisor_contracts as supervisor_contracts
from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ContractValidationError
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionExecutionLifecycleIdentity,
    RunActionIntent,
    RunActionResultInterpreterIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionLedgerSnapshot
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionCreateInspectProjection,
    DockerRunActionExecutionPolicy,
    DockerRunActionKeeperCreateInspectProjection,
    DockerRunActionResourceLimits,
    DockerRunActionSafeCreateDefaults,
    DockerRunActionSandboxSpec,
    RUN_ACTION_BARRIER_CONTROL_DESTINATION,
    RUN_ACTION_BARRIER_DUMMY_ARGUMENT,
    RUN_ACTION_BARRIER_PROTOCOL_VERSION,
    RUN_ACTION_BARRIER_RELEASE_DESTINATION,
    RUN_ACTION_BARRIER_SCRIPT,
    RUN_ACTION_DOCKER_INIT_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionActivatedFileObservation,
    RunActionActivatedSentinelObservation,
    RunActionActivatedRuntimeDirectoryObservation,
    RunActionActivatedWorkspaceObservation,
    RunActionActivationNetworkMode,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionCredentialPolicy,
    RunActionDockerInitSourceEvidence,
    RunActionFilesystemPolicy,
    RunActionInertContainerEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionMountedKeeperHelperEvidence,
    RunActionNetworkPolicy,
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    RunActionPreparedDeliverySlot,
    RunActionPreparedExecution,
    RunActionPreparedFile,
    RunActionPreparedFileKind,
    RunActionPreparedMount,
    RunActionPreparedMountAccess,
    RunActionPreparedMountKind,
    RunActionPreparedRuntimeDirectory,
    RunActionPreparedRuntimeDirectoryKind,
    RunActionPreparedWorkspaceProof,
    RunActionRuntimeVolumeAuthority,
    RunActionRuntimeVolumeEvidence,
    RunActionRuntimeVolumeLayoutProof,
    RunActionRuntimeVolumeSentinelEvidence,
    RunActionResultCaptureReceipt,
    RunActionStaticEnvironmentVariable,
    RunActionSupervisorContractError,
    RunActionSupervisorLimits,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
    issue_runtime_volume_authority,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
    runtime_volume_driver_options,
    run_action_docker_init_authority_id,
    run_action_supervisor_helper_authority_id,
    runtime_volume_sentinel_identity,
    run_action_activated_volume_evidence_matches,
    run_action_keeper_process_cgroup_path,
    run_action_runtime_volume_occurrence_matches,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.settings import CrossRunSettings

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_BARRIER_POLL_INTERVAL_SECONDS = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).docker.run_action_barrier_poll_interval_seconds
_RUN_ACTION_RELEASE_RECEIPT_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_release_receipt_size_bytes
_RUN_ACTION_TIMEOUT_DIRECTIVE_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_timeout_directive_size_bytes
_RUN_ACTION_RELEASE_COMMIT_TIMEOUT_SECONDS = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_release_commit_timeout_seconds


def _fixture_content_id(namespace: str, label: str) -> str:
    return content_id(namespace, {"fixture": label})


def _boundary(
    kind=RunFrontierActionKind.CODING_AGENT,
    *,
    execution_policy_id=None,
):
    execution_policy_id = (
        _fixture_content_id("test-run-action-execution-policy", kind.value)
        if execution_policy_id is None
        else execution_policy_id
    )
    lifecycle = RunActionExecutionLifecycleIdentity.mint(
        kind=kind,
        implementation_id=f"test.{kind.value}.execution",
        implementation_version="test.execution.v1",
        recovery_protocol_version="test.recovery.v1",
        execution_policy_id=execution_policy_id,
    )
    return RunActionBoundaryIdentity.mint(
        kind=kind,
        execution_lifecycle_identity=lifecycle,
        result_interpreter_identity=RunActionResultInterpreterIdentity.mint(
            kind=kind,
            implementation_id=f"test.{kind.value}.interpreter",
            implementation_version="test.interpreter.v1",
            interpretation_protocol_version="test.interpretation.v1",
        ),
    )


def _credential_policy(mode=RunActionCredentialMode.SUPERVISOR_FILE):
    if mode is RunActionCredentialMode.NONE:
        return RunActionCredentialPolicy.mint(
            mode=mode,
            broker_id=None,
            broker_protocol_version=None,
            principal_id=None,
            audience_id=None,
            scope_ids=(),
            maximum_lease_seconds=None,
            maximum_delivery_size_bytes=None,
        )
    return RunActionCredentialPolicy.mint(
        mode=mode,
        broker_id="test.credential.broker",
        broker_protocol_version="test.credential.broker.v1",
        principal_id="test.provider.principal",
        audience_id="test.provider.audience",
        scope_ids=("test.provider.invoke",),
        maximum_lease_seconds=900,
        maximum_delivery_size_bytes=4096,
    )


def _execution_policy(
    *,
    kind=RunFrontierActionKind.CODING_AGENT,
    workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
    credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    command_template_label="default",
):
    credential_policy = _credential_policy(credential_mode)
    helper_digest = tree_or_blob_digest(b"volume keeper helper")
    helper_source_path = "/usr/bin/busybox"
    init_digest = tree_or_blob_digest(b"Docker init executable")
    init_source_path = "/usr/bin/docker-init"
    return DockerRunActionExecutionPolicy.mint(
        kind=kind,
        supervisor_protocol_version="kapso.run_action_supervisor.v1",
        projection_protocol_version="kapso.docker_create_inspect_projection.v1",
        raw_field_schema_id=_fixture_content_id(
            "docker-raw-field-schema",
            "docker-api-closed-schema",
        ),
        docker_runtime_settings_digest=tree_or_blob_digest(b"docker settings"),
        image_authority=DockerImageAuthority.mint(
            image_reference=(
                "registry.example/kapso/run-action@"
                + tree_or_blob_digest(b"image manifest")
            ),
            image_config_digest=tree_or_blob_digest(b"image config"),
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
        ),
        supervisor_helper_source_path=helper_source_path,
        supervisor_helper_executable_authority_id=(
            run_action_supervisor_helper_authority_id(
                helper_source_path,
                helper_digest,
            )
        ),
        supervisor_helper_executable_digest=helper_digest,
        docker_init_source_path=init_source_path,
        docker_init_executable_authority_id=(
            run_action_docker_init_authority_id(
                init_source_path,
                init_digest,
            )
        ),
        docker_init_executable_digest=init_digest,
        command_template_id=content_id(
            "docker-run-action-command-template",
            {
                "arguments": (command_template_label,),
                "entrypoint": "/bin/tool",
            },
        ),
        static_environment=(
            RunActionStaticEnvironmentVariable(key="LANG", value="C"),
            RunActionStaticEnvironmentVariable(key="PATH", value="/usr/bin:/bin"),
        ),
        user_id=1000,
        group_id=1000,
        hostname="kapso-run-action",
        safe_create_defaults=DockerRunActionSafeCreateDefaults.mint(
            open_stdin=False,
            terminal=False,
            stdin_once=False,
            attach_stdin=False,
            exposed_port_ids=(),
            port_binding_ids=(),
            publish_all_ports=False,
            link_ids=(),
            extra_host_ids=(),
            dns_server_ids=(),
            dns_search_ids=(),
            dns_option_ids=(),
            endpoint_alias_ids=(),
            volume_from_ids=(),
            storage_option_ids=(),
            anonymous_volume_count=0,
        ),
        sandbox_spec=DockerRunActionSandboxSpec.mint(
            read_only_root_filesystem=True,
            privileged=False,
            capability_additions=(),
            capability_drops=("ALL",),
            device_authority_ids=(),
            device_request_authority_ids=(),
            device_cgroup_rule_ids=(),
            supplementary_group_ids=(),
            pid_namespace_mode="private",
            ipc_namespace_mode="private",
            uts_namespace_mode="private",
            cgroup_namespace_mode="private",
            user_namespace_mode="daemon_default_unmapped",
            cgroup_parent_id="test.kapso.run_action.slice",
            sysctl_ids=(),
            no_new_privileges=True,
            seccomp_profile_id="builtin",
            apparmor_profile_id="docker-default",
            security_option_ids=(
                "apparmor:docker-default",
                "no-new-privileges",
                "seccomp:builtin",
            ),
            masked_system_paths=(
                "/proc/acpi",
                "/proc/asound",
                "/proc/interrupts",
                "/proc/kcore",
                "/proc/keys",
                "/proc/latency_stats",
                "/proc/sched_debug",
                "/proc/scsi",
                "/proc/timer_list",
                "/proc/timer_stats",
                "/sys/devices/virtual/powercap",
                "/sys/firmware",
            ),
            read_only_system_paths=(
                "/proc/bus",
                "/proc/fs",
                "/proc/irq",
                "/proc/sys",
                "/proc/sysrq-trigger",
            ),
            runtime_id="runc",
            log_driver="none",
            log_option_ids=(),
            init_process=True,
            isolation_mode="default",
        ),
        filesystem_policy=RunActionFilesystemPolicy.mint(
            workspace_access=workspace_access,
            workspace_destination=(
                None
                if workspace_access is RunFrontierWorkspaceAccess.NONE
                else "/kapso/workspace"
            ),
            input_destination="/kapso/input",
            result_destination="/kapso/result",
            credential_destination=(
                None
                if credential_mode is RunActionCredentialMode.NONE
                else "/kapso/credentials"
            ),
            working_directory=(
                "/kapso/input"
                if workspace_access is RunFrontierWorkspaceAccess.NONE
                else "/kapso/workspace"
            ),
            temporary_filesystem_destination="/kapso/tmp",
        ),
        network_policy=RunActionNetworkPolicy.mint(
            activation_mode=RunActionActivationNetworkMode.NONE,
            broker_endpoint_ids=(),
        ),
        credential_policy=credential_policy,
        docker_resource_limits=DockerRunActionResourceLimits.mint(
            cpu_period_microseconds=100000,
            cpu_quota_microseconds=200000,
            cpu_shares=1024,
            cpuset_cpu_ids=(),
            cpuset_memory_node_ids=(),
            memory_size_bytes=1073741824,
            memory_reservation_size_bytes=536870912,
            memory_swap_size_bytes=1073741824,
            oom_score_adjustment=0,
            process_limit=128,
            block_io_weight=500,
            shared_memory_size_bytes=67108864,
            runtime_volume_size_bytes=536870912,
            runtime_volume_inode_limit=4096,
            runtime_temporary_reservation_size_bytes=67108864,
            runtime_temporary_reservation_inode_count=1024,
        ),
        supervisor_limits=RunActionSupervisorLimits.mint(
            execution_timeout_seconds=600,
            termination_grace_seconds=30,
            release_commit_timeout_seconds=(_RUN_ACTION_RELEASE_COMMIT_TIMEOUT_SECONDS),
            result_size_bytes=268435456,
            release_receipt_size_bytes=_RUN_ACTION_RELEASE_RECEIPT_SIZE_BYTES,
            timeout_directive_size_bytes=_RUN_ACTION_TIMEOUT_DIRECTIVE_SIZE_BYTES,
        ),
    )


def _claim(
    *,
    policy=None,
    boundary=None,
    request_digest=None,
    request_payload=None,
    security_observation_id=None,
):
    policy = _execution_policy() if policy is None else policy
    boundary = (
        _boundary(
            policy.kind,
            execution_policy_id=policy.docker_execution_policy_id,
        )
        if boundary is None
        else boundary
    )
    if request_payload is not None:
        if type(request_payload) is not bytes or not request_payload:
            raise AssertionError("test request payload must be nonempty bytes")
        request_digest = tree_or_blob_digest(request_payload)
    else:
        request_digest = (
            tree_or_blob_digest(b"complete request")
            if request_digest is None
            else request_digest
        )
    has_workspace = (
        policy.filesystem_policy.workspace_access is not RunFrontierWorkspaceAccess.NONE
    )
    request_payload = (
        b"complete request" if request_payload is None else request_payload
    )
    if (
        request_payload == b"complete request"
        and request_digest != tree_or_blob_digest(request_payload)
    ):
        request_payload = b"another request"
    if tree_or_blob_digest(request_payload) != request_digest:
        raise AssertionError("test request digest lacks matching fixture bytes")
    intent = RunActionIntent.from_request(
        kind=policy.kind,
        boundary=(
            RunSafetyBoundary.IMPLEMENTATION
            if (
                policy.kind is RunFrontierActionKind.CODING_AGENT
                and policy.filesystem_policy.workspace_access
                is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            )
            else {
                RunFrontierActionKind.CODING_AGENT: RunSafetyBoundary.IDEATION,
                RunFrontierActionKind.EMBEDDING: RunSafetyBoundary.IDEATION,
                RunFrontierActionKind.EVALUATOR: RunSafetyBoundary.EVALUATION,
            }[policy.kind]
        ),
        operation_id=f"test.{policy.kind.value}.operation",
        request_payload=request_payload,
        workspace_access=policy.filesystem_policy.workspace_access,
        boundary_identity=boundary,
    )
    workspace = (
        RunActionWorkspaceBinding(
            workspace_device=101,
            workspace_inode=202,
            branch="main",
            commit_sha="a" * 40,
            parent_commit_shas=(),
            git_tree_sha="b" * 40,
            source_tree_digest=tree_or_blob_digest(b"source"),
            git_closure_digest=tree_or_blob_digest(b"git closure"),
            source_entry_count=1,
            source_size_bytes=1,
        )
        if has_workspace
        else None
    )
    frontier = RunActionFrontierBinding.mint(
        bootstrap_pin_id=_fixture_content_id("bootstrap-pin", "pin"),
        run_checkpoint_id=_fixture_content_id("run-checkpoint", "checkpoint"),
        safety_state_id=_fixture_content_id("run-safety-state", "safety"),
        security_observation_id=(
            _fixture_content_id("security-denylist-observation", "security")
            if security_observation_id is None
            else security_observation_id
        ),
        generation_id=_fixture_content_id(
            "run-derived-state-generation",
            "generation",
        ),
        journal_head_id=_fixture_content_id(
            "run-checkpoint-head",
            "checkpoint-head",
        ),
        journal_size_bytes=1,
        bundle_digest=tree_or_blob_digest(b"bundle"),
        bundle_size_bytes=1,
        view_bindings=(
            RunActionViewBinding(
                relative_path=".kapso/view.json",
                digest=tree_or_blob_digest(b"view"),
                size_bytes=1,
            ),
        ),
        workspace_before=workspace,
    )
    reservation = RunActionReservation.build(
        intent=intent,
        frontier=frontier,
        predecessor_ledger=RunActionLedgerSnapshot.empty(),
    )
    return RunActionPreparationClaim.mint(
        reservation=reservation,
        execution_policy=policy,
    )


def _volume_authority(claim, *, nonce):
    return issue_runtime_volume_authority(claim, nonce)


def _prepared_delivery_slot(
    claim,
    authority,
    kind,
    *,
    mount_id,
    device,
    inode,
):
    limits = {
        RunActionPreparedFileKind.INPUT: claim.reservation.request_blob.size_bytes,
        RunActionPreparedFileKind.CREDENTIAL: (
            claim.execution_policy.credential_policy.maximum_delivery_size_bytes
        ),
    }
    paths = {
        RunActionPreparedFileKind.INPUT: ("input", "request.blob"),
        RunActionPreparedFileKind.CREDENTIAL: ("credential", "credentials"),
    }
    return RunActionPreparedDeliverySlot.mint(
        preparation_claim_id=claim.preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=kind,
        directory_relative_path=paths[kind][0],
        final_file_name=paths[kind][1],
        directory_type="directory",
        owner_user_id=claim.execution_policy.user_id,
        owner_group_id=claim.execution_policy.group_id,
        mode=0o700,
        observed_entry_count=0,
        payload_size_limit_bytes=limits[kind],
        mount_id=mount_id,
        device=device,
        inode=inode,
    )


def _prepared_result_file(
    claim,
    authority,
    parent_directory,
    *,
    mount_id,
    device,
    inode,
):
    return RunActionPreparedFile.mint(
        prepared_parent_directory_id=(parent_directory.prepared_runtime_directory_id),
        preparation_claim_id=claim.preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=RunActionPreparedFileKind.RESULT,
        relative_path="result/result.blob",
        file_type="regular",
        owner_user_id=claim.execution_policy.user_id,
        owner_group_id=claim.execution_policy.group_id,
        mode=0o600,
        link_count=1,
        size_bytes=0,
        payload_size_limit_bytes=(
            claim.execution_policy.supervisor_limits.result_size_bytes
        ),
        mount_id=mount_id,
        device=device,
        inode=inode,
    )


def _prepared_runtime_directory(
    claim,
    authority,
    kind,
    *,
    mount_id,
    device,
    inode,
):
    return RunActionPreparedRuntimeDirectory.mint(
        preparation_claim_id=claim.preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=kind,
        directory_relative_path=kind.value,
        directory_type="directory",
        owner_user_id=claim.execution_policy.user_id,
        owner_group_id=claim.execution_policy.group_id,
        mode=0o700,
        observed_entry_count=(
            1 if kind is RunActionPreparedRuntimeDirectoryKind.RESULT else 0
        ),
        mount_id=mount_id,
        device=device,
        inode=inode,
    )


def _mounts(claim, volume_name):
    filesystem = claim.execution_policy.filesystem_policy
    specifications = [
        (
            RunActionPreparedMountKind.INPUT,
            "input",
            filesystem.input_destination,
            RunActionPreparedMountAccess.READ_ONLY,
        ),
        (
            RunActionPreparedMountKind.RESULT,
            "result",
            filesystem.result_destination,
            RunActionPreparedMountAccess.READ_WRITE,
        ),
        (
            RunActionPreparedMountKind.TEMPORARY,
            "temporary",
            filesystem.temporary_filesystem_destination,
            RunActionPreparedMountAccess.READ_WRITE,
        ),
        (
            RunActionPreparedMountKind.CONTROL,
            "control",
            RUN_ACTION_BARRIER_CONTROL_DESTINATION,
            RunActionPreparedMountAccess.READ_ONLY,
        ),
    ]
    if filesystem.credential_destination is not None:
        specifications.append(
            (
                RunActionPreparedMountKind.CREDENTIAL,
                "credential",
                filesystem.credential_destination,
                RunActionPreparedMountAccess.READ_ONLY,
            )
        )
    if filesystem.workspace_destination is not None:
        specifications.append(
            (
                RunActionPreparedMountKind.WORKSPACE,
                "workspace",
                filesystem.workspace_destination,
                (
                    RunActionPreparedMountAccess.READ_WRITE
                    if filesystem.workspace_access
                    is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                    else RunActionPreparedMountAccess.READ_ONLY
                ),
            )
        )
    return tuple(
        sorted(
            (
                RunActionPreparedMount(
                    kind=kind,
                    volume_name=volume_name,
                    volume_subpath=subpath,
                    container_destination=destination,
                    mount_type="volume",
                    source_access=RunActionPreparedMountAccess.READ_WRITE,
                    container_access=access,
                    host_config_volume_subpath=subpath,
                )
                for kind, subpath, destination, access in specifications
            ),
            key=lambda mount: mount.container_destination,
        )
    )


def _prepared_execution(
    *,
    claim=None,
    authority=None,
    container_id="a" * 64,
    inode_offset=0,
):
    claim = _claim() if claim is None else claim
    nonce = f"{inode_offset + 1:032x}"
    authority = (
        _volume_authority(claim, nonce=nonce) if authority is None else authority
    )
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=claim.execution_policy.user_id,
        owner_group_id=claim.execution_policy.group_id,
        mode=0o400,
        link_count=1,
        size_bytes=len(authority.generation_nonce),
        content_digest=tree_or_blob_digest(authority.generation_nonce.encode("ascii")),
        mount_id=1000 + inode_offset,
        device=500,
        inode=10000 + inode_offset,
    )
    file_mount_id = 1000 + inode_offset
    file_device = 500
    first_artifact_inode = 20000 + inode_offset * 8
    input_delivery_slot = _prepared_delivery_slot(
        claim,
        authority,
        RunActionPreparedFileKind.INPUT,
        mount_id=file_mount_id,
        device=file_device,
        inode=first_artifact_inode,
    )
    result_directory = _prepared_runtime_directory(
        claim,
        authority,
        RunActionPreparedRuntimeDirectoryKind.RESULT,
        mount_id=file_mount_id,
        device=file_device,
        inode=first_artifact_inode + 1,
    )
    control_directory = _prepared_runtime_directory(
        claim,
        authority,
        RunActionPreparedRuntimeDirectoryKind.CONTROL,
        mount_id=file_mount_id,
        device=file_device,
        inode=first_artifact_inode + 6,
    )
    temporary_directory = _prepared_runtime_directory(
        claim,
        authority,
        RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
        mount_id=file_mount_id,
        device=file_device,
        inode=first_artifact_inode + 2,
    )
    result_file = _prepared_result_file(
        claim,
        authority,
        result_directory,
        mount_id=file_mount_id,
        device=file_device,
        inode=first_artifact_inode + 3,
    )
    credential_delivery_slot = (
        None
        if claim.execution_policy.credential_policy.mode is RunActionCredentialMode.NONE
        else _prepared_delivery_slot(
            claim,
            authority,
            RunActionPreparedFileKind.CREDENTIAL,
            mount_id=file_mount_id,
            device=file_device,
            inode=first_artifact_inode + 4,
        )
    )
    delivery_slots = tuple(
        delivery_slot
        for delivery_slot in (input_delivery_slot, credential_delivery_slot)
        if delivery_slot is not None
    )
    policy = claim.execution_policy
    workspace_binding = claim.reservation.frontier.workspace_before
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
            owner_user_id=policy.user_id,
            owner_group_id=policy.group_id,
            root_mode=0o700,
            unexpected_entry_count=0,
            mount_id=file_mount_id,
            device=file_device,
            inode=first_artifact_inode + 5,
        )
    )
    directories = tuple(
        sorted(
            {
                "control",
                "input",
                "result",
                "temporary",
                *(("credential",) if credential_delivery_slot is not None else ()),
                *(("workspace",) if workspace_proof is not None else ()),
            }
        )
    )
    workspace_size = (
        0 if workspace_binding is None else workspace_binding.source_size_bytes
    )
    workspace_entries = (
        0 if workspace_binding is None else workspace_binding.source_entry_count
    )
    logical_entry_count = len(directories) + 2 + workspace_entries
    observed_used_size = 32768
    observed_used_inodes = logical_entry_count + 2
    helper_evidence = RunActionSupervisorHelperEvidence.mint(
        helper_authority_id=policy.supervisor_helper_executable_authority_id,
        source_path=policy.supervisor_helper_source_path,
        destination="/kapso-supervisor/busybox",
        mount_type="bind",
        mount_access=RunActionPreparedMountAccess.READ_ONLY,
        recursive_bind=False,
        file_type="regular",
        owner_user_id=0,
        owner_group_id=0,
        mode=0o755,
        link_count=1,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.supervisor_helper_executable_digest,
        mount_id=3000 + inode_offset,
        device=700,
        inode=800,
    )
    init_source_evidence = RunActionDockerInitSourceEvidence.mint(
        init_authority_id=policy.docker_init_executable_authority_id,
        source_path=policy.docker_init_source_path,
        file_type="regular",
        owner_user_id=0,
        owner_group_id=0,
        mode=0o755,
        link_count=1,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.docker_init_executable_digest,
        mount_id=3000 + inode_offset,
        device=700,
        inode=801,
    )
    keeper_projection = DockerRunActionKeeperCreateInspectProjection.mint(
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        preparation_claim_id=claim.preparation_claim_id,
        execution_policy=policy,
        volume_authority=authority,
        command_executable="/kapso-supervisor/busybox",
        command_arguments=("tail", "-f", "/dev/null"),
        helper_evidence=helper_evidence,
        docker_init_source_evidence=init_source_evidence,
        volume_mount_type="volume",
        volume_mount_destination="/kapso/runtime-volume",
        volume_mount_access=RunActionPreparedMountAccess.READ_WRITE,
        network_mode="none",
        exact_mount_count=2,
        healthcheck_present=False,
        docker_socket_mounted=False,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=4,
    )
    keeper_container_id = f"{inode_offset + 2:064x}"
    keeper_process_id = 1000 + inode_offset
    mounted_helper_evidence = RunActionMountedKeeperHelperEvidence.mint(
        source_helper_evidence=helper_evidence,
        container_id=keeper_container_id,
        process_id=keeper_process_id,
        process_start_time_ticks=2000 + inode_offset,
        process_cgroup_path=(
            "/test.kapso.run_action.slice/" f"docker-{keeper_container_id}.scope"
        ),
        destination="/kapso-supervisor/busybox",
        mount_id=helper_evidence.mount_id + 1,
        device=helper_evidence.device,
        inode=helper_evidence.inode,
        executable_digest=helper_evidence.executable_digest,
    )
    keeper_evidence = RunActionVolumeKeeperEvidence.mint(
        preparation_claim_id=claim.preparation_claim_id,
        container_id=keeper_container_id,
        container_name=preparation_keeper_container_name(claim),
        labels=preparation_keeper_container_labels(claim),
        issued_create_projection=keeper_projection,
        observed_inspect_projection=keeper_projection,
        mounted_helper_evidence=mounted_helper_evidence,
        container_status="running",
        process_id=keeper_process_id,
        process_start_time_ticks=mounted_helper_evidence.process_start_time_ticks,
        restart_count=0,
        restart_policy_name="no",
        auto_remove=False,
    )
    volume_evidence = RunActionRuntimeVolumeEvidence.mint(
        volume_authority=authority,
        docker_volume_occurrence_digest=tree_or_blob_digest(
            f"volume-occurrence-{inode_offset}".encode()
        ),
        volume_keeper_evidence_id=keeper_evidence.volume_keeper_evidence_id,
        keeper_container_id=keeper_evidence.container_id,
        keeper_process_id=keeper_evidence.process_id,
        keeper_process_start_time_ticks=keeper_evidence.process_start_time_ticks,
        keeper_process_cgroup_path=(
            keeper_evidence.mounted_helper_evidence.process_cgroup_path
        ),
        root_mount_id=sentinel_evidence.mount_id,
        root_device=sentinel_evidence.device,
        root_inode=9000 + inode_offset,
        observed_volume_name=authority.volume_name,
        observed_labels=authority.labels,
        observed_scope="local",
        observed_driver=authority.driver,
        observed_driver_options=authority.driver_options,
        observed_filesystem_type="tmpfs",
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        observed_owner_user_id=authority.owner_user_id,
        observed_owner_group_id=authority.owner_group_id,
        observed_root_mode=authority.root_mode,
        allocation_block_size_bytes=4096,
        effective_block_count=authority.size_limit_bytes // 4096,
        effective_size_bytes=authority.size_limit_bytes,
        effective_inode_limit=authority.inode_limit,
        used_block_count=observed_used_size // 4096,
        used_size_bytes=observed_used_size,
        used_inode_count=observed_used_inodes,
        available_block_count=(
            authority.size_limit_bytes // 4096 - observed_used_size // 4096
        ),
        available_size_bytes=authority.size_limit_bytes - observed_used_size,
        available_inode_count=authority.inode_limit - observed_used_inodes,
        sentinel_evidence=sentinel_evidence,
    )
    layout_proof = RunActionRuntimeVolumeLayoutProof.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        runtime_volume_evidence_id=volume_evidence.runtime_volume_evidence_id,
        generation_nonce=authority.generation_nonce,
        empty_size_bytes=0,
        empty_entry_count=0,
        directory_relative_paths=directories,
        prepared_delivery_slot_ids=tuple(
            sorted(
                delivery_slot.prepared_delivery_slot_id
                for delivery_slot in delivery_slots
            )
        ),
        prepared_runtime_directory_ids=tuple(
            sorted(
                (
                    result_directory.prepared_runtime_directory_id,
                    temporary_directory.prepared_runtime_directory_id,
                    control_directory.prepared_runtime_directory_id,
                )
            )
        ),
        prepared_result_file_id=result_file.prepared_file_id,
        prepared_workspace_proof_id=(
            None
            if workspace_proof is None
            else workspace_proof.prepared_workspace_proof_id
        ),
        logical_content_size_bytes=len(authority.generation_nonce) + workspace_size,
        logical_entry_count=logical_entry_count,
        observed_used_size_bytes=observed_used_size,
        observed_used_inode_count=observed_used_inodes,
        unexpected_entry_count=0,
    )
    issued_projection = DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        execution_policy=policy,
        supervisor_helper_evidence=helper_evidence,
        docker_init_source_evidence=init_source_evidence,
        barrier_protocol_version=RUN_ACTION_BARRIER_PROTOCOL_VERSION,
        barrier_poll_interval_seconds=_BARRIER_POLL_INTERVAL_SECONDS,
        command_executable=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        command_arguments=(
            "sh",
            "-eu",
            "-c",
            RUN_ACTION_BARRIER_SCRIPT,
            RUN_ACTION_BARRIER_DUMMY_ARGUMENT,
            RUN_ACTION_BARRIER_RELEASE_DESTINATION,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            str(_BARRIER_POLL_INTERVAL_SECONDS),
            "/bin/tool",
            "default",
        ),
        mounts=_mounts(claim, authority.volume_name),
        exact_mount_count=len(_mounts(claim, authority.volume_name)) + 1,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=4,
    )
    evidence = RunActionInertContainerEvidence.mint(
        preparation_claim_id=claim.preparation_claim_id,
        container_id=container_id,
        container_name=preparation_container_name(claim),
        labels=preparation_container_labels(claim),
        image_authority_id=claim.execution_policy.image_authority.image_authority_id,
        docker_runtime_settings_digest=(
            claim.execution_policy.docker_runtime_settings_digest
        ),
        issued_create_projection=issued_projection,
        observed_inspect_projection=issued_projection,
        container_status="created",
        process_id=0,
        restart_count=0,
        started_at="0001-01-01T00:00:00Z",
        finished_at="0001-01-01T00:00:00Z",
        restart_policy_name="no",
        auto_remove=False,
        network_mode="none",
        healthcheck_present=False,
        volume_plugin_mount_count=0,
        docker_socket_mounted=False,
    )
    return RunActionPreparedExecution.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
        runtime_volume_evidence=volume_evidence,
        volume_keeper_evidence=keeper_evidence,
        input_delivery_slot=input_delivery_slot,
        control_directory=control_directory,
        result_directory=result_directory,
        temporary_directory=temporary_directory,
        result_file=result_file,
        credential_delivery_slot=credential_delivery_slot,
        workspace_proof=workspace_proof,
        layout_proof=layout_proof,
        inert_container_evidence=evidence,
    )


def _projection_with_mounts(projection, mounts):
    return DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=projection.projection_protocol_version,
        raw_field_schema_id=projection.raw_field_schema_id,
        execution_policy=projection.execution_policy,
        supervisor_helper_evidence=projection.supervisor_helper_evidence,
        docker_init_source_evidence=projection.docker_init_source_evidence,
        barrier_protocol_version=projection.barrier_protocol_version,
        barrier_poll_interval_seconds=projection.barrier_poll_interval_seconds,
        command_executable=projection.command_executable,
        command_arguments=projection.command_arguments,
        mounts=mounts,
        exact_mount_count=len(mounts) + 1,
        unclassified_raw_field_count=projection.unclassified_raw_field_count,
        nonauthoritative_raw_field_count=(projection.nonauthoritative_raw_field_count),
    )


def _evidence_with_projection(evidence, projection):
    return RunActionInertContainerEvidence.mint(
        **{
            key: value
            for key, value in evidence.to_dict().items()
            if key
            not in {
                "inert_container_evidence_id",
                "issued_create_projection",
                "observed_inspect_projection",
            }
        },
        issued_create_projection=projection,
        observed_inspect_projection=projection,
    )


def _remint_contract(contract, **changes):
    values = {
        key: value
        for key, value in contract.to_dict().items()
        if key != contract.IDENTITY_FIELD
    }
    values.update(changes)
    return type(contract).mint(**values)


def _volume_with_added_blocks(
    evidence,
    added_block_count,
    *,
    added_inode_count=0,
):
    block_size = evidence.allocation_block_size_bytes
    return _remint_contract(
        evidence,
        used_block_count=evidence.used_block_count + added_block_count,
        used_size_bytes=(evidence.used_size_bytes + added_block_count * block_size),
        available_block_count=(evidence.available_block_count - added_block_count),
        available_size_bytes=(
            evidence.available_size_bytes - added_block_count * block_size
        ),
        used_inode_count=evidence.used_inode_count + added_inode_count,
        available_inode_count=evidence.available_inode_count - added_inode_count,
    )


def test_prepared_execution_round_trips_with_complete_content_identity():
    prepared = _prepared_execution()
    layout = prepared.layout_proof

    assert (
        RunActionPreparedExecution.from_json_bytes(prepared.to_json_bytes()) == prepared
    )
    assert (
        RunActionRuntimeVolumeLayoutProof.from_json_bytes(layout.to_json_bytes())
        == layout
    )
    assert layout.prepared_delivery_slot_ids == tuple(
        sorted(
            (
                prepared.input_delivery_slot.prepared_delivery_slot_id,
                prepared.credential_delivery_slot.prepared_delivery_slot_id,
            )
        )
    )
    assert layout.prepared_runtime_directory_ids == tuple(
        sorted(
            (
                prepared.control_directory.prepared_runtime_directory_id,
                prepared.result_directory.prepared_runtime_directory_id,
                prepared.temporary_directory.prepared_runtime_directory_id,
            )
        )
    )
    assert layout.prepared_result_file_id == prepared.result_file.prepared_file_id
    assert prepared.result_file.prepared_parent_directory_id == (
        prepared.result_directory.prepared_runtime_directory_id
    )
    assert (
        RunActionPreparedRuntimeDirectory.from_json_bytes(
            prepared.result_directory.to_json_bytes()
        )
        == prepared.result_directory
    )
    assert prepared.prepared_execution_id.startswith(
        "run-action-prepared-execution:sha256:"
    )
    assert (
        prepared.preparation_claim.execution_policy.image_authority.image_authority_id
    )
    assert {
        "RUN_ACTION_BARRIER_CONTROL_DESTINATION",
        "RUN_ACTION_BARRIER_PROTOCOL_VERSION",
        "RUN_ACTION_DOCKER_INIT_DESTINATION",
        "RunActionActivatedRuntimeDirectoryObservation",
        "RunActionDockerInitSourceEvidence",
        "RunActionPreparedRuntimeDirectory",
        "RunActionPreparedRuntimeDirectoryKind",
        "RunActionSupervisorHelperEvidence",
        "run_action_activated_volume_evidence_matches",
        "run_action_docker_init_authority_id",
        "run_action_supervisor_helper_authority_id",
    }.issubset(supervisor_contracts.__all__)


def test_activation_revalidation_binds_fresh_exact_prepared_observations():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    request_blob = prepared.preparation_claim.reservation.request_blob
    input_observation = _activated_file_observation(
        prepared.input_delivery_slot,
        spawn,
        size_bytes=request_blob.size_bytes,
        content_digest=request_blob.digest,
        content_authority_id=request_blob.request_blob_id,
    )
    result_observation = _activated_file_observation(
        prepared.result_file,
        spawn,
        parent_authority=prepared.result_directory,
        size_bytes=0,
        content_digest=None,
        content_authority_id=None,
    )
    credential_observation = _activated_file_observation(
        prepared.credential_delivery_slot,
        spawn,
        size_bytes=32,
        content_digest=None,
        content_authority_id="test.credential.lease",
    )
    reobserved_volume = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        2,
        added_inode_count=2,
    )
    receipt = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=reobserved_volume,
        reobserved_keeper_evidence=prepared.volume_keeper_evidence,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=(
            None
            if prepared.workspace_proof is None
            else _activated_workspace_observation(prepared, spawn)
        ),
        activated_runtime_directory_observations=(
            _activated_runtime_directory_observations(prepared, spawn)
        ),
        activated_sentinel_observation=_activated_sentinel_observation(
            prepared,
            spawn,
        ),
        input_file_observation=input_observation,
        result_file_observation=result_observation,
        credential_file_observation=credential_observation,
    )

    assert (
        RunActionActivationRevalidationReceipt.from_json_bytes(receipt.to_json_bytes())
        == receipt
    )
    assert receipt.credential_file_observation.content_digest is None
    activated_file_fields = {field.name for field in fields(type(input_observation))}
    assert {
        "prepared_delivery_slot_id",
        "delivery_slot_mount_id",
        "delivery_slot_device",
        "delivery_slot_inode",
    }.isdisjoint(activated_file_fields)
    assert run_action_activated_volume_evidence_matches(
        prepared=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=receipt.reobserved_volume_evidence,
        activated_workspace_observation=(receipt.activated_workspace_observation),
        activated_runtime_directory_observations=(
            receipt.activated_runtime_directory_observations
        ),
        activated_sentinel_observation=receipt.activated_sentinel_observation,
        input_file_observation=receipt.input_file_observation,
        result_file_observation=receipt.result_file_observation,
        credential_file_observation=receipt.credential_file_observation,
    )
    foreign_prepared = _prepared_execution(
        claim=_claim(request_digest=tree_or_blob_digest(b"another request"))
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, spawn_commit=_spawn_commit(foreign_prepared))
    with pytest.raises(
        ContractValidationError,
        match="must be an object",
    ):
        replace(
            receipt,
            activated_workspace_observation=prepared.workspace_proof,
        )
    wrong_spawn = _spawn_commit(
        prepared,
        invocation_nonce="2" * 32,
    )
    wrong_spawn_workspace = _activated_workspace_observation(
        prepared,
        wrong_spawn,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_workspace_observation=wrong_spawn_workspace,
        )
    substituted_workspace_inode = _remint_contract(
        receipt.activated_workspace_observation,
        inode=receipt.activated_workspace_observation.inode + 1,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_workspace_observation=substituted_workspace_inode,
        )
    wrong_spawn_runtime_directories = _activated_runtime_directory_observations(
        prepared,
        wrong_spawn,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_runtime_directory_observations=wrong_spawn_runtime_directories,
        )
    substituted_control_inode = _remint_contract(
        receipt.activated_runtime_directory_observations[0],
        inode=receipt.activated_runtime_directory_observations[0].inode + 1,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_runtime_directory_observations=(
                substituted_control_inode,
                receipt.activated_runtime_directory_observations[1],
            ),
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="invalid or nonempty",
    ):
        replace(
            receipt.activated_runtime_directory_observations[0],
            observed_entry_count=1,
        )
    moved_sentinel = _remint_contract(
        receipt.activated_sentinel_observation,
        inode=receipt.activated_sentinel_observation.inode + 1,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_sentinel_observation=moved_sentinel,
        )
    wrong_spawn_sentinel = _activated_sentinel_observation(
        prepared,
        wrong_spawn,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            activated_sentinel_observation=wrong_spawn_sentinel,
        )
    wrong_spawn_delivery = _remint_contract(
        receipt.input_file_observation,
        spawn_commit_id=wrong_spawn.spawn_commit_id,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, input_file_observation=wrong_spawn_delivery)
    substituted_delivery = _remint_contract(
        receipt.input_file_observation,
        content_digest=tree_or_blob_digest(b"same-size-substitute"),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, input_file_observation=substituted_delivery)
    foreign_delivery = _remint_contract(
        receipt.input_file_observation,
        prepared_parent_authority_id=(
            foreign_prepared.input_delivery_slot.prepared_delivery_slot_id
        ),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, input_file_observation=foreign_delivery)
    substituted_delivery_parent = _remint_contract(
        receipt.input_file_observation,
        parent_inode=receipt.input_file_observation.parent_inode + 1,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(
            receipt,
            input_file_observation=substituted_delivery_parent,
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="carries a prepared file",
    ):
        _remint_contract(
            receipt.input_file_observation,
            prepared_file_id=prepared.result_file.prepared_file_id,
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="another namespace",
    ):
        _remint_contract(
            receipt.result_file_observation,
            prepared_parent_authority_id=(
                prepared.input_delivery_slot.prepared_delivery_slot_id
            ),
        )
    foreign_result_parent = _remint_contract(
        receipt.result_file_observation,
        prepared_parent_authority_id=(
            foreign_prepared.result_directory.prepared_runtime_directory_id
        ),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, result_file_observation=foreign_result_parent)
    substituted_result_parent = _remint_contract(
        receipt.result_file_observation,
        parent_inode=receipt.result_file_observation.parent_inode + 1,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, result_file_observation=substituted_result_parent)

    with pytest.raises(RunActionSupervisorContractError, match="file observation"):
        replace(
            receipt.input_file_observation,
            file_type="symlink",
        )
    incomplete_usage = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        1,
        added_inode_count=2,
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=incomplete_usage,
        )
    unexplained_extra_block = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        3,
        added_inode_count=2,
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=unexplained_extra_block,
        )
    unexplained_extra_inode = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        2,
        added_inode_count=3,
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=unexplained_extra_inode,
        )
    colliding_delivery = _remint_contract(
        receipt.input_file_observation,
        inode=prepared.result_file.inode,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        _remint_contract(
            receipt,
            input_file_observation=colliding_delivery,
        )
    limits = prepared.preparation_claim.execution_policy.docker_resource_limits
    block_size = reobserved_volume.allocation_block_size_bytes
    remaining_bytes = (
        prepared.result_file.payload_size_limit_bytes
        + limits.runtime_temporary_reservation_size_bytes
    )
    exact_exhaustion = (
        reobserved_volume.effective_size_bytes
        - ((remaining_bytes + block_size - 1) // block_size) * block_size
    )
    exhausted_usage = _remint_contract(
        reobserved_volume,
        used_block_count=exact_exhaustion // block_size,
        used_size_bytes=exact_exhaustion,
        available_block_count=(
            reobserved_volume.effective_block_count - exact_exhaustion // block_size
        ),
        available_size_bytes=(
            reobserved_volume.effective_size_bytes - exact_exhaustion
        ),
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=exhausted_usage,
        )


def test_activation_revalidation_requires_exact_live_volume_generation_and_keeper():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    request_blob = prepared.preparation_claim.reservation.request_blob
    values = {
        "prepared_execution": prepared,
        "spawn_commit": spawn,
        "reobserved_volume_evidence": _volume_with_added_blocks(
            prepared.runtime_volume_evidence,
            2,
            added_inode_count=2,
        ),
        "reobserved_keeper_evidence": prepared.volume_keeper_evidence,
        "reobserved_container_evidence": prepared.inert_container_evidence,
        "activated_workspace_observation": _activated_workspace_observation(
            prepared,
            spawn,
        ),
        "activated_runtime_directory_observations": (
            _activated_runtime_directory_observations(prepared, spawn)
        ),
        "activated_sentinel_observation": _activated_sentinel_observation(
            prepared,
            spawn,
        ),
        "input_file_observation": _activated_file_observation(
            prepared.input_delivery_slot,
            spawn,
            size_bytes=request_blob.size_bytes,
            content_digest=request_blob.digest,
            content_authority_id=request_blob.request_blob_id,
        ),
        "result_file_observation": _activated_file_observation(
            prepared.result_file,
            spawn,
            parent_authority=prepared.result_directory,
            size_bytes=0,
            content_digest=None,
            content_authority_id=None,
        ),
        "credential_file_observation": _activated_file_observation(
            prepared.credential_delivery_slot,
            spawn,
            size_bytes=32,
            content_digest=None,
            content_authority_id="test.credential.lease",
        ),
    }
    wrong_generation = _remint_contract(
        values["input_file_observation"],
        generation_nonce="f" * 32,
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        RunActionActivationRevalidationReceipt.mint(
            **values | {"input_file_observation": wrong_generation}
        )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        RunActionActivationRevalidationReceipt.mint(
            **values
            | {
                "reobserved_keeper_evidence": _prepared_execution(
                    inode_offset=8
                ).volume_keeper_evidence
            }
        )
    recycled_process_volume = _remint_contract(
        values["reobserved_volume_evidence"],
        keeper_process_start_time_ticks=(
            prepared.runtime_volume_evidence.keeper_process_start_time_ticks + 1
        ),
    )
    with pytest.raises(RunActionSupervisorContractError, match="prepared authority"):
        RunActionActivationRevalidationReceipt.mint(
            **values | {"reobserved_volume_evidence": recycled_process_volume}
        )


def test_activation_revalidation_uses_absence_for_credential_free_policy():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    spawn = _spawn_commit(prepared)
    request_blob = prepared.preparation_claim.reservation.request_blob
    receipt = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=_volume_with_added_blocks(
            prepared.runtime_volume_evidence,
            1,
            added_inode_count=1,
        ),
        reobserved_keeper_evidence=prepared.volume_keeper_evidence,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=None,
        activated_runtime_directory_observations=(
            _activated_runtime_directory_observations(prepared, spawn)
        ),
        activated_sentinel_observation=_activated_sentinel_observation(
            prepared,
            spawn,
        ),
        input_file_observation=_activated_file_observation(
            prepared.input_delivery_slot,
            spawn,
            size_bytes=request_blob.size_bytes,
            content_digest=request_blob.digest,
            content_authority_id=request_blob.request_blob_id,
        ),
        result_file_observation=_activated_file_observation(
            prepared.result_file,
            spawn,
            parent_authority=prepared.result_directory,
            size_bytes=0,
            content_digest=None,
            content_authority_id=None,
        ),
        credential_file_observation=None,
    )

    assert receipt.credential_file_observation is None
    assert receipt.activated_workspace_observation is None
    foreign_credential_slot = _prepared_execution().credential_delivery_slot
    forbidden_credential = _activated_file_observation(
        foreign_credential_slot,
        spawn,
        size_bytes=32,
        content_digest=None,
        content_authority_id="test.credential.lease",
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=_volume_with_added_blocks(
                prepared.runtime_volume_evidence,
                2,
                added_inode_count=2,
            ),
            credential_file_observation=forbidden_credential,
        )
    with pytest.raises(ContractValidationError, match="must be an object"):
        replace(
            receipt,
            activated_workspace_observation=_prepared_execution().workspace_proof,
        )


def _spawn_commit(prepared, *, invocation_nonce="1" * 32):
    reservation = prepared.preparation_claim.reservation
    return RunActionSpawnCommit.mint(
        reservation_id=reservation.reservation_id,
        prepared_execution_id=prepared.prepared_execution_id,
        provider_execution_id=prepared.inert_container_evidence.container_id,
        invocation_nonce=invocation_nonce,
        security_observation_id=reservation.frontier.security_observation_id,
        boundary_identity=reservation.intent.boundary_identity,
    )


def test_terminal_observation_and_result_capture_bind_the_physical_result():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    payload = b'{"provider":"complete"}'
    activation = _activation_revalidation_receipt(prepared, spawn)
    terminal = _terminal_observation(prepared, spawn)
    capture = _result_capture_receipt(prepared, activation, terminal, payload)
    empty_capture = _result_capture_receipt(
        prepared,
        activation,
        terminal,
        b"",
    )

    assert (
        RunActionTerminalObservation.from_json_bytes(terminal.to_json_bytes())
        == terminal
    )
    assert (
        RunActionResultCaptureReceipt.from_json_bytes(capture.to_json_bytes())
        == capture
    )
    assert capture.terminal_observation_id == terminal.terminal_observation_id
    assert capture.prepared_parent_authority_id == (
        prepared.result_directory.prepared_runtime_directory_id
    )
    assert capture.parent_inode == prepared.result_directory.inode
    assert (
        capture.reobserved_volume_evidence.root_inode
        == prepared.runtime_volume_evidence.root_inode
    )
    assert empty_capture.size_bytes == 0
    assert empty_capture.content_digest == tree_or_blob_digest(b"")
    with pytest.raises(
        RunActionSupervisorContractError,
        match="result capture receipt is invalid",
    ):
        _remint_contract(
            empty_capture,
            content_digest=tree_or_blob_digest(b"not empty"),
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="terminal observation is invalid",
    ):
        replace(terminal, paused=True)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="result capture receipt is invalid",
    ):
        replace(
            capture,
            inode=prepared.runtime_volume_evidence.sentinel_evidence.inode,
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="result capture receipt is invalid",
    ):
        replace(capture, parent_inode=capture.inode)


def _activated_workspace_observation(prepared, spawn):
    workspace = prepared.workspace_proof
    if workspace is None:
        raise AssertionError("activated workspace fixture requires a workspace")
    return RunActionActivatedWorkspaceObservation.mint(
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_workspace_proof_id=workspace.prepared_workspace_proof_id,
        runtime_volume_authority_id=workspace.runtime_volume_authority_id,
        generation_nonce=workspace.generation_nonce,
        source_tree_digest=workspace.observed_source_tree_digest,
        git_closure_digest=workspace.observed_git_closure_digest,
        source_entry_count=workspace.observed_source_entry_count,
        source_size_bytes=workspace.observed_source_size_bytes,
        owner_user_id=workspace.owner_user_id,
        owner_group_id=workspace.owner_group_id,
        root_mode=workspace.root_mode,
        mount_id=workspace.mount_id,
        device=workspace.device,
        inode=workspace.inode,
    )


def _activated_runtime_directory_observations(prepared, spawn):
    return tuple(
        RunActionActivatedRuntimeDirectoryObservation.mint(
            spawn_commit_id=spawn.spawn_commit_id,
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


def _activated_sentinel_observation(prepared, spawn):
    sentinel = prepared.runtime_volume_evidence.sentinel_evidence
    return RunActionActivatedSentinelObservation.mint(
        spawn_commit_id=spawn.spawn_commit_id,
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


def _activated_file_observation(
    prepared_authority,
    spawn_commit,
    *,
    parent_authority=None,
    size_bytes,
    content_digest,
    content_authority_id,
):
    is_delivery = type(prepared_authority) is RunActionPreparedDeliverySlot
    parent_authority = prepared_authority if is_delivery else parent_authority
    if (
        is_delivery and type(parent_authority) is not RunActionPreparedDeliverySlot
    ) or (
        not is_delivery
        and type(parent_authority) is not RunActionPreparedRuntimeDirectory
    ):
        raise AssertionError("activated file fixture requires its exact parent")
    relative_path = (
        (
            f"{prepared_authority.directory_relative_path}/"
            f"{prepared_authority.final_file_name}"
        )
        if is_delivery
        else prepared_authority.relative_path
    )
    return RunActionActivatedFileObservation.mint(
        spawn_commit_id=spawn_commit.spawn_commit_id,
        prepared_parent_authority_id=(
            parent_authority.prepared_delivery_slot_id
            if is_delivery
            else parent_authority.prepared_runtime_directory_id
        ),
        prepared_file_id=(None if is_delivery else prepared_authority.prepared_file_id),
        parent_mount_id=parent_authority.mount_id,
        parent_device=parent_authority.device,
        parent_inode=parent_authority.inode,
        runtime_volume_authority_id=prepared_authority.runtime_volume_authority_id,
        generation_nonce=prepared_authority.generation_nonce,
        kind=prepared_authority.kind,
        relative_path=relative_path,
        file_type="regular",
        owner_user_id=prepared_authority.owner_user_id,
        owner_group_id=prepared_authority.owner_group_id,
        mode=(
            0o600
            if prepared_authority.kind is RunActionPreparedFileKind.RESULT
            else 0o400
        ),
        link_count=1,
        mount_id=prepared_authority.mount_id,
        device=prepared_authority.device,
        inode=(
            prepared_authority.inode + 100_000
            if is_delivery
            else prepared_authority.inode
        ),
        size_bytes=size_bytes,
        content_digest=content_digest,
        content_authority_id=content_authority_id,
    )


def _terminal_observation(prepared, spawn, workload_release_adoption=None):
    activation_receipt = _activation_revalidation_receipt(prepared, spawn)
    workload_release_adoption_id = (
        content_id(
            "run-action-workload-release-adoption",
            {"fixture": "terminal release adoption"},
        )
        if workload_release_adoption is None
        else workload_release_adoption.workload_release_adoption_id
    )
    started_at = (
        "2026-01-01T00:00:00Z"
        if workload_release_adoption is None
        else workload_release_adoption.workload_release_receipt.resolved_workload_observation.running_container_observation.started_at
    )
    return RunActionTerminalObservation.mint(
        prepared_execution_id=prepared.prepared_execution_id,
        spawn_commit_id=spawn.spawn_commit_id,
        provider_execution_id=spawn.provider_execution_id,
        runtime_volume_authority_id=(
            prepared.runtime_volume_authority.runtime_volume_authority_id
        ),
        generation_nonce=prepared.runtime_volume_authority.generation_nonce,
        activation_revalidation_receipt_id=(
            activation_receipt.activation_revalidation_receipt_id
        ),
        workload_release_adoption_id=workload_release_adoption_id,
        observed_inspect_projection=(
            prepared.inert_container_evidence.observed_inspect_projection
        ),
        complete_inspection_digest=tree_or_blob_digest(b"terminal inspection"),
        container_status="exited",
        process_id=0,
        restart_count=0,
        paused=False,
        restarting=False,
        dead=False,
        started_at=started_at,
        finished_at="2026-07-25T01:02:04.123456789Z",
        exit_code=0,
        oom_killed=False,
        state_error="",
    )


def _result_capture_receipt(prepared, activation, terminal, payload):
    result_file = prepared.result_file
    activation_volume = activation.reobserved_volume_evidence
    result_block_count = (
        len(payload) + activation_volume.allocation_block_size_bytes - 1
    ) // activation_volume.allocation_block_size_bytes
    volume = _volume_with_added_blocks(
        activation_volume,
        result_block_count,
    )
    return RunActionResultCaptureReceipt.mint(
        terminal_observation_id=terminal.terminal_observation_id,
        prepared_parent_authority_id=(
            prepared.result_directory.prepared_runtime_directory_id
        ),
        prepared_file_id=result_file.prepared_file_id,
        parent_mount_id=prepared.result_directory.mount_id,
        parent_device=prepared.result_directory.device,
        parent_inode=prepared.result_directory.inode,
        runtime_volume_authority_id=result_file.runtime_volume_authority_id,
        reobserved_volume_evidence=volume,
        prepared_sentinel_evidence_id=(
            volume.sentinel_evidence.runtime_volume_sentinel_evidence_id
        ),
        generation_nonce=result_file.generation_nonce,
        relative_path=result_file.relative_path,
        file_type=result_file.file_type,
        owner_user_id=result_file.owner_user_id,
        owner_group_id=result_file.owner_group_id,
        mode=result_file.mode,
        link_count=result_file.link_count,
        size_bytes=len(payload),
        content_digest=tree_or_blob_digest(payload),
        mount_id=volume.root_mount_id,
        device=volume.root_device,
        inode=result_file.inode,
    )


def _activation_revalidation_receipt(prepared, spawn):
    request_blob = prepared.preparation_claim.reservation.request_blob
    block_size = prepared.runtime_volume_evidence.allocation_block_size_bytes
    delivered_sizes = [request_blob.size_bytes]
    credential_observation = None
    if prepared.credential_delivery_slot is not None:
        delivered_sizes.append(32)
        credential_observation = _activated_file_observation(
            prepared.credential_delivery_slot,
            spawn,
            size_bytes=32,
            content_digest=None,
            content_authority_id="test.credential.lease",
        )
    delivered_block_count = sum(
        (size_bytes + block_size - 1) // block_size for size_bytes in delivered_sizes
    )
    return RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=_volume_with_added_blocks(
            prepared.runtime_volume_evidence,
            delivered_block_count,
            added_inode_count=len(delivered_sizes),
        ),
        reobserved_keeper_evidence=prepared.volume_keeper_evidence,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=(
            None
            if prepared.workspace_proof is None
            else _activated_workspace_observation(prepared, spawn)
        ),
        activated_runtime_directory_observations=(
            _activated_runtime_directory_observations(prepared, spawn)
        ),
        activated_sentinel_observation=_activated_sentinel_observation(
            prepared,
            spawn,
        ),
        input_file_observation=_activated_file_observation(
            prepared.input_delivery_slot,
            spawn,
            size_bytes=request_blob.size_bytes,
            content_digest=request_blob.digest,
            content_authority_id=request_blob.request_blob_id,
        ),
        result_file_observation=_activated_file_observation(
            prepared.result_file,
            spawn,
            parent_authority=prepared.result_directory,
            size_bytes=0,
            content_digest=None,
            content_authority_id=None,
        ),
        credential_file_observation=credential_observation,
    )


def test_semantic_claim_changes_with_request_or_execution_policy():
    original = _claim()
    changed_request = _claim(request_digest=tree_or_blob_digest(b"another request"))
    changed_policy = _claim(
        policy=_execution_policy(command_template_label="another-command")
    )

    assert original.preparation_claim_id != changed_request.preparation_claim_id
    assert original.preparation_claim_id != changed_policy.preparation_claim_id


def test_preparation_allocation_rejects_authority_not_exactly_derived_from_claim():
    claim = _claim()
    authority = issue_runtime_volume_authority(claim, "a" * 32)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
    )
    assert (
        RunActionPreparationAllocation.from_json_bytes(allocation.to_json_bytes())
        == allocation
    )
    substituted_authority = _remint_contract(
        authority,
        labels=tuple(
            supervisor_contracts.RunActionContainerLabel(
                key=label.key,
                value=(
                    _fixture_content_id("run-action-reservation", "foreign")
                    if label.key == "com.kapso.run-action.reservation"
                    else label.value
                ),
            )
            for label in authority.labels
        ),
    )

    with pytest.raises(
        RunActionSupervisorContractError,
        match="differs from its exact claim",
    ):
        RunActionPreparationAllocation.mint(
            preparation_claim=claim,
            runtime_volume_authority=substituted_authority,
        )


def test_runtime_volume_authority_rejects_a_foreign_generation_label():
    claim = _claim()
    authority = issue_runtime_volume_authority(claim, "a" * 32)
    labels = tuple(
        supervisor_contracts.RunActionContainerLabel(
            key=label.key,
            value=(
                runtime_volume_sentinel_identity("b" * 32)
                if label.key == "com.kapso.run-action.generation"
                else label.value
            ),
        )
        for label in authority.labels
    )

    with pytest.raises(
        RunActionSupervisorContractError,
        match="runtime volume authority is invalid",
    ):
        _remint_contract(authority, labels=labels)


def test_claim_embeds_one_complete_reservation_and_rejects_cross_kind_splicing():
    coding_claim = _claim()
    embedding_spec = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    embedding_claim = _claim(policy=embedding_spec)

    with pytest.raises(
        RunActionSupervisorContractError,
        match="durable reservation",
    ):
        RunActionPreparationClaim.mint(
            reservation=coding_claim.reservation,
            execution_policy=embedding_spec,
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="durable reservation",
    ):
        RunActionPreparationClaim.mint(
            reservation=embedding_claim.reservation,
            execution_policy=coding_claim.execution_policy,
        )
    assert coding_claim.reservation.intent.boundary_identity == _boundary(
        execution_policy_id=(coding_claim.execution_policy.docker_execution_policy_id)
    )
    assert (
        coding_claim.reservation.request_blob.digest
        == coding_claim.reservation.intent.request_digest
    )


def test_claim_rejects_same_kind_execution_lifecycle_substitution():
    policy = _execution_policy()
    boundary = _boundary(
        execution_policy_id=policy.docker_execution_policy_id,
    )
    substituted_lifecycle = RunActionExecutionLifecycleIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        implementation_id="test.coding_agent.execution.substituted",
        implementation_version="test.execution.v2",
        recovery_protocol_version="test.recovery.v1",
        execution_policy_id=_fixture_content_id(
            "docker-run-action-execution-policy",
            "substituted",
        ),
    )
    substituted_boundary = RunActionBoundaryIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        execution_lifecycle_identity=substituted_lifecycle,
        result_interpreter_identity=boundary.result_interpreter_identity,
    )

    with pytest.raises(
        RunActionSupervisorContractError,
        match="durable reservation",
    ):
        _claim(policy=policy, boundary=substituted_boundary)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("container_status", "running"),
        ("process_id", 1),
        ("restart_count", 1),
        ("started_at", "2026-01-01T00:00:00Z"),
        ("restart_policy_name", "always"),
        ("auto_remove", True),
        ("network_mode", "bridge"),
        ("healthcheck_present", True),
        ("volume_plugin_mount_count", 1),
        ("docker_socket_mounted", True),
    ),
)
def test_inert_evidence_rejects_any_started_or_expandable_resource_fact(
    field_name,
    value,
):
    evidence = _prepared_execution().inert_container_evidence

    with pytest.raises(RunActionSupervisorContractError, match="exact inert"):
        replace(evidence, **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("read_only_root_filesystem", False),
        ("privileged", True),
        ("capability_additions", ("SYS_ADMIN",)),
        ("capability_drops", ()),
        ("device_authority_ids", ("test.gpu",)),
        ("device_request_authority_ids", ("test.gpu.request",)),
        ("device_cgroup_rule_ids", ("test.gpu.rule",)),
        ("supplementary_group_ids", (7,)),
        ("pid_namespace_mode", "host"),
        ("sysctl_ids", ("net.ipv4.ip_forward=1",)),
        ("no_new_privileges", False),
        ("security_option_ids", ()),
        ("masked_system_paths", ("/proc/kcore",)),
        ("read_only_system_paths", ("/proc/sys",)),
        ("log_driver", "json-file"),
        ("init_process", False),
        ("cgroup_parent_id", "test.kapso.run_action"),
        ("user_namespace_mode", "private"),
        ("seccomp_profile_id", "unconfined"),
        ("apparmor_profile_id", "unconfined"),
        ("runtime_id", "crun"),
    ),
)
def test_sandbox_spec_rejects_every_privilege_expansion(field_name, value):
    sandbox = _execution_policy().sandbox_spec

    with pytest.raises(
        RunActionSupervisorContractError,
        match="expanded privilege",
    ):
        replace(sandbox, **{field_name: value})


def _remint_sandbox(sandbox, **changes):
    values = {
        key: value
        for key, value in sandbox.to_dict().items()
        if key != "docker_sandbox_spec_id"
    }
    values.update(changes)
    return DockerRunActionSandboxSpec.mint(**values)


def _remint_policy(policy, **changes):
    values = {
        key: value
        for key, value in policy.to_dict().items()
        if key != "docker_execution_policy_id"
    }
    values.update(changes)
    return DockerRunActionExecutionPolicy.mint(**values)


def _remint_resource_limits(resource_limits, **changes):
    values = {
        key: value
        for key, value in resource_limits.to_dict().items()
        if key != "docker_resource_limits_id"
    }
    values.update(changes)
    return DockerRunActionResourceLimits.mint(**values)


@pytest.mark.parametrize("delimiter", (",", "\r", "\n", '"'))
def test_filesystem_policy_rejects_docker_mount_delimiters(delimiter):
    filesystem = _execution_policy().filesystem_policy
    values = {
        key: value
        for key, value in filesystem.to_dict().items()
        if key != "filesystem_policy_id"
    }
    values["input_destination"] = f"/kapso/input{delimiter}readonly"

    with pytest.raises(
        RunActionSupervisorContractError,
        match="normalized and absolute",
    ):
        RunActionFilesystemPolicy.mint(**values)


@pytest.mark.parametrize(
    ("field_name", "destination"),
    tuple(
        (field_name, destination)
        for field_name in (
            "workspace_destination",
            "input_destination",
            "result_destination",
            "credential_destination",
            "temporary_filesystem_destination",
        )
        for destination in (
            "/sbin",
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            f"{RUN_ACTION_DOCKER_INIT_DESTINATION}/nested",
        )
    ),
)
def test_filesystem_policy_rejects_docker_init_mount_collisions(
    field_name,
    destination,
):
    filesystem = _execution_policy().filesystem_policy
    values = {
        key: value
        for key, value in filesystem.to_dict().items()
        if key != "filesystem_policy_id"
    }
    values[field_name] = destination

    with pytest.raises(
        RunActionSupervisorContractError,
        match="mount destinations overlap",
    ):
        RunActionFilesystemPolicy.mint(**values)


@pytest.mark.parametrize("delimiter", (",", "\r", "\n", '"'))
def test_supervisor_helper_authority_rejects_docker_mount_delimiters(delimiter):
    with pytest.raises(
        RunActionSupervisorContractError,
        match="normalized and absolute",
    ):
        run_action_supervisor_helper_authority_id(
            f"/usr/bin/busybox{delimiter}readonly",
            tree_or_blob_digest(b"helper"),
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="normalized and absolute",
    ):
        run_action_docker_init_authority_id(
            f"/usr/bin/docker-init{delimiter}readonly",
            tree_or_blob_digest(b"init"),
        )


def test_lifecycle_policy_binding_rejects_alternate_valid_sandbox():
    claim = _claim()
    sandbox = _remint_sandbox(
        claim.execution_policy.sandbox_spec,
        cgroup_parent_id="test.kapso.alternate.slice",
    )
    substituted_policy = _remint_policy(
        claim.execution_policy,
        sandbox_spec=sandbox,
    )

    with pytest.raises(RunActionSupervisorContractError, match="durable reservation"):
        RunActionPreparationClaim.mint(
            reservation=claim.reservation,
            execution_policy=substituted_policy,
        )


def test_lifecycle_policy_binding_pins_the_supervisor_helper_bytes():
    claim = _claim()
    alternate_digest = tree_or_blob_digest(b"alternate keeper helper")
    substituted_policy = _remint_policy(
        claim.execution_policy,
        supervisor_helper_executable_authority_id=(
            run_action_supervisor_helper_authority_id(
                claim.execution_policy.supervisor_helper_source_path,
                alternate_digest,
            )
        ),
        supervisor_helper_executable_digest=alternate_digest,
    )

    assert (
        substituted_policy.docker_execution_policy_id
        != claim.execution_policy.docker_execution_policy_id
    )
    with pytest.raises(RunActionSupervisorContractError, match="durable reservation"):
        RunActionPreparationClaim.mint(
            reservation=claim.reservation,
            execution_policy=substituted_policy,
        )
    with pytest.raises(RunActionSupervisorContractError, match="supervisor helper"):
        _remint_policy(
            claim.execution_policy,
            supervisor_helper_executable_digest=alternate_digest,
        )


def test_lifecycle_policy_binding_pins_the_docker_init_bytes():
    claim = _claim()
    alternate_digest = tree_or_blob_digest(b"alternate Docker init")
    substituted_policy = _remint_policy(
        claim.execution_policy,
        docker_init_executable_authority_id=run_action_docker_init_authority_id(
            claim.execution_policy.docker_init_source_path,
            alternate_digest,
        ),
        docker_init_executable_digest=alternate_digest,
    )

    assert (
        substituted_policy.docker_execution_policy_id
        != claim.execution_policy.docker_execution_policy_id
    )
    with pytest.raises(RunActionSupervisorContractError, match="durable reservation"):
        RunActionPreparationClaim.mint(
            reservation=claim.reservation,
            execution_policy=substituted_policy,
        )
    with pytest.raises(RunActionSupervisorContractError, match="Docker init"):
        _remint_policy(
            claim.execution_policy,
            docker_init_executable_digest=alternate_digest,
        )


def test_execution_policy_exposes_only_renderable_resource_degrees_of_freedom():
    assert {
        "nano_cpus",
        "cpu_realtime_period_microseconds",
        "cpu_realtime_runtime_microseconds",
        "memory_swappiness_percentage",
        "oom_kill_disabled",
        "block_io_read_bandwidth_rule_ids",
        "block_io_write_bandwidth_rule_ids",
        "block_io_read_iops_rule_ids",
        "block_io_write_iops_rule_ids",
        "ulimits",
        "open_file_limit",
        "temporary_filesystem_size_bytes",
    }.isdisjoint({field.name for field in fields(DockerRunActionResourceLimits)})
    assert {"stdout_size_bytes", "stderr_size_bytes"}.isdisjoint(
        {field.name for field in fields(RunActionSupervisorLimits)}
    )
    assert "timeout_directive_size_bytes" in {
        field.name for field in fields(RunActionSupervisorLimits)
    }
    assert "safe_create_defaults" in {
        field.name for field in fields(DockerRunActionExecutionPolicy)
    }
    assert tuple(RunActionActivationNetworkMode) == (
        RunActionActivationNetworkMode.NONE,
    )


def test_supervisor_timeout_directive_bound_is_positive_and_config_sourced():
    limits = _execution_policy().supervisor_limits

    assert (
        limits.timeout_directive_size_bytes == _RUN_ACTION_TIMEOUT_DIRECTIVE_SIZE_BYTES
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="supervisor limits",
    ):
        _remint_contract(limits, timeout_directive_size_bytes=0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("cpu_quota_microseconds", 999),
        ("memory_size_bytes", 6 * 1024 * 1024 - 1),
        ("memory_reservation_size_bytes", 6 * 1024 * 1024 - 1),
    ),
)
def test_resource_limits_reject_values_below_docker_boundaries(field_name, value):
    policy = _execution_policy()

    with pytest.raises(
        RunActionSupervisorContractError,
        match="resource limits",
    ):
        _remint_resource_limits(
            policy.docker_resource_limits,
            **{field_name: value},
        )


def test_resource_limits_accept_exact_docker_boundaries():
    limits = _remint_resource_limits(
        _execution_policy().docker_resource_limits,
        cpu_quota_microseconds=1_000,
        memory_size_bytes=6 * 1024 * 1024,
        memory_reservation_size_bytes=6 * 1024 * 1024,
        memory_swap_size_bytes=6 * 1024 * 1024,
    )

    assert limits.cpu_quota_microseconds == 1_000
    assert limits.memory_size_bytes == 6 * 1024 * 1024


@pytest.mark.parametrize(
    "cgroup_parent_id",
    (
        "test/kapso.slice",
        "-test.slice",
        "test-.slice",
        "test.slice/child",
        "foo--bar.slice",
        f"{'a' * 250}.slice",
    ),
)
def test_sandbox_rejects_noncanonical_systemd_slice(cgroup_parent_id):
    with pytest.raises(RunActionSupervisorContractError, match="expanded privilege"):
        _remint_sandbox(
            _execution_policy().sandbox_spec,
            cgroup_parent_id=cgroup_parent_id,
        )


def test_sandbox_accepts_maximum_length_systemd_slice():
    cgroup_parent_id = f"{'a' * 249}.slice"

    sandbox = _remint_sandbox(
        _execution_policy().sandbox_spec,
        cgroup_parent_id=cgroup_parent_id,
    )

    assert len(cgroup_parent_id.encode("ascii")) == 255
    assert sandbox.cgroup_parent_id == cgroup_parent_id


def test_keeper_cgroup_path_expands_systemd_slice_hierarchy():
    container_id = "a" * 64
    policy = _execution_policy()
    hierarchical_policy = _remint_policy(
        policy,
        sandbox_spec=_remint_sandbox(
            policy.sandbox_spec,
            cgroup_parent_id="kapso-workers-actions.slice",
        ),
    )

    assert run_action_keeper_process_cgroup_path(
        hierarchical_policy,
        container_id,
    ) == (
        "/kapso.slice/kapso-workers.slice/"
        f"kapso-workers-actions.slice/docker-{container_id}.scope"
    )


@pytest.mark.parametrize("field_name", ("user_id", "group_id"))
def test_execution_policy_rejects_identity_above_docker_boundary(field_name):
    policy = _execution_policy()

    assert getattr(
        _remint_policy(policy, **{field_name: 2_147_483_647}), field_name
    ) == (2_147_483_647)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="execution policy is invalid",
    ):
        _remint_policy(policy, **{field_name: 2_147_483_648})


def test_durable_policy_has_no_argv_or_request_secret_channel():
    secret_request = b"agent --api-key sk-live-secret"
    claim = _claim(request_payload=secret_request)
    serialized = claim.to_json_bytes()

    assert "argv" not in {
        field.name for field in fields(DockerRunActionExecutionPolicy)
    }
    assert secret_request not in serialized
    assert b"sk-live-secret" not in serialized
    assert (
        claim.execution_policy.command_template_id
        == _execution_policy().command_template_id
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("unclassified_raw_field_count", 1),
        ("projection_protocol_version", "kapso.unknown_projection.v2"),
    ),
)
def test_closed_projection_rejects_unclassified_or_wrong_version(
    field_name,
    value,
):
    projection = _prepared_execution().inert_container_evidence.issued_create_projection

    with pytest.raises(
        RunActionSupervisorContractError,
        match="incomplete or noncanonical",
    ):
        replace(projection, **{field_name: value})


def test_runtime_volume_binds_exact_docker_inspect_and_statfs_evidence():
    prepared = _prepared_execution()
    authority = prepared.runtime_volume_authority
    evidence = prepared.runtime_volume_evidence

    assert authority.volume_name == preparation_volume_name(prepared.preparation_claim)
    assert authority.labels == preparation_volume_labels(
        prepared.preparation_claim,
        authority.generation_nonce,
    )
    assert authority.driver_options == runtime_volume_driver_options(authority)
    assert "noswap" in authority.driver_options[1]
    assert evidence.observed_scope == "local"
    assert evidence.observed_volume_name == authority.volume_name
    assert evidence.observed_labels == authority.labels
    assert evidence.volume_keeper_evidence_id == (
        prepared.volume_keeper_evidence.volume_keeper_evidence_id
    )
    assert evidence.keeper_container_id == prepared.volume_keeper_evidence.container_id
    assert evidence.keeper_process_id == prepared.volume_keeper_evidence.process_id
    assert evidence.keeper_process_start_time_ticks == (
        prepared.volume_keeper_evidence.process_start_time_ticks
    )
    assert evidence.keeper_process_cgroup_path == (
        prepared.volume_keeper_evidence.mounted_helper_evidence.process_cgroup_path
    )
    assert evidence.root_mount_id == evidence.sentinel_evidence.mount_id
    assert evidence.root_device == evidence.sentinel_evidence.device
    assert evidence.root_inode != evidence.sentinel_evidence.inode
    assert evidence.used_size_bytes == prepared.layout_proof.observed_used_size_bytes
    assert evidence.used_inode_count == prepared.layout_proof.observed_used_inode_count
    assert evidence.used_size_bytes == (
        evidence.used_block_count * evidence.allocation_block_size_bytes
    )
    assert (
        evidence.used_block_count + evidence.available_block_count
        == evidence.effective_block_count
    )
    assert (
        evidence.used_size_bytes + evidence.available_size_bytes
        == evidence.effective_size_bytes
    )
    assert (
        evidence.used_inode_count + evidence.available_inode_count
        == evidence.effective_inode_limit
    )

    with pytest.raises(RunActionSupervisorContractError, match="authority is invalid"):
        replace(authority, noswap=False)
    with pytest.raises(RunActionSupervisorContractError, match="bounded tmpfs"):
        replace(
            evidence,
            available_block_count=evidence.available_block_count - 1,
        )
    with pytest.raises(RunActionSupervisorContractError, match="bounded tmpfs"):
        replace(evidence, observed_scope="global")
    with pytest.raises(RunActionSupervisorContractError, match="bounded tmpfs"):
        replace(evidence, observed_labels=())
    with pytest.raises(RunActionSupervisorContractError, match="bounded tmpfs"):
        replace(
            evidence,
            keeper_process_cgroup_path="/test/docker-" + "f" * 64 + ".scope",
        )
    with pytest.raises(RunActionSupervisorContractError, match="bounded tmpfs"):
        replace(evidence, root_device=evidence.root_device + 1)


def test_prepared_volume_rejects_keeper_and_layout_evidence_splices():
    prepared = _prepared_execution()
    foreign_prepared = _prepared_execution(inode_offset=100)
    evidence = prepared.runtime_volume_evidence
    substituted_container_id = "f" * 64
    substituted_evidence = _remint_contract(
        evidence,
        volume_keeper_evidence_id=_fixture_content_id(
            "run-action-volume-keeper-evidence",
            "substituted",
        ),
        keeper_container_id=substituted_container_id,
        keeper_process_id=evidence.keeper_process_id + 1,
        keeper_process_start_time_ticks=(evidence.keeper_process_start_time_ticks + 1),
        keeper_process_cgroup_path=(
            "/test.kapso.run_action.slice/" f"docker-{substituted_container_id}.scope"
        ),
    )

    with pytest.raises(
        RunActionSupervisorContractError,
        match="keeper differs from prepared authority",
    ):
        replace(prepared, runtime_volume_evidence=substituted_evidence)

    substituted_sentinel = _remint_contract(
        evidence.sentinel_evidence,
        mount_id=evidence.root_mount_id + 1,
        device=evidence.root_device + 1,
    )
    substituted_root_evidence = _remint_contract(
        evidence,
        root_mount_id=evidence.root_mount_id + 1,
        root_device=evidence.root_device + 1,
        sentinel_evidence=substituted_sentinel,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="artifacts differ from their preparation claim",
    ):
        replace(prepared, runtime_volume_evidence=substituted_root_evidence)
    substituted_layout = _remint_contract(
        prepared.layout_proof,
        runtime_volume_evidence_id=_fixture_content_id(
            "run-action-runtime-volume-evidence",
            "substituted",
        ),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="positive byte or inode headroom",
    ):
        replace(prepared, layout_proof=substituted_layout)

    substituted_slot_layout = _remint_contract(
        prepared.layout_proof,
        prepared_delivery_slot_ids=(
            foreign_prepared.layout_proof.prepared_delivery_slot_ids
        ),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="positive byte or inode headroom",
    ):
        replace(prepared, layout_proof=substituted_slot_layout)
    substituted_result_layout = _remint_contract(
        prepared.layout_proof,
        prepared_result_file_id=(foreign_prepared.layout_proof.prepared_result_file_id),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="positive byte or inode headroom",
    ):
        replace(prepared, layout_proof=substituted_result_layout)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="artifacts differ from their preparation claim",
    ):
        replace(
            prepared,
            input_delivery_slot=foreign_prepared.input_delivery_slot,
        )


def test_runtime_volume_occurrence_rejects_daemon_occurrence_splice():
    evidence = _prepared_execution().runtime_volume_evidence
    spliced = _remint_contract(
        evidence,
        docker_volume_occurrence_digest=tree_or_blob_digest(
            b"replacement Docker volume occurrence"
        ),
    )

    assert not run_action_runtime_volume_occurrence_matches(spliced, evidence)


def test_prepared_keeper_rejects_coordinated_wrong_cgroup_parent():
    prepared = _prepared_execution()
    keeper = prepared.volume_keeper_evidence
    wrong_cgroup_path = f"/wrong.slice/docker-{keeper.container_id}.scope"
    wrong_mounted_helper = _remint_contract(
        keeper.mounted_helper_evidence,
        process_cgroup_path=wrong_cgroup_path,
    )

    with pytest.raises(
        RunActionSupervisorContractError,
        match="not exact and running",
    ):
        _remint_contract(
            keeper,
            mounted_helper_evidence=wrong_mounted_helper,
        )


def test_runtime_volume_sentinel_is_one_stable_physical_file():
    prepared = _prepared_execution()
    sentinel = prepared.runtime_volume_evidence.sentinel_evidence

    assert sentinel.relative_path == ".kapso-generation"
    assert sentinel.file_type == "regular"
    assert sentinel.mode == 0o400
    assert sentinel.link_count == 1
    assert sentinel.content_digest == tree_or_blob_digest(
        sentinel.generation_nonce.encode("ascii")
    )
    assert (sentinel.mount_id, sentinel.device, sentinel.inode) == (
        1000,
        500,
        10000,
    )
    with pytest.raises(RunActionSupervisorContractError, match="stable physical file"):
        replace(sentinel, file_type="symlink")
    with pytest.raises(RunActionSupervisorContractError, match="stable physical file"):
        replace(sentinel, content_digest=tree_or_blob_digest(b"substitute"))


def test_prepared_workspace_proves_the_observed_copied_tree_and_git_closure():
    prepared = _prepared_execution()
    workspace = prepared.workspace_proof
    binding = prepared.preparation_claim.reservation.frontier.workspace_before

    assert workspace.observed_source_tree_digest == binding.source_tree_digest
    assert workspace.observed_git_closure_digest == binding.git_closure_digest
    assert workspace.observed_source_entry_count == binding.source_entry_count
    assert workspace.observed_source_size_bytes == binding.source_size_bytes
    assert (workspace.mount_id, workspace.device) == (
        prepared.runtime_volume_evidence.root_mount_id,
        prepared.runtime_volume_evidence.root_device,
    )
    with pytest.raises(RunActionSupervisorContractError, match="incomplete"):
        replace(
            workspace,
            observed_source_tree_digest=tree_or_blob_digest(b"substituted source"),
        )
    with pytest.raises(RunActionSupervisorContractError, match="incomplete"):
        replace(workspace, observed_source_entry_count=binding.source_entry_count + 1)
    substituted_workspace = _remint_contract(
        workspace,
        inode=workspace.inode + 100,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="positive byte or inode headroom",
    ):
        replace(prepared, workspace_proof=substituted_workspace)


def test_prepared_delivery_slots_are_empty_and_result_file_is_precreated():
    prepared = _prepared_execution()

    for delivery_slot in (
        prepared.input_delivery_slot,
        prepared.credential_delivery_slot,
    ):
        assert delivery_slot.directory_type == "directory"
        assert delivery_slot.mode == 0o700
        assert delivery_slot.observed_entry_count == 0
        assert (
            RunActionPreparedDeliverySlot.from_json_bytes(delivery_slot.to_json_bytes())
            == delivery_slot
        )
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(prepared.input_delivery_slot, observed_entry_count=1)
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(
            prepared.input_delivery_slot,
            final_file_name="substituted.blob",
        )
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(
            prepared.input_delivery_slot,
            kind=RunActionPreparedFileKind.RESULT,
        )

    assert prepared.result_file.file_type == "regular"
    assert prepared.result_file.mode == 0o600
    assert prepared.result_file.link_count == 1
    assert prepared.result_file.size_bytes == 0
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(prepared.result_file, link_count=2)
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(prepared.result_file, kind=RunActionPreparedFileKind.INPUT)


def test_prepared_runtime_directories_pin_result_and_temporary_subpaths():
    prepared = _prepared_execution()
    result = prepared.result_directory
    temporary = prepared.temporary_directory

    assert (
        result.kind,
        result.directory_relative_path,
        result.observed_entry_count,
    ) == (RunActionPreparedRuntimeDirectoryKind.RESULT, "result", 1)
    assert (
        temporary.kind,
        temporary.directory_relative_path,
        temporary.observed_entry_count,
    ) == (RunActionPreparedRuntimeDirectoryKind.TEMPORARY, "temporary", 0)
    assert (result.mount_id, result.device) == (
        prepared.runtime_volume_evidence.root_mount_id,
        prepared.runtime_volume_evidence.root_device,
    )
    assert (temporary.mount_id, temporary.device) == (
        prepared.runtime_volume_evidence.root_mount_id,
        prepared.runtime_volume_evidence.root_device,
    )
    assert len({result.inode, temporary.inode, prepared.result_file.inode}) == 3

    with pytest.raises(
        RunActionSupervisorContractError,
        match="runtime directory is invalid",
    ):
        replace(result, observed_entry_count=0)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="runtime directory is invalid",
    ):
        replace(temporary, observed_entry_count=1)
    substituted_result = _remint_contract(result, inode=result.inode + 100)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="artifacts differ from their preparation claim",
    ):
        replace(prepared, result_directory=substituted_result)
    foreign = _prepared_execution(inode_offset=8)
    with pytest.raises(
        RunActionSupervisorContractError,
        match="positive byte or inode headroom",
    ):
        replace(
            prepared,
            layout_proof=_remint_contract(
                prepared.layout_proof,
                prepared_runtime_directory_ids=(
                    foreign.layout_proof.prepared_runtime_directory_ids
                ),
            ),
        )


def test_runtime_volume_keeper_binds_helper_and_exact_live_generation():
    prepared = _prepared_execution()
    keeper = prepared.volume_keeper_evidence
    projection = keeper.issued_create_projection
    policy = prepared.preparation_claim.execution_policy

    assert keeper.container_status == "running"
    assert projection.network_mode == "none"
    helper = projection.helper_evidence
    assert helper.helper_authority_id == (
        policy.supervisor_helper_executable_authority_id
    )
    assert helper.source_path == "/usr/bin/busybox"
    assert helper.destination == "/kapso-supervisor/busybox"
    assert helper.mount_access is RunActionPreparedMountAccess.READ_ONLY
    assert helper.recursive_bind is False
    assert (helper.owner_user_id, helper.owner_group_id, helper.mode) == (
        0,
        0,
        0o755,
    )
    assert helper.dynamic_dependency_count == 0
    assert helper.elf_interpreter_present is False
    docker_init = projection.docker_init_source_evidence
    assert docker_init.init_authority_id == (policy.docker_init_executable_authority_id)
    assert docker_init.source_path == "/usr/bin/docker-init"
    assert docker_init.dynamic_dependency_count == 0
    assert docker_init.elf_interpreter_present is False
    mounted_helper = keeper.mounted_helper_evidence
    with pytest.raises(ContractValidationError, match="device must be an integer"):
        replace(mounted_helper, device=float(mounted_helper.device))
    with pytest.raises(ContractValidationError, match="inode must be an integer"):
        replace(mounted_helper, inode=float(mounted_helper.inode))
    with pytest.raises(
        RunActionSupervisorContractError,
        match="differs from its source inode or process",
    ):
        replace(
            mounted_helper,
            process_cgroup_path=mounted_helper.process_cgroup_path + "\x00",
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="differs from its source inode or process",
    ):
        replace(
            mounted_helper,
            process_cgroup_path=mounted_helper.process_cgroup_path.replace(
                "/test.",
                "/tést.",
            ),
        )
    assert projection.volume_authority == prepared.runtime_volume_authority
    assert (
        prepared.runtime_volume_evidence.sentinel_evidence.runtime_volume_authority_id
        == projection.volume_authority.runtime_volume_authority_id
    )
    with pytest.raises(RunActionSupervisorContractError, match="unsafe or substituted"):
        replace(
            helper,
            recursive_bind=True,
        )
    substituted_source = "/opt/substituted/busybox"
    substituted_helper = _remint_contract(
        helper,
        source_path=substituted_source,
        helper_authority_id=run_action_supervisor_helper_authority_id(
            substituted_source,
            helper.executable_digest,
        ),
    )
    with pytest.raises(RunActionSupervisorContractError, match="incomplete or unsafe"):
        replace(projection, helper_evidence=substituted_helper)
    substituted_init = _remint_contract(
        docker_init,
        inode=docker_init.inode + 1,
    )
    substituted_projection = _remint_contract(
        projection,
        docker_init_source_evidence=substituted_init,
    )
    substituted_keeper = _remint_contract(
        keeper,
        issued_create_projection=substituted_projection,
        observed_inspect_projection=substituted_projection,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="keeper differs from prepared authority",
    ):
        replace(prepared, volume_keeper_evidence=substituted_keeper)
    changed_observation = _remint_contract(
        projection,
        nonauthoritative_raw_field_count=5,
    )
    with pytest.raises(RunActionSupervisorContractError, match="exact and running"):
        _remint_contract(
            keeper,
            observed_inspect_projection=changed_observation,
        )
    aliased_container_id = prepared.inert_container_evidence.container_id
    aliased_mounted_helper = _remint_contract(
        keeper.mounted_helper_evidence,
        container_id=aliased_container_id,
        process_cgroup_path=(
            "/test.kapso.run_action.slice/" f"docker-{aliased_container_id}.scope"
        ),
    )
    aliased_keeper = _remint_contract(
        keeper,
        container_id=aliased_container_id,
        mounted_helper_evidence=aliased_mounted_helper,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="keeper differs from prepared authority",
    ):
        replace(prepared, volume_keeper_evidence=aliased_keeper)
    with pytest.raises(ContractValidationError, match="must be an object"):
        replace(prepared, volume_keeper_evidence=None)


def test_prepared_execution_requires_strict_byte_and_inode_headroom():
    prepared = _prepared_execution()
    evidence = prepared.runtime_volume_evidence
    limits = prepared.preparation_claim.execution_policy.docker_resource_limits
    block_size = evidence.allocation_block_size_bytes
    allocated_file_bytes = sum(
        ((size + block_size - 1) // block_size) * block_size
        for size in (
            prepared.input_delivery_slot.payload_size_limit_bytes,
            prepared.result_file.payload_size_limit_bytes,
            prepared.credential_delivery_slot.payload_size_limit_bytes,
            prepared.preparation_claim.execution_policy.supervisor_limits.release_receipt_size_bytes,
            prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes,
        )
    )
    exact_byte_cap = (
        evidence.used_size_bytes
        + allocated_file_bytes
        + (
            (limits.runtime_temporary_reservation_size_bytes + block_size - 1)
            // block_size
        )
        * block_size
    )
    exhausted_bytes = _remint_contract(
        evidence,
        effective_block_count=exact_byte_cap // block_size,
        effective_size_bytes=exact_byte_cap,
        available_block_count=(
            exact_byte_cap // block_size - evidence.used_block_count
        ),
        available_size_bytes=exact_byte_cap - evidence.used_size_bytes,
    )
    with pytest.raises(RunActionSupervisorContractError, match="positive byte"):
        replace(prepared, runtime_volume_evidence=exhausted_bytes)

    delivery_slot_count = 1 + (
        1 if prepared.credential_delivery_slot is not None else 0
    )
    exact_inode_cap = (
        evidence.used_inode_count
        + delivery_slot_count
        + limits.runtime_temporary_reservation_inode_count
        + 2
    )
    exhausted_inodes = _remint_contract(
        evidence,
        effective_inode_limit=exact_inode_cap,
        available_inode_count=exact_inode_cap - evidence.used_inode_count,
    )
    with pytest.raises(RunActionSupervisorContractError, match="positive byte"):
        replace(prepared, runtime_volume_evidence=exhausted_inodes)


def test_main_mounts_hide_volume_root_and_sentinel_and_bind_hostconfig_subpath():
    prepared = _prepared_execution()
    projection = prepared.inert_container_evidence.issued_create_projection

    assert all(
        mount.volume_subpath not in (".", ".kapso-generation")
        for mount in projection.mounts
    )
    assert all(
        mount.host_config_volume_subpath == mount.volume_subpath
        for mount in projection.mounts
    )
    with pytest.raises(RunActionSupervisorContractError, match="mount is invalid"):
        replace(
            projection.mounts[0],
            volume_subpath=".",
            host_config_volume_subpath=".",
        )
    with pytest.raises(RunActionSupervisorContractError, match="mount is invalid"):
        replace(
            projection.mounts[0],
            volume_subpath=".kapso-generation",
            host_config_volume_subpath=".kapso-generation",
        )


def test_main_mounts_reject_prefix_overlap_and_wrong_access():
    prepared = _prepared_execution()
    projection = prepared.inert_container_evidence.issued_create_projection
    input_mount = next(
        mount
        for mount in projection.mounts
        if mount.kind is RunActionPreparedMountKind.INPUT
    )
    result_mount = next(
        mount
        for mount in projection.mounts
        if mount.kind is RunActionPreparedMountKind.RESULT
    )

    with pytest.raises(RunActionSupervisorContractError, match="delivery mount"):
        replace(
            input_mount,
            container_access=RunActionPreparedMountAccess.READ_WRITE,
        )
    with pytest.raises(RunActionSupervisorContractError, match="mount is invalid"):
        replace(
            input_mount,
            source_access=RunActionPreparedMountAccess.READ_ONLY,
        )
    with pytest.raises(RunActionSupervisorContractError, match="output mount"):
        replace(
            result_mount,
            container_access=RunActionPreparedMountAccess.READ_ONLY,
        )
    with pytest.raises(RunActionSupervisorContractError, match="mount is invalid"):
        replace(
            input_mount,
            volume_subpath="result/input",
            host_config_volume_subpath="result/input",
        )


def test_prepared_execution_rejects_main_mount_substitution():
    prepared = _prepared_execution()
    evidence = prepared.inert_container_evidence
    projection = evidence.issued_create_projection
    mounts = list(projection.mounts)
    workspace_position = next(
        position
        for position, mount in enumerate(mounts)
        if mount.kind is RunActionPreparedMountKind.WORKSPACE
    )
    mounts[workspace_position] = replace(
        mounts[workspace_position],
        container_access=RunActionPreparedMountAccess.READ_WRITE,
    )
    substituted_projection = _projection_with_mounts(projection, tuple(mounts))

    with pytest.raises(RunActionSupervisorContractError, match="evidence differs"):
        replace(
            prepared,
            inert_container_evidence=_evidence_with_projection(
                evidence,
                substituted_projection,
            ),
        )


def test_superseded_contract_surfaces_are_removed():
    legacy_names = {
        "RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION",
        "RunActionActivatedTemporaryDirectoryObservation",
        "RunActionDescriptorWalkObservation",
        "RunActionFilesystemIdentity",
        "RunActionFilesystemNodeObservation",
        "RunActionKeeperHelperEvidence",
        "RunActionPreparedSlot",
        "RunActionQuotaObservation",
        "runtime_volume_keeper_helper_authority_id",
        "quota_scope_id",
    }

    assert legacy_names.isdisjoint(supervisor_contracts.__all__)
    assert all(not hasattr(supervisor_contracts, name) for name in legacy_names)


def test_inert_evidence_rejects_observed_projection_substitution():
    prepared = _prepared_execution()
    evidence = prepared.inert_container_evidence
    observed = evidence.observed_inspect_projection
    substituted_policy = _execution_policy(command_template_label="substituted")
    substituted_observed = DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=substituted_policy.projection_protocol_version,
        raw_field_schema_id=substituted_policy.raw_field_schema_id,
        execution_policy=substituted_policy,
        supervisor_helper_evidence=observed.supervisor_helper_evidence,
        docker_init_source_evidence=observed.docker_init_source_evidence,
        barrier_protocol_version=observed.barrier_protocol_version,
        barrier_poll_interval_seconds=observed.barrier_poll_interval_seconds,
        command_executable=observed.command_executable,
        command_arguments=(*observed.command_arguments[:-1], "substituted"),
        mounts=observed.mounts,
        exact_mount_count=observed.exact_mount_count,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=(observed.nonauthoritative_raw_field_count),
    )

    with pytest.raises(RunActionSupervisorContractError, match="exact inert"):
        RunActionInertContainerEvidence.mint(
            **{
                key: value
                for key, value in evidence.to_dict().items()
                if key
                not in {
                    "inert_container_evidence_id",
                    "observed_inspect_projection",
                }
            },
            observed_inspect_projection=substituted_observed,
        )


def test_legacy_arbitrary_create_projection_digest_is_rejected():
    evidence = _prepared_execution().inert_container_evidence

    with pytest.raises(ContractValidationError, match="unknown"):
        RunActionInertContainerEvidence.from_dict(
            evidence.to_dict()
            | {"create_projection_digest": tree_or_blob_digest(b"arbitrary")}
        )


@pytest.mark.parametrize(
    "key",
    ("OPENAI_API_KEY", "AUTH_TOKEN", "AWS_SECRET", "CONFIG"),
)
def test_static_environment_rejects_non_allowlisted_or_secret_like_keys(key):
    with pytest.raises(RunActionSupervisorContractError, match="secret-like"):
        RunActionStaticEnvironmentVariable(key=key, value="must-not-enter-policy")


def test_static_environment_rejects_arbitrary_value_in_allowlisted_path_key():
    with pytest.raises(RunActionSupervisorContractError, match="exact allowlist"):
        RunActionStaticEnvironmentVariable(
            key="PATH",
            value="/sk-live-secret-123",
        )


def test_container_identity_has_no_prepared_execution_back_edge():
    first = _prepared_execution(container_id="a" * 64)
    second = _prepared_execution(container_id="b" * 64, inode_offset=100)

    assert first.preparation_claim == second.preparation_claim
    assert (
        first.inert_container_evidence.container_name
        == second.inert_container_evidence.container_name
    )
    assert (
        first.inert_container_evidence.labels == second.inert_container_evidence.labels
    )
    assert first.prepared_execution_id != second.prepared_execution_id
    labels = {label.key: label.value for label in first.inert_container_evidence.labels}
    assert labels == {
        "com.kapso.run-action.claim": (first.preparation_claim.preparation_claim_id),
        "com.kapso.run-action.reservation": (
            first.preparation_claim.reservation.reservation_id
        ),
        "com.kapso.run-action.role": "execution",
    }
    label_values = {label.value for label in first.inert_container_evidence.labels}
    assert first.prepared_execution_id not in label_values
    assert not any(
        "prepared" in label.key for label in first.inert_container_evidence.labels
    )


def test_credential_policy_and_file_contracts_carry_no_secret_or_host_path_fields():
    forbidden_fragments = ("secret", "value", "host_path", "credential_path")

    for contract_type in (
        RunActionCredentialPolicy,
        RunActionActivatedFileObservation,
        RunActionPreparedDeliverySlot,
        RunActionPreparedFile,
    ):
        names = tuple(field.name for field in fields(contract_type))
        assert not any(
            fragment in name for name in names for fragment in forbidden_fragments
        )
    prepared = _prepared_execution()
    activation = _activation_revalidation_receipt(
        prepared,
        _spawn_commit(prepared),
    )
    credential = activation.credential_file_observation
    assert credential.content_digest is None
    with pytest.raises(
        RunActionSupervisorContractError,
        match="file observation is invalid",
    ):
        replace(
            credential,
            content_digest=tree_or_blob_digest(b"credential secret bytes"),
        )


def test_credential_free_preparation_has_no_credential_slot_or_workspace_mount():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))

    assert prepared.credential_delivery_slot is None
    assert {
        mount.kind
        for mount in prepared.inert_container_evidence.issued_create_projection.mounts
    } == {
        RunActionPreparedMountKind.CONTROL,
        RunActionPreparedMountKind.INPUT,
        RunActionPreparedMountKind.RESULT,
        RunActionPreparedMountKind.TEMPORARY,
    }
