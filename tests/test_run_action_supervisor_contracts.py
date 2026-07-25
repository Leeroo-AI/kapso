"""Contracts for deterministic claims and concrete inert Docker occurrences."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

import kapso.cross_run.launch.run_action_supervisor_contracts as supervisor_contracts
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
    RunActionActivatedFileObservation,
    RunActionActivatedSentinelObservation,
    RunActionActivatedWorkspaceObservation,
    RunActionActivationNetworkMode,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionCredentialPolicy,
    RunActionFilesystemPolicy,
    RunActionInertContainerEvidence,
    RunActionKeeperHelperEvidence,
    RunActionMountedKeeperHelperEvidence,
    RunActionNetworkPolicy,
    RunActionPreparationClaim,
    RunActionPreparedExecution,
    RunActionPreparedFile,
    RunActionPreparedFileKind,
    RunActionPreparedMount,
    RunActionPreparedMountAccess,
    RunActionPreparedMountKind,
    RunActionPreparedWorkspaceProof,
    RunActionRuntimeVolumeAuthority,
    RunActionRuntimeVolumeEvidence,
    RunActionRuntimeVolumeLayoutProof,
    RunActionRuntimeVolumeSentinelEvidence,
    RunActionStaticEnvironmentVariable,
    RunActionSupervisorContractError,
    RunActionSupervisorLimits,
    RunActionVolumeKeeperEvidence,
    issue_runtime_volume_authority,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
    runtime_volume_driver_options,
    runtime_volume_keeper_helper_authority_id,
    runtime_volume_sentinel_identity,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary


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
        maximum_lease_seconds=300,
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
        keeper_helper_source_path=helper_source_path,
        keeper_helper_executable_authority_id=(
            runtime_volume_keeper_helper_authority_id(
                helper_source_path,
                helper_digest,
            )
        ),
        keeper_helper_executable_digest=helper_digest,
        command_template_id=_fixture_content_id(
            "docker-run-action-command-template",
            command_template_label,
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
            result_size_bytes=268435456,
        ),
    )


def _claim(
    *,
    policy=None,
    boundary=None,
    request_digest=None,
    request_payload=None,
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
        security_observation_id=_fixture_content_id(
            "security-denylist-observation",
            "security",
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


def _prepared_file(claim, authority, kind):
    limits = {
        RunActionPreparedFileKind.INPUT: claim.reservation.request_blob.size_bytes,
        RunActionPreparedFileKind.RESULT: (
            claim.execution_policy.supervisor_limits.result_size_bytes
        ),
        RunActionPreparedFileKind.CREDENTIAL: (
            claim.execution_policy.credential_policy.maximum_delivery_size_bytes
        ),
    }
    paths = {
        RunActionPreparedFileKind.INPUT: "input/request.blob",
        RunActionPreparedFileKind.RESULT: "result/result.blob",
        RunActionPreparedFileKind.CREDENTIAL: "credential/credentials",
    }
    return RunActionPreparedFile.mint(
        preparation_claim_id=claim.preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=kind,
        relative_path=paths[kind],
        file_type="regular",
        owner_user_id=claim.execution_policy.user_id,
        owner_group_id=claim.execution_policy.group_id,
        mode=0o600,
        link_count=1,
        size_bytes=0,
        payload_size_limit_bytes=limits[kind],
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
    container_id="a" * 64,
    inode_offset=0,
):
    claim = _claim() if claim is None else claim
    nonce = f"{inode_offset + 1:032x}"
    authority = _volume_authority(claim, nonce=nonce)
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
    input_file = _prepared_file(claim, authority, RunActionPreparedFileKind.INPUT)
    result_file = _prepared_file(claim, authority, RunActionPreparedFileKind.RESULT)
    credential_file = (
        None
        if claim.execution_policy.credential_policy.mode is RunActionCredentialMode.NONE
        else _prepared_file(
            claim,
            authority,
            RunActionPreparedFileKind.CREDENTIAL,
        )
    )
    files = tuple(
        prepared_file
        for prepared_file in (input_file, result_file, credential_file)
        if prepared_file is not None
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
        )
    )
    directories = tuple(
        sorted(
            {
                "input",
                "result",
                "temporary",
                *(("credential",) if credential_file is not None else ()),
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
    logical_entry_count = len(directories) + len(files) + 1 + workspace_entries
    observed_used_size = 32768
    observed_used_inodes = logical_entry_count + 2
    volume_evidence = RunActionRuntimeVolumeEvidence.mint(
        volume_authority=authority,
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
        generation_nonce=authority.generation_nonce,
        empty_size_bytes=0,
        empty_entry_count=0,
        directory_relative_paths=directories,
        prepared_file_ids=tuple(
            sorted(prepared_file.prepared_file_id for prepared_file in files)
        ),
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
    helper_evidence = RunActionKeeperHelperEvidence.mint(
        helper_authority_id=policy.keeper_helper_executable_authority_id,
        source_path=policy.keeper_helper_source_path,
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
        executable_digest=policy.keeper_helper_executable_digest,
        mount_id=3000 + inode_offset,
        device=700,
        inode=800,
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
        restart_count=0,
        restart_policy_name="no",
        auto_remove=False,
    )
    issued_projection = DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        execution_policy=policy,
        mounts=_mounts(claim, authority.volume_name),
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
        input_file=input_file,
        result_file=result_file,
        credential_file=credential_file,
        workspace_proof=workspace_proof,
        layout_proof=layout_proof,
        inert_container_evidence=evidence,
    )


def _projection_with_mounts(projection, mounts):
    return DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=projection.projection_protocol_version,
        raw_field_schema_id=projection.raw_field_schema_id,
        execution_policy=projection.execution_policy,
        mounts=mounts,
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


def _volume_with_added_blocks(evidence, added_block_count):
    block_size = evidence.allocation_block_size_bytes
    return _remint_contract(
        evidence,
        used_block_count=evidence.used_block_count + added_block_count,
        used_size_bytes=(evidence.used_size_bytes + added_block_count * block_size),
        available_block_count=(evidence.available_block_count - added_block_count),
        available_size_bytes=(
            evidence.available_size_bytes - added_block_count * block_size
        ),
    )


def test_prepared_execution_round_trips_with_complete_content_identity():
    prepared = _prepared_execution()

    assert (
        RunActionPreparedExecution.from_json_bytes(prepared.to_json_bytes()) == prepared
    )
    assert prepared.prepared_execution_id.startswith(
        "run-action-prepared-execution:sha256:"
    )
    assert (
        prepared.preparation_claim.execution_policy.image_authority.image_authority_id
    )


def test_activation_revalidation_binds_fresh_exact_prepared_observations():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    request_blob = prepared.preparation_claim.reservation.request_blob
    input_observation = _activated_file_observation(
        prepared.input_file,
        size_bytes=request_blob.size_bytes,
        content_digest=request_blob.digest,
        content_authority_id=request_blob.request_blob_id,
    )
    result_observation = _activated_file_observation(
        prepared.result_file,
        size_bytes=0,
        content_digest=None,
        content_authority_id=None,
    )
    credential_observation = _activated_file_observation(
        prepared.credential_file,
        size_bytes=32,
        content_digest=None,
        content_authority_id="test.credential.lease",
    )
    reobserved_volume = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        2,
    )
    receipt = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=reobserved_volume,
        reobserved_keeper_evidence=prepared.volume_keeper_evidence,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=_activated_workspace_observation(
            prepared,
            spawn,
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
    substituted_delivery = _remint_contract(
        receipt.input_file_observation,
        content_digest=tree_or_blob_digest(b"same-size-substitute"),
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, input_file_observation=substituted_delivery)

    with pytest.raises(RunActionSupervisorContractError, match="file observation"):
        replace(
            receipt.input_file_observation,
            file_type="symlink",
        )
    incomplete_usage = _volume_with_added_blocks(
        prepared.runtime_volume_evidence,
        1,
    )
    with pytest.raises(RunActionSupervisorContractError, match="statfs usage"):
        _remint_contract(
            receipt,
            reobserved_volume_evidence=incomplete_usage,
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
    with pytest.raises(RunActionSupervisorContractError, match="headroom"):
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
        ),
        "reobserved_keeper_evidence": prepared.volume_keeper_evidence,
        "reobserved_container_evidence": prepared.inert_container_evidence,
        "activated_workspace_observation": _activated_workspace_observation(
            prepared,
            spawn,
        ),
        "activated_sentinel_observation": _activated_sentinel_observation(
            prepared,
            spawn,
        ),
        "input_file_observation": _activated_file_observation(
            prepared.input_file,
            size_bytes=request_blob.size_bytes,
            content_digest=request_blob.digest,
            content_authority_id=request_blob.request_blob_id,
        ),
        "result_file_observation": _activated_file_observation(
            prepared.result_file,
            size_bytes=0,
            content_digest=None,
            content_authority_id=None,
        ),
        "credential_file_observation": _activated_file_observation(
            prepared.credential_file,
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
        ),
        reobserved_keeper_evidence=prepared.volume_keeper_evidence,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=None,
        activated_sentinel_observation=_activated_sentinel_observation(
            prepared,
            spawn,
        ),
        input_file_observation=_activated_file_observation(
            prepared.input_file,
            size_bytes=request_blob.size_bytes,
            content_digest=request_blob.digest,
            content_authority_id=request_blob.request_blob_id,
        ),
        result_file_observation=_activated_file_observation(
            prepared.result_file,
            size_bytes=0,
            content_digest=None,
            content_authority_id=None,
        ),
        credential_file_observation=None,
    )

    assert receipt.credential_file_observation is None
    assert receipt.activated_workspace_observation is None
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
    prepared_file,
    *,
    size_bytes,
    content_digest,
    content_authority_id,
):
    return RunActionActivatedFileObservation.mint(
        prepared_file_id=prepared_file.prepared_file_id,
        runtime_volume_authority_id=prepared_file.runtime_volume_authority_id,
        generation_nonce=prepared_file.generation_nonce,
        kind=prepared_file.kind,
        file_type=prepared_file.file_type,
        owner_user_id=prepared_file.owner_user_id,
        owner_group_id=prepared_file.owner_group_id,
        mode=prepared_file.mode,
        link_count=prepared_file.link_count,
        size_bytes=size_bytes,
        content_digest=content_digest,
        content_authority_id=content_authority_id,
    )


def test_semantic_claim_changes_with_request_or_execution_policy():
    original = _claim()
    changed_request = _claim(request_digest=tree_or_blob_digest(b"another request"))
    changed_policy = _claim(
        policy=_execution_policy(command_template_label="another-command")
    )

    assert original.preparation_claim_id != changed_request.preparation_claim_id
    assert original.preparation_claim_id != changed_policy.preparation_claim_id


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


@pytest.mark.parametrize("delimiter", (",", "\r", "\n", '"'))
def test_keeper_helper_authority_rejects_docker_mount_delimiters(delimiter):
    with pytest.raises(
        RunActionSupervisorContractError,
        match="normalized and absolute",
    ):
        runtime_volume_keeper_helper_authority_id(
            f"/usr/bin/busybox{delimiter}readonly",
            tree_or_blob_digest(b"helper"),
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


def test_lifecycle_policy_binding_pins_the_keeper_helper_bytes():
    claim = _claim()
    alternate_digest = tree_or_blob_digest(b"alternate keeper helper")
    substituted_policy = _remint_policy(
        claim.execution_policy,
        keeper_helper_executable_authority_id=(
            runtime_volume_keeper_helper_authority_id(
                claim.execution_policy.keeper_helper_source_path,
                alternate_digest,
            )
        ),
        keeper_helper_executable_digest=alternate_digest,
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
    with pytest.raises(RunActionSupervisorContractError, match="keeper helper"):
        _remint_policy(
            claim.execution_policy,
            keeper_helper_executable_digest=alternate_digest,
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
    assert "safe_create_defaults" in {
        field.name for field in fields(DockerRunActionExecutionPolicy)
    }
    assert tuple(RunActionActivationNetworkMode) == (
        RunActionActivationNetworkMode.NONE,
    )


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
    assert authority.labels == preparation_volume_labels(prepared.preparation_claim)
    assert authority.driver_options == runtime_volume_driver_options(authority)
    assert "noswap" in authority.driver_options[1]
    assert evidence.observed_scope == "local"
    assert evidence.observed_volume_name == authority.volume_name
    assert evidence.observed_labels == authority.labels
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
    with pytest.raises(RunActionSupervisorContractError, match="incomplete"):
        replace(
            workspace,
            observed_source_tree_digest=tree_or_blob_digest(b"substituted source"),
        )
    with pytest.raises(RunActionSupervisorContractError, match="incomplete"):
        replace(workspace, observed_source_entry_count=binding.source_entry_count + 1)


def test_prepared_logical_files_are_empty_private_regular_files():
    prepared = _prepared_execution()

    for prepared_file in (
        prepared.input_file,
        prepared.result_file,
        prepared.credential_file,
    ):
        assert prepared_file.file_type == "regular"
        assert prepared_file.mode == 0o600
        assert prepared_file.link_count == 1
        assert prepared_file.size_bytes == 0
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(prepared.input_file, mode=0o400)
    with pytest.raises(RunActionSupervisorContractError, match="invalid or nonempty"):
        replace(prepared.result_file, link_count=2)


def test_runtime_volume_keeper_binds_helper_and_exact_live_generation():
    prepared = _prepared_execution()
    keeper = prepared.volume_keeper_evidence
    projection = keeper.issued_create_projection
    policy = prepared.preparation_claim.execution_policy

    assert keeper.container_status == "running"
    assert projection.network_mode == "none"
    helper = projection.helper_evidence
    assert helper.helper_authority_id == (policy.keeper_helper_executable_authority_id)
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
        helper_authority_id=runtime_volume_keeper_helper_authority_id(
            substituted_source,
            helper.executable_digest,
        ),
    )
    with pytest.raises(RunActionSupervisorContractError, match="incomplete or unsafe"):
        replace(projection, helper_evidence=substituted_helper)
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
    with pytest.raises(RunActionSupervisorContractError, match="evidence differs"):
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
            prepared.input_file.payload_size_limit_bytes,
            prepared.result_file.payload_size_limit_bytes,
            prepared.credential_file.payload_size_limit_bytes,
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

    exact_inode_cap = (
        evidence.used_inode_count + limits.runtime_temporary_reservation_inode_count
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


def test_old_host_quota_and_descriptor_contracts_are_not_exported():
    legacy_names = {
        "RunActionDescriptorWalkObservation",
        "RunActionFilesystemIdentity",
        "RunActionFilesystemNodeObservation",
        "RunActionPreparedSlot",
        "RunActionQuotaObservation",
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
        mounts=observed.mounts,
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
        RunActionPreparedFile,
    ):
        names = tuple(field.name for field in fields(contract_type))
        assert not any(
            fragment in name for name in names for fragment in forbidden_fragments
        )


def test_credential_free_preparation_has_no_credential_file_or_workspace_mount():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))

    assert prepared.credential_file is None
    assert {
        mount.kind
        for mount in prepared.inert_container_evidence.issued_create_projection.mounts
    } == {
        RunActionPreparedMountKind.INPUT,
        RunActionPreparedMountKind.RESULT,
        RunActionPreparedMountKind.TEMPORARY,
    }
