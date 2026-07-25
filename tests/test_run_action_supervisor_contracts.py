"""Contracts for deterministic claims and concrete inert Docker occurrences."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

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
    DockerRunActionResourceLimits,
    DockerRunActionSafeCreateDefaults,
    DockerRunActionSandboxSpec,
    DockerRunActionUlimit,
    RunActionActivatedSlotObservation,
    RunActionActivationNetworkMode,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionCredentialDeliveryReceipt,
    RunActionCredentialPolicy,
    RunActionDescriptorWalkObservation,
    RunActionFilesystemIdentity,
    RunActionFilesystemNodeObservation,
    RunActionFilesystemPolicy,
    RunActionInertContainerEvidence,
    RunActionNetworkPolicy,
    RunActionNoCredentialsProof,
    RunActionPreparationClaim,
    RunActionPreparedExecution,
    RunActionPreparedMount,
    RunActionPreparedMountAccess,
    RunActionPreparedMountKind,
    RunActionPreparedSlot,
    RunActionPreparedSlotKind,
    RunActionQuotaObservation,
    RunActionRequestDeliveryReceipt,
    RunActionStaticEnvironmentVariable,
    RunActionSupervisorContractError,
    RunActionSupervisorLimits,
    preparation_container_labels,
    preparation_container_name,
    quota_scope_id,
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
            user_namespace_mode="private",
            cgroup_parent_id="test.kapso.run_action",
            sysctl_ids=(),
            no_new_privileges=True,
            seccomp_profile_id="test.seccomp.v1",
            apparmor_profile_id="test.apparmor.v1",
            security_option_ids=(
                "apparmor:test.apparmor.v1",
                "no-new-privileges",
                "seccomp:test.seccomp.v1",
            ),
            masked_system_paths=("/proc/kcore",),
            read_only_system_paths=("/proc/acpi",),
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
            temporary_filesystem_mode=0o700,
            temporary_filesystem_read_only=False,
            temporary_filesystem_nosuid=True,
            temporary_filesystem_nodev=True,
            temporary_filesystem_noexec=True,
        ),
        network_policy=RunActionNetworkPolicy.mint(
            activation_mode=(
                RunActionActivationNetworkMode.NONE
                if credential_mode is RunActionCredentialMode.NONE
                else RunActionActivationNetworkMode.BROKER_ONLY
            ),
            broker_endpoint_ids=(
                ()
                if credential_mode is RunActionCredentialMode.NONE
                else ("test.provider.broker.endpoint",)
            ),
        ),
        credential_policy=credential_policy,
        docker_resource_limits=DockerRunActionResourceLimits.mint(
            cpu_period_microseconds=100000,
            cpu_quota_microseconds=200000,
            cpu_shares=1024,
            nano_cpus=2000000000,
            cpu_realtime_period_microseconds=0,
            cpu_realtime_runtime_microseconds=0,
            cpuset_cpu_ids=(),
            cpuset_memory_node_ids=(),
            memory_size_bytes=1073741824,
            memory_reservation_size_bytes=536870912,
            memory_swap_size_bytes=1073741824,
            memory_swappiness_percentage=0,
            oom_kill_disabled=False,
            oom_score_adjustment=0,
            process_limit=128,
            block_io_weight=500,
            block_io_read_bandwidth_rule_ids=(),
            block_io_write_bandwidth_rule_ids=(),
            block_io_read_iops_rule_ids=(),
            block_io_write_iops_rule_ids=(),
            ulimits=(
                DockerRunActionUlimit(
                    name="nofile",
                    soft_limit=1024,
                    hard_limit=1024,
                ),
            ),
            shared_memory_size_bytes=67108864,
            temporary_filesystem_size_bytes=134217728,
        ),
        supervisor_limits=RunActionSupervisorLimits.mint(
            execution_timeout_seconds=600,
            termination_grace_seconds=30,
            stdout_size_bytes=16777216,
            stderr_size_bytes=16777216,
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
        boundary={
            RunFrontierActionKind.CODING_AGENT: RunSafetyBoundary.IDEATION,
            RunFrontierActionKind.EMBEDDING: RunSafetyBoundary.IDEATION,
            RunFrontierActionKind.EVALUATOR: RunSafetyBoundary.EVALUATION,
        }[policy.kind],
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


def _slot(claim, kind, *, inode_offset=0):
    destinations = {
        RunActionPreparedSlotKind.INPUT: (
            claim.execution_policy.filesystem_policy.input_destination
        ),
        RunActionPreparedSlotKind.RESULT: (
            claim.execution_policy.filesystem_policy.result_destination
        ),
        RunActionPreparedSlotKind.CREDENTIAL: (
            claim.execution_policy.filesystem_policy.credential_destination
        ),
    }
    maximum_sizes = {
        RunActionPreparedSlotKind.INPUT: claim.reservation.request_blob.size_bytes,
        RunActionPreparedSlotKind.RESULT: (
            claim.execution_policy.supervisor_limits.result_size_bytes
        ),
        RunActionPreparedSlotKind.CREDENTIAL: (
            claim.execution_policy.credential_policy.maximum_delivery_size_bytes
        ),
    }
    position = tuple(RunActionPreparedSlotKind).index(kind)
    descriptor_walk = RunActionDescriptorWalkObservation.mint(
        root_authority_id=_fixture_content_id(
            "run-action-storage-root-authority",
            "supervisor-slots",
        ),
        resolution_protocol_version="openat2-beneath-no-symlink.v1",
        nodes=(
            _filesystem_node(
                mount_id=1,
                device=1,
                inode=2,
                generation=1,
            ),
            _filesystem_node(
                mount_id=1,
                device=201,
                inode=211 + inode_offset,
                generation=11 + inode_offset,
            ),
            _filesystem_node(
                mount_id=1,
                device=201,
                inode=401 + position + inode_offset,
                generation=21 + position + inode_offset,
            ),
        ),
    )
    return RunActionPreparedSlot.mint(
        preparation_claim_id=claim.preparation_claim_id,
        kind=kind,
        descriptor_walk=descriptor_walk,
        quota_observation=RunActionQuotaObservation.mint(
            preparation_claim_id=claim.preparation_claim_id,
            slot_kind=kind,
            leaf_identity=descriptor_walk.nodes[-1].identity,
            quota_backend_authority_id=_fixture_content_id(
                "run-action-quota-backend-authority",
                "project-quota",
            ),
            filesystem_instance_id="test.slot.filesystem",
            filesystem_mount_id=descriptor_walk.nodes[-1].identity.mount_id,
            filesystem_device=descriptor_walk.nodes[-1].identity.device,
            exclusive_scope_id=quota_scope_id(
                claim.preparation_claim_id,
                kind,
                descriptor_walk.nodes[-1].identity,
            ),
            enabled=True,
            enforced=True,
            hard_size_bytes=maximum_sizes[kind],
            hard_entry_count=1,
            current_size_bytes=0,
            current_entry_count=0,
        ),
        expected_owner_user_id=1000,
        expected_owner_group_id=1000,
        expected_mode=0o700,
        container_destination=destinations[kind],
        payload_size_limit_bytes=maximum_sizes[kind],
    )


def _filesystem_node(*, mount_id, device, inode, generation):
    return RunActionFilesystemNodeObservation(
        identity=RunActionFilesystemIdentity(
            mount_id=mount_id,
            device=device,
            inode=inode,
            inode_generation=generation,
        ),
        file_type="directory",
        owner_user_id=1000,
        owner_group_id=1000,
        mode=0o700,
        unexpected_acl_count=0,
        unexpected_link_count=0,
    )


def _workspace_walk(claim):
    workspace = claim.reservation.frontier.workspace_before
    if workspace is None:
        raise AssertionError("workspace walk requires one workspace binding")
    return RunActionDescriptorWalkObservation.mint(
        root_authority_id=_fixture_content_id(
            "run-action-storage-root-authority",
            "workspace",
        ),
        resolution_protocol_version="openat2-beneath-no-symlink.v1",
        nodes=(
            _filesystem_node(
                mount_id=2,
                device=2,
                inode=3,
                generation=1,
            ),
            _filesystem_node(
                mount_id=2,
                device=workspace.workspace_device,
                inode=workspace.workspace_inode,
                generation=2,
            ),
        ),
    )


def _mounts(claim, slots):
    mounts = [
        RunActionPreparedMount(
            kind={
                RunActionPreparedSlotKind.INPUT: RunActionPreparedMountKind.INPUT,
                RunActionPreparedSlotKind.RESULT: RunActionPreparedMountKind.RESULT,
                RunActionPreparedSlotKind.CREDENTIAL: (
                    RunActionPreparedMountKind.CREDENTIAL
                ),
            }[slot.kind],
            prepared_slot_id=slot.prepared_slot_id,
            source_walk=slot.descriptor_walk,
            container_destination=slot.container_destination,
            mount_type="bind",
            access=(
                RunActionPreparedMountAccess.READ_WRITE
                if slot.kind is RunActionPreparedSlotKind.RESULT
                else RunActionPreparedMountAccess.READ_ONLY
            ),
            bind_propagation="rprivate",
            recursive_read_only=slot.kind is not RunActionPreparedSlotKind.RESULT,
            nested_mount_count=0,
        )
        for slot in slots
    ]
    workspace_access = claim.reservation.intent.workspace_access
    if workspace_access is not RunFrontierWorkspaceAccess.NONE:
        mounts.append(
            RunActionPreparedMount(
                kind=RunActionPreparedMountKind.WORKSPACE,
                prepared_slot_id=None,
                source_walk=_workspace_walk(claim),
                container_destination=(
                    claim.execution_policy.filesystem_policy.workspace_destination
                ),
                mount_type="bind",
                access=(
                    RunActionPreparedMountAccess.READ_WRITE
                    if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                    else RunActionPreparedMountAccess.READ_ONLY
                ),
                bind_propagation="rprivate",
                recursive_read_only=(
                    workspace_access is not RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                ),
                nested_mount_count=0,
            )
        )
    return tuple(sorted(mounts, key=lambda mount: mount.container_destination))


def _prepared_execution(
    *,
    claim=None,
    container_id="a" * 64,
    inode_offset=0,
):
    claim = _claim() if claim is None else claim
    input_slot = _slot(
        claim,
        RunActionPreparedSlotKind.INPUT,
        inode_offset=inode_offset,
    )
    result_slot = _slot(
        claim,
        RunActionPreparedSlotKind.RESULT,
        inode_offset=inode_offset,
    )
    credential_slot = (
        None
        if claim.execution_policy.credential_policy.mode is RunActionCredentialMode.NONE
        else _slot(
            claim,
            RunActionPreparedSlotKind.CREDENTIAL,
            inode_offset=inode_offset,
        )
    )
    slots = tuple(
        slot for slot in (input_slot, result_slot, credential_slot) if slot is not None
    )
    policy = claim.execution_policy
    issued_projection = DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        execution_policy=policy,
        mounts=_mounts(claim, slots),
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
        input_slot=input_slot,
        result_slot=result_slot,
        credential_slot=credential_slot,
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
    input_observation = _activated_slot_observation(
        prepared.input_slot,
        current_size_bytes=(
            prepared.preparation_claim.reservation.request_blob.size_bytes
        ),
        current_entry_count=1,
    )
    result_observation = _activated_slot_observation(
        prepared.result_slot,
        current_size_bytes=0,
        current_entry_count=0,
    )
    credential_observation = _activated_slot_observation(
        prepared.credential_slot,
        current_size_bytes=32,
        current_entry_count=1,
    )
    receipt = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=_spawn_commit(prepared),
        reobserved_container_evidence=prepared.inert_container_evidence,
        input_slot_observation=input_observation,
        result_slot_observation=result_observation,
        credential_slot_observation=credential_observation,
        request_delivery_receipt=RunActionRequestDeliveryReceipt.mint(
            prepared_execution_id=prepared.prepared_execution_id,
            spawn_commit_id=_spawn_commit(prepared).spawn_commit_id,
            input_slot_id=prepared.input_slot.prepared_slot_id,
            request_blob_id=(
                prepared.preparation_claim.reservation.request_blob.request_blob_id
            ),
            delivered_digest=(
                prepared.preparation_claim.reservation.request_blob.digest
            ),
            delivered_size_bytes=(
                prepared.preparation_claim.reservation.request_blob.size_bytes
            ),
            delivered_entry_count=1,
            delivered_relative_name="request.blob",
            delivered_file_type="regular",
            delivered_owner_user_id=prepared.preparation_claim.execution_policy.user_id,
            delivered_owner_group_id=prepared.preparation_claim.execution_policy.group_id,
            delivered_mode=0o400,
            delivered_link_count=1,
        ),
        credential_delivery_receipt=RunActionCredentialDeliveryReceipt.mint(
            prepared_execution_id=prepared.prepared_execution_id,
            spawn_commit_id=_spawn_commit(prepared).spawn_commit_id,
            credential_slot_id=prepared.credential_slot.prepared_slot_id,
            credential_policy_id=(
                prepared.preparation_claim.execution_policy.credential_policy.credential_policy_id
            ),
            lease_authority_id="test.credential.lease",
            delivered_size_bytes=credential_observation.current_size_bytes,
            delivered_entry_count=1,
            delivered_relative_name="credentials",
            delivered_file_type="regular",
            delivered_owner_user_id=prepared.preparation_claim.execution_policy.user_id,
            delivered_owner_group_id=prepared.preparation_claim.execution_policy.group_id,
            delivered_mode=0o400,
            delivered_link_count=1,
        ),
        no_credentials_proof=None,
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
    alternate_spawn = RunActionSpawnCommit.mint(
        reservation_id=receipt.spawn_commit.reservation_id,
        prepared_execution_id=receipt.spawn_commit.prepared_execution_id,
        provider_execution_id=receipt.spawn_commit.provider_execution_id,
        invocation_nonce="2" * 32,
        security_observation_id=receipt.spawn_commit.security_observation_id,
        boundary_identity=receipt.spawn_commit.boundary_identity,
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, spawn_commit=alternate_spawn)

    substituted_delivery = RunActionRequestDeliveryReceipt.mint(
        **{
            key: value
            for key, value in receipt.request_delivery_receipt.to_dict().items()
            if key != "request_delivery_receipt_id"
        }
        | {"delivered_digest": tree_or_blob_digest(b"same-size-substitute")}
    )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="prepared authority",
    ):
        replace(receipt, request_delivery_receipt=substituted_delivery)

    with pytest.raises(RunActionSupervisorContractError, match="request delivery"):
        replace(
            receipt.request_delivery_receipt,
            delivered_file_type="symlink",
        )


def test_activation_revalidation_requires_positive_no_credentials_proof():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    request_blob = prepared.preparation_claim.reservation.request_blob
    receipt = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=_spawn_commit(prepared),
        reobserved_container_evidence=prepared.inert_container_evidence,
        input_slot_observation=_activated_slot_observation(
            prepared.input_slot,
            current_size_bytes=request_blob.size_bytes,
            current_entry_count=1,
        ),
        result_slot_observation=_activated_slot_observation(
            prepared.result_slot,
            current_size_bytes=0,
            current_entry_count=0,
        ),
        credential_slot_observation=None,
        request_delivery_receipt=RunActionRequestDeliveryReceipt.mint(
            prepared_execution_id=prepared.prepared_execution_id,
            spawn_commit_id=_spawn_commit(prepared).spawn_commit_id,
            input_slot_id=prepared.input_slot.prepared_slot_id,
            request_blob_id=request_blob.request_blob_id,
            delivered_digest=request_blob.digest,
            delivered_size_bytes=request_blob.size_bytes,
            delivered_entry_count=1,
            delivered_relative_name="request.blob",
            delivered_file_type="regular",
            delivered_owner_user_id=prepared.preparation_claim.execution_policy.user_id,
            delivered_owner_group_id=prepared.preparation_claim.execution_policy.group_id,
            delivered_mode=0o400,
            delivered_link_count=1,
        ),
        credential_delivery_receipt=None,
        no_credentials_proof=RunActionNoCredentialsProof.mint(
            prepared_execution_id=prepared.prepared_execution_id,
            spawn_commit_id=_spawn_commit(prepared).spawn_commit_id,
            credential_policy_id=(
                prepared.preparation_claim.execution_policy.credential_policy.credential_policy_id
            ),
        ),
    )

    assert receipt.credential_slot_observation is None
    assert receipt.no_credentials_proof is not None


def _spawn_commit(prepared):
    reservation = prepared.preparation_claim.reservation
    return RunActionSpawnCommit.mint(
        reservation_id=reservation.reservation_id,
        prepared_execution_id=prepared.prepared_execution_id,
        provider_execution_id=prepared.inert_container_evidence.container_id,
        invocation_nonce="1" * 32,
        security_observation_id=reservation.frontier.security_observation_id,
        boundary_identity=reservation.intent.boundary_identity,
    )


def _activated_slot_observation(
    slot,
    *,
    current_size_bytes,
    current_entry_count,
):
    quota = slot.quota_observation
    return RunActionActivatedSlotObservation.mint(
        prepared_slot_id=slot.prepared_slot_id,
        descriptor_walk=slot.descriptor_walk,
        quota_backend_authority_id=quota.quota_backend_authority_id,
        filesystem_instance_id=quota.filesystem_instance_id,
        exclusive_scope_id=quota.exclusive_scope_id,
        quota_enabled=quota.enabled,
        quota_enforced=quota.enforced,
        hard_size_bytes=quota.hard_size_bytes,
        hard_entry_count=quota.hard_entry_count,
        current_size_bytes=current_size_bytes,
        current_entry_count=current_entry_count,
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
        ("device_authority_ids", ("test.gpu", "test.gpu")),
        ("supplementary_group_ids", (7,)),
        ("pid_namespace_mode", "host"),
        ("sysctl_ids", ("net.ipv4.ip_forward=1",)),
        ("no_new_privileges", False),
        ("security_option_ids", ()),
        ("log_driver", "json-file"),
        ("init_process", False),
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


@pytest.mark.parametrize(
    "sandbox",
    (
        _remint_sandbox(
            _execution_policy().sandbox_spec,
            seccomp_profile_id="unconfined",
            security_option_ids=(
                "apparmor:test.apparmor.v1",
                "no-new-privileges",
                "seccomp:unconfined",
            ),
        ),
        _remint_sandbox(
            _execution_policy().sandbox_spec,
            masked_system_paths=("/kapso/noop-mask",),
            read_only_system_paths=("/kapso/noop-read-only",),
        ),
        _remint_sandbox(
            _execution_policy().sandbox_spec,
            device_authority_ids=("test.unauthorized.device",),
            device_cgroup_rule_ids=("test.unauthorized.device.rule",),
        ),
    ),
)
def test_lifecycle_policy_binding_rejects_alternate_valid_sandbox(sandbox):
    claim = _claim()
    substituted_policy = _remint_policy(
        claim.execution_policy,
        sandbox_spec=sandbox,
    )

    with pytest.raises(RunActionSupervisorContractError, match="durable reservation"):
        RunActionPreparationClaim.mint(
            reservation=claim.reservation,
            execution_policy=substituted_policy,
        )


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


def test_prepared_slot_rejects_changed_owner_mode_or_acl():
    slot = _prepared_execution().result_slot
    walk = slot.descriptor_walk
    changed_leaf = replace(walk.nodes[-1], mode=0o755)
    changed_walk = RunActionDescriptorWalkObservation.mint(
        root_authority_id=walk.root_authority_id,
        resolution_protocol_version=walk.resolution_protocol_version,
        nodes=(*walk.nodes[:-1], changed_leaf),
    )

    with pytest.raises(RunActionSupervisorContractError, match="not private"):
        RunActionPreparedSlot.mint(
            preparation_claim_id=slot.preparation_claim_id,
            kind=slot.kind,
            descriptor_walk=changed_walk,
            quota_observation=slot.quota_observation,
            expected_owner_user_id=slot.expected_owner_user_id,
            expected_owner_group_id=slot.expected_owner_group_id,
            expected_mode=slot.expected_mode,
            container_destination=slot.container_destination,
            payload_size_limit_bytes=slot.payload_size_limit_bytes,
        )


def test_prepared_slot_rejects_disabled_or_unenforced_quota():
    quota = _prepared_execution().result_slot.quota_observation

    with pytest.raises(RunActionSupervisorContractError, match="quota"):
        replace(quota, enforced=False)


def test_prepared_slot_rejects_reused_or_oversized_quota_scope():
    prepared = _prepared_execution()
    input_slot = prepared.input_slot

    with pytest.raises(RunActionSupervisorContractError, match="exact leaf"):
        RunActionPreparedSlot.mint(
            preparation_claim_id=input_slot.preparation_claim_id,
            kind=input_slot.kind,
            descriptor_walk=input_slot.descriptor_walk,
            quota_observation=prepared.result_slot.quota_observation,
            expected_owner_user_id=input_slot.expected_owner_user_id,
            expected_owner_group_id=input_slot.expected_owner_group_id,
            expected_mode=input_slot.expected_mode,
            container_destination=input_slot.container_destination,
            payload_size_limit_bytes=input_slot.payload_size_limit_bytes,
        )

    oversized_quota = RunActionQuotaObservation.mint(
        **{
            key: value
            for key, value in input_slot.quota_observation.to_dict().items()
            if key != "quota_observation_id"
        }
        | {"hard_size_bytes": input_slot.payload_size_limit_bytes + 1}
    )
    with pytest.raises(RunActionSupervisorContractError, match="exact leaf"):
        RunActionPreparedSlot.mint(
            preparation_claim_id=input_slot.preparation_claim_id,
            kind=input_slot.kind,
            descriptor_walk=input_slot.descriptor_walk,
            quota_observation=oversized_quota,
            expected_owner_user_id=input_slot.expected_owner_user_id,
            expected_owner_group_id=input_slot.expected_owner_group_id,
            expected_mode=input_slot.expected_mode,
            container_destination=input_slot.container_destination,
            payload_size_limit_bytes=input_slot.payload_size_limit_bytes,
        )


def test_quota_observation_rejects_wrong_leaf_filesystem_binding():
    quota = _prepared_execution().input_slot.quota_observation

    with pytest.raises(RunActionSupervisorContractError, match="quota"):
        replace(quota, filesystem_device=quota.filesystem_device + 1)


def test_prepared_execution_rejects_slot_mount_substitution():
    prepared = _prepared_execution()
    evidence = prepared.inert_container_evidence
    projection = evidence.issued_create_projection
    mounts = list(projection.mounts)
    source_walk = mounts[0].source_walk
    changed_leaf = replace(
        source_walk.nodes[-1],
        identity=replace(
            source_walk.nodes[-1].identity,
            inode=source_walk.nodes[-1].identity.inode + 1,
            inode_generation=source_walk.nodes[-1].identity.inode_generation + 1,
        ),
    )
    mounts[0] = replace(
        mounts[0],
        source_walk=RunActionDescriptorWalkObservation.mint(
            root_authority_id=source_walk.root_authority_id,
            resolution_protocol_version=source_walk.resolution_protocol_version,
            nodes=(*source_walk.nodes[:-1], changed_leaf),
        ),
    )
    substituted_projection = _projection_with_mounts(projection, tuple(mounts))
    substituted_evidence = _evidence_with_projection(
        evidence,
        substituted_projection,
    )

    with pytest.raises(RunActionSupervisorContractError, match="evidence differs"):
        RunActionPreparedExecution.mint(
            preparation_claim=prepared.preparation_claim,
            input_slot=prepared.input_slot,
            result_slot=prepared.result_slot,
            credential_slot=prepared.credential_slot,
            inert_container_evidence=substituted_evidence,
        )


def test_prepared_execution_rejects_bind_source_aliasing():
    prepared = _prepared_execution()
    projection = prepared.inert_container_evidence.issued_create_projection
    mounts = list(projection.mounts)
    workspace_position = next(
        position
        for position, mount in enumerate(mounts)
        if mount.kind is RunActionPreparedMountKind.WORKSPACE
    )
    workspace_mount = mounts[workspace_position]
    workspace_walk = workspace_mount.source_walk
    aliased_workspace_walk = RunActionDescriptorWalkObservation.mint(
        root_authority_id=workspace_walk.root_authority_id,
        resolution_protocol_version=workspace_walk.resolution_protocol_version,
        nodes=(
            *workspace_walk.nodes[:-1],
            replace(
                workspace_walk.nodes[-1],
                identity=prepared.result_slot.descriptor_walk.nodes[-1].identity,
            ),
        ),
    )
    mounts[workspace_position] = replace(
        workspace_mount,
        source_walk=aliased_workspace_walk,
    )
    projection = _projection_with_mounts(
        projection,
        tuple(mounts),
    )

    with pytest.raises(RunActionSupervisorContractError, match="alias"):
        RunActionPreparedExecution.mint(
            preparation_claim=prepared.preparation_claim,
            input_slot=prepared.input_slot,
            result_slot=prepared.result_slot,
            credential_slot=prepared.credential_slot,
            inert_container_evidence=_evidence_with_projection(
                prepared.inert_container_evidence,
                projection,
            ),
        )


def test_prepared_execution_rejects_nested_bind_sources():
    prepared = _prepared_execution()
    projection = prepared.inert_container_evidence.issued_create_projection
    mounts = list(projection.mounts)
    result_mount = next(
        mount for mount in mounts if mount.kind is RunActionPreparedMountKind.RESULT
    )
    workspace_position = next(
        position
        for position, mount in enumerate(mounts)
        if mount.kind is RunActionPreparedMountKind.WORKSPACE
    )
    workspace_mount = mounts[workspace_position]
    workspace_walk = workspace_mount.source_walk
    nested_workspace_walk = RunActionDescriptorWalkObservation.mint(
        root_authority_id=workspace_walk.root_authority_id,
        resolution_protocol_version=workspace_walk.resolution_protocol_version,
        nodes=(
            workspace_walk.nodes[0],
            replace(
                result_mount.source_walk.nodes[-1],
                owner_user_id=workspace_walk.nodes[-1].owner_user_id,
                owner_group_id=workspace_walk.nodes[-1].owner_group_id,
            ),
            workspace_walk.nodes[-1],
        ),
    )
    mounts[workspace_position] = replace(
        workspace_mount,
        source_walk=nested_workspace_walk,
    )
    substituted_projection = _projection_with_mounts(projection, tuple(mounts))

    with pytest.raises(RunActionSupervisorContractError, match="contain"):
        RunActionPreparedExecution.mint(
            preparation_claim=prepared.preparation_claim,
            input_slot=prepared.input_slot,
            result_slot=prepared.result_slot,
            credential_slot=prepared.credential_slot,
            inert_container_evidence=_evidence_with_projection(
                prepared.inert_container_evidence,
                substituted_projection,
            ),
        )


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
    label_values = {label.value for label in first.inert_container_evidence.labels}
    assert first.prepared_execution_id not in label_values
    assert not any(
        "prepared" in label.key for label in first.inert_container_evidence.labels
    )


def test_credential_policy_and_slot_contracts_carry_no_secret_or_host_path_fields():
    forbidden_fragments = ("secret", "value", "host_path", "credential_path")

    for contract_type in (
        RunActionCredentialPolicy,
        RunActionCredentialDeliveryReceipt,
        RunActionPreparedSlot,
    ):
        names = tuple(field.name for field in fields(contract_type))
        assert not any(
            fragment in name for name in names for fragment in forbidden_fragments
        )


def test_credential_free_preparation_has_no_credential_slot_or_workspace_mount():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))

    assert prepared.credential_slot is None
    assert {
        mount.kind
        for mount in prepared.inert_container_evidence.issued_create_projection.mounts
    } == {
        RunActionPreparedMountKind.INPUT,
        RunActionPreparedMountKind.RESULT,
    }
