"""Canonical production policies for native coding-agent run actions."""

from __future__ import annotations

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_coding_agent_consumer import (
    NATIVE_CODING_AGENT_CONSUMER_ID,
    NATIVE_CODING_AGENT_CONSUMER_VERSION,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_NATIVE_TOOL_POLICY_VERSION,
    CODING_AGENT_REQUEST_PROTOCOL_VERSION,
    CODING_AGENT_RESULT_PROTOCOL_VERSION,
    CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
    CodingAgentInterpretationPolicy,
    CodingAgentProviderEgressMode,
)
from kapso.cross_run.launch.run_action_coding_agent_interpreter import (
    coding_agent_result_interpreter_identity,
)
from kapso.cross_run.launch.run_action_coding_agent_supervisor import (
    coding_agent_supervisor_command,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionExecutionLifecycleIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
    DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    DockerRunActionResourceLimits,
    DockerRunActionSafeCreateDefaults,
    DockerRunActionSandboxSpec,
    RunActionActivationNetworkMode,
    RunActionCredentialMode,
    RunActionCredentialPolicy,
    RunActionFilesystemPolicy,
    RunActionNetworkPolicy,
    RunActionStaticEnvironmentVariable,
    RunActionSupervisorLimits,
    RUN_ACTION_NETWORK_BROKER_DESTINATION,
    run_action_network_broker_endpoint_id,
    run_action_docker_init_authority_id,
    run_action_supervisor_helper_authority_id,
)
from kapso.cross_run.settings import (
    CodingAgentSettings,
    CrossRunSettings,
)

CODING_AGENT_EXECUTION_IMPLEMENTATION_ID = "kapso.coding_agent_docker_execution"
CODING_AGENT_EXECUTION_IMPLEMENTATION_VERSION = "kapso.coding_agent_docker_execution.v1"
CODING_AGENT_RECOVERY_PROTOCOL_VERSION = "kapso.run_action_recovery.v1"
CODING_AGENT_SUPERVISOR_PROTOCOL_VERSION = "kapso.run_action_supervisor.v1"
NATIVE_CODEX_CREDENTIAL_BROKER_ID = "kapso.native_codex_credential_broker"
NATIVE_CODEX_CREDENTIAL_BROKER_PROTOCOL_VERSION = (
    "kapso.native_codex_credential_broker.v1"
)
NATIVE_CODEX_CREDENTIAL_PRINCIPAL_ID = "kapso.native_codex"
NATIVE_CODEX_CREDENTIAL_AUDIENCE_ID = "codex_cli"
NATIVE_CODEX_CREDENTIAL_SCOPE_IDS = ("codex.invoke",)


class ProductionCodingAgentPolicyError(RuntimeError):
    """A coding-agent production policy cannot be derived exactly."""


def build_coding_agent_interpretation_policy(
    *,
    settings: CrossRunSettings,
    agent: CodingAgentSettings,
    principal_id: str,
    role: str,
    workspace_access: RunFrontierWorkspaceAccess,
    web_search_enabled: bool,
    provider_network_enabled: bool,
) -> CodingAgentInterpretationPolicy:
    """Bind configured provider behavior to the native consumer contract."""

    if (
        type(settings) is not CrossRunSettings
        or type(agent) is not CodingAgentSettings
        or type(workspace_access) is not RunFrontierWorkspaceAccess
        or type(web_search_enabled) is not bool
        or type(provider_network_enabled) is not bool
        or web_search_enabled
        and not provider_network_enabled
        or agent.timeout_seconds
        > settings.launch.coding_agent_action_execution_timeout_seconds
    ):
        raise ProductionCodingAgentPolicyError(
            "coding-agent interpretation inputs exceed configured authority"
        )
    launch = settings.launch
    egress_mode = (
        CodingAgentProviderEgressMode.HTTPS_CONNECT_PROXY
        if provider_network_enabled
        else CodingAgentProviderEgressMode.NONE
    )
    return CodingAgentInterpretationPolicy.mint(
        request_protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
        result_protocol_version=CODING_AGENT_RESULT_PROTOCOL_VERSION,
        schema_protocol_version=CODING_AGENT_SCHEMA_PROTOCOL_VERSION,
        consumer_id=NATIVE_CODING_AGENT_CONSUMER_ID,
        consumer_version=NATIVE_CODING_AGENT_CONSUMER_VERSION,
        principal_id=principal_id,
        role=role,
        cli=agent.cli,
        model=agent.model,
        effort=agent.effort,
        native_tool_policy_version=CODING_AGENT_NATIVE_TOOL_POLICY_VERSION,
        web_search_enabled=web_search_enabled,
        provider_egress_mode=egress_mode,
        egress_relay_port=(
            launch.coding_agent_egress_relay_port if provider_network_enabled else None
        ),
        egress_connect_authorities=(
            launch.coding_agent_egress_connect_authorities
            if provider_network_enabled
            else ()
        ),
        maximum_egress_connect_header_bytes=(
            launch.coding_agent_egress_connect_header_size_bytes
            if provider_network_enabled
            else None
        ),
        egress_relay_backlog=(
            launch.coding_agent_egress_relay_backlog
            if provider_network_enabled
            else None
        ),
        egress_relay_chunk_size_bytes=(
            launch.coding_agent_egress_relay_chunk_size_bytes
            if provider_network_enabled
            else None
        ),
        timeout_nanoseconds=agent.timeout_seconds * 1_000_000_000,
        termination_grace_nanoseconds=(
            launch.coding_agent_action_termination_grace_seconds * 1_000_000_000
        ),
        supervisor_user_id=launch.coding_agent_supervisor_user_id,
        supervisor_group_id=launch.coding_agent_supervisor_group_id,
        provider_user_id=launch.coding_agent_provider_user_id,
        provider_group_id=launch.coding_agent_provider_group_id,
        landlock_abi_version=launch.coding_agent_landlock_abi_version,
        workspace_access=workspace_access,
        workspace_git_branch=launch.workspace_git_branch,
        git_commit_author_name="Kapso Coding Agent",
        git_commit_author_email="coding-agent@kapso.invalid",
        maximum_request_bytes=launch.run_action_request_size_bytes,
        maximum_response_schema_bytes=(launch.coding_agent_response_schema_size_bytes),
        maximum_cli_argument_bytes=launch.coding_agent_cli_argument_size_bytes,
        maximum_provider_output_bytes=(launch.coding_agent_provider_output_size_bytes),
        maximum_provider_diagnostic_bytes=(
            launch.coding_agent_provider_diagnostic_size_bytes
        ),
        maximum_prior_knowledge_audit_bytes=(
            launch.coding_agent_prior_knowledge_audit_size_bytes
        ),
        prior_knowledge_relay_chunk_size_bytes=(
            launch.coding_agent_prior_knowledge_relay_chunk_size_bytes
        ),
        maximum_native_credential_bytes=(
            launch.coding_agent_native_credential_size_bytes
        ),
        maximum_workspace_entries=launch.run_workspace_entry_limit,
        maximum_workspace_bytes=launch.run_workspace_size_bytes,
        maximum_workspace_git_entries=launch.run_workspace_git_entry_limit,
        maximum_workspace_git_bytes=launch.run_workspace_git_metadata_size_bytes,
        maximum_raw_result_bytes=launch.run_action_result_size_bytes,
    )


def build_coding_agent_execution_policy(
    *,
    settings: CrossRunSettings,
    image_authority: DockerImageAuthority,
    interpretation_policy: CodingAgentInterpretationPolicy,
    credential_mode: RunActionCredentialMode,
    egress_broker_socket_source_path: str | None,
) -> tuple[DockerRunActionExecutionPolicy, DockerRunActionCommand]:
    """Build the exact Docker policy and supervisor command for one provider."""

    if (
        type(settings) is not CrossRunSettings
        or type(image_authority) is not DockerImageAuthority
        or type(interpretation_policy) is not CodingAgentInterpretationPolicy
        or type(credential_mode) is not RunActionCredentialMode
        or interpretation_policy.workspace_git_branch
        != settings.launch.workspace_git_branch
        or interpretation_policy.supervisor_user_id
        != settings.launch.coding_agent_supervisor_user_id
        or interpretation_policy.provider_user_id
        != settings.launch.coding_agent_provider_user_id
        or (
            interpretation_policy.provider_egress_mode
            is CodingAgentProviderEgressMode.HTTPS_CONNECT_PROXY
        )
        != (egress_broker_socket_source_path is not None)
    ):
        raise ProductionCodingAgentPolicyError(
            "coding-agent Docker inputs contain mixed authority"
        )
    if (interpretation_policy.cli == "codex") != (
        credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
    ):
        raise ProductionCodingAgentPolicyError(
            "native Codex requires its supervisor-delivered credential"
        )
    launch = settings.launch
    docker = settings.docker
    command_arguments = coding_agent_supervisor_command(interpretation_policy)
    command = DockerRunActionCommand.build(
        entrypoint=command_arguments[0],
        arguments=command_arguments[1:],
    )
    credential_policy = _credential_policy(settings, credential_mode)
    return (
        DockerRunActionExecutionPolicy.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            supervisor_protocol_version=CODING_AGENT_SUPERVISOR_PROTOCOL_VERSION,
            projection_protocol_version=(DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION),
            raw_field_schema_id=DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
            docker_runtime_settings_digest=tree_or_blob_digest(docker.to_json_bytes()),
            image_authority=image_authority,
            supervisor_helper_source_path=docker.helper_executable_path,
            supervisor_helper_executable_authority_id=(
                run_action_supervisor_helper_authority_id(
                    docker.helper_executable_path,
                    docker.helper_executable_digest,
                )
            ),
            supervisor_helper_executable_digest=docker.helper_executable_digest,
            docker_init_source_path=docker.init_executable_path,
            docker_init_executable_authority_id=(
                run_action_docker_init_authority_id(
                    docker.init_executable_path,
                    docker.init_executable_digest,
                )
            ),
            docker_init_executable_digest=docker.init_executable_digest,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(
                    key="PATH",
                    value="/usr/local/bin:/usr/bin:/bin",
                ),
            ),
            user_id=launch.coding_agent_supervisor_user_id,
            group_id=launch.coding_agent_supervisor_group_id,
            hostname=launch.coding_agent_action_hostname,
            safe_create_defaults=_safe_create_defaults(),
            sandbox_spec=_coding_agent_sandbox(settings),
            filesystem_policy=RunActionFilesystemPolicy.mint(
                workspace_access=interpretation_policy.workspace_access,
                workspace_destination="/kapso/workspace",
                input_destination="/kapso/input",
                result_destination="/kapso/result",
                credential_destination=(
                    "/kapso/credentials"
                    if credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
                    else None
                ),
                working_directory="/kapso/workspace",
                temporary_filesystem_destination="/kapso/tmp",
            ),
            network_policy=RunActionNetworkPolicy.mint(
                activation_mode=RunActionActivationNetworkMode.NONE,
                broker_endpoint_ids=(
                    ()
                    if egress_broker_socket_source_path is None
                    else (
                        run_action_network_broker_endpoint_id(
                            egress_broker_socket_source_path,
                            RUN_ACTION_NETWORK_BROKER_DESTINATION,
                        ),
                    )
                ),
                broker_socket_source_path=egress_broker_socket_source_path,
                broker_socket_destination_path=(
                    None
                    if egress_broker_socket_source_path is None
                    else RUN_ACTION_NETWORK_BROKER_DESTINATION
                ),
            ),
            credential_policy=credential_policy,
            docker_resource_limits=_resource_limits(settings),
            supervisor_limits=RunActionSupervisorLimits.mint(
                execution_timeout_seconds=(
                    launch.coding_agent_action_execution_timeout_seconds
                ),
                termination_grace_seconds=(
                    launch.coding_agent_action_termination_grace_seconds
                ),
                release_commit_timeout_seconds=(
                    launch.run_action_release_commit_timeout_seconds
                ),
                result_size_bytes=launch.run_action_result_size_bytes,
                release_receipt_size_bytes=(
                    launch.run_action_release_receipt_size_bytes
                ),
                timeout_directive_size_bytes=(
                    launch.run_action_timeout_directive_size_bytes
                ),
                process_snapshot_size_bytes=(
                    launch.run_action_process_snapshot_size_bytes
                ),
            ),
        ),
        command,
    )


def build_coding_agent_boundary_identity(
    execution_policy: DockerRunActionExecutionPolicy,
    interpretation_policy: CodingAgentInterpretationPolicy,
) -> RunActionBoundaryIdentity:
    """Join the production lifecycle and pure result interpreter identities."""

    if (
        type(execution_policy) is not DockerRunActionExecutionPolicy
        or type(interpretation_policy) is not CodingAgentInterpretationPolicy
        or execution_policy.kind is not RunFrontierActionKind.CODING_AGENT
    ):
        raise ProductionCodingAgentPolicyError(
            "coding-agent boundary requires exact production policies"
        )
    return RunActionBoundaryIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        execution_lifecycle_identity=RunActionExecutionLifecycleIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            implementation_id=CODING_AGENT_EXECUTION_IMPLEMENTATION_ID,
            implementation_version=CODING_AGENT_EXECUTION_IMPLEMENTATION_VERSION,
            recovery_protocol_version=CODING_AGENT_RECOVERY_PROTOCOL_VERSION,
            execution_policy_id=execution_policy.docker_execution_policy_id,
        ),
        result_interpreter_identity=coding_agent_result_interpreter_identity(
            interpretation_policy
        ),
    )


def _credential_policy(
    settings: CrossRunSettings,
    credential_mode: RunActionCredentialMode,
) -> RunActionCredentialPolicy:
    if credential_mode is RunActionCredentialMode.NONE:
        return RunActionCredentialPolicy.mint(
            mode=credential_mode,
            broker_id=None,
            broker_protocol_version=None,
            principal_id=None,
            audience_id=None,
            scope_ids=(),
            maximum_lease_seconds=None,
            maximum_delivery_size_bytes=None,
        )
    return RunActionCredentialPolicy.mint(
        mode=credential_mode,
        broker_id=NATIVE_CODEX_CREDENTIAL_BROKER_ID,
        broker_protocol_version=NATIVE_CODEX_CREDENTIAL_BROKER_PROTOCOL_VERSION,
        principal_id=NATIVE_CODEX_CREDENTIAL_PRINCIPAL_ID,
        audience_id=NATIVE_CODEX_CREDENTIAL_AUDIENCE_ID,
        scope_ids=NATIVE_CODEX_CREDENTIAL_SCOPE_IDS,
        maximum_lease_seconds=(
            settings.launch.coding_agent_action_credential_lease_seconds
        ),
        maximum_delivery_size_bytes=(
            settings.launch.coding_agent_native_credential_size_bytes
        ),
    )


def _safe_create_defaults() -> DockerRunActionSafeCreateDefaults:
    return DockerRunActionSafeCreateDefaults.mint(
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
    )


def _coding_agent_sandbox(settings: CrossRunSettings) -> DockerRunActionSandboxSpec:
    launch = settings.launch
    return DockerRunActionSandboxSpec.mint(
        read_only_root_filesystem=True,
        privileged=False,
        capability_additions=("KILL", "SETGID", "SETPCAP", "SETUID"),
        capability_drops=("ALL",),
        device_authority_ids=(),
        device_request_authority_ids=(),
        device_cgroup_rule_ids=(),
        supplementary_group_ids=(launch.coding_agent_provider_group_id,),
        pid_namespace_mode="private",
        ipc_namespace_mode="private",
        uts_namespace_mode="private",
        cgroup_namespace_mode="private",
        user_namespace_mode="daemon_default_unmapped",
        cgroup_parent_id=launch.coding_agent_action_cgroup_parent_id,
        sysctl_ids=(),
        no_new_privileges=False,
        seccomp_profile_id="builtin",
        apparmor_profile_id="docker-default",
        security_option_ids=("apparmor:docker-default", "seccomp:builtin"),
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
        runtime_id=settings.docker.runtime_default_runtime,
        log_driver="none",
        log_option_ids=(),
        init_process=True,
        isolation_mode="default",
    )


def _resource_limits(settings: CrossRunSettings) -> DockerRunActionResourceLimits:
    launch = settings.launch
    return DockerRunActionResourceLimits.mint(
        cpu_period_microseconds=(launch.coding_agent_action_cpu_period_microseconds),
        cpu_quota_microseconds=launch.coding_agent_action_cpu_quota_microseconds,
        cpu_shares=launch.coding_agent_action_cpu_shares,
        cpuset_cpu_ids=(),
        cpuset_memory_node_ids=(),
        memory_size_bytes=launch.coding_agent_action_memory_size_bytes,
        memory_reservation_size_bytes=(
            launch.coding_agent_action_memory_reservation_size_bytes
        ),
        memory_swap_size_bytes=launch.coding_agent_action_memory_swap_size_bytes,
        oom_score_adjustment=0,
        process_limit=launch.coding_agent_action_process_limit,
        block_io_weight=launch.coding_agent_action_block_io_weight,
        shared_memory_size_bytes=(launch.coding_agent_action_shared_memory_size_bytes),
        runtime_volume_size_bytes=(
            launch.coding_agent_action_runtime_volume_size_bytes
        ),
        runtime_volume_inode_limit=(
            launch.coding_agent_action_runtime_volume_inode_limit
        ),
        runtime_temporary_reservation_size_bytes=(
            launch.coding_agent_action_temporary_reservation_size_bytes
        ),
        runtime_temporary_reservation_inode_count=(
            launch.coding_agent_action_temporary_reservation_inode_count
        ),
    )


__all__ = [
    "build_coding_agent_boundary_identity",
    "build_coding_agent_execution_policy",
    "build_coding_agent_interpretation_policy",
    "CODING_AGENT_EXECUTION_IMPLEMENTATION_ID",
    "CODING_AGENT_EXECUTION_IMPLEMENTATION_VERSION",
    "CODING_AGENT_RECOVERY_PROTOCOL_VERSION",
    "NATIVE_CODEX_CREDENTIAL_BROKER_ID",
    "NATIVE_CODEX_CREDENTIAL_BROKER_PROTOCOL_VERSION",
    "ProductionCodingAgentPolicyError",
]
