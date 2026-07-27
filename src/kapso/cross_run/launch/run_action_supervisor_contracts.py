"""Content-addressed preparation contracts for inert run-action containers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar

from kapso.cross_run.canonical import (
    content_id,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit

_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:[.](?P<fraction>[0-9]{1,9}))?Z$"
)
_GENERATION_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_ENVIRONMENT_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:ACCESS_KEY(?:_ID)?|ACCESS_TOKEN|API_KEY|AUTH_CONFIG|AUTH_TOKEN|"
    r"CREDENTIALS?|NETRC|OAUTH_TOKEN|PASSWORD|PASSWD|PAT|PRIVATE_KEY|"
    r"SECRET(?:_ACCESS_KEY)?|SECRETS?|TOKEN)(?:_|$)"
)
_STATIC_ENVIRONMENT_KEYS = {"LANG", "LC_ALL", "NO_COLOR", "PATH", "TERM"}
_STATIC_ENVIRONMENT_EXACT_VALUES = {
    "LANG": {"C", "C.UTF-8"},
    "LC_ALL": {"C", "C.UTF-8"},
    "NO_COLOR": {"1"},
    "PATH": {
        "/bin",
        "/usr/bin:/bin",
        "/usr/local/bin:/usr/bin:/bin",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    },
    "TERM": {"dumb", "xterm", "xterm-256color"},
}
_LABEL_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9-]+)*$")
_SYSTEMD_CGROUP_SLICE_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.]*[A-Za-z0-9])?"
    r"(?:-[A-Za-z0-9](?:[A-Za-z0-9_.]*[A-Za-z0-9])?)*\.slice$"
)
_DOCKER_29_1_3_MASKED_SYSTEM_PATHS = (
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
)
_DOCKER_29_1_3_READ_ONLY_SYSTEM_PATHS = (
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
)
_DOCKER_MINIMUM_MEMORY_BYTES = 6 * 1024 * 1024
_DOCKER_MAXIMUM_USER_OR_GROUP_ID = 2_147_483_647
RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER = (1 << 64) - 1
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_CONTAINER_NAME_PREFIX = "kapso-run-action-"
_KEEPER_CONTAINER_NAME_PREFIX = "kapso-run-action-keeper-"
_RUNTIME_VOLUME_NAME_PREFIX = "kapso-run-action-volume-"
_PREPARATION_LABEL_PREFIX = "com.kapso.run-action."
_RUNTIME_VOLUME_SENTINEL_PATH = ".kapso-generation"
RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION = "/kapso/runtime-volume"
RUN_ACTION_SUPERVISOR_HELPER_DESTINATION = "/kapso-supervisor/busybox"
RUN_ACTION_DOCKER_INIT_DESTINATION = "/sbin/docker-init"
RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE = (
    "run-action-credential-lease-authority"
)
RUN_ACTION_BARRIER_CONTROL_DESTINATION = "/kapso-supervisor/control"
RUN_ACTION_BARRIER_RELEASE_DESTINATION = "/kapso-supervisor/control/release"
RUN_ACTION_BARRIER_PROTOCOL_VERSION = "kapso.run_action_barrier.v2"
RUN_ACTION_BARRIER_SCRIPT = (
    'while [ ! -f "$1" ] || [ ! -r "$1" ]'
    ' || ! "$2" grep -Fq "$4" "$1"; do "$2" sleep "$3"; done; '
    'shift 4; exec "$@"'
)
RUN_ACTION_BARRIER_DUMMY_ARGUMENT = "kapso-run-action-barrier"
_RUNTIME_VOLUME_SUBPATHS = {
    "workspace": "workspace",
    "input": "input",
    "result": "result",
    "credential": "credential",
    "temporary": "temporary",
    "control": "control",
}


class RunActionSupervisorContractError(ValueError):
    """A prepared execution cannot prove the exact inert Docker occurrence."""


def _bounded_physical_integer(value: object, minimum: int) -> bool:
    return (
        type(value) is int and minimum <= value <= RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
    )


class RunActionCredentialMode(str, Enum):
    """How a committed execution may receive provider credentials."""

    NONE = "none"
    SUPERVISOR_FILE = "supervisor_file"


class RunActionActivationNetworkMode(str, Enum):
    """Network authority that may be attached only after spawn commit."""

    NONE = "none"


class RunActionPreparedFileKind(str, Enum):
    """Purpose of one logical payload inside the private runtime volume."""

    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"


class RunActionPreparedRuntimeDirectoryKind(str, Enum):
    """Purpose of one non-delivery runtime directory mounted into the main."""

    RESULT = "result"
    TEMPORARY = "temporary"
    CONTROL = "control"


class RunActionPreparedMountKind(str, Enum):
    """Identity of one runtime-volume subpath admitted to the main container."""

    WORKSPACE = "workspace"
    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"
    TEMPORARY = "temporary"
    CONTROL = "control"


class RunActionPreparedMountAccess(str, Enum):
    """Access granted to a volume source or its container mount."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class RunActionStaticEnvironmentVariable(StrictContract):
    """One non-secret static environment entry in an execution policy."""

    key: str
    value: str

    def _validate(self) -> None:
        if (
            _ENVIRONMENT_KEY_PATTERN.fullmatch(self.key) is None
            or self.key not in _STATIC_ENVIRONMENT_KEYS
            or _SECRET_ENVIRONMENT_KEY_PATTERN.search(self.key) is not None
            or not isinstance(self.value, str)
            or not self.value
            or "\x00" in self.value
        ):
            raise RunActionSupervisorContractError(
                "run action static environment contains an invalid or secret-like entry"
            )
        if (
            self.key in _STATIC_ENVIRONMENT_EXACT_VALUES
            and self.value not in _STATIC_ENVIRONMENT_EXACT_VALUES[self.key]
        ):
            raise RunActionSupervisorContractError(
                "run action static environment value is outside its exact allowlist"
            )


@dataclass(frozen=True)
class RunActionCredentialPolicy(StrictContract):
    """Non-secret credential authority; values are delivered only after commit."""

    credential_policy_id: str
    mode: RunActionCredentialMode
    broker_id: str | None
    broker_protocol_version: str | None
    principal_id: str | None
    audience_id: str | None
    scope_ids: tuple[str, ...]
    maximum_lease_seconds: int | None
    maximum_delivery_size_bytes: int | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-policy"
    IDENTITY_FIELD: ClassVar[str] = "credential_policy_id"

    def _validate(self) -> None:
        if type(self.mode) is not RunActionCredentialMode:
            raise RunActionSupervisorContractError(
                "run action credential policy uses an unknown mode"
            )
        credential_fields = (
            self.broker_id,
            self.broker_protocol_version,
            self.principal_id,
            self.audience_id,
            self.maximum_lease_seconds,
            self.maximum_delivery_size_bytes,
        )
        if self.mode is RunActionCredentialMode.NONE:
            if any(value is not None for value in credential_fields) or self.scope_ids:
                raise RunActionSupervisorContractError(
                    "credential-free run action policy carries credential authority"
                )
            return
        for value, name in (
            (self.broker_id, "broker"),
            (self.broker_protocol_version, "broker protocol"),
            (self.principal_id, "principal"),
            (self.audience_id, "audience"),
        ):
            require_identifier(value, f"run action credential {name}")
        if (
            not self.scope_ids
            or self.scope_ids != tuple(sorted(set(self.scope_ids)))
            or any(
                not isinstance(scope_id, str)
                or require_identifier(scope_id, "run action credential scope")
                != scope_id
                for scope_id in self.scope_ids
            )
            or type(self.maximum_lease_seconds) is not int
            or self.maximum_lease_seconds <= 0
            or type(self.maximum_delivery_size_bytes) is not int
            or self.maximum_delivery_size_bytes <= 0
        ):
            raise RunActionSupervisorContractError(
                "brokered run action credential policy is invalid"
            )


@dataclass(frozen=True)
class RunActionNetworkPolicy(StrictContract):
    """Network authority attached after commit, never during preparation."""

    network_policy_id: str
    activation_mode: RunActionActivationNetworkMode
    broker_endpoint_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-network-policy"
    IDENTITY_FIELD: ClassVar[str] = "network_policy_id"

    def _validate(self) -> None:
        if (
            self.activation_mode is not RunActionActivationNetworkMode.NONE
            or self.broker_endpoint_ids
        ):
            raise RunActionSupervisorContractError(
                "run action network policy must deny all network authority"
            )


@dataclass(frozen=True)
class DockerRunActionResourceLimits(StrictContract):
    """Resource controls observable in the retained Docker configuration."""

    docker_resource_limits_id: str
    cpu_period_microseconds: int
    cpu_quota_microseconds: int
    cpu_shares: int
    cpuset_cpu_ids: tuple[int, ...]
    cpuset_memory_node_ids: tuple[int, ...]
    memory_size_bytes: int
    memory_reservation_size_bytes: int
    memory_swap_size_bytes: int
    oom_score_adjustment: int
    process_limit: int
    block_io_weight: int
    shared_memory_size_bytes: int
    runtime_volume_size_bytes: int
    runtime_volume_inode_limit: int
    runtime_temporary_reservation_size_bytes: int
    runtime_temporary_reservation_inode_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-resource-limits"
    IDENTITY_FIELD: ClassVar[str] = "docker_resource_limits_id"

    def _validate(self) -> None:
        values = (
            self.cpu_period_microseconds,
            self.cpu_quota_microseconds,
            self.cpu_shares,
            self.memory_size_bytes,
            self.memory_reservation_size_bytes,
            self.memory_swap_size_bytes,
            self.process_limit,
            self.block_io_weight,
            self.shared_memory_size_bytes,
            self.runtime_volume_size_bytes,
            self.runtime_volume_inode_limit,
            self.runtime_temporary_reservation_size_bytes,
            self.runtime_temporary_reservation_inode_count,
        )
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or not 1_000 <= self.cpu_period_microseconds <= 1_000_000
            or self.cpu_quota_microseconds < 1_000
            or not 2 <= self.cpu_shares <= 262_144
            or self.memory_size_bytes < _DOCKER_MINIMUM_MEMORY_BYTES
            or self.memory_reservation_size_bytes < _DOCKER_MINIMUM_MEMORY_BYTES
            or self.memory_swap_size_bytes < self.memory_size_bytes
            or self.memory_reservation_size_bytes > self.memory_size_bytes
            or type(self.oom_score_adjustment) is not int
            or not -1000 <= self.oom_score_adjustment <= 1000
            or self.cpuset_cpu_ids != tuple(sorted(set(self.cpuset_cpu_ids)))
            or any(type(value) is not int or value < 0 for value in self.cpuset_cpu_ids)
            or self.cpuset_memory_node_ids
            != tuple(sorted(set(self.cpuset_memory_node_ids)))
            or any(
                type(value) is not int or value < 0
                for value in self.cpuset_memory_node_ids
            )
            or not 10 <= self.block_io_weight <= 1_000
        ):
            raise RunActionSupervisorContractError(
                "Docker run action resource limits are invalid"
            )


@dataclass(frozen=True)
class RunActionSupervisorLimits(StrictContract):
    """Non-Docker time and byte bounds enforced by the trusted supervisor."""

    supervisor_limits_id: str
    execution_timeout_seconds: int
    termination_grace_seconds: int
    release_commit_timeout_seconds: int
    result_size_bytes: int
    release_receipt_size_bytes: int
    timeout_directive_size_bytes: int
    process_snapshot_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-supervisor-limits"
    IDENTITY_FIELD: ClassVar[str] = "supervisor_limits_id"

    def _validate(self) -> None:
        values = (
            self.execution_timeout_seconds,
            self.termination_grace_seconds,
            self.release_commit_timeout_seconds,
            self.result_size_bytes,
            self.release_receipt_size_bytes,
            self.timeout_directive_size_bytes,
            self.process_snapshot_size_bytes,
        )
        if (
            any(not _bounded_physical_integer(value, 1) for value in values)
            or self.termination_grace_seconds >= self.execution_timeout_seconds
            or self.release_commit_timeout_seconds >= self.execution_timeout_seconds
        ):
            raise RunActionSupervisorContractError(
                "run action supervisor limits are invalid"
            )


@dataclass(frozen=True)
class DockerRunActionSandboxSpec(StrictContract):
    """Exact privilege envelope corresponding to one lifecycle sandbox policy."""

    docker_sandbox_spec_id: str
    read_only_root_filesystem: bool
    privileged: bool
    capability_additions: tuple[str, ...]
    capability_drops: tuple[str, ...]
    device_authority_ids: tuple[str, ...]
    device_request_authority_ids: tuple[str, ...]
    device_cgroup_rule_ids: tuple[str, ...]
    supplementary_group_ids: tuple[int, ...]
    pid_namespace_mode: str
    ipc_namespace_mode: str
    uts_namespace_mode: str
    cgroup_namespace_mode: str
    user_namespace_mode: str
    cgroup_parent_id: str
    sysctl_ids: tuple[str, ...]
    no_new_privileges: bool
    seccomp_profile_id: str
    apparmor_profile_id: str
    security_option_ids: tuple[str, ...]
    masked_system_paths: tuple[str, ...]
    read_only_system_paths: tuple[str, ...]
    runtime_id: str
    log_driver: str
    log_option_ids: tuple[str, ...]
    init_process: bool
    isolation_mode: str

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-sandbox-spec"
    IDENTITY_FIELD: ClassVar[str] = "docker_sandbox_spec_id"

    def _validate(self) -> None:
        if (
            self.read_only_root_filesystem is not True
            or self.privileged is not False
            or self.capability_additions
            or self.capability_drops != ("ALL",)
            or self.device_authority_ids
            or self.device_request_authority_ids
            or self.device_cgroup_rule_ids
            or self.supplementary_group_ids
            or self.pid_namespace_mode != "private"
            or self.ipc_namespace_mode != "private"
            or self.uts_namespace_mode != "private"
            or self.cgroup_namespace_mode != "private"
            or self.user_namespace_mode != "daemon_default_unmapped"
            or not self.cgroup_parent_id.isascii()
            or len(self.cgroup_parent_id.encode("ascii")) > 255
            or _SYSTEMD_CGROUP_SLICE_PATTERN.fullmatch(self.cgroup_parent_id) is None
            or self.sysctl_ids
            or self.no_new_privileges is not True
            or self.seccomp_profile_id != "builtin"
            or self.apparmor_profile_id != "docker-default"
            or self.security_option_ids
            != (
                "apparmor:docker-default",
                "no-new-privileges",
                "seccomp:builtin",
            )
            or self.masked_system_paths != _DOCKER_29_1_3_MASKED_SYSTEM_PATHS
            or self.read_only_system_paths != _DOCKER_29_1_3_READ_ONLY_SYSTEM_PATHS
            or self.runtime_id != "runc"
            or self.log_driver != "none"
            or self.log_option_ids
            or self.init_process is not True
            or self.isolation_mode != "default"
        ):
            raise RunActionSupervisorContractError(
                "Docker run action sandbox permits expanded privilege"
            )


@dataclass(frozen=True)
class RunActionFilesystemPolicy(StrictContract):
    """Prefix-disjoint container destinations for runtime-volume subpaths."""

    filesystem_policy_id: str
    workspace_access: RunFrontierWorkspaceAccess
    workspace_destination: str | None
    input_destination: str
    result_destination: str
    credential_destination: str | None
    working_directory: str
    temporary_filesystem_destination: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-filesystem-policy"
    IDENTITY_FIELD: ClassVar[str] = "filesystem_policy_id"

    def _validate(self) -> None:
        if type(self.workspace_access) is not RunFrontierWorkspaceAccess:
            raise RunActionSupervisorContractError(
                "run action filesystem policy uses unknown workspace access"
            )
        if (self.workspace_access is RunFrontierWorkspaceAccess.NONE) != (
            self.workspace_destination is None
        ):
            raise RunActionSupervisorContractError(
                "run action workspace destination differs from its access"
            )
        destinations = (
            self.input_destination,
            self.result_destination,
            self.working_directory,
            self.temporary_filesystem_destination,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            RUN_ACTION_BARRIER_CONTROL_DESTINATION,
            *(
                ()
                if self.workspace_destination is None
                else (self.workspace_destination,)
            ),
            *(
                ()
                if self.credential_destination is None
                else (self.credential_destination,)
            ),
        )
        for destination in destinations:
            _require_absolute_container_path(destination)
        workload_mount_destinations = tuple(
            destination
            for destination in (
                self.workspace_destination,
                self.input_destination,
                self.result_destination,
                self.credential_destination,
                self.temporary_filesystem_destination,
            )
            if destination is not None
        )
        mount_destinations = (
            *workload_mount_destinations,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            RUN_ACTION_BARRIER_CONTROL_DESTINATION,
        )
        if any(
            left == right
            or left in PurePosixPath(right).parents
            or right in PurePosixPath(left).parents
            for position, left in enumerate(
                PurePosixPath(destination) for destination in mount_destinations
            )
            for right in (
                PurePosixPath(destination)
                for destination in mount_destinations[position + 1 :]
            )
        ):
            raise RunActionSupervisorContractError(
                "run action container mount destinations overlap"
            )
        working_directory = PurePosixPath(self.working_directory)
        if not any(
            working_directory == PurePosixPath(destination)
            or PurePosixPath(destination) in working_directory.parents
            for destination in workload_mount_destinations
        ):
            raise RunActionSupervisorContractError(
                "run action working directory is outside its mounted filesystems"
            )


@dataclass(frozen=True)
class DockerRunActionSafeCreateDefaults(StrictContract):
    """Security-relevant Docker create fields that must remain inert or absent."""

    safe_create_defaults_id: str
    open_stdin: bool
    terminal: bool
    stdin_once: bool
    attach_stdin: bool
    exposed_port_ids: tuple[str, ...]
    port_binding_ids: tuple[str, ...]
    publish_all_ports: bool
    link_ids: tuple[str, ...]
    extra_host_ids: tuple[str, ...]
    dns_server_ids: tuple[str, ...]
    dns_search_ids: tuple[str, ...]
    dns_option_ids: tuple[str, ...]
    endpoint_alias_ids: tuple[str, ...]
    volume_from_ids: tuple[str, ...]
    storage_option_ids: tuple[str, ...]
    anonymous_volume_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-safe-create-defaults"
    IDENTITY_FIELD: ClassVar[str] = "safe_create_defaults_id"

    def _validate(self) -> None:
        if (
            self.open_stdin is not False
            or self.terminal is not False
            or self.stdin_once is not False
            or self.attach_stdin is not False
            or self.publish_all_ports is not False
            or self.anonymous_volume_count != 0
            or any(
                values
                for values in (
                    self.exposed_port_ids,
                    self.port_binding_ids,
                    self.link_ids,
                    self.extra_host_ids,
                    self.dns_server_ids,
                    self.dns_search_ids,
                    self.dns_option_ids,
                    self.endpoint_alias_ids,
                    self.volume_from_ids,
                    self.storage_option_ids,
                )
            )
        ):
            raise RunActionSupervisorContractError(
                "Docker run action create defaults expand authority"
            )


@dataclass(frozen=True)
class DockerRunActionExecutionPolicy(StrictContract):
    """Lifecycle-owned, content-addressed Docker execution policy."""

    docker_execution_policy_id: str
    kind: RunFrontierActionKind
    supervisor_protocol_version: str
    projection_protocol_version: str
    raw_field_schema_id: str
    docker_runtime_settings_digest: str
    image_authority: DockerImageAuthority
    supervisor_helper_source_path: str
    supervisor_helper_executable_authority_id: str
    supervisor_helper_executable_digest: str
    docker_init_source_path: str
    docker_init_executable_authority_id: str
    docker_init_executable_digest: str
    command_template_id: str
    static_environment: tuple[RunActionStaticEnvironmentVariable, ...]
    user_id: int
    group_id: int
    hostname: str
    safe_create_defaults: DockerRunActionSafeCreateDefaults
    sandbox_spec: DockerRunActionSandboxSpec
    filesystem_policy: RunActionFilesystemPolicy
    network_policy: RunActionNetworkPolicy
    credential_policy: RunActionCredentialPolicy
    docker_resource_limits: DockerRunActionResourceLimits
    supervisor_limits: RunActionSupervisorLimits

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-execution-policy"
    IDENTITY_FIELD: ClassVar[str] = "docker_execution_policy_id"

    def _validate(self) -> None:
        if (
            type(self.kind) is not RunFrontierActionKind
            or type(self.image_authority) is not DockerImageAuthority
            or type(self.safe_create_defaults) is not DockerRunActionSafeCreateDefaults
            or type(self.filesystem_policy) is not RunActionFilesystemPolicy
            or type(self.network_policy) is not RunActionNetworkPolicy
            or type(self.credential_policy) is not RunActionCredentialPolicy
            or type(self.docker_resource_limits) is not DockerRunActionResourceLimits
            or type(self.supervisor_limits) is not RunActionSupervisorLimits
            or type(self.sandbox_spec) is not DockerRunActionSandboxSpec
            or _SHA256_DIGEST_PATTERN.fullmatch(self.docker_runtime_settings_digest)
            is None
            or any(
                type(variable) is not RunActionStaticEnvironmentVariable
                for variable in self.static_environment
            )
            or tuple(variable.key for variable in self.static_environment)
            != tuple(sorted({variable.key for variable in self.static_environment}))
            or type(self.user_id) is not int
            or self.user_id <= 0
            or self.user_id > _DOCKER_MAXIMUM_USER_OR_GROUP_ID
            or type(self.group_id) is not int
            or self.group_id <= 0
            or self.group_id > _DOCKER_MAXIMUM_USER_OR_GROUP_ID
        ):
            raise RunActionSupervisorContractError(
                "Docker run action execution policy is invalid"
            )
        for value, name in (
            (self.supervisor_protocol_version, "supervisor protocol"),
            (self.projection_protocol_version, "projection protocol"),
            (self.hostname, "hostname"),
        ):
            require_identifier(value, f"run action {name}")
        _require_namespaced_content_id(
            self.command_template_id,
            "docker-run-action-command-template",
            "Docker run action command template",
        )
        _require_namespaced_content_id(
            self.raw_field_schema_id,
            "docker-raw-field-schema",
            "Docker raw-field schema",
        )
        _require_namespaced_content_id(
            self.supervisor_helper_executable_authority_id,
            "run-action-helper-executable-authority",
            "runtime supervisor helper",
        )
        _require_absolute_host_path(
            self.supervisor_helper_source_path,
            "runtime supervisor helper source",
        )
        if _SHA256_DIGEST_PATTERN.fullmatch(
            self.supervisor_helper_executable_digest
        ) is None or self.supervisor_helper_executable_authority_id != run_action_supervisor_helper_authority_id(
            self.supervisor_helper_source_path,
            self.supervisor_helper_executable_digest,
        ):
            raise RunActionSupervisorContractError(
                "runtime supervisor helper differs from execution policy"
            )
        _require_namespaced_content_id(
            self.docker_init_executable_authority_id,
            "run-action-docker-init-executable-authority",
            "Docker init executable",
        )
        _require_absolute_host_path(
            self.docker_init_source_path,
            "Docker init executable source",
        )
        if _SHA256_DIGEST_PATTERN.fullmatch(
            self.docker_init_executable_digest
        ) is None or self.docker_init_executable_authority_id != run_action_docker_init_authority_id(
            self.docker_init_source_path,
            self.docker_init_executable_digest,
        ):
            raise RunActionSupervisorContractError(
                "Docker init executable differs from execution policy"
            )
        has_credential_destination = (
            self.filesystem_policy.credential_destination is not None
        )
        if has_credential_destination != (
            self.credential_policy.mode is RunActionCredentialMode.SUPERVISOR_FILE
        ):
            raise RunActionSupervisorContractError(
                "run action credential destination differs from credential policy"
            )
        if (
            self.credential_policy.mode is RunActionCredentialMode.SUPERVISOR_FILE
            and self.credential_policy.maximum_lease_seconds
            < (
                self.supervisor_limits.execution_timeout_seconds
                + self.supervisor_limits.termination_grace_seconds
            )
        ):
            raise RunActionSupervisorContractError(
                "run action credential lease cannot span containment"
            )


@dataclass(frozen=True)
class RunActionPreparationClaim(StrictContract):
    """Deterministic semantic claim used before Docker allocates an occurrence."""

    preparation_claim_id: str
    reservation: RunActionReservation
    execution_policy: DockerRunActionExecutionPolicy

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-preparation-claim"
    IDENTITY_FIELD: ClassVar[str] = "preparation_claim_id"

    def _validate(self) -> None:
        if (
            type(self.reservation) is not RunActionReservation
            or type(self.execution_policy) is not DockerRunActionExecutionPolicy
            or self.reservation.intent.kind is not self.execution_policy.kind
            or self.reservation.intent.boundary_identity.execution_lifecycle_identity.execution_policy_id
            != self.execution_policy.docker_execution_policy_id
            or self.reservation.intent.workspace_access
            is not self.execution_policy.filesystem_policy.workspace_access
        ):
            raise RunActionSupervisorContractError(
                "run action preparation claim differs from its durable reservation"
            )


@dataclass(frozen=True)
class RunActionRuntimeVolumeAuthority(StrictContract):
    """Issued local-driver tmpfs authority for one private runtime volume."""

    runtime_volume_authority_id: str
    preparation_claim_id: str
    volume_name: str
    labels: tuple[RunActionContainerLabel, ...]
    driver: str
    driver_options: tuple[str, ...]
    generation_nonce: str
    sentinel_relative_path: str
    sentinel_identity: str
    owner_user_id: int
    owner_group_id: int
    root_mode: int
    size_limit_bytes: int
    inode_limit: int
    nosuid: bool
    nodev: bool
    noswap: bool
    execution_allowed: bool

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-runtime-volume-authority"
    IDENTITY_FIELD: ClassVar[str] = "runtime_volume_authority_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "runtime volume preparation claim",
        )
        _require_namespaced_content_id(
            self.sentinel_identity,
            "run-action-runtime-volume-sentinel",
            "runtime volume sentinel",
        )
        label_values = {
            label.key: label.value
            for label in self.labels
            if type(label) is RunActionContainerLabel
        }
        expected_label_keys = tuple(
            sorted(
                (
                    f"{_PREPARATION_LABEL_PREFIX}claim",
                    f"{_PREPARATION_LABEL_PREFIX}generation",
                    f"{_PREPARATION_LABEL_PREFIX}reservation",
                    f"{_PREPARATION_LABEL_PREFIX}role",
                )
            )
        )
        if (
            any(type(label) is not RunActionContainerLabel for label in self.labels)
            or tuple(label.key for label in self.labels) != expected_label_keys
        ):
            raise RunActionSupervisorContractError(
                "run action runtime volume authority labels are invalid"
            )
        _require_namespaced_content_id(
            label_values[f"{_PREPARATION_LABEL_PREFIX}reservation"],
            RunActionReservation.CONTENT_NAMESPACE,
            "runtime volume reservation label",
        )
        if (
            self.volume_name
            != _RUNTIME_VOLUME_NAME_PREFIX
            + self.preparation_claim_id.rsplit(":sha256:", 1)[1]
            or label_values[f"{_PREPARATION_LABEL_PREFIX}claim"]
            != self.preparation_claim_id
            or label_values[f"{_PREPARATION_LABEL_PREFIX}generation"]
            != self.sentinel_identity
            or label_values[f"{_PREPARATION_LABEL_PREFIX}role"] != "runtime-volume"
            or self.driver != "local"
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.sentinel_relative_path != _RUNTIME_VOLUME_SENTINEL_PATH
            or self.sentinel_identity
            != runtime_volume_sentinel_identity(self.generation_nonce)
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.root_mode != 0o700
            or not _bounded_physical_integer(self.size_limit_bytes, 1)
            or not _bounded_physical_integer(self.inode_limit, 1)
            or self.nosuid is not True
            or self.nodev is not True
            or self.noswap is not True
            or self.execution_allowed is not True
            or self.driver_options != runtime_volume_driver_options(self)
        ):
            raise RunActionSupervisorContractError(
                "run action runtime volume authority is invalid"
            )


@dataclass(frozen=True)
class RunActionPreparationAllocation(StrictContract):
    """Logical authority for one generation before physical materialization."""

    preparation_allocation_id: str
    preparation_claim: RunActionPreparationClaim
    runtime_volume_authority: RunActionRuntimeVolumeAuthority

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-preparation-allocation"
    IDENTITY_FIELD: ClassVar[str] = "preparation_allocation_id"

    def _validate(self) -> None:
        expected_authority = (
            issue_runtime_volume_authority(
                self.preparation_claim,
                self.runtime_volume_authority.generation_nonce,
            )
            if type(self.preparation_claim) is RunActionPreparationClaim
            and type(self.runtime_volume_authority) is RunActionRuntimeVolumeAuthority
            else None
        )
        if (
            type(self.preparation_claim) is not RunActionPreparationClaim
            or type(self.runtime_volume_authority)
            is not RunActionRuntimeVolumeAuthority
            or self.runtime_volume_authority != expected_authority
        ):
            raise RunActionSupervisorContractError(
                "run action preparation allocation differs from its exact claim"
            )


@dataclass(frozen=True)
class RunActionRuntimeVolumeSentinelEvidence(StrictContract):
    """No-follow physical evidence for one in-volume generation sentinel."""

    runtime_volume_sentinel_evidence_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-runtime-volume-sentinel-evidence"
    IDENTITY_FIELD: ClassVar[str] = "runtime_volume_sentinel_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "runtime volume sentinel authority",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != _RUNTIME_VOLUME_SENTINEL_PATH
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o400
            or self.link_count != 1
            or self.size_bytes != len(self.generation_nonce)
            or self.content_digest
            != tree_or_blob_digest(self.generation_nonce.encode("ascii"))
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "runtime volume sentinel evidence is not one stable physical file"
            )


@dataclass(frozen=True)
class RunActionRuntimeVolumeEvidence(StrictContract):
    """Effective tmpfs identity and limits observed through the mounted keeper."""

    runtime_volume_evidence_id: str
    volume_authority: RunActionRuntimeVolumeAuthority
    docker_volume_occurrence_digest: str
    volume_keeper_evidence_id: str
    keeper_container_id: str
    keeper_process_id: int
    keeper_process_start_time_ticks: int
    keeper_process_cgroup_path: str
    root_mount_id: int
    root_device: int
    root_inode: int
    observed_volume_name: str
    observed_labels: tuple[RunActionContainerLabel, ...]
    observed_scope: str
    observed_driver: str
    observed_driver_options: tuple[str, ...]
    observed_filesystem_type: str
    observed_mount_flags: tuple[str, ...]
    observed_owner_user_id: int
    observed_owner_group_id: int
    observed_root_mode: int
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
    sentinel_evidence: RunActionRuntimeVolumeSentinelEvidence

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-runtime-volume-evidence"
    IDENTITY_FIELD: ClassVar[str] = "runtime_volume_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.volume_keeper_evidence_id,
            "run-action-volume-keeper-evidence",
            "runtime volume keeper evidence",
        )
        if type(self.volume_authority) is not RunActionRuntimeVolumeAuthority:
            raise RunActionSupervisorContractError(
                "runtime volume evidence lacks issued authority"
            )
        authority = self.volume_authority
        keeper_cgroup_path = (
            PurePosixPath(self.keeper_process_cgroup_path)
            if type(self.keeper_process_cgroup_path) is str
            else None
        )
        if (
            _SHA256_DIGEST_PATTERN.fullmatch(self.docker_volume_occurrence_digest)
            is None
            or _DOCKER_CONTAINER_ID_PATTERN.fullmatch(self.keeper_container_id) is None
            or not _bounded_physical_integer(self.keeper_process_id, 1)
            or not _bounded_physical_integer(
                self.keeper_process_start_time_ticks,
                1,
            )
            or keeper_cgroup_path is None
            or not self.keeper_process_cgroup_path.isascii()
            or "\x00" in self.keeper_process_cgroup_path
            or not keeper_cgroup_path.is_absolute()
            or keeper_cgroup_path.as_posix() != self.keeper_process_cgroup_path
            or ".." in keeper_cgroup_path.parts
            or not self.keeper_process_cgroup_path.endswith(
                f"/docker-{self.keeper_container_id}.scope"
            )
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (
                    self.root_mount_id,
                    self.root_device,
                    self.root_inode,
                )
            )
            or self.observed_volume_name != authority.volume_name
            or self.observed_labels != authority.labels
            or self.observed_scope != "local"
            or self.observed_driver != authority.driver
            or self.observed_driver_options != authority.driver_options
            or self.observed_filesystem_type != "tmpfs"
            or self.observed_mount_flags != ("nodev", "nosuid", "noswap")
            or self.observed_owner_user_id != authority.owner_user_id
            or self.observed_owner_group_id != authority.owner_group_id
            or self.observed_root_mode != authority.root_mode
            or not _bounded_physical_integer(
                self.allocation_block_size_bytes,
                1,
            )
            or self.allocation_block_size_bytes & (self.allocation_block_size_bytes - 1)
            != 0
            or not _bounded_physical_integer(self.effective_block_count, 1)
            or not _bounded_physical_integer(self.effective_size_bytes, 1)
            or not 0 < self.effective_size_bytes <= authority.size_limit_bytes
            or self.effective_size_bytes
            != self.effective_block_count * self.allocation_block_size_bytes
            or not _bounded_physical_integer(self.effective_inode_limit, 1)
            or not 0 < self.effective_inode_limit <= authority.inode_limit
            or not _bounded_physical_integer(self.used_block_count, 0)
            or not 0 <= self.used_block_count < self.effective_block_count
            or not _bounded_physical_integer(self.used_size_bytes, 0)
            or not 0 <= self.used_size_bytes < self.effective_size_bytes
            or self.used_size_bytes
            != self.used_block_count * self.allocation_block_size_bytes
            or not _bounded_physical_integer(self.used_inode_count, 0)
            or not 0 <= self.used_inode_count < self.effective_inode_limit
            or not _bounded_physical_integer(self.available_block_count, 1)
            or self.used_block_count + self.available_block_count
            != self.effective_block_count
            or not _bounded_physical_integer(self.available_size_bytes, 1)
            or self.available_size_bytes
            != self.available_block_count * self.allocation_block_size_bytes
            or self.used_size_bytes + self.available_size_bytes
            != self.effective_size_bytes
            or not _bounded_physical_integer(self.available_inode_count, 1)
            or self.used_inode_count + self.available_inode_count
            != self.effective_inode_limit
            or type(self.sentinel_evidence)
            is not RunActionRuntimeVolumeSentinelEvidence
            or self.sentinel_evidence.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.sentinel_evidence.generation_nonce != authority.generation_nonce
            or self.sentinel_evidence.owner_user_id != authority.owner_user_id
            or self.sentinel_evidence.owner_group_id != authority.owner_group_id
            or self.sentinel_evidence.mount_id != self.root_mount_id
            or self.sentinel_evidence.device != self.root_device
            or self.sentinel_evidence.inode == self.root_inode
        ):
            raise RunActionSupervisorContractError(
                "runtime volume evidence differs from effective bounded tmpfs"
            )


@dataclass(frozen=True)
class RunActionPreparedDeliverySlot(StrictContract):
    """One empty directory that authorizes atomic publication of a final payload."""

    prepared_delivery_slot_id: str
    preparation_claim_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedFileKind
    directory_relative_path: str
    final_file_name: str
    directory_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    observed_entry_count: int
    payload_size_limit_bytes: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-delivery-slot"
    IDENTITY_FIELD: ClassVar[str] = "prepared_delivery_slot_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "prepared delivery slot claim",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "prepared delivery slot runtime volume",
        )
        expected_paths = {
            RunActionPreparedFileKind.INPUT: ("input", "request.blob"),
            RunActionPreparedFileKind.CREDENTIAL: ("credential", "credentials"),
        }
        if (
            type(self.kind) is not RunActionPreparedFileKind
            or self.kind not in expected_paths
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or (
                self.kind in expected_paths
                and (
                    self.directory_relative_path != expected_paths[self.kind][0]
                    or self.final_file_name != expected_paths[self.kind][1]
                )
            )
            or self.directory_type != "directory"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o700
            or self.observed_entry_count != 0
            or type(self.payload_size_limit_bytes) is not int
            or self.payload_size_limit_bytes <= 0
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "run action prepared delivery slot is invalid or nonempty"
            )


@dataclass(frozen=True)
class RunActionPreparedRuntimeDirectory(StrictContract):
    """Exact prepared result or temporary subpath mounted into the main."""

    prepared_runtime_directory_id: str
    preparation_claim_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedRuntimeDirectoryKind
    directory_relative_path: str
    directory_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    observed_entry_count: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-runtime-directory"
    IDENTITY_FIELD: ClassVar[str] = "prepared_runtime_directory_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "prepared runtime directory claim",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "prepared runtime directory volume",
        )
        expected_entries = {
            RunActionPreparedRuntimeDirectoryKind.RESULT: 1,
            RunActionPreparedRuntimeDirectoryKind.TEMPORARY: 0,
            RunActionPreparedRuntimeDirectoryKind.CONTROL: 0,
        }
        if (
            type(self.kind) is not RunActionPreparedRuntimeDirectoryKind
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or (
                self.kind in expected_entries
                and (
                    self.directory_relative_path
                    != _RUNTIME_VOLUME_SUBPATHS[self.kind.value]
                    or self.observed_entry_count != expected_entries[self.kind]
                )
            )
            or self.directory_type != "directory"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o700
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "run action prepared runtime directory is invalid"
            )


@dataclass(frozen=True)
class RunActionPreparedFile(StrictContract):
    """The empty result file prepared inside the exact runtime generation."""

    prepared_file_id: str
    prepared_parent_directory_id: str
    preparation_claim_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedFileKind
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    payload_size_limit_bytes: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-file"
    IDENTITY_FIELD: ClassVar[str] = "prepared_file_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_parent_directory_id,
            RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE,
            "prepared result parent directory",
        )
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "prepared file claim",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "prepared file runtime volume",
        )
        if (
            self.kind is not RunActionPreparedFileKind.RESULT
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != "result/result.blob"
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o600
            or self.link_count != 1
            or self.size_bytes != 0
            or type(self.payload_size_limit_bytes) is not int
            or self.payload_size_limit_bytes <= 0
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "run action prepared logical file is invalid or nonempty"
            )


@dataclass(frozen=True)
class RunActionPreparedWorkspaceProof(StrictContract):
    """Exact frontier workspace copied into one runtime-volume generation."""

    prepared_workspace_proof_id: str
    preparation_claim_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    volume_subpath: str
    workspace_binding: RunActionWorkspaceBinding
    observed_source_tree_digest: str
    observed_git_closure_digest: str
    observed_source_entry_count: int
    observed_source_size_bytes: int
    owner_user_id: int
    owner_group_id: int
    root_mode: int
    unexpected_entry_count: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-workspace-proof"
    IDENTITY_FIELD: ClassVar[str] = "prepared_workspace_proof_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "prepared workspace claim",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "prepared workspace runtime volume",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.volume_subpath != _RUNTIME_VOLUME_SUBPATHS["workspace"]
            or type(self.workspace_binding) is not RunActionWorkspaceBinding
            or self.observed_source_tree_digest
            != self.workspace_binding.source_tree_digest
            or self.observed_git_closure_digest
            != self.workspace_binding.git_closure_digest
            or self.observed_source_entry_count
            != self.workspace_binding.source_entry_count
            or self.observed_source_size_bytes
            != self.workspace_binding.source_size_bytes
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.root_mode != 0o700
            or self.unexpected_entry_count != 0
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "prepared workspace proof is incomplete"
            )


@dataclass(frozen=True)
class RunActionRuntimeVolumeLayoutProof(StrictContract):
    """Empty-before-use and exact prepared layout proof for one generation."""

    runtime_volume_layout_proof_id: str
    runtime_volume_authority_id: str
    runtime_volume_evidence_id: str
    generation_nonce: str
    empty_size_bytes: int
    empty_entry_count: int
    directory_relative_paths: tuple[str, ...]
    prepared_delivery_slot_ids: tuple[str, ...]
    prepared_runtime_directory_ids: tuple[str, ...]
    prepared_result_file_id: str
    prepared_workspace_proof_id: str | None
    logical_content_size_bytes: int
    logical_entry_count: int
    observed_used_size_bytes: int
    observed_used_inode_count: int
    unexpected_entry_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-runtime-volume-layout-proof"
    IDENTITY_FIELD: ClassVar[str] = "runtime_volume_layout_proof_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "runtime volume layout authority",
        )
        _require_namespaced_content_id(
            self.runtime_volume_evidence_id,
            RunActionRuntimeVolumeEvidence.CONTENT_NAMESPACE,
            "runtime volume layout evidence",
        )
        if self.prepared_workspace_proof_id is not None:
            _require_namespaced_content_id(
                self.prepared_workspace_proof_id,
                RunActionPreparedWorkspaceProof.CONTENT_NAMESPACE,
                "runtime volume layout workspace",
            )
        _require_namespaced_content_id(
            self.prepared_result_file_id,
            RunActionPreparedFile.CONTENT_NAMESPACE,
            "runtime volume layout prepared result file",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.empty_size_bytes != 0
            or self.empty_entry_count != 0
            or not self.directory_relative_paths
            or self.directory_relative_paths
            != tuple(sorted(set(self.directory_relative_paths)))
            or not self.prepared_delivery_slot_ids
            or self.prepared_delivery_slot_ids
            != tuple(sorted(set(self.prepared_delivery_slot_ids)))
            or any(
                require_content_id(
                    slot_id,
                    "runtime volume layout prepared delivery slot",
                )
                != slot_id
                or slot_id.split(":sha256:", 1)[0]
                != RunActionPreparedDeliverySlot.CONTENT_NAMESPACE
                for slot_id in self.prepared_delivery_slot_ids
            )
            or len(self.prepared_runtime_directory_ids) != 3
            or self.prepared_runtime_directory_ids
            != tuple(sorted(set(self.prepared_runtime_directory_ids)))
            or any(
                require_content_id(
                    directory_id,
                    "runtime volume layout prepared runtime directory",
                )
                != directory_id
                or directory_id.split(":sha256:", 1)[0]
                != RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE
                for directory_id in self.prepared_runtime_directory_ids
            )
            or type(self.logical_content_size_bytes) is not int
            or self.logical_content_size_bytes < len(self.generation_nonce)
            or type(self.logical_entry_count) is not int
            or self.logical_entry_count <= 0
            or not _bounded_physical_integer(self.observed_used_size_bytes, 1)
            or not _bounded_physical_integer(
                self.observed_used_inode_count,
                1,
            )
            or self.observed_used_inode_count < self.logical_entry_count
            or self.unexpected_entry_count != 0
        ):
            raise RunActionSupervisorContractError(
                "runtime volume layout proof is incomplete or noncanonical"
            )


@dataclass(frozen=True)
class RunActionContainerLabel(StrictContract):
    """One canonical, non-secret Docker label."""

    key: str
    value: str

    def _validate(self) -> None:
        if (
            _LABEL_KEY_PATTERN.fullmatch(self.key) is None
            or not isinstance(self.value, str)
            or not self.value
            or "\x00" in self.value
        ):
            raise RunActionSupervisorContractError(
                "run action container label is invalid"
            )


@dataclass(frozen=True)
class RunActionPreparedMount(StrictContract):
    """One exact named-volume subpath observed on the inert main container."""

    kind: RunActionPreparedMountKind
    volume_name: str
    volume_subpath: str
    container_destination: str
    mount_type: str
    source_access: RunActionPreparedMountAccess
    container_access: RunActionPreparedMountAccess
    host_config_volume_subpath: str

    def _validate(self) -> None:
        if type(self.kind) is not RunActionPreparedMountKind:
            raise RunActionSupervisorContractError(
                "run action prepared mount kind is invalid"
            )
        expected_subpath = _RUNTIME_VOLUME_SUBPATHS[self.kind.value]
        if (
            type(self.source_access) is not RunActionPreparedMountAccess
            or type(self.container_access) is not RunActionPreparedMountAccess
            or not isinstance(self.volume_name, str)
            or not self.volume_name.startswith(_RUNTIME_VOLUME_NAME_PREFIX)
            or self.mount_type != "volume"
            or self.source_access is not RunActionPreparedMountAccess.READ_WRITE
            or self.volume_subpath != expected_subpath
            or self.host_config_volume_subpath != self.volume_subpath
        ):
            raise RunActionSupervisorContractError(
                "run action prepared mount is invalid"
            )
        _require_absolute_container_path(self.container_destination)
        if (
            self.kind
            in (
                RunActionPreparedMountKind.INPUT,
                RunActionPreparedMountKind.CREDENTIAL,
                RunActionPreparedMountKind.CONTROL,
            )
            and self.container_access is not RunActionPreparedMountAccess.READ_ONLY
        ):
            raise RunActionSupervisorContractError(
                "run action delivery mount must be read-only"
            )
        if (
            self.kind
            in (
                RunActionPreparedMountKind.RESULT,
                RunActionPreparedMountKind.TEMPORARY,
            )
            and self.container_access is not RunActionPreparedMountAccess.READ_WRITE
        ):
            raise RunActionSupervisorContractError(
                "run action output mount must be read-write"
            )


@dataclass(frozen=True)
class DockerRunActionCreateInspectProjection(StrictContract):
    """Closed normalized projection shared by Docker create and inspect."""

    create_inspect_projection_id: str
    projection_protocol_version: str
    raw_field_schema_id: str
    execution_policy: DockerRunActionExecutionPolicy
    supervisor_helper_evidence: RunActionSupervisorHelperEvidence
    docker_init_source_evidence: RunActionDockerInitSourceEvidence
    barrier_protocol_version: str
    barrier_poll_interval_seconds: int
    barrier_generation_nonce: str
    command_executable: str
    command_arguments: tuple[str, ...]
    mounts: tuple[RunActionPreparedMount, ...]
    exact_mount_count: int
    unclassified_raw_field_count: int
    nonauthoritative_raw_field_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-create-inspect-projection"
    IDENTITY_FIELD: ClassVar[str] = "create_inspect_projection_id"

    def _validate(self) -> None:
        if (
            type(self.execution_policy) is not DockerRunActionExecutionPolicy
            or type(self.supervisor_helper_evidence)
            is not RunActionSupervisorHelperEvidence
            or self.supervisor_helper_evidence.helper_authority_id
            != self.execution_policy.supervisor_helper_executable_authority_id
            or self.supervisor_helper_evidence.source_path
            != self.execution_policy.supervisor_helper_source_path
            or self.supervisor_helper_evidence.executable_digest
            != self.execution_policy.supervisor_helper_executable_digest
            or type(self.docker_init_source_evidence)
            is not RunActionDockerInitSourceEvidence
            or self.docker_init_source_evidence.init_authority_id
            != self.execution_policy.docker_init_executable_authority_id
            or self.docker_init_source_evidence.source_path
            != self.execution_policy.docker_init_source_path
            or self.docker_init_source_evidence.executable_digest
            != self.execution_policy.docker_init_executable_digest
            or self.barrier_protocol_version != RUN_ACTION_BARRIER_PROTOCOL_VERSION
            or type(self.barrier_poll_interval_seconds) is not int
            or self.barrier_poll_interval_seconds <= 0
            or _GENERATION_NONCE_PATTERN.fullmatch(self.barrier_generation_nonce)
            is None
            or not _barrier_command_matches_policy(
                self.command_executable,
                self.command_arguments,
                self.barrier_poll_interval_seconds,
                self.barrier_generation_nonce,
                self.execution_policy,
            )
            or self.projection_protocol_version
            != self.execution_policy.projection_protocol_version
            or self.raw_field_schema_id != self.execution_policy.raw_field_schema_id
            or any(type(mount) is not RunActionPreparedMount for mount in self.mounts)
            or tuple(mount.container_destination for mount in self.mounts)
            != tuple(sorted({mount.container_destination for mount in self.mounts}))
            or len({mount.kind for mount in self.mounts}) != len(self.mounts)
            or len({mount.volume_name for mount in self.mounts}) != 1
            or self.exact_mount_count != len(self.mounts) + 1
            or any(
                left == right or left in right.parents or right in left.parents
                for position, left in enumerate(
                    PurePosixPath(mount.volume_subpath) for mount in self.mounts
                )
                for right in (
                    PurePosixPath(mount.volume_subpath)
                    for mount in self.mounts[position + 1 :]
                )
            )
            or self.unclassified_raw_field_count != 0
            or not _bounded_physical_integer(
                self.nonauthoritative_raw_field_count,
                0,
            )
        ):
            raise RunActionSupervisorContractError(
                "Docker create/inspect projection is incomplete or noncanonical"
            )


@dataclass(frozen=True)
class RunActionSupervisorHelperEvidence(StrictContract):
    """Physical proof of the root-owned static BusyBox supervisor bind."""

    supervisor_helper_evidence_id: str
    helper_authority_id: str
    source_path: str
    destination: str
    mount_type: str
    mount_access: RunActionPreparedMountAccess
    recursive_bind: bool
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    file_format: str
    dynamic_dependency_count: int
    elf_interpreter_present: bool
    executable_digest: str
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-supervisor-helper-evidence"
    IDENTITY_FIELD: ClassVar[str] = "supervisor_helper_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.helper_authority_id,
            "run-action-helper-executable-authority",
            "runtime supervisor helper authority",
        )
        _require_absolute_host_path(
            self.source_path,
            "runtime supervisor helper source",
        )
        if (
            self.helper_authority_id
            != run_action_supervisor_helper_authority_id(
                self.source_path,
                self.executable_digest,
            )
            or self.destination != RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
            or self.mount_type != "bind"
            or self.mount_access is not RunActionPreparedMountAccess.READ_ONLY
            or self.recursive_bind is not False
            or self.file_type != "regular"
            or self.owner_user_id != 0
            or self.owner_group_id != 0
            or self.mode != 0o755
            or self.link_count != 1
            or self.file_format != "elf"
            or self.dynamic_dependency_count != 0
            or self.elf_interpreter_present is not False
            or _SHA256_DIGEST_PATTERN.fullmatch(self.executable_digest) is None
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "runtime supervisor helper evidence is unsafe or substituted"
            )


@dataclass(frozen=True)
class RunActionDockerInitSourceEvidence(StrictContract):
    """Physical proof of the configured host Docker-init source executable."""

    docker_init_source_evidence_id: str
    init_authority_id: str
    source_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    file_format: str
    dynamic_dependency_count: int
    elf_interpreter_present: bool
    executable_digest: str
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-docker-init-source-evidence"
    IDENTITY_FIELD: ClassVar[str] = "docker_init_source_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.init_authority_id,
            "run-action-docker-init-executable-authority",
            "Docker init executable authority",
        )
        _require_absolute_host_path(
            self.source_path,
            "Docker init executable source",
        )
        if (
            self.init_authority_id
            != run_action_docker_init_authority_id(
                self.source_path,
                self.executable_digest,
            )
            or self.file_type != "regular"
            or self.owner_user_id != 0
            or self.owner_group_id != 0
            or self.mode != 0o755
            or self.link_count != 1
            or self.file_format != "elf"
            or self.dynamic_dependency_count != 0
            or self.elf_interpreter_present is not False
            or _SHA256_DIGEST_PATTERN.fullmatch(self.executable_digest) is None
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "Docker init source evidence is unsafe or substituted"
            )


@dataclass(frozen=True)
class RunActionMountedKeeperHelperEvidence(StrictContract):
    """Spawn-bound proof that the keeper executes the issued helper inode."""

    mounted_keeper_helper_evidence_id: str
    source_helper_evidence: RunActionSupervisorHelperEvidence
    container_id: str
    process_id: int
    process_start_time_ticks: int
    process_cgroup_path: str
    destination: str
    mount_id: int
    device: int
    inode: int
    executable_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-mounted-keeper-helper-evidence"
    IDENTITY_FIELD: ClassVar[str] = "mounted_keeper_helper_evidence_id"

    def _validate(self) -> None:
        if (
            type(self.source_helper_evidence) is not RunActionSupervisorHelperEvidence
            or _DOCKER_CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
            or not _bounded_physical_integer(self.process_id, 1)
            or not _bounded_physical_integer(
                self.process_start_time_ticks,
                1,
            )
            or not isinstance(self.process_cgroup_path, str)
            or not self.process_cgroup_path.isascii()
            or "\x00" in self.process_cgroup_path
            or not self.process_cgroup_path.startswith("/")
            or PurePosixPath(self.process_cgroup_path).as_posix()
            != self.process_cgroup_path
            or ".." in PurePosixPath(self.process_cgroup_path).parts
            or not self.process_cgroup_path.endswith(
                f"/docker-{self.container_id}.scope"
            )
            or self.destination != RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
            or not _bounded_physical_integer(self.mount_id, 1)
            or self.mount_id == self.source_helper_evidence.mount_id
            or not _bounded_physical_integer(self.device, 1)
            or not _bounded_physical_integer(self.inode, 1)
            or self.device != self.source_helper_evidence.device
            or self.inode != self.source_helper_evidence.inode
            or self.executable_digest != self.source_helper_evidence.executable_digest
        ):
            raise RunActionSupervisorContractError(
                "mounted keeper helper differs from its source inode or process"
            )


@dataclass(frozen=True)
class DockerRunActionKeeperCreateInspectProjection(StrictContract):
    """Closed normalized projection for the sole runtime-volume keeper."""

    keeper_create_inspect_projection_id: str
    projection_protocol_version: str
    raw_field_schema_id: str
    preparation_claim_id: str
    execution_policy: DockerRunActionExecutionPolicy
    volume_authority: RunActionRuntimeVolumeAuthority
    command_executable: str
    command_arguments: tuple[str, ...]
    helper_evidence: RunActionSupervisorHelperEvidence
    docker_init_source_evidence: RunActionDockerInitSourceEvidence
    volume_mount_type: str
    volume_mount_destination: str
    volume_mount_access: RunActionPreparedMountAccess
    network_mode: str
    exact_mount_count: int
    healthcheck_present: bool
    docker_socket_mounted: bool
    unclassified_raw_field_count: int
    nonauthoritative_raw_field_count: int

    CONTENT_NAMESPACE: ClassVar[str] = (
        "docker-run-action-keeper-create-inspect-projection"
    )
    IDENTITY_FIELD: ClassVar[str] = "keeper_create_inspect_projection_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "runtime volume keeper projection claim",
        )
        if (
            type(self.execution_policy) is not DockerRunActionExecutionPolicy
            or type(self.volume_authority) is not RunActionRuntimeVolumeAuthority
            or type(self.helper_evidence) is not RunActionSupervisorHelperEvidence
            or type(self.docker_init_source_evidence)
            is not RunActionDockerInitSourceEvidence
            or self.projection_protocol_version
            != self.execution_policy.projection_protocol_version
            or self.raw_field_schema_id != self.execution_policy.raw_field_schema_id
            or self.volume_authority.preparation_claim_id != self.preparation_claim_id
            or self.command_executable != RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
            or self.command_arguments != ("tail", "-f", "/dev/null")
            or self.helper_evidence.helper_authority_id
            != self.execution_policy.supervisor_helper_executable_authority_id
            or self.helper_evidence.source_path
            != self.execution_policy.supervisor_helper_source_path
            or self.helper_evidence.executable_digest
            != self.execution_policy.supervisor_helper_executable_digest
            or self.docker_init_source_evidence.init_authority_id
            != self.execution_policy.docker_init_executable_authority_id
            or self.docker_init_source_evidence.source_path
            != self.execution_policy.docker_init_source_path
            or self.docker_init_source_evidence.executable_digest
            != self.execution_policy.docker_init_executable_digest
            or self.volume_mount_type != "volume"
            or self.volume_mount_destination
            != RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION
            or self.volume_mount_access is not RunActionPreparedMountAccess.READ_WRITE
            or self.network_mode != "none"
            or self.exact_mount_count != 2
            or self.healthcheck_present is not False
            or self.docker_socket_mounted is not False
            or self.unclassified_raw_field_count != 0
            or not _bounded_physical_integer(
                self.nonauthoritative_raw_field_count,
                0,
            )
        ):
            raise RunActionSupervisorContractError(
                "Docker keeper create/inspect projection is incomplete or unsafe"
            )


@dataclass(frozen=True)
class RunActionVolumeKeeperEvidence(StrictContract):
    """Issued-equals-observed proof of one running network-free keeper."""

    volume_keeper_evidence_id: str
    preparation_claim_id: str
    container_id: str
    container_name: str
    labels: tuple[RunActionContainerLabel, ...]
    issued_create_projection: DockerRunActionKeeperCreateInspectProjection
    observed_inspect_projection: DockerRunActionKeeperCreateInspectProjection
    mounted_helper_evidence: RunActionMountedKeeperHelperEvidence
    container_status: str
    process_id: int
    process_start_time_ticks: int
    restart_count: int
    restart_policy_name: str
    auto_remove: bool

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-volume-keeper-evidence"
    IDENTITY_FIELD: ClassVar[str] = "volume_keeper_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "runtime volume keeper claim",
        )
        if (
            _DOCKER_CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
            or self.container_name
            != _KEEPER_CONTAINER_NAME_PREFIX
            + self.preparation_claim_id.rsplit(":sha256:", 1)[1]
            or any(type(label) is not RunActionContainerLabel for label in self.labels)
            or tuple(label.key for label in self.labels)
            != tuple(sorted({label.key for label in self.labels}))
            or type(self.issued_create_projection)
            is not DockerRunActionKeeperCreateInspectProjection
            or type(self.observed_inspect_projection)
            is not DockerRunActionKeeperCreateInspectProjection
            or self.observed_inspect_projection != self.issued_create_projection
            or self.issued_create_projection.preparation_claim_id
            != self.preparation_claim_id
            or type(self.mounted_helper_evidence)
            is not RunActionMountedKeeperHelperEvidence
            or self.mounted_helper_evidence.source_helper_evidence
            != self.issued_create_projection.helper_evidence
            or self.mounted_helper_evidence.container_id != self.container_id
            or self.mounted_helper_evidence.process_id != self.process_id
            or self.mounted_helper_evidence.process_start_time_ticks
            != self.process_start_time_ticks
            or self.mounted_helper_evidence.process_cgroup_path
            != run_action_keeper_process_cgroup_path(
                self.issued_create_projection.execution_policy,
                self.container_id,
            )
            or self.container_status != "running"
            or not _bounded_physical_integer(self.process_id, 1)
            or not _bounded_physical_integer(
                self.process_start_time_ticks,
                1,
            )
            or self.restart_count != 0
            or self.restart_policy_name != "no"
            or self.auto_remove is not False
        ):
            raise RunActionSupervisorContractError(
                "runtime volume keeper evidence is not exact and running"
            )


@dataclass(frozen=True)
class RunActionInertContainerEvidence(StrictContract):
    """Exact Docker inspection proving one prepared container never started."""

    inert_container_evidence_id: str
    preparation_claim_id: str
    container_id: str
    container_name: str
    labels: tuple[RunActionContainerLabel, ...]
    image_authority_id: str
    docker_runtime_settings_digest: str
    issued_create_projection: DockerRunActionCreateInspectProjection
    observed_inspect_projection: DockerRunActionCreateInspectProjection
    container_status: str
    process_id: int
    restart_count: int
    started_at: str
    finished_at: str
    restart_policy_name: str
    auto_remove: bool
    network_mode: str
    healthcheck_present: bool
    volume_plugin_mount_count: int
    docker_socket_mounted: bool

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-inert-container-evidence"
    IDENTITY_FIELD: ClassVar[str] = "inert_container_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "inert run action container claim",
        )
        _require_namespaced_content_id(
            self.image_authority_id,
            DockerImageAuthority.CONTENT_NAMESPACE,
            "inert run action container image",
        )
        if (
            _DOCKER_CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
            or self.container_name
            != _CONTAINER_NAME_PREFIX
            + self.preparation_claim_id.rsplit(":sha256:", 1)[1]
            or any(type(label) is not RunActionContainerLabel for label in self.labels)
            or tuple(label.key for label in self.labels)
            != tuple(sorted({label.key for label in self.labels}))
            or _SHA256_DIGEST_PATTERN.fullmatch(self.docker_runtime_settings_digest)
            is None
            or type(self.issued_create_projection)
            is not DockerRunActionCreateInspectProjection
            or type(self.observed_inspect_projection)
            is not DockerRunActionCreateInspectProjection
            or self.observed_inspect_projection != self.issued_create_projection
            or self.container_status != "created"
            or self.process_id != 0
            or self.restart_count != 0
            or self.started_at != _ZERO_DOCKER_TIMESTAMP
            or self.finished_at != _ZERO_DOCKER_TIMESTAMP
            or self.restart_policy_name != "no"
            or self.auto_remove is not False
            or self.network_mode != "none"
            or self.healthcheck_present is not False
            or self.volume_plugin_mount_count != 0
            or self.docker_socket_mounted is not False
        ):
            raise RunActionSupervisorContractError(
                "run action container evidence does not prove an exact inert resource"
            )


@dataclass(frozen=True)
class RunActionPreparedExecution(StrictContract):
    """One concrete, inert Docker occurrence prepared for a semantic claim."""

    prepared_execution_id: str
    preparation_claim: RunActionPreparationClaim
    runtime_volume_authority: RunActionRuntimeVolumeAuthority
    runtime_volume_evidence: RunActionRuntimeVolumeEvidence
    volume_keeper_evidence: RunActionVolumeKeeperEvidence
    input_delivery_slot: RunActionPreparedDeliverySlot
    result_directory: RunActionPreparedRuntimeDirectory
    temporary_directory: RunActionPreparedRuntimeDirectory
    control_directory: RunActionPreparedRuntimeDirectory
    result_file: RunActionPreparedFile
    credential_delivery_slot: RunActionPreparedDeliverySlot | None
    workspace_proof: RunActionPreparedWorkspaceProof | None
    layout_proof: RunActionRuntimeVolumeLayoutProof
    inert_container_evidence: RunActionInertContainerEvidence

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-execution"
    IDENTITY_FIELD: ClassVar[str] = "prepared_execution_id"

    def _validate(self) -> None:
        if (
            type(self.preparation_claim) is not RunActionPreparationClaim
            or type(self.runtime_volume_authority)
            is not RunActionRuntimeVolumeAuthority
            or type(self.runtime_volume_evidence) is not RunActionRuntimeVolumeEvidence
            or type(self.volume_keeper_evidence) is not RunActionVolumeKeeperEvidence
            or type(self.input_delivery_slot) is not RunActionPreparedDeliverySlot
            or type(self.result_directory) is not RunActionPreparedRuntimeDirectory
            or type(self.temporary_directory) is not RunActionPreparedRuntimeDirectory
            or type(self.control_directory) is not RunActionPreparedRuntimeDirectory
            or type(self.result_file) is not RunActionPreparedFile
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
            or type(self.inert_container_evidence)
            is not RunActionInertContainerEvidence
        ):
            raise RunActionSupervisorContractError(
                "prepared run action execution has invalid components"
            )
        claim = self.preparation_claim
        authority = self.runtime_volume_authority
        delivery_slots = tuple(
            delivery_slot
            for delivery_slot in (
                self.input_delivery_slot,
                self.credential_delivery_slot,
            )
            if delivery_slot is not None
        )
        expected_kinds = (
            RunActionPreparedFileKind.INPUT,
            *(
                ()
                if claim.execution_policy.credential_policy.mode
                is RunActionCredentialMode.NONE
                else (RunActionPreparedFileKind.CREDENTIAL,)
            ),
        )
        if (
            tuple(delivery_slot.kind for delivery_slot in delivery_slots)
            != expected_kinds
            or any(
                delivery_slot.preparation_claim_id != claim.preparation_claim_id
                or delivery_slot.runtime_volume_authority_id
                != authority.runtime_volume_authority_id
                or delivery_slot.generation_nonce != authority.generation_nonce
                or delivery_slot.owner_user_id != claim.execution_policy.user_id
                or delivery_slot.owner_group_id != claim.execution_policy.group_id
                for delivery_slot in delivery_slots
            )
            or self.input_delivery_slot.payload_size_limit_bytes
            != claim.reservation.request_blob.size_bytes
            or self.result_directory.preparation_claim_id != claim.preparation_claim_id
            or self.result_directory.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.result_directory.generation_nonce != authority.generation_nonce
            or self.result_directory.kind
            is not RunActionPreparedRuntimeDirectoryKind.RESULT
            or self.temporary_directory.preparation_claim_id
            != claim.preparation_claim_id
            or self.temporary_directory.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.temporary_directory.generation_nonce != authority.generation_nonce
            or self.temporary_directory.kind
            is not RunActionPreparedRuntimeDirectoryKind.TEMPORARY
            or self.control_directory.preparation_claim_id != claim.preparation_claim_id
            or self.control_directory.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.control_directory.generation_nonce != authority.generation_nonce
            or self.control_directory.kind
            is not RunActionPreparedRuntimeDirectoryKind.CONTROL
            or self.result_file.preparation_claim_id != claim.preparation_claim_id
            or self.result_file.prepared_parent_directory_id
            != self.result_directory.prepared_runtime_directory_id
            or self.result_file.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.result_file.generation_nonce != authority.generation_nonce
            or self.result_file.owner_user_id != claim.execution_policy.user_id
            or self.result_file.owner_group_id != claim.execution_policy.group_id
            or self.result_file.payload_size_limit_bytes
            != claim.execution_policy.supervisor_limits.result_size_bytes
            or (
                self.credential_delivery_slot is not None
                and self.credential_delivery_slot.payload_size_limit_bytes
                != claim.execution_policy.credential_policy.maximum_delivery_size_bytes
            )
            or any(
                delivery_slot.mount_id != self.runtime_volume_evidence.root_mount_id
                or delivery_slot.device != self.runtime_volume_evidence.root_device
                for delivery_slot in delivery_slots
            )
            or self.result_file.mount_id != self.runtime_volume_evidence.root_mount_id
            or self.result_file.device != self.runtime_volume_evidence.root_device
            or any(
                directory.mount_id != self.runtime_volume_evidence.root_mount_id
                or directory.device != self.runtime_volume_evidence.root_device
                for directory in (
                    self.result_directory,
                    self.temporary_directory,
                    self.control_directory,
                )
            )
            or len(
                {
                    *(delivery_slot.inode for delivery_slot in delivery_slots),
                    self.result_directory.inode,
                    self.temporary_directory.inode,
                    self.control_directory.inode,
                    self.result_file.inode,
                }
            )
            != len(delivery_slots) + 4
            or {
                *(delivery_slot.inode for delivery_slot in delivery_slots),
                self.result_directory.inode,
                self.temporary_directory.inode,
                self.control_directory.inode,
                self.result_file.inode,
            }
            & {
                self.runtime_volume_evidence.root_inode,
                self.runtime_volume_evidence.sentinel_evidence.inode,
            }
        ):
            raise RunActionSupervisorContractError(
                "prepared run action artifacts differ from their preparation claim"
            )
        limits = claim.execution_policy.docker_resource_limits
        policy = claim.execution_policy
        if (
            authority.preparation_claim_id != claim.preparation_claim_id
            or authority.volume_name != preparation_volume_name(claim)
            or authority.labels
            != preparation_volume_labels(claim, authority.generation_nonce)
            or authority.owner_user_id != policy.user_id
            or authority.owner_group_id != policy.group_id
            or authority.size_limit_bytes != limits.runtime_volume_size_bytes
            or authority.inode_limit != limits.runtime_volume_inode_limit
            or self.runtime_volume_evidence.volume_authority != authority
        ):
            raise RunActionSupervisorContractError(
                "prepared runtime volume differs from its execution policy"
            )
        volume_evidence = self.runtime_volume_evidence
        keeper = self.volume_keeper_evidence
        keeper_projection = keeper.issued_create_projection
        if (
            keeper.preparation_claim_id != claim.preparation_claim_id
            or keeper.container_name != preparation_keeper_container_name(claim)
            or keeper.labels != preparation_keeper_container_labels(claim)
            or keeper_projection.execution_policy != policy
            or keeper_projection.volume_authority != authority
            or volume_evidence.volume_keeper_evidence_id
            != keeper.volume_keeper_evidence_id
            or volume_evidence.keeper_container_id != keeper.container_id
            or volume_evidence.keeper_process_id != keeper.process_id
            or volume_evidence.keeper_process_start_time_ticks
            != keeper.process_start_time_ticks
            or volume_evidence.keeper_process_cgroup_path
            != keeper.mounted_helper_evidence.process_cgroup_path
            or volume_evidence.keeper_process_cgroup_path
            != run_action_keeper_process_cgroup_path(policy, keeper.container_id)
        ):
            raise RunActionSupervisorContractError(
                "runtime volume keeper differs from prepared authority"
            )
        workspace_binding = claim.reservation.frontier.workspace_before
        workspace_access = claim.reservation.intent.workspace_access
        if workspace_access is RunFrontierWorkspaceAccess.NONE:
            if self.workspace_proof is not None or workspace_binding is not None:
                raise RunActionSupervisorContractError(
                    "workspace-free preparation carries a workspace proof"
                )
        elif (
            type(self.workspace_proof) is not RunActionPreparedWorkspaceProof
            or workspace_binding is None
            or self.workspace_proof.preparation_claim_id != claim.preparation_claim_id
            or self.workspace_proof.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.workspace_proof.generation_nonce != authority.generation_nonce
            or self.workspace_proof.workspace_binding != workspace_binding
            or self.workspace_proof.owner_user_id != policy.user_id
            or self.workspace_proof.owner_group_id != policy.group_id
            or self.workspace_proof.mount_id != volume_evidence.root_mount_id
            or self.workspace_proof.device != volume_evidence.root_device
            or self.workspace_proof.inode
            in {
                volume_evidence.root_inode,
                volume_evidence.sentinel_evidence.inode,
                self.input_delivery_slot.inode,
                self.result_directory.inode,
                self.temporary_directory.inode,
                self.control_directory.inode,
                self.result_file.inode,
                *(
                    ()
                    if self.credential_delivery_slot is None
                    else (self.credential_delivery_slot.inode,)
                ),
            }
        ):
            raise RunActionSupervisorContractError(
                "prepared workspace proof differs from the durable frontier"
            )
        layout = self.layout_proof
        expected_directories = tuple(
            sorted(
                {
                    _RUNTIME_VOLUME_SUBPATHS["input"],
                    _RUNTIME_VOLUME_SUBPATHS["result"],
                    _RUNTIME_VOLUME_SUBPATHS["temporary"],
                    _RUNTIME_VOLUME_SUBPATHS["control"],
                    *(
                        ()
                        if self.credential_delivery_slot is None
                        else (_RUNTIME_VOLUME_SUBPATHS["credential"],)
                    ),
                    *(
                        ()
                        if self.workspace_proof is None
                        else (_RUNTIME_VOLUME_SUBPATHS["workspace"],)
                    ),
                }
            )
        )
        workspace_size_bytes = (
            0 if workspace_binding is None else workspace_binding.source_size_bytes
        )
        workspace_entry_count = (
            0 if workspace_binding is None else workspace_binding.source_entry_count
        )
        expected_prepared_size_bytes = len(authority.generation_nonce) + (
            workspace_size_bytes
        )
        expected_prepared_entry_count = (
            len(expected_directories) + 2 + workspace_entry_count
        )
        evidence = volume_evidence
        required_available_size_bytes = (
            sum(
                _allocated_size(
                    delivery_slot.payload_size_limit_bytes,
                    evidence.allocation_block_size_bytes,
                )
                for delivery_slot in delivery_slots
            )
            + _allocated_size(
                self.result_file.payload_size_limit_bytes,
                evidence.allocation_block_size_bytes,
            )
            + _allocated_size(
                limits.runtime_temporary_reservation_size_bytes,
                evidence.allocation_block_size_bytes,
            )
            + _allocated_size(
                claim.execution_policy.supervisor_limits.release_receipt_size_bytes,
                evidence.allocation_block_size_bytes,
            )
            + _allocated_size(
                claim.execution_policy.supervisor_limits.timeout_directive_size_bytes,
                evidence.allocation_block_size_bytes,
            )
        )
        required_available_inode_count = (
            len(delivery_slots) + limits.runtime_temporary_reservation_inode_count + 2
        )
        if (
            layout.runtime_volume_authority_id != authority.runtime_volume_authority_id
            or layout.runtime_volume_evidence_id != evidence.runtime_volume_evidence_id
            or layout.generation_nonce != authority.generation_nonce
            or layout.directory_relative_paths != expected_directories
            or layout.prepared_delivery_slot_ids
            != tuple(
                sorted(
                    delivery_slot.prepared_delivery_slot_id
                    for delivery_slot in delivery_slots
                )
            )
            or layout.prepared_runtime_directory_ids
            != tuple(
                sorted(
                    (
                        self.result_directory.prepared_runtime_directory_id,
                        self.temporary_directory.prepared_runtime_directory_id,
                        self.control_directory.prepared_runtime_directory_id,
                    )
                )
            )
            or layout.prepared_result_file_id != self.result_file.prepared_file_id
            or layout.prepared_workspace_proof_id
            != (
                None
                if self.workspace_proof is None
                else self.workspace_proof.prepared_workspace_proof_id
            )
            or layout.logical_content_size_bytes != expected_prepared_size_bytes
            or layout.logical_entry_count != expected_prepared_entry_count
            or layout.observed_used_size_bytes != evidence.used_size_bytes
            or layout.observed_used_inode_count != evidence.used_inode_count
            or required_available_size_bytes >= evidence.available_size_bytes
            or required_available_inode_count >= evidence.available_inode_count
        ):
            raise RunActionSupervisorContractError(
                "prepared runtime volume lacks positive byte or inode headroom"
            )
        container_evidence = self.inert_container_evidence
        issued_projection = container_evidence.issued_create_projection
        if (
            container_evidence.preparation_claim_id != claim.preparation_claim_id
            or container_evidence.container_name != preparation_container_name(claim)
            or container_evidence.labels != preparation_container_labels(claim)
            or container_evidence.image_authority_id
            != policy.image_authority.image_authority_id
            or container_evidence.docker_runtime_settings_digest
            != policy.docker_runtime_settings_digest
            or container_evidence.container_id == keeper.container_id
            or container_evidence.container_name == keeper.container_name
            or issued_projection.execution_policy != policy
            or issued_projection.supervisor_helper_evidence
            != keeper_projection.helper_evidence
            or issued_projection.docker_init_source_evidence
            != keeper_projection.docker_init_source_evidence
            or issued_projection.barrier_generation_nonce != authority.generation_nonce
            or issued_projection.mounts
            != _expected_prepared_mounts(claim, authority.volume_name)
        ):
            raise RunActionSupervisorContractError(
                "inert run action evidence differs from the prepared execution"
            )


@dataclass(frozen=True)
class RunActionCredentialLeaseRequest(StrictContract):
    """Deterministic non-secret request for one spawn-bound broker lease."""

    credential_lease_request_id: str
    credential_policy: RunActionCredentialPolicy
    reservation_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    credential_delivery_slot_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-lease-request"
    IDENTITY_FIELD: ClassVar[str] = "credential_lease_request_id"

    def _validate(self) -> None:
        if (
            type(self.credential_policy) is not RunActionCredentialPolicy
            or self.credential_policy.mode
            is not RunActionCredentialMode.SUPERVISOR_FILE
        ):
            raise RunActionSupervisorContractError(
                "credential lease request lacks one brokered policy"
            )
        for value, namespace, name in (
            (
                self.reservation_id,
                RunActionReservation.CONTENT_NAMESPACE,
                "credential lease reservation",
            ),
            (
                self.prepared_execution_id,
                RunActionPreparedExecution.CONTENT_NAMESPACE,
                "credential lease prepared execution",
            ),
            (
                self.spawn_commit_id,
                RunActionSpawnCommit.CONTENT_NAMESPACE,
                "credential lease spawn commit",
            ),
            (
                self.credential_delivery_slot_id,
                RunActionPreparedDeliverySlot.CONTENT_NAMESPACE,
                "credential lease delivery slot",
            ),
        ):
            _require_namespaced_content_id(value, namespace, name)


def run_action_credential_lease_request(
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
) -> RunActionCredentialLeaseRequest:
    """Derive the sole non-secret broker request for one exact committed spawn."""

    if (
        type(prepared_execution) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
    ):
        raise RunActionSupervisorContractError(
            "credential lease request requires exact prepared and spawn authority"
        )
    reservation = prepared_execution.preparation_claim.reservation
    credential_policy = (
        prepared_execution.preparation_claim.execution_policy.credential_policy
    )
    delivery_slot = prepared_execution.credential_delivery_slot
    if (
        credential_policy.mode is not RunActionCredentialMode.SUPERVISOR_FILE
        or type(delivery_slot) is not RunActionPreparedDeliverySlot
        or spawn_commit.reservation_id != reservation.reservation_id
        or spawn_commit.prepared_execution_id
        != prepared_execution.prepared_execution_id
        or spawn_commit.provider_execution_id
        != prepared_execution.inert_container_evidence.container_id
        or spawn_commit.boundary_identity != reservation.intent.boundary_identity
        or spawn_commit.security_observation_id
        != reservation.frontier.security_observation_id
    ):
        raise RunActionSupervisorContractError(
            "credential lease request differs from its committed spawn"
        )
    return RunActionCredentialLeaseRequest.mint(
        credential_policy=credential_policy,
        reservation_id=reservation.reservation_id,
        prepared_execution_id=prepared_execution.prepared_execution_id,
        spawn_commit_id=spawn_commit.spawn_commit_id,
        credential_delivery_slot_id=delivery_slot.prepared_delivery_slot_id,
    )


def run_action_credential_lease_authority_id_from_request(
    request: RunActionCredentialLeaseRequest,
) -> str:
    """Derive fixed-width authority from one already-validated lease request."""

    if type(request) is not RunActionCredentialLeaseRequest:
        raise RunActionSupervisorContractError(
            "credential lease authority requires one exact request"
        )
    return content_id(
        RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
        {"credential_lease_request_id": request.credential_lease_request_id},
    )


def run_action_credential_lease_authority_id(
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
) -> str:
    """Derive the fixed-width lease authority; the broker never chooses it."""

    return run_action_credential_lease_authority_id_from_request(
        run_action_credential_lease_request(prepared_execution, spawn_commit)
    )


@dataclass(frozen=True)
class RunActionActivatedFileObservation(StrictContract):
    """Fresh post-delivery shape and non-secret identity of one logical file."""

    activated_file_observation_id: str
    spawn_commit_id: str
    prepared_parent_authority_id: str
    prepared_file_id: str | None
    parent_mount_id: int
    parent_device: int
    parent_inode: int
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedFileKind
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    mount_id: int
    device: int
    inode: int
    content_digest: str | None
    content_authority_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activated-file-observation"
    IDENTITY_FIELD: ClassVar[str] = "activated_file_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "activated file spawn commit",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "activated runtime volume",
        )
        if type(self.kind) is not RunActionPreparedFileKind:
            raise RunActionSupervisorContractError(
                "activated run action file kind is invalid"
            )
        if self.content_authority_id is not None:
            require_identifier(
                self.content_authority_id,
                "activated file content authority",
            )
            if self.kind is RunActionPreparedFileKind.CREDENTIAL:
                _require_namespaced_content_id(
                    self.content_authority_id,
                    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
                    "activated credential lease authority",
                )
        expected_paths = {
            RunActionPreparedFileKind.INPUT: "input/request.blob",
            RunActionPreparedFileKind.RESULT: "result/result.blob",
            RunActionPreparedFileKind.CREDENTIAL: "credential/credentials",
        }
        if self.kind is RunActionPreparedFileKind.RESULT:
            _require_namespaced_content_id(
                self.prepared_parent_authority_id,
                RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE,
                "activated result parent directory",
            )
            _require_namespaced_content_id(
                self.prepared_file_id,
                RunActionPreparedFile.CONTENT_NAMESPACE,
                "activated prepared result file",
            )
        else:
            if self.prepared_file_id is not None:
                raise RunActionSupervisorContractError(
                    "activated delivered file carries a prepared file"
                )
            _require_namespaced_content_id(
                self.prepared_parent_authority_id,
                RunActionPreparedDeliverySlot.CONTENT_NAMESPACE,
                "activated prepared delivery slot",
            )
        expected_mode = (
            0o600 if self.kind is RunActionPreparedFileKind.RESULT else 0o400
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != expected_paths[self.kind]
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != expected_mode
            or self.link_count != 1
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (
                    self.parent_mount_id,
                    self.parent_device,
                    self.parent_inode,
                    self.mount_id,
                    self.device,
                    self.inode,
                )
            )
            or self.parent_mount_id != self.mount_id
            or self.parent_device != self.device
            or self.parent_inode == self.inode
            or (
                self.content_digest is not None
                and _SHA256_DIGEST_PATTERN.fullmatch(self.content_digest) is None
            )
            or (
                self.kind is RunActionPreparedFileKind.INPUT
                and (
                    self.size_bytes <= 0
                    or self.content_digest is None
                    or self.content_authority_id is None
                )
            )
            or (
                self.kind is RunActionPreparedFileKind.RESULT
                and (
                    self.size_bytes != 0
                    or self.content_digest is not None
                    or self.content_authority_id is not None
                )
            )
            or (
                self.kind is RunActionPreparedFileKind.CREDENTIAL
                and (
                    self.size_bytes <= 0
                    or self.content_digest is not None
                    or self.content_authority_id is None
                )
            )
        ):
            raise RunActionSupervisorContractError(
                "activated run action file observation is invalid"
            )


@dataclass(frozen=True)
class RunActionActivatedWorkspaceObservation(StrictContract):
    """Fresh post-commit observation of the still-exact copied workspace."""

    activated_workspace_observation_id: str
    spawn_commit_id: str
    prepared_workspace_proof_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    source_tree_digest: str
    git_closure_digest: str
    source_entry_count: int
    source_size_bytes: int
    owner_user_id: int
    owner_group_id: int
    root_mode: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activated-workspace-observation"
    IDENTITY_FIELD: ClassVar[str] = "activated_workspace_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "activated workspace spawn commit",
        )
        _require_namespaced_content_id(
            self.prepared_workspace_proof_id,
            RunActionPreparedWorkspaceProof.CONTENT_NAMESPACE,
            "activated prepared workspace",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "activated workspace runtime volume",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.source_tree_digest) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.git_closure_digest) is None
            or type(self.source_entry_count) is not int
            or self.source_entry_count < 0
            or type(self.source_size_bytes) is not int
            or self.source_size_bytes < 0
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.root_mode != 0o700
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "activated workspace observation is invalid"
            )


@dataclass(frozen=True)
class RunActionActivatedRuntimeDirectoryObservation(StrictContract):
    """Fresh pre-start proof that one exact runtime subpath remains empty."""

    activated_runtime_directory_observation_id: str
    spawn_commit_id: str
    prepared_runtime_directory_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedRuntimeDirectoryKind
    directory_relative_path: str
    directory_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    observed_entry_count: int
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = (
        "run-action-activated-runtime-directory-observation"
    )
    IDENTITY_FIELD: ClassVar[str] = "activated_runtime_directory_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "activated runtime directory spawn commit",
        )
        _require_namespaced_content_id(
            self.prepared_runtime_directory_id,
            RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE,
            "activated prepared runtime directory",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "activated runtime directory volume",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.kind
            not in {
                RunActionPreparedRuntimeDirectoryKind.CONTROL,
                RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
            }
            or self.directory_relative_path != _RUNTIME_VOLUME_SUBPATHS[self.kind.value]
            or self.directory_type != "directory"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o700
            or self.observed_entry_count != 0
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "activated runtime directory observation is invalid or nonempty"
            )


@dataclass(frozen=True)
class RunActionActivatedSentinelObservation(StrictContract):
    """Fresh no-follow observation of the prepared generation sentinel."""

    activated_sentinel_observation_id: str
    spawn_commit_id: str
    prepared_sentinel_evidence_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activated-sentinel-observation"
    IDENTITY_FIELD: ClassVar[str] = "activated_sentinel_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "activated sentinel spawn commit",
        )
        _require_namespaced_content_id(
            self.prepared_sentinel_evidence_id,
            RunActionRuntimeVolumeSentinelEvidence.CONTENT_NAMESPACE,
            "activated prepared sentinel",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "activated sentinel runtime volume",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != _RUNTIME_VOLUME_SENTINEL_PATH
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o400
            or self.link_count != 1
            or self.size_bytes != len(self.generation_nonce)
            or self.content_digest
            != tree_or_blob_digest(self.generation_nonce.encode("ascii"))
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "activated sentinel observation is not the stable physical file"
            )


@dataclass(frozen=True)
class RunActionActivationRevalidationReceipt(StrictContract):
    """Fresh pre-start observation; durable evidence, never start authority."""

    activation_revalidation_receipt_id: str
    prepared_execution: RunActionPreparedExecution
    spawn_commit: RunActionSpawnCommit
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence
    reobserved_keeper_evidence: RunActionVolumeKeeperEvidence
    reobserved_container_evidence: RunActionInertContainerEvidence
    activated_workspace_observation: RunActionActivatedWorkspaceObservation | None
    activated_runtime_directory_observations: tuple[
        RunActionActivatedRuntimeDirectoryObservation, ...
    ]
    activated_sentinel_observation: RunActionActivatedSentinelObservation
    input_file_observation: RunActionActivatedFileObservation
    result_file_observation: RunActionActivatedFileObservation
    credential_file_observation: RunActionActivatedFileObservation | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activation-revalidation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "activation_revalidation_receipt_id"

    def _validate(self) -> None:
        if type(self.prepared_execution) is not RunActionPreparedExecution:
            raise RunActionSupervisorContractError(
                "activation revalidation requires one prepared execution"
            )
        prepared = self.prepared_execution
        if (
            type(self.reobserved_container_evidence)
            is not RunActionInertContainerEvidence
            or self.reobserved_keeper_evidence != prepared.volume_keeper_evidence
            or self.reobserved_container_evidence != prepared.inert_container_evidence
            or not run_action_activated_volume_evidence_matches(
                prepared=prepared,
                spawn_commit=self.spawn_commit,
                reobserved_volume_evidence=self.reobserved_volume_evidence,
                activated_workspace_observation=(self.activated_workspace_observation),
                activated_runtime_directory_observations=(
                    self.activated_runtime_directory_observations
                ),
                activated_sentinel_observation=(self.activated_sentinel_observation),
                input_file_observation=self.input_file_observation,
                result_file_observation=self.result_file_observation,
                credential_file_observation=self.credential_file_observation,
            )
        ):
            raise RunActionSupervisorContractError(
                "activation revalidation differs from prepared authority"
            )


@dataclass(frozen=True)
class RunActionTerminalObservation(StrictContract):
    """Stable terminal state of the exact durably spawned main container."""

    terminal_observation_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    provider_execution_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    activation_revalidation_receipt_id: str
    workload_release_adoption_id: str
    observed_inspect_projection: DockerRunActionCreateInspectProjection
    complete_inspection_digest: str
    container_status: str
    process_id: int
    restart_count: int
    paused: bool
    restarting: bool
    dead: bool
    started_at: str
    finished_at: str
    exit_code: int
    oom_killed: bool
    state_error: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-terminal-observation"
    IDENTITY_FIELD: ClassVar[str] = "terminal_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_execution_id,
            RunActionPreparedExecution.CONTENT_NAMESPACE,
            "terminal prepared execution",
        )
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "terminal spawn commit",
        )
        require_identifier(
            self.provider_execution_id,
            "terminal provider execution",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "terminal runtime volume",
        )
        _require_namespaced_content_id(
            self.activation_revalidation_receipt_id,
            RunActionActivationRevalidationReceipt.CONTENT_NAMESPACE,
            "terminal activation revalidation",
        )
        _require_namespaced_content_id(
            self.workload_release_adoption_id,
            "run-action-workload-release-adoption",
            "terminal workload release adoption",
        )
        if (
            type(self.observed_inspect_projection)
            is not DockerRunActionCreateInspectProjection
        ):
            raise RunActionSupervisorContractError(
                "terminal observation lacks its closed Docker projection"
            )
        started_at = _docker_timestamp_order_key(self.started_at)
        finished_at = _docker_timestamp_order_key(self.finished_at)
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.complete_inspection_digest) is None
            or self.container_status != "exited"
            or self.process_id != 0
            or self.restart_count != 0
            or self.paused is not False
            or self.restarting is not False
            or self.dead is not False
            or self.started_at == _ZERO_DOCKER_TIMESTAMP
            or self.finished_at == _ZERO_DOCKER_TIMESTAMP
            or started_at is None
            or finished_at is None
            or finished_at < started_at
            or type(self.exit_code) is not int
            or not 0 <= self.exit_code <= 255
            or type(self.oom_killed) is not bool
            or self.state_error != ""
        ):
            raise RunActionSupervisorContractError(
                "run action terminal observation is invalid"
            )


@dataclass(frozen=True)
class RunActionResultCaptureReceipt(StrictContract):
    """Descriptor-bound capture of one exact result file after terminal state."""

    result_capture_receipt_id: str
    terminal_observation_id: str
    prepared_parent_authority_id: str
    prepared_file_id: str
    parent_mount_id: int
    parent_device: int
    parent_inode: int
    runtime_volume_authority_id: str
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence
    prepared_sentinel_evidence_id: str
    generation_nonce: str
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str
    mount_id: int
    device: int
    inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-result-capture-receipt"
    IDENTITY_FIELD: ClassVar[str] = "result_capture_receipt_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.terminal_observation_id,
            RunActionTerminalObservation.CONTENT_NAMESPACE,
            "result capture terminal observation",
        )
        _require_namespaced_content_id(
            self.prepared_parent_authority_id,
            RunActionPreparedRuntimeDirectory.CONTENT_NAMESPACE,
            "result capture prepared parent directory",
        )
        _require_namespaced_content_id(
            self.prepared_file_id,
            RunActionPreparedFile.CONTENT_NAMESPACE,
            "result capture prepared file",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "result capture runtime volume",
        )
        if type(self.reobserved_volume_evidence) is not RunActionRuntimeVolumeEvidence:
            raise RunActionSupervisorContractError(
                "result capture lacks its reobserved physical volume"
            )
        _require_namespaced_content_id(
            self.prepared_sentinel_evidence_id,
            RunActionRuntimeVolumeSentinelEvidence.CONTENT_NAMESPACE,
            "result capture generation sentinel",
        )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != "result/result.blob"
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o600
            or self.link_count != 1
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
            or _SHA256_DIGEST_PATTERN.fullmatch(self.content_digest) is None
            or (
                self.size_bytes == 0 and self.content_digest != tree_or_blob_digest(b"")
            )
            or self.reobserved_volume_evidence.volume_authority.runtime_volume_authority_id
            != self.runtime_volume_authority_id
            or self.reobserved_volume_evidence.volume_authority.generation_nonce
            != self.generation_nonce
            or self.reobserved_volume_evidence.sentinel_evidence.runtime_volume_sentinel_evidence_id
            != self.prepared_sentinel_evidence_id
            or self.parent_mount_id != self.mount_id
            or self.parent_device != self.device
            or self.parent_inode
            in {
                self.inode,
                self.reobserved_volume_evidence.root_inode,
                self.reobserved_volume_evidence.sentinel_evidence.inode,
            }
            or self.mount_id != self.reobserved_volume_evidence.root_mount_id
            or self.device != self.reobserved_volume_evidence.root_device
            or self.inode
            in {
                self.reobserved_volume_evidence.root_inode,
                self.reobserved_volume_evidence.sentinel_evidence.inode,
            }
            or any(
                not _bounded_physical_integer(value, 1)
                for value in (
                    self.parent_mount_id,
                    self.parent_device,
                    self.parent_inode,
                    self.mount_id,
                    self.device,
                    self.inode,
                )
            )
        ):
            raise RunActionSupervisorContractError(
                "run action result capture receipt is invalid"
            )


def _activated_delivery_matches_slot(
    observed: RunActionActivatedFileObservation,
    prepared: RunActionPreparedDeliverySlot | None,
    spawn_commit_id: str,
) -> bool:
    if (
        type(observed) is not RunActionActivatedFileObservation
        or type(prepared) is not RunActionPreparedDeliverySlot
    ):
        return False
    return (
        observed.spawn_commit_id == spawn_commit_id
        and observed.prepared_parent_authority_id == prepared.prepared_delivery_slot_id
        and observed.prepared_file_id is None
        and observed.parent_mount_id == prepared.mount_id
        and observed.parent_device == prepared.device
        and observed.parent_inode == prepared.inode
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.kind is prepared.kind
        and observed.relative_path
        == f"{prepared.directory_relative_path}/{prepared.final_file_name}"
        and observed.file_type == "regular"
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.mode == 0o400
        and observed.link_count == 1
        and observed.mount_id == prepared.mount_id
        and observed.device == prepared.device
        and observed.inode != prepared.inode
        and observed.size_bytes <= prepared.payload_size_limit_bytes
    )


def _activated_result_matches_prepared_file(
    observed: RunActionActivatedFileObservation,
    prepared: RunActionPreparedFile,
    parent: RunActionPreparedRuntimeDirectory,
    spawn_commit_id: str,
) -> bool:
    if (
        type(observed) is not RunActionActivatedFileObservation
        or type(prepared) is not RunActionPreparedFile
        or type(parent) is not RunActionPreparedRuntimeDirectory
    ):
        return False
    return (
        observed.spawn_commit_id == spawn_commit_id
        and parent.kind is RunActionPreparedRuntimeDirectoryKind.RESULT
        and observed.prepared_parent_authority_id
        == parent.prepared_runtime_directory_id
        and observed.prepared_file_id == prepared.prepared_file_id
        and observed.parent_mount_id == parent.mount_id
        and observed.parent_device == parent.device
        and observed.parent_inode == parent.inode
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.kind is RunActionPreparedFileKind.RESULT
        and observed.relative_path == prepared.relative_path
        and observed.file_type == prepared.file_type
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.mode == prepared.mode
        and observed.link_count == prepared.link_count
        and observed.parent_mount_id == observed.mount_id
        and observed.parent_device == observed.device
        and observed.mount_id == prepared.mount_id
        and observed.device == prepared.device
        and observed.inode == prepared.inode
        and observed.size_bytes == prepared.size_bytes
    )


def _activated_workspace_matches_prepared(
    observed: RunActionActivatedWorkspaceObservation | None,
    prepared: RunActionPreparedWorkspaceProof | None,
    spawn_commit_id: str,
) -> bool:
    if prepared is None:
        return observed is None
    if type(observed) is not RunActionActivatedWorkspaceObservation:
        return False
    return (
        observed.spawn_commit_id == spawn_commit_id
        and observed.prepared_workspace_proof_id == prepared.prepared_workspace_proof_id
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.source_tree_digest == prepared.observed_source_tree_digest
        and observed.git_closure_digest == prepared.observed_git_closure_digest
        and observed.source_entry_count == prepared.observed_source_entry_count
        and observed.source_size_bytes == prepared.observed_source_size_bytes
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.root_mode == prepared.root_mode
        and observed.mount_id == prepared.mount_id
        and observed.device == prepared.device
        and observed.inode == prepared.inode
    )


def _barrier_command_matches_policy(
    executable: str,
    arguments: tuple[str, ...],
    poll_interval_seconds: int,
    generation_nonce: str,
    policy: DockerRunActionExecutionPolicy,
) -> bool:
    if (
        executable != RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
        or type(arguments) is not tuple
        or len(arguments) < 11
        or arguments[:8]
        != (
            "sh",
            "-eu",
            "-c",
            RUN_ACTION_BARRIER_SCRIPT,
            RUN_ACTION_BARRIER_DUMMY_ARGUMENT,
            RUN_ACTION_BARRIER_RELEASE_DESTINATION,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            str(poll_interval_seconds),
        )
        or arguments[8] != _barrier_release_generation_marker(generation_nonce)
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in arguments[9:]
        )
    ):
        return False
    target_entrypoint = arguments[9]
    target_path = PurePosixPath(target_entrypoint)
    if (
        not target_path.is_absolute()
        or target_path == PurePosixPath("/")
        or ".." in target_path.parts
        or target_path.as_posix() != target_entrypoint
    ):
        return False
    return policy.command_template_id == content_id(
        "docker-run-action-command-template",
        {
            "arguments": arguments[10:],
            "entrypoint": target_entrypoint,
        },
    )


def _barrier_release_generation_marker(generation_nonce: str) -> str:
    if (
        not isinstance(generation_nonce, str)
        or _GENERATION_NONCE_PATTERN.fullmatch(generation_nonce) is None
    ):
        return ""
    return f'"generation_nonce":"{generation_nonce}"'


def _activated_runtime_directory_matches_prepared(
    observed: RunActionActivatedRuntimeDirectoryObservation,
    prepared: RunActionPreparedRuntimeDirectory,
    spawn_commit_id: str,
) -> bool:
    if (
        type(observed) is not RunActionActivatedRuntimeDirectoryObservation
        or type(prepared) is not RunActionPreparedRuntimeDirectory
    ):
        return False
    return (
        prepared.kind
        in {
            RunActionPreparedRuntimeDirectoryKind.CONTROL,
            RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
        }
        and observed.kind is prepared.kind
        and observed.spawn_commit_id == spawn_commit_id
        and observed.prepared_runtime_directory_id
        == prepared.prepared_runtime_directory_id
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.directory_relative_path == prepared.directory_relative_path
        and observed.directory_type == prepared.directory_type
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.mode == prepared.mode
        and observed.observed_entry_count == 0
        and observed.mount_id == prepared.mount_id
        and observed.device == prepared.device
        and observed.inode == prepared.inode
    )


def _activated_sentinel_matches_prepared(
    observed: RunActionActivatedSentinelObservation,
    prepared: RunActionRuntimeVolumeSentinelEvidence,
    spawn_commit_id: str,
) -> bool:
    if (
        type(observed) is not RunActionActivatedSentinelObservation
        or type(prepared) is not RunActionRuntimeVolumeSentinelEvidence
    ):
        return False
    return (
        observed.spawn_commit_id == spawn_commit_id
        and observed.prepared_sentinel_evidence_id
        == prepared.runtime_volume_sentinel_evidence_id
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.relative_path == prepared.relative_path
        and observed.file_type == prepared.file_type
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.mode == prepared.mode
        and observed.link_count == prepared.link_count
        and observed.size_bytes == prepared.size_bytes
        and observed.content_digest == prepared.content_digest
        and observed.mount_id == prepared.mount_id
        and observed.device == prepared.device
        and observed.inode == prepared.inode
    )


def _reobserved_volume_matches_prepared(
    reobserved: RunActionRuntimeVolumeEvidence,
    prepared: RunActionRuntimeVolumeEvidence,
    delivered_files: tuple[RunActionActivatedFileObservation, ...],
) -> bool:
    if (
        type(reobserved) is not RunActionRuntimeVolumeEvidence
        or type(prepared) is not RunActionRuntimeVolumeEvidence
        or not delivered_files
        or any(
            type(observed) is not RunActionActivatedFileObservation
            for observed in delivered_files
        )
    ):
        return False
    delivered_file_count = len(delivered_files)
    delivered_size_bytes = sum(
        _allocated_size(
            observed.size_bytes,
            prepared.allocation_block_size_bytes,
        )
        for observed in delivered_files
    )
    delivered_block_count = delivered_size_bytes // prepared.allocation_block_size_bytes
    return (
        run_action_runtime_volume_occurrence_matches(reobserved, prepared)
        and reobserved.used_block_count
        == prepared.used_block_count + delivered_block_count
        and reobserved.used_size_bytes
        == prepared.used_size_bytes + delivered_size_bytes
        and reobserved.used_inode_count
        == prepared.used_inode_count + delivered_file_count
        and reobserved.available_block_count
        == prepared.available_block_count - delivered_block_count
        and reobserved.available_size_bytes
        == prepared.available_size_bytes - delivered_size_bytes
        and reobserved.available_inode_count
        == prepared.available_inode_count - delivered_file_count
    )


def run_action_activated_volume_evidence_matches(
    *,
    prepared: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence,
    activated_workspace_observation: RunActionActivatedWorkspaceObservation | None,
    activated_runtime_directory_observations: tuple[
        RunActionActivatedRuntimeDirectoryObservation, ...
    ],
    activated_sentinel_observation: RunActionActivatedSentinelObservation,
    input_file_observation: RunActionActivatedFileObservation,
    result_file_observation: RunActionActivatedFileObservation,
    credential_file_observation: RunActionActivatedFileObservation | None,
) -> bool:
    """Match one complete descriptor-observed post-delivery volume closure."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
        or type(reobserved_volume_evidence) is not RunActionRuntimeVolumeEvidence
        or type(activated_runtime_directory_observations) is not tuple
        or tuple(
            observation.kind
            for observation in activated_runtime_directory_observations
            if type(observation) is RunActionActivatedRuntimeDirectoryObservation
        )
        != (
            RunActionPreparedRuntimeDirectoryKind.CONTROL,
            RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
        )
        or any(
            type(observation) is not RunActionActivatedRuntimeDirectoryObservation
            for observation in activated_runtime_directory_observations
        )
        or type(activated_sentinel_observation)
        is not RunActionActivatedSentinelObservation
        or type(input_file_observation) is not RunActionActivatedFileObservation
        or type(result_file_observation) is not RunActionActivatedFileObservation
        or (
            activated_workspace_observation is not None
            and type(activated_workspace_observation)
            is not RunActionActivatedWorkspaceObservation
        )
        or (
            credential_file_observation is not None
            and type(credential_file_observation)
            is not RunActionActivatedFileObservation
        )
    ):
        return False
    reservation = prepared.preparation_claim.reservation
    spawn_commit_id = spawn_commit.spawn_commit_id
    credential_required = (
        prepared.preparation_claim.execution_policy.credential_policy.mode
        is RunActionCredentialMode.SUPERVISOR_FILE
    )
    delivered_files = tuple(
        observed
        for observed in (
            input_file_observation,
            credential_file_observation,
        )
        if observed is not None
    )
    stable_inodes = {
        prepared.runtime_volume_evidence.root_inode,
        prepared.runtime_volume_evidence.sentinel_evidence.inode,
        prepared.input_delivery_slot.inode,
        prepared.result_directory.inode,
        prepared.temporary_directory.inode,
        prepared.control_directory.inode,
        prepared.result_file.inode,
        *(
            ()
            if prepared.credential_delivery_slot is None
            else (prepared.credential_delivery_slot.inode,)
        ),
        *(
            ()
            if prepared.workspace_proof is None
            else (prepared.workspace_proof.inode,)
        ),
    }
    limits = prepared.preparation_claim.execution_policy.docker_resource_limits
    block_size = reobserved_volume_evidence.allocation_block_size_bytes
    remaining_requirement_bytes = (
        _allocated_size(
            prepared.result_file.payload_size_limit_bytes,
            block_size,
        )
        + _allocated_size(
            limits.runtime_temporary_reservation_size_bytes,
            block_size,
        )
        + _allocated_size(
            prepared.preparation_claim.execution_policy.supervisor_limits.release_receipt_size_bytes,
            block_size,
        )
        + _allocated_size(
            prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes,
            block_size,
        )
    )
    return (
        spawn_commit.reservation_id == reservation.reservation_id
        and spawn_commit.prepared_execution_id == prepared.prepared_execution_id
        and spawn_commit.boundary_identity == reservation.intent.boundary_identity
        and spawn_commit.security_observation_id
        == reservation.frontier.security_observation_id
        and spawn_commit.provider_execution_id
        == prepared.inert_container_evidence.container_id
        and _activated_workspace_matches_prepared(
            activated_workspace_observation,
            prepared.workspace_proof,
            spawn_commit_id,
        )
        and all(
            _activated_runtime_directory_matches_prepared(
                observed,
                expected,
                spawn_commit_id,
            )
            for observed, expected in zip(
                activated_runtime_directory_observations,
                (prepared.control_directory, prepared.temporary_directory),
                strict=True,
            )
        )
        and _activated_sentinel_matches_prepared(
            activated_sentinel_observation,
            prepared.runtime_volume_evidence.sentinel_evidence,
            spawn_commit_id,
        )
        and _activated_delivery_matches_slot(
            input_file_observation,
            prepared.input_delivery_slot,
            spawn_commit_id,
        )
        and _activated_result_matches_prepared_file(
            result_file_observation,
            prepared.result_file,
            prepared.result_directory,
            spawn_commit_id,
        )
        and input_file_observation.size_bytes == reservation.request_blob.size_bytes
        and input_file_observation.content_authority_id
        == reservation.request_blob.request_blob_id
        and input_file_observation.content_digest == reservation.request_blob.digest
        and (prepared.credential_delivery_slot is not None) == credential_required
        and (credential_file_observation is not None) == credential_required
        and (
            not credential_required
            or (
                _activated_delivery_matches_slot(
                    credential_file_observation,
                    prepared.credential_delivery_slot,
                    spawn_commit_id,
                )
                and credential_file_observation.content_authority_id
                == run_action_credential_lease_authority_id(prepared, spawn_commit)
            )
        )
        and len({observed.inode for observed in delivered_files})
        == len(delivered_files)
        and not ({observed.inode for observed in delivered_files} & stable_inodes)
        and _reobserved_volume_matches_prepared(
            reobserved_volume_evidence,
            prepared.runtime_volume_evidence,
            delivered_files,
        )
        and remaining_requirement_bytes
        < reobserved_volume_evidence.available_size_bytes
        and limits.runtime_temporary_reservation_inode_count + 2
        < reobserved_volume_evidence.available_inode_count
    )


def run_action_runtime_volume_occurrence_matches(
    observed: RunActionRuntimeVolumeEvidence,
    prepared: RunActionRuntimeVolumeEvidence,
) -> bool:
    """Match immutable physical identity while permitting execution-time usage."""

    if (
        type(observed) is not RunActionRuntimeVolumeEvidence
        or type(prepared) is not RunActionRuntimeVolumeEvidence
    ):
        return False
    return (
        observed.volume_authority == prepared.volume_authority
        and observed.docker_volume_occurrence_digest
        == prepared.docker_volume_occurrence_digest
        and observed.volume_keeper_evidence_id == prepared.volume_keeper_evidence_id
        and observed.keeper_container_id == prepared.keeper_container_id
        and observed.keeper_process_id == prepared.keeper_process_id
        and observed.keeper_process_start_time_ticks
        == prepared.keeper_process_start_time_ticks
        and observed.keeper_process_cgroup_path == prepared.keeper_process_cgroup_path
        and observed.root_mount_id == prepared.root_mount_id
        and observed.root_device == prepared.root_device
        and observed.root_inode == prepared.root_inode
        and observed.observed_volume_name == prepared.observed_volume_name
        and observed.observed_labels == prepared.observed_labels
        and observed.observed_scope == prepared.observed_scope
        and observed.observed_driver == prepared.observed_driver
        and observed.observed_driver_options == prepared.observed_driver_options
        and observed.observed_filesystem_type == prepared.observed_filesystem_type
        and observed.observed_mount_flags == prepared.observed_mount_flags
        and observed.observed_owner_user_id == prepared.observed_owner_user_id
        and observed.observed_owner_group_id == prepared.observed_owner_group_id
        and observed.observed_root_mode == prepared.observed_root_mode
        and observed.allocation_block_size_bytes == prepared.allocation_block_size_bytes
        and observed.effective_block_count == prepared.effective_block_count
        and observed.effective_size_bytes == prepared.effective_size_bytes
        and observed.effective_inode_limit == prepared.effective_inode_limit
        and observed.sentinel_evidence == prepared.sentinel_evidence
    )


def preparation_container_name(claim: RunActionPreparationClaim) -> str:
    """Derive the sole Docker name from the semantic preparation claim."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "run action container name requires an exact preparation claim"
        )
    return _CONTAINER_NAME_PREFIX + claim.preparation_claim_id.rsplit(":sha256:", 1)[1]


def preparation_container_labels(
    claim: RunActionPreparationClaim,
) -> tuple[RunActionContainerLabel, ...]:
    """Derive the complete label set without a prepared-execution back-edge."""

    return _preparation_resource_labels(claim, "execution")


def run_action_keeper_process_cgroup_path(
    policy: DockerRunActionExecutionPolicy,
    container_id: str,
) -> str:
    """Derive the exact physical cgroup path issued for one keeper process."""

    if (
        type(policy) is not DockerRunActionExecutionPolicy
        or type(container_id) is not str
        or _DOCKER_CONTAINER_ID_PATTERN.fullmatch(container_id) is None
    ):
        raise RunActionSupervisorContractError(
            "keeper process cgroup path requires exact policy and container identity"
        )
    slice_name = policy.sandbox_spec.cgroup_parent_id
    stem_parts = slice_name.removesuffix(".slice").split("-")
    slice_path = "/".join(
        f"{'-'.join(stem_parts[:part_count])}.slice"
        for part_count in range(1, len(stem_parts) + 1)
    )
    return f"/{slice_path}/docker-{container_id}.scope"


def preparation_volume_name(claim: RunActionPreparationClaim) -> str:
    """Derive the sole runtime-volume name from its semantic claim."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "runtime volume name requires an exact preparation claim"
        )
    return (
        _RUNTIME_VOLUME_NAME_PREFIX
        + claim.preparation_claim_id.rsplit(
            ":sha256:",
            1,
        )[1]
    )


def preparation_volume_labels(
    claim: RunActionPreparationClaim,
    generation_nonce: str,
) -> tuple[RunActionContainerLabel, ...]:
    """Derive the generation-bound labels owned by one runtime volume."""

    labels = (
        *_preparation_resource_labels(claim, "runtime-volume"),
        RunActionContainerLabel(
            key=f"{_PREPARATION_LABEL_PREFIX}generation",
            value=runtime_volume_sentinel_identity(generation_nonce),
        ),
    )
    return tuple(sorted(labels, key=lambda label: label.key))


def preparation_keeper_container_name(claim: RunActionPreparationClaim) -> str:
    """Derive the sole keeper-container name from its semantic claim."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "runtime volume keeper name requires an exact preparation claim"
        )
    return (
        _KEEPER_CONTAINER_NAME_PREFIX
        + claim.preparation_claim_id.rsplit(
            ":sha256:",
            1,
        )[1]
    )


def preparation_keeper_container_labels(
    claim: RunActionPreparationClaim,
) -> tuple[RunActionContainerLabel, ...]:
    """Derive the exact labels owned by the claim's keeper container."""

    return _preparation_resource_labels(claim, "volume-keeper")


def preparation_main_mounts(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
) -> tuple[RunActionPreparedMount, ...]:
    """Derive the sole main-container mount set from durable preparation authority."""

    if (
        type(claim) is not RunActionPreparationClaim
        or type(authority) is not RunActionRuntimeVolumeAuthority
        or authority.preparation_claim_id != claim.preparation_claim_id
    ):
        raise RunActionSupervisorContractError(
            "run action main mounts require one exact claim and runtime volume"
        )
    return _expected_prepared_mounts(claim, authority.volume_name)


def issue_runtime_volume_authority(
    claim: RunActionPreparationClaim,
    generation_nonce: str,
) -> RunActionRuntimeVolumeAuthority:
    """Issue one host-local generation under a deterministic preparation claim."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "runtime volume authority requires an exact preparation claim"
        )
    sentinel_identity = runtime_volume_sentinel_identity(generation_nonce)
    policy = claim.execution_policy
    limits = policy.docker_resource_limits
    return RunActionRuntimeVolumeAuthority.mint(
        preparation_claim_id=claim.preparation_claim_id,
        volume_name=preparation_volume_name(claim),
        labels=preparation_volume_labels(claim, generation_nonce),
        driver="local",
        driver_options=_runtime_volume_driver_options(
            owner_user_id=policy.user_id,
            owner_group_id=policy.group_id,
            root_mode=0o700,
            size_limit_bytes=limits.runtime_volume_size_bytes,
            inode_limit=limits.runtime_volume_inode_limit,
        ),
        generation_nonce=generation_nonce,
        sentinel_relative_path=".kapso-generation",
        sentinel_identity=sentinel_identity,
        owner_user_id=policy.user_id,
        owner_group_id=policy.group_id,
        root_mode=0o700,
        size_limit_bytes=limits.runtime_volume_size_bytes,
        inode_limit=limits.runtime_volume_inode_limit,
        nosuid=True,
        nodev=True,
        noswap=True,
        execution_allowed=True,
    )


def runtime_volume_sentinel_identity(generation_nonce: str) -> str:
    """Bind one unguessable volume generation to its in-volume sentinel."""

    if (
        not isinstance(generation_nonce, str)
        or _GENERATION_NONCE_PATTERN.fullmatch(generation_nonce) is None
    ):
        raise RunActionSupervisorContractError(
            "runtime volume generation nonce must be 32 lowercase hex characters"
        )
    return content_id(
        "run-action-runtime-volume-sentinel",
        {"generation_nonce": generation_nonce},
    )


def run_action_supervisor_helper_authority_id(
    source_path: str,
    executable_digest: str,
) -> str:
    """Bind the supervisor helper bytes to their sole read-only mount destination."""

    _require_absolute_host_path(
        source_path,
        "runtime supervisor helper source",
    )
    if (
        not isinstance(executable_digest, str)
        or _SHA256_DIGEST_PATTERN.fullmatch(executable_digest) is None
    ):
        raise RunActionSupervisorContractError(
            "runtime supervisor helper digest is invalid"
        )
    return content_id(
        "run-action-helper-executable-authority",
        {
            "destination": RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            "digest": executable_digest,
            "mount_access": RunActionPreparedMountAccess.READ_ONLY.value,
            "source_path": source_path,
        },
    )


def run_action_docker_init_authority_id(
    source_path: str,
    executable_digest: str,
) -> str:
    """Bind intended Docker-init source bytes to the fixed destination."""

    _require_absolute_host_path(
        source_path,
        "Docker init executable source",
    )
    if (
        not isinstance(executable_digest, str)
        or _SHA256_DIGEST_PATTERN.fullmatch(executable_digest) is None
    ):
        raise RunActionSupervisorContractError(
            "Docker init executable digest is invalid"
        )
    return content_id(
        "run-action-docker-init-executable-authority",
        {
            "destination": RUN_ACTION_DOCKER_INIT_DESTINATION,
            "digest": executable_digest,
            "mount_access": RunActionPreparedMountAccess.READ_ONLY.value,
            "source_path": source_path,
        },
    )


def runtime_volume_driver_options(
    authority: RunActionRuntimeVolumeAuthority,
) -> tuple[str, ...]:
    """Render the exact normalized local-driver tmpfs option mapping."""

    if type(authority) is not RunActionRuntimeVolumeAuthority:
        raise RunActionSupervisorContractError(
            "runtime volume options require exact authority"
        )
    return _runtime_volume_driver_options(
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        root_mode=authority.root_mode,
        size_limit_bytes=authority.size_limit_bytes,
        inode_limit=authority.inode_limit,
    )


def _runtime_volume_driver_options(
    *,
    owner_user_id: int,
    owner_group_id: int,
    root_mode: int,
    size_limit_bytes: int,
    inode_limit: int,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                "device=tmpfs",
                (
                    "o=nodev,nosuid,noswap,"
                    f"size={size_limit_bytes},"
                    f"nr_inodes={inode_limit},"
                    f"mode={root_mode:04o},"
                    f"uid={owner_user_id},"
                    f"gid={owner_group_id}"
                ),
                "type=tmpfs",
            )
        )
    )


def _preparation_resource_labels(
    claim: RunActionPreparationClaim,
    role: str,
) -> tuple[RunActionContainerLabel, ...]:
    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "run action resource labels require an exact preparation claim"
        )
    require_identifier(role, "run action preparation resource role")
    values = {
        f"{_PREPARATION_LABEL_PREFIX}claim": claim.preparation_claim_id,
        f"{_PREPARATION_LABEL_PREFIX}reservation": claim.reservation.reservation_id,
        f"{_PREPARATION_LABEL_PREFIX}role": role,
    }
    return tuple(
        RunActionContainerLabel(key=key, value=value)
        for key, value in sorted(values.items())
    )


def _expected_prepared_mounts(
    claim: RunActionPreparationClaim,
    volume_name: str,
) -> tuple[RunActionPreparedMount, ...]:
    filesystem = claim.execution_policy.filesystem_policy
    specifications = [
        (
            RunActionPreparedMountKind.INPUT,
            filesystem.input_destination,
            RunActionPreparedMountAccess.READ_ONLY,
        ),
        (
            RunActionPreparedMountKind.RESULT,
            filesystem.result_destination,
            RunActionPreparedMountAccess.READ_WRITE,
        ),
        (
            RunActionPreparedMountKind.TEMPORARY,
            filesystem.temporary_filesystem_destination,
            RunActionPreparedMountAccess.READ_WRITE,
        ),
        (
            RunActionPreparedMountKind.CONTROL,
            RUN_ACTION_BARRIER_CONTROL_DESTINATION,
            RunActionPreparedMountAccess.READ_ONLY,
        ),
    ]
    if filesystem.credential_destination is not None:
        specifications.append(
            (
                RunActionPreparedMountKind.CREDENTIAL,
                filesystem.credential_destination,
                RunActionPreparedMountAccess.READ_ONLY,
            )
        )
    workspace_access = claim.reservation.intent.workspace_access
    if workspace_access is not RunFrontierWorkspaceAccess.NONE:
        specifications.append(
            (
                RunActionPreparedMountKind.WORKSPACE,
                filesystem.workspace_destination,
                (
                    RunActionPreparedMountAccess.READ_WRITE
                    if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                    else RunActionPreparedMountAccess.READ_ONLY
                ),
            )
        )
    mounts = tuple(
        RunActionPreparedMount(
            kind=kind,
            volume_name=volume_name,
            volume_subpath=_RUNTIME_VOLUME_SUBPATHS[kind.value],
            container_destination=destination,
            mount_type="volume",
            source_access=RunActionPreparedMountAccess.READ_WRITE,
            container_access=container_access,
            host_config_volume_subpath=_RUNTIME_VOLUME_SUBPATHS[kind.value],
        )
        for kind, destination, container_access in specifications
    )
    return tuple(sorted(mounts, key=lambda mount: mount.container_destination))


def _allocated_size(size_bytes: int, block_size_bytes: int) -> int:
    return ((size_bytes + block_size_bytes - 1) // block_size_bytes) * block_size_bytes


def _require_absolute_container_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", ",", "\r", "\n", '"'))
    ):
        raise RunActionSupervisorContractError(
            "run action container path must be normalized and absolute"
        )
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
        or path == PurePosixPath("/")
    ):
        raise RunActionSupervisorContractError(
            "run action container path must be normalized and absolute"
        )


def _require_absolute_host_path(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", ",", "\r", "\n", '"'))
    ):
        raise RunActionSupervisorContractError(
            f"{name} must be normalized and absolute"
        )
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
        or path == PurePosixPath("/")
    ):
        raise RunActionSupervisorContractError(
            f"{name} must be normalized and absolute"
        )


def _require_namespaced_content_id(
    value: str | None,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionSupervisorContractError(f"{name} uses another namespace")


def _docker_timestamp_order_key(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    match = _DOCKER_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        return None
    parts = tuple(
        int(match.group(name))
        for name in ("year", "month", "day", "hour", "minute", "second")
    )
    year, month, day, hour, minute, second = parts
    leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = (
        31,
        29 if leap_year else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    if (
        year <= 0
        or not 1 <= month <= 12
        or not 1 <= day <= month_lengths[month - 1]
        or not 0 <= hour <= 23
        or not 0 <= minute <= 59
        or not 0 <= second <= 59
    ):
        return None
    fraction = match.group("fraction")
    nanosecond = int(fraction.ljust(9, "0")) if fraction is not None else 0
    return (*parts, nanosecond)


__all__ = [
    "DockerRunActionCreateInspectProjection",
    "DockerRunActionExecutionPolicy",
    "DockerRunActionKeeperCreateInspectProjection",
    "DockerRunActionResourceLimits",
    "DockerRunActionSafeCreateDefaults",
    "DockerRunActionSandboxSpec",
    "RUN_ACTION_BARRIER_CONTROL_DESTINATION",
    "RUN_ACTION_BARRIER_DUMMY_ARGUMENT",
    "RUN_ACTION_BARRIER_PROTOCOL_VERSION",
    "RUN_ACTION_BARRIER_RELEASE_DESTINATION",
    "RUN_ACTION_BARRIER_SCRIPT",
    "RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE",
    "RUN_ACTION_DOCKER_INIT_DESTINATION",
    "RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER",
    "RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION",
    "RUN_ACTION_SUPERVISOR_HELPER_DESTINATION",
    "RunActionActivatedFileObservation",
    "RunActionActivatedSentinelObservation",
    "RunActionActivatedRuntimeDirectoryObservation",
    "RunActionActivatedWorkspaceObservation",
    "RunActionActivationRevalidationReceipt",
    "RunActionActivationNetworkMode",
    "RunActionContainerLabel",
    "RunActionCredentialMode",
    "RunActionCredentialPolicy",
    "RunActionDockerInitSourceEvidence",
    "RunActionFilesystemPolicy",
    "RunActionInertContainerEvidence",
    "RunActionSupervisorHelperEvidence",
    "RunActionMountedKeeperHelperEvidence",
    "RunActionNetworkPolicy",
    "RunActionPreparationAllocation",
    "RunActionPreparationClaim",
    "RunActionPreparedDeliverySlot",
    "RunActionPreparedExecution",
    "RunActionPreparedFile",
    "RunActionPreparedFileKind",
    "RunActionPreparedMount",
    "RunActionPreparedMountAccess",
    "RunActionPreparedMountKind",
    "RunActionPreparedRuntimeDirectory",
    "RunActionPreparedRuntimeDirectoryKind",
    "RunActionPreparedWorkspaceProof",
    "RunActionRuntimeVolumeAuthority",
    "RunActionRuntimeVolumeEvidence",
    "RunActionRuntimeVolumeLayoutProof",
    "RunActionRuntimeVolumeSentinelEvidence",
    "RunActionResultCaptureReceipt",
    "RunActionStaticEnvironmentVariable",
    "RunActionSupervisorLimits",
    "RunActionSupervisorContractError",
    "RunActionTerminalObservation",
    "RunActionVolumeKeeperEvidence",
    "preparation_container_labels",
    "preparation_container_name",
    "preparation_keeper_container_labels",
    "preparation_keeper_container_name",
    "preparation_main_mounts",
    "preparation_volume_labels",
    "preparation_volume_name",
    "issue_runtime_volume_authority",
    "runtime_volume_driver_options",
    "run_action_supervisor_helper_authority_id",
    "runtime_volume_sentinel_identity",
    "run_action_keeper_process_cgroup_path",
    "run_action_activated_volume_evidence_matches",
    "run_action_docker_init_authority_id",
    "run_action_runtime_volume_occurrence_matches",
]
