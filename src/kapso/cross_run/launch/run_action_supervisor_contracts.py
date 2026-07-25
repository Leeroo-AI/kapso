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
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit

_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
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
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_CONTAINER_NAME_PREFIX = "kapso-run-action-"
_PREPARATION_LABEL_PREFIX = "com.kapso.run-action."


class RunActionSupervisorContractError(ValueError):
    """A prepared execution cannot prove the exact inert Docker occurrence."""


class RunActionCredentialMode(str, Enum):
    """How a committed execution may receive provider credentials."""

    NONE = "none"
    SUPERVISOR_FILE = "supervisor_file"


class RunActionActivationNetworkMode(str, Enum):
    """Network authority that may be attached only after spawn commit."""

    NONE = "none"
    BROKER_ONLY = "broker_only"


class RunActionPreparedSlotKind(str, Enum):
    """Purpose of one empty supervisor-owned pre-commit directory."""

    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"


class RunActionPreparedMountKind(str, Enum):
    """Identity of one bind source admitted to an inert container."""

    WORKSPACE = "workspace"
    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"


class RunActionPreparedMountAccess(str, Enum):
    """Container-side access to one exact bind source."""

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
        if type(self.activation_mode) is not RunActionActivationNetworkMode:
            raise RunActionSupervisorContractError(
                "run action network policy uses an unknown mode"
            )
        if self.activation_mode is RunActionActivationNetworkMode.NONE:
            if self.broker_endpoint_ids:
                raise RunActionSupervisorContractError(
                    "network-free run action policy carries broker endpoints"
                )
            return
        if not self.broker_endpoint_ids or self.broker_endpoint_ids != tuple(
            sorted(set(self.broker_endpoint_ids))
        ):
            raise RunActionSupervisorContractError(
                "brokered run action network policy must be sorted and non-empty"
            )
        for endpoint_id in self.broker_endpoint_ids:
            require_identifier(endpoint_id, "run action network broker endpoint")


@dataclass(frozen=True)
class DockerRunActionUlimit(StrictContract):
    """One exact Docker process limit."""

    name: str
    soft_limit: int
    hard_limit: int

    def _validate(self) -> None:
        require_identifier(self.name, "Docker run action ulimit")
        if (
            type(self.soft_limit) is not int
            or self.soft_limit <= 0
            or type(self.hard_limit) is not int
            or self.hard_limit < self.soft_limit
        ):
            raise RunActionSupervisorContractError(
                "Docker run action ulimit is invalid"
            )


@dataclass(frozen=True)
class DockerRunActionResourceLimits(StrictContract):
    """Resource controls observable in the retained Docker configuration."""

    docker_resource_limits_id: str
    cpu_period_microseconds: int
    cpu_quota_microseconds: int
    cpu_shares: int
    nano_cpus: int
    cpu_realtime_period_microseconds: int
    cpu_realtime_runtime_microseconds: int
    cpuset_cpu_ids: tuple[int, ...]
    cpuset_memory_node_ids: tuple[int, ...]
    memory_size_bytes: int
    memory_reservation_size_bytes: int
    memory_swap_size_bytes: int
    memory_swappiness_percentage: int
    oom_kill_disabled: bool
    oom_score_adjustment: int
    process_limit: int
    block_io_weight: int
    block_io_read_bandwidth_rule_ids: tuple[str, ...]
    block_io_write_bandwidth_rule_ids: tuple[str, ...]
    block_io_read_iops_rule_ids: tuple[str, ...]
    block_io_write_iops_rule_ids: tuple[str, ...]
    ulimits: tuple[DockerRunActionUlimit, ...]
    shared_memory_size_bytes: int
    temporary_filesystem_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-resource-limits"
    IDENTITY_FIELD: ClassVar[str] = "docker_resource_limits_id"

    def _validate(self) -> None:
        values = (
            self.cpu_period_microseconds,
            self.cpu_quota_microseconds,
            self.cpu_shares,
            self.nano_cpus,
            self.memory_size_bytes,
            self.memory_reservation_size_bytes,
            self.memory_swap_size_bytes,
            self.process_limit,
            self.block_io_weight,
            self.shared_memory_size_bytes,
            self.temporary_filesystem_size_bytes,
        )
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or not 1_000 <= self.cpu_period_microseconds <= 1_000_000
            or not 2 <= self.cpu_shares <= 262_144
            or self.memory_swap_size_bytes < self.memory_size_bytes
            or self.memory_reservation_size_bytes > self.memory_size_bytes
            or type(self.cpu_realtime_period_microseconds) is not int
            or self.cpu_realtime_period_microseconds < 0
            or type(self.cpu_realtime_runtime_microseconds) is not int
            or self.cpu_realtime_runtime_microseconds < 0
            or (
                bool(self.cpu_realtime_period_microseconds)
                != bool(self.cpu_realtime_runtime_microseconds)
            )
            or (
                self.cpu_realtime_runtime_microseconds
                > self.cpu_realtime_period_microseconds
            )
            or type(self.memory_swappiness_percentage) is not int
            or not 0 <= self.memory_swappiness_percentage <= 100
            or self.oom_kill_disabled is not False
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
            or any(type(limit) is not DockerRunActionUlimit for limit in self.ulimits)
            or tuple(limit.name for limit in self.ulimits)
            != tuple(sorted({limit.name for limit in self.ulimits}))
            or not 10 <= self.block_io_weight <= 1_000
        ):
            raise RunActionSupervisorContractError(
                "Docker run action resource limits are invalid"
            )
        for rule_ids in (
            self.block_io_read_bandwidth_rule_ids,
            self.block_io_write_bandwidth_rule_ids,
            self.block_io_read_iops_rule_ids,
            self.block_io_write_iops_rule_ids,
        ):
            if rule_ids != tuple(sorted(set(rule_ids))):
                raise RunActionSupervisorContractError(
                    "Docker block-I/O rules are not canonical"
                )
            for rule_id in rule_ids:
                require_identifier(rule_id, "Docker block-I/O rule")


@dataclass(frozen=True)
class RunActionSupervisorLimits(StrictContract):
    """Non-Docker time and byte bounds enforced by the trusted supervisor."""

    supervisor_limits_id: str
    execution_timeout_seconds: int
    termination_grace_seconds: int
    stdout_size_bytes: int
    stderr_size_bytes: int
    result_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-supervisor-limits"
    IDENTITY_FIELD: ClassVar[str] = "supervisor_limits_id"

    def _validate(self) -> None:
        values = (
            self.execution_timeout_seconds,
            self.termination_grace_seconds,
            self.stdout_size_bytes,
            self.stderr_size_bytes,
            self.result_size_bytes,
        )
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or self.termination_grace_seconds >= self.execution_timeout_seconds
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
        for value, name in (
            (self.seccomp_profile_id, "seccomp profile"),
            (self.apparmor_profile_id, "AppArmor profile"),
            (self.runtime_id, "runtime"),
        ):
            require_identifier(value, f"Docker run action sandbox {name}")
        if (
            self.read_only_root_filesystem is not True
            or self.privileged is not False
            or self.capability_additions
            or self.capability_drops != ("ALL",)
            or self.device_authority_ids
            != tuple(sorted(set(self.device_authority_ids)))
            or self.device_request_authority_ids
            != tuple(sorted(set(self.device_request_authority_ids)))
            or self.device_cgroup_rule_ids
            != tuple(sorted(set(self.device_cgroup_rule_ids)))
            or bool(self.device_authority_ids) != bool(self.device_cgroup_rule_ids)
            or self.supplementary_group_ids
            or self.pid_namespace_mode != "private"
            or self.ipc_namespace_mode != "private"
            or self.uts_namespace_mode != "private"
            or self.cgroup_namespace_mode != "private"
            or self.user_namespace_mode not in {"private", "daemon_remapped"}
            or self.sysctl_ids
            or self.no_new_privileges is not True
            or self.security_option_ids
            != tuple(
                sorted(
                    {
                        f"apparmor:{self.apparmor_profile_id}",
                        "no-new-privileges",
                        f"seccomp:{self.seccomp_profile_id}",
                    }
                )
            )
            or not self.masked_system_paths
            or self.masked_system_paths != tuple(sorted(set(self.masked_system_paths)))
            or not self.read_only_system_paths
            or self.read_only_system_paths
            != tuple(sorted(set(self.read_only_system_paths)))
            or self.log_driver != "none"
            or self.log_option_ids
            or self.init_process is not True
            or self.isolation_mode != "default"
        ):
            raise RunActionSupervisorContractError(
                "Docker run action sandbox permits expanded privilege"
            )
        require_identifier(
            self.cgroup_parent_id,
            "Docker run action cgroup parent",
        )
        for values, name in (
            (self.device_authority_ids, "device authority"),
            (self.device_request_authority_ids, "device request authority"),
            (self.device_cgroup_rule_ids, "device cgroup rule"),
        ):
            for value in values:
                require_identifier(value, f"Docker run action {name}")
        for paths, name in (
            (self.masked_system_paths, "masked system"),
            (self.read_only_system_paths, "read-only system"),
        ):
            for path in paths:
                _require_absolute_container_path(path)


@dataclass(frozen=True)
class RunActionFilesystemPolicy(StrictContract):
    """Container destinations and workspace authority for one action kind."""

    filesystem_policy_id: str
    workspace_access: RunFrontierWorkspaceAccess
    workspace_destination: str | None
    input_destination: str
    result_destination: str
    credential_destination: str | None
    working_directory: str
    temporary_filesystem_destination: str
    temporary_filesystem_mode: int
    temporary_filesystem_read_only: bool
    temporary_filesystem_nosuid: bool
    temporary_filesystem_nodev: bool
    temporary_filesystem_noexec: bool

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
        if (
            self.temporary_filesystem_mode != 0o700
            or self.temporary_filesystem_read_only is not False
            or self.temporary_filesystem_nosuid is not True
            or self.temporary_filesystem_nodev is not True
            or self.temporary_filesystem_noexec is not True
        ):
            raise RunActionSupervisorContractError(
                "run action temporary filesystem policy is unsafe"
            )
        destinations = (
            self.input_destination,
            self.result_destination,
            self.working_directory,
            self.temporary_filesystem_destination,
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
        mount_destinations = tuple(
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
            for destination in mount_destinations
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
            or type(self.group_id) is not int
            or self.group_id <= 0
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
        has_credential_destination = (
            self.filesystem_policy.credential_destination is not None
        )
        if has_credential_destination != (
            self.credential_policy.mode is RunActionCredentialMode.SUPERVISOR_FILE
        ):
            raise RunActionSupervisorContractError(
                "run action credential destination differs from credential policy"
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
class RunActionFilesystemIdentity(StrictContract):
    """One descriptor-observed filesystem object identity."""

    mount_id: int
    device: int
    inode: int
    inode_generation: int

    def _validate(self) -> None:
        if (
            type(self.mount_id) is not int
            or self.mount_id <= 0
            or type(self.device) is not int
            or self.device <= 0
            or type(self.inode) is not int
            or self.inode <= 0
            or type(self.inode_generation) is not int
            or self.inode_generation <= 0
        ):
            raise RunActionSupervisorContractError(
                "run action filesystem identity is invalid"
            )


@dataclass(frozen=True)
class RunActionFilesystemNodeObservation(StrictContract):
    """One no-follow node observed during a descriptor-relative walk."""

    identity: RunActionFilesystemIdentity
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    unexpected_acl_count: int
    unexpected_link_count: int

    def _validate(self) -> None:
        if (
            type(self.identity) is not RunActionFilesystemIdentity
            or self.file_type != "directory"
            or type(self.owner_user_id) is not int
            or self.owner_user_id < 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id < 0
            or type(self.mode) is not int
            or not 0 <= self.mode <= 0o777
            or self.unexpected_acl_count != 0
            or self.unexpected_link_count != 0
        ):
            raise RunActionSupervisorContractError(
                "run action filesystem node observation is unsafe"
            )


@dataclass(frozen=True)
class RunActionDescriptorWalkObservation(StrictContract):
    """Ordered root-to-leaf observation made beneath one trusted open root."""

    descriptor_walk_observation_id: str
    root_authority_id: str
    resolution_protocol_version: str
    nodes: tuple[RunActionFilesystemNodeObservation, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-descriptor-walk-observation"
    IDENTITY_FIELD: ClassVar[str] = "descriptor_walk_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.root_authority_id,
            "run-action-storage-root-authority",
            "run action storage root authority",
        )
        if (
            self.resolution_protocol_version != "openat2-beneath-no-symlink.v1"
            or len(self.nodes) < 2
            or any(
                type(node) is not RunActionFilesystemNodeObservation
                for node in self.nodes
            )
            or len({node.identity for node in self.nodes}) != len(self.nodes)
        ):
            raise RunActionSupervisorContractError(
                "run action descriptor walk is incomplete or unsafe"
            )


@dataclass(frozen=True)
class RunActionQuotaObservation(StrictContract):
    """Enforced exclusive filesystem quota for one prepared delivery slot."""

    quota_observation_id: str
    preparation_claim_id: str
    slot_kind: RunActionPreparedSlotKind
    leaf_identity: RunActionFilesystemIdentity
    quota_backend_authority_id: str
    filesystem_instance_id: str
    filesystem_mount_id: int
    filesystem_device: int
    exclusive_scope_id: str
    enabled: bool
    enforced: bool
    hard_size_bytes: int
    hard_entry_count: int
    current_size_bytes: int
    current_entry_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-quota-observation"
    IDENTITY_FIELD: ClassVar[str] = "quota_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "run action quota preparation claim",
        )
        _require_namespaced_content_id(
            self.quota_backend_authority_id,
            "run-action-quota-backend-authority",
            "run action quota backend",
        )
        require_identifier(
            self.filesystem_instance_id,
            "run action filesystem instance",
        )
        _require_namespaced_content_id(
            self.exclusive_scope_id,
            "run-action-quota-scope",
            "run action exclusive quota scope",
        )
        if (
            type(self.slot_kind) is not RunActionPreparedSlotKind
            or type(self.leaf_identity) is not RunActionFilesystemIdentity
            or self.filesystem_mount_id != self.leaf_identity.mount_id
            or self.filesystem_device != self.leaf_identity.device
            or self.enabled is not True
            or self.enforced is not True
            or type(self.hard_size_bytes) is not int
            or self.hard_size_bytes <= 0
            or type(self.hard_entry_count) is not int
            or self.hard_entry_count != 1
            or self.current_size_bytes != 0
            or self.current_entry_count != 0
        ):
            raise RunActionSupervisorContractError(
                "run action quota is absent, shared, or nonempty"
            )


@dataclass(frozen=True)
class RunActionPreparedSlot(StrictContract):
    """One empty, private, supervisor-owned directory prepared before commit."""

    prepared_slot_id: str
    preparation_claim_id: str
    kind: RunActionPreparedSlotKind
    descriptor_walk: RunActionDescriptorWalkObservation
    quota_observation: RunActionQuotaObservation
    expected_owner_user_id: int
    expected_owner_group_id: int
    expected_mode: int
    container_destination: str
    payload_size_limit_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-slot"
    IDENTITY_FIELD: ClassVar[str] = "prepared_slot_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.preparation_claim_id,
            RunActionPreparationClaim.CONTENT_NAMESPACE,
            "run action prepared slot claim",
        )
        _require_absolute_container_path(self.container_destination)
        if (
            type(self.kind) is not RunActionPreparedSlotKind
            or type(self.descriptor_walk) is not RunActionDescriptorWalkObservation
            or type(self.quota_observation) is not RunActionQuotaObservation
            or type(self.expected_owner_user_id) is not int
            or self.expected_owner_user_id < 0
            or type(self.expected_owner_group_id) is not int
            or self.expected_owner_group_id < 0
            or self.expected_mode != 0o700
            or type(self.payload_size_limit_bytes) is not int
            or self.payload_size_limit_bytes <= 0
            or self.quota_observation.hard_size_bytes < self.payload_size_limit_bytes
        ):
            raise RunActionSupervisorContractError(
                "run action prepared slot is invalid or not empty"
            )
        if any(
            node.owner_user_id != self.expected_owner_user_id
            or node.owner_group_id != self.expected_owner_group_id
            or node.mode != self.expected_mode
            for node in self.descriptor_walk.nodes[1:]
        ):
            raise RunActionSupervisorContractError(
                "run action prepared slot is not private"
            )
        leaf_identity = self.descriptor_walk.nodes[-1].identity
        quota = self.quota_observation
        if (
            quota.preparation_claim_id != self.preparation_claim_id
            or quota.slot_kind is not self.kind
            or quota.leaf_identity != leaf_identity
            or quota.exclusive_scope_id
            != quota_scope_id(
                self.preparation_claim_id,
                self.kind,
                leaf_identity,
            )
            or quota.hard_size_bytes != self.payload_size_limit_bytes
        ):
            raise RunActionSupervisorContractError(
                "run action prepared slot quota differs from its exact leaf"
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
    """One exact bind mount observed on the inert container."""

    kind: RunActionPreparedMountKind
    prepared_slot_id: str | None
    source_walk: RunActionDescriptorWalkObservation
    container_destination: str
    mount_type: str
    access: RunActionPreparedMountAccess
    bind_propagation: str
    recursive_read_only: bool
    nested_mount_count: int

    def _validate(self) -> None:
        if (
            type(self.kind) is not RunActionPreparedMountKind
            or type(self.access) is not RunActionPreparedMountAccess
            or type(self.source_walk) is not RunActionDescriptorWalkObservation
            or self.mount_type != "bind"
            or self.bind_propagation != "rprivate"
            or self.recursive_read_only
            is not (self.access is RunActionPreparedMountAccess.READ_ONLY)
            or self.nested_mount_count != 0
        ):
            raise RunActionSupervisorContractError(
                "run action prepared mount is invalid"
            )
        _require_absolute_container_path(self.container_destination)
        if self.kind is RunActionPreparedMountKind.WORKSPACE:
            if self.prepared_slot_id is not None:
                raise RunActionSupervisorContractError(
                    "run action workspace mount cannot name a prepared slot"
                )
        else:
            _require_namespaced_content_id(
                self.prepared_slot_id,
                RunActionPreparedSlot.CONTENT_NAMESPACE,
                "run action prepared mount slot",
            )


@dataclass(frozen=True)
class DockerRunActionCreateInspectProjection(StrictContract):
    """Closed normalized projection shared by Docker create and inspect."""

    create_inspect_projection_id: str
    projection_protocol_version: str
    raw_field_schema_id: str
    execution_policy: DockerRunActionExecutionPolicy
    mounts: tuple[RunActionPreparedMount, ...]
    unclassified_raw_field_count: int
    nonauthoritative_raw_field_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "docker-run-action-create-inspect-projection"
    IDENTITY_FIELD: ClassVar[str] = "create_inspect_projection_id"

    def _validate(self) -> None:
        if (
            type(self.execution_policy) is not DockerRunActionExecutionPolicy
            or self.projection_protocol_version
            != self.execution_policy.projection_protocol_version
            or self.raw_field_schema_id != self.execution_policy.raw_field_schema_id
            or any(type(mount) is not RunActionPreparedMount for mount in self.mounts)
            or tuple(mount.container_destination for mount in self.mounts)
            != tuple(sorted({mount.container_destination for mount in self.mounts}))
            or self.unclassified_raw_field_count != 0
            or type(self.nonauthoritative_raw_field_count) is not int
            or self.nonauthoritative_raw_field_count < 0
        ):
            raise RunActionSupervisorContractError(
                "Docker create/inspect projection is incomplete or noncanonical"
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
    input_slot: RunActionPreparedSlot
    result_slot: RunActionPreparedSlot
    credential_slot: RunActionPreparedSlot | None
    inert_container_evidence: RunActionInertContainerEvidence

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-execution"
    IDENTITY_FIELD: ClassVar[str] = "prepared_execution_id"

    def _validate(self) -> None:
        if (
            type(self.preparation_claim) is not RunActionPreparationClaim
            or type(self.input_slot) is not RunActionPreparedSlot
            or type(self.result_slot) is not RunActionPreparedSlot
            or (
                self.credential_slot is not None
                and type(self.credential_slot) is not RunActionPreparedSlot
            )
            or type(self.inert_container_evidence)
            is not RunActionInertContainerEvidence
        ):
            raise RunActionSupervisorContractError(
                "prepared run action execution has invalid components"
            )
        claim = self.preparation_claim
        slots = tuple(
            slot
            for slot in (self.input_slot, self.result_slot, self.credential_slot)
            if slot is not None
        )
        expected_kinds = (
            RunActionPreparedSlotKind.INPUT,
            RunActionPreparedSlotKind.RESULT,
            *(
                ()
                if claim.execution_policy.credential_policy.mode
                is RunActionCredentialMode.NONE
                else (RunActionPreparedSlotKind.CREDENTIAL,)
            ),
        )
        if (
            tuple(slot.kind for slot in slots) != expected_kinds
            or any(
                slot.preparation_claim_id != claim.preparation_claim_id
                for slot in slots
            )
            or self.input_slot.payload_size_limit_bytes
            != claim.reservation.request_blob.size_bytes
            or self.result_slot.payload_size_limit_bytes
            != claim.execution_policy.supervisor_limits.result_size_bytes
            or (
                self.credential_slot is not None
                and self.credential_slot.payload_size_limit_bytes
                != claim.execution_policy.credential_policy.maximum_delivery_size_bytes
            )
            or tuple(slot.container_destination for slot in slots)
            != (
                claim.execution_policy.filesystem_policy.input_destination,
                claim.execution_policy.filesystem_policy.result_destination,
                *(
                    ()
                    if claim.execution_policy.filesystem_policy.credential_destination
                    is None
                    else (
                        claim.execution_policy.filesystem_policy.credential_destination,
                    )
                ),
            )
        ):
            raise RunActionSupervisorContractError(
                "prepared run action slots differ from their preparation claim"
            )
        quotas = tuple(slot.quota_observation for slot in slots)
        if (
            len({quota.exclusive_scope_id for quota in quotas}) != len(quotas)
            or len({quota.quota_backend_authority_id for quota in quotas}) != 1
            or len({quota.filesystem_instance_id for quota in quotas}) != 1
            or len({quota.filesystem_mount_id for quota in quotas}) != 1
            or len({quota.filesystem_device for quota in quotas}) != 1
        ):
            raise RunActionSupervisorContractError(
                "prepared run action slot quotas are shared or span storage roots"
            )
        evidence = self.inert_container_evidence
        issued_projection = evidence.issued_create_projection
        workspace_mounts = tuple(
            mount
            for mount in issued_projection.mounts
            if mount.kind is RunActionPreparedMountKind.WORKSPACE
        )
        workspace_access = claim.reservation.intent.workspace_access
        expected_workspace_mount_count = (
            0 if workspace_access is RunFrontierWorkspaceAccess.NONE else 1
        )
        if len(workspace_mounts) != expected_workspace_mount_count:
            raise RunActionSupervisorContractError(
                "prepared run action workspace mount differs from its claim"
            )
        slot_walk_prefixes = {
            (
                slot.descriptor_walk.root_authority_id,
                slot.descriptor_walk.nodes[:-1],
            )
            for slot in slots
        }
        if len(slot_walk_prefixes) != 1:
            raise RunActionSupervisorContractError(
                "prepared run action slots do not share one private claim root"
            )
        source_walks = tuple(mount.source_walk for mount in issued_projection.mounts)
        if any(
            left.nodes[-1].identity in {node.identity for node in right.nodes}
            or right.nodes[-1].identity in {node.identity for node in left.nodes}
            for position, left in enumerate(source_walks)
            for right in source_walks[position + 1 :]
        ):
            raise RunActionSupervisorContractError(
                "prepared run action bind sources alias or contain one another"
            )
        workspace_walk = (
            None if not workspace_mounts else workspace_mounts[0].source_walk
        )
        workspace_binding = claim.reservation.frontier.workspace_before
        if workspace_walk is not None and (
            workspace_walk.root_authority_id
            == self.input_slot.descriptor_walk.root_authority_id
            or workspace_walk.nodes[-1].identity.device
            != workspace_binding.workspace_device
            or workspace_walk.nodes[-1].identity.inode
            != workspace_binding.workspace_inode
        ):
            raise RunActionSupervisorContractError(
                "prepared workspace walk differs from its durable binding"
            )
        policy = claim.execution_policy
        if (
            evidence.preparation_claim_id != claim.preparation_claim_id
            or evidence.container_name != preparation_container_name(claim)
            or evidence.labels != preparation_container_labels(claim)
            or evidence.image_authority_id != policy.image_authority.image_authority_id
            or evidence.docker_runtime_settings_digest
            != policy.docker_runtime_settings_digest
            or issued_projection.execution_policy != policy
            or issued_projection.mounts
            != _expected_prepared_mounts(claim, slots, workspace_walk)
        ):
            raise RunActionSupervisorContractError(
                "inert run action evidence differs from the prepared execution"
            )


@dataclass(frozen=True)
class RunActionActivatedSlotObservation(StrictContract):
    """Fresh post-delivery slot state without persisting payload contents."""

    activated_slot_observation_id: str
    prepared_slot_id: str
    descriptor_walk: RunActionDescriptorWalkObservation
    quota_backend_authority_id: str
    filesystem_instance_id: str
    exclusive_scope_id: str
    quota_enabled: bool
    quota_enforced: bool
    hard_size_bytes: int
    hard_entry_count: int
    current_size_bytes: int
    current_entry_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activated-slot-observation"
    IDENTITY_FIELD: ClassVar[str] = "activated_slot_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_slot_id,
            RunActionPreparedSlot.CONTENT_NAMESPACE,
            "activated run action slot",
        )
        require_identifier(
            self.filesystem_instance_id,
            "activated run action filesystem instance",
        )
        _require_namespaced_content_id(
            self.exclusive_scope_id,
            "run-action-quota-scope",
            "activated run action exclusive quota scope",
        )
        _require_namespaced_content_id(
            self.quota_backend_authority_id,
            "run-action-quota-backend-authority",
            "activated run action quota backend",
        )
        if (
            type(self.descriptor_walk) is not RunActionDescriptorWalkObservation
            or self.quota_enabled is not True
            or self.quota_enforced is not True
            or type(self.hard_size_bytes) is not int
            or self.hard_size_bytes <= 0
            or type(self.hard_entry_count) is not int
            or self.hard_entry_count <= 0
            or type(self.current_size_bytes) is not int
            or not 0 <= self.current_size_bytes <= self.hard_size_bytes
            or type(self.current_entry_count) is not int
            or not 0 <= self.current_entry_count <= self.hard_entry_count
        ):
            raise RunActionSupervisorContractError(
                "activated run action slot observation is unsafe"
            )


@dataclass(frozen=True)
class RunActionRequestDeliveryReceipt(StrictContract):
    """Exact non-secret identity of request bytes delivered after spawn commit."""

    request_delivery_receipt_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    input_slot_id: str
    request_blob_id: str
    delivered_digest: str
    delivered_size_bytes: int
    delivered_entry_count: int
    delivered_relative_name: str
    delivered_file_type: str
    delivered_owner_user_id: int
    delivered_owner_group_id: int
    delivered_mode: int
    delivered_link_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-request-delivery-receipt"
    IDENTITY_FIELD: ClassVar[str] = "request_delivery_receipt_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.prepared_execution_id,
                RunActionPreparedExecution.CONTENT_NAMESPACE,
                "request delivery prepared execution",
            ),
            (
                self.spawn_commit_id,
                RunActionSpawnCommit.CONTENT_NAMESPACE,
                "request delivery spawn commit",
            ),
            (
                self.input_slot_id,
                RunActionPreparedSlot.CONTENT_NAMESPACE,
                "request delivery input slot",
            ),
            (
                self.request_blob_id,
                "run-action-request-blob",
                "request delivery blob",
            ),
        ):
            _require_namespaced_content_id(value, namespace, name)
        if (
            _SHA256_DIGEST_PATTERN.fullmatch(self.delivered_digest) is None
            or type(self.delivered_size_bytes) is not int
            or self.delivered_size_bytes <= 0
            or self.delivered_entry_count != 1
            or self.delivered_relative_name != "request.blob"
            or self.delivered_file_type != "regular"
            or type(self.delivered_owner_user_id) is not int
            or self.delivered_owner_user_id <= 0
            or type(self.delivered_owner_group_id) is not int
            or self.delivered_owner_group_id <= 0
            or self.delivered_mode != 0o400
            or self.delivered_link_count != 1
        ):
            raise RunActionSupervisorContractError(
                "run action request delivery receipt is invalid"
            )


@dataclass(frozen=True)
class RunActionCredentialDeliveryReceipt(StrictContract):
    """Credential lease delivery identity without secret bytes, digest, or path."""

    credential_delivery_receipt_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    credential_slot_id: str
    credential_policy_id: str
    lease_authority_id: str
    delivered_size_bytes: int
    delivered_entry_count: int
    delivered_relative_name: str
    delivered_file_type: str
    delivered_owner_user_id: int
    delivered_owner_group_id: int
    delivered_mode: int
    delivered_link_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-delivery-receipt"
    IDENTITY_FIELD: ClassVar[str] = "credential_delivery_receipt_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.prepared_execution_id,
                RunActionPreparedExecution.CONTENT_NAMESPACE,
                "credential delivery prepared execution",
            ),
            (
                self.spawn_commit_id,
                RunActionSpawnCommit.CONTENT_NAMESPACE,
                "credential delivery spawn commit",
            ),
            (
                self.credential_slot_id,
                RunActionPreparedSlot.CONTENT_NAMESPACE,
                "credential delivery slot",
            ),
            (
                self.credential_policy_id,
                RunActionCredentialPolicy.CONTENT_NAMESPACE,
                "credential delivery policy",
            ),
        ):
            _require_namespaced_content_id(value, namespace, name)
        require_identifier(
            self.lease_authority_id,
            "run action credential lease authority",
        )
        if (
            type(self.delivered_size_bytes) is not int
            or self.delivered_size_bytes <= 0
            or self.delivered_entry_count != 1
            or self.delivered_relative_name != "credentials"
            or self.delivered_file_type != "regular"
            or type(self.delivered_owner_user_id) is not int
            or self.delivered_owner_user_id <= 0
            or type(self.delivered_owner_group_id) is not int
            or self.delivered_owner_group_id <= 0
            or self.delivered_mode != 0o400
            or self.delivered_link_count != 1
        ):
            raise RunActionSupervisorContractError(
                "run action credential delivery receipt is invalid"
            )


@dataclass(frozen=True)
class RunActionNoCredentialsProof(StrictContract):
    """Positive proof that a credential-free policy received no credential slot."""

    no_credentials_proof_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    credential_policy_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-no-credentials-proof"
    IDENTITY_FIELD: ClassVar[str] = "no_credentials_proof_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_execution_id,
            RunActionPreparedExecution.CONTENT_NAMESPACE,
            "no-credentials prepared execution",
        )
        _require_namespaced_content_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "no-credentials spawn commit",
        )
        _require_namespaced_content_id(
            self.credential_policy_id,
            RunActionCredentialPolicy.CONTENT_NAMESPACE,
            "no-credentials policy",
        )


@dataclass(frozen=True)
class RunActionActivationRevalidationReceipt(StrictContract):
    """Fresh pre-start observation; durable evidence, never start authority."""

    activation_revalidation_receipt_id: str
    prepared_execution: RunActionPreparedExecution
    spawn_commit: RunActionSpawnCommit
    reobserved_container_evidence: RunActionInertContainerEvidence
    input_slot_observation: RunActionActivatedSlotObservation
    result_slot_observation: RunActionActivatedSlotObservation
    credential_slot_observation: RunActionActivatedSlotObservation | None
    request_delivery_receipt: RunActionRequestDeliveryReceipt
    credential_delivery_receipt: RunActionCredentialDeliveryReceipt | None
    no_credentials_proof: RunActionNoCredentialsProof | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activation-revalidation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "activation_revalidation_receipt_id"

    def _validate(self) -> None:
        if type(self.prepared_execution) is not RunActionPreparedExecution:
            raise RunActionSupervisorContractError(
                "activation revalidation requires one prepared execution"
            )
        prepared = self.prepared_execution
        credential_mode = (
            prepared.preparation_claim.execution_policy.credential_policy.mode
        )
        if (
            type(self.reobserved_container_evidence)
            is not RunActionInertContainerEvidence
            or type(self.spawn_commit) is not RunActionSpawnCommit
            or self.spawn_commit.reservation_id
            != prepared.preparation_claim.reservation.reservation_id
            or self.spawn_commit.prepared_execution_id != prepared.prepared_execution_id
            or self.spawn_commit.boundary_identity
            != prepared.preparation_claim.reservation.intent.boundary_identity
            or self.spawn_commit.security_observation_id
            != prepared.preparation_claim.reservation.frontier.security_observation_id
            or self.spawn_commit.provider_execution_id
            != prepared.inert_container_evidence.container_id
            or self.reobserved_container_evidence != prepared.inert_container_evidence
            or not _activated_slot_matches_prepared(
                self.input_slot_observation,
                prepared.input_slot,
            )
            or not _activated_slot_matches_prepared(
                self.result_slot_observation,
                prepared.result_slot,
            )
            or self.input_slot_observation.current_size_bytes
            != prepared.preparation_claim.reservation.request_blob.size_bytes
            or self.input_slot_observation.current_entry_count != 1
            or self.result_slot_observation.current_size_bytes != 0
            or self.result_slot_observation.current_entry_count != 0
            or type(self.request_delivery_receipt)
            is not RunActionRequestDeliveryReceipt
            or self.request_delivery_receipt.prepared_execution_id
            != prepared.prepared_execution_id
            or self.request_delivery_receipt.spawn_commit_id
            != self.spawn_commit.spawn_commit_id
            or self.request_delivery_receipt.input_slot_id
            != prepared.input_slot.prepared_slot_id
            or self.request_delivery_receipt.request_blob_id
            != prepared.preparation_claim.reservation.request_blob.request_blob_id
            or self.request_delivery_receipt.delivered_digest
            != prepared.preparation_claim.reservation.request_blob.digest
            or self.request_delivery_receipt.delivered_size_bytes
            != prepared.preparation_claim.reservation.request_blob.size_bytes
            or self.request_delivery_receipt.delivered_owner_user_id
            != prepared.preparation_claim.execution_policy.user_id
            or self.request_delivery_receipt.delivered_owner_group_id
            != prepared.preparation_claim.execution_policy.group_id
        ):
            raise RunActionSupervisorContractError(
                "activation revalidation differs from prepared authority"
            )
        if credential_mode is RunActionCredentialMode.NONE:
            if self.credential_delivery_receipt is not None:
                raise RunActionSupervisorContractError(
                    "credential-free activation carries delivery authority"
                )
            if self.credential_slot_observation is not None:
                raise RunActionSupervisorContractError(
                    "credential-free activation carries a credential slot"
                )
            if (
                type(self.no_credentials_proof) is not RunActionNoCredentialsProof
                or self.no_credentials_proof.prepared_execution_id
                != prepared.prepared_execution_id
                or self.no_credentials_proof.spawn_commit_id
                != self.spawn_commit.spawn_commit_id
                or self.no_credentials_proof.credential_policy_id
                != prepared.preparation_claim.execution_policy.credential_policy.credential_policy_id
            ):
                raise RunActionSupervisorContractError(
                    "credential-free activation lacks an exact proof"
                )
        else:
            if self.no_credentials_proof is not None:
                raise RunActionSupervisorContractError(
                    "credentialed activation carries a no-credentials proof"
                )
            if (
                self.credential_slot_observation is None
                or not _activated_slot_matches_prepared(
                    self.credential_slot_observation,
                    prepared.credential_slot,
                )
                or self.credential_slot_observation.current_size_bytes <= 0
                or self.credential_slot_observation.current_entry_count != 1
                or type(self.credential_delivery_receipt)
                is not RunActionCredentialDeliveryReceipt
                or self.credential_delivery_receipt.prepared_execution_id
                != prepared.prepared_execution_id
                or self.credential_delivery_receipt.spawn_commit_id
                != self.spawn_commit.spawn_commit_id
                or self.credential_delivery_receipt.credential_slot_id
                != prepared.credential_slot.prepared_slot_id
                or self.credential_delivery_receipt.credential_policy_id
                != prepared.preparation_claim.execution_policy.credential_policy.credential_policy_id
                or self.credential_delivery_receipt.delivered_size_bytes
                != self.credential_slot_observation.current_size_bytes
                or self.credential_delivery_receipt.delivered_owner_user_id
                != prepared.preparation_claim.execution_policy.user_id
                or self.credential_delivery_receipt.delivered_owner_group_id
                != prepared.preparation_claim.execution_policy.group_id
            ):
                raise RunActionSupervisorContractError(
                    "credentialed activation lacks an exact delivered slot"
                )


def _activated_slot_matches_prepared(
    observed: RunActionActivatedSlotObservation,
    prepared: RunActionPreparedSlot | None,
) -> bool:
    if (
        type(observed) is not RunActionActivatedSlotObservation
        or type(prepared) is not RunActionPreparedSlot
    ):
        return False
    quota = prepared.quota_observation
    return (
        observed.prepared_slot_id == prepared.prepared_slot_id
        and observed.descriptor_walk == prepared.descriptor_walk
        and observed.quota_backend_authority_id == quota.quota_backend_authority_id
        and observed.filesystem_instance_id == quota.filesystem_instance_id
        and observed.exclusive_scope_id == quota.exclusive_scope_id
        and observed.quota_enabled == quota.enabled
        and observed.quota_enforced == quota.enforced
        and observed.hard_size_bytes == quota.hard_size_bytes
        and observed.hard_entry_count == quota.hard_entry_count
    )


def quota_scope_id(
    preparation_claim_id: str,
    slot_kind: RunActionPreparedSlotKind,
    leaf_identity: RunActionFilesystemIdentity,
) -> str:
    """Derive one exclusive logical-quota scope from its exact slot leaf."""

    _require_namespaced_content_id(
        preparation_claim_id,
        RunActionPreparationClaim.CONTENT_NAMESPACE,
        "run action quota scope claim",
    )
    if (
        type(slot_kind) is not RunActionPreparedSlotKind
        or type(leaf_identity) is not RunActionFilesystemIdentity
    ):
        raise RunActionSupervisorContractError(
            "run action quota scope requires an exact slot and leaf"
        )
    return content_id(
        "run-action-quota-scope",
        {
            "preparation_claim_id": preparation_claim_id,
            "slot_kind": slot_kind.value,
            "leaf_identity": leaf_identity.to_dict(),
        },
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

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionSupervisorContractError(
            "run action container labels require an exact preparation claim"
        )
    values = {
        f"{_PREPARATION_LABEL_PREFIX}boundary": (
            claim.reservation.intent.boundary_identity.boundary_identity_id
        ),
        f"{_PREPARATION_LABEL_PREFIX}claim": claim.preparation_claim_id,
        f"{_PREPARATION_LABEL_PREFIX}reservation": (claim.reservation.reservation_id),
        f"{_PREPARATION_LABEL_PREFIX}policy": (
            claim.execution_policy.docker_execution_policy_id
        ),
    }
    return tuple(
        RunActionContainerLabel(key=key, value=value)
        for key, value in sorted(values.items())
    )


def _expected_prepared_mounts(
    claim: RunActionPreparationClaim,
    slots: tuple[RunActionPreparedSlot, ...],
    workspace_walk: RunActionDescriptorWalkObservation | None,
) -> tuple[RunActionPreparedMount, ...]:
    filesystem = claim.execution_policy.filesystem_policy
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
        if workspace_walk is None:
            raise RunActionSupervisorContractError(
                "run action workspace mount lacks a descriptor walk"
            )
        mounts.append(
            RunActionPreparedMount(
                kind=RunActionPreparedMountKind.WORKSPACE,
                prepared_slot_id=None,
                source_walk=workspace_walk,
                container_destination=filesystem.workspace_destination,
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


def _require_absolute_container_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
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


def _require_namespaced_content_id(
    value: str | None,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionSupervisorContractError(f"{name} uses another namespace")


__all__ = [
    "DockerRunActionCreateInspectProjection",
    "DockerRunActionExecutionPolicy",
    "DockerRunActionResourceLimits",
    "DockerRunActionSafeCreateDefaults",
    "DockerRunActionSandboxSpec",
    "DockerRunActionUlimit",
    "RunActionActivatedSlotObservation",
    "RunActionActivationRevalidationReceipt",
    "RunActionActivationNetworkMode",
    "RunActionContainerLabel",
    "RunActionCredentialMode",
    "RunActionCredentialDeliveryReceipt",
    "RunActionCredentialPolicy",
    "RunActionDescriptorWalkObservation",
    "RunActionFilesystemIdentity",
    "RunActionFilesystemNodeObservation",
    "RunActionFilesystemPolicy",
    "RunActionInertContainerEvidence",
    "RunActionNetworkPolicy",
    "RunActionNoCredentialsProof",
    "RunActionPreparationClaim",
    "RunActionPreparedExecution",
    "RunActionPreparedMount",
    "RunActionPreparedMountAccess",
    "RunActionPreparedMountKind",
    "RunActionPreparedSlot",
    "RunActionPreparedSlotKind",
    "RunActionQuotaObservation",
    "RunActionRequestDeliveryReceipt",
    "RunActionStaticEnvironmentVariable",
    "RunActionSupervisorLimits",
    "RunActionSupervisorContractError",
    "preparation_container_labels",
    "preparation_container_name",
    "quota_scope_id",
]
