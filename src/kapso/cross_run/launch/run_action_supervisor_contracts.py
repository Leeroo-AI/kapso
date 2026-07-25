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
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_CONTAINER_NAME_PREFIX = "kapso-run-action-"
_KEEPER_CONTAINER_NAME_PREFIX = "kapso-run-action-keeper-"
_RUNTIME_VOLUME_NAME_PREFIX = "kapso-run-action-volume-"
_PREPARATION_LABEL_PREFIX = "com.kapso.run-action."
_RUNTIME_VOLUME_SENTINEL_PATH = ".kapso-generation"
RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION = "/kapso/runtime-volume"
RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION = "/kapso-supervisor/busybox"
_RUNTIME_VOLUME_SUBPATHS = {
    "workspace": "workspace",
    "input": "input",
    "result": "result",
    "credential": "credential",
    "temporary": "temporary",
}


class RunActionSupervisorContractError(ValueError):
    """A prepared execution cannot prove the exact inert Docker occurrence."""


class RunActionCredentialMode(str, Enum):
    """How a committed execution may receive provider credentials."""

    NONE = "none"
    SUPERVISOR_FILE = "supervisor_file"


class RunActionActivationNetworkMode(str, Enum):
    """Network authority that may be attached only after spawn commit."""

    NONE = "none"


class RunActionPreparedFileKind(str, Enum):
    """Purpose of one empty logical file inside the private runtime volume."""

    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"


class RunActionPreparedMountKind(str, Enum):
    """Identity of one runtime-volume subpath admitted to the main container."""

    WORKSPACE = "workspace"
    INPUT = "input"
    RESULT = "result"
    CREDENTIAL = "credential"
    TEMPORARY = "temporary"


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
    result_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-supervisor-limits"
    IDENTITY_FIELD: ClassVar[str] = "supervisor_limits_id"

    def _validate(self) -> None:
        values = (
            self.execution_timeout_seconds,
            self.termination_grace_seconds,
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
    keeper_helper_source_path: str
    keeper_helper_executable_authority_id: str
    keeper_helper_executable_digest: str
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
            self.keeper_helper_executable_authority_id,
            "run-action-helper-executable-authority",
            "runtime volume keeper helper",
        )
        _require_absolute_host_path(
            self.keeper_helper_source_path,
            "runtime volume keeper helper source",
        )
        if _SHA256_DIGEST_PATTERN.fullmatch(
            self.keeper_helper_executable_digest
        ) is None or self.keeper_helper_executable_authority_id != runtime_volume_keeper_helper_authority_id(
            self.keeper_helper_source_path,
            self.keeper_helper_executable_digest,
        ):
            raise RunActionSupervisorContractError(
                "runtime volume keeper helper differs from execution policy"
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
        if (
            self.volume_name
            != _RUNTIME_VOLUME_NAME_PREFIX
            + self.preparation_claim_id.rsplit(":sha256:", 1)[1]
            or any(type(label) is not RunActionContainerLabel for label in self.labels)
            or tuple(label.key for label in self.labels)
            != tuple(sorted({label.key for label in self.labels}))
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
            or type(self.size_limit_bytes) is not int
            or self.size_limit_bytes <= 0
            or type(self.inode_limit) is not int
            or self.inode_limit <= 0
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
                type(value) is not int or value <= 0
                for value in (
                    self.mount_id,
                    self.device,
                    self.inode,
                )
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
        if type(self.volume_authority) is not RunActionRuntimeVolumeAuthority:
            raise RunActionSupervisorContractError(
                "runtime volume evidence lacks issued authority"
            )
        authority = self.volume_authority
        if (
            self.observed_volume_name != authority.volume_name
            or self.observed_labels != authority.labels
            or self.observed_scope != "local"
            or self.observed_driver != authority.driver
            or self.observed_driver_options != authority.driver_options
            or self.observed_filesystem_type != "tmpfs"
            or self.observed_mount_flags != ("nodev", "nosuid", "noswap")
            or self.observed_owner_user_id != authority.owner_user_id
            or self.observed_owner_group_id != authority.owner_group_id
            or self.observed_root_mode != authority.root_mode
            or type(self.allocation_block_size_bytes) is not int
            or self.allocation_block_size_bytes <= 0
            or self.allocation_block_size_bytes & (self.allocation_block_size_bytes - 1)
            != 0
            or type(self.effective_block_count) is not int
            or self.effective_block_count <= 0
            or type(self.effective_size_bytes) is not int
            or not 0 < self.effective_size_bytes <= authority.size_limit_bytes
            or self.effective_size_bytes
            != self.effective_block_count * self.allocation_block_size_bytes
            or type(self.effective_inode_limit) is not int
            or not 0 < self.effective_inode_limit <= authority.inode_limit
            or type(self.used_block_count) is not int
            or not 0 <= self.used_block_count < self.effective_block_count
            or type(self.used_size_bytes) is not int
            or not 0 <= self.used_size_bytes < self.effective_size_bytes
            or self.used_size_bytes
            != self.used_block_count * self.allocation_block_size_bytes
            or type(self.used_inode_count) is not int
            or not 0 <= self.used_inode_count < self.effective_inode_limit
            or type(self.available_block_count) is not int
            or self.available_block_count <= 0
            or self.used_block_count + self.available_block_count
            != self.effective_block_count
            or type(self.available_size_bytes) is not int
            or self.available_size_bytes
            != self.available_block_count * self.allocation_block_size_bytes
            or self.used_size_bytes + self.available_size_bytes
            != self.effective_size_bytes
            or type(self.available_inode_count) is not int
            or self.available_inode_count <= 0
            or self.used_inode_count + self.available_inode_count
            != self.effective_inode_limit
            or type(self.sentinel_evidence)
            is not RunActionRuntimeVolumeSentinelEvidence
            or self.sentinel_evidence.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.sentinel_evidence.generation_nonce != authority.generation_nonce
            or self.sentinel_evidence.owner_user_id != authority.owner_user_id
            or self.sentinel_evidence.owner_group_id != authority.owner_group_id
        ):
            raise RunActionSupervisorContractError(
                "runtime volume evidence differs from effective bounded tmpfs"
            )


@dataclass(frozen=True)
class RunActionPreparedFile(StrictContract):
    """One empty payload file prepared inside the exact runtime generation."""

    prepared_file_id: str
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

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-prepared-file"
    IDENTITY_FIELD: ClassVar[str] = "prepared_file_id"

    def _validate(self) -> None:
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
        expected_paths = {
            RunActionPreparedFileKind.INPUT: "input/request.blob",
            RunActionPreparedFileKind.RESULT: "result/result.blob",
            RunActionPreparedFileKind.CREDENTIAL: "credential/credentials",
        }
        if (
            type(self.kind) is not RunActionPreparedFileKind
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.relative_path != expected_paths[self.kind]
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
        ):
            raise RunActionSupervisorContractError(
                "prepared workspace proof is incomplete"
            )


@dataclass(frozen=True)
class RunActionRuntimeVolumeLayoutProof(StrictContract):
    """Empty-before-use and exact prepared layout proof for one generation."""

    runtime_volume_layout_proof_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    empty_size_bytes: int
    empty_entry_count: int
    directory_relative_paths: tuple[str, ...]
    prepared_file_ids: tuple[str, ...]
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
        if self.prepared_workspace_proof_id is not None:
            _require_namespaced_content_id(
                self.prepared_workspace_proof_id,
                RunActionPreparedWorkspaceProof.CONTENT_NAMESPACE,
                "runtime volume layout workspace",
            )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or self.empty_size_bytes != 0
            or self.empty_entry_count != 0
            or not self.directory_relative_paths
            or self.directory_relative_paths
            != tuple(sorted(set(self.directory_relative_paths)))
            or not self.prepared_file_ids
            or self.prepared_file_ids != tuple(sorted(set(self.prepared_file_ids)))
            or any(
                require_content_id(file_id, "runtime volume layout prepared file")
                != file_id
                or file_id.split(":sha256:", 1)[0]
                != RunActionPreparedFile.CONTENT_NAMESPACE
                for file_id in self.prepared_file_ids
            )
            or type(self.logical_content_size_bytes) is not int
            or self.logical_content_size_bytes < len(self.generation_nonce)
            or type(self.logical_entry_count) is not int
            or self.logical_entry_count <= 0
            or type(self.observed_used_size_bytes) is not int
            or self.observed_used_size_bytes <= 0
            or type(self.observed_used_inode_count) is not int
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
            or len({mount.kind for mount in self.mounts}) != len(self.mounts)
            or len({mount.volume_name for mount in self.mounts}) != 1
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
            or type(self.nonauthoritative_raw_field_count) is not int
            or self.nonauthoritative_raw_field_count < 0
        ):
            raise RunActionSupervisorContractError(
                "Docker create/inspect projection is incomplete or noncanonical"
            )


@dataclass(frozen=True)
class RunActionKeeperHelperEvidence(StrictContract):
    """Physical proof of the root-owned static BusyBox keeper bind."""

    keeper_helper_evidence_id: str
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

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-keeper-helper-evidence"
    IDENTITY_FIELD: ClassVar[str] = "keeper_helper_evidence_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.helper_authority_id,
            "run-action-helper-executable-authority",
            "runtime volume keeper helper authority",
        )
        _require_absolute_host_path(
            self.source_path,
            "runtime volume keeper helper source",
        )
        if (
            self.helper_authority_id
            != runtime_volume_keeper_helper_authority_id(
                self.source_path,
                self.executable_digest,
            )
            or self.destination != RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION
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
                type(value) is not int or value <= 0
                for value in (self.mount_id, self.device, self.inode)
            )
        ):
            raise RunActionSupervisorContractError(
                "runtime volume keeper helper evidence is unsafe or substituted"
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
    helper_evidence: RunActionKeeperHelperEvidence
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
            or type(self.helper_evidence) is not RunActionKeeperHelperEvidence
            or self.projection_protocol_version
            != self.execution_policy.projection_protocol_version
            or self.raw_field_schema_id != self.execution_policy.raw_field_schema_id
            or self.volume_authority.preparation_claim_id != self.preparation_claim_id
            or self.command_executable
            != RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION
            or self.command_arguments != ("tail", "-f", "/dev/null")
            or self.helper_evidence.helper_authority_id
            != self.execution_policy.keeper_helper_executable_authority_id
            or self.helper_evidence.source_path
            != self.execution_policy.keeper_helper_source_path
            or self.helper_evidence.executable_digest
            != self.execution_policy.keeper_helper_executable_digest
            or self.volume_mount_type != "volume"
            or self.volume_mount_destination
            != RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION
            or self.volume_mount_access is not RunActionPreparedMountAccess.READ_WRITE
            or self.network_mode != "none"
            or self.exact_mount_count != 2
            or self.healthcheck_present is not False
            or self.docker_socket_mounted is not False
            or self.unclassified_raw_field_count != 0
            or type(self.nonauthoritative_raw_field_count) is not int
            or self.nonauthoritative_raw_field_count < 0
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
    container_status: str
    process_id: int
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
            or self.container_status != "running"
            or type(self.process_id) is not int
            or self.process_id <= 0
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
    input_file: RunActionPreparedFile
    result_file: RunActionPreparedFile
    credential_file: RunActionPreparedFile | None
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
            or type(self.input_file) is not RunActionPreparedFile
            or type(self.result_file) is not RunActionPreparedFile
            or (
                self.credential_file is not None
                and type(self.credential_file) is not RunActionPreparedFile
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
        files = tuple(
            prepared_file
            for prepared_file in (
                self.input_file,
                self.result_file,
                self.credential_file,
            )
            if prepared_file is not None
        )
        expected_kinds = (
            RunActionPreparedFileKind.INPUT,
            RunActionPreparedFileKind.RESULT,
            *(
                ()
                if claim.execution_policy.credential_policy.mode
                is RunActionCredentialMode.NONE
                else (RunActionPreparedFileKind.CREDENTIAL,)
            ),
        )
        if (
            tuple(prepared_file.kind for prepared_file in files) != expected_kinds
            or any(
                prepared_file.preparation_claim_id != claim.preparation_claim_id
                or prepared_file.runtime_volume_authority_id
                != authority.runtime_volume_authority_id
                or prepared_file.generation_nonce != authority.generation_nonce
                or prepared_file.owner_user_id != claim.execution_policy.user_id
                or prepared_file.owner_group_id != claim.execution_policy.group_id
                for prepared_file in files
            )
            or self.input_file.payload_size_limit_bytes
            != claim.reservation.request_blob.size_bytes
            or self.result_file.payload_size_limit_bytes
            != claim.execution_policy.supervisor_limits.result_size_bytes
            or (
                self.credential_file is not None
                and self.credential_file.payload_size_limit_bytes
                != claim.execution_policy.credential_policy.maximum_delivery_size_bytes
            )
        ):
            raise RunActionSupervisorContractError(
                "prepared run action files differ from their preparation claim"
            )
        limits = claim.execution_policy.docker_resource_limits
        policy = claim.execution_policy
        if (
            authority.preparation_claim_id != claim.preparation_claim_id
            or authority.volume_name != preparation_volume_name(claim)
            or authority.labels != preparation_volume_labels(claim)
            or authority.owner_user_id != policy.user_id
            or authority.owner_group_id != policy.group_id
            or authority.size_limit_bytes != limits.runtime_volume_size_bytes
            or authority.inode_limit != limits.runtime_volume_inode_limit
            or self.runtime_volume_evidence.volume_authority != authority
        ):
            raise RunActionSupervisorContractError(
                "prepared runtime volume differs from its execution policy"
            )
        keeper = self.volume_keeper_evidence
        keeper_projection = keeper.issued_create_projection
        if (
            keeper.preparation_claim_id != claim.preparation_claim_id
            or keeper.container_name != preparation_keeper_container_name(claim)
            or keeper.labels != preparation_keeper_container_labels(claim)
            or keeper_projection.execution_policy != policy
            or keeper_projection.volume_authority != authority
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
                    *(
                        ()
                        if self.credential_file is None
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
            len(expected_directories) + len(files) + 1 + workspace_entry_count
        )
        evidence = self.runtime_volume_evidence
        required_available_size_bytes = sum(
            _allocated_size(
                prepared_file.payload_size_limit_bytes,
                evidence.allocation_block_size_bytes,
            )
            for prepared_file in files
        ) + _allocated_size(
            limits.runtime_temporary_reservation_size_bytes,
            evidence.allocation_block_size_bytes,
        )
        if (
            layout.runtime_volume_authority_id != authority.runtime_volume_authority_id
            or layout.generation_nonce != authority.generation_nonce
            or layout.directory_relative_paths != expected_directories
            or layout.prepared_file_ids
            != tuple(sorted(prepared_file.prepared_file_id for prepared_file in files))
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
            or limits.runtime_temporary_reservation_inode_count
            >= evidence.available_inode_count
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
            or issued_projection.mounts
            != _expected_prepared_mounts(claim, authority.volume_name)
        ):
            raise RunActionSupervisorContractError(
                "inert run action evidence differs from the prepared execution"
            )


@dataclass(frozen=True)
class RunActionActivatedFileObservation(StrictContract):
    """Fresh post-delivery shape and non-secret identity of one logical file."""

    activated_file_observation_id: str
    prepared_file_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedFileKind
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str | None
    content_authority_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-activated-file-observation"
    IDENTITY_FIELD: ClassVar[str] = "activated_file_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_file_id,
            RunActionPreparedFile.CONTENT_NAMESPACE,
            "activated prepared file",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            RunActionRuntimeVolumeAuthority.CONTENT_NAMESPACE,
            "activated runtime volume",
        )
        if self.content_authority_id is not None:
            require_identifier(
                self.content_authority_id,
                "activated file content authority",
            )
        if (
            _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or type(self.kind) is not RunActionPreparedFileKind
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o600
            or self.link_count != 1
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
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
        ):
            raise RunActionSupervisorContractError(
                "activated workspace observation is invalid"
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
                type(value) is not int or value <= 0
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
            or not _reobserved_volume_matches_prepared(
                self.reobserved_volume_evidence,
                prepared.runtime_volume_evidence,
            )
            or self.reobserved_keeper_evidence != prepared.volume_keeper_evidence
            or self.reobserved_container_evidence != prepared.inert_container_evidence
            or not _activated_workspace_matches_prepared(
                self.activated_workspace_observation,
                prepared.workspace_proof,
                self.spawn_commit.spawn_commit_id,
            )
            or not _activated_sentinel_matches_prepared(
                self.activated_sentinel_observation,
                prepared.runtime_volume_evidence.sentinel_evidence,
                self.spawn_commit.spawn_commit_id,
            )
            or not _activated_file_matches_prepared(
                self.input_file_observation,
                prepared.input_file,
            )
            or not _activated_file_matches_prepared(
                self.result_file_observation,
                prepared.result_file,
            )
            or self.input_file_observation.size_bytes
            != prepared.preparation_claim.reservation.request_blob.size_bytes
            or self.input_file_observation.content_authority_id
            != prepared.preparation_claim.reservation.request_blob.request_blob_id
            or self.input_file_observation.content_digest
            != prepared.preparation_claim.reservation.request_blob.digest
        ):
            raise RunActionSupervisorContractError(
                "activation revalidation differs from prepared authority"
            )
        limits = prepared.preparation_claim.execution_policy.docker_resource_limits
        reobserved_volume = self.reobserved_volume_evidence
        remaining_requirement_bytes = _allocated_size(
            prepared.result_file.payload_size_limit_bytes,
            reobserved_volume.allocation_block_size_bytes,
        ) + _allocated_size(
            limits.runtime_temporary_reservation_size_bytes,
            reobserved_volume.allocation_block_size_bytes,
        )
        if (
            remaining_requirement_bytes >= reobserved_volume.available_size_bytes
            or limits.runtime_temporary_reservation_inode_count
            >= reobserved_volume.available_inode_count
        ):
            raise RunActionSupervisorContractError(
                "activation revalidation lacks result and temporary headroom"
            )
        if credential_mode is RunActionCredentialMode.NONE:
            if self.credential_file_observation is not None:
                raise RunActionSupervisorContractError(
                    "credential-free activation carries a credential file"
                )
        else:
            if (
                self.credential_file_observation is None
                or not _activated_file_matches_prepared(
                    self.credential_file_observation,
                    prepared.credential_file,
                )
                or self.credential_file_observation.content_authority_id is None
            ):
                raise RunActionSupervisorContractError(
                    "credentialed activation lacks an exact delivered file"
                )
        minimum_reobserved_size_bytes = (
            prepared.runtime_volume_evidence.used_size_bytes
            + _allocated_size(
                self.input_file_observation.size_bytes,
                reobserved_volume.allocation_block_size_bytes,
            )
            + (
                0
                if self.credential_file_observation is None
                else _allocated_size(
                    self.credential_file_observation.size_bytes,
                    reobserved_volume.allocation_block_size_bytes,
                )
            )
        )
        if (
            reobserved_volume.used_size_bytes < minimum_reobserved_size_bytes
            or reobserved_volume.used_inode_count
            != prepared.runtime_volume_evidence.used_inode_count
        ):
            raise RunActionSupervisorContractError(
                "activation revalidation statfs usage omits delivered payloads"
            )


def _activated_file_matches_prepared(
    observed: RunActionActivatedFileObservation,
    prepared: RunActionPreparedFile | None,
) -> bool:
    if (
        type(observed) is not RunActionActivatedFileObservation
        or type(prepared) is not RunActionPreparedFile
    ):
        return False
    return (
        observed.prepared_file_id == prepared.prepared_file_id
        and observed.runtime_volume_authority_id == prepared.runtime_volume_authority_id
        and observed.generation_nonce == prepared.generation_nonce
        and observed.kind is prepared.kind
        and observed.file_type == prepared.file_type
        and observed.owner_user_id == prepared.owner_user_id
        and observed.owner_group_id == prepared.owner_group_id
        and observed.mode == prepared.mode
        and observed.link_count == prepared.link_count
        and observed.size_bytes <= prepared.payload_size_limit_bytes
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
) -> bool:
    if (
        type(reobserved) is not RunActionRuntimeVolumeEvidence
        or type(prepared) is not RunActionRuntimeVolumeEvidence
    ):
        return False
    return (
        reobserved.volume_authority == prepared.volume_authority
        and reobserved.observed_volume_name == prepared.observed_volume_name
        and reobserved.observed_labels == prepared.observed_labels
        and reobserved.observed_scope == prepared.observed_scope
        and reobserved.observed_driver == prepared.observed_driver
        and reobserved.observed_driver_options == prepared.observed_driver_options
        and reobserved.observed_filesystem_type == prepared.observed_filesystem_type
        and reobserved.observed_mount_flags == prepared.observed_mount_flags
        and reobserved.observed_owner_user_id == prepared.observed_owner_user_id
        and reobserved.observed_owner_group_id == prepared.observed_owner_group_id
        and reobserved.observed_root_mode == prepared.observed_root_mode
        and reobserved.allocation_block_size_bytes
        == prepared.allocation_block_size_bytes
        and reobserved.effective_block_count == prepared.effective_block_count
        and reobserved.effective_size_bytes == prepared.effective_size_bytes
        and reobserved.effective_inode_limit == prepared.effective_inode_limit
        and reobserved.sentinel_evidence == prepared.sentinel_evidence
        and reobserved.used_size_bytes >= prepared.used_size_bytes
        and reobserved.used_block_count >= prepared.used_block_count
        and reobserved.used_inode_count == prepared.used_inode_count
        and reobserved.available_block_count <= prepared.available_block_count
        and reobserved.available_size_bytes <= prepared.available_size_bytes
        and reobserved.available_inode_count == prepared.available_inode_count
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
) -> tuple[RunActionContainerLabel, ...]:
    """Derive the exact labels owned by the claim's runtime volume."""

    return _preparation_resource_labels(claim, "runtime-volume")


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


def runtime_volume_keeper_helper_authority_id(
    source_path: str,
    executable_digest: str,
) -> str:
    """Bind the keeper helper bytes to their sole read-only mount destination."""

    _require_absolute_host_path(
        source_path,
        "runtime volume keeper helper source",
    )
    if (
        not isinstance(executable_digest, str)
        or _SHA256_DIGEST_PATTERN.fullmatch(executable_digest) is None
    ):
        raise RunActionSupervisorContractError(
            "runtime volume keeper helper digest is invalid"
        )
    return content_id(
        "run-action-helper-executable-authority",
        {
            "destination": RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION,
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
    return tuple(
        sorted(
            (
                "device=tmpfs",
                (
                    "o=nodev,nosuid,noswap,"
                    f"size={authority.size_limit_bytes},"
                    f"nr_inodes={authority.inode_limit},"
                    f"mode={authority.root_mode:04o},"
                    f"uid={authority.owner_user_id},"
                    f"gid={authority.owner_group_id}"
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


__all__ = [
    "DockerRunActionCreateInspectProjection",
    "DockerRunActionExecutionPolicy",
    "DockerRunActionKeeperCreateInspectProjection",
    "DockerRunActionResourceLimits",
    "DockerRunActionSafeCreateDefaults",
    "DockerRunActionSandboxSpec",
    "RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION",
    "RUN_ACTION_RUNTIME_VOLUME_KEEPER_HELPER_DESTINATION",
    "RunActionActivatedFileObservation",
    "RunActionActivatedSentinelObservation",
    "RunActionActivatedWorkspaceObservation",
    "RunActionActivationRevalidationReceipt",
    "RunActionActivationNetworkMode",
    "RunActionContainerLabel",
    "RunActionCredentialMode",
    "RunActionCredentialPolicy",
    "RunActionFilesystemPolicy",
    "RunActionInertContainerEvidence",
    "RunActionKeeperHelperEvidence",
    "RunActionNetworkPolicy",
    "RunActionPreparationClaim",
    "RunActionPreparedExecution",
    "RunActionPreparedFile",
    "RunActionPreparedFileKind",
    "RunActionPreparedMount",
    "RunActionPreparedMountAccess",
    "RunActionPreparedMountKind",
    "RunActionPreparedWorkspaceProof",
    "RunActionRuntimeVolumeAuthority",
    "RunActionRuntimeVolumeEvidence",
    "RunActionRuntimeVolumeLayoutProof",
    "RunActionRuntimeVolumeSentinelEvidence",
    "RunActionStaticEnvironmentVariable",
    "RunActionSupervisorLimits",
    "RunActionSupervisorContractError",
    "RunActionVolumeKeeperEvidence",
    "preparation_container_labels",
    "preparation_container_name",
    "preparation_keeper_container_labels",
    "preparation_keeper_container_name",
    "preparation_main_mounts",
    "preparation_volume_labels",
    "preparation_volume_name",
    "runtime_volume_driver_options",
    "runtime_volume_keeper_helper_authority_id",
    "runtime_volume_sentinel_identity",
]
