"""Closed issued Docker projection for one prepared run-action occurrence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RUN_ACTION_BARRIER_DUMMY_ARGUMENT,
    RUN_ACTION_BARRIER_RELEASE_DESTINATION,
    RUN_ACTION_BARRIER_SCRIPT,
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionPreparationClaim,
    RunActionPreparedMountAccess,
    RunActionRuntimeVolumeAuthority,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_main_mounts,
    preparation_volume_labels,
    preparation_volume_name,
    runtime_volume_driver_options,
)
from kapso.cross_run.settings import DockerRuntimeSettings

DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION = (
    "kapso.docker_run_action_create_inspect.v4"
)

_CONTAINER_ROOT_FIELDS = (
    "AppArmorProfile",
    "Args",
    "Config",
    "Created",
    "Driver",
    "ExecIDs",
    "GraphDriver",
    "HostConfig",
    "HostnamePath",
    "HostsPath",
    "Id",
    "Image",
    "LogPath",
    "MountLabel",
    "Mounts",
    "Name",
    "NetworkSettings",
    "Path",
    "Platform",
    "ProcessLabel",
    "ResolvConfPath",
    "RestartCount",
    "State",
)
_CONTAINER_CONFIG_FIELDS = (
    "AttachStderr",
    "AttachStdin",
    "AttachStdout",
    "Cmd",
    "Domainname",
    "Entrypoint",
    "Env",
    "Hostname",
    "Image",
    "Labels",
    "OpenStdin",
    "StdinOnce",
    "StopSignal",
    "StopTimeout",
    "Tty",
    "User",
    "Volumes",
    "WorkingDir",
)
_HOST_CONFIG_FIELDS = (
    "AutoRemove",
    "Binds",
    "BlkioDeviceReadBps",
    "BlkioDeviceReadIOps",
    "BlkioDeviceWriteBps",
    "BlkioDeviceWriteIOps",
    "BlkioWeight",
    "BlkioWeightDevice",
    "CapAdd",
    "CapDrop",
    "Cgroup",
    "CgroupParent",
    "CgroupnsMode",
    "ConsoleSize",
    "ContainerIDFile",
    "CpuCount",
    "CpuPercent",
    "CpuPeriod",
    "CpuQuota",
    "CpuRealtimePeriod",
    "CpuRealtimeRuntime",
    "CpuShares",
    "CpusetCpus",
    "CpusetMems",
    "DeviceCgroupRules",
    "DeviceRequests",
    "Devices",
    "Dns",
    "DnsOptions",
    "DnsSearch",
    "ExtraHosts",
    "GroupAdd",
    "IOMaximumBandwidth",
    "IOMaximumIOps",
    "Init",
    "IpcMode",
    "Isolation",
    "Links",
    "LogConfig",
    "MaskedPaths",
    "Memory",
    "MemoryReservation",
    "MemorySwap",
    "MemorySwappiness",
    "Mounts",
    "NanoCpus",
    "NetworkMode",
    "OomKillDisable",
    "OomScoreAdj",
    "PidMode",
    "PidsLimit",
    "PortBindings",
    "Privileged",
    "PublishAllPorts",
    "ReadonlyPaths",
    "ReadonlyRootfs",
    "RestartPolicy",
    "Runtime",
    "SecurityOpt",
    "ShmSize",
    "UTSMode",
    "Ulimits",
    "UsernsMode",
    "VolumeDriver",
    "VolumesFrom",
)
_CONTAINER_STATE_FIELDS = (
    "Dead",
    "Error",
    "ExitCode",
    "FinishedAt",
    "OOMKilled",
    "Paused",
    "Pid",
    "Restarting",
    "Running",
    "StartedAt",
    "Status",
)
_GRAPH_DRIVER_FIELDS = ("Data", "Name")
_GRAPH_DRIVER_DATA_FIELDS = ("ID", "LowerDir", "MergedDir", "UpperDir", "WorkDir")
_HOST_CONFIG_LOG_CONFIG_FIELDS = ("Config", "Type")
_HOST_CONFIG_RESTART_POLICY_FIELDS = ("MaximumRetryCount", "Name")
_HOST_CONFIG_READ_ONLY_VOLUME_MOUNT_FIELDS = (
    "ReadOnly",
    "Source",
    "Target",
    "Type",
    "VolumeOptions",
)
_HOST_CONFIG_READ_WRITE_VOLUME_MOUNT_FIELDS = (
    "Source",
    "Target",
    "Type",
    "VolumeOptions",
)
_HOST_CONFIG_ROOT_VOLUME_OPTIONS_FIELDS = ("DriverConfig", "NoCopy")
_HOST_CONFIG_SUBPATH_VOLUME_OPTIONS_FIELDS = (
    "DriverConfig",
    "NoCopy",
    "Subpath",
)
_HOST_CONFIG_BIND_MOUNT_FIELDS = (
    "BindOptions",
    "ReadOnly",
    "Source",
    "Target",
    "Type",
)
_HOST_CONFIG_BIND_OPTIONS_FIELDS = ("NonRecursive", "Propagation")
_NETWORK_SETTINGS_FIELDS = ("Networks", "Ports", "SandboxID", "SandboxKey")
_NONE_NETWORK_FIELDS = (
    "Aliases",
    "DNSNames",
    "DriverOpts",
    "EndpointID",
    "Gateway",
    "GlobalIPv6Address",
    "GlobalIPv6PrefixLen",
    "GwPriority",
    "IPAMConfig",
    "IPAddress",
    "IPPrefixLen",
    "IPv6Gateway",
    "Links",
    "MacAddress",
    "NetworkID",
)
_VOLUME_INSPECT_FIELDS = (
    "CreatedAt",
    "Driver",
    "Labels",
    "Mountpoint",
    "Name",
    "Options",
    "Scope",
)
_VOLUME_INSPECT_OPTIONS_FIELDS = ("device", "o", "type")
_TOP_LEVEL_VOLUME_MOUNT_FIELDS = (
    "Destination",
    "Driver",
    "Mode",
    "Name",
    "Propagation",
    "RW",
    "Source",
    "Type",
)
_TOP_LEVEL_BIND_MOUNT_FIELDS = (
    "Destination",
    "Mode",
    "Propagation",
    "RW",
    "Source",
    "Type",
)
_IMAGE_ROOT_REQUIRED_FIELDS = (
    "Architecture",
    "Config",
    "GraphDriver",
    "Id",
    "Metadata",
    "Os",
    "RepoDigests",
    "RepoTags",
    "RootFS",
    "Size",
)
_IMAGE_ROOT_OPTIONAL_FIELDS = ("Comment", "Created", "Variant")
_IMAGE_METADATA_FIELDS = ("LastTagTime",)
_IMAGE_ROOT_FILESYSTEM_FIELDS = ("Layers", "Type")
_IMAGE_CONFIG_FIELDS = (
    "ArgsEscaped",
    "AttachStderr",
    "AttachStdin",
    "AttachStdout",
    "Cmd",
    "Domainname",
    "Entrypoint",
    "Env",
    "ExposedPorts",
    "Healthcheck",
    "Hostname",
    "Image",
    "Labels",
    "MacAddress",
    "NetworkDisabled",
    "OnBuild",
    "OpenStdin",
    "Shell",
    "StdinOnce",
    "StopSignal",
    "Tty",
    "User",
    "Volumes",
    "WorkingDir",
)
_RAW_FIELD_SCHEMA = {
    "container_config": _CONTAINER_CONFIG_FIELDS,
    "container_root": _CONTAINER_ROOT_FIELDS,
    "container_state": _CONTAINER_STATE_FIELDS,
    "graph_driver": _GRAPH_DRIVER_FIELDS,
    "graph_driver_data": _GRAPH_DRIVER_DATA_FIELDS,
    "host_config": _HOST_CONFIG_FIELDS,
    "host_config_bind_mount": _HOST_CONFIG_BIND_MOUNT_FIELDS,
    "host_config_bind_options": _HOST_CONFIG_BIND_OPTIONS_FIELDS,
    "host_config_log_config": _HOST_CONFIG_LOG_CONFIG_FIELDS,
    "host_config_restart_policy": _HOST_CONFIG_RESTART_POLICY_FIELDS,
    "host_config_read_only_volume_mount": (_HOST_CONFIG_READ_ONLY_VOLUME_MOUNT_FIELDS),
    "host_config_read_write_volume_mount": (
        _HOST_CONFIG_READ_WRITE_VOLUME_MOUNT_FIELDS
    ),
    "host_config_root_volume_options": (_HOST_CONFIG_ROOT_VOLUME_OPTIONS_FIELDS),
    "host_config_subpath_volume_options": (_HOST_CONFIG_SUBPATH_VOLUME_OPTIONS_FIELDS),
    "image_config": _IMAGE_CONFIG_FIELDS,
    "image_metadata": _IMAGE_METADATA_FIELDS,
    "image_root_optional": _IMAGE_ROOT_OPTIONAL_FIELDS,
    "image_root_required": _IMAGE_ROOT_REQUIRED_FIELDS,
    "image_root_filesystem": _IMAGE_ROOT_FILESYSTEM_FIELDS,
    "network_none": _NONE_NETWORK_FIELDS,
    "network_settings": _NETWORK_SETTINGS_FIELDS,
    "top_level_bind_mount": _TOP_LEVEL_BIND_MOUNT_FIELDS,
    "top_level_volume_mount": _TOP_LEVEL_VOLUME_MOUNT_FIELDS,
    "volume_inspect": _VOLUME_INSPECT_FIELDS,
    "volume_inspect_options": _VOLUME_INSPECT_OPTIONS_FIELDS,
}
DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID = content_id(
    "docker-raw-field-schema",
    {
        "fields": _RAW_FIELD_SCHEMA,
        "projection_protocol_version": (DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION),
    },
)
_READ_ONLY_RAW_FIELD_SCHEMA = MappingProxyType(_RAW_FIELD_SCHEMA)

_SECRET_ENVIRONMENT_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:ACCESS_KEY(?:_ID)?|ACCESS_TOKEN|API_KEY|AUTH_CONFIG|AUTH_TOKEN|"
    r"CREDENTIALS?|NETRC|OAUTH_TOKEN|PASSWORD|PASSWD|PAT|PRIVATE_KEY|"
    r"SECRET(?:_ACCESS_KEY)?|SECRETS?|TOKEN)(?:_|$)"
)
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_COMMAND_TEMPLATE_ID_PATTERN = re.compile(
    r"^docker-run-action-command-template:sha256:[0-9a-f]{64}$"
)
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T" r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)
_HOST_TO_IMAGE_ARCHITECTURE = {"x86_64": "amd64"}
_KEEPER_COMMAND = ("tail", "-f", "/dev/null")


class DockerRunActionProjectionError(ValueError):
    """An issued Docker create request exceeds its closed action authority."""


@dataclass(frozen=True)
class DockerRunActionCommand:
    """Ephemeral fixed command rendered by one lifecycle-owned implementation."""

    command_template_id: str
    entrypoint: str
    arguments: tuple[str, ...]

    @classmethod
    def build(
        cls,
        *,
        entrypoint: str,
        arguments: tuple[str, ...],
    ) -> DockerRunActionCommand:
        """Build one command whose identity binds every persisted argument."""

        return cls(
            command_template_id=docker_run_action_command_template_id(
                entrypoint,
                arguments,
            ),
            entrypoint=entrypoint,
            arguments=arguments,
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.command_template_id, str)
            or _COMMAND_TEMPLATE_ID_PATTERN.fullmatch(self.command_template_id) is None
            or self.command_template_id
            != docker_run_action_command_template_id(
                self.entrypoint,
                self.arguments,
            )
        ):
            raise DockerRunActionProjectionError(
                "run action command template identity is invalid"
            )


def docker_run_action_command_template_id(
    entrypoint: str,
    arguments: tuple[str, ...],
) -> str:
    """Bind an immutable entrypoint and complete argument vector."""

    _require_fixed_command(entrypoint, arguments)
    return content_id(
        "docker-run-action-command-template",
        {
            "arguments": arguments,
            "entrypoint": entrypoint,
        },
    )


def main_barrier_command(
    command: DockerRunActionCommand,
    settings: DockerRuntimeSettings,
) -> tuple[str, tuple[str, ...]]:
    """Render the fixed barrier while keeping the target as positional data."""

    if (
        type(command) is not DockerRunActionCommand
        or type(settings) is not DockerRuntimeSettings
    ):
        raise DockerRunActionProjectionError(
            "run action barrier requires exact target and Docker settings"
        )
    return (
        RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        (
            "sh",
            "-eu",
            "-c",
            RUN_ACTION_BARRIER_SCRIPT,
            RUN_ACTION_BARRIER_DUMMY_ARGUMENT,
            RUN_ACTION_BARRIER_RELEASE_DESTINATION,
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            str(settings.run_action_barrier_poll_interval_seconds),
            command.entrypoint,
            *command.arguments,
        ),
    )


def docker_run_action_raw_field_schema() -> Mapping[str, tuple[str, ...]]:
    """Return the immutable schema whose identity is embedded in every policy."""

    return _READ_ONLY_RAW_FIELD_SCHEMA


def require_run_action_image(
    image: Mapping[str, Any],
    policy: DockerRunActionExecutionPolicy,
    settings: DockerRuntimeSettings,
) -> None:
    """Admit one exact image whose inherited runtime state is fully constrained."""

    _require_projection_policy(policy, settings)
    if not isinstance(image, Mapping):
        raise DockerRunActionProjectionError(
            "run action image inspection is not an object"
        )
    image_keys = set(image)
    required_keys = set(_IMAGE_ROOT_REQUIRED_FIELDS)
    optional_keys = set(_IMAGE_ROOT_OPTIONAL_FIELDS)
    if (
        not required_keys.issubset(image_keys)
        or image_keys - required_keys - optional_keys
    ):
        raise DockerRunActionProjectionError(
            "run action image inspection has an unknown or missing field"
        )
    authority = policy.image_authority
    repo_digests = image["RepoDigests"]
    repo_tags = image["RepoTags"]
    variant = image.get("Variant")
    if variant is not None and not isinstance(variant, str):
        raise DockerRunActionProjectionError("run action image variant is malformed")
    normalized_variant = None if variant in {None, ""} else variant
    expected_architecture = _HOST_TO_IMAGE_ARCHITECTURE.get(
        settings.runtime_host_architecture
    )
    if (
        image["Id"] != authority.image_config_digest
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) or not item for item in repo_digests)
        or len(repo_digests) != len(set(repo_digests))
        or authority.image_reference not in repo_digests
        or (
            repo_tags is not None
            and (
                not isinstance(repo_tags, list)
                or any(not isinstance(item, str) or not item for item in repo_tags)
                or len(repo_tags) != len(set(repo_tags))
            )
        )
        or type(image["Size"]) is not int
        or image["Size"] < 0
        or ("Comment" in image and not isinstance(image["Comment"], str))
        or (
            "Created" in image
            and (
                not isinstance(image["Created"], str)
                or _UTC_TIMESTAMP_PATTERN.fullmatch(image["Created"]) is None
            )
        )
        or image["Os"] != authority.operating_system
        or image["Architecture"] != authority.architecture
        or normalized_variant != authority.architecture_variant
        or authority.operating_system != settings.runtime_host_operating_system
        or expected_architecture is None
        or authority.architecture != expected_architecture
    ):
        raise DockerRunActionProjectionError(
            "run action image differs from its host and content authority"
        )
    _require_image_metadata(image, settings)
    config = _require_mapping(image["Config"], "run action image Config")
    if set(config) - set(_IMAGE_CONFIG_FIELDS):
        raise DockerRunActionProjectionError(
            "run action image Config has an unknown field"
        )
    for field_name in ("Volumes", "ExposedPorts", "Labels"):
        value = config.get(field_name)
        if value is not None and (not isinstance(value, Mapping) or value):
            raise DockerRunActionProjectionError(
                f"run action image inherits forbidden {field_name}"
            )
    if config.get("Healthcheck") is not None:
        raise DockerRunActionProjectionError("run action image inherits a healthcheck")
    _require_image_environment(config.get("Env"), policy)
    _require_overridden_image_fields(config)


def volume_create_arguments(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    settings: DockerRuntimeSettings,
) -> tuple[str, ...]:
    """Render the sole bounded local-driver tmpfs volume creation request."""

    _require_claim_authority(claim, authority)
    _require_projection_policy(claim.execution_policy, settings)
    arguments = [
        "volume",
        "create",
        "--driver",
        authority.driver,
    ]
    _append_labels(
        arguments,
        preparation_volume_labels(claim, authority.generation_nonce),
    )
    for option in runtime_volume_driver_options(authority):
        arguments.extend(("--opt", option))
    arguments.append(preparation_volume_name(claim))
    return tuple(arguments)


def keeper_create_arguments(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    image: Mapping[str, Any],
    settings: DockerRuntimeSettings,
) -> tuple[str, ...]:
    """Render the exact running keeper creation request without initializing data."""

    _require_claim_authority(claim, authority)
    policy = claim.execution_policy
    require_run_action_image(image, policy, settings)
    arguments = [
        "container",
        "create",
        "--name",
        preparation_keeper_container_name(claim),
    ]
    _append_labels(arguments, preparation_keeper_container_labels(claim))
    arguments.extend(
        _common_container_arguments(
            policy,
            working_directory=RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
    )
    arguments.extend(
        (
            "--mount",
            (
                "type=bind,"
                f"source={policy.supervisor_helper_source_path},"
                f"target={RUN_ACTION_SUPERVISOR_HELPER_DESTINATION},"
                "readonly,bind-recursive=disabled,bind-propagation=rprivate"
            ),
            "--mount",
            (
                "type=volume,"
                f"source={authority.volume_name},"
                f"target={RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION},"
                "volume-nocopy"
            ),
            "--entrypoint",
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            policy.image_authority.image_reference,
            *_KEEPER_COMMAND,
        )
    )
    return tuple(arguments)


def main_create_arguments(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    command: DockerRunActionCommand,
    image: Mapping[str, Any],
    settings: DockerRuntimeSettings,
) -> tuple[str, ...]:
    """Render the exact inert main-container request from durable authority."""

    _require_claim_authority(claim, authority)
    if type(command) is not DockerRunActionCommand:
        raise DockerRunActionProjectionError(
            "run action main creation requires one exact fixed command"
        )
    policy = claim.execution_policy
    if command.command_template_id != policy.command_template_id:
        raise DockerRunActionProjectionError(
            "run action command differs from its execution policy template"
        )
    require_run_action_image(image, policy, settings)
    barrier_executable, barrier_arguments = main_barrier_command(command, settings)
    arguments = [
        "container",
        "create",
        "--name",
        preparation_container_name(claim),
    ]
    _append_labels(arguments, preparation_container_labels(claim))
    arguments.extend(
        _common_container_arguments(
            policy,
            working_directory=policy.filesystem_policy.working_directory,
        )
    )
    arguments.extend(
        (
            "--mount",
            (
                "type=bind,"
                f"source={policy.supervisor_helper_source_path},"
                f"target={RUN_ACTION_SUPERVISOR_HELPER_DESTINATION},"
                "readonly,bind-recursive=disabled,bind-propagation=rprivate"
            ),
        )
    )
    for mount in preparation_main_mounts(claim, authority):
        mount_parts = [
            "type=volume",
            f"source={mount.volume_name}",
            f"target={mount.container_destination}",
        ]
        if mount.container_access is RunActionPreparedMountAccess.READ_ONLY:
            mount_parts.append("readonly")
        mount_parts.extend(
            (
                "volume-nocopy",
                f"volume-subpath={mount.volume_subpath}",
            )
        )
        arguments.extend(("--mount", ",".join(mount_parts)))
    arguments.extend(
        (
            "--entrypoint",
            barrier_executable,
            policy.image_authority.image_reference,
            *barrier_arguments,
        )
    )
    return tuple(arguments)


def _require_projection_policy(
    policy: DockerRunActionExecutionPolicy,
    settings: DockerRuntimeSettings,
) -> None:
    if (
        type(policy) is not DockerRunActionExecutionPolicy
        or type(settings) is not DockerRuntimeSettings
        or policy.projection_protocol_version
        != DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION
        or policy.raw_field_schema_id != DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID
        or policy.docker_runtime_settings_digest
        != tree_or_blob_digest(settings.to_json_bytes())
        or policy.supervisor_helper_source_path != settings.helper_executable_path
        or policy.supervisor_helper_executable_digest
        != settings.helper_executable_digest
        or policy.docker_init_source_path != settings.init_executable_path
        or policy.docker_init_executable_digest != settings.init_executable_digest
    ):
        raise DockerRunActionProjectionError(
            "run action execution policy differs from closed Docker authority"
        )


def _require_claim_authority(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    if (
        type(claim) is not RunActionPreparationClaim
        or type(authority) is not RunActionRuntimeVolumeAuthority
        or authority.preparation_claim_id != claim.preparation_claim_id
        or authority.volume_name != preparation_volume_name(claim)
        or authority.labels
        != preparation_volume_labels(claim, authority.generation_nonce)
        or authority.driver != "local"
        or authority.driver_options != runtime_volume_driver_options(authority)
        or authority.owner_user_id != claim.execution_policy.user_id
        or authority.owner_group_id != claim.execution_policy.group_id
        or authority.root_mode != 0o700
        or authority.size_limit_bytes
        != claim.execution_policy.docker_resource_limits.runtime_volume_size_bytes
        or authority.inode_limit
        != claim.execution_policy.docker_resource_limits.runtime_volume_inode_limit
        or authority.nosuid is not True
        or authority.nodev is not True
        or authority.noswap is not True
        or authority.execution_allowed is not True
    ):
        raise DockerRunActionProjectionError(
            "run action volume authority differs from its preparation claim "
            "or execution policy"
        )


def _common_container_arguments(
    policy: DockerRunActionExecutionPolicy,
    *,
    working_directory: str,
) -> tuple[str, ...]:
    limits = policy.docker_resource_limits
    sandbox = policy.sandbox_spec
    arguments = [
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
    ]
    for security_option in sandbox.security_option_ids:
        arguments.extend(("--security-opt", security_option))
    arguments.extend(
        (
            "--cgroupns",
            sandbox.cgroup_namespace_mode,
            "--ipc",
            sandbox.ipc_namespace_mode,
            "--cgroup-parent",
            sandbox.cgroup_parent_id,
            "--runtime",
            sandbox.runtime_id,
            "--log-driver",
            sandbox.log_driver,
            "--init",
            "--restart",
            "no",
            "--hostname",
            policy.hostname,
            "--user",
            f"{policy.user_id}:{policy.group_id}",
            "--workdir",
            working_directory,
            "--stop-signal",
            "SIGTERM",
            "--stop-timeout",
            str(policy.supervisor_limits.termination_grace_seconds),
            "--cpu-period",
            str(limits.cpu_period_microseconds),
            "--cpu-quota",
            str(limits.cpu_quota_microseconds),
            "--cpu-shares",
            str(limits.cpu_shares),
        )
    )
    if limits.cpuset_cpu_ids:
        arguments.extend(
            ("--cpuset-cpus", ",".join(str(value) for value in limits.cpuset_cpu_ids))
        )
    if limits.cpuset_memory_node_ids:
        arguments.extend(
            (
                "--cpuset-mems",
                ",".join(str(value) for value in limits.cpuset_memory_node_ids),
            )
        )
    arguments.extend(
        (
            "--memory",
            str(limits.memory_size_bytes),
            "--memory-reservation",
            str(limits.memory_reservation_size_bytes),
            "--memory-swap",
            str(limits.memory_swap_size_bytes),
            "--oom-score-adj",
            str(limits.oom_score_adjustment),
            "--pids-limit",
            str(limits.process_limit),
            "--blkio-weight",
            str(limits.block_io_weight),
            "--shm-size",
            str(limits.shared_memory_size_bytes),
        )
    )
    for variable in policy.static_environment:
        arguments.extend(("--env", f"{variable.key}={variable.value}"))
    return tuple(arguments)


def _append_labels(arguments: list[str], labels: tuple) -> None:
    for label in labels:
        arguments.extend(("--label", f"{label.key}={label.value}"))


def _require_image_metadata(
    image: Mapping[str, Any],
    settings: DockerRuntimeSettings,
) -> None:
    graph_driver = _require_mapping(
        image["GraphDriver"], "run action image GraphDriver"
    )
    graph_data = graph_driver["Data"]
    if (
        set(graph_driver) != set(_GRAPH_DRIVER_FIELDS)
        or graph_driver["Name"] != settings.runtime_storage_driver
        or not isinstance(graph_data, Mapping)
        or not {"MergedDir", "UpperDir", "WorkDir"}.issubset(graph_data)
        or set(graph_data) - set(_GRAPH_DRIVER_DATA_FIELDS)
    ):
        raise DockerRunActionProjectionError(
            "run action image GraphDriver has an unknown or missing field"
        )
    docker_root = PurePosixPath(settings.runtime_root_directory)
    path_values = tuple(
        path_value
        for field_name, encoded_paths in graph_data.items()
        for path_value in (
            encoded_paths.split(":")
            if field_name == "LowerDir" and isinstance(encoded_paths, str)
            else (encoded_paths,)
        )
    )
    for path_value in path_values:
        path = (
            PurePosixPath(path_value)
            if isinstance(path_value, str)
            else PurePosixPath(".")
        )
        if (
            not isinstance(path_value, str)
            or not path_value
            or "\x00" in path_value
            or not path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != path_value
            or docker_root not in path.parents
        ):
            raise DockerRunActionProjectionError(
                "run action image GraphDriver path escapes Docker authority"
            )
    metadata = _require_mapping(image["Metadata"], "run action image Metadata")
    root_filesystem = _require_mapping(image["RootFS"], "run action image RootFS")
    layers = root_filesystem.get("Layers")
    if (
        set(metadata) != {"LastTagTime"}
        or not isinstance(metadata["LastTagTime"], str)
        or set(root_filesystem) != {"Layers", "Type"}
        or root_filesystem["Type"] != "layers"
        or not isinstance(layers, list)
        or not layers
        or any(
            not isinstance(layer, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", layer) is None
            for layer in layers
        )
    ):
        raise DockerRunActionProjectionError(
            "run action image metadata has an unknown or missing field"
        )


def _require_image_environment(
    value: Any,
    policy: DockerRunActionExecutionPolicy,
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise DockerRunActionProjectionError(
            "run action image environment is not a list"
        )
    expected = {variable.key: variable.value for variable in policy.static_environment}
    observed: dict[str, str] = {}
    for assignment in value:
        if (
            not isinstance(assignment, str)
            or "\x00" in assignment
            or "=" not in assignment
        ):
            raise DockerRunActionProjectionError(
                "run action image environment contains a malformed assignment"
            )
        key, assignment_value = assignment.split("=", 1)
        if (
            _ENVIRONMENT_KEY_PATTERN.fullmatch(key) is None
            or _SECRET_ENVIRONMENT_KEY_PATTERN.search(key) is not None
            or key in observed
            or expected.get(key) != assignment_value
        ):
            raise DockerRunActionProjectionError(
                "run action image environment exceeds its exact policy"
            )
        observed[key] = assignment_value


def _require_overridden_image_fields(config: Mapping[str, Any]) -> None:
    for field_name in ("Cmd", "Entrypoint", "OnBuild", "Shell"):
        value = config.get(field_name)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) or "\x00" in item for item in value)
        ):
            raise DockerRunActionProjectionError(
                f"run action image {field_name} is malformed"
            )
    for field_name in (
        "Domainname",
        "Hostname",
        "Image",
        "MacAddress",
        "StopSignal",
        "User",
        "WorkingDir",
    ):
        value = config.get(field_name)
        if value is not None and (not isinstance(value, str) or "\x00" in value):
            raise DockerRunActionProjectionError(
                f"run action image {field_name} is malformed"
            )
    for field_name in (
        "ArgsEscaped",
        "AttachStderr",
        "AttachStdin",
        "AttachStdout",
        "NetworkDisabled",
        "OpenStdin",
        "StdinOnce",
        "Tty",
    ):
        value = config.get(field_name)
        if value is not None and type(value) is not bool:
            raise DockerRunActionProjectionError(
                f"run action image {field_name} is malformed"
            )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DockerRunActionProjectionError(f"{name} is not an object")
    return value


def _require_fixed_command(
    entrypoint: str,
    arguments: tuple[str, ...],
) -> None:
    _require_container_path(entrypoint, "run action command entrypoint")
    if (
        not isinstance(arguments, tuple)
        or not arguments
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in arguments
        )
    ):
        raise DockerRunActionProjectionError("run action command arguments are invalid")


def _require_container_path(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DockerRunActionProjectionError(f"{name} is not an absolute path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise DockerRunActionProjectionError(f"{name} is not an absolute path")


__all__ = [
    "DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION",
    "DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID",
    "DockerRunActionCommand",
    "DockerRunActionProjectionError",
    "docker_run_action_command_template_id",
    "docker_run_action_raw_field_schema",
    "keeper_create_arguments",
    "main_barrier_command",
    "main_create_arguments",
    "require_run_action_image",
    "volume_create_arguments",
]
