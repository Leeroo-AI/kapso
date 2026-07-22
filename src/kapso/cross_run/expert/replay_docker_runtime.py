"""Pinned Docker CLI authority for isolated expert source replay."""

from __future__ import annotations

import json
import os
import stat
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import TaskAdapterRuntimeContract
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
    BoundedProcessRunner,
)
from kapso.cross_run.settings import SourceReplayDockerProviderSettings

_AUTHORITY_DIRECTORY_NAME = "authority"
_DOCKER_CONFIG_DIRECTORY_PREFIX = "docker-config-"
_DOCKER_CONFIG_FILENAME = "config.json"
_PINNED_DOCKER_FILENAME_PREFIX = "docker-"
_EMPTY_DOCKER_CONFIG = b'{"auths":{}}\n'
_DOCKER_HOST_PREFIX = "unix://"


class SourceReplayDockerRuntimeError(RuntimeError):
    """The Docker client, daemon, image, or command violates pinned authority."""


class SourceReplayDockerProcessRunner(Protocol):
    """The bounded host-process primitive used by the Docker runtime."""

    def run(self, request: BoundedProcessRequest) -> BoundedProcessResult: ...


class SourceReplayDockerRuntime:
    """Execute Docker only through privately pinned bytes and an exact daemon."""

    def __init__(
        self,
        *,
        trusted_root: Path,
        settings: SourceReplayDockerProviderSettings,
        process_runner: SourceReplayDockerProcessRunner,
    ) -> None:
        if not isinstance(settings, SourceReplayDockerProviderSettings):
            raise SourceReplayDockerRuntimeError(
                "source replay Docker runtime requires exact provider settings"
            )
        if not callable(getattr(process_runner, "run", None)):
            raise SourceReplayDockerRuntimeError(
                "source replay Docker runtime requires a bounded process runner"
            )
        _require_private_root(trusted_root)
        settings_digest_suffix = tree_or_blob_digest(
            settings.to_json_bytes()
        ).removeprefix("sha256:")
        executable_digest_suffix = settings.runtime_executable_digest.removeprefix(
            "sha256:"
        )
        authority_root = trusted_root / _AUTHORITY_DIRECTORY_NAME
        docker_config_root = trusted_root / (
            f"{_DOCKER_CONFIG_DIRECTORY_PREFIX}{settings_digest_suffix}"
        )
        _ensure_private_directory(authority_root, trusted_root)
        _ensure_private_directory(docker_config_root, trusted_root)
        docker_path = authority_root / (
            f"{_PINNED_DOCKER_FILENAME_PREFIX}{executable_digest_suffix}"
        )
        docker_bytes = read_verified_root_executable(
            Path(settings.runtime_executable_path),
            settings.runtime_executable_digest,
        )
        _publish_or_verify_private_executable(docker_path, docker_bytes)
        _publish_or_verify_private_file(
            docker_config_root / _DOCKER_CONFIG_FILENAME,
            _EMPTY_DOCKER_CONFIG,
        )
        self._trusted_root = trusted_root
        self._settings = settings
        self._process_runner = process_runner
        self._docker_path = docker_path
        self._docker_digest = settings.runtime_executable_digest
        self._docker_config_root = docker_config_root
        self._environment = MappingProxyType(
            {
                "DOCKER_API_VERSION": settings.runtime_api_version,
                "DOCKER_CONFIG": str(docker_config_root),
                "HOME": str(trusted_root),
                "LANG": "C",
                "LC_ALL": "C",
            }
        )
        self.require_live_authority()

    @classmethod
    def create(
        cls,
        *,
        trusted_root: Path,
        settings: SourceReplayDockerProviderSettings,
    ) -> SourceReplayDockerRuntime:
        return cls(
            trusted_root=trusted_root,
            settings=settings,
            process_runner=BoundedProcessRunner(),
        )

    @property
    def trusted_root(self) -> Path:
        return self._trusted_root

    @property
    def settings(self) -> SourceReplayDockerProviderSettings:
        return self._settings

    def require_live_authority(self) -> None:
        self._require_local_authority()
        version = self.run_json_control(("version", "--format", "{{json .}}"))
        info = self.run_json_control(("info", "--format", "{{json .}}"))
        _require_daemon_authority(version, info, self._settings)

    def require_exact_image(self, runtime: TaskAdapterRuntimeContract) -> None:
        if not isinstance(runtime, TaskAdapterRuntimeContract):
            raise SourceReplayDockerRuntimeError(
                "source replay Docker image requires an exact runtime contract"
            )
        self.require_live_authority()
        image = self.run_json_control(
            (
                "image",
                "inspect",
                "--format",
                "{{json .}}",
                runtime.image_reference,
            )
        )
        _require_image_authority(image, runtime)

    def run_control(self, arguments: tuple[str, ...]) -> BoundedProcessResult:
        result = self.run_bounded(
            arguments,
            timeout_seconds=self._settings.command_timeout_seconds,
            cleanup_timeout_seconds=self._settings.cleanup_timeout_seconds,
            stdout_byte_limit=self._settings.command_output_byte_limit,
            stderr_byte_limit=self._settings.command_output_byte_limit,
        )
        if (
            result.outcome is not BoundedProcessOutcome.COMPLETED
            or result.returncode != 0
            or result.stderr
        ):
            raise SourceReplayDockerRuntimeError(
                "source replay Docker control command failed or emitted stderr"
            )
        return result

    def run_json_control(self, arguments: tuple[str, ...]) -> Mapping[str, Any]:
        return _parse_single_json_object(self.run_control(arguments).stdout)

    def run_bounded(
        self,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: int,
        cleanup_timeout_seconds: int,
        stdout_byte_limit: int,
        stderr_byte_limit: int,
    ) -> BoundedProcessResult:
        if (
            not isinstance(arguments, tuple)
            or not arguments
            or any(
                not isinstance(argument, str) or not argument or "\x00" in argument
                for argument in arguments
            )
        ):
            raise SourceReplayDockerRuntimeError(
                "source replay Docker arguments must be non-empty strings"
            )
        self._require_local_authority()
        request = BoundedProcessRequest(
            argv=(
                str(self._docker_path),
                "--host",
                f"{_DOCKER_HOST_PREFIX}{self._settings.runtime_socket_path}",
                "--config",
                str(self._docker_config_root),
                *arguments,
            ),
            trusted_root=self._trusted_root,
            cwd=self._trusted_root,
            timeout_seconds=timeout_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            stdout_byte_limit=stdout_byte_limit,
            stderr_byte_limit=stderr_byte_limit,
            environment=self._environment,
        )
        result = self._process_runner.run(request)
        if type(result) is not BoundedProcessResult or result.request != request:
            raise SourceReplayDockerRuntimeError(
                "source replay Docker runner changed its exact request"
            )
        return result

    def _require_local_authority(self) -> None:
        if read_verified_private_executable(self._docker_path) != self._docker_digest:
            raise SourceReplayDockerRuntimeError(
                "source replay pinned Docker executable changed"
            )
        _require_runtime_socket(Path(self._settings.runtime_socket_path))


def read_verified_root_executable(path: Path, expected_digest: str) -> bytes:
    """Read one immutable-enough root-owned executable through a no-follow fd."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay authority executable path must be absolute and normalized"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata_before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata_before.st_mode)
            or metadata_before.st_nlink != 1
            or metadata_before.st_uid != 0
            or stat.S_IMODE(metadata_before.st_mode) & 0o022
            or not metadata_before.st_mode & stat.S_IXUSR
        ):
            raise SourceReplayDockerRuntimeError(
                "source replay authority executable is not immutable root-owned code"
            )
        payload = handle.read()
        metadata_after = os.fstat(handle.fileno())
    if (
        (metadata_before.st_dev, metadata_before.st_ino, metadata_before.st_size)
        != (metadata_after.st_dev, metadata_after.st_ino, metadata_after.st_size)
        or len(payload) != metadata_before.st_size
        or tree_or_blob_digest(payload) != expected_digest
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay authority executable differs from its pinned digest"
        )
    return payload


def read_verified_private_executable(path: Path) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata_before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata_before.st_mode)
            or metadata_before.st_nlink != 1
            or metadata_before.st_uid != os.geteuid()
            or stat.S_IMODE(metadata_before.st_mode) != 0o500
        ):
            raise SourceReplayDockerRuntimeError(
                "source replay private executable is unsafe"
            )
        payload = handle.read()
        metadata_after = os.fstat(handle.fileno())
    if (metadata_before.st_dev, metadata_before.st_ino, metadata_before.st_size) != (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
    ) or len(payload) != metadata_before.st_size:
        raise SourceReplayDockerRuntimeError(
            "source replay private executable changed while reading"
        )
    return tree_or_blob_digest(payload)


def _require_daemon_authority(
    version: Mapping[str, Any],
    info: Mapping[str, Any],
    settings: SourceReplayDockerProviderSettings,
) -> None:
    client = _require_mapping(version, "Client", "Docker version client")
    server = _require_mapping(version, "Server", "Docker version server")
    plugins = _require_mapping(info, "Plugins", "Docker daemon plugins")
    runtimes = _require_mapping(info, "Runtimes", "Docker daemon runtimes")
    required_boolean_capabilities = (
        "MemoryLimit",
        "SwapLimit",
        "CpuCfsPeriod",
        "CpuCfsQuota",
        "PidsLimit",
    )
    if (
        client.get("ApiVersion") != settings.runtime_api_version
        or server.get("Version") != settings.runtime_server_version
        or server.get("ApiVersion") != settings.runtime_api_version
        or server.get("Os") != settings.runtime_host_operating_system
        or info.get("ServerVersion") != settings.runtime_server_version
        or info.get("OSType") != settings.runtime_host_operating_system
        or info.get("Architecture") != settings.runtime_host_architecture
        or info.get("Driver") != settings.runtime_storage_driver
        or str(info.get("CgroupVersion")) != settings.runtime_cgroup_version
        or info.get("DefaultRuntime") != settings.runtime_default_runtime
        or any(info.get(name) is not True for name in required_boolean_capabilities)
        or set(settings.required_security_options)
        != set(
            _require_string_set(info.get("SecurityOptions"), "Docker security options")
        )
        or settings.runtime_default_runtime not in runtimes
        or "local"
        not in _require_string_set(plugins.get("Volume"), "Docker volume plugins")
        or "null"
        not in _require_string_set(plugins.get("Network"), "Docker network plugins")
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker daemon differs from its exact authority"
        )


def _require_image_authority(
    image: Mapping[str, Any],
    runtime: TaskAdapterRuntimeContract,
) -> None:
    config = _require_mapping(image, "Config", "Docker image config")
    repo_digests = _require_string_set(image.get("RepoDigests"), "Docker image digests")
    environment = _optional_string_tuple(config.get("Env"), "Docker image environment")
    command = _optional_string_tuple(config.get("Cmd"), "Docker image command")
    volumes = config.get("Volumes")
    variant = image.get("Variant")
    normalized_variant = None if variant in {None, ""} else variant
    expected_environment = tuple(
        f"{key}={value}" for key, value in runtime.environment.items()
    )
    if (
        image.get("Id") != runtime.image_config_digest
        or runtime.image_reference not in repo_digests
        or image.get("Os") != runtime.operating_system
        or image.get("Architecture") != runtime.architecture
        or normalized_variant != runtime.architecture_variant
        or environment != expected_environment
        or command
        or (volumes is not None and volumes != {})
        or config.get("Healthcheck") is not None
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker image differs from its exact runtime contract"
        )


def _parse_single_json_object(payload: bytes) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker JSON output lacks its exact line ending"
        )
    encoded = payload[:-1]
    if not encoded or b"\n" in encoded or b"\r" in encoded:
        raise SourceReplayDockerRuntimeError(
            "source replay Docker JSON output is not one document"
        )
    decoded = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(decoded, dict):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker JSON output is not an object"
        )
    return MappingProxyType(decoded)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise SourceReplayDockerRuntimeError(
                "source replay Docker JSON output contains a duplicate key"
            )
        decoded[key] = value
    return decoded


def _reject_nonstandard_json_constant(value: str) -> None:
    raise SourceReplayDockerRuntimeError(
        f"source replay Docker JSON output contains nonstandard constant {value}"
    )


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
    name: str,
) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise SourceReplayDockerRuntimeError(f"{name} is not an object")
    return child


def _require_string_set(value: Any, name: str) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise SourceReplayDockerRuntimeError(f"{name} is not a string list")
    return frozenset(value)


def _optional_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SourceReplayDockerRuntimeError(f"{name} is not a string list")
    return tuple(value)


def _require_runtime_socket(path: Path) -> None:
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker socket path is not absolute and direct"
        )
    metadata = path.lstat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o002
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker authority is not a root-owned Unix socket"
        )


def _require_private_root(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker trusted root must be absolute and resolved"
        )
    metadata = path.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker trusted root must be owner-private"
        )


def _ensure_private_directory(path: Path, parent: Path) -> None:
    if path.parent != parent:
        raise SourceReplayDockerRuntimeError(
            "source replay Docker private directory is outside its trusted parent"
        )
    if not path.exists():
        os.mkdir(path, mode=0o700)
        _fsync_directory(parent)
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay Docker private directory is unsafe"
        )


def _publish_or_verify_private_executable(path: Path, payload: bytes) -> None:
    if not path.exists():
        _write_private_file(path, payload, 0o500)
    if read_verified_private_executable(path) != tree_or_blob_digest(payload):
        raise SourceReplayDockerRuntimeError(
            "source replay pinned Docker executable conflicts with authority"
        )


def _publish_or_verify_private_file(path: Path, payload: bytes) -> None:
    if not path.exists():
        _write_private_file(path, payload, 0o400)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        observed = handle.read()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or observed != payload
    ):
        raise SourceReplayDockerRuntimeError(
            "source replay private Docker configuration is unsafe"
        )


def _write_private_file(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), mode)
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
        ):
            raise SourceReplayDockerRuntimeError(
                "source replay private Docker file is unsafe"
            )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        os.fsync(descriptor)
