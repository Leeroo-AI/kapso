"""Domain-neutral pinned Docker CLI and daemon authority."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Protocol

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
    BoundedProcessRunner,
)
from kapso.cross_run.settings import DockerRuntimeSettings

_AUTHORITY_DIRECTORY_NAME = "authority"
_DOCKER_CONFIG_DIRECTORY_PREFIX = "docker-config-"
_DOCKER_CONFIG_FILENAME = "config.json"
_PINNED_DOCKER_FILENAME_PREFIX = "docker-"
_EMPTY_DOCKER_CONFIG = b'{"auths":{}}\n'
_DOCKER_HOST_PREFIX = "unix://"
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLATFORM_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_REGISTRY_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_REPOSITORY_COMPONENT_PATTERN = re.compile(r"^[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*$")


class PinnedDockerRuntimeError(RuntimeError):
    """The Docker client, daemon, image, or command violates pinned authority."""


class PinnedDockerProcessRunner(Protocol):
    """The bounded host-process primitive used by the Docker runtime."""

    def run(self, request: BoundedProcessRequest) -> BoundedProcessResult: ...


@dataclass(frozen=True)
class DockerImageAuthority(StrictContract):
    """Exact content and platform identity admitted by the pinned runtime."""

    image_authority_id: str
    image_reference: str
    image_config_digest: str
    operating_system: str
    architecture: str
    architecture_variant: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "docker-image-authority"
    IDENTITY_FIELD: ClassVar[str] = "image_authority_id"

    def _validate(self) -> None:
        if (
            not _is_canonical_pinned_image_reference(self.image_reference)
            or not isinstance(self.image_config_digest, str)
            or _DIGEST_PATTERN.fullmatch(self.image_config_digest) is None
            or not isinstance(self.operating_system, str)
            or _PLATFORM_IDENTIFIER_PATTERN.fullmatch(self.operating_system) is None
            or not isinstance(self.architecture, str)
            or _PLATFORM_IDENTIFIER_PATTERN.fullmatch(self.architecture) is None
            or (
                self.architecture_variant is not None
                and (
                    not isinstance(self.architecture_variant, str)
                    or _PLATFORM_IDENTIFIER_PATTERN.fullmatch(self.architecture_variant)
                    is None
                )
            )
        ):
            raise PinnedDockerRuntimeError("Docker image authority is invalid")


class PinnedDockerRuntime:
    """Execute Docker only through privately pinned bytes and an exact daemon."""

    def __init__(
        self,
        *,
        trusted_root: Path,
        settings: DockerRuntimeSettings,
        process_runner: PinnedDockerProcessRunner,
    ) -> None:
        if type(settings) is not DockerRuntimeSettings:
            raise PinnedDockerRuntimeError(
                "pinned Docker runtime requires exact runtime settings"
            )
        if not callable(getattr(process_runner, "run", None)):
            raise PinnedDockerRuntimeError(
                "pinned Docker runtime requires a bounded process runner"
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
        docker_path = authority_root / (
            f"{_PINNED_DOCKER_FILENAME_PREFIX}{executable_digest_suffix}"
        )
        with ExitStack() as initialization_descriptors:
            trusted_root_descriptor = os.open(
                trusted_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            initialization_descriptors.callback(os.close, trusted_root_descriptor)
            fcntl.flock(trusted_root_descriptor, fcntl.LOCK_EX)
            _ensure_private_directory(authority_root, trusted_root)
            _ensure_private_directory(docker_config_root, trusted_root)
            docker_bytes = read_verified_root_executable(
                Path(settings.runtime_executable_path),
                settings.runtime_executable_digest,
            )
            _publish_or_verify_private_executable(docker_path, docker_bytes)
            _publish_or_verify_private_file(
                docker_config_root / _DOCKER_CONFIG_FILENAME,
                _EMPTY_DOCKER_CONFIG,
            )
            fcntl.flock(trusted_root_descriptor, fcntl.LOCK_UN)
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
        settings: DockerRuntimeSettings,
    ) -> PinnedDockerRuntime:
        return cls(
            trusted_root=trusted_root,
            settings=settings,
            process_runner=BoundedProcessRunner(),
        )

    @property
    def trusted_root(self) -> Path:
        return self._trusted_root

    @property
    def settings(self) -> DockerRuntimeSettings:
        return self._settings

    def require_live_authority(self) -> None:
        self._require_local_authority()
        version = self.run_json_control(("version", "--format", "{{json .}}"))
        info = self.run_json_control(("info", "--format", "{{json .}}"))
        _require_daemon_authority(version, info, self._settings)

    def inspect_exact_image(
        self,
        authority: DockerImageAuthority,
    ) -> Mapping[str, Any]:
        if type(authority) is not DockerImageAuthority:
            raise PinnedDockerRuntimeError(
                "pinned Docker image requires an exact authority"
            )
        self.require_live_authority()
        image = self.run_json_control(
            (
                "image",
                "inspect",
                "--format",
                "{{json .}}",
                authority.image_reference,
            )
        )
        _require_image_identity(image, authority)
        return image

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
            raise PinnedDockerRuntimeError(
                "pinned Docker control command failed or emitted stderr"
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
            raise PinnedDockerRuntimeError(
                "pinned Docker arguments must be non-empty strings"
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
            raise PinnedDockerRuntimeError(
                "pinned Docker runner changed its exact request"
            )
        return result

    def _require_local_authority(self) -> None:
        if read_verified_private_executable(self._docker_path) != self._docker_digest:
            raise PinnedDockerRuntimeError("pinned Docker executable changed")
        _require_runtime_socket(Path(self._settings.runtime_socket_path))


def read_verified_root_executable(path: Path, expected_digest: str) -> bytes:
    """Read one immutable-enough root-owned executable through a no-follow fd."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise PinnedDockerRuntimeError(
            "Docker authority executable path must be absolute and normalized"
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
            raise PinnedDockerRuntimeError(
                "Docker authority executable is not immutable root-owned code"
            )
        payload = handle.read()
        metadata_after = os.fstat(handle.fileno())
    if (
        (metadata_before.st_dev, metadata_before.st_ino, metadata_before.st_size)
        != (metadata_after.st_dev, metadata_after.st_ino, metadata_after.st_size)
        or len(payload) != metadata_before.st_size
        or tree_or_blob_digest(payload) != expected_digest
    ):
        raise PinnedDockerRuntimeError(
            "Docker authority executable differs from its pinned digest"
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
            raise PinnedDockerRuntimeError("private Docker executable is unsafe")
        payload = handle.read()
        metadata_after = os.fstat(handle.fileno())
    if (metadata_before.st_dev, metadata_before.st_ino, metadata_before.st_size) != (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
    ) or len(payload) != metadata_before.st_size:
        raise PinnedDockerRuntimeError(
            "private Docker executable changed while reading"
        )
    return tree_or_blob_digest(payload)


def _require_daemon_authority(
    version: Mapping[str, Any],
    info: Mapping[str, Any],
    settings: DockerRuntimeSettings,
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
        raise PinnedDockerRuntimeError("Docker daemon differs from its exact authority")


def _require_image_identity(
    image: Mapping[str, Any],
    authority: DockerImageAuthority,
) -> None:
    repo_digests = _require_string_set(image.get("RepoDigests"), "Docker image digests")
    variant = image.get("Variant")
    normalized_variant = None if variant in {None, ""} else variant
    if (
        image.get("Id") != authority.image_config_digest
        or authority.image_reference not in repo_digests
        or image.get("Os") != authority.operating_system
        or image.get("Architecture") != authority.architecture
        or normalized_variant != authority.architecture_variant
    ):
        raise PinnedDockerRuntimeError(
            "Docker image differs from its exact content identity"
        )


def _is_canonical_pinned_image_reference(value: Any) -> bool:
    if not isinstance(value, str) or value != value.lower() or "\x00" in value:
        return False
    repository, separator, manifest_digest = value.rpartition("@")
    if (
        separator != "@"
        or _DIGEST_PATTERN.fullmatch(manifest_digest) is None
        or "/" not in repository
    ):
        return False
    registry, *components = repository.split("/")
    if (
        not registry
        or not components
        or any(
            _REPOSITORY_COMPONENT_PATTERN.fullmatch(component) is None
            for component in components
        )
    ):
        return False
    host = registry
    has_explicit_port = False
    if ":" in registry:
        has_explicit_port = True
        host, port_separator, port = registry.rpartition(":")
        if (
            port_separator != ":"
            or not port.isascii()
            or not port.isdigit()
            or len(port) > 5
            or not 1 <= int(port) <= 65535
        ):
            return False
    if (
        not host
        or (host != "localhost" and "." not in host and not has_explicit_port)
        or any(
            _REGISTRY_HOST_LABEL_PATTERN.fullmatch(label) is None
            for label in host.split(".")
        )
    ):
        return False
    return True


def _parse_single_json_object(payload: bytes) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise PinnedDockerRuntimeError("Docker JSON output lacks its exact line ending")
    encoded = payload[:-1]
    if not encoded or b"\n" in encoded or b"\r" in encoded:
        raise PinnedDockerRuntimeError("Docker JSON output is not one document")
    decoded = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(decoded, dict):
        raise PinnedDockerRuntimeError("Docker JSON output is not an object")
    return MappingProxyType(decoded)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise PinnedDockerRuntimeError(
                "Docker JSON output contains a duplicate key"
            )
        decoded[key] = value
    return decoded


def _reject_nonstandard_json_constant(value: str) -> None:
    raise PinnedDockerRuntimeError(
        f"Docker JSON output contains nonstandard constant {value}"
    )


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
    name: str,
) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise PinnedDockerRuntimeError(f"{name} is not an object")
    return child


def _require_string_set(value: Any, name: str) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise PinnedDockerRuntimeError(f"{name} is not a string list")
    return frozenset(value)


def _require_runtime_socket(path: Path) -> None:
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise PinnedDockerRuntimeError("Docker socket path is not absolute and direct")
    metadata = path.lstat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o002
    ):
        raise PinnedDockerRuntimeError(
            "Docker authority is not a root-owned Unix socket"
        )


def _require_private_root(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise PinnedDockerRuntimeError(
            "Docker trusted root must be absolute and resolved"
        )
    metadata = path.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise PinnedDockerRuntimeError("Docker trusted root must be owner-private")


def _ensure_private_directory(path: Path, parent: Path) -> None:
    if path.parent != parent:
        raise PinnedDockerRuntimeError(
            "Docker private directory is outside its trusted parent"
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
        raise PinnedDockerRuntimeError("Docker private directory is unsafe")


def _publish_or_verify_private_executable(path: Path, payload: bytes) -> None:
    if not path.exists():
        _write_private_file(path, payload, 0o500)
    if read_verified_private_executable(path) != tree_or_blob_digest(payload):
        raise PinnedDockerRuntimeError(
            "pinned Docker executable conflicts with authority"
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
        raise PinnedDockerRuntimeError("private Docker configuration is unsafe")


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
            raise PinnedDockerRuntimeError("private Docker file is unsafe")
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
