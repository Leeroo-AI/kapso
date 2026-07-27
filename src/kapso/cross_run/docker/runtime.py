"""Domain-neutral pinned Docker CLI and daemon authority."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import re
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident, Lock
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Protocol
from weakref import WeakKeyDictionary

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
_LIBC_FLOCK = ctypes.CDLL(None, use_errno=True).flock
_LIBC_FLOCK.argtypes = (ctypes.c_int, ctypes.c_int)
_LIBC_FLOCK.restype = ctypes.c_int
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_RESOURCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLATFORM_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_REGISTRY_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_REPOSITORY_COMPONENT_PATTERN = re.compile(r"^[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*$")
_DOCKER_OBSERVATION_AUTHORITY_ISSUANCE = object()
_DOCKER_OBSERVATION_AUTHORITIES: WeakKeyDictionary[
    PinnedDockerObservationAuthority, PinnedDockerRuntime
] = WeakKeyDictionary()
_DOCKER_OBSERVATION_AUTHORITY_LOCK = Lock()
_DOCKER_START_AUTHORITY_ISSUANCE = object()
_DOCKER_START_EXCLUSION_ISSUANCE = object()
_DOCKER_START_CONTAINER_AUTHORITY = object()
_DOCKER_START_AUTHORITIES: WeakKeyDictionary[
    PinnedDockerStartAuthority, PinnedDockerRuntime
] = WeakKeyDictionary()
_DOCKER_START_AUTHORITY_LOCK = Lock()
_DOCKER_START_EXCLUSION_LEASES: WeakKeyDictionary[
    PinnedDockerStartExclusionLease, PinnedDockerStartAuthority
] = WeakKeyDictionary()
_DOCKER_START_EXCLUSION_LOCK = Lock()
_DOCKER_CONTAINMENT_AUTHORITY_ISSUANCE = object()
_DOCKER_CONTAINMENT_SIGNAL_AUTHORITY = object()
_DOCKER_CONTAINMENT_AUTHORITIES: WeakKeyDictionary[
    PinnedDockerContainmentAuthority, PinnedDockerRuntime
] = WeakKeyDictionary()
_DOCKER_CONTAINMENT_AUTHORITY_LOCK = Lock()
_DOCKER_CLEANUP_AUTHORITY_ISSUANCE = object()
_DOCKER_CLEANUP_EXCLUSION_ISSUANCE = object()
_DOCKER_CLEANUP_REMOVE_AUTHORITY = object()
_DOCKER_CLEANUP_AUTHORITIES: WeakKeyDictionary[
    PinnedDockerCleanupAuthority, PinnedDockerRuntime
] = WeakKeyDictionary()
_DOCKER_CLEANUP_AUTHORITY_LOCK = Lock()
_DOCKER_CLEANUP_EXCLUSION_LEASES: WeakKeyDictionary[
    PinnedDockerCleanupExclusionLease, PinnedDockerCleanupAuthority
] = WeakKeyDictionary()
_DOCKER_CLEANUP_EXCLUSION_LOCK = Lock()
_DOCKER_MUTATION_LEASE_OWNERS: set[tuple[int, int, str]] = set()
_DOCKER_MUTATION_LEASE_OWNER_LOCK = Lock()


class PinnedDockerRuntimeError(RuntimeError):
    """The Docker client, daemon, image, or command violates pinned authority."""


class PinnedDockerProcessRunner(Protocol):
    """The bounded host-process primitive used by the Docker runtime."""

    def run(self, request: BoundedProcessRequest) -> BoundedProcessResult: ...


class PinnedDockerObservationAuthority:
    """Issued read-only Docker authority with no provider mutation surface."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DOCKER_OBSERVATION_AUTHORITY_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker observation authority lacks issuance authority"
            )
        self._owner_process_id = os.getpid()

    def require_live_authority(self) -> None:
        _docker_observation_runtime(self).require_live_authority()

    @property
    def settings(self) -> DockerRuntimeSettings:
        """Return the immutable settings of the issuing pinned runtime."""

        return _docker_observation_runtime(self).settings

    def run_control(self, arguments: tuple[str, ...]) -> BoundedProcessResult:
        _require_docker_observation_arguments(arguments)
        return _docker_observation_runtime(self).run_control(arguments)

    def run_json_control(self, arguments: tuple[str, ...]) -> Mapping[str, Any]:
        return _parse_single_json_object(self.run_control(arguments).stdout)


class PinnedDockerStartAuthority:
    """Issued start-only projection; the trusted launch leaf seals the exact ID."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DOCKER_START_AUTHORITY_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker start authority lacks issuance authority"
            )
        self._owner_process_id = os.getpid()

    @property
    def settings(self) -> DockerRuntimeSettings:
        """Return the immutable settings of the issuing pinned runtime."""

        return _docker_start_runtime(self).settings

    def _issue_exclusion_lease(
        self,
        *,
        _authority: object,
    ) -> PinnedDockerStartExclusionLease:
        if _authority is not _DOCKER_START_EXCLUSION_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker start exclusion lacks closed authority"
            )
        runtime = _docker_start_runtime(self)
        mutation_lease = _open_docker_mutation_lease(
            runtime,
            timeout_seconds=runtime.settings.command_timeout_seconds,
        )
        lease = PinnedDockerStartExclusionLease(
            start_authority=self,
            mutation_lease=mutation_lease,
            _authority=_DOCKER_START_EXCLUSION_ISSUANCE,
        )
        with _DOCKER_START_EXCLUSION_LOCK:
            if _DOCKER_START_EXCLUSION_LEASES.get(lease) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker start exclusion identity is already issued"
                )
            _DOCKER_START_EXCLUSION_LEASES[lease] = self
        return lease

    def _start_created_container_once(
        self,
        *,
        container_id: str,
        exclusion_lease: PinnedDockerStartExclusionLease,
        _authority: object,
    ) -> BoundedProcessResult:
        """Linearize one created-state check and start under mutation exclusion."""

        if (
            type(container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or not _docker_start_exclusion_matches(self, exclusion_lease)
            or _authority is not _DOCKER_START_CONTAINER_AUTHORITY
        ):
            raise PinnedDockerRuntimeError(
                "Docker container start lacks exact closed authority"
            )
        runtime = _docker_start_runtime(self)
        settings = runtime.settings
        before_start = runtime.run_json_control(
            (
                "container",
                "inspect",
                "--format",
                "{{json .}}",
                container_id,
            )
        )
        if _observed_container_status(before_start, container_id) != "created":
            raise PinnedDockerRuntimeError(
                "Docker start lacks an exact created occurrence"
            )
        exclusion_lease.require_current()
        return runtime._run_bounded_under_mutation_lease(
            ("container", "start", container_id),
            exclusion_lease._mutation_lease,
            timeout_seconds=settings.command_timeout_seconds,
            cleanup_timeout_seconds=settings.cleanup_timeout_seconds,
            stdout_byte_limit=settings.command_output_byte_limit,
            stderr_byte_limit=settings.command_output_byte_limit,
        )


class PinnedDockerContainmentAuthority:
    """Issued Docker authority limited to TERM/KILL of one caller-bound ID."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DOCKER_CONTAINMENT_AUTHORITY_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker containment authority lacks issuance authority"
            )
        self._owner_process_id = os.getpid()

    def require_live_authority(self) -> None:
        _docker_containment_runtime(self).require_live_authority()

    @property
    def settings(self) -> DockerRuntimeSettings:
        """Return the immutable settings of the issuing pinned runtime."""

        return _docker_containment_runtime(self).settings

    def _signal_container_once(
        self,
        *,
        container_id: str,
        signal_name: str,
        _authority: object,
    ) -> BoundedProcessResult:
        if (
            type(container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or signal_name not in {"SIGTERM", "SIGKILL"}
            or _authority is not _DOCKER_CONTAINMENT_SIGNAL_AUTHORITY
        ):
            raise PinnedDockerRuntimeError(
                "Docker containment signal lacks exact closed authority"
            )
        runtime = _docker_containment_runtime(self)
        settings = runtime.settings
        return runtime._run_bounded_without_mutation_lock(
            (
                "container",
                "kill",
                "--signal",
                signal_name,
                container_id,
            ),
            timeout_seconds=settings.command_timeout_seconds,
            cleanup_timeout_seconds=settings.cleanup_timeout_seconds,
            stdout_byte_limit=settings.command_output_byte_limit,
            stderr_byte_limit=settings.command_output_byte_limit,
        )


class _PinnedDockerMutationLease:
    """Retained daemon-wide flock shared by ordinary trusted Docker mutators."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        descriptor: int,
        path: Path,
        identity: tuple[int, int],
        owner_key: tuple[int, int, str],
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(descriptor) is not int
            or descriptor < 0
            or not isinstance(path, Path)
            or not path.is_absolute()
            or type(identity) is not tuple
            or len(identity) != 2
            or type(owner_key) is not tuple
            or len(owner_key) != 3
        ):
            raise PinnedDockerRuntimeError(
                "Docker mutation lease lacks exact lock authority"
            )
        self._descriptors = descriptors
        self._descriptor = descriptor
        self._path = path
        self._identity = identity
        self._owner_key = owner_key
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise PinnedDockerRuntimeError("Docker mutation lease is closed or foreign")
        metadata = os.fstat(self._descriptor)
        current = os.stat(self._path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o640
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or (metadata.st_dev, metadata.st_ino) != self._identity
            or (current.st_dev, current.st_ino) != self._identity
        ):
            raise PinnedDockerRuntimeError(
                "Docker mutation lock changed while retained"
            )

    def __enter__(self) -> _PinnedDockerMutationLease:
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise PinnedDockerRuntimeError("Docker mutation lease is closed or foreign")
        metadata = os.fstat(self._descriptor)
        integrity_changed = (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o640
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or (metadata.st_dev, metadata.st_ino) != self._identity
        )
        self._closed = True
        with _DOCKER_MUTATION_LEASE_OWNER_LOCK:
            removed = self._owner_key in _DOCKER_MUTATION_LEASE_OWNERS
            _DOCKER_MUTATION_LEASE_OWNERS.discard(self._owner_key)
        self._descriptors.close()
        if not removed or integrity_changed:
            raise PinnedDockerRuntimeError(
                "Docker mutation lock changed while retained"
            )


class PinnedDockerStartExclusionLease:
    """Owner-bound proof that start retains daemon-wide mutation exclusion."""

    def __init__(
        self,
        *,
        start_authority: PinnedDockerStartAuthority,
        mutation_lease: _PinnedDockerMutationLease,
        _authority: object,
    ) -> None:
        if (
            type(start_authority) is not PinnedDockerStartAuthority
            or type(mutation_lease) is not _PinnedDockerMutationLease
            or _authority is not _DOCKER_START_EXCLUSION_ISSUANCE
        ):
            raise PinnedDockerRuntimeError(
                "Docker start exclusion lacks issuance authority"
            )
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._mutation_lease = mutation_lease
        self._closed = False

    def require_current(self) -> None:
        with _DOCKER_START_EXCLUSION_LOCK:
            start_authority = _DOCKER_START_EXCLUSION_LEASES.get(self)
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or type(start_authority) is not PinnedDockerStartAuthority
        ):
            raise PinnedDockerRuntimeError(
                "Docker start exclusion is closed or foreign"
            )
        _docker_start_runtime(start_authority)
        self._mutation_lease.require_current()

    def __enter__(self) -> PinnedDockerStartExclusionLease:
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        with _DOCKER_START_EXCLUSION_LOCK:
            issuing_authority = _DOCKER_START_EXCLUSION_LEASES.get(self)
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or type(issuing_authority) is not PinnedDockerStartAuthority
        ):
            raise PinnedDockerRuntimeError(
                "Docker start exclusion is closed or foreign"
            )
        self._closed = True
        with _DOCKER_START_EXCLUSION_LOCK:
            removed = _DOCKER_START_EXCLUSION_LEASES.pop(self, None)
        self._mutation_lease.close()
        if removed is not issuing_authority:
            raise PinnedDockerRuntimeError(
                "Docker start exclusion lost its issuing authority"
            )


class PinnedDockerCleanupExclusionLease:
    """Owner-bound proof that one cleanup caller retains mutation exclusion."""

    def __init__(
        self,
        *,
        cleanup_authority: PinnedDockerCleanupAuthority,
        mutation_lease: _PinnedDockerMutationLease,
        _authority: object,
    ) -> None:
        if (
            type(cleanup_authority) is not PinnedDockerCleanupAuthority
            or type(mutation_lease) is not _PinnedDockerMutationLease
            or _authority is not _DOCKER_CLEANUP_EXCLUSION_ISSUANCE
        ):
            raise PinnedDockerRuntimeError(
                "Docker cleanup exclusion lacks issuance authority"
            )
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._mutation_lease = mutation_lease
        self._closed = False

    def require_current(self) -> None:
        """Require the original process/thread and live issuing authority."""

        with _DOCKER_CLEANUP_EXCLUSION_LOCK:
            cleanup_authority = _DOCKER_CLEANUP_EXCLUSION_LEASES.get(self)
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or type(cleanup_authority) is not PinnedDockerCleanupAuthority
        ):
            raise PinnedDockerRuntimeError(
                "Docker cleanup exclusion is closed or foreign"
            )
        _docker_cleanup_runtime(cleanup_authority)
        self._mutation_lease.require_current()

    def __enter__(self) -> PinnedDockerCleanupExclusionLease:
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        with _DOCKER_CLEANUP_EXCLUSION_LOCK:
            issuing_authority = _DOCKER_CLEANUP_EXCLUSION_LEASES.get(self)
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or type(issuing_authority) is not PinnedDockerCleanupAuthority
        ):
            raise PinnedDockerRuntimeError(
                "Docker cleanup exclusion is closed or foreign"
            )
        self._closed = True
        with _DOCKER_CLEANUP_EXCLUSION_LOCK:
            removed = _DOCKER_CLEANUP_EXCLUSION_LEASES.pop(self, None)
        self._mutation_lease.close()
        if type(removed) is not PinnedDockerCleanupAuthority:
            raise PinnedDockerRuntimeError(
                "Docker cleanup exclusion lost its issuing authority"
            )


class PinnedDockerCleanupAuthority:
    """Issued Docker authority limited to exact container and volume removal."""

    def __init__(self, *, _authority: object) -> None:
        if _authority is not _DOCKER_CLEANUP_AUTHORITY_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker cleanup authority lacks issuance authority"
            )
        self._owner_process_id = os.getpid()

    @property
    def settings(self) -> DockerRuntimeSettings:
        """Return the immutable settings of the issuing pinned runtime."""

        return _docker_cleanup_runtime(self).settings

    def _issue_exclusion_lease(
        self,
        *,
        _authority: object,
    ) -> PinnedDockerCleanupExclusionLease:
        if _authority is not _DOCKER_CLEANUP_EXCLUSION_ISSUANCE:
            raise PinnedDockerRuntimeError(
                "Docker cleanup exclusion lacks closed authority"
            )
        runtime = _docker_cleanup_runtime(self)
        mutation_lease = _open_docker_mutation_lease(
            runtime,
            timeout_seconds=runtime.settings.command_timeout_seconds,
        )
        lease = PinnedDockerCleanupExclusionLease(
            cleanup_authority=self,
            mutation_lease=mutation_lease,
            _authority=_DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
        )
        with _DOCKER_CLEANUP_EXCLUSION_LOCK:
            if _DOCKER_CLEANUP_EXCLUSION_LEASES.get(lease) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker cleanup exclusion identity is already issued"
                )
            _DOCKER_CLEANUP_EXCLUSION_LEASES[lease] = self
        return lease

    def _remove_stopped_container_once(
        self,
        *,
        container_id: str,
        exclusion_lease: PinnedDockerCleanupExclusionLease,
        _authority: object,
    ) -> BoundedProcessResult:
        if (
            type(container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or _authority is not _DOCKER_CLEANUP_REMOVE_AUTHORITY
            or not _docker_cleanup_exclusion_matches(self, exclusion_lease)
        ):
            raise PinnedDockerRuntimeError(
                "Docker container cleanup lacks exact closed authority"
            )
        runtime = _docker_cleanup_runtime(self)
        settings = runtime.settings
        return runtime._run_bounded_under_mutation_lease(
            (
                "container",
                "rm",
                container_id,
            ),
            exclusion_lease._mutation_lease,
            timeout_seconds=settings.command_timeout_seconds,
            cleanup_timeout_seconds=settings.cleanup_timeout_seconds,
            stdout_byte_limit=settings.command_output_byte_limit,
            stderr_byte_limit=settings.command_output_byte_limit,
        )

    def _remove_running_keeper_once(
        self,
        *,
        container_id: str,
        exclusion_lease: PinnedDockerCleanupExclusionLease,
        _authority: object,
    ) -> BoundedProcessResult:
        if (
            type(container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or _authority is not _DOCKER_CLEANUP_REMOVE_AUTHORITY
            or not _docker_cleanup_exclusion_matches(self, exclusion_lease)
        ):
            raise PinnedDockerRuntimeError(
                "Docker keeper cleanup lacks exact closed authority"
            )
        runtime = _docker_cleanup_runtime(self)
        settings = runtime.settings
        return runtime._run_bounded_under_mutation_lease(
            (
                "container",
                "rm",
                "--force",
                container_id,
            ),
            exclusion_lease._mutation_lease,
            timeout_seconds=settings.command_timeout_seconds,
            cleanup_timeout_seconds=settings.cleanup_timeout_seconds,
            stdout_byte_limit=settings.command_output_byte_limit,
            stderr_byte_limit=settings.command_output_byte_limit,
        )

    def _remove_volume_once(
        self,
        *,
        volume_name: str,
        exclusion_lease: PinnedDockerCleanupExclusionLease,
        _authority: object,
    ) -> BoundedProcessResult:
        if (
            type(volume_name) is not str
            or _DOCKER_RESOURCE_NAME_PATTERN.fullmatch(volume_name) is None
            or _authority is not _DOCKER_CLEANUP_REMOVE_AUTHORITY
            or not _docker_cleanup_exclusion_matches(self, exclusion_lease)
        ):
            raise PinnedDockerRuntimeError(
                "Docker volume cleanup lacks exact closed authority"
            )
        runtime = _docker_cleanup_runtime(self)
        settings = runtime.settings
        return runtime._run_bounded_under_mutation_lease(
            ("volume", "rm", volume_name),
            exclusion_lease._mutation_lease,
            timeout_seconds=settings.command_timeout_seconds,
            cleanup_timeout_seconds=settings.cleanup_timeout_seconds,
            stdout_byte_limit=settings.command_output_byte_limit,
            stderr_byte_limit=settings.command_output_byte_limit,
        )


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
        self._owner_process_id = os.getpid()
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

    def issue_observation_authority(self) -> PinnedDockerObservationAuthority:
        """Issue a read-only projection that cannot execute Docker mutations."""

        self.require_live_authority()
        authority = PinnedDockerObservationAuthority(
            _authority=_DOCKER_OBSERVATION_AUTHORITY_ISSUANCE
        )
        with _DOCKER_OBSERVATION_AUTHORITY_LOCK:
            if _DOCKER_OBSERVATION_AUTHORITIES.get(authority) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker observation authority identity is already issued"
                )
            _DOCKER_OBSERVATION_AUTHORITIES[authority] = self
        return authority

    def issue_start_authority(self) -> PinnedDockerStartAuthority:
        """Issue a projection limited to one exact created-container start."""

        self.require_live_authority()
        authority = PinnedDockerStartAuthority(
            _authority=_DOCKER_START_AUTHORITY_ISSUANCE
        )
        with _DOCKER_START_AUTHORITY_LOCK:
            if _DOCKER_START_AUTHORITIES.get(authority) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker start authority identity is already issued"
                )
            _DOCKER_START_AUTHORITIES[authority] = self
        return authority

    def issue_containment_authority(self) -> PinnedDockerContainmentAuthority:
        """Issue a projection limited to exact container TERM/KILL signals."""

        self.require_live_authority()
        authority = PinnedDockerContainmentAuthority(
            _authority=_DOCKER_CONTAINMENT_AUTHORITY_ISSUANCE
        )
        with _DOCKER_CONTAINMENT_AUTHORITY_LOCK:
            if _DOCKER_CONTAINMENT_AUTHORITIES.get(authority) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker containment authority identity is already issued"
                )
            _DOCKER_CONTAINMENT_AUTHORITIES[authority] = self
        return authority

    def issue_cleanup_authority(self) -> PinnedDockerCleanupAuthority:
        """Issue a projection limited to exact container and volume removal."""

        self.require_live_authority()
        authority = PinnedDockerCleanupAuthority(
            _authority=_DOCKER_CLEANUP_AUTHORITY_ISSUANCE
        )
        with _DOCKER_CLEANUP_AUTHORITY_LOCK:
            if _DOCKER_CLEANUP_AUTHORITIES.get(authority) is not None:
                raise PinnedDockerRuntimeError(
                    "Docker cleanup authority identity is already issued"
                )
            _DOCKER_CLEANUP_AUTHORITIES[authority] = self
        return authority

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
        _require_docker_arguments(arguments)
        if _is_docker_read_only_control(arguments):
            return self._run_bounded_without_mutation_lock(
                arguments,
                timeout_seconds=timeout_seconds,
                cleanup_timeout_seconds=cleanup_timeout_seconds,
                stdout_byte_limit=stdout_byte_limit,
                stderr_byte_limit=stderr_byte_limit,
            )
        if _is_docker_attached_container_start(arguments):
            return self._run_bounded_attached_container_start(
                arguments,
                timeout_seconds=timeout_seconds,
                cleanup_timeout_seconds=cleanup_timeout_seconds,
                stdout_byte_limit=stdout_byte_limit,
                stderr_byte_limit=stderr_byte_limit,
            )
        return self._run_bounded_with_mutation_lock(
            arguments,
            timeout_seconds=timeout_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            stdout_byte_limit=stdout_byte_limit,
            stderr_byte_limit=stderr_byte_limit,
            mutation_timeout_seconds=timeout_seconds,
        )

    def _run_bounded_attached_container_start(
        self,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: int,
        cleanup_timeout_seconds: int,
        stdout_byte_limit: int,
        stderr_byte_limit: int,
    ) -> BoundedProcessResult:
        container_id = arguments[3]
        with ThreadPoolExecutor(max_workers=1) as execution:
            with _open_docker_mutation_lease(
                self,
                timeout_seconds=timeout_seconds,
            ) as mutation_lease:
                before_start = self.run_json_control(
                    (
                        "container",
                        "inspect",
                        "--format",
                        "{{json .}}",
                        container_id,
                    )
                )
                if _observed_container_status(before_start, container_id) != "created":
                    raise PinnedDockerRuntimeError(
                        "attached Docker start lacks an exact created occurrence"
                    )
                attached = execution.submit(
                    self._run_bounded_without_mutation_lock,
                    arguments,
                    timeout_seconds=timeout_seconds,
                    cleanup_timeout_seconds=cleanup_timeout_seconds,
                    stdout_byte_limit=stdout_byte_limit,
                    stderr_byte_limit=stderr_byte_limit,
                )
                while not attached.done():
                    observation = self.run_json_control(
                        (
                            "container",
                            "inspect",
                            "--format",
                            "{{json .}}",
                            container_id,
                        )
                    )
                    if (
                        _observed_container_status(observation, container_id)
                        != "created"
                    ):
                        break
                    time.sleep(self._settings.run_action_barrier_poll_interval_seconds)
                mutation_lease.require_current()
            return attached.result()

    def _run_bounded_with_mutation_lock(
        self,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: int,
        cleanup_timeout_seconds: int,
        stdout_byte_limit: int,
        stderr_byte_limit: int,
        mutation_timeout_seconds: int,
    ) -> BoundedProcessResult:
        with _open_docker_mutation_lease(
            self,
            timeout_seconds=mutation_timeout_seconds,
        ) as mutation_lease:
            return self._run_bounded_under_mutation_lease(
                arguments,
                mutation_lease,
                timeout_seconds=timeout_seconds,
                cleanup_timeout_seconds=cleanup_timeout_seconds,
                stdout_byte_limit=stdout_byte_limit,
                stderr_byte_limit=stderr_byte_limit,
            )

    def _run_bounded_under_mutation_lease(
        self,
        arguments: tuple[str, ...],
        mutation_lease: _PinnedDockerMutationLease,
        *,
        timeout_seconds: int,
        cleanup_timeout_seconds: int,
        stdout_byte_limit: int,
        stderr_byte_limit: int,
    ) -> BoundedProcessResult:
        _require_docker_arguments(arguments)
        if type(mutation_lease) is not _PinnedDockerMutationLease:
            raise PinnedDockerRuntimeError(
                "Docker mutation lacks daemon-wide exclusion"
            )
        mutation_lease.require_current()
        result = self._run_bounded_without_mutation_lock(
            arguments,
            timeout_seconds=timeout_seconds,
            cleanup_timeout_seconds=cleanup_timeout_seconds,
            stdout_byte_limit=stdout_byte_limit,
            stderr_byte_limit=stderr_byte_limit,
        )
        mutation_lease.require_current()
        return result

    def _run_bounded_without_mutation_lock(
        self,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: int,
        cleanup_timeout_seconds: int,
        stdout_byte_limit: int,
        stderr_byte_limit: int,
    ) -> BoundedProcessResult:
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
        if self._owner_process_id != os.getpid():
            raise PinnedDockerRuntimeError(
                "pinned Docker runtime belongs to another process"
            )
        if read_verified_private_executable(self._docker_path) != self._docker_digest:
            raise PinnedDockerRuntimeError("pinned Docker executable changed")
        _require_runtime_socket(Path(self._settings.runtime_socket_path))


def _docker_observation_runtime(
    authority: PinnedDockerObservationAuthority,
) -> PinnedDockerRuntime:
    with _DOCKER_OBSERVATION_AUTHORITY_LOCK:
        runtime = _DOCKER_OBSERVATION_AUTHORITIES.get(authority)
    if (
        type(authority) is not PinnedDockerObservationAuthority
        or authority._owner_process_id != os.getpid()
        or type(runtime) is not PinnedDockerRuntime
    ):
        raise PinnedDockerRuntimeError(
            "Docker observation authority is unissued or foreign"
        )
    return runtime


def _docker_start_runtime(
    authority: PinnedDockerStartAuthority,
) -> PinnedDockerRuntime:
    with _DOCKER_START_AUTHORITY_LOCK:
        runtime = _DOCKER_START_AUTHORITIES.get(authority)
    if (
        type(authority) is not PinnedDockerStartAuthority
        or authority._owner_process_id != os.getpid()
        or type(runtime) is not PinnedDockerRuntime
    ):
        raise PinnedDockerRuntimeError("Docker start authority is unissued or foreign")
    return runtime


def _docker_containment_runtime(
    authority: PinnedDockerContainmentAuthority,
) -> PinnedDockerRuntime:
    with _DOCKER_CONTAINMENT_AUTHORITY_LOCK:
        runtime = _DOCKER_CONTAINMENT_AUTHORITIES.get(authority)
    if (
        type(authority) is not PinnedDockerContainmentAuthority
        or authority._owner_process_id != os.getpid()
        or type(runtime) is not PinnedDockerRuntime
    ):
        raise PinnedDockerRuntimeError(
            "Docker containment authority is unissued or foreign"
        )
    return runtime


def _docker_cleanup_runtime(
    authority: PinnedDockerCleanupAuthority,
) -> PinnedDockerRuntime:
    with _DOCKER_CLEANUP_AUTHORITY_LOCK:
        runtime = _DOCKER_CLEANUP_AUTHORITIES.get(authority)
    if (
        type(authority) is not PinnedDockerCleanupAuthority
        or authority._owner_process_id != os.getpid()
        or type(runtime) is not PinnedDockerRuntime
    ):
        raise PinnedDockerRuntimeError(
            "Docker cleanup authority is unissued or foreign"
        )
    return runtime


def _docker_authorities_share_runtime(
    observation_authority: PinnedDockerObservationAuthority,
    containment_authority: PinnedDockerContainmentAuthority,
) -> bool:
    """Join read and mutation projections without exposing their runtime."""

    return _docker_observation_runtime(
        observation_authority
    ) is _docker_containment_runtime(containment_authority)


def _docker_observation_and_start_authorities_share_runtime(
    observation_authority: PinnedDockerObservationAuthority,
    start_authority: PinnedDockerStartAuthority,
) -> bool:
    """Join read and start projections without exposing their runtime."""

    return _docker_observation_runtime(observation_authority) is _docker_start_runtime(
        start_authority
    )


def _docker_observation_and_cleanup_authorities_share_runtime(
    observation_authority: PinnedDockerObservationAuthority,
    cleanup_authority: PinnedDockerCleanupAuthority,
) -> bool:
    """Join read and cleanup projections without exposing their runtime."""

    return _docker_observation_runtime(
        observation_authority
    ) is _docker_cleanup_runtime(cleanup_authority)


def _docker_start_exclusion_matches(
    start_authority: PinnedDockerStartAuthority,
    exclusion_lease: PinnedDockerStartExclusionLease,
) -> bool:
    """Require one current mutation lease issued by the exact start authority."""

    if type(exclusion_lease) is not PinnedDockerStartExclusionLease:
        return False
    exclusion_lease.require_current()
    with _DOCKER_START_EXCLUSION_LOCK:
        issuing_authority = _DOCKER_START_EXCLUSION_LEASES.get(exclusion_lease)
    return issuing_authority is start_authority


def _docker_cleanup_exclusion_matches(
    cleanup_authority: PinnedDockerCleanupAuthority,
    exclusion_lease: PinnedDockerCleanupExclusionLease,
) -> bool:
    """Require one current lease issued by the exact cleanup authority."""

    if type(exclusion_lease) is not PinnedDockerCleanupExclusionLease:
        return False
    exclusion_lease.require_current()
    with _DOCKER_CLEANUP_EXCLUSION_LOCK:
        issuing_authority = _DOCKER_CLEANUP_EXCLUSION_LEASES.get(exclusion_lease)
    return issuing_authority is cleanup_authority


def _open_docker_mutation_lease(
    runtime: PinnedDockerRuntime,
    *,
    timeout_seconds: int,
) -> _PinnedDockerMutationLease:
    if type(runtime) is not PinnedDockerRuntime:
        raise PinnedDockerRuntimeError(
            "Docker mutation lease requires one pinned runtime"
        )
    if type(timeout_seconds) is not int or timeout_seconds < 0:
        raise PinnedDockerRuntimeError("Docker mutation lease timeout is invalid")
    path = Path(runtime.settings.runtime_mutation_lock_path)
    parent = path.parent
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or parent.resolve() != parent
    ):
        raise PinnedDockerRuntimeError(
            "Docker mutation lock path is not absolute and normalized"
        )
    parent_metadata = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != 0
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise PinnedDockerRuntimeError(
            "Docker mutation lock parent is not an immutable root-owned directory"
        )
    owner_key = (os.getpid(), get_ident(), str(path))
    with _DOCKER_MUTATION_LEASE_OWNER_LOCK:
        if owner_key in _DOCKER_MUTATION_LEASE_OWNERS:
            raise PinnedDockerRuntimeError(
                "Docker mutation lease cannot be acquired recursively"
            )
    with ExitStack() as descriptors:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, descriptor)
        os.set_inheritable(descriptor, False)
        deadline = time.monotonic() + timeout_seconds
        acquired = False
        while not acquired:
            acquired = _acquire_docker_mutation_flock_once(descriptor)
            if not acquired:
                if time.monotonic() >= deadline:
                    raise PinnedDockerRuntimeError(
                        "Docker mutation exclusion deadline elapsed"
                    )
                time.sleep(
                    min(
                        runtime.settings.run_action_barrier_poll_interval_seconds,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
        metadata = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        socket_metadata = os.stat(
            runtime.settings.runtime_socket_path,
            follow_symlinks=False,
        )
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != socket_metadata.st_gid
            or stat.S_IMODE(metadata.st_mode) != 0o640
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or (current.st_dev, current.st_ino) != identity
        ):
            raise PinnedDockerRuntimeError(
                "Docker mutation lock is unsafe or changed during acquisition"
            )
        with _DOCKER_MUTATION_LEASE_OWNER_LOCK:
            if owner_key in _DOCKER_MUTATION_LEASE_OWNERS:
                raise PinnedDockerRuntimeError(
                    "Docker mutation lease identity is already retained"
                )
            _DOCKER_MUTATION_LEASE_OWNERS.add(owner_key)
        lease = _PinnedDockerMutationLease(
            descriptors=descriptors,
            descriptor=descriptor,
            path=path,
            identity=identity,
            owner_key=owner_key,
        )
        lease._descriptors = descriptors.pop_all()
        return lease


def _acquire_docker_mutation_flock_once(descriptor: int) -> bool:
    if type(descriptor) is not int or descriptor < 0:
        raise PinnedDockerRuntimeError("Docker mutation lock descriptor is invalid")
    result = _LIBC_FLOCK(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EACCES, errno.EAGAIN}:
        return False
    raise PinnedDockerRuntimeError(
        f"Docker mutation lock acquisition failed with errno {error_number}"
    )


def _require_docker_arguments(arguments: tuple[str, ...]) -> None:
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


def _is_docker_resource_observation(arguments: tuple[str, ...]) -> bool:
    return _is_docker_list_observation(arguments) or _is_docker_inspect_observation(
        arguments
    )


def _is_docker_attached_container_start(arguments: tuple[str, ...]) -> bool:
    return (
        len(arguments) == 4
        and arguments[:3] == ("container", "start", "--attach")
        and _CONTAINER_ID_PATTERN.fullmatch(arguments[3]) is not None
    )


def _observed_container_status(
    observation: Mapping[str, Any],
    container_id: str,
) -> str:
    if (
        not isinstance(observation, Mapping)
        or type(container_id) is not str
        or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        or observation.get("Id") != container_id
    ):
        raise PinnedDockerRuntimeError("Docker container observation changed identity")
    state = _require_mapping(
        observation,
        "State",
        "Docker container state",
    )
    status = state.get("Status")
    if type(status) is not str or not status:
        raise PinnedDockerRuntimeError("Docker container state lacks exact status")
    return status


def _is_docker_read_only_control(arguments: tuple[str, ...]) -> bool:
    return (
        arguments
        in {
            ("version", "--format", "{{json .}}"),
            ("info", "--format", "{{json .}}"),
        }
        or (
            len(arguments) == 5
            and arguments[:4] == ("image", "inspect", "--format", "{{json .}}")
        )
        or (
            len(arguments) == 3
            and arguments[:2] == ("container", "wait")
            and _CONTAINER_ID_PATTERN.fullmatch(arguments[2]) is not None
        )
        or _is_docker_resource_observation(arguments)
    )


def _require_docker_observation_arguments(arguments: tuple[str, ...]) -> None:
    _require_docker_arguments(arguments)
    if _is_docker_resource_observation(arguments):
        return
    raise PinnedDockerRuntimeError(
        "Docker observation authority cannot execute provider mutations"
    )


def _is_docker_list_observation(arguments: tuple[str, ...]) -> bool:
    if arguments[:2] == ("container", "ls"):
        fixed_prefix = ("container", "ls", "--all", "--no-trunc")
        expected_format = "{{json .ID}}"
    elif arguments[:2] == ("volume", "ls"):
        fixed_prefix = ("volume", "ls")
        expected_format = "{{json .Name}}"
    else:
        return False
    if (
        arguments[: len(fixed_prefix)] != fixed_prefix
        or len(arguments) < len(fixed_prefix) + 2
        or arguments[-2:] != ("--format", expected_format)
    ):
        return False
    filters = arguments[len(fixed_prefix) : -2]
    return len(filters) % 2 == 0 and all(
        filters[position] == "--filter" for position in range(0, len(filters), 2)
    )


def _is_docker_inspect_observation(arguments: tuple[str, ...]) -> bool:
    return (
        len(arguments) == 5
        and arguments[0] in {"container", "volume"}
        and arguments[1:4] == ("inspect", "--format", "{{json .}}")
    )


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
        or info.get("DockerRootDir") != settings.runtime_root_directory
        or info.get("CgroupDriver") != settings.runtime_cgroup_driver
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
