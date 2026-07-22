"""Handle-owned Docker resources for isolated expert source replay."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.contracts import ExpertSourceReplayComputeBinding
from kapso.cross_run.expert.replay_docker_runtime import SourceReplayDockerRuntime
from kapso.cross_run.expert.replay_execution import (
    SourceReplayProviderExecutionHandle,
)
from kapso.cross_run.settings import SourceReplayDockerProviderSettings

_HANDLE_LABEL = "io.kapso.source-replay.handle"
_ROLE_LABEL = "io.kapso.source-replay.role"
_EVALUATOR_ROLE = "evaluator"
_KEEPER_ROLE = "keeper"
_VOLUME_ROLE = "volume"
_RESOURCE_NAME_PREFIX = "kapso-source-replay"
_WORKSPACE_NAME_PREFIX = "replay-"
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SourceReplayDockerResourceError(RuntimeError):
    """A handle-owned Docker resource is absent, substituted, or unsafe."""


@dataclass(frozen=True)
class SourceReplayDockerResourceIdentity:
    provider_handle_id: str
    workspace_root: Path
    evaluator_name: str
    keeper_name: str
    volume_name: str

    def labels_for(self, role: str) -> Mapping[str, str]:
        if role not in {_EVALUATOR_ROLE, _KEEPER_ROLE, _VOLUME_ROLE}:
            raise SourceReplayDockerResourceError(
                "source replay Docker resource role is unsupported"
            )
        return MappingProxyType(
            {
                _HANDLE_LABEL: self.provider_handle_id,
                _ROLE_LABEL: role,
            }
        )


@dataclass(frozen=True)
class SourceReplayDockerContainerObservation:
    container_id: str
    name: str
    role: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class SourceReplayDockerVolumeObservation:
    name: str
    payload: Mapping[str, Any]


class SourceReplayDockerResourceManager:
    """Create and reap only resources bearing one unpredictable handle label."""

    def __init__(self, runtime: SourceReplayDockerRuntime) -> None:
        if type(runtime) is not SourceReplayDockerRuntime:
            raise SourceReplayDockerResourceError(
                "source replay resources require the exact Docker runtime"
            )
        self._runtime = runtime

    @property
    def runtime(self) -> SourceReplayDockerRuntime:
        return self._runtime

    def identity(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> SourceReplayDockerResourceIdentity:
        if type(provider_handle) is not SourceReplayProviderExecutionHandle:
            raise SourceReplayDockerResourceError(
                "source replay Docker resources require an exact provider handle"
            )
        suffix = provider_handle.provider_handle_id.rsplit(":sha256:", 1)[-1]
        if re.fullmatch(r"[0-9a-f]{64}", suffix) is None:
            raise SourceReplayDockerResourceError(
                "source replay Docker provider handle digest is invalid"
            )
        return SourceReplayDockerResourceIdentity(
            provider_handle_id=provider_handle.provider_handle_id,
            workspace_root=(
                self._runtime.trusted_root / f"{_WORKSPACE_NAME_PREFIX}{suffix}"
            ),
            evaluator_name=f"{_RESOURCE_NAME_PREFIX}-{_EVALUATOR_ROLE}-{suffix}",
            keeper_name=f"{_RESOURCE_NAME_PREFIX}-{_KEEPER_ROLE}-{suffix}",
            volume_name=f"{_RESOURCE_NAME_PREFIX}-{_VOLUME_ROLE}-{suffix}",
        )

    def require_absent(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> SourceReplayDockerResourceIdentity:
        identity = self.identity(provider_handle)
        observations = self.observe(identity)
        if any(observation is not None for observation in observations):
            raise SourceReplayDockerResourceError(
                "source replay Docker handle already owns daemon resources"
            )
        if os.path.lexists(identity.workspace_root):
            raise SourceReplayDockerResourceError(
                "source replay Docker handle workspace already exists"
            )
        return identity

    def observe(
        self,
        identity: SourceReplayDockerResourceIdentity,
    ) -> tuple[
        SourceReplayDockerContainerObservation | None,
        SourceReplayDockerContainerObservation | None,
        SourceReplayDockerVolumeObservation | None,
    ]:
        if type(identity) is not SourceReplayDockerResourceIdentity:
            raise SourceReplayDockerResourceError(
                "source replay Docker observation requires exact resource identity"
            )
        evaluator = self._observe_container(
            identity.evaluator_name,
            identity.labels_for(_EVALUATOR_ROLE),
            _EVALUATOR_ROLE,
        )
        keeper = self._observe_container(
            identity.keeper_name,
            identity.labels_for(_KEEPER_ROLE),
            _KEEPER_ROLE,
        )
        volume = self._observe_volume(identity)
        return evaluator, keeper, volume

    def create_writable_volume(
        self,
        identity: SourceReplayDockerResourceIdentity,
        compute: ExpertSourceReplayComputeBinding,
    ) -> SourceReplayDockerVolumeObservation:
        if type(identity) is not SourceReplayDockerResourceIdentity or not isinstance(
            compute, ExpertSourceReplayComputeBinding
        ):
            raise SourceReplayDockerResourceError(
                "source replay writable volume requires exact identity and compute"
            )
        if self._observe_volume(identity) is not None:
            raise SourceReplayDockerResourceError(
                "source replay writable volume is not fresh"
            )
        settings = self._runtime.settings
        options = _writable_volume_options(identity, compute, settings)
        labels = identity.labels_for(_VOLUME_ROLE)
        result = self._runtime.run_control(
            (
                "volume",
                "create",
                "--driver",
                "local",
                "--label",
                f"{_HANDLE_LABEL}={labels[_HANDLE_LABEL]}",
                "--label",
                f"{_ROLE_LABEL}={labels[_ROLE_LABEL]}",
                "--opt",
                "type=tmpfs",
                "--opt",
                "device=tmpfs",
                "--opt",
                f"o={options['o']}",
                identity.volume_name,
            )
        )
        _require_exact_line(result.stdout, identity.volume_name)
        observation = self._observe_volume(identity)
        if observation is None:
            raise SourceReplayDockerResourceError(
                "source replay writable volume disappeared after creation"
            )
        payload = observation.payload
        if (
            payload.get("Driver") != "local"
            or payload.get("Scope") != "local"
            or payload.get("Options") != options
        ):
            raise SourceReplayDockerResourceError(
                "source replay writable volume differs from exact tmpfs authority"
            )
        return observation

    def remove_container(
        self,
        identity: SourceReplayDockerResourceIdentity,
        observation: SourceReplayDockerContainerObservation,
    ) -> None:
        if (
            type(identity) is not SourceReplayDockerResourceIdentity
            or type(observation) is not SourceReplayDockerContainerObservation
            or observation.role not in {_EVALUATOR_ROLE, _KEEPER_ROLE}
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker removal requires an exact container observation"
            )
        expected_name = (
            identity.evaluator_name
            if observation.role == _EVALUATOR_ROLE
            else identity.keeper_name
        )
        current = self._observe_container(
            expected_name,
            identity.labels_for(observation.role),
            observation.role,
        )
        if not source_replay_docker_container_observations_match(
            current,
            observation,
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker container changed before removal"
            )
        result = self._runtime.run_control(
            (
                "container",
                "rm",
                "--force",
                "--volumes",
                observation.container_id,
            )
        )
        _require_exact_line(result.stdout, observation.container_id)
        if (
            self._observe_container(
                expected_name,
                identity.labels_for(observation.role),
                observation.role,
            )
            is not None
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker container survived removal"
            )

    def stop_container(
        self,
        identity: SourceReplayDockerResourceIdentity,
        observation: SourceReplayDockerContainerObservation,
        grace_seconds: int,
    ) -> SourceReplayDockerContainerObservation:
        if (
            type(identity) is not SourceReplayDockerResourceIdentity
            or type(observation) is not SourceReplayDockerContainerObservation
            or observation.role != _EVALUATOR_ROLE
            or type(grace_seconds) is not int
            or grace_seconds <= 0
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker stop requires an exact evaluator and grace"
            )
        current = self._observe_container(
            identity.evaluator_name,
            identity.labels_for(_EVALUATOR_ROLE),
            _EVALUATOR_ROLE,
        )
        if not source_replay_docker_container_observations_match(
            current,
            observation,
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker evaluator changed before stop"
            )
        result = self._runtime.run_control(
            (
                "container",
                "stop",
                "--time",
                str(grace_seconds),
                observation.container_id,
            )
        )
        _require_exact_line(result.stdout, observation.container_id)
        stopped = self._observe_container(
            identity.evaluator_name,
            identity.labels_for(_EVALUATOR_ROLE),
            _EVALUATOR_ROLE,
        )
        if stopped is None or stopped.container_id != observation.container_id:
            raise SourceReplayDockerResourceError(
                "source replay Docker evaluator disappeared while stopping"
            )
        return stopped

    def cleanup_daemon_resources(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> SourceReplayDockerResourceIdentity:
        identity = self.identity(provider_handle)
        self._runtime.require_live_authority()
        evaluator, keeper, volume = self.observe(identity)
        for container in (evaluator, keeper):
            if container is not None:
                result = self._runtime.run_control(
                    (
                        "container",
                        "rm",
                        "--force",
                        "--volumes",
                        container.container_id,
                    )
                )
                _require_exact_line(result.stdout, container.container_id)
        if volume is not None:
            current_volume = self._observe_volume(identity)
            if current_volume is None or current_volume != volume:
                raise SourceReplayDockerResourceError(
                    "source replay writable volume changed before removal"
                )
            result = self._runtime.run_control(("volume", "rm", current_volume.name))
            _require_exact_line(result.stdout, current_volume.name)
        if any(observation is not None for observation in self.observe(identity)):
            raise SourceReplayDockerResourceError(
                "source replay Docker resources survived cleanup"
            )
        return identity

    def _observe_container(
        self,
        name: str,
        expected_labels: Mapping[str, str],
        role: str,
    ) -> SourceReplayDockerContainerObservation | None:
        result = self._runtime.run_control(
            (
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"name=^/{name}$",
                "--format",
                "{{json .Names}}",
            )
        )
        observed_name = _parse_optional_json_string(result.stdout)
        if observed_name is None:
            return None
        if observed_name != name:
            raise SourceReplayDockerResourceError(
                "source replay Docker container lookup was not exact"
            )
        payload = self._runtime.run_json_control(
            ("container", "inspect", "--format", "{{json .}}", name)
        )
        config = _require_mapping(payload, "Config", "Docker container config")
        container_id = payload.get("Id")
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
            or payload.get("Name") != f"/{name}"
            or config.get("Labels") != dict(expected_labels)
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker container differs from its handle labels"
            )
        return SourceReplayDockerContainerObservation(
            container_id=container_id,
            name=name,
            role=role,
            payload=payload,
        )

    def _observe_volume(
        self,
        identity: SourceReplayDockerResourceIdentity,
    ) -> SourceReplayDockerVolumeObservation | None:
        result = self._runtime.run_control(
            (
                "volume",
                "ls",
                "--filter",
                f"name=^{identity.volume_name}$",
                "--format",
                "{{json .Name}}",
            )
        )
        observed_name = _parse_optional_json_string(result.stdout)
        if observed_name is None:
            return None
        if observed_name != identity.volume_name:
            raise SourceReplayDockerResourceError(
                "source replay Docker volume lookup was not exact"
            )
        payload = self._runtime.run_json_control(
            ("volume", "inspect", "--format", "{{json .}}", identity.volume_name)
        )
        if payload.get("Name") != identity.volume_name or payload.get("Labels") != dict(
            identity.labels_for(_VOLUME_ROLE)
        ):
            raise SourceReplayDockerResourceError(
                "source replay Docker volume differs from its handle labels"
            )
        return SourceReplayDockerVolumeObservation(
            name=identity.volume_name,
            payload=payload,
        )


def _writable_volume_options(
    identity: SourceReplayDockerResourceIdentity,
    compute: ExpertSourceReplayComputeBinding,
    settings: SourceReplayDockerProviderSettings,
) -> dict[str, str]:
    if not identity.volume_name:
        raise SourceReplayDockerResourceError(
            "source replay writable volume identity is empty"
        )
    return {
        "device": "tmpfs",
        "o": (
            f"uid={settings.container_user_id},"
            f"gid={settings.container_group_id},"
            "mode=0700,"
            f"size={compute.writable_storage_byte_limit},"
            f"nr_inodes={compute.writable_inode_limit},"
            "nosuid,nodev,noexec"
        ),
        "type": "tmpfs",
    }


def source_replay_docker_container_observations_match(
    current: SourceReplayDockerContainerObservation | None,
    expected: SourceReplayDockerContainerObservation,
) -> bool:
    if (
        type(current) is not SourceReplayDockerContainerObservation
        or current.container_id != expected.container_id
        or current.name != expected.name
        or current.role != expected.role
    ):
        return False
    current_mounts = _mounts_by_destination(current.payload.get("Mounts"))
    expected_mounts = _mounts_by_destination(expected.payload.get("Mounts"))
    if current_mounts is None or expected_mounts is None:
        return False
    return current_mounts == expected_mounts and {
        key: value for key, value in current.payload.items() if key != "Mounts"
    } == {key: value for key, value in expected.payload.items() if key != "Mounts"}


def _mounts_by_destination(value: Any) -> Mapping[str, Mapping[str, Any]] | None:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    mounts = {item.get("Destination"): item for item in value}
    if len(mounts) != len(value) or any(
        not isinstance(destination, str) or not destination for destination in mounts
    ):
        return None
    return MappingProxyType(mounts)


def _parse_optional_json_string(payload: bytes) -> str | None:
    if payload == b"":
        return None
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise SourceReplayDockerResourceError(
            "source replay Docker resource lookup lacks an exact line ending"
        )
    encoded = payload[:-1]
    if not encoded or b"\n" in encoded or b"\r" in encoded:
        raise SourceReplayDockerResourceError(
            "source replay Docker resource lookup is ambiguous"
        )
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, str) or not decoded:
        raise SourceReplayDockerResourceError(
            "source replay Docker resource lookup is not a name"
        )
    return decoded


def _require_exact_line(payload: bytes, expected: str) -> None:
    if payload != f"{expected}\n".encode():
        raise SourceReplayDockerResourceError(
            "source replay Docker mutation returned an unexpected identity"
        )


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    name: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SourceReplayDockerResourceError(f"{name} is not an object")
    return value
