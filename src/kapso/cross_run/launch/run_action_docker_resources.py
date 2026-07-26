"""Race-bound inventory for Docker resources owned by one preparation allocation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping
from weakref import WeakKeyDictionary

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    PinnedDockerObservationAuthority,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionContainerLabel,
    RunActionPreparationAllocation,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
)
from kapso.cross_run.settings import DockerRuntimeSettings

_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_RESOURCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_ACTION_RESOURCE_MANAGER_LOCK = Lock()
_RUN_ACTION_RESOURCE_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionResourceManager, PinnedDockerObservationAuthority
] = WeakKeyDictionary()


class DockerRunActionResourceError(RuntimeError):
    """A deterministic action resource is ambiguous, substituted, or changed."""


@dataclass(frozen=True)
class DockerRunActionResourceInventory:
    """Stable two-scan inventory for one allocation's three deterministic names."""

    preparation_allocation: RunActionPreparationAllocation
    volume_inspection_digest: str | None
    keeper_container_id: str | None
    main_container_id: str | None

    def __post_init__(self) -> None:
        if (
            type(self.preparation_allocation) is not RunActionPreparationAllocation
            or (
                self.volume_inspection_digest is not None
                and (
                    type(self.volume_inspection_digest) is not str
                    or _SHA256_DIGEST_PATTERN.fullmatch(self.volume_inspection_digest)
                    is None
                )
            )
            or (
                self.keeper_container_id is not None
                and (
                    type(self.keeper_container_id) is not str
                    or _CONTAINER_ID_PATTERN.fullmatch(self.keeper_container_id) is None
                )
            )
            or (
                self.main_container_id is not None
                and (
                    type(self.main_container_id) is not str
                    or _CONTAINER_ID_PATTERN.fullmatch(self.main_container_id) is None
                )
            )
            or (
                self.keeper_container_id is not None
                and self.keeper_container_id == self.main_container_id
            )
        ):
            raise DockerRunActionResourceError(
                "Docker run-action inventory has invalid resource identities"
            )

    @property
    def is_absent(self) -> bool:
        return (
            self.volume_inspection_digest is None
            and self.keeper_container_id is None
            and self.main_container_id is None
        )

    @property
    def volume_present(self) -> bool:
        return self.volume_inspection_digest is not None


class DockerRunActionResourceManager:
    """Discover and rebind only exact name-and-label-owned Docker resources."""

    def __init__(self, runtime: PinnedDockerRuntime) -> None:
        if type(runtime) is not PinnedDockerRuntime:
            raise DockerRunActionResourceError(
                "Docker run-action resources require the pinned runtime"
            )
        observation_authority = runtime.issue_observation_authority()
        with _RUN_ACTION_RESOURCE_MANAGER_LOCK:
            if _RUN_ACTION_RESOURCE_MANAGER_AUTHORITIES.get(self) is not None:
                raise DockerRunActionResourceError(
                    "Docker run-action resource manager is already issued"
                )
            _RUN_ACTION_RESOURCE_MANAGER_AUTHORITIES[self] = observation_authority

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return the immutable settings joined to this observation authority."""

        return _run_action_observation_authority(self).settings

    def observe(
        self,
        allocation: RunActionPreparationAllocation,
    ) -> DockerRunActionResourceInventory:
        """Require two identical full inventories around exact ID inspections."""

        if type(allocation) is not RunActionPreparationAllocation:
            raise DockerRunActionResourceError(
                "Docker run-action inventory requires an exact preparation allocation"
            )
        _run_action_observation_authority(self).require_live_authority()
        first = self._scan_once(allocation)
        second = self._scan_once(allocation)
        if first != second:
            raise DockerRunActionResourceError(
                "Docker run-action resources changed during inventory"
            )
        return second

    def inspect_volume(
        self,
        inventory: DockerRunActionResourceInventory,
    ) -> Mapping[str, Any]:
        """Rebind the complete inventory before inspecting its exact volume name."""

        self._require_current_inventory(inventory)
        if not inventory.volume_present:
            raise DockerRunActionResourceError(
                "Docker run-action inventory has no volume to inspect"
            )
        allocation = inventory.preparation_allocation
        claim = allocation.preparation_claim
        authority = allocation.runtime_volume_authority
        payload = self._inspect_volume_identity(
            preparation_volume_name(claim),
            preparation_volume_labels(claim, authority.generation_nonce),
        )
        if _inspection_digest(payload) != inventory.volume_inspection_digest:
            raise DockerRunActionResourceError(
                "Docker run-action volume changed after inventory"
            )
        return payload

    def inspect_keeper(
        self,
        inventory: DockerRunActionResourceInventory,
    ) -> Mapping[str, Any]:
        """Rebind the complete inventory before inspecting the keeper by ID."""

        self._require_current_inventory(inventory)
        if inventory.keeper_container_id is None:
            raise DockerRunActionResourceError(
                "Docker run-action inventory has no keeper to inspect"
            )
        claim = inventory.preparation_allocation.preparation_claim
        return self._inspect_container_identity(
            inventory.keeper_container_id,
            preparation_keeper_container_name(claim),
            preparation_keeper_container_labels(claim),
        )

    def inspect_main(
        self,
        inventory: DockerRunActionResourceInventory,
    ) -> Mapping[str, Any]:
        """Rebind the complete inventory before inspecting the main by ID."""

        self._require_current_inventory(inventory)
        if inventory.main_container_id is None:
            raise DockerRunActionResourceError(
                "Docker run-action inventory has no main container to inspect"
            )
        claim = inventory.preparation_allocation.preparation_claim
        return self._inspect_container_identity(
            inventory.main_container_id,
            preparation_container_name(claim),
            preparation_container_labels(claim),
        )

    def _require_current_inventory(
        self,
        inventory: DockerRunActionResourceInventory,
    ) -> None:
        if type(inventory) is not DockerRunActionResourceInventory:
            raise DockerRunActionResourceError(
                "Docker run-action inspection requires an exact inventory"
            )
        if self.observe(inventory.preparation_allocation) != inventory:
            raise DockerRunActionResourceError(
                "Docker run-action inventory changed before inspection"
            )

    def _scan_once(
        self,
        allocation: RunActionPreparationAllocation,
    ) -> DockerRunActionResourceInventory:
        claim = allocation.preparation_claim
        authority = allocation.runtime_volume_authority
        volume_inspection_digest = self._observe_volume(
            preparation_volume_name(claim),
            preparation_volume_labels(claim, authority.generation_nonce),
        )
        keeper_container_id = self._observe_container_id(
            preparation_keeper_container_name(claim),
            preparation_keeper_container_labels(claim),
        )
        main_container_id = self._observe_container_id(
            preparation_container_name(claim),
            preparation_container_labels(claim),
        )
        return DockerRunActionResourceInventory(
            preparation_allocation=allocation,
            volume_inspection_digest=volume_inspection_digest,
            keeper_container_id=keeper_container_id,
            main_container_id=main_container_id,
        )

    def _observe_container_id(
        self,
        name: str,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> str | None:
        named_ids = self._list_container_ids(name, ())
        named_labeled_ids = self._list_container_ids(name, labels)
        labeled_ids = self._list_container_ids(None, labels)
        if not named_ids and not named_labeled_ids and not labeled_ids:
            return None
        if (
            len(named_ids) != 1
            or named_labeled_ids != named_ids
            or labeled_ids != named_ids
        ):
            raise DockerRunActionResourceError(
                "Docker run-action container name or labels are conflicted"
            )
        container_id = named_ids[0]
        self._inspect_container_identity(container_id, name, labels)
        return container_id

    def _observe_volume(
        self,
        name: str,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> str | None:
        named_volumes = self._list_volume_names(name, ())
        named_labeled_volumes = self._list_volume_names(name, labels)
        labeled_volumes = self._list_volume_names(None, labels)
        if not named_volumes and not named_labeled_volumes and not labeled_volumes:
            return None
        if (
            named_volumes != (name,)
            or named_labeled_volumes != named_volumes
            or labeled_volumes != named_volumes
        ):
            raise DockerRunActionResourceError(
                "Docker run-action volume name or labels are conflicted"
            )
        return _inspection_digest(self._inspect_volume_identity(name, labels))

    def _list_container_ids(
        self,
        name: str | None,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> tuple[str, ...]:
        arguments = [
            "container",
            "ls",
            "--all",
            "--no-trunc",
        ]
        if name is not None:
            arguments.extend(("--filter", f"name=^/{name}$"))
        _append_label_filters(arguments, labels)
        arguments.extend(("--format", "{{json .ID}}"))
        return _parse_json_string_lines(
            _run_action_observation_authority(self)
            .run_control(tuple(arguments))
            .stdout,
            _CONTAINER_ID_PATTERN,
            "Docker run-action container lookup",
        )

    def _list_volume_names(
        self,
        name: str | None,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> tuple[str, ...]:
        arguments = [
            "volume",
            "ls",
        ]
        if name is not None:
            arguments.extend(("--filter", f"name=^{name}$"))
        _append_label_filters(arguments, labels)
        arguments.extend(("--format", "{{json .Name}}"))
        return _parse_json_string_lines(
            _run_action_observation_authority(self)
            .run_control(tuple(arguments))
            .stdout,
            (
                re.compile(re.escape(name))
                if name is not None
                else _DOCKER_RESOURCE_NAME_PATTERN
            ),
            "Docker run-action volume lookup",
        )

    def _inspect_container_identity(
        self,
        container_id: str,
        name: str,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> Mapping[str, Any]:
        payload = _run_action_observation_authority(self).run_json_control(
            ("container", "inspect", "--format", "{{json .}}", container_id)
        )
        config = payload.get("Config")
        if (
            not isinstance(config, Mapping)
            or payload.get("Id") != container_id
            or payload.get("Name") != f"/{name}"
            or config.get("Labels") != _label_mapping(labels)
        ):
            raise DockerRunActionResourceError(
                "Docker run-action container differs from exact name and labels"
            )
        return payload

    def _inspect_volume_identity(
        self,
        name: str,
        labels: tuple[RunActionContainerLabel, ...],
    ) -> Mapping[str, Any]:
        payload = _run_action_observation_authority(self).run_json_control(
            ("volume", "inspect", "--format", "{{json .}}", name)
        )
        if payload.get("Name") != name or payload.get("Labels") != _label_mapping(
            labels
        ):
            raise DockerRunActionResourceError(
                "Docker run-action volume differs from exact name and labels"
            )
        return payload


def _run_action_observation_authority(
    manager: DockerRunActionResourceManager,
) -> PinnedDockerObservationAuthority:
    with _RUN_ACTION_RESOURCE_MANAGER_LOCK:
        authority = _RUN_ACTION_RESOURCE_MANAGER_AUTHORITIES.get(manager)
    if (
        type(manager) is not DockerRunActionResourceManager
        or type(authority) is not PinnedDockerObservationAuthority
    ):
        raise DockerRunActionResourceError(
            "Docker run-action resource manager is unissued or foreign"
        )
    return authority


def _append_label_filters(
    arguments: list[str],
    labels: tuple[RunActionContainerLabel, ...],
) -> None:
    for label in labels:
        if type(label) is not RunActionContainerLabel:
            raise DockerRunActionResourceError(
                "Docker run-action label filter is malformed"
            )
        arguments.extend(("--filter", f"label={label.key}={label.value}"))


def _label_mapping(
    labels: tuple[RunActionContainerLabel, ...],
) -> dict[str, str]:
    return {label.key: label.value for label in labels}


def _inspection_digest(payload: Mapping[str, Any]) -> str:
    return tree_or_blob_digest(canonical_json_bytes(payload))


def _parse_json_string_lines(
    payload: bytes,
    pattern: re.Pattern[str],
    name: str,
) -> tuple[str, ...]:
    if type(payload) is not bytes:
        raise DockerRunActionResourceError(f"{name} is not bytes")
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise DockerRunActionResourceError(f"{name} lacks a final line ending")
    encoded_lines = payload.splitlines()
    if not encoded_lines or any(not line or b"\r" in line for line in encoded_lines):
        raise DockerRunActionResourceError(f"{name} has malformed lines")
    values = tuple(json.loads(line.decode("ascii")) for line in encoded_lines)
    if any(
        type(value) is not str or pattern.fullmatch(value) is None for value in values
    ) or len(values) != len(set(values)):
        raise DockerRunActionResourceError(
            f"{name} contains malformed or duplicate identities"
        )
    return values


__all__ = [
    "DockerRunActionResourceError",
    "DockerRunActionResourceInventory",
    "DockerRunActionResourceManager",
]
