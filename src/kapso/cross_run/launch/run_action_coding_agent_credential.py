"""Native Codex credential broker with durable non-secret lease identity."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from threading import Lock
from typing import Mapping

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.launch.run_action_coding_agent_production import (
    NATIVE_CODEX_CREDENTIAL_BROKER_ID,
    NATIVE_CODEX_CREDENTIAL_BROKER_PROTOCOL_VERSION,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialBrokerBackend,
    RunActionCredentialIssueResponse,
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionCredentialLeaseRequest,
)
from kapso.cross_run.settings import LaunchSettings

_LEASE_STATE_PROTOCOL_VERSION = "kapso.native_codex_credential_leases.v1"
_NANOSECONDS_PER_SECOND = 1_000_000_000


class NativeCodexCredentialBrokerError(RuntimeError):
    """The configured native Codex credential or lease state is unsafe."""


class NativeCodexCredentialBroker(RunActionCredentialBrokerBackend):
    """Issue exact auth-file bytes while persisting only digest and lease expiry."""

    def __init__(self, *, settings: LaunchSettings, state_root: Path) -> None:
        if (
            type(settings) is not LaunchSettings
            or not isinstance(state_root, Path)
            or not state_root.is_absolute()
            or state_root != Path(os.path.abspath(state_root))
            or state_root in {Path("/"), Path.home()}
        ):
            raise NativeCodexCredentialBrokerError(
                "native Codex credential broker requires an exact private root"
            )
        _require_private_directory(state_root)
        state_path = state_root / settings.coding_agent_credential_lease_state_path
        if state_root not in state_path.parents:
            raise NativeCodexCredentialBrokerError(
                "native Codex credential lease state escapes its private root"
            )
        _require_private_directory(state_path.parent)
        super().__init__(
            broker_id=NATIVE_CODEX_CREDENTIAL_BROKER_ID,
            broker_protocol_version=NATIVE_CODEX_CREDENTIAL_BROKER_PROTOCOL_VERSION,
        )
        self._settings = settings
        self._credential_path = Path(settings.coding_agent_codex_auth_source_path)
        self._state_path = state_path
        self._lock = Lock()

    def issue_or_replay_exact(
        self,
        request: RunActionCredentialLeaseRequest,
    ) -> RunActionCredentialIssueResponse:
        """Return current configured bytes under the request's durable lease."""

        self._require_request(request)
        with self._lock:
            payload = _read_private_credential(
                self._credential_path,
                self._settings.coding_agent_native_credential_size_bytes,
            )
            digest = tree_or_blob_digest(payload)
            entries = self._read_entries()
            entry = entries.get(request.credential_lease_request_id)
            if entry is None:
                if (
                    len(entries)
                    >= self._settings.coding_agent_credential_lease_state_entry_limit
                ):
                    raise NativeCodexCredentialBrokerError(
                        "native Codex credential lease state is full"
                    )
                valid_until = (
                    time.time_ns()
                    + (self._settings.coding_agent_action_credential_lease_seconds - 1)
                    * _NANOSECONDS_PER_SECOND
                )
                if valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER:
                    raise NativeCodexCredentialBrokerError(
                        "native Codex credential lease exceeds physical time"
                    )
                entries[request.credential_lease_request_id] = {
                    "credential_digest": digest,
                    "valid_until_realtime_nanoseconds": valid_until,
                }
                self._write_entries(entries)
            else:
                valid_until = entry["valid_until_realtime_nanoseconds"]
                if entry["credential_digest"] != digest:
                    raise NativeCodexCredentialBrokerError(
                        "native Codex credential changed during an exact lease"
                    )
        return RunActionCredentialIssueResponse(
            credential_lease_request_id=request.credential_lease_request_id,
            payload=payload,
            valid_until_realtime_nanoseconds=valid_until,
        )

    def observe_exact(
        self,
        request: RunActionCredentialLeaseRequest,
    ) -> RunActionCredentialLeaseStatus:
        """Observe the previously persisted non-secret lease expiry."""

        self._require_request(request)
        with self._lock:
            entries = self._read_entries()
            entry = entries.get(request.credential_lease_request_id)
            if entry is None:
                raise NativeCodexCredentialBrokerError(
                    "native Codex credential lease was never issued"
                )
            payload = _read_private_credential(
                self._credential_path,
                self._settings.coding_agent_native_credential_size_bytes,
            )
            if entry["credential_digest"] != tree_or_blob_digest(payload):
                raise NativeCodexCredentialBrokerError(
                    "native Codex credential changed during an exact lease"
                )
        return RunActionCredentialLeaseStatus.mint(
            credential_lease_request_id=request.credential_lease_request_id,
            valid_until_realtime_nanoseconds=(
                entry["valid_until_realtime_nanoseconds"]
            ),
        )

    def _require_request(self, request: RunActionCredentialLeaseRequest) -> None:
        if (
            type(request) is not RunActionCredentialLeaseRequest
            or request.credential_policy.broker_id != self.broker_id
            or request.credential_policy.broker_protocol_version
            != self.broker_protocol_version
            or request.credential_policy.maximum_lease_seconds
            != self._settings.coding_agent_action_credential_lease_seconds
            or request.credential_policy.maximum_delivery_size_bytes
            != self._settings.coding_agent_native_credential_size_bytes
        ):
            raise NativeCodexCredentialBrokerError(
                "native Codex credential request differs from configured authority"
            )

    def _read_entries(self) -> dict[str, dict[str, int | str]]:
        if not self._state_path.exists():
            return {}
        _require_private_regular_file(
            self._state_path,
            maximum_bytes=(
                self._settings.coding_agent_credential_lease_state_size_bytes
            ),
        )
        payload = self._state_path.read_bytes()
        decoded = json.loads(payload)
        if (
            not isinstance(decoded, Mapping)
            or set(decoded) != {"entries", "protocol_version"}
            or decoded["protocol_version"] != _LEASE_STATE_PROTOCOL_VERSION
            or not isinstance(decoded["entries"], Mapping)
            or len(decoded["entries"])
            > self._settings.coding_agent_credential_lease_state_entry_limit
            or canonical_json_bytes(decoded) != payload
        ):
            raise NativeCodexCredentialBrokerError(
                "native Codex credential lease state is malformed"
            )
        entries = {}
        for request_id, entry in decoded["entries"].items():
            if (
                not isinstance(request_id, str)
                or not isinstance(entry, Mapping)
                or set(entry)
                != {
                    "credential_digest",
                    "valid_until_realtime_nanoseconds",
                }
                or not isinstance(entry["credential_digest"], str)
                or not isinstance(entry["valid_until_realtime_nanoseconds"], int)
                or entry["valid_until_realtime_nanoseconds"] <= 0
                or entry["valid_until_realtime_nanoseconds"]
                > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            ):
                raise NativeCodexCredentialBrokerError(
                    "native Codex credential lease entry is malformed"
                )
            entries[request_id] = {
                "credential_digest": entry["credential_digest"],
                "valid_until_realtime_nanoseconds": (
                    entry["valid_until_realtime_nanoseconds"]
                ),
            }
        return entries

    def _write_entries(self, entries: Mapping[str, Mapping[str, int | str]]) -> None:
        payload = canonical_json_bytes(
            {
                "entries": entries,
                "protocol_version": _LEASE_STATE_PROTOCOL_VERSION,
            }
        )
        if len(payload) > self._settings.coding_agent_credential_lease_state_size_bytes:
            raise NativeCodexCredentialBrokerError(
                "native Codex credential lease state exceeds its byte bound"
            )
        temporary = self._state_path.with_name(
            f".{self._state_path.name}.{os.getpid()}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise NativeCodexCredentialBrokerError(
                    "native Codex credential lease write made no progress"
                )
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        os.replace(temporary, self._state_path)
        parent_descriptor = os.open(
            self._state_path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(parent_descriptor)
        os.close(parent_descriptor)
        _require_private_regular_file(
            self._state_path,
            maximum_bytes=(
                self._settings.coding_agent_credential_lease_state_size_bytes
            ),
        )


def _read_private_credential(path: Path, maximum_bytes: int) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or not 0 < metadata.st_size <= maximum_bytes
    ):
        os.close(descriptor)
        raise NativeCodexCredentialBrokerError(
            "native Codex credential source is not a private regular file"
        )
    payload = bytearray()
    while len(payload) < metadata.st_size:
        chunk = os.read(descriptor, metadata.st_size - len(payload))
        if not chunk:
            os.close(descriptor)
            raise NativeCodexCredentialBrokerError(
                "native Codex credential read made no progress"
            )
        payload.extend(chunk)
    observed = os.fstat(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(observed) != identity(metadata):
        os.close(descriptor)
        raise NativeCodexCredentialBrokerError(
            "native Codex credential changed while reading"
        )
    os.close(descriptor)
    return bytes(payload)


def _require_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    metadata = os.stat(path, follow_symlinks=False)
    if (
        path.resolve() != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise NativeCodexCredentialBrokerError(
            "native Codex credential state directory is unsafe"
        )


def _require_private_regular_file(path: Path, *, maximum_bytes: int) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= maximum_bytes
    ):
        raise NativeCodexCredentialBrokerError(
            "native Codex credential lease state file is unsafe"
        )


__all__ = [
    "NativeCodexCredentialBroker",
    "NativeCodexCredentialBrokerError",
]
