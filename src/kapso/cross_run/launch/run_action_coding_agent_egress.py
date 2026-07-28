"""Process-bound host egress broker authority for native coding-agent actions."""

from __future__ import annotations

import errno
import os
import select
import signal
import socket
import stat
import subprocess
import sys
from pathlib import Path

from kapso.cross_run.settings import LaunchSettings


class NativeCodingAgentEgressError(RuntimeError):
    """The configured coding-agent egress broker could not start exactly."""


class NativeCodingAgentEgressBroker:
    """Own one authenticated-by-mount Unix broker occurrence."""

    def __init__(
        self,
        *,
        settings: LaunchSettings,
        state_root: Path,
    ) -> None:
        if (
            type(settings) is not LaunchSettings
            or not isinstance(state_root, Path)
            or not state_root.is_absolute()
            or state_root != Path(os.path.abspath(state_root))
            or state_root in {Path("/"), Path.home()}
        ):
            raise NativeCodingAgentEgressError(
                "coding-agent egress broker requires an exact private root"
            )
        socket_path = state_root / settings.coding_agent_egress_broker_socket_path
        if state_root not in socket_path.parents or len(os.fsencode(socket_path)) > 107:
            raise NativeCodingAgentEgressError(
                "coding-agent egress broker socket escapes its root or Unix bound"
            )
        _require_private_directory(socket_path.parent)
        _remove_proven_stale_socket(socket_path)
        readiness_read_descriptor, readiness_write_descriptor = os.pipe2(os.O_CLOEXEC)
        command = [
            sys.executable,
            "-m",
            "kapso.cross_run.launch.run_action_coding_agent_egress_broker",
            "--socket-path",
            socket_path.as_posix(),
        ]
        for authority in settings.coding_agent_egress_connect_authorities:
            command.extend(("--authority", authority))
        command.extend(
            (
                "--maximum-header-bytes",
                str(settings.coding_agent_egress_connect_header_size_bytes),
                "--backlog",
                str(settings.coding_agent_egress_relay_backlog),
                "--chunk-size-bytes",
                str(settings.coding_agent_egress_relay_chunk_size_bytes),
                "--connect-timeout-seconds",
                str(settings.coding_agent_egress_connect_timeout_seconds),
                "--readiness-descriptor",
                str(readiness_write_descriptor),
            )
        )
        process = subprocess.Popen(
            tuple(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(readiness_write_descriptor,),
        )
        os.close(readiness_write_descriptor)
        readable, _writable, _exceptional = select.select(
            (readiness_read_descriptor,),
            (),
            (),
            settings.coding_agent_egress_connect_timeout_seconds,
        )
        ready = (
            readable == [readiness_read_descriptor]
            and os.read(readiness_read_descriptor, 1) == b"\x01"
            and process.poll() is None
        )
        os.close(readiness_read_descriptor)
        if not ready:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise NativeCodingAgentEgressError(
                "coding-agent egress broker did not become ready exactly"
            )
        _require_private_socket(socket_path)
        self._settings = settings
        self._socket_path = socket_path
        self._process = process
        self._owner_process_id = os.getpid()
        self._closed = False

    @property
    def socket_path(self) -> Path:
        """Return the exact path admitted into the Docker execution policy."""

        self.require_current()
        return self._socket_path

    def require_current(self) -> None:
        """Prove process ownership, liveness, and socket identity."""

        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._process.poll() is not None
        ):
            raise NativeCodingAgentEgressError(
                "coding-agent egress broker is closed, cloned, or terminal"
            )
        _require_private_socket(self._socket_path)

    def close(self) -> None:
        """Terminate the complete broker group and remove its now-stale socket."""

        if self._closed:
            return
        if self._owner_process_id != os.getpid():
            raise NativeCodingAgentEgressError(
                "coding-agent egress broker cannot close from another process"
            )
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGKILL)
        self._process.wait()
        _require_private_socket(self._socket_path)
        self._socket_path.unlink()
        self._closed = True

    def __enter__(self) -> NativeCodingAgentEgressBroker:
        self.require_current()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def _remove_proven_stale_socket(path: Path) -> None:
    if not path.exists():
        return
    _require_private_socket(path)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    result = probe.connect_ex(path.as_posix())
    probe.close()
    if result == 0:
        raise NativeCodingAgentEgressError(
            "coding-agent egress broker socket is already active"
        )
    if result not in {errno.ECONNREFUSED, errno.ENOENT}:
        raise NativeCodingAgentEgressError(
            "coding-agent egress broker staleness is not proven"
        )
    if path.exists():
        _require_private_socket(path)
        path.unlink()


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
        raise NativeCodingAgentEgressError(
            "coding-agent egress broker directory is unsafe"
        )


def _require_private_socket(path: Path) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise NativeCodingAgentEgressError(
            "coding-agent egress broker is not an owned private Unix socket"
        )


__all__ = [
    "NativeCodingAgentEgressBroker",
    "NativeCodingAgentEgressError",
]
