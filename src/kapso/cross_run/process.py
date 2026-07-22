"""Typed bounded subprocess execution for isolated cross-run providers."""

from __future__ import annotations

import ctypes
import errno
import io
import math
import selectors
import signal
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Thread
from types import MappingProxyType
from typing import Any, Mapping

_LIBC_KILL = ctypes.CDLL(None, use_errno=True).kill
_LIBC_KILL.argtypes = (ctypes.c_int, ctypes.c_int)
_LIBC_KILL.restype = ctypes.c_int


class BoundedProcessError(RuntimeError):
    """A bounded process request or completion state is invalid."""


class BoundedProcessOutcome(str, Enum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    STDOUT_LIMIT_EXCEEDED = "stdout_limit_exceeded"
    STDERR_LIMIT_EXCEEDED = "stderr_limit_exceeded"


@dataclass(frozen=True)
class BoundedProcessRequest:
    argv: tuple[str, ...]
    trusted_root: Path
    cwd: Path
    timeout_seconds: int
    cleanup_timeout_seconds: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    environment: Mapping[str, str]
    stdin: bytes | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(
                not isinstance(argument, str) or not argument or "\x00" in argument
                for argument in self.argv
            )
        ):
            raise BoundedProcessError("process argv must contain non-empty strings")
        if (
            not isinstance(self.trusted_root, Path)
            or not self.trusted_root.is_absolute()
            or not self.trusted_root.is_dir()
            or self.trusted_root.resolve() != self.trusted_root
        ):
            raise BoundedProcessError(
                "process trusted_root must be a resolved existing directory"
            )
        if (
            not isinstance(self.cwd, Path)
            or not self.cwd.is_absolute()
            or not self.cwd.is_dir()
            or self.cwd.resolve() != self.cwd
            or not self.cwd.is_relative_to(self.trusted_root)
        ):
            raise BoundedProcessError(
                "process cwd must be a resolved directory under trusted_root"
            )
        for value, name in (
            (self.timeout_seconds, "timeout_seconds"),
            (self.cleanup_timeout_seconds, "cleanup_timeout_seconds"),
            (self.stdout_byte_limit, "stdout_byte_limit"),
            (self.stderr_byte_limit, "stderr_byte_limit"),
        ):
            if type(value) is not int or value <= 0:
                raise BoundedProcessError(f"process {name} must be a positive integer")
        if self.stdin is not None and not isinstance(self.stdin, bytes):
            raise BoundedProcessError("process stdin must be bytes")
        if not isinstance(self.environment, Mapping) or any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
            for key, value in self.environment.items()
        ):
            raise BoundedProcessError("process environment must contain valid strings")
        for path, name in (
            (self.stdout_path, "stdout_path"),
            (self.stderr_path, "stderr_path"),
        ):
            if path is not None and (
                not isinstance(path, Path)
                or not path.is_absolute()
                or not path.parent.is_dir()
                or not path.parent.resolve().is_relative_to(self.trusted_root)
                or path.exists()
            ):
                raise BoundedProcessError(
                    f"process {name} must be an absent absolute path under an existing directory"
                )
        if self.stdout_path is not None and self.stdout_path == self.stderr_path:
            raise BoundedProcessError("process stream output paths must differ")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(self.environment)),
        )


@dataclass(frozen=True)
class BoundedProcessResult:
    request: BoundedProcessRequest
    outcome: BoundedProcessOutcome
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_bytes_observed: int
    stderr_bytes_observed: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise BoundedProcessError("process returncode must be an integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise BoundedProcessError("process streams must be bytes")
        if (
            type(self.stdout_bytes_observed) is not int
            or self.stdout_bytes_observed < 0
            or type(self.stderr_bytes_observed) is not int
            or self.stderr_bytes_observed < 0
            or not isinstance(self.duration_seconds, float)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0.0
        ):
            raise BoundedProcessError("process observations are invalid")


class BoundedProcessRunner:
    """Execute fixed argv without a shell and bound each stream independently."""

    def run(self, request: BoundedProcessRequest) -> BoundedProcessResult:
        started_at = time.monotonic()
        with ExitStack() as stack:
            stdin_handle = stack.enter_context(tempfile.TemporaryFile())
            stdout_handle = stack.enter_context(
                request.stdout_path.open("xb")
                if request.stdout_path is not None
                else io.BytesIO()
            )
            stderr_handle = stack.enter_context(
                request.stderr_path.open("xb")
                if request.stderr_path is not None
                else io.BytesIO()
            )
            if request.stdin is not None:
                stdin_handle.write(request.stdin)
                stdin_handle.seek(0)
            outcome, returncode, observations = self._execute(
                request,
                stdin_handle,
                stdout_handle,
                stderr_handle,
            )
            stdout = (
                b"" if request.stdout_path is not None else stdout_handle.getvalue()
            )
            stderr = (
                b"" if request.stderr_path is not None else stderr_handle.getvalue()
            )
        return BoundedProcessResult(
            request=request,
            outcome=outcome,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_bytes_observed=observations["stdout"],
            stderr_bytes_observed=observations["stderr"],
            duration_seconds=float(time.monotonic() - started_at),
        )

    def _execute(
        self,
        request: BoundedProcessRequest,
        stdin: Any,
        stdout: Any,
        stderr: Any,
    ) -> tuple[BoundedProcessOutcome, int, dict[str, int]]:
        process = subprocess.Popen(
            list(request.argv),
            cwd=request.cwd,
            stdin=stdin if request.stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(request.environment),
            shell=False,
            start_new_session=True,
        )
        with ExitStack() as process_owner:
            ownership_state: dict[str, Thread] = {}
            process_owner.callback(
                self._cleanup_owned_process,
                process,
                request.cleanup_timeout_seconds,
                ownership_state,
            )
            process_completed = Event()
            process_waiter = Thread(
                target=self._wait_for_process,
                args=(process, process_completed),
                daemon=True,
            )
            process_waiter.start()
            ownership_state["waiter"] = process_waiter
            if process.stdout is None or process.stderr is None:
                raise BoundedProcessError("process pipes were not created")
            streams = {
                process.stdout: (stdout, "stdout", request.stdout_byte_limit),
                process.stderr: (stderr, "stderr", request.stderr_byte_limit),
            }
            observations = {"stdout": 0, "stderr": 0}
            deadline = time.monotonic() + request.timeout_seconds
            terminal_outcome = None
            with selectors.DefaultSelector() as selector:
                for stream in streams:
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map() and terminal_outcome is None:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        terminal_outcome = BoundedProcessOutcome.TIMED_OUT
                        continue
                    events = selector.select(remaining_seconds)
                    if not events:
                        terminal_outcome = BoundedProcessOutcome.TIMED_OUT
                        continue
                    for key, _ in events:
                        stream = key.fileobj
                        destination, label, byte_limit = streams[stream]
                        chunk = stream.read1(io.DEFAULT_BUFFER_SIZE)
                        if not chunk:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        available = max(byte_limit - observations[label], 0)
                        destination.write(chunk[:available])
                        observations[label] += len(chunk)
                        if observations[label] > byte_limit:
                            terminal_outcome = (
                                BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED
                                if label == "stdout"
                                else BoundedProcessOutcome.STDERR_LIMIT_EXCEEDED
                            )
                            break
            if terminal_outcome is not None:
                self._terminate_process(
                    process,
                    process_completed,
                    process_waiter,
                    request.cleanup_timeout_seconds,
                )
                process_owner.pop_all()
                if process.returncode is None:
                    raise BoundedProcessError("terminated process has no returncode")
                return terminal_outcome, process.returncode, observations
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0 or not process_completed.wait(remaining_seconds):
                self._terminate_process(
                    process,
                    process_completed,
                    process_waiter,
                    request.cleanup_timeout_seconds,
                )
                outcome = BoundedProcessOutcome.TIMED_OUT
            else:
                process_waiter.join(request.cleanup_timeout_seconds)
                if process_waiter.is_alive():
                    raise BoundedProcessError("process waiter did not terminate")
                outcome = BoundedProcessOutcome.COMPLETED
            self._kill_process_group(process.pid)
            process_owner.pop_all()
            if process.returncode is None:
                raise BoundedProcessError("completed process has no returncode")
            return outcome, process.returncode, observations

    @staticmethod
    def _wait_for_process(process: subprocess.Popen, completed: Event) -> None:
        process.wait()
        completed.set()

    @staticmethod
    def _terminate_process(
        process: subprocess.Popen,
        completed: Event,
        waiter: Thread,
        cleanup_timeout_seconds: int,
    ) -> None:
        BoundedProcessRunner._kill_process_group(process.pid)
        deadline = time.monotonic() + cleanup_timeout_seconds
        if not completed.wait(cleanup_timeout_seconds):
            raise BoundedProcessError("terminated process did not complete")
        waiter.join(max(deadline - time.monotonic(), 0.0))
        if waiter.is_alive():
            raise BoundedProcessError("terminated process waiter did not terminate")

    @staticmethod
    def _cleanup_owned_process(
        process: subprocess.Popen,
        cleanup_timeout_seconds: int,
        ownership_state: dict[str, Thread],
    ) -> None:
        BoundedProcessRunner._kill_process_group(process.pid)
        process.wait(timeout=cleanup_timeout_seconds)
        waiter = ownership_state.get("waiter")
        if waiter is not None:
            waiter.join(cleanup_timeout_seconds)
            if waiter.is_alive():
                raise BoundedProcessError("owned process waiter did not terminate")

    @staticmethod
    def _kill_process_group(process_group_id: int) -> None:
        ctypes.set_errno(0)
        result = _LIBC_KILL(-process_group_id, int(signal.SIGKILL))
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number != errno.ESRCH:
                raise OSError(
                    error_number,
                    "process-group termination failed",
                )
