"""Linux primitives for one anonymous-file, no-replace publication."""

from __future__ import annotations

import ctypes
import os

_AT_EMPTY_PATH = 0x1000
_LIBC = ctypes.CDLL(None, use_errno=True)
_LINK_AT = getattr(_LIBC, "linkat", None)
if _LINK_AT is not None:
    _LINK_AT.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    _LINK_AT.restype = ctypes.c_int


class RunActionAtomicPublicationError(RuntimeError):
    """The host lacks the required anonymous no-replace publication primitive."""


def open_run_action_anonymous_file(
    directory_descriptor: int,
    mode: int,
) -> int:
    """Open one unlinked regular file inside an exact retained directory."""

    if (
        type(directory_descriptor) is not int
        or directory_descriptor < 0
        or type(mode) is not int
        or not 0 < mode <= 0o777
    ):
        raise RunActionAtomicPublicationError(
            "anonymous publication requires an exact directory and mode"
        )
    if not hasattr(os, "O_TMPFILE"):
        raise RunActionAtomicPublicationError(
            "anonymous publication requires O_TMPFILE"
        )
    if _LINK_AT is None:
        raise RunActionAtomicPublicationError(
            "anonymous publication requires linkat with AT_EMPTY_PATH"
        )
    return os.open(
        ".",
        os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC,
        mode,
        dir_fd=directory_descriptor,
    )


def write_run_action_full_payload(descriptor: int, payload: bytes) -> None:
    """Write every byte or fail on the first non-progressing syscall."""

    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(payload) is not bytes
        or not payload
    ):
        raise RunActionAtomicPublicationError(
            "anonymous publication requires one nonempty payload"
        )
    written_size = 0
    while written_size < len(payload):
        written = os.write(descriptor, payload[written_size:])
        if (
            type(written) is not int
            or written <= 0
            or written > len(payload) - written_size
        ):
            raise RunActionAtomicPublicationError(
                "anonymous publication write made no valid progress"
            )
        written_size += written


def require_run_action_descriptor_payload(
    descriptor: int,
    expected_payload: bytes,
) -> None:
    """Read full EOF plus one byte and require byte-exact payload equality."""

    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(expected_payload) is not bytes
        or not expected_payload
    ):
        raise RunActionAtomicPublicationError(
            "anonymous publication payload proof is invalid"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    remaining = len(expected_payload) + 1
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if b"".join(chunks) != expected_payload:
        raise RunActionAtomicPublicationError(
            "publication file bytes differ from the complete bounded payload"
        )


def link_run_action_anonymous_file_no_replace(
    anonymous_descriptor: int,
    directory_descriptor: int,
    final_file_name: str,
) -> None:
    """Make one anonymous inode visible without replacement."""

    if (
        type(anonymous_descriptor) is not int
        or anonymous_descriptor < 0
        or type(directory_descriptor) is not int
        or directory_descriptor < 0
        or not isinstance(final_file_name, str)
        or not final_file_name
        or "/" in final_file_name
        or "\x00" in final_file_name
        or final_file_name in {".", ".."}
    ):
        raise RunActionAtomicPublicationError(
            "anonymous publication link target is invalid"
        )
    if _LINK_AT is None:
        raise RunActionAtomicPublicationError(
            "anonymous publication requires linkat with AT_EMPTY_PATH"
        )
    ctypes.set_errno(0)
    result = _LINK_AT(
        anonymous_descriptor,
        b"",
        directory_descriptor,
        os.fsencode(final_file_name),
        _AT_EMPTY_PATH,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            final_file_name,
        )


__all__ = [
    "RunActionAtomicPublicationError",
    "link_run_action_anonymous_file_no_replace",
    "open_run_action_anonymous_file",
    "require_run_action_descriptor_payload",
    "write_run_action_full_payload",
]
