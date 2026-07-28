"""Privilege erasure shared by trusted in-container coding-agent sidecars."""

from __future__ import annotations

import ctypes
import errno
import os
import signal

_PR_SET_PDEATHSIG = 1
_PR_GET_NO_NEW_PRIVS = 39
_PR_CAPBSET_DROP = 24
_PR_CAPBSET_READ = 23
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_IS_SET = 1
_TRANSITION_CAPABILITY_NUMBERS = (5, 6, 7, 8)
_LINUX_CAPABILITY_VERSION_3 = 0x20080522
_LINUX_CAPABILITY_SCAN_LIMIT = 64


class RunActionCodingAgentTrustedChildError(RuntimeError):
    """A trusted child retained authority outside its fixed role."""


class _UserCapabilityHeader(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_uint32),
        ("process_id", ctypes.c_int),
    )


class _UserCapabilityData(ctypes.Structure):
    _fields_ = (
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    )


def bind_trusted_parent_death_signal(expected_parent_process_id: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_process_id:
        raise RunActionCodingAgentTrustedChildError(
            "coding-agent trusted child parent changed during containment"
        )


def erase_trusted_child_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    for capability_number in _TRANSITION_CAPABILITY_NUMBERS:
        if libc.prctl(_PR_CAPBSET_DROP, capability_number, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
    header = _UserCapabilityHeader(
        version=_LINUX_CAPABILITY_VERSION_3,
        process_id=0,
    )
    capability_data = (_UserCapabilityData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(capability_data)) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def require_zero_linux_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    header = _UserCapabilityHeader(
        version=_LINUX_CAPABILITY_VERSION_3,
        process_id=0,
    )
    capability_data = (_UserCapabilityData * 2)()
    if libc.capget(ctypes.byref(header), ctypes.byref(capability_data)) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if any(
        value != 0
        for item in capability_data
        for value in (item.effective, item.permitted, item.inheritable)
    ):
        raise RunActionCodingAgentTrustedChildError(
            "coding-agent child retained process capabilities"
        )
    for capability_number in range(_LINUX_CAPABILITY_SCAN_LIMIT):
        ctypes.set_errno(0)
        bounding = libc.prctl(
            _PR_CAPBSET_READ,
            capability_number,
            0,
            0,
            0,
        )
        bounding_error = ctypes.get_errno()
        if bounding == 1:
            raise RunActionCodingAgentTrustedChildError(
                "coding-agent child retained a bounding capability"
            )
        if bounding == -1 and bounding_error != errno.EINVAL:
            raise OSError(bounding_error, os.strerror(bounding_error))
        ctypes.set_errno(0)
        ambient = libc.prctl(
            _PR_CAP_AMBIENT,
            _PR_CAP_AMBIENT_IS_SET,
            capability_number,
            0,
            0,
        )
        ambient_error = ctypes.get_errno()
        if ambient == 1:
            raise RunActionCodingAgentTrustedChildError(
                "coding-agent child retained an ambient capability"
            )
        if ambient == -1 and ambient_error != errno.EINVAL:
            raise OSError(ambient_error, os.strerror(ambient_error))


def require_no_new_privileges() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    observed = libc.prctl(_PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0)
    if observed != 1:
        if observed == -1:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        raise RunActionCodingAgentTrustedChildError(
            "coding-agent child lacks no-new-privileges"
        )


def require_unprivileged_supervisor_child(
    *,
    supervisor_user_id: int,
    supervisor_group_id: int,
    provider_group_id: int,
) -> None:
    if (
        os.getresuid() != (supervisor_user_id,) * 3
        or os.getresgid() != (supervisor_group_id,) * 3
        or os.getgroups() != sorted({supervisor_group_id, provider_group_id})
    ):
        raise RunActionCodingAgentTrustedChildError(
            "coding-agent trusted child identity is not exact"
        )
    require_zero_linux_capabilities()
