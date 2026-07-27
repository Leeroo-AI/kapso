"""Trusted loopback-to-Unix relay for one network-isolated coding agent."""

from __future__ import annotations

import argparse
import ctypes
import os
import select
import signal
import socket

from kapso.cross_run.launch.run_action_coding_agent_layout import (
    PROVIDER_EGRESS_BROKER_PATH,
)

_PR_SET_PDEATHSIG = 1
_PR_CAPBSET_DROP = 24
_ZERO_CAPABILITIES = "0000000000000000"
_TRANSITION_CAPABILITY_NUMBERS = (5, 6, 7, 8)
_LINUX_CAPABILITY_VERSION_3 = 0x20080522


class RunActionCodingAgentEgressRelayError(RuntimeError):
    """The in-container egress relay cannot prove or retain its narrow authority."""


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


def _erase_transition_capabilities() -> None:
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


def _bind_parent_death_signal(expected_parent_process_id: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_process_id:
        raise RunActionCodingAgentEgressRelayError(
            "coding-agent egress relay parent changed during containment"
        )


def _require_unprivileged_relay_identity(
    supervisor_user_id: int,
    supervisor_group_id: int,
    provider_group_id: int,
) -> None:
    if (
        os.geteuid() != supervisor_user_id
        or os.getegid() != supervisor_group_id
        or os.getgroups() != sorted({supervisor_group_id, provider_group_id})
    ):
        raise RunActionCodingAgentEgressRelayError(
            "coding-agent egress relay identity is not exact"
        )
    capability_fields = {
        line.split(":", 1)[0]: line.split()[1]
        for line in open("/proc/self/status", encoding="ascii").read().splitlines()
        if line.startswith(("CapInh:", "CapPrm:", "CapEff:", "CapBnd:", "CapAmb:"))
    }
    if set(capability_fields.values()) != {_ZERO_CAPABILITIES}:
        raise RunActionCodingAgentEgressRelayError(
            "coding-agent egress relay retained Linux capabilities"
        )


def _tunnel(left: socket.socket, right: socket.socket, chunk_size_bytes: int) -> None:
    readable = [left, right]
    peers = {left: right, right: left}
    while readable:
        ready, _writable, _exceptional = select.select(readable, (), ())
        for source in ready:
            payload = source.recv(chunk_size_bytes)
            destination = peers[source]
            if payload:
                destination.sendall(payload)
            else:
                readable.remove(source)
                destination.shutdown(socket.SHUT_WR)


def serve_egress_relay(
    *,
    port: int,
    backlog: int,
    chunk_size_bytes: int,
    readiness_descriptor: int,
    supervisor_user_id: int,
    supervisor_group_id: int,
    provider_group_id: int,
) -> None:
    """Serve raw provider proxy connections until the trusted parent terminates."""

    if (
        not 0 < port <= 65_535
        or backlog <= 0
        or chunk_size_bytes <= 0
        or readiness_descriptor <= 2
    ):
        raise RunActionCodingAgentEgressRelayError(
            "coding-agent egress relay arguments are invalid or unbounded"
        )
    _erase_transition_capabilities()
    _require_unprivileged_relay_identity(
        supervisor_user_id,
        supervisor_group_id,
        provider_group_id,
    )
    _bind_parent_death_signal(os.getppid())
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", port))
    listener.listen(backlog)
    if os.write(readiness_descriptor, b"\x01") != 1:
        raise RunActionCodingAgentEgressRelayError(
            "coding-agent egress relay readiness made no progress"
        )
    os.close(readiness_descriptor)
    while True:
        provider, _address = listener.accept()
        parent_process_id = os.getpid()
        child_process_id = os.fork()
        if child_process_id == 0:
            listener.close()
            _bind_parent_death_signal(parent_process_id)
            broker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            broker.connect(PROVIDER_EGRESS_BROKER_PATH)
            _tunnel(provider, broker, chunk_size_bytes)
            provider.close()
            broker.close()
            os._exit(0)
        provider.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backlog", type=int, required=True)
    parser.add_argument("--chunk-size-bytes", type=int, required=True)
    parser.add_argument("--readiness-descriptor", type=int, required=True)
    parser.add_argument("--supervisor-user-id", type=int, required=True)
    parser.add_argument("--supervisor-group-id", type=int, required=True)
    parser.add_argument("--provider-group-id", type=int, required=True)
    arguments = parser.parse_args()
    serve_egress_relay(
        port=arguments.port,
        backlog=arguments.backlog,
        chunk_size_bytes=arguments.chunk_size_bytes,
        readiness_descriptor=arguments.readiness_descriptor,
        supervisor_user_id=arguments.supervisor_user_id,
        supervisor_group_id=arguments.supervisor_group_id,
        provider_group_id=arguments.provider_group_id,
    )


if __name__ == "__main__":
    main()
