"""Trusted loopback-to-Unix relay for one network-isolated coding agent."""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket

from kapso.cross_run.launch.run_action_coding_agent_layout import (
    PROVIDER_EGRESS_BROKER_PATH,
)
from kapso.cross_run.launch.run_action_coding_agent_trusted_child import (
    bind_trusted_parent_death_signal,
    erase_trusted_child_capabilities,
    require_unprivileged_supervisor_child,
)


class RunActionCodingAgentEgressRelayError(RuntimeError):
    """The in-container egress relay cannot prove or retain its narrow authority."""


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
                if destination in readable:
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
    erase_trusted_child_capabilities()
    require_unprivileged_supervisor_child(
        supervisor_user_id=supervisor_user_id,
        supervisor_group_id=supervisor_group_id,
        provider_group_id=provider_group_id,
    )
    bind_trusted_parent_death_signal(os.getppid())
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
            bind_trusted_parent_death_signal(parent_process_id)
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
