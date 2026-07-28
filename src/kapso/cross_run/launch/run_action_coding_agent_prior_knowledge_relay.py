"""Provider-side stdio relay for one authenticated prior-knowledge session."""

from __future__ import annotations

import argparse
import os
import re
import select
import socket
import struct

from kapso.cross_run.launch.run_action_coding_agent_trusted_child import (
    require_no_new_privileges,
    require_zero_linux_capabilities,
)

_SOCKET_NAME_PATTERN = re.compile(r"^kapso-prior-knowledge[.]agent_call_[0-9a-f]{32}$")
_SESSION_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PEER_CREDENTIAL_FORMAT = "3i"


class RunActionCodingAgentPriorKnowledgeRelayError(RuntimeError):
    """The provider relay cannot prove its sidecar or bounded session."""


def _require_provider_identity(provider_user_id: int, provider_group_id: int) -> None:
    if (
        os.getresuid() != (provider_user_id,) * 3
        or os.getresgid() != (provider_group_id,) * 3
        or os.getgroups()
    ):
        raise RunActionCodingAgentPriorKnowledgeRelayError(
            "prior-knowledge relay provider identity is not exact"
        )
    require_zero_linux_capabilities()
    require_no_new_privileges()


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RunActionCodingAgentPriorKnowledgeRelayError(
                "prior-knowledge relay write made no progress"
            )
        remaining = remaining[written:]


def _relay_stdio(connection: socket.socket, chunk_size_bytes: int) -> None:
    stdin_open = True
    while True:
        readable = [connection]
        if stdin_open:
            readable.append(0)
        ready, _writable, _exceptional = select.select(readable, (), ())
        if 0 in ready:
            payload = os.read(0, chunk_size_bytes)
            if payload:
                connection.sendall(payload)
            else:
                stdin_open = False
                connection.shutdown(socket.SHUT_WR)
        if connection in ready:
            payload = connection.recv(chunk_size_bytes)
            if not payload:
                return
            _write_all(1, payload)


def run_prior_knowledge_relay(
    *,
    socket_name: str,
    session_token: str,
    chunk_size_bytes: int,
    provider_user_id: int,
    provider_group_id: int,
    sidecar_user_id: int,
    sidecar_group_id: int,
) -> None:
    if (
        _SOCKET_NAME_PATTERN.fullmatch(socket_name) is None
        or _SESSION_TOKEN_PATTERN.fullmatch(session_token) is None
        or type(chunk_size_bytes) is not int
        or chunk_size_bytes <= 0
        or any(
            type(identity) is not int or identity <= 0
            for identity in (
                provider_user_id,
                provider_group_id,
                sidecar_user_id,
                sidecar_group_id,
            )
        )
        or provider_user_id == sidecar_user_id
        or provider_group_id == sidecar_group_id
    ):
        raise RunActionCodingAgentPriorKnowledgeRelayError(
            "prior-knowledge relay arguments are invalid"
        )
    _require_provider_identity(provider_user_id, provider_group_id)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect("\x00" + socket_name)
    peer_payload = connection.getsockopt(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        struct.calcsize(_PEER_CREDENTIAL_FORMAT),
    )
    peer_process_id, peer_user_id, peer_group_id = struct.unpack(
        _PEER_CREDENTIAL_FORMAT,
        peer_payload,
    )
    if (
        peer_process_id <= 1
        or peer_user_id != sidecar_user_id
        or peer_group_id != sidecar_group_id
    ):
        raise RunActionCodingAgentPriorKnowledgeRelayError(
            "prior-knowledge relay peer identity is invalid"
        )
    connection.sendall(session_token.encode("ascii") + b"\n")
    if connection.recv(1) != b"\x01":
        raise RunActionCodingAgentPriorKnowledgeRelayError(
            "prior-knowledge relay authentication was not acknowledged"
        )
    _relay_stdio(connection, chunk_size_bytes)
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-name", required=True)
    parser.add_argument("--session-token", required=True)
    parser.add_argument("--chunk-size-bytes", type=int, required=True)
    parser.add_argument("--provider-user-id", type=int, required=True)
    parser.add_argument("--provider-group-id", type=int, required=True)
    parser.add_argument("--sidecar-user-id", type=int, required=True)
    parser.add_argument("--sidecar-group-id", type=int, required=True)
    arguments = parser.parse_args()
    run_prior_knowledge_relay(
        socket_name=arguments.socket_name,
        session_token=arguments.session_token,
        chunk_size_bytes=arguments.chunk_size_bytes,
        provider_user_id=arguments.provider_user_id,
        provider_group_id=arguments.provider_group_id,
        sidecar_user_id=arguments.sidecar_user_id,
        sidecar_group_id=arguments.sidecar_group_id,
    )


if __name__ == "__main__":
    main()
