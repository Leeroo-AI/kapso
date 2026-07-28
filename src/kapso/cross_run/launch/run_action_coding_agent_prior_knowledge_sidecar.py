"""Trusted descriptor-only prior-knowledge MCP sidecar."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import os
import re
import select
import signal
import socket
import stat
import struct
import sys
from dataclasses import dataclass
from io import FileIO, TextIOWrapper

from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.launch.run_action_coding_agent_trusted_child import (
    bind_trusted_parent_death_signal,
    erase_trusted_child_capabilities,
    require_unprivileged_supervisor_child,
)
from kapso.gated_mcp.gates.base import GateConfig
from kapso.gated_mcp.gates.prior_knowledge_gate import PriorKnowledgeGate
from kapso.gated_mcp.prior_knowledge_cli import (
    _create_prior_knowledge_server,
    _run_prior_knowledge_server,
)

_SOCKET_NAME_PATTERN = re.compile(r"^kapso-prior-knowledge[.]agent_call_[0-9a-f]{32}$")
_SESSION_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PEER_CREDENTIAL_FORMAT = "3i"


class RunActionCodingAgentPriorKnowledgeSidecarError(RuntimeError):
    """The trusted MCP sidecar cannot prove its descriptors or provider peer."""


@dataclass(frozen=True)
class _DescriptorAuditSink:
    descriptor: int
    maximum_bytes: int
    owner_user_id: int
    owner_group_id: int

    def __call__(self, payload: bytes) -> None:
        metadata = os.fstat(self.descriptor)
        if (
            type(payload) is not bytes
            or not payload
            or not payload.endswith(b"\n")
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != self.owner_user_id
            or metadata.st_gid != self.owner_group_id
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size + len(payload) > self.maximum_bytes
        ):
            raise RunActionCodingAgentPriorKnowledgeSidecarError(
                "prior-knowledge anonymous audit is unsafe or full"
            )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(self.descriptor, remaining)
            if written <= 0:
                raise RunActionCodingAgentPriorKnowledgeSidecarError(
                    "prior-knowledge anonymous audit write made no progress"
                )
            remaining = remaining[written:]
        os.fsync(self.descriptor)


def _read_packet_descriptor(
    descriptor: int,
    *,
    exact_bytes: int,
    owner_user_id: int,
    owner_group_id: int,
) -> bytes:
    metadata = os.fstat(descriptor)
    if (
        exact_bytes <= 0
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != owner_user_id
        or metadata.st_gid != owner_group_id
        or metadata.st_nlink != 0
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_size != exact_bytes
    ):
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge anonymous packet descriptor is invalid"
        )
    payload = os.pread(descriptor, exact_bytes + 1, 0)
    if len(payload) != exact_bytes:
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge anonymous packet is incomplete"
        )
    return payload


def _read_authentication_frame(connection: socket.socket) -> bytes:
    payload = bytearray()
    while not payload.endswith(b"\n"):
        if len(payload) >= 65:
            raise RunActionCodingAgentPriorKnowledgeSidecarError(
                "prior-knowledge authentication frame exceeds its exact bound"
            )
        chunk = connection.recv(65 - len(payload))
        if not chunk:
            raise RunActionCodingAgentPriorKnowledgeSidecarError(
                "prior-knowledge authentication frame is incomplete"
            )
        payload.extend(chunk)
    return bytes(payload)


def _replace_inherited_standard_stream_wrappers() -> None:
    sys.stdin = TextIOWrapper(
        FileIO(os.dup(0), mode="rb", closefd=True),
        encoding="utf-8",
    )
    sys.stdout = TextIOWrapper(
        FileIO(os.dup(1), mode="wb", closefd=True),
        encoding="utf-8",
    )


def serve_prior_knowledge_sidecar(
    *,
    packet_descriptor: int,
    packet_bytes: int,
    audit_descriptor: int,
    audit_maximum_bytes: int,
    control_descriptor: int,
    readiness_descriptor: int,
    socket_name: str,
    session_token: str,
    operation_id: str,
    supervisor_user_id: int,
    supervisor_group_id: int,
    provider_user_id: int,
    provider_group_id: int,
) -> None:
    if (
        packet_descriptor <= 2
        or audit_descriptor <= 2
        or control_descriptor <= 2
        or readiness_descriptor <= 2
        or len(
            {
                packet_descriptor,
                audit_descriptor,
                control_descriptor,
                readiness_descriptor,
            }
        )
        != 4
        or packet_bytes <= 0
        or audit_maximum_bytes <= 0
        or _SOCKET_NAME_PATTERN.fullmatch(socket_name) is None
        or _SESSION_TOKEN_PATTERN.fullmatch(session_token) is None
    ):
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge sidecar arguments are invalid or unbounded"
        )
    erase_trusted_child_capabilities()
    require_unprivileged_supervisor_child(
        supervisor_user_id=supervisor_user_id,
        supervisor_group_id=supervisor_group_id,
        provider_group_id=provider_group_id,
    )
    bind_trusted_parent_death_signal(os.getppid())
    packet_payload = _read_packet_descriptor(
        packet_descriptor,
        exact_bytes=packet_bytes,
        owner_user_id=supervisor_user_id,
        owner_group_id=supervisor_group_id,
    )
    materialization = PriorKnowledgeAccessMaterialization.from_json_bytes(
        packet_payload
    )
    if materialization.to_json_bytes() != packet_payload:
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge anonymous packet is not canonical"
        )
    audit_sink = _DescriptorAuditSink(
        descriptor=audit_descriptor,
        maximum_bytes=audit_maximum_bytes,
        owner_user_id=supervisor_user_id,
        owner_group_id=supervisor_group_id,
    )
    audit_sink_metadata = os.fstat(audit_descriptor)
    if audit_sink_metadata.st_size != 0:
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge anonymous audit is not empty"
        )
    gate = PriorKnowledgeGate(
        GateConfig(enabled=True),
        access=PriorKnowledgeAccess(materialization),
        audit_sink=audit_sink,
        operation_id=operation_id,
    )
    server = _create_prior_knowledge_server(gate=gate)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\x00" + socket_name)
    listener.listen(1)
    if os.write(readiness_descriptor, b"\x01") != 1:
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge sidecar readiness made no progress"
        )
    os.close(readiness_descriptor)
    ready, _writable, _exceptional = select.select(
        (listener, control_descriptor),
        (),
        (),
    )
    if control_descriptor in ready:
        if os.read(control_descriptor, 2) != b"\x01":
            raise RunActionCodingAgentPriorKnowledgeSidecarError(
                "prior-knowledge sidecar early control is invalid"
            )
        os.close(control_descriptor)
        listener.close()
        return
    connection, _address = listener.accept()
    listener.close()
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
        or peer_user_id != provider_user_id
        or peer_group_id != provider_group_id
    ):
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge sidecar peer identity is invalid"
        )
    authentication = _read_authentication_frame(connection)
    expected_authentication = session_token.encode("ascii") + b"\n"
    if not hmac.compare_digest(authentication, expected_authentication):
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge sidecar authentication failed"
        )
    connection.sendall(b"\x01")
    sidecar_process_id = os.getpid()
    server_process_id = os.fork()
    if server_process_id == 0:
        os.close(control_descriptor)
        bind_trusted_parent_death_signal(sidecar_process_id)
        os.dup2(connection.fileno(), 0)
        os.dup2(connection.fileno(), 1)
        connection.close()
        _replace_inherited_standard_stream_wrappers()
        asyncio.run(_run_prior_knowledge_server(server))
        signal.pause()
    shutdown_payload = os.read(control_descriptor, 2)
    if shutdown_payload != b"\x01":
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge sidecar control is invalid"
        )
    os.close(control_descriptor)
    connection.close()
    os.kill(server_process_id, signal.SIGKILL)
    observed_process_id, child_status = os.waitpid(server_process_id, 0)
    if (
        observed_process_id != server_process_id
        or not os.WIFSIGNALED(child_status)
        or os.WTERMSIG(child_status) != signal.SIGKILL
    ):
        raise RunActionCodingAgentPriorKnowledgeSidecarError(
            "prior-knowledge MCP child did not accept exact termination"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-descriptor", type=int, required=True)
    parser.add_argument("--packet-bytes", type=int, required=True)
    parser.add_argument("--audit-descriptor", type=int, required=True)
    parser.add_argument("--audit-maximum-bytes", type=int, required=True)
    parser.add_argument("--control-descriptor", type=int, required=True)
    parser.add_argument("--readiness-descriptor", type=int, required=True)
    parser.add_argument("--socket-name", required=True)
    parser.add_argument("--session-token", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--supervisor-user-id", type=int, required=True)
    parser.add_argument("--supervisor-group-id", type=int, required=True)
    parser.add_argument("--provider-user-id", type=int, required=True)
    parser.add_argument("--provider-group-id", type=int, required=True)
    arguments = parser.parse_args()
    serve_prior_knowledge_sidecar(
        packet_descriptor=arguments.packet_descriptor,
        packet_bytes=arguments.packet_bytes,
        audit_descriptor=arguments.audit_descriptor,
        audit_maximum_bytes=arguments.audit_maximum_bytes,
        control_descriptor=arguments.control_descriptor,
        readiness_descriptor=arguments.readiness_descriptor,
        socket_name=arguments.socket_name,
        session_token=arguments.session_token,
        operation_id=arguments.operation_id,
        supervisor_user_id=arguments.supervisor_user_id,
        supervisor_group_id=arguments.supervisor_group_id,
        provider_user_id=arguments.provider_user_id,
        provider_group_id=arguments.provider_group_id,
    )


if __name__ == "__main__":
    main()
