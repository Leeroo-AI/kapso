"""Host-side exact-authority HTTPS CONNECT broker for one coding-agent action."""

from __future__ import annotations

import argparse
import ctypes
import os
import select
import signal
import socket
from pathlib import Path

_PR_SET_PDEATHSIG = 1
_HEADER_TERMINATOR = b"\r\n\r\n"
_SUCCESS_RESPONSE = b"HTTP/1.1 200 Connection Established\r\n\r\n"
_DENIED_RESPONSE = b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n"


class RunActionCodingAgentEgressBrokerError(RuntimeError):
    """The host broker rejected or could not contain one CONNECT request."""


def _bind_parent_death_signal(expected_parent_process_id: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_process_id:
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent egress broker parent changed during containment"
        )


def _read_connect_request(
    client: socket.socket,
    maximum_header_bytes: int,
) -> tuple[str, bytes]:
    payload = bytearray()
    while _HEADER_TERMINATOR not in payload:
        remaining = maximum_header_bytes - len(payload)
        if remaining <= 0:
            raise RunActionCodingAgentEgressBrokerError(
                "coding-agent CONNECT header exceeded its exact byte bound"
            )
        chunk = client.recv(remaining)
        if not chunk:
            raise RunActionCodingAgentEgressBrokerError(
                "coding-agent CONNECT request ended before its complete header"
            )
        payload.extend(chunk)
    header, separator, remainder = bytes(payload).partition(_HEADER_TERMINATOR)
    if separator != _HEADER_TERMINATOR:
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent CONNECT request lacks a complete header"
        )
    lines = header.decode("ascii").split("\r\n")
    request_line = lines[0].split(" ")
    if (
        len(request_line) != 3
        or request_line[0] != "CONNECT"
        or request_line[2] != "HTTP/1.1"
        or any(":" not in line or not line.split(":", 1)[0] for line in lines[1:])
    ):
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent proxy request is not strict HTTP/1.1 CONNECT"
        )
    return request_line[1], remainder


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


def _serve_connect(
    client: socket.socket,
    *,
    authorities: frozenset[str],
    maximum_header_bytes: int,
    chunk_size_bytes: int,
    connect_timeout_seconds: int,
) -> None:
    authority, remainder = _read_connect_request(client, maximum_header_bytes)
    if authority not in authorities:
        client.sendall(_DENIED_RESPONSE)
        raise RunActionCodingAgentEgressBrokerError(
            f"coding-agent CONNECT authority was denied: {authority}"
        )
    hostname, separator, port = authority.rpartition(":")
    if not hostname or separator != ":" or port != "443":
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent CONNECT authority is not HTTPS"
        )
    upstream = socket.create_connection(
        (hostname, 443),
        timeout=connect_timeout_seconds,
    )
    upstream.settimeout(None)
    client.sendall(_SUCCESS_RESPONSE)
    if remainder:
        upstream.sendall(remainder)
    _tunnel(client, upstream, chunk_size_bytes)
    upstream.close()


def serve_egress_broker(
    *,
    socket_path: Path,
    authorities: tuple[str, ...],
    maximum_header_bytes: int,
    backlog: int,
    chunk_size_bytes: int,
    connect_timeout_seconds: int,
    readiness_descriptor: int,
) -> None:
    """Serve exact CONNECT authorities until the launching process terminates."""

    if (
        not socket_path.is_absolute()
        or socket_path.exists()
        or not socket_path.parent.is_dir()
        or authorities != tuple(sorted(set(authorities)))
        or not authorities
        or maximum_header_bytes <= 0
        or backlog <= 0
        or chunk_size_bytes <= 0
        or connect_timeout_seconds <= 0
        or readiness_descriptor <= 2
    ):
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent egress broker arguments are invalid or unbounded"
        )
    _bind_parent_death_signal(os.getppid())
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path.as_posix())
    os.chmod(socket_path, 0o600)
    listener.listen(backlog)
    if os.write(readiness_descriptor, b"\x01") != 1:
        raise RunActionCodingAgentEgressBrokerError(
            "coding-agent egress broker readiness made no progress"
        )
    os.close(readiness_descriptor)
    admitted = frozenset(authorities)
    while True:
        client, _address = listener.accept()
        parent_process_id = os.getpid()
        child_process_id = os.fork()
        if child_process_id == 0:
            listener.close()
            _bind_parent_death_signal(parent_process_id)
            _serve_connect(
                client,
                authorities=admitted,
                maximum_header_bytes=maximum_header_bytes,
                chunk_size_bytes=chunk_size_bytes,
                connect_timeout_seconds=connect_timeout_seconds,
            )
            client.close()
            os._exit(0)
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--authority", action="append", required=True)
    parser.add_argument("--maximum-header-bytes", type=int, required=True)
    parser.add_argument("--backlog", type=int, required=True)
    parser.add_argument("--chunk-size-bytes", type=int, required=True)
    parser.add_argument("--connect-timeout-seconds", type=int, required=True)
    parser.add_argument("--readiness-descriptor", type=int, required=True)
    arguments = parser.parse_args()
    serve_egress_broker(
        socket_path=arguments.socket_path,
        authorities=tuple(arguments.authority),
        maximum_header_bytes=arguments.maximum_header_bytes,
        backlog=arguments.backlog,
        chunk_size_bytes=arguments.chunk_size_bytes,
        connect_timeout_seconds=arguments.connect_timeout_seconds,
        readiness_descriptor=arguments.readiness_descriptor,
    )


if __name__ == "__main__":
    main()
