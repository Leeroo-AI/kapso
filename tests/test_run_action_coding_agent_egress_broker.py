import socket
from contextlib import ExitStack

import pytest

from kapso.cross_run.launch.run_action_coding_agent_egress_broker import (
    RunActionCodingAgentEgressBrokerError,
    _read_connect_request,
    _serve_connect,
)


def test_connect_parser_preserves_post_header_bytes():
    client, broker = socket.socketpair()
    with ExitStack() as resources:
        resources.callback(client.close)
        resources.callback(broker.close)
        client.sendall(
            b"CONNECT chatgpt.com:443 HTTP/1.1\r\n" b"Host: chatgpt.com:443\r\n\r\nTLS"
        )

        authority, remainder = _read_connect_request(broker, 1_024)

    assert authority == "chatgpt.com:443"
    assert remainder == b"TLS"


def test_connect_broker_denies_every_unpinned_authority_before_dialing():
    client, broker = socket.socketpair()
    with ExitStack() as resources:
        resources.callback(client.close)
        resources.callback(broker.close)
        client.sendall(
            b"CONNECT api.openai.com:443 HTTP/1.1\r\n"
            b"Host: api.openai.com:443\r\n\r\n"
        )

        with pytest.raises(
            RunActionCodingAgentEgressBrokerError,
            match="api.openai.com:443",
        ):
            _serve_connect(
                broker,
                authorities=frozenset({"chatgpt.com:443"}),
                maximum_header_bytes=1_024,
                chunk_size_bytes=4_096,
                connect_timeout_seconds=1,
            )

        assert client.recv(1_024).startswith(b"HTTP/1.1 403 Forbidden\r\n")


@pytest.mark.parametrize(
    "payload,maximum_bytes,message",
    (
        (
            b"GET https://chatgpt.com/ HTTP/1.1\r\nHost: chatgpt.com\r\n\r\n",
            1_024,
            "strict HTTP/1.1 CONNECT",
        ),
        (
            b"CONNECT chatgpt.com:443 HTTP/1.1\r\nHost: chatgpt.com:443",
            32,
            "exact byte bound",
        ),
    ),
)
def test_connect_parser_fails_loud_on_other_or_unbounded_protocols(
    payload,
    maximum_bytes,
    message,
):
    client, broker = socket.socketpair()
    with ExitStack() as resources:
        resources.callback(client.close)
        resources.callback(broker.close)
        client.sendall(payload)

        with pytest.raises(RunActionCodingAgentEgressBrokerError, match=message):
            _read_connect_request(broker, maximum_bytes)
