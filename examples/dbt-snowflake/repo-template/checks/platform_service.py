"""Shared by dev_reset.py and release.py: how a script of this folder reaches the data platform's service."""
import json
import pathlib
import socket
import sys
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SERVICE = json.loads((HERE / "service.json").read_text())
ADDRESS = urllib.parse.urlparse(SERVICE["url"])


class Quiet(urllib.request.HTTPErrorProcessor):
    """Hand every answer back, whatever its status, so a refusal is read rather than raised."""

    def http_response(self, request, response):
        return response

    https_response = http_response


def call(method, path, payload=None, timeout_key="request_timeout_seconds"):
    """(status, answer) of one request; `payload` is sent as JSON, and the wait is the named setting's seconds."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(SERVICE["url"] + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.build_opener(Quiet()).open(request, timeout=SERVICE[timeout_key]) as response:
        return response.status, json.loads(response.read().decode())


def require_service(what):
    """Exit 2 with the reason when the service does not accept a connection."""
    probe = socket.socket()
    probe.settimeout(SERVICE["connect_timeout_seconds"])
    error = probe.connect_ex((ADDRESS.hostname, ADDRESS.port))
    probe.close()
    if error != 0:
        print(f"{what}: the platform service at {SERVICE['url']} is not reachable (connect error {error}). "
              "See README.md, 'The ticket checks'.")
        sys.exit(2)
