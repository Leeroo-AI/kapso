"""The harness: an OpenAI-compatible service between the task runner and the model endpoint.

    python -m harness.serve --port <port> --upstream <base url>      # e.g. http://localhost:18000/v1

The runner is pointed at this service instead of the model endpoint. Every chat completion
the runner sends goes through `Harness.prepare` on the way to the model and `Harness.review`
on the way back; both return their input unchanged here, which makes this service the bare
agent and the baseline. The harness's own endpoint settings live in `SETTINGS` and are
reported by `/health` with `VERSION`, so an evaluation and a deployment record what ran.

Routes:
    POST /v1/chat/completions   the runner's request: prepared, sent upstream, reviewed, returned
    GET  /v1/models             the upstream's list, unchanged
    GET  /health                200 with the version, the settings and the upstream's models once
                                the upstream answers; 503 while it does not
A request with `stream: true` is relayed chunk by chunk without review (the runner does not
stream). An upstream error (a context-window overflow, say) is returned with its own status and
body, so the runner sees the failure it would see from the model. One line per request goes to
stderr: the path, the status, the duration, the message and tool counts, the token usage; never
the content. Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "0.1.0"   # the harness revision: bump it with every change that is measured
SETTINGS = {
    # The harness's own endpoint settings, the ones it may change. What the runner sets
    # (its sampling, its tools, its step limits) stays the runner's.
    "request_timeout_seconds": 900,   # one upstream completion, thinking included
}
CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
MODELS_PATHS = ("/v1/models", "/models")


class Harness:
    """The two seams: what reaches the model, and what reaches the runner."""

    def prepare(self, request):
        """The runner's request (messages, tools, sampling) as it will be sent to the model."""
        return request

    def review(self, request, response):
        """The model's reply as it will be returned to the runner."""
        return response


def upstream_call(base, path, payload, headers, timeout):
    """One upstream request: (status, body bytes, content type). An HTTP error comes back with
    its own status and body; a connection failure or a timeout as 502 with a JSON error."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    for name, value in headers.items():
        request.add_header(name, value)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return reply.status, reply.read(), reply.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        message = f"the harness could not reach its upstream {base}: {error}"
        return 502, json.dumps({"error": {"message": message, "type": "harness_upstream_error"}}).encode("utf-8"), "application/json"


class Handler(BaseHTTPRequestHandler):
    harness = Harness()
    upstream = ""

    def log_message(self, format, *args):
        return   # the request line is logged by `note`, without content

    def note(self, status, started, **fields):
        line = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": self.path,
                "status": status, "ms": int((time.monotonic() - started) * 1000), **fields}
        print(json.dumps(line), file=sys.stderr, flush=True)

    def reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forwarded_headers(self):
        return {"Authorization": self.headers["Authorization"]} if self.headers.get("Authorization") else {}

    def do_GET(self):
        started = time.monotonic()
        if self.path == "/health":
            status, body, _ = upstream_call(self.upstream, "/models", None, self.forwarded_headers(), 10)
            models = [item.get("id") for item in json.loads(body).get("data", [])] if status == 200 else []
            health = {"status": "ok" if status == 200 else "upstream unreachable",
                      "harness": {"version": VERSION, "settings": SETTINGS},
                      "upstream": self.upstream, "upstream_models": models}
            code = 200 if status == 200 else 503
            self.reply(code, json.dumps(health).encode("utf-8"))
            self.note(code, started)
            return
        if self.path in MODELS_PATHS:
            status, body, content_type = upstream_call(self.upstream, "/models", None, self.forwarded_headers(), 30)
            self.reply(status, body, content_type)
            self.note(status, started)
            return
        self.reply(404, json.dumps({"error": {"message": f"no route {self.path}", "type": "not_found"}}).encode("utf-8"))
        self.note(404, started)

    def do_POST(self):
        started = time.monotonic()
        if self.path not in CHAT_PATHS:
            self.reply(404, json.dumps({"error": {"message": f"no route {self.path}", "type": "not_found"}}).encode("utf-8"))
            self.note(404, started)
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as error:
            self.reply(400, json.dumps({"error": {"message": f"the request body is not JSON: {error}",
                                                  "type": "invalid_request_error"}}).encode("utf-8"))
            self.note(400, started)
            return
        prepared = self.harness.prepare(request)
        counts = {"messages": len(prepared.get("messages") or []), "tools": len(prepared.get("tools") or [])}
        if prepared.get("stream"):
            status = self.relay_stream(prepared)
            self.note(status, started, streamed=True, **counts)
            return
        status, body, content_type = upstream_call(self.upstream, "/chat/completions", prepared,
                                                   self.forwarded_headers(), SETTINGS["request_timeout_seconds"])
        usage = {}
        if status == 200:
            response = self.harness.review(prepared, json.loads(body))
            usage = response.get("usage") or {}
            body, content_type = json.dumps(response).encode("utf-8"), "application/json"
        self.reply(status, body, content_type)
        self.note(status, started, prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"), **counts)

    def relay_stream(self, prepared):
        """A streaming completion passed through as it arrives; the status it ended with."""
        request = urllib.request.Request(self.upstream + "/chat/completions",
                                         data=json.dumps(prepared).encode("utf-8"), method="POST")
        request.add_header("Content-Type", "application/json")
        for name, value in self.forwarded_headers().items():
            request.add_header(name, value)
        try:
            reply = urllib.request.urlopen(request, timeout=SETTINGS["request_timeout_seconds"])
        except urllib.error.HTTPError as error:
            self.reply(error.code, error.read(), error.headers.get("Content-Type", "application/json"))
            return error.code
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            message = f"the harness could not reach its upstream {self.upstream}: {error}"
            self.reply(502, json.dumps({"error": {"message": message, "type": "harness_upstream_error"}}).encode("utf-8"))
            return 502
        with reply:
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.headers.get("Content-Type", "text/event-stream"))
            self.end_headers()
            while True:
                chunk = reply.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            return reply.status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help="the port to listen on")
    parser.add_argument("--upstream", required=True, help="the model endpoint's base URL, ending in /v1")
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on")
    args = parser.parse_args()
    Handler.upstream = args.upstream.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"harness": VERSION, "listening": f"http://{args.host}:{args.port}",
                      "upstream": Handler.upstream, "settings": SETTINGS}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
