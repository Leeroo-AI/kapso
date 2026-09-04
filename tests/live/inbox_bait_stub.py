"""A local stand-in for the OpenAI embeddings endpoint, for the bait suite
(plan §3.3 B7 and H4). Stdlib only, one process per fixture run:

    python inbox_bait_stub.py <port> transient   # 429 rate-limit to the first
                                                 # four requests, then embeddings
    python inbox_bait_stub.py <port> quota       # insufficient_quota, always

The embeddings are deterministic feature-hashed bags of words, so a
ranking computed against them is meaningful and reproducible.
"""

import hashlib
import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIMENSIONS = 64
TRANSIENT_FAILURES = 4  # more than the openai client's default two retries

RATE_LIMIT = {
    "error": {
        "message": "Rate limit reached for text-embedding-3-small: 3 requests per minute. "
                   "Please try again in 20s.",
        "type": "requests", "param": None, "code": "rate_limit_exceeded",
    }
}
QUOTA = {
    "error": {
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "type": "insufficient_quota", "param": None, "code": "insufficient_quota",
    }
}


def embed(text: str):
    vector = [0.0] * DIMENSIONS
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % DIMENSIONS
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Handler(BaseHTTPRequestHandler):
    mode = "transient"
    served = 0

    def log_message(self, fmt, *args):
        sys.stderr.write(f"stub {self.mode}: {fmt % args}\n")

    def _send(self, code: int, body: dict, headers=None):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "text-embedding-3-small", "object": "model", "owned_by": "stub"},
            ]})
            return
        self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/embeddings"):
            self._send(404, {"error": {"message": f"unknown path {self.path}", "type": "invalid_request_error"}})
            return
        Handler.served += 1
        if Handler.mode == "quota":
            self._send(429, QUOTA)
            return
        if Handler.served <= TRANSIENT_FAILURES:
            self._send(429, RATE_LIMIT, {"Retry-After": "2"})
            return
        inputs = request.get("input")
        texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
        self._send(200, {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": embed(t)} for i, t in enumerate(texts)],
            "model": request.get("model", "text-embedding-3-small"),
            "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
        })


def main(argv):
    port, mode = int(argv[0]), argv[1]
    if mode not in ("transient", "quota"):
        raise SystemExit(f"mode must be transient or quota, got {mode!r}")
    Handler.mode = mode
    server = HTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"stub {mode} listening on 127.0.0.1:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
