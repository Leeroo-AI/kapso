"""Vertex AI Claude Code adapter — the Claude Code CLI driven by an open model
served by Vertex AI Model Garden (e.g. ``zai-org/glm-5.2-maas``).

Vertex serves its open models through an OpenAI-compatible endpoint and
authenticates with Google credentials, so neither of the two things the OSS
adapter needs (an Anthropic-compatible URL and a bearer token) exists. A
local LiteLLM proxy supplies both: it accepts the CLI's Anthropic requests,
calls Vertex through LiteLLM's native ``vertex_ai`` provider (google-auth
finds and refreshes the credential: the gcloud application-default login on
a workstation, the service account on a Compute Engine machine), and guards
its port with a random key minted for this process. Everything downstream of
the proxy is the OSS adapter unchanged. A bridge is not "up" until one
one-token request has gone through it to the model and back: that probe is
what turns a missing credential, a wrong project or an unknown model into an
error at start rather than mid-session.

One bridge per process for each (project, location, model): the learner runs
about ten sessions per repository and a proxy takes seconds to start. The
bridge dies with the process (``atexit``).

Required ``agent_specific`` keys:

- ``vertex_project``: the Google Cloud project billed for the model.

Optional: ``vertex_location`` (default ``global``, where the open models are
served), ``bridge_startup_timeout`` seconds, and ``effort``: the session's
reasoning effort, which here is the open model's ``reasoning_effort`` set on
the bridge for every request (Vertex accepts minimal / low / medium / high /
max for GLM 5.2; ``xhigh`` is translated to ``max``; the model's own default
is max). The CLI's own ``--effort`` flag is an Anthropic-only field the bridge
would drop, so this adapter does not pass it; LiteLLM's whitelist for these
models lacks the field too, which the bridge's hook module fixes.

Probe-verified 2026-09-28 (LiteLLM 1.103, ``vertex_ai/zai-org/glm-5.2-maas``,
global location): the Claude Code CLI reads and writes files through the
bridge; the model id is what ``--model`` carries, so no alias table.
"""

import atexit
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters.oss_claude_code_agent import (
    OssClaudeCodeCodingAgent,
)

BRIDGE_HOST = "127.0.0.1"
BRIDGE_HEALTH_PATH = "/health/liveliness"
BRIDGE_PROBE_BODY = {"max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
BRIDGE_KEY_ENV = "KAPSO_VERTEX_BRIDGE_KEY"
BRIDGE_LOG_TAIL_BYTES = 4000
# Claude's effort vocabulary is the config language; Vertex validates GLM's
# reasoning_effort against minimal/low/medium/high/max and rejects xhigh,
# which Z.ai defines as max.
EFFORT_TRANSLATION = {"xhigh": "max"}
# Copied next to the proxy config so LiteLLM loads it as a callback module.
BRIDGE_HOOK_SOURCE = Path(__file__).with_name("vertex_bridge_litellm_hook.py")


class VertexBridge:
    """A LiteLLM proxy translating Anthropic requests into Vertex calls."""

    def __init__(
        self, *, project: str, location: str, model: str, startup_timeout: float,
        effort: Optional[str] = None,
    ):
        self.project = project
        self.location = location
        self.model = model
        self.effort = effort
        self.startup_timeout = startup_timeout
        self.port: Optional[int] = None
        self.master_key = secrets.token_urlsafe(32)
        self._process: Optional[subprocess.Popen] = None
        self._dir: Optional[Path] = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise RuntimeError("Vertex bridge has not been started")
        return f"http://{BRIDGE_HOST}:{self.port}"

    def proxy_config(self) -> dict:
        """The LiteLLM config: one model, native Vertex provider, our key,
        and the reasoning effort as a default parameter of every request."""
        litellm_params = {
            "model": f"vertex_ai/{self.model}",
            "vertex_project": self.project,
            "vertex_location": self.location,
        }
        if self.effort is not None:
            litellm_params["reasoning_effort"] = EFFORT_TRANSLATION.get(self.effort, self.effort)
        return {
            "model_list": [{"model_name": self.model, "litellm_params": litellm_params}],
            "general_settings": {"master_key": self.master_key},
            "litellm_settings": {
                "drop_params": True,
                "callbacks": [f"{BRIDGE_HOOK_SOURCE.stem}.hook"],
            },
        }

    def start(self) -> None:
        litellm = shutil.which("litellm")
        if litellm is None:
            raise RuntimeError(
                "litellm not found. Install the proxy with: "
                "pip install 'litellm[proxy]'"
            )
        self._dir = Path(tempfile.mkdtemp(prefix="kapso-vertex-bridge-"))
        config_path = self._dir / "litellm.json"
        config_path.write_text(json.dumps(self.proxy_config(), indent=2))
        shutil.copy(BRIDGE_HOOK_SOURCE, self._dir / BRIDGE_HOOK_SOURCE.name)
        self.port = _free_port()
        log_path = self._dir / "bridge.log"
        env = os.environ.copy()
        # ADC needs a quota project for the billed API calls; the ADC file on
        # a box rarely carries one. Set for the bridge only, never read.
        env["GOOGLE_CLOUD_QUOTA_PROJECT"] = self.project
        with open(log_path, "w", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                [
                    litellm, "--config", str(config_path),
                    "--host", BRIDGE_HOST, "--port", str(self.port),
                    "--telemetry", "False",
                ],
                stdout=log, stderr=subprocess.STDOUT, env=env,
                cwd=self._dir, start_new_session=True,
            )
        atexit.register(self.stop)
        self._wait_until_serving(log_path)

    def _wait_until_serving(self, log_path: Path) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"litellm exited with code {self._process.returncode} "
                    f"before serving:\n{_tail(log_path)}"
                )
            if _port_open(self.port):
                break
            if time.monotonic() >= deadline:
                self.stop()
                raise RuntimeError(
                    f"litellm did not start serving on port {self.port} "
                    f"within {self.startup_timeout:.0f}s:\n{_tail(log_path)}"
                )
            time.sleep(0.25)
        with urllib.request.urlopen(
            f"{self.base_url}{BRIDGE_HEALTH_PATH}", timeout=10
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"litellm health check returned {response.status}:\n"
                    f"{_tail(log_path)}"
                )
        self._probe_model(log_path)

    def _probe_model(self, log_path: Path) -> None:
        """One one-token request through the proxy to the model: proves the
        credential, the project and the model name before any session."""
        request = urllib.request.Request(
            f"{self.base_url}/v1/messages", method="POST",
            data=json.dumps({"model": self.model, **BRIDGE_PROBE_BODY}).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.master_key,
                "anthropic-version": "2023-06-01",
            },
        )
        response = _PROBE_OPENER.open(request, timeout=120)
        if response.status != 200:
            reason = response.read()[:BRIDGE_LOG_TAIL_BYTES].decode(errors="replace")
            self.stop()
            raise RuntimeError(
                f"Vertex probe through the bridge failed with HTTP "
                f"{response.status}: {reason}\n{_tail(log_path)}"
            )

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            grace = time.monotonic() + 10
            while self._process.poll() is None and time.monotonic() < grace:
                time.sleep(0.1)
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
        if self._dir is not None and self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir = None


class _ReturnErrorResponses(urllib.request.HTTPErrorProcessor):
    """urlopen turns 4xx/5xx into exceptions that drop the body; the probe
    wants the body (the provider's reason) and raises its own error."""

    def http_response(self, request, response):
        return response

    https_response = http_response


_PROBE_OPENER = urllib.request.build_opener(_ReturnErrorResponses)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((BRIDGE_HOST, 0))
        return probe.getsockname()[1]


def _port_open(port: int) -> bool:
    """True once something accepts connections on the port; never raises."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((BRIDGE_HOST, port)) == 0


def _tail(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")[-BRIDGE_LOG_TAIL_BYTES:]


# Bridges shared by every agent of this process with the same target.
_BRIDGES: Dict[Tuple[str, str, str, Optional[str]], VertexBridge] = {}


def shared_bridge(
    *, project: str, location: str, model: str, startup_timeout: float,
    effort: Optional[str] = None,
) -> VertexBridge:
    key = (project, location, model, effort)
    bridge = _BRIDGES.get(key)
    if bridge is None or bridge.port is None:
        bridge = VertexBridge(
            project=project, location=location, model=model,
            startup_timeout=startup_timeout, effort=effort,
        )
        bridge.start()
        _BRIDGES[key] = bridge
    return bridge


class VertexClaudeCodeCodingAgent(OssClaudeCodeCodingAgent):
    """Claude Code CLI against a Vertex AI open model, through a local bridge."""

    def __init__(self, config: CodingAgentConfig):
        project = config.agent_specific.get("vertex_project")
        if not isinstance(project, str) or not project.strip():
            raise ValueError(
                "vertex_claude_code requires agent_specific.vertex_project "
                "(the Google Cloud project billed for the model)"
            )
        location = config.agent_specific.get("vertex_location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(
                "vertex_claude_code requires agent_specific.vertex_location"
            )
        startup_timeout = config.agent_specific.get("bridge_startup_timeout")
        if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float)) or startup_timeout <= 0:
            raise ValueError(
                "vertex_claude_code requires a positive "
                "agent_specific.bridge_startup_timeout (seconds)"
            )
        effort = config.agent_specific.get("effort")
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            raise ValueError(
                "vertex_claude_code: agent_specific.effort must be the open "
                "model's reasoning effort level (a non-empty string) or null"
            )
        self._bridge = shared_bridge(
            project=project.strip(), location=location.strip(),
            model=config.model, startup_timeout=float(startup_timeout),
            effort=effort.strip() if effort is not None else None,
        )
        # The OSS adapter authenticates with a token it finds under a
        # configured env var name; hand it the bridge's key the same way,
        # as a per-agent override, so nothing about the bridge leaks into
        # the process environment.
        overrides = dict(config.agent_specific.get("env_overrides") or {})
        overrides[BRIDGE_KEY_ENV] = self._bridge.master_key
        # The effort lives on the bridge; the CLI flag would be dropped there.
        config.agent_specific = {
            **{k: v for k, v in config.agent_specific.items() if k != "effort"},
            "base_url": self._bridge.base_url,
            "auth_token_env": BRIDGE_KEY_ENV,
            "env_overrides": overrides,
        }
        super().__init__(config)

    def _get_env(self) -> Dict[str, str]:
        env = super()._get_env()
        # The CLI's update checks and telemetry would go to Anthropic, which
        # the bridge is not; keep the child's traffic on the bridge alone.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env
