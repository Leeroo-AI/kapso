"""Hermetic tests for the vertex_claude_code adapter.

Contract under test: the adapter starts ONE local LiteLLM bridge per
(project, location, model) in the process, hands the Claude Code CLI the
bridge as its Anthropic endpoint with a key minted for this process, and
otherwise behaves as the OSS adapter — first-party credentials stripped,
model slots pinned. A fake ``litellm`` executable stands in for the proxy:
it serves the health endpoint and records the config it was given.
"""

import json
import os
import stat
import sys
import urllib.request

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters import vertex_claude_code_agent as module
from kapso.execution.coding_agents.adapters.oss_claude_code_agent import (
    FIRST_PARTY_ENV_VARS,
)
from kapso.execution.coding_agents.adapters.vertex_claude_code_agent import (
    BRIDGE_KEY_ENV,
    VertexClaudeCodeCodingAgent,
)

MODEL = "zai-org/glm-5.2-maas"
PROJECT = "example-project"

FAKE_LITELLM = '''#!/usr/bin/env python3
"""Stands in for `litellm --config C --host H --port P`: serves the health
route and copies the config next to it so a test can read what it got."""
import json, shutil, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
config = args[args.index("--config") + 1]
port = int(args[args.index("--port") + 1])
shutil.copy(config, config + ".seen")
with open(config + ".env", "w") as handle:
    import os
    json.dump({"cwd": os.getcwd(), **{k: v for k, v in os.environ.items() if k.startswith("GOOGLE_")}}, handle)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health/liveliness" else 404)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


@pytest.fixture
def fake_litellm(tmp_path, monkeypatch):
    script = tmp_path / "litellm"
    script.write_text(FAKE_LITELLM)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    binaries = {"claude": "/usr/bin/claude", "litellm": str(script)}
    for target in (
        "kapso.execution.coding_agents.adapters.vertex_claude_code_agent.shutil.which",
        "kapso.execution.coding_agents.adapters.oss_claude_code_agent.shutil.which",
    ):
        monkeypatch.setattr(target, lambda command: binaries.get(command))
    monkeypatch.setattr(module, "ADC_PATH", tmp_path / "adc.json")
    (tmp_path / "adc.json").write_text("{}")
    for name in FIRST_PARTY_ENV_VARS + ("ANTHROPIC_AUTH_TOKEN", BRIDGE_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_BRIDGES", {})
    yield script
    for bridge in list(module._BRIDGES.values()):
        bridge.stop()


def make_agent(**overrides):
    agent_specific = {
        "vertex_project": PROJECT,
        "vertex_location": "global",
        "bridge_startup_timeout": 20,
        **overrides,
    }
    return VertexClaudeCodeCodingAgent(CodingAgentConfig(
        agent_type="vertex_claude_code", model=MODEL, debug_model=MODEL,
        agent_specific=agent_specific,
    ))


def test_bridge_serves_the_model_through_litellm_and_the_cli_talks_only_to_it(
    fake_litellm, monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-first-party")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    agent = make_agent()
    bridge = agent._bridge

    seen = json.loads(next(bridge._dir.glob("*.seen")).read_text())
    assert seen["model_list"] == [{
        "model_name": MODEL,
        "litellm_params": {
            "model": f"vertex_ai/{MODEL}",
            "vertex_project": PROJECT,
            "vertex_location": "global",
        },
    }]
    assert seen["general_settings"]["master_key"] == bridge.master_key
    assert len(bridge.master_key) >= 32
    bridge_env = json.loads(next(bridge._dir.glob("*.env")).read_text())
    assert bridge_env["GOOGLE_CLOUD_QUOTA_PROJECT"] == PROJECT
    # The parameter-whitelist hook is loaded from the proxy's own directory.
    assert seen["litellm_settings"]["callbacks"] == ["vertex_bridge_litellm_hook.hook"]
    assert "VertexAILlama3Config" in (bridge._dir / "vertex_bridge_litellm_hook.py").read_text()
    assert bridge_env["cwd"] == str(bridge._dir)
    with urllib.request.urlopen(f"{bridge.base_url}/health/liveliness") as response:
        assert response.status == 200

    env = agent._get_env()
    assert env["ANTHROPIC_BASE_URL"] == bridge.base_url
    assert env["ANTHROPIC_AUTH_TOKEN"] == bridge.master_key
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == MODEL
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert name not in env
    # The key is handed to the child only; the process environment is untouched.
    assert BRIDGE_KEY_ENV not in os.environ


def test_agents_with_the_same_target_share_one_bridge(fake_litellm):
    first = make_agent()
    second = make_agent()
    assert second._bridge is first._bridge
    other_model = VertexClaudeCodeCodingAgent(CodingAgentConfig(
        agent_type="vertex_claude_code", model="qwen/qwen3-coder-480b-a35b-instruct-maas",
        debug_model="qwen/qwen3-coder-480b-a35b-instruct-maas",
        agent_specific={"vertex_project": PROJECT, "vertex_location": "global",
                        "bridge_startup_timeout": 20},
    ))
    assert other_model._bridge is not first._bridge
    assert other_model._bridge.port != first._bridge.port


def test_effort_becomes_the_bridge_reasoning_effort_not_a_cli_flag(fake_litellm):
    agent = make_agent(effort="low")
    seen = json.loads(next(agent._bridge._dir.glob("*.seen")).read_text())
    assert seen["model_list"][0]["litellm_params"]["reasoning_effort"] == "low"
    assert "--effort" not in agent._build_command(MODEL)
    # A different effort is a different deployment, so a different bridge.
    assert make_agent(effort="max")._bridge is not agent._bridge
    assert make_agent(effort="low")._bridge is agent._bridge

    plain = make_agent()
    seen = json.loads(next(plain._bridge._dir.glob("*.seen")).read_text())
    assert "reasoning_effort" not in seen["model_list"][0]["litellm_params"]
    # Claude's top level is the config language; Vertex only knows GLM's.
    xhigh = make_agent(effort="xhigh")
    seen = json.loads(next(xhigh._bridge._dir.glob("*.seen")).read_text())
    assert seen["model_list"][0]["litellm_params"]["reasoning_effort"] == "max"


def test_stop_ends_the_proxy_and_removes_its_files(fake_litellm):
    bridge = make_agent()._bridge
    process, directory = bridge._process, bridge._dir
    bridge.stop()
    assert process.poll() is not None
    assert not directory.exists()


@pytest.mark.parametrize("overrides, message", [
    ({"vertex_project": None}, "vertex_project"),
    ({"vertex_project": "  "}, "vertex_project"),
    ({"vertex_location": ""}, "vertex_location"),
    ({"bridge_startup_timeout": 0}, "bridge_startup_timeout"),
    ({"bridge_startup_timeout": True}, "bridge_startup_timeout"),
    ({"effort": ""}, "effort"),
    ({"effort": 3}, "effort"),
])
def test_bad_wiring_fails_loud_before_any_proxy_starts(fake_litellm, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_agent(**overrides)
    assert module._BRIDGES == {}


def test_missing_credentials_fail_loud_before_any_proxy_starts(fake_litellm, tmp_path):
    (tmp_path / "adc.json").unlink()
    with pytest.raises(ValueError, match="application-default"):
        make_agent()
    assert module._BRIDGES == {}


def test_missing_litellm_fails_loud(fake_litellm, monkeypatch):
    monkeypatch.setattr(module.shutil, "which", lambda command: None)
    with pytest.raises(RuntimeError, match="litellm"):
        make_agent()


def test_a_proxy_that_dies_reports_its_log(fake_litellm, tmp_path):
    fake_litellm.write_text(
        "#!/usr/bin/env python3\nimport sys\nprint('boom: bad config')\nsys.exit(3)\n"
    )
    with pytest.raises(RuntimeError, match="exited with code 3") as failure:
        make_agent()
    assert "boom: bad config" in str(failure.value)


def test_a_proxy_that_never_listens_times_out(fake_litellm):
    fake_litellm.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
    with pytest.raises(RuntimeError, match="did not start serving"):
        make_agent(bridge_startup_timeout=1)
    assert all(bridge._process.poll() is not None for bridge in module._BRIDGES.values())
