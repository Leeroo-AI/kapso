"""Hermetic tests for the text-only OpenAI-compatible adapter."""

import json
from types import SimpleNamespace

import httpx
from openai import OpenAI

from kapso.core.cli_inference import CliInference
from kapso.core import api_endpoint
from kapso.core.api_endpoint import openai_compatible_options

import pytest

from kapso.execution.coding_agents.adapters import (
    openai_compatible_agent as agent_module,
)
from kapso.execution.coding_agents.adapters.openai_compatible_agent import (
    OpenAICompatibleInferenceAgent,
)
from kapso.execution.coding_agents.base import CodingAgentConfig


class FakeCompletions:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.output))
            ],
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=5, total_tokens=8
            ),
        )


class FakeClient:
    def close(self):
        self.closed = True

    def __init__(self, **kwargs):
        self.options = kwargs
        self.chat = SimpleNamespace(
            completions=FakeCompletions(
                "<<<FILE: src/app.py>>>\nprint('ok')\n<<<END FILE>>>"
            )
        )


def make_agent(monkeypatch, tmp_path, **agent_specific):
    monkeypatch.setattr(agent_module, "OpenAI", FakeClient)
    config = CodingAgentConfig(
        agent_type="openai_compatible",
        model="provider/model",
        debug_model="provider/debug-model",
        agent_specific=agent_specific,
    )
    agent = OpenAICompatibleInferenceAgent(config)
    agent.initialize(str(tmp_path))
    return agent


def test_custom_endpoint_and_named_key_are_used(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER_KEY", "secret")
    agent = make_agent(
        monkeypatch,
        tmp_path,
        base_url="https://api.example.test/v1",
        api_key_env="PROVIDER_KEY",
    )

    result = agent.generate_code("create the app")

    assert result.success
    assert result.files_changed == []
    assert not list(tmp_path.iterdir())
    assert agent._client.options == {
        "api_key": "secret",
        "max_retries": 0,
        "base_url": "https://api.example.test/v1",
    }
    call = agent._client.chat.completions.calls[0]
    assert call["model"] == "provider/model"
    assert call["messages"][1]["content"] == "create the app"
    assert result.metadata["usage"]["total_tokens"] == 8


def test_debug_model_ignores_environment_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    agent = make_agent(monkeypatch, tmp_path)

    agent.generate_code("debug it", debug_mode=True)

    assert (
        agent._client.options["base_url"]
        == openai_compatible_options()["base_url"]
    )
    assert (
        agent._client.chat.completions.calls[0]["model"]
        == "provider/debug-model"
    )


def test_missing_key_fails_loud(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_agent(monkeypatch, tmp_path)


def test_local_endpoint_can_opt_out_of_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = make_agent(
        monkeypatch,
        tmp_path,
        base_url="http://localhost:11434/v1",
        allow_missing_api_key=True,
    )
    assert agent._client.options["api_key"] == "not-needed"


@pytest.mark.parametrize("reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_completion_is_not_applied(monkeypatch, tmp_path, reason):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    completion = agent._client.chat.completions
    original = completion.create

    def create(**kwargs):
        response = original(**kwargs)
        response.choices[0].finish_reason = reason
        return response

    monkeypatch.setattr(completion, "create", create)
    result = agent.generate_code("implement")
    assert not result.success
    assert reason in result.error
    assert not list(tmp_path.iterdir())


def test_configured_timeout_and_reasoning_request_options(
    monkeypatch, tmp_path
):
    agent = make_agent(
        monkeypatch,
        tmp_path,
        allow_missing_api_key=True,
        timeout=37,
        temperature=None,
        request_options={"max_completion_tokens": 1000},
    )
    assert agent.generate_code("implement").success
    call = agent._client.chat.completions.calls[-1]
    assert call["timeout"] == 37
    assert call["max_completion_tokens"] == 1000
    assert "max_tokens" not in call and "temperature" not in call
    agent.generate_code("debug", debug_mode=True, timeout_seconds=2)
    assert agent._client.chat.completions.calls[-1]["timeout"] == 2
    assert agent._client.options["max_retries"] == 0


@pytest.mark.parametrize(
    "option",
    [
        "model",
        "messages",
        "stream",
        "timeout",
        "tools",
        "extra_body",
        "extra_headers",
        "extra_query",
    ],
)
def test_request_options_cannot_bypass_session_contract(
    monkeypatch, tmp_path, option
):
    with pytest.raises(ValueError, match="session fields"):
        make_agent(
            monkeypatch,
            tmp_path,
            allow_missing_api_key=True,
            request_options={option: True},
        )


def test_cleanup_closes_client(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    agent.cleanup()
    assert agent._client.closed
    assert agent.workspace is None


def test_real_sdk_request_and_response(monkeypatch, tmp_path):
    """Exercise SDK serialization and parsing without a provider or API key."""

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "provider/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"ok": true}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            },
        )

    def client(**kwargs):
        return OpenAI(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.test/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://legacy.test/v1")
    monkeypatch.setattr(agent_module, "OpenAI", client)
    backend = CliInference(
        inference={
            "default": {
                "cli": "openai_compatible",
                "model": "provider/model",
                "timeout_seconds": 19,
                "agent_specific": {
                    "base_url": "https://provider.test/v1",
                    "api_key_env": "TEST_PROVIDER_KEY",
                },
            },
        }
    )
    assert (
        backend.llm_completion(messages=[{"role": "user", "content": "JSON"}])
        == '{"ok": true}'
    )
    assert str(requests[0].url) == "https://provider.test/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["messages"][1]["content"] == "JSON"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.2/v1",
        "http://[::1]:8000/v1",
        "https://provider.test/v1",
    ],
)
def test_configured_secure_or_loopback_endpoint(
    monkeypatch, tmp_path, base_url
):
    agent = make_agent(
        monkeypatch, tmp_path, base_url=base_url, allow_missing_api_key=True
    )
    assert agent._client.options["base_url"] == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "",
        "http://provider.test/v1",
        "http://192.168.1.1/v1",
        "http://[::]/v1",
        "http://localhost.evil.test/v1",
        "ftp://localhost/v1",
        "https://user:secret@provider.test/v1",
        "https://provider.test/v1?api_key=secret",
    ],
)
def test_unsafe_endpoint_rejected_before_client(
    monkeypatch, tmp_path, base_url
):
    def forbidden_client(**kwargs):
        pytest.fail("unsafe endpoint reached the SDK")

    monkeypatch.setattr(agent_module, "OpenAI", forbidden_client)
    with pytest.raises(ValueError):
        OpenAICompatibleInferenceAgent(
            CodingAgentConfig(
                agent_type="openai_compatible",
                model="m",
                debug_model="m",
                agent_specific={
                    "base_url": base_url,
                    "allow_missing_api_key": True,
                },
            )
        )


@pytest.mark.parametrize("kind", ["empty", "no_choices", "tool_calls"])
def test_invalid_completion_returns_failure(monkeypatch, tmp_path, kind):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    completion = agent._client.chat.completions
    original = completion.create

    def create(**kwargs):
        response = original(**kwargs)
        if kind == "no_choices":
            response.choices = []
        elif kind == "empty":
            response.choices[0].message.content = " "
        else:
            response.choices[0].message.tool_calls = [{"id": "call"}]
        return response

    monkeypatch.setattr(completion, "create", create)
    result = agent.generate_code("test")
    assert not result.success and result.error
    assert not list(tmp_path.iterdir())


def test_transport_error_propagates(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)

    def fail(**kwargs):
        raise RuntimeError("transport failure")

    monkeypatch.setattr(agent._client.chat.completions, "create", fail)
    with pytest.raises(RuntimeError, match="transport failure"):
        agent.generate_code("test")


def test_adapter_defaults_follow_manifest(monkeypatch, tmp_path):
    read_config = api_endpoint.load_agent_manifest

    def changed_config():
        manifest = read_config()
        manifest["agents"]["openai_compatible"]["agent_specific"].update(
            {
                "max_output_tokens": 321,
                "temperature": 0.7,
                "timeout": 29,
            }
        )
        return manifest

    monkeypatch.setattr(api_endpoint, "load_agent_manifest", changed_config)
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    agent.generate_code("test")
    call = agent._client.chat.completions.calls[-1]
    assert (call["max_tokens"], call["temperature"], call["timeout"]) == (
        321,
        0.7,
        29,
    )
