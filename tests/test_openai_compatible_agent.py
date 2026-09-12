"""Hermetic tests for the OpenAI-compatible coding-agent adapter."""

from types import SimpleNamespace

import pytest

from kapso.execution.coding_agents.adapters import (
    openai_compatible_agent as agent_module,
)
from kapso.execution.coding_agents.adapters.openai_compatible_agent import (
    OpenAICompatibleCodingAgent,
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
    agent = OpenAICompatibleCodingAgent(config)
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
    assert (tmp_path / "src/app.py").read_text() == "print('ok')\n"
    assert agent._client.options == {
        "api_key": "secret",
        "max_retries": 0,
        "base_url": "https://api.example.test/v1",
    }
    call = agent._client.chat.completions.calls[0]
    assert call["model"] == "provider/model"
    assert call["messages"][1]["content"] == "create the app"
    assert result.metadata["usage"]["total_tokens"] == 8


def test_debug_model_and_environment_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    agent = make_agent(monkeypatch, tmp_path)

    agent.generate_code("debug it", debug_mode=True)

    assert agent._client.options["base_url"] == "http://localhost:8000/v1"
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


def test_model_cannot_write_outside_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    agent = make_agent(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="unsafe file path"):
        agent._write_files("<<<FILE: ../outside.py>>>\nx\n<<<END FILE>>>")


@pytest.mark.parametrize(
    "output",
    [
        "No changes required.",
        "```python app.py\nprint(1)\n```",
        "<<<FILE: app.py>>>\npartial",
        "<<<FILE: ok.py>>>\nx\n<<<END FILE>>>\n<<<FILE: broken.py>>>\n",
        "<<<FILE: ok.py>>>\nx\n<<<END FILE>>>\n"
        "<<<FILE: ../bad>>>\nx\n<<<END FILE>>>",
        "<<<FILE: ok.py>>>\nx\n<<<END FILE>>>\n"
        "<<<FILE: ok.py>>>\ny\n<<<END FILE>>>",
        "<<<FILE: a>>>\nx\n<<<END FILE>>>\n<<<FILE: a/b>>>\ny\n<<<END FILE>>>",
    ],
)
def test_invalid_response_does_not_write_any_files(
    monkeypatch, tmp_path, output
):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    agent._client.chat.completions.output = output
    result = agent.generate_code("implement")
    assert not result.success
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["../outside", ".git", "/tmp"])
def test_symlink_escape_is_rejected(monkeypatch, tmp_path, target):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    result = agent.generate_code("implement")
    assert result.success
    with pytest.raises(ValueError, match="unsafe file path"):
        agent._write_files("<<<FILE: link/config>>>\nx\n<<<END FILE>>>")


def test_file_contents_keep_leading_whitespace(monkeypatch, tmp_path):
    agent = make_agent(monkeypatch, tmp_path, allow_missing_api_key=True)
    output = "<<<FILE: app.py>>>\n\n    indented\n<<<END FILE>>>\n"
    agent._write_files(output)
    assert (tmp_path / "app.py").read_text() == "\n    indented\n"
    assert agent._write_files(output) == []


def test_read_only_completion_never_interprets_files(monkeypatch, tmp_path):
    agent = make_agent(
        monkeypatch, tmp_path, allow_missing_api_key=True, read_only=True
    )
    result = agent.generate_code("explain the file format")
    assert result.success and result.files_changed == []
    assert not list(tmp_path.iterdir())
    system = agent._client.chat.completions.calls[0]["messages"][0]["content"]
    assert "Return each changed file" not in system


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
    "option", ["model", "messages", "stream", "timeout", "tools", "extra_body"]
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
    import json
    import httpx
    from openai import OpenAI
    from kapso.core.cli_inference import CliInference

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
