"""Hermetic tests for shared model routing and LLM retry behavior."""

import asyncio
from types import SimpleNamespace

import pytest

import kapso.core.llm as llm_module
import kapso.execution.experiment_workspace.experiment_workspace as workspace_module
import kapso.kapso as kapso_module
from kapso.core.config import load_mode_config
from kapso.core.llm import (
    LLMBackend,
    LLMRetryError,
    ModelRouter,
    RetryPolicy,
    is_transient_llm_error,
)
from kapso.execution.coding_agents.commit_message_generator import (
    COMMIT_MESSAGE_MODEL,
)
from kapso.execution.memories.experiment_memory.insight_extractor import (
    InsightExtractor,
)
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.knowledge_base.types import Source
from kapso.researcher import Researcher


def response(text, cost=0.0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        _hidden_params={"response_cost": cost},
    )


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class AuthenticationError(Exception):
    pass


def no_jitter_policy(**overrides):
    values = {
        "max_attempts": 2,
        "initial_delay_seconds": 1,
        "max_delay_seconds": 10,
        "multiplier": 2,
        "jitter": False,
    }
    values.update(overrides)
    return RetryPolicy(**values)


def test_model_router_supports_roles_partial_overrides_and_explicit_models():
    router = ModelRouter(
        {
            "utility": "vendor/cheap",
            "web_search": "vendor/search",
        }
    )

    assert router.resolve(None) == "vendor/cheap"
    assert router.resolve("utility") == "vendor/cheap"
    assert router.resolve("reasoning") == "gpt-5-mini"
    assert router.resolve("vendor/custom") == "vendor/custom"
    assert router.resolve("gpt-4.1", default_role="web_search") == (
        "vendor/search"
    )
    assert router.to_dict()["web_search"] == "vendor/search"


def test_model_router_rich_form_resolves_model_and_carries_effort():
    router = ModelRouter(
        {
            "utility": {
                "model": "openai/gpt-5.6-luna",
                "reasoning_effort": "xhigh",
            },
            "reasoning": "vendor/plain",
        }
    )

    assert router.resolve("utility") == "openai/gpt-5.6-luna"
    assert router.resolve(None) == "openai/gpt-5.6-luna"
    assert router.effort_for("utility") == "xhigh"
    assert router.effort_for(None) == "xhigh"
    # Roles configured as bare strings, and explicit model strings, carry none.
    assert router.effort_for("reasoning") is None
    assert router.effort_for("openai/gpt-5.6-luna") is None


@pytest.mark.parametrize(
    "routes,match",
    [
        ({"unknown": "model"}, "Unknown model role"),
        ({"utility": ""}, "non-empty string"),
        ({"reasoning": None}, "non-empty string"),
        ({"utility": {"model": "m", "oops": "x"}}, "unknown keys"),
        ({"utility": {"reasoning_effort": "high"}}, "non-empty string"),
        ({"utility": {"model": "m", "reasoning_effort": ""}}, "non-empty string"),
    ],
)
def test_invalid_model_routes_fail_during_configuration(routes, match):
    with pytest.raises(ValueError, match=match):
        ModelRouter(routes)


def test_role_reasoning_effort_applies_only_when_caller_passes_none(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response("ok")

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(
        models={
            "utility": {
                "model": "openai/gpt-5.6-luna",
                "reasoning_effort": "xhigh",
            }
        }
    )

    backend.llm_completion("utility", [{"role": "user", "content": "x"}])
    assert calls[-1]["reasoning_effort"] == "xhigh"

    backend.llm_completion(
        "utility", [{"role": "user", "content": "x"}], reasoning_effort="low"
    )
    assert calls[-1]["reasoning_effort"] == "low"

    backend.llm_completion("openai/other", [{"role": "user", "content": "x"}])
    assert calls[-1]["reasoning_effort"] is None


def test_retry_policy_computes_capped_backoff_and_full_jitter():
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay_seconds=2,
        max_delay_seconds=5,
        multiplier=2,
        jitter=False,
    )

    assert [policy.delay_for_retry(index) for index in (1, 2, 3)] == [2, 4, 5]

    jittered = RetryPolicy(
        initial_delay_seconds=4,
        max_delay_seconds=10,
        jitter=True,
    )
    assert jittered.delay_for_retry(1, lambda: 0.25) == 1


@pytest.mark.parametrize(
    "config,match",
    [
        ({"max_attempts": 0}, "positive integer"),
        ({"initial_delay_seconds": -1}, "non-negative"),
        (
            {"initial_delay_seconds": 10, "max_delay_seconds": 5},
            "at least initial_delay_seconds",
        ),
        ({"multiplier": 0.5}, "at least 1"),
        ({"jitter": "yes"}, "boolean"),
        ({"unknown": 1}, "Unknown retry setting"),
    ],
)
def test_invalid_retry_policy_fails_during_configuration(config, match):
    with pytest.raises(ValueError, match=match):
        RetryPolicy.from_config(config)


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), StatusError(429), StatusError(503)])
def test_transient_classifier_accepts_transport_throttle_and_server_errors(error):
    assert is_transient_llm_error(error) is True


@pytest.mark.parametrize(
    "error",
    [AuthenticationError(), StatusError(400), ValueError("bad config"), RuntimeError("bug")],
)
def test_transient_classifier_rejects_auth_config_and_programming_errors(error):
    assert is_transient_llm_error(error) is False


def test_sync_completion_retries_transient_error_with_resolved_role(monkeypatch):
    calls = []
    sleeps = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("temporary")
        return response("done", cost=0.25)

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(
        models={"utility": "vendor/utility"},
        retry_policy=no_jitter_policy(initial_delay_seconds=2),
        sleep_fn=sleeps.append,
    )

    result = backend.llm_completion(
        model="utility",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result == "done"
    assert [call["model"] for call in calls] == [
        "vendor/utility",
        "vendor/utility",
    ]
    assert sleeps == [2]
    assert backend.get_cumulative_cost() == 0.25


def test_non_transient_error_is_raised_immediately_without_sleep(monkeypatch):
    error = AuthenticationError("invalid key")
    calls = []
    sleeps = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(
        retry_policy=no_jitter_policy(max_attempts=4),
        sleep_fn=sleeps.append,
    )

    with pytest.raises(AuthenticationError) as exc_info:
        backend.llm_completion("utility", [{"role": "user", "content": "x"}])

    assert exc_info.value is error
    assert len(calls) == 1
    assert sleeps == []


def test_transient_exhaustion_reports_attempt_count(monkeypatch):
    calls = []
    sleeps = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise StatusError(503)

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(
        retry_policy=no_jitter_policy(max_attempts=3),
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMRetryError) as exc_info:
        backend.llm_completion("reasoning", [{"role": "user", "content": "x"}])

    assert exc_info.value.attempts == 3
    assert exc_info.value.model == "gpt-5-mini"
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_parallel_completion_retries_only_the_failed_model(monkeypatch):
    counts = {"stable": 0, "flaky": 0}
    sleeps = []

    async def fake_acompletion(**kwargs):
        model = kwargs["model"]
        counts[model] += 1
        await asyncio.sleep(0)
        if model == "flaky" and counts[model] == 1:
            raise ConnectionError("temporary")
        return response(model, cost=0.1)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    backend = LLMBackend(
        models={"utility": "stable", "reasoning": "flaky"},
        retry_policy=no_jitter_policy(initial_delay_seconds=0.5),
        async_sleep_fn=fake_sleep,
    )

    results = backend.llm_multiple_completions(
        ["utility", "reasoning"],
        [{"role": "user", "content": "hello"}],
    )

    assert results == ["stable", "flaky"]
    assert counts == {"stable": 1, "flaky": 2}
    assert sleeps == [0.5]
    assert backend.get_cumulative_cost() == pytest.approx(0.2)


def test_web_search_uses_configured_role_and_legacy_alias(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return response("search result")

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(models={"web_search": "vendor/search"})

    result = backend.llm_completion_with_web_search(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "latest"}],
        search_context_size="high",
        temperature=0,
    )

    assert result == "search result"
    assert calls[0]["model"] == "vendor/search"
    assert calls[0]["web_search_options"] == {"search_context_size": "high"}
    assert "temperature" not in calls[0]


def test_web_search_uses_shared_transient_retry_policy(monkeypatch):
    calls = []
    sleeps = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise StatusError(429)
        return response("search result")

    monkeypatch.setattr(llm_module, "completion", fake_completion)
    backend = LLMBackend(
        retry_policy=no_jitter_policy(initial_delay_seconds=3),
        sleep_fn=sleeps.append,
    )

    result = backend.llm_completion_with_web_search(
        model="web_search",
        messages=[{"role": "user", "content": "latest"}],
    )

    assert result == "search result"
    assert len(calls) == 2
    assert sleeps == [3]


def test_parallel_web_search_uses_individual_efforts(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return response(kwargs["model"])

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    backend = LLMBackend(models={"web_search": "vendor/search"})

    results = backend.llm_multiple_completions_with_web_search(
        ["web_search", "vendor/other-search"],
        [{"role": "user", "content": "latest"}],
        reasoning_efforts=["low", "high"],
    )

    assert results == ["vendor/search", "vendor/other-search"]
    assert [call["reasoning_effort"] for call in calls] == ["low", "high"]


def test_internal_optional_enrichment_uses_utility_role():
    assert COMMIT_MESSAGE_MODEL == "utility"
    assert InsightExtractor.DEFAULT_MODEL == "utility"
    assert RepoMemoryManager.DEFAULT_REPO_MODEL_LLM == "utility"


def test_experiment_store_uses_shared_backend_for_insights(tmp_path):
    backend = LLMBackend(models={"utility": "vendor/utility"})
    store = ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"),
        llm=backend,
    )

    assert store._get_insight_extractor()._llm is backend


def test_experiment_workspace_threads_shared_backend_to_sessions(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class CapturingSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(workspace_module, "ExperimentSession", CapturingSession)
    workspace = workspace_module.ExperimentWorkspace.__new__(
        workspace_module.ExperimentWorkspace
    )
    workspace.workspace_dir = str(tmp_path)
    workspace.repo = object()
    workspace.coding_agent_config = object()
    workspace.repo_memory_failure_policy = "warn"
    workspace.repo_memory_max_retries = 2
    workspace.llm_backend = object()

    session = workspace.create_experiment_session("candidate", "main")

    assert isinstance(session, CapturingSession)
    assert captured["llm_backend"] is workspace.llm_backend
    assert captured["repo_memory_llm"] is workspace.llm_backend


def test_shipped_mode_config_constructs_shared_backend():
    mode = load_mode_config("src/kapso/config.yaml", "GENERIC")
    backend = LLMBackend(
        models=mode["models"],
        retry_policy=mode["retry"],
    )

    assert backend.resolve_model("utility") == "gpt-4.1-mini"
    assert backend.resolve_model("reasoning") == "gpt-5-mini"
    assert backend.resolve_model(None, default_role="web_search") == (
        "openai/gpt-4o-search-preview"
    )
    assert backend.retry_policy.max_attempts == 2


def test_researcher_routes_web_search_through_shared_backend():
    class CapturingBackend:
        def __init__(self):
            self.calls = []

        def llm_completion_with_web_search(self, **kwargs):
            self.calls.append(kwargs)
            return (
                "<research_result><research_item>"
                "<source>https://example.com</source>"
                "<content>Useful result</content>"
                "</research_item></research_result>"
            )

    backend = CapturingBackend()
    researcher = Researcher(llm_backend=backend)

    ideas = researcher.research(
        "test query",
        mode="idea",
        top_k=1,
        depth="deep",
    )

    assert ideas == [
        Source.Idea(
            query="test query",
            source="https://example.com",
            content="Useful result",
        )
    ]
    call = backend.calls[0]
    assert call["model"] == "web_search"
    assert call["search_context_size"] == "high"
    assert call["reasoning_effort"] == "high"
    assert call["max_tokens"] == 32000
    assert "test query" in call["messages"][0]["content"]


def test_researcher_preserves_explicit_model_and_light_depth():
    class CapturingBackend:
        def __init__(self):
            self.call = None

        def llm_completion_with_web_search(self, **kwargs):
            self.call = kwargs
            return "<research_result>Report</research_result>"

    backend = CapturingBackend()
    researcher = Researcher(
        model="vendor/custom-search",
        llm_backend=backend,
    )

    report = researcher.research("test query", mode="study", depth="light")

    assert report == Source.ResearchReport(query="test query", content="Report")
    assert backend.call["model"] == "vendor/custom-search"
    assert backend.call["search_context_size"] == "medium"
    assert backend.call["reasoning_effort"] == "medium"


def test_researcher_failure_returns_typed_empty_report():
    class FailingBackend:
        def llm_completion_with_web_search(self, **kwargs):
            raise AuthenticationError("bad key")

    researcher = Researcher(llm_backend=FailingBackend())

    report = researcher.research("test query", mode="study", depth="light")

    assert report == Source.ResearchReport(query="test query", content="")


def test_kapso_research_uses_active_mode_routes_and_retry(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
default_mode: CUSTOM
modes:
  CUSTOM:
    models:
      web_search: vendor/search
    retry:
      max_attempts: 4
""".strip(),
        encoding="utf-8",
    )
    created = []

    class CapturingResearcher:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def research(self, objective, **kwargs):
            return objective, kwargs

    monkeypatch.setattr(kapso_module, "Researcher", CapturingResearcher)
    kapso = kapso_module.Kapso(config_path=str(config_path))

    result = kapso.research("objective", mode="study", depth="light")

    assert created == [
        {
            "models": {"web_search": "vendor/search"},
            "retry_policy": {"max_attempts": 4},
        }
    ]
    assert result == (
        "objective",
        {"mode": "study", "depth": "light"},
    )
