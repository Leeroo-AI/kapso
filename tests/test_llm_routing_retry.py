"""Hermetic tests for shared model routing and LLM retry behavior."""

from types import SimpleNamespace

import pytest

import kapso.core.llm as llm_module
from kapso.core.llm import (
    LLMBackend,
    ModelRouter,
    RetryPolicy,
    is_transient_llm_error,
)
from kapso.researcher import Researcher


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


def test_model_router_is_embedding_only_after_cli_conversion():
    # Every non-embedding role died with the completion surface
    # (cli-only-inference §5); the router resolves the embedding route
    # and preserves explicit model strings, nothing else.
    router = ModelRouter({"embedding": "vendor/embedder"})

    assert router.resolve(None) == "vendor/embedder"
    assert router.resolve("embedding") == "vendor/embedder"
    assert router.resolve("vendor/custom") == "vendor/custom"
    assert router.to_dict() == {"embedding": "vendor/embedder"}
    # Retired roles are now configuration errors, not silent routes.
    with pytest.raises(ValueError, match="Unknown model role"):
        ModelRouter({"utility": "vendor/cheap"})
    with pytest.raises(ValueError, match="Unknown default model role"):
        router.resolve(None, default_role="web_search")


@pytest.mark.parametrize(
    "routes,match",
    [
        ({"unknown": "model"}, "Unknown model role"),
        ({"embedding": ""}, "non-empty string"),
        ({"embedding": None}, "non-empty string"),
        # The rich {model, reasoning_effort} form died with completions.
        ({"embedding": {"model": "m"}}, "non-empty string"),
    ],
)
def test_invalid_model_routes_fail_during_configuration(routes, match):
    with pytest.raises(ValueError, match=match):
        ModelRouter(routes)


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


def test_researcher_failure_propagates():
    # Fail loud: a research failure returned as empty findings poisons
    # every downstream consumer silently — learn_knowledge ingested
    # nothing and reported success (found live 2026-08-24 by the facade
    # E2E). Provider errors must reach the caller.
    class FailingBackend:
        def llm_completion_with_web_search(self, **kwargs):
            raise AuthenticationError("bad key")

    researcher = Researcher(llm_backend=FailingBackend())

    with pytest.raises(AuthenticationError, match="bad key"):
        researcher.research("test query", mode="study", depth="light")


def test_request_timeout_must_be_positive():
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        LLMBackend(retry_policy={"request_timeout_seconds": 0})


def embedding_response(vector, cost=0.0):
    return SimpleNamespace(
        data=[{"embedding": list(vector)}],
        _hidden_params={"response_cost": cost},
    )


def test_create_embedding_routes_role_and_caps_at_provider_limit(monkeypatch):
    """Embedding inputs are truncated to the provider's 8192-token hard cap
    (user-approved 2026-07-28): an over-cap request raises a non-transient
    400 that would kill a campaign at bookkeeping. Under-cap inputs pass
    through byte-identical."""
    import tiktoken

    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return embedding_response([0.1, 0.2], cost=0.001)

    monkeypatch.setattr(llm_module, "embedding", fake_embedding)
    backend = LLMBackend(
        models={"embedding": "text-embedding-3-small"},
        retry_policy=no_jitter_policy(),
    )

    encoder = tiktoken.get_encoding("cl100k_base")
    over_cap = "solution body " * 5000  # ~15k tokens, over the 8192 cap
    vector = backend.create_embedding(over_cap)
    assert vector == [0.1, 0.2]
    assert calls[0]["model"] == "text-embedding-3-small"
    sent = calls[0]["input"][0]
    assert len(encoder.encode(sent)) == llm_module.EMBEDDING_MAX_TOKENS
    assert sent == encoder.decode(
        encoder.encode(over_cap)[: llm_module.EMBEDDING_MAX_TOKENS]
    )

    under_cap = "short document"
    backend.create_embedding(under_cap)
    assert calls[1]["input"] == [under_cap]
    assert backend.get_cumulative_cost() == pytest.approx(0.002)


def test_create_embedding_default_role_and_explicit_override(monkeypatch):
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return embedding_response([1.0])

    monkeypatch.setattr(llm_module, "embedding", fake_embedding)
    backend = LLMBackend(retry_policy=no_jitter_policy())

    backend.create_embedding("a")
    backend.create_embedding("b", model="custom-embedder")

    assert calls[0]["model"] == "text-embedding-3-small"  # router default
    assert calls[1]["model"] == "custom-embedder"          # explicit wins


def test_create_embedding_retries_transient_then_raises_loud(monkeypatch):
    attempts = []

    def fake_embedding(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise StatusError(503)
        return embedding_response([0.5])

    monkeypatch.setattr(llm_module, "embedding", fake_embedding)
    backend = LLMBackend(
        retry_policy=no_jitter_policy(max_attempts=2),
        sleep_fn=lambda _s: None,
    )
    assert backend.create_embedding("x") == [0.5]
    assert len(attempts) == 2

    def auth_error(**kwargs):
        raise AuthenticationError("no credentials")

    monkeypatch.setattr(llm_module, "embedding", auth_error)
    with pytest.raises(AuthenticationError):
        backend.create_embedding("x")


def test_completion_surface_is_gone_from_llm_backend():
    # Rule 7: the CLI conversion deleted the direct-completion surface;
    # a resurrected method would mean a dual inference path.
    backend = LLMBackend()
    for name in (
        "llm_completion",
        "llm_completion_with_system_prompt",
        "llm_multiple_completions",
        "llm_completion_with_web_search",
        "llm_multiple_completions_with_web_search",
    ):
        assert not hasattr(backend, name), name
