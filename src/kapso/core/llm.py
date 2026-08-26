"""Unified LLM model routing, retry behavior, and cost tracking."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

import tiktoken
from litellm import acompletion, completion, embedding

# Suppress verbose LiteLLM logs.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Provider hard cap for embedding inputs (OpenAI text-embedding-3 family):
# requests above this raise a non-transient 400. Structural API limit, not a
# tunable. cl100k_base is the tokenizer for the text-embedding-3 models.
EMBEDDING_MAX_TOKENS = 8192
_EMBEDDING_ENCODING = "cl100k_base"

MODEL_ROLES = frozenset({"utility", "reasoning", "web_search", "embedding"})
DEFAULT_MODEL_ROUTES: Dict[str, str] = {
    "utility": "gpt-4.1-mini",
    "reasoning": "gpt-5-mini",
    # Web search runs through the Responses API web_search TOOL, which the
    # retired search-preview family rejects ("not supported with the
    # Responses API" — a gate whose subprocess fell back to this default
    # 400'd on every call, E2E review 2026-08-24). Any current chat model
    # works; deployments override via the config's models.web_search.
    "web_search": "gpt-4.1-mini",
    "embedding": "text-embedding-3-small",
}


def _effort_kwargs(reasoning_effort: Optional[str]) -> Dict[str, Any]:
    """Completion kwargs for a reasoning effort; empty when none is set.

    A None effort must OMIT the parameter entirely: passing the kwarg with
    a None value can reach the provider as an explicit null once litellm's
    capability map and the allowed-params whitelist interact (run #9,
    R9-I-2: gpt-5.6-luna rejects null with a 400).
    """
    if reasoning_effort is None:
        return {}
    return {
        "reasoning_effort": reasoning_effort,
        **_effort_passthrough(reasoning_effort),
    }


def _effort_passthrough(reasoning_effort: Optional[str]) -> Dict[str, Any]:
    """Force reasoning_effort past litellm's static capability map.

    `drop_params=True` silently discards reasoning_effort for models newer
    than the installed litellm's model registry (e.g. the gpt-5.6 family),
    which would quietly ignore a configured effort level. Whitelisting the
    parameter keeps it in the request while drop_params still prunes anything
    else unsupported.
    """
    if reasoning_effort is None:
        return {}
    return {"allowed_openai_params": ["reasoning_effort"]}


_ANTHROPIC_ROUTE_HINTS = ("anthropic", "claude")


def _prepare_effort(
    model: Optional[str],
    reasoning_effort: Optional[str],
    kwargs: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Translate effort levels litellm's Anthropic mapper does not know.

    litellm maps reasoning_effort for Anthropic-routed models but raises on
    levels outside {low, medium, high} (e.g. "xhigh"). Current Claude models
    (Opus 4.8+) control this natively via adaptive thinking plus
    output_config.effort — send those verbatim through Bedrock's request
    pass-through so litellm neither validates nor rewrites them.
    Non-Anthropic models pass through unchanged.
    """
    if reasoning_effort != "xhigh":
        return reasoning_effort, kwargs
    lowered = (model or "").lower()
    if not any(hint in lowered for hint in _ANTHROPIC_ROUTE_HINTS):
        return reasoning_effort, kwargs
    # litellm forwards these into the provider request body verbatim
    # (unknown kwargs are collected into Bedrock's additionalModelRequestFields).
    kwargs = dict(kwargs)
    kwargs.setdefault("thinking", {"type": "adaptive"})
    kwargs.setdefault("output_config", {"effort": "xhigh"})
    kwargs.setdefault("max_tokens", 16384)
    return None, kwargs

# These inputs were historically rewritten by the web-search methods. They
# remain aliases, but now target the configured web_search role.
LEGACY_WEB_SEARCH_ALIASES = frozenset(
    {"gpt-5", "gpt-5.1", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"}
)


class ModelRouter:
    """Resolve semantic model roles while preserving explicit model strings.

    A route value is either a bare model string or a mapping
    {model: <str>, reasoning_effort: <str>} — the rich form attaches a
    default reasoning effort to every call resolved through that role
    (callers passing an explicit effort still win). This is how config
    reaches call sites that never plumbed an effort parameter (e.g.
    repo memory).
    """

    def __init__(self, routes: Optional[Mapping[str, Any]] = None):
        supplied = dict(routes or {})
        unknown = sorted(set(supplied) - MODEL_ROLES)
        if unknown:
            raise ValueError(f"Unknown model role(s): {', '.join(unknown)}")

        merged = dict(DEFAULT_MODEL_ROUTES)
        efforts: Dict[str, str] = {}
        for role, value in supplied.items():
            if isinstance(value, Mapping):
                model = value.get("model")
                extra = sorted(set(value) - {"model", "reasoning_effort"})
                if extra:
                    raise ValueError(
                        f"Model route '{role}' has unknown keys: {', '.join(extra)}"
                    )
                effort = value.get("reasoning_effort")
                if effort is not None:
                    if not isinstance(effort, str) or not effort.strip():
                        raise ValueError(
                            f"Model route '{role}' reasoning_effort must be a non-empty string"
                        )
                    efforts[role] = effort.strip()
            else:
                model = value
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Model route '{role}' must be a non-empty string")
            merged[role] = model.strip()
        self._routes = merged
        self._efforts = efforts

    def resolve(
        self,
        model: Optional[str],
        *,
        default_role: str = "utility",
    ) -> str:
        if default_role not in MODEL_ROLES:
            raise ValueError(f"Unknown default model role: {default_role}")
        if model is None:
            return self._routes[default_role]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string or None")

        requested = model.strip()
        if requested in MODEL_ROLES:
            return self._routes[requested]
        if (
            default_role == "web_search"
            and requested in LEGACY_WEB_SEARCH_ALIASES
        ):
            return self._routes["web_search"]
        return requested

    def effort_for(
        self, model: Optional[str], *, default_role: str = "utility"
    ) -> Optional[str]:
        """The configured effort for whichever role this call resolves to."""
        if model is None:
            return self._efforts.get(default_role)
        requested = str(model).strip()
        if requested in MODEL_ROLES:
            return self._efforts.get(requested)
        return None

    def to_dict(self) -> Dict[str, str]:
        return dict(self._routes)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff shared by every completion surface."""

    max_attempts: int = 2
    initial_delay_seconds: float = 5.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    # Hard per-request wall clock handed to the provider client. Without it
    # a connection the server abandons mid-stream blocks the calling thread
    # forever: the first live wedge (rel-event/user-ignore re-run,
    # 2026-08-09) held three CLOSE-WAIT sockets for 50 minutes with the
    # campaign frozen. A timeout surfaces as APITimeoutError/Timeout, which
    # is classified transient and consumed by this same retry loop.
    request_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise ValueError("retry.max_attempts must be a positive integer")
        if self.max_attempts < 1:
            raise ValueError("retry.max_attempts must be a positive integer")

        numeric_fields = {
            "initial_delay_seconds": self.initial_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "multiplier": self.multiplier,
            "request_timeout_seconds": self.request_timeout_seconds,
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"retry.{name} must be numeric")
        if self.initial_delay_seconds < 0:
            raise ValueError("retry.initial_delay_seconds must be non-negative")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "retry.max_delay_seconds must be at least initial_delay_seconds"
            )
        if self.multiplier < 1:
            raise ValueError("retry.multiplier must be at least 1")
        if not isinstance(self.jitter, bool):
            raise ValueError("retry.jitter must be a boolean")
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "retry.request_timeout_seconds must be positive"
            )

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any] | "RetryPolicy"],
    ) -> "RetryPolicy":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise ValueError("retry configuration must be a mapping")

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"Unknown retry setting(s): {', '.join(unknown)}")
        return cls(**dict(config))

    def delay_for_retry(
        self,
        retry_number: int,
        random_fn: Callable[[], float] = random.random,
    ) -> float:
        """Return delay before retry 1, 2, ... using capped full jitter."""
        if retry_number < 1:
            raise ValueError("retry_number must be at least 1")
        delay = min(
            self.max_delay_seconds,
            self.initial_delay_seconds
            * (self.multiplier ** (retry_number - 1)),
        )
        return delay * random_fn() if self.jitter else delay

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMRetryError(RuntimeError):
    """A transient LLM call exhausted its configured attempts."""

    def __init__(self, operation: str, model: str, attempts: int, cause: Exception):
        self.operation = operation
        self.model = model
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"Transient {operation} failed for model {model} after "
            f"{attempts} attempt(s): {type(cause).__name__}"
        )


_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }
)
_NON_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "ContextWindowExceededError",
        "NotFoundError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
    }
)


def _status_code(error: Exception) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_transient_llm_error(error: Exception) -> bool:
    """Classify retryable transport, throttling, and server failures."""
    if isinstance(error, (TypeError, ValueError, AssertionError)):
        return False

    current: Optional[BaseException] = error
    seen = set()
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        status = _status_code(current)
        if status is not None:
            return status in _TRANSIENT_STATUS_CODES

        name = type(current).__name__
        if name in _NON_TRANSIENT_EXCEPTION_NAMES:
            return False
        if name in _TRANSIENT_EXCEPTION_NAMES:
            return True
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True

        current = current.__cause__ or current.__context__
    return False


class LLMBackend:
    """LLM completions with role routing, bounded retries, and cost tracking."""

    def __init__(
        self,
        models: Optional[Mapping[str, str] | ModelRouter] = None,
        retry_policy: Optional[Mapping[str, Any] | RetryPolicy] = None,
        *,
        sleep_fn: Optional[Callable[[float], None]] = None,
        async_sleep_fn: Optional[Callable[[float], Awaitable[None]]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ):
        self.model_router = (
            models if isinstance(models, ModelRouter) else ModelRouter(models)
        )
        self.retry_policy = RetryPolicy.from_config(retry_policy)
        self._sleep = sleep_fn or time.sleep
        self._async_sleep = async_sleep_fn or asyncio.sleep
        self._random = random_fn or random.random
        self._cumulative_cost = 0.0

    def get_cumulative_cost(self) -> float:
        return self._cumulative_cost

    def resolve_model(
        self,
        model: Optional[str],
        *,
        default_role: str = "utility",
    ) -> str:
        return self.model_router.resolve(model, default_role=default_role)

    def _record_cost(self, response: Any) -> None:
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, Mapping):
            cost = hidden.get("response_cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self._cumulative_cost += float(cost)

    @staticmethod
    def _content(response: Any) -> str:
        return response.choices[0].message.content

    def _run_sync(
        self,
        operation: str,
        model: str,
        call: Callable[[], Any],
    ) -> Any:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                response = call()
                self._record_cost(response)
                return response
            except KeyboardInterrupt:
                raise
            except Exception as error:
                if not is_transient_llm_error(error):
                    raise
                if attempt == self.retry_policy.max_attempts:
                    raise LLMRetryError(
                        operation, model, attempt, error
                    ) from error
                delay = self.retry_policy.delay_for_retry(
                    attempt, self._random
                )
                logger.warning(
                    "Transient %s failure for model %s (%d/%d, %s); "
                    "retrying in %.2fs",
                    operation,
                    model,
                    attempt,
                    self.retry_policy.max_attempts,
                    type(error).__name__,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("retry loop exited unexpectedly")

    async def _run_async(
        self,
        operation: str,
        model: str,
        call: Callable[[], Awaitable[Any]],
    ) -> Any:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                response = await call()
                self._record_cost(response)
                return response
            except Exception as error:
                if not is_transient_llm_error(error):
                    raise
                if attempt == self.retry_policy.max_attempts:
                    raise LLMRetryError(
                        operation, model, attempt, error
                    ) from error
                delay = self.retry_policy.delay_for_retry(
                    attempt, self._random
                )
                logger.warning(
                    "Transient %s failure for model %s (%d/%d, %s); "
                    "retrying in %.2fs",
                    operation,
                    model,
                    attempt,
                    self.retry_policy.max_attempts,
                    type(error).__name__,
                    delay,
                )
                await self._async_sleep(delay)
        raise AssertionError("retry loop exited unexpectedly")

    # Completions were removed 2026-08-26 (cli-only-inference-design):
    # every non-embedding completion now runs as a coding-agent CLI
    # session via kapso.core.cli_inference.CliInference. This backend
    # survives for embeddings, model-role resolution, and the cost
    # meter only.

    def create_embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Embed the text via the router's embedding role.

        Inputs beyond the provider's EMBEDDING_MAX_TOKENS hard cap are
        truncated to the cap (user-approved exception to the no-truncation
        rule, 2026-07-28: an over-cap request 400s non-transiently and kills
        the campaign at bookkeeping; a prefix embedding of an already
        model-authored solution is an acceptable similarity key). Transient
        provider failures retry under the backend's policy; genuine errors
        propagate.
        """
        encoder = tiktoken.get_encoding(_EMBEDDING_ENCODING)
        tokens = encoder.encode(text)
        if len(tokens) > EMBEDDING_MAX_TOKENS:
            logger.warning(
                "Embedding input truncated from %d to %d tokens",
                len(tokens),
                EMBEDDING_MAX_TOKENS,
            )
            text = encoder.decode(tokens[:EMBEDDING_MAX_TOKENS])
        resolved = self.model_router.resolve(model, default_role="embedding")
        response = self._run_sync(
            "embedding",
            resolved,
            lambda: embedding(
                model=resolved,
                input=[text],
                timeout=self.retry_policy.request_timeout_seconds,
            ),
        )
        return list(response.data[0]["embedding"])


def main() -> None:
    llm = LLMBackend()
    response = llm.llm_completion(
        model="reasoning",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print(response)
    print(f"Cost: ${llm.get_cumulative_cost():.6f}")


if __name__ == "__main__":
    main()
