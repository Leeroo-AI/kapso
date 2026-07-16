"""Unified LLM model routing, retry behavior, and cost tracking."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from litellm import acompletion, completion

# Suppress verbose LiteLLM logs.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


MODEL_ROLES = frozenset({"utility", "reasoning", "web_search"})
DEFAULT_MODEL_ROUTES: Dict[str, str] = {
    "utility": "gpt-4.1-mini",
    "reasoning": "gpt-5-mini",
    "web_search": "openai/gpt-4o-search-preview",
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
    reaches call sites that never plumbed an effort parameter (repo
    memory, insight extraction).
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

    def llm_completion(
        self,
        model: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float = 1,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        resolved_model = self.resolve_model(model, default_role="utility")
        if reasoning_effort is None:
            reasoning_effort = self.model_router.effort_for(
                model, default_role="utility"
            )
        effective_effort, kwargs = _prepare_effort(
            resolved_model, reasoning_effort, kwargs
        )
        response = self._run_sync(
            "completion",
            resolved_model,
            lambda: completion(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                drop_params=True,
                **_effort_kwargs(effective_effort),
                **kwargs,
            ),
        )
        return self._content(response)

    def llm_completion_with_system_prompt(
        self,
        model: Optional[str],
        system_prompt: str,
        user_message: str,
        temperature: float = 1,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self.llm_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    def llm_multiple_completions(
        self,
        models: Sequence[Optional[str]],
        messages: List[Dict[str, str]],
        temperature: float = 1,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        resolved_models = [
            self.resolve_model(model, default_role="utility") for model in models
        ]

        async def _run() -> List[str]:
            tasks = []
            for requested, model in zip(models, resolved_models):
                model_effort, model_kwargs = _prepare_effort(
                    model,
                    reasoning_effort
                    if reasoning_effort is not None
                    else self.model_router.effort_for(
                        requested, default_role="utility"
                    ),
                    kwargs,
                )
                tasks.append(
                    self._run_async(
                        "parallel completion",
                        model,
                        lambda model=model, model_effort=model_effort, model_kwargs=model_kwargs: acompletion(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            drop_params=True,
                            **_effort_kwargs(model_effort),
                            **model_kwargs,
                        ),
                    )
                )
            return [self._content(item) for item in await asyncio.gather(*tasks)]

        return asyncio.run(_run())

    def llm_completion_with_web_search(
        self,
        model: Optional[str],
        messages: List[Dict[str, str]],
        search_context_size: str = "medium",
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        resolved_model = self.resolve_model(model, default_role="web_search")
        if reasoning_effort is None:
            reasoning_effort = self.model_router.effort_for(
                model, default_role="web_search"
            )
        kwargs.pop("temperature", None)
        response = self._run_sync(
            "web-search completion",
            resolved_model,
            lambda: completion(
                model=resolved_model,
                messages=messages,
                web_search_options={"search_context_size": search_context_size},
                drop_params=True,
                **_effort_kwargs(reasoning_effort),
                **kwargs,
            ),
        )
        return self._content(response)

    def llm_multiple_completions_with_web_search(
        self,
        models: Sequence[Optional[str]],
        messages: List[Dict[str, str]],
        search_context_size: str = "medium",
        reasoning_efforts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        resolved_models = [
            self.resolve_model(model, default_role="web_search") for model in models
        ]
        kwargs.pop("temperature", None)

        async def _run() -> List[str]:
            tasks = []
            for index, model in enumerate(resolved_models):
                effort = (
                    reasoning_efforts[index]
                    if reasoning_efforts and index < len(reasoning_efforts)
                    else self.model_router.effort_for(
                        models[index], default_role="web_search"
                    )
                )
                tasks.append(
                    self._run_async(
                        "parallel web-search completion",
                        model,
                        lambda model=model, effort=effort: acompletion(
                            model=model,
                            messages=messages,
                            web_search_options={
                                "search_context_size": search_context_size
                            },
                            drop_params=True,
                            **_effort_kwargs(effort),
                            **kwargs,
                        ),
                    )
                )
            return [self._content(item) for item in await asyncio.gather(*tasks)]

        return asyncio.run(_run())

    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large",
        max_chars: Optional[int] = None,
    ) -> List[float]:
        """Create an embedding, returning an empty list on provider failure."""
        try:
            import openai

            if max_chars is None:
                env_val = os.getenv("KAPSO_EMBEDDING_MAX_CHARS")
                if env_val:
                    try:
                        parsed = int(env_val)
                        max_chars = parsed if parsed > 0 else None
                    except (TypeError, ValueError):
                        max_chars = None

            input_text = text if max_chars is None else text[:max_chars]
            response = openai.embeddings.create(model=model, input=input_text)
            return response.data[0].embedding
        except Exception:
            return []


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
