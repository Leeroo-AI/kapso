"""Unified LLM model routing, retry behavior, and cost tracking."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import tiktoken
from litellm import embedding

# Suppress verbose LiteLLM logs.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Provider hard cap for embedding inputs (OpenAI text-embedding-3 family):
# requests above this raise a non-transient 400. Structural API limit, not a
# tunable. cl100k_base is the tokenizer for the text-embedding-3 models.
EMBEDDING_MAX_TOKENS = 8192
_EMBEDDING_ENCODING = "cl100k_base"

# Embedding is the ONLY model role left: every other completion moved to
# coding-agent CLI sessions (cli-only-inference design, 2026-08-26).
MODEL_ROLES = frozenset({"embedding"})
DEFAULT_MODEL_ROUTES: Dict[str, str] = {
    "embedding": "text-embedding-3-small",
}


class ModelRouter:
    """Resolve the embedding role while preserving explicit model strings.

    Reasoning-effort routing died with the completion surface: a CLI
    session's model/effort come from the `inference:` role specs, so a
    route value is just a bare model string.
    """

    def __init__(self, routes: Optional[Mapping[str, Any]] = None):
        supplied = dict(routes or {})
        unknown = sorted(set(supplied) - MODEL_ROLES)
        if unknown:
            raise ValueError(f"Unknown model role(s): {', '.join(unknown)}")

        merged = dict(DEFAULT_MODEL_ROUTES)
        for role, value in supplied.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Model route '{role}' must be a non-empty string")
            merged[role] = value.strip()
        self._routes = merged

    def resolve(
        self,
        model: Optional[str],
        *,
        default_role: str = "embedding",
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
        return requested

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
    """Embedding backend with bounded retries and cost tracking."""

    def __init__(
        self,
        models: Optional[Mapping[str, str] | ModelRouter] = None,
        retry_policy: Optional[Mapping[str, Any] | RetryPolicy] = None,
        *,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ):
        self.model_router = (
            models if isinstance(models, ModelRouter) else ModelRouter(models)
        )
        self.retry_policy = RetryPolicy.from_config(retry_policy)
        self._sleep = sleep_fn or time.sleep
        self._random = random_fn or random.random
        self._cumulative_cost = 0.0

    def get_cumulative_cost(self) -> float:
        return self._cumulative_cost

    def resolve_model(
        self,
        model: Optional[str],
        *,
        default_role: str = "embedding",
    ) -> str:
        return self.model_router.resolve(model, default_role=default_role)

    def _record_cost(self, response: Any) -> None:
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, Mapping):
            cost = hidden.get("response_cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self._cumulative_cost += float(cost)

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
