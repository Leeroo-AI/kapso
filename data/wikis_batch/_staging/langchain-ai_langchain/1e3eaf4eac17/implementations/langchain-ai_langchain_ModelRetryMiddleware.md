{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Error Handling]], [[domain::Resilience]], [[domain::Exponential Backoff]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Agent middleware that automatically retries failed model calls with configurable exponential backoff and exception filtering.

=== Description ===
ModelRetryMiddleware wraps the agent's model invocation with automatic retry logic. It intercepts model calls through the wrap_model_call and awrap_model_call hooks, catching exceptions and retrying based on configurable criteria. The middleware supports filtering which exceptions trigger retries (via exception types or callable predicates) and implements exponential backoff with optional jitter to prevent thundering herd problems.

When retries are exhausted, the middleware offers three failure handling modes: "continue" (return an AIMessage with error details, allowing the agent to proceed), "error" (re-raise the exception to halt execution), or a custom callable that formats error messages. This allows agents to gracefully handle transient failures like rate limits or network timeouts without manual retry logic.

The backoff calculation uses the formula: initial_delay * (backoff_factor ** retry_number), capped at max_delay. Jitter adds random variance (±25%) to prevent synchronized retry storms when multiple agents hit the same failure.

=== Usage ===
Use ModelRetryMiddleware when building agents that interact with unreliable external services (LLM APIs with rate limits, network calls prone to timeouts). Configure retry_on to target specific exceptions (e.g., RateLimitError, APITimeoutError) rather than all exceptions. Set max_retries based on expected failure frequency and acceptable latency. Use "continue" mode for resilient agents that should adapt to failures, or "error" mode for strict execution requirements.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/middleware/model_retry.py libs/langchain_v1/langchain/agents/middleware/model_retry.py]

=== Signature ===
<syntaxhighlight lang="python">
class ModelRetryMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None: ...

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage: ...

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| max_retries
| int
| 2
| Maximum retry attempts after initial call (must be >= 0)
|-
| retry_on
| tuple[Exception] or Callable
| (Exception,)
| Exception types to retry, or predicate function
|-
| on_failure
| "continue", "error", or Callable
| "continue"
| Behavior when all retries exhausted
|-
| backoff_factor
| float
| 2.0
| Exponential backoff multiplier (0.0 for constant delay)
|-
| initial_delay
| float
| 1.0
| Initial delay in seconds before first retry
|-
| max_delay
| float
| 60.0
| Maximum delay between retries (caps exponential growth)
|-
| jitter
| bool
| True
| Add random ±25% variance to delays
|}

=== RetryOn Types ===
{| class="wikitable"
|-
! Type
! Description
|-
| tuple[type[Exception], ...]
| Retry if exception matches any type in tuple
|-
| Callable[[Exception], bool]
| Retry if callable returns True for exception
|}

=== OnFailure Types ===
{| class="wikitable"
|-
! Type
! Description
|-
| "continue"
| Return AIMessage with formatted error details
|-
| "error"
| Re-raise the exception to halt agent execution
|-
| Callable[[Exception], str]
| Custom function to format error message for AIMessage
|}

=== Methods ===
{| class="wikitable"
|-
! Method
! Returns
! Description
|-
| wrap_model_call(request, handler)
| ModelResponse or AIMessage
| Synchronous retry wrapper for model calls
|-
| awrap_model_call(request, handler)
| ModelResponse or AIMessage
| Asynchronous retry wrapper for model calls
|}

== Usage Examples ==

=== Basic Usage with Defaults ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain_openai import ChatOpenAI

# Default: 2 retries, exponential backoff, continue on failure
middleware = ModelRetryMiddleware()

agent = create_agent(
    model=ChatOpenAI(),
    tools=[search_tool],
    middleware=[middleware]
)

# Transient failures automatically retried
response = agent.invoke({"messages": [{"role": "user", "content": "Search for news"}]})
</syntaxhighlight>

=== Retry Specific Exceptions ===
<syntaxhighlight lang="python">
from anthropic import RateLimitError
from openai import APITimeoutError
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Only retry rate limits and timeouts
middleware = ModelRetryMiddleware(
    max_retries=4,
    retry_on=(APITimeoutError, RateLimitError),
    backoff_factor=1.5,
)

# Other exceptions (e.g., InvalidRequestError) won't be retried
</syntaxhighlight>

=== Custom Exception Filtering ===
<syntaxhighlight lang="python">
from anthropic import APIStatusError
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


def should_retry(exc: Exception) -> bool:
    """Only retry server errors (5xx), not client errors (4xx)."""
    if isinstance(exc, APIStatusError):
        return 500 <= exc.status_code < 600
    return False


middleware = ModelRetryMiddleware(
    max_retries=3,
    retry_on=should_retry,
)
</syntaxhighlight>

=== Custom Error Message Formatting ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


def format_error(exc: Exception) -> str:
    """Provide user-friendly error message."""
    if "rate_limit" in str(exc).lower():
        return "The AI service is experiencing high demand. Please try again in a few moments."
    if "timeout" in str(exc).lower():
        return "The request took too long to complete. Please try a simpler query."
    return f"An error occurred: {type(exc).__name__}"


middleware = ModelRetryMiddleware(
    max_retries=3,
    on_failure=format_error,
)
</syntaxhighlight>

=== Constant Backoff (No Exponential Growth) ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Wait exactly 2 seconds between each retry
middleware = ModelRetryMiddleware(
    max_retries=5,
    backoff_factor=0.0,  # Disables exponential growth
    initial_delay=2.0,   # Constant 2 second delay
    jitter=False,        # No randomization
)
</syntaxhighlight>

=== Aggressive Retry with Short Delays ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Quick retries for transient network issues
middleware = ModelRetryMiddleware(
    max_retries=5,
    backoff_factor=1.5,
    initial_delay=0.5,  # Start with 500ms
    max_delay=10.0,     # Cap at 10 seconds
    jitter=True,
)

# Delay sequence: ~0.5s, ~0.75s, ~1.1s, ~1.7s, ~2.5s (with jitter)
</syntaxhighlight>

=== Fail Fast Mode ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Re-raise exceptions instead of continuing
middleware = ModelRetryMiddleware(
    max_retries=2,
    on_failure="error",  # Raise exception after retries exhausted
)

# Use when failures should halt agent execution immediately
</syntaxhighlight>

=== Combining with Other Middleware ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

retry_middleware = ModelRetryMiddleware(max_retries=2)
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"dangerous_tool": True}
)

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[retry_middleware, hitl_middleware]
)

# Retry handles transient failures, HITL handles approvals
</syntaxhighlight>

=== Rate Limit Specific Configuration ===
<syntaxhighlight lang="python">
from openai import RateLimitError
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

# Tuned for OpenAI rate limits
middleware = ModelRetryMiddleware(
    max_retries=4,
    retry_on=(RateLimitError,),
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True,  # Prevent thundering herd
)

# Delay sequence: ~1s, ~2s, ~4s, ~8s (with jitter, capped at 60s)
</syntaxhighlight>

=== Monitoring Retry Behavior ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


def log_error(exc: Exception) -> str:
    """Log failures for monitoring."""
    import logging
    logger = logging.getLogger(__name__)

    logger.error(f"Model call failed after retries: {type(exc).__name__}: {exc}")

    # Return user-facing message
    return "The AI service is temporarily unavailable. Our team has been notified."


middleware = ModelRetryMiddleware(
    max_retries=3,
    on_failure=log_error,
)
</syntaxhighlight>

=== Timeout-Specific Handling ===
<syntaxhighlight lang="python">
import asyncio
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


def should_retry_timeout(exc: Exception) -> bool:
    """Retry timeouts but not other errors."""
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError))


middleware = ModelRetryMiddleware(
    max_retries=3,
    retry_on=should_retry_timeout,
    initial_delay=2.0,
    backoff_factor=1.5,
)
</syntaxhighlight>

=== Testing Retry Logic ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from unittest.mock import Mock


# Create middleware
middleware = ModelRetryMiddleware(
    max_retries=2,
    initial_delay=0.1,  # Short delay for testing
    jitter=False,        # Deterministic for testing
)

# Mock handler that fails twice then succeeds
call_count = 0


def mock_handler(request):
    global call_count
    call_count += 1
    if call_count < 3:
        raise Exception("Transient failure")
    return {"result": "success"}


# Test retry behavior
result = middleware.wrap_model_call(
    request={"model": "test"},
    handler=mock_handler
)

print(f"Succeeded after {call_count} attempts")
# Output: "Succeeded after 3 attempts"
</syntaxhighlight>

=== Exponential Backoff Calculation ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.model_retry import ModelRetryMiddleware

middleware = ModelRetryMiddleware(
    max_retries=5,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=False,  # For clarity
)

# Delay for each retry attempt:
# Attempt 0 (initial): 1.0s
# Attempt 1: 1.0 * 2^0 = 1.0s
# Attempt 2: 1.0 * 2^1 = 2.0s
# Attempt 3: 1.0 * 2^2 = 4.0s
# Attempt 4: 1.0 * 2^3 = 8.0s
# Attempt 5: 1.0 * 2^4 = 16.0s

# With jitter=True, each delay gets ±25% random variance
</syntaxhighlight>

=== Async Usage ===
<syntaxhighlight lang="python">
import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware.model_retry import ModelRetryMiddleware
from langchain_openai import ChatOpenAI

middleware = ModelRetryMiddleware(max_retries=3)

agent = create_agent(
    model=ChatOpenAI(),
    tools=[async_search_tool],
    middleware=[middleware]
)


async def run_agent():
    # Async retries with asyncio.sleep
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Search for news"}]
    })
    return response


result = asyncio.run(run_agent())
</syntaxhighlight>

=== Production Configuration Example ===
<syntaxhighlight lang="python">
from anthropic import RateLimitError, APITimeoutError, APIStatusError
from langchain.agents.middleware.model_retry import ModelRetryMiddleware


def is_retryable(exc: Exception) -> bool:
    """Production retry policy."""
    # Always retry rate limits and timeouts
    if isinstance(exc, (RateLimitError, APITimeoutError)):
        return True

    # Retry server errors but not client errors
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500

    # Don't retry other exceptions
    return False


def format_production_error(exc: Exception) -> str:
    """User-facing error messages."""
    if isinstance(exc, RateLimitError):
        return "Service capacity reached. Please try again in a moment."
    if isinstance(exc, APITimeoutError):
        return "Request timed out. Please try a simpler query."
    return "An unexpected error occurred. Please contact support if this persists."


middleware = ModelRetryMiddleware(
    max_retries=4,
    retry_on=is_retryable,
    on_failure=format_production_error,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=30.0,
    jitter=True,
)
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware]] - Base class for all middleware
* [[langchain-ai_langchain_HumanInTheLoopMiddleware]] - Human approval middleware
* [[principle::Exponential Backoff]]
* [[principle::Graceful Degradation]]
* [[principle::Resilience Patterns]]
* [[environment::Production AI Agents]]
* [[environment::Rate-Limited APIs]]
