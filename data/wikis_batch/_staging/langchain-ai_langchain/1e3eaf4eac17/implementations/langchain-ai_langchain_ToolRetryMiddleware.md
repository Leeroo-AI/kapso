{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Error Handling]], [[domain::Reliability]], [[domain::Retry Logic]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
ToolRetryMiddleware automatically retries failed tool calls with configurable exponential backoff, exception filtering, and failure handling to improve agent reliability when working with unreliable or rate-limited services.

=== Description ===
This middleware intercepts tool execution and implements sophisticated retry logic with exponential backoff. It provides fine-grained control over which exceptions trigger retries, how long to wait between attempts, and what to do when all retries are exhausted.

Key features:
* Exponential backoff with configurable parameters (base delay, multiplier, max delay)
* Optional random jitter to prevent thundering herd
* Exception type filtering (retry only on specific exceptions)
* Custom exception filtering via callable predicates
* Tool-specific retry policies (apply to specific tools or all tools)
* Configurable failure handling (continue, error, or custom formatting)
* Both sync and async support

The middleware uses the `wrap_tool_call` interception pattern to wrap tool execution without modifying tool implementations.

=== Usage ===
Use this middleware when:
* Working with unreliable external APIs or services
* Handling transient network failures
* Dealing with rate-limited services (with backoff)
* Improving agent resilience to temporary failures
* Implementing circuit breaker patterns
* Debugging: Retry specific exceptions to gather more information

== Code Reference ==
'''Source location:''' `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/tool_retry.py`

'''Signature:'''
<syntaxhighlight lang="python">
class ToolRetryMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        max_retries: int = 2,
        tools: list[BaseTool | str] | None = None,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None
</syntaxhighlight>

'''Import statement:'''
<syntaxhighlight lang="python">
from langchain.agents.middleware import ToolRetryMiddleware
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| max_retries || int || 2 || Maximum retry attempts after initial call (must be >= 0)
|-
| tools || list[BaseTool/str]/None || None || Tools to apply retry logic to. If None, applies to all tools
|-
| retry_on || RetryOn || (Exception,) || Exception types to retry, or callable that returns bool
|-
| on_failure || OnFailure || "continue" || Behavior when retries exhausted: "continue", "error", or custom callable
|-
| backoff_factor || float || 2.0 || Multiplier for exponential backoff. 0.0 = constant delay
|-
| initial_delay || float || 1.0 || Initial delay in seconds before first retry
|-
| max_delay || float || 60.0 || Maximum delay in seconds (caps exponential growth)
|-
| jitter || bool || True || Add random jitter (±25%) to delay
|}

=== Types ===
{| class="wikitable"
! Type !! Definition !! Description
|-
| RetryOn || tuple[type[Exception], ...] / Callable[[Exception], bool] || Exception types or predicate function
|-
| OnFailure || Literal["continue", "error"] / Callable[[Exception], str] || Failure handling strategy or custom formatter
|}

=== Backoff Calculation ===
Delay formula:
```python
delay = initial_delay * (backoff_factor ** retry_number)
delay = min(delay, max_delay)
if jitter:
    delay = delay * random.uniform(0.75, 1.25)
```

Examples:
* Retry 0: 1.0 * (2.0 ** 0) = 1.0s
* Retry 1: 1.0 * (2.0 ** 1) = 2.0s
* Retry 2: 1.0 * (2.0 ** 2) = 4.0s
* Retry 3: 1.0 * (2.0 ** 3) = 8.0s

=== Hook Methods ===
{| class="wikitable"
! Method !! Purpose !! Input !! Output
|-
| wrap_tool_call() || Intercept sync tool execution || ToolCallRequest, handler || ToolMessage or Command
|-
| awrap_tool_call() || Intercept async tool execution || ToolCallRequest, async handler || ToolMessage or Command
|}

=== Failure Handling ===
{| class="wikitable"
! on_failure Value !! Behavior !! Use Case
|-
| "continue" || Return ToolMessage with error, let model handle || Default: graceful degradation
|-
| "error" || Re-raise exception, stop execution || Strict: fail fast on tool errors
|-
| Callable || Call function(exc) -> str, return as ToolMessage || Custom error formatting
|}

== Usage Examples ==

=== Example 1: Basic Retry (Default Settings) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

# Retry all tools up to 2 times with exponential backoff
agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[
        ToolRetryMiddleware()  # max_retries=2, backoff_factor=2.0
    ],
)

# Tool failures automatically retried:
# - Initial call
# - Retry 1 (after ~1s)
# - Retry 2 (after ~2s)
# - If still failing, return error ToolMessage
</syntaxhighlight>

=== Example 2: Retry Specific Exceptions Only ===
<syntaxhighlight lang="python">
from requests.exceptions import RequestException, Timeout, ConnectionError

# Only retry network-related exceptions
agent = create_agent(
    "openai:gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=4,
            retry_on=(RequestException, Timeout, ConnectionError)
        )
    ],
)

# Retries network failures
# Other exceptions (ValueError, etc.) fail immediately
</syntaxhighlight>

=== Example 3: Custom Exception Filter ===
<syntaxhighlight lang="python">
from requests.exceptions import HTTPError

def should_retry(exc: Exception) -> bool:
    """Only retry on 5xx server errors."""
    if isinstance(exc, HTTPError):
        if exc.response is not None:
            return 500 <= exc.response.status_code < 600
    return False

agent = create_agent(
    "openai:gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            retry_on=should_retry
        )
    ],
)

# Retries only when server returns 5xx errors
# Client errors (4xx) fail immediately
</syntaxhighlight>

=== Example 4: Apply to Specific Tools ===
<syntaxhighlight lang="python">
# Retry only specific tools
agent = create_agent(
    "openai:gpt-4o",
    tools=[unreliable_search_tool, reliable_calculator_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=5,
            tools=["search", "database"],  # By name
            backoff_factor=1.5
        )
    ],
)

# Only search and database tools have retry logic
# Calculator tool fails immediately on error
</syntaxhighlight>

=== Example 5: Apply Using BaseTool Instances ===
<syntaxhighlight lang="python">
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    '''Search the database.'''
    return results

@tool
def send_email(recipient: str, message: str) -> str:
    '''Send an email.'''
    return status

agent = create_agent(
    "openai:gpt-4o",
    tools=[search_database, send_email],
    middleware=[
        ToolRetryMiddleware(
            max_retries=4,
            tools=[search_database],  # Pass BaseTool instance
        )
    ],
)

# Only search_database has retry logic
</syntaxhighlight>

=== Example 6: Custom Failure Formatting ===
<syntaxhighlight lang="python">
def format_error(exc: Exception) -> str:
    """Custom error message for model."""
    return f"Service temporarily unavailable: {type(exc).__name__}. Try a different approach or ask the user for help."

agent = create_agent(
    "openai:gpt-4o",
    tools=[external_api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            on_failure=format_error
        )
    ],
)

# Model receives custom-formatted error message
# Can guide model to alternative strategies
</syntaxhighlight>

=== Example 7: Raise Exception on Failure ===
<syntaxhighlight lang="python">
# Strict mode: stop execution on persistent tool failure
agent = create_agent(
    "openai:gpt-4o",
    tools=[critical_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            on_failure="error"  # Re-raise exception
        )
    ],
)

try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Use critical tool"}]})
except Exception as e:
    print(f"Critical tool failed after retries: {e}")
    # Handle appropriately
</syntaxhighlight>

=== Example 8: Constant Backoff (No Exponential Growth) ===
<syntaxhighlight lang="python">
# Fixed delay between retries
agent = create_agent(
    "openai:gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=5,
            backoff_factor=0.0,  # No exponential growth
            initial_delay=2.0,   # Always wait 2 seconds
            jitter=False         # No randomness
        )
    ],
)

# Retries every 2 seconds exactly (no variation)
</syntaxhighlight>

=== Example 9: Aggressive Retry with High Max Delay ===
<syntaxhighlight lang="python">
# For very unreliable services
agent = create_agent(
    "openai:gpt-4o",
    tools=[flaky_service_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=10,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=300.0,   # Cap at 5 minutes
            jitter=True
        )
    ],
)

# Retries up to 10 times with exponential backoff
# Delays capped at 5 minutes to prevent excessive waits
</syntaxhighlight>

=== Example 10: Rate Limiting Scenario ===
<syntaxhighlight lang="python">
from requests.exceptions import HTTPError

def is_rate_limited(exc: Exception) -> bool:
    """Check for 429 Too Many Requests."""
    if isinstance(exc, HTTPError):
        if exc.response is not None:
            return exc.response.status_code == 429
    return False

agent = create_agent(
    "openai:gpt-4o",
    tools=[rate_limited_api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=5,
            retry_on=is_rate_limited,
            backoff_factor=3.0,   # Aggressive backoff for rate limits
            initial_delay=5.0,    # Start with 5 second delay
            jitter=True
        )
    ],
)

# Handles rate limiting gracefully:
# - Retry 1: ~5s delay
# - Retry 2: ~15s delay
# - Retry 3: ~45s delay
# - etc.
</syntaxhighlight>

=== Example 11: Combining with Other Middleware ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import ToolCallLimitMiddleware

# Retry failures but enforce limits
agent = create_agent(
    "openai:gpt-4o",
    tools=[api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            tools=["api"],
        ),
        ToolCallLimitMiddleware(
            tool_name="api",
            thread_limit=20,  # Limit total successful calls
        ),
    ],
)

# Middleware execution:
# 1. Retry middleware wraps tool execution
# 2. Each retry attempt counts toward limit (if successful)
# 3. Failed retries don't count toward limit
# 4. Limit middleware blocks after 20 successful calls
</syntaxhighlight>

=== Example 12: No Jitter for Predictable Timing ===
<syntaxhighlight lang="python">
# Deterministic retry timing (useful for testing)
agent = create_agent(
    "openai:gpt-4o",
    tools=[test_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            jitter=False  # Predictable delays: 1s, 2s, 4s
        )
    ],
)
</syntaxhighlight>

== Implementation Details ==

=== Retry Loop ===
Both sync and async implementations use the same logic:
```python
for attempt in range(max_retries + 1):  # +1 for initial attempt
    try:
        return handler(request)
    except Exception as exc:
        attempts_made = attempt + 1

        if not should_retry_exception(exc, retry_on):
            return handle_failure(...)  # Exception not retryable

        if attempt < max_retries:
            delay = calculate_delay(attempt, ...)
            sleep(delay)  # or await asyncio.sleep(delay)
        else:
            return handle_failure(...)  # No more retries
```

=== Tool Filtering ===
The `_should_retry_tool` method checks if retry applies:
```python
def _should_retry_tool(self, tool_name: str) -> bool:
    if self._tool_filter is None:
        return True  # Apply to all tools
    return tool_name in self._tool_filter
```

If tool not in filter, `handler(request)` called directly (no retry wrapper).

=== Exception Filtering ===
The `should_retry_exception` helper determines if exception is retryable:
```python
def should_retry_exception(exc: Exception, retry_on: RetryOn) -> bool:
    if callable(retry_on):
        return retry_on(exc)
    else:
        return isinstance(exc, retry_on)
```

If exception not retryable, fails immediately without retries.

=== Backoff Calculation ===
The `calculate_delay` function computes wait time:
```python
def calculate_delay(
    attempt: int,
    backoff_factor: float,
    initial_delay: float,
    max_delay: float,
    jitter: bool
) -> float:
    if backoff_factor == 0.0:
        delay = initial_delay
    else:
        delay = initial_delay * (backoff_factor ** attempt)

    delay = min(delay, max_delay)

    if jitter:
        delay = delay * random.uniform(0.75, 1.25)

    return delay
```

Jitter range: ±25% (prevents synchronized retries across multiple agents).

=== Failure Handling ===
The `_handle_failure` method processes exhausted retries:
```python
def _handle_failure(
    self, tool_name: str, tool_call_id: str | None, exc: Exception, attempts_made: int
) -> ToolMessage:
    if on_failure == "error":
        raise exc

    if callable(on_failure):
        content = on_failure(exc)
    else:
        content = self._format_failure_message(...)

    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error"
    )
```

=== Default Failure Message Format ===
```python
f"Tool '{tool_name}' failed after {attempts_made} {attempt_word} with {exc_type}: {exc_msg}. Please try again."
```

Example:
```
Tool 'search' failed after 3 attempts with RequestException: Connection timeout. Please try again.
```

=== Tool Name Extraction ===
Tool name determined from request:
```python
tool_name = request.tool.name if request.tool else request.tool_call["name"]
```

Prefers `BaseTool.name` if available, falls back to tool call dictionary.

=== Wrap Pattern ===
The middleware uses `wrap_tool_call` pattern:
* Does NOT register new tools (`self.tools = []`)
* Intercepts execution of existing tools
* Handler callable executes the actual tool
* Handler can be called multiple times (for retries)

=== Async Implementation ===
Async version mirrors sync logic:
* Uses `await handler(request)` instead of `handler(request)`
* Uses `await asyncio.sleep(delay)` instead of `time.sleep(delay)`
* No other differences (same retry logic)

=== Attempts Counting ===
`attempts_made` includes initial call:
* Attempt 0 (initial) fails: attempts_made = 1
* Retry 0 fails: attempts_made = 2
* Retry 1 fails: attempts_made = 3

Error message reflects total attempts: "failed after 3 attempts"

=== Parameter Validation ===
The `validate_retry_params` helper ensures:
* `max_retries >= 0`
* `initial_delay >= 0`
* `max_delay >= 0`
* `backoff_factor >= 0`

Raises `ValueError` for invalid parameters.

=== Backwards Compatibility ===
Deprecated `on_failure` values handled with warnings:
* `"raise"` -> Use `"error"` (warning issued)
* `"return_message"` -> Use `"continue"` (warning issued)

=== Performance Considerations ===
* Retry delay adds latency (by design)
* Jitter adds 0-50% variation to configured delay
* No retry when tool not in filter (zero overhead)
* Exception checking is fast (isinstance or callable)

=== Integration with Agent Execution ===
The middleware integrates seamlessly:
# Agent calls model, receives tool calls
# For each tool call, agent executor invokes tool
# If ToolRetryMiddleware configured, it wraps execution
# On failure, middleware retries with backoff
# On success or exhausted retries, returns ToolMessage
# Agent continues with ToolMessage(s)

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware|AgentMiddleware]] - Base middleware class
* [[langchain-ai_langchain_ToolCallLimitMiddleware|ToolCallLimitMiddleware]] - Limit tool call counts
* [[Error Handling Patterns]] - Agent resilience strategies
* [[Exponential Backoff Guide]] - Retry timing strategies
* [[Tool Reliability Best Practices]] - Building resilient tool integrations
