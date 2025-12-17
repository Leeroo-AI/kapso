# File: `libs/langchain_v1/langchain/agents/middleware/tool_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 396 |
| Classes | `ToolRetryMiddleware` |
| Imports | __future__, asyncio, langchain, langchain_core, time, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically retries failed tool calls with configurable exponential backoff, exception filtering, and failure handling strategies, improving agent reliability by handling transient errors (network timeouts, rate limits, temporary service unavailability) without manual intervention.

**Mechanism:** The ToolRetryMiddleware intercepts tool execution in `wrap_tool_call`, wrapping the handler in a retry loop (initial attempt + max_retries). When an exception occurs, it checks if the exception type is retryable (matches retry_on tuple or passes retry_on callable), calculates a backoff delay using exponential growth with optional jitter, sleeps for the delay, and retries. If retries are exhausted or the exception isn't retryable, it handles failure per on_failure config: "continue" returns an error ToolMessage with details (letting the agent potentially recover), "error" re-raises the exception, or a custom callable formats the error message. The middleware supports per-tool filtering to apply retry logic selectively.

**Significance:** This middleware is essential for production-grade agent systems that interact with external services prone to transient failures. It implements industry-standard retry patterns (exponential backoff, jitter, max delay caps) to handle common failure modes like network glitches, rate limiting, and temporary service outages. The flexible exception filtering (tuple of types or custom callable) allows fine-grained control over what gets retried (e.g., only retry 5xx errors, not 4xx). The configurable failure handling supports different reliability strategies: graceful degradation with "continue" for fault-tolerant agents, or strict failure with "error" for critical operations. The per-tool filtering enables different retry policies for different tools based on their reliability characteristics.
