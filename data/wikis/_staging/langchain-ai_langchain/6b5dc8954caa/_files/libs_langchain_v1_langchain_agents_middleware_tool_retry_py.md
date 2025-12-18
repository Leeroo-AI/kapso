# File: `libs/langchain_v1/langchain/agents/middleware/tool_retry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 396 |
| Classes | `ToolRetryMiddleware` |
| Imports | __future__, asyncio, langchain, langchain_core, time, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically retries failed tool calls with configurable exponential backoff and exception filtering.

**Mechanism:** Implements ToolRetryMiddleware that wraps tool execution in retry loop. Uses wrap_tool_call to intercept tool execution and call handler multiple times on failure. Checks if exception matches retry_on criteria (tuple of exception types or callable). Applies exponential backoff: delay = initial_delay * (backoff_factor ** attempt), capped at max_delay, with optional jitter (±25%). On final failure, handles based on on_failure: "continue" (returns formatted error ToolMessage), "error" (re-raises), or custom callable for error formatting.

**Significance:** Production reliability middleware that makes agents resilient to transient tool failures. Essential for agents using network tools, external APIs, or unreliable services. Implements industry-standard retry patterns (exponential backoff with jitter) while maintaining flexibility through exception filtering and custom error handling. Can target specific tools or apply globally.
