# Heuristic: Middleware Deprecation Patterns

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Middleware|https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware]]
* [[source::Doc|Middleware Types|libs/langchain_v1/langchain/agents/middleware/types.py]]
|-
! Domains
| [[domain::API_Design]], [[domain::Migration]], [[domain::Best_Practices]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

## Overview

Guidelines for handling deprecated middleware APIs and migrating to new patterns in LangChain agent middleware.

### Description

LangChain's middleware system is evolving, with several APIs being deprecated in favor of more explicit, immutable patterns. Understanding these deprecation patterns helps maintain code that will continue to work with future versions and follows best practices for middleware composition.

### Usage

Apply this heuristic when:
- Seeing `DeprecationWarning` messages in your middleware code
- Migrating from older middleware patterns to new APIs
- Writing new middleware and choosing between equivalent approaches
- Reviewing code that directly mutates `ModelRequest` attributes

## The Insight (Rule of Thumb)

### ModelRequest Immutability

* **Action:** Use `request.override(...)` instead of direct attribute assignment
* **Old Pattern (Deprecated):** `request.system_prompt = "..."`
* **New Pattern:** `request.override(system_message=SystemMessage(...))`
* **Trade-off:** More verbose but thread-safe and explicit

### Tool Retry Failure Modes

* **Action:** Use new failure mode names for `ToolRetryMiddleware`
* **Old Values:** `"raise"`, `"return_message"`
* **New Values:** `"error"`, `"continue"`
* **Migration:**
  - `on_failure="raise"` → `on_failure="error"`
  - `on_failure="return_message"` → `on_failure="continue"`

### Summarization Middleware Configuration

* **Action:** Use tuple-based configuration for summarization triggers
* **Old Pattern:** `max_tokens_before_summary=1000`
* **New Pattern:** `trigger=("tokens", 1000)`
* **Old Pattern:** `messages_to_keep=5`
* **New Pattern:** `keep=("messages", 5)`

## Reasoning

### Why Immutable Requests?

1. **Thread Safety:** Direct mutation is unsafe in async/concurrent contexts
2. **Debugging:** Immutable patterns make state changes traceable
3. **Composition:** Multiple middleware can propose changes without conflicts
4. **Testing:** Easier to test when state changes are explicit

### Deprecation Warning Pattern

From `libs/langchain_v1/langchain/agents/middleware/types.py:168-185`:
```python
# Special handling for system_prompt - convert to system_message
if name == "system_prompt":
    warnings.warn(
        "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
        "Use request.override(system_message=SystemMessage(...)) instead to create "
        "a new request with the updated value. Direct assignment will be removed "
        "in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    ...
    return

warnings.warn(
    f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
    f"Use request.override({name}=...) instead to create a new request "
    ...
)
```

### Tool Retry Migration

From `libs/langchain_v1/langchain/agents/middleware/tool_retry.py:192-204`:
```python
if on_failure == "raise":  # type: ignore[comparison-overlap]
    msg = (
        "on_failure='raise' is deprecated and will be removed "
        "in a future version. Use on_failure='error' instead."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    on_failure = "error"
elif on_failure == "return_message":  # type: ignore[comparison-overlap]
    msg = (
        "on_failure='return_message' is deprecated and will be removed "
        "in a future version. Use on_failure='continue' instead."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    on_failure = "continue"
```

### Summarization Migration

From `libs/langchain_v1/langchain/agents/middleware/summarization.py:206-220`:
```python
if "max_tokens_before_summary" in deprecated_kwargs:
    value = deprecated_kwargs["max_tokens_before_summary"]
    warnings.warn(
        "max_tokens_before_summary is deprecated. Use trigger=('tokens', value) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    trigger = ("tokens", value)

if "messages_to_keep" in deprecated_kwargs:
    value = deprecated_kwargs["messages_to_keep"]
    warnings.warn(
        "messages_to_keep is deprecated. Use keep=('messages', value) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

### Best Practice: stacklevel=2

* **Action:** Always use `stacklevel=2` in deprecation warnings
* **Reason:** Points to the caller's code, not the library code
* **Evidence:** Consistent pattern across all middleware deprecation warnings

## Related Pages

* [[applied_to::Implementation:langchain-ai_langchain_AgentMiddleware_class]]
* [[applied_to::Implementation:langchain-ai_langchain_middleware_hooks]]
* [[applied_to::Workflow:langchain-ai_langchain_Middleware_Composition]]
* [[applied_to::Principle:langchain-ai_langchain_Middleware_Configuration]]
