# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_overrides.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 378 |
| Classes | `TestModelRequestOverride`, `TestToolCallRequestOverride` |
| Imports | langchain, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test immutable request modification via the override() method pattern

**Mechanism:** Validates the .override() method on request objects that enables middleware to create modified copies while preserving immutability:

**ModelRequest.override() Tests:**
- Single attribute override (system_prompt, tool_choice, messages)
- Multiple attribute override in one call
- Messages list modification
- model_settings dict override
- Setting attributes to None
- Object identity preservation for unchanged attributes
- Method chaining for sequential overrides

**ToolCallRequest.override() Tests:**
- tool_call dict modification
- State dict override
- Multiple attributes at once
- Tool instance replacement
- Common copy-and-modify pattern (e.g., `{**request.tool_call, "args": new_args}`)
- Object identity preservation
- Method chaining

**Override Patterns Validated:**
- Immutability: Original request unchanged after override
- Identity preservation: Unchanged references point to same objects
- Composability: Chaining multiple override() calls works correctly
- Copy semantics: Modified attributes create new instances

**Significance:** Critical for middleware safety and correctness - the override() pattern ensures middleware cannot accidentally mutate shared request objects. Tests verify that modifications are isolated, original requests remain intact, and the API supports common modification patterns like chaining and partial updates. This is fundamental to composable middleware that doesn't have side effects on other handlers.
