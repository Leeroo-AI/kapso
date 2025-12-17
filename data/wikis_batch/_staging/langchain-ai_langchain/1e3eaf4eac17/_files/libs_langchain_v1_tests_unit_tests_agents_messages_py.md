# File: `libs/langchain_v1/tests/unit_tests/agents/messages.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 28 |
| Imports | any_str, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides test helper functions to create message objects with flexible ID matching for unit tests. This is a workaround for a Pydantic issue where the `__eq__` method is ignored on subclassed strings.

**Mechanism:** Defines factory functions `_AnyIdHumanMessage` and `_AnyIdToolMessage` that create message instances and then assign an `AnyStr()` instance to the `id` field after creation. This post-creation assignment bypasses Pydantic validation issues and allows test assertions to match messages regardless of their actual ID values.

**Significance:** Essential test utility that enables writing robust unit tests for agent functionality without needing to predict or mock exact message IDs. This pattern is used throughout agent tests to verify message content while being flexible about system-generated IDs.
