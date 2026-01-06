# File: `libs/langchain_v1/tests/unit_tests/agents/messages.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 28 |
| Imports | any_str, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides factory functions for creating message instances with flexible ID matching in tests.

**Mechanism:** Defines `_AnyIdHumanMessage` and `_AnyIdToolMessage` functions that create standard LangChain message objects but replace their `id` field with an `AnyStr()` instance post-construction. This workaround addresses a Pydantic limitation where `__eq__` on subclassed strings is ignored during model validation.

**Significance:** Testing utility that enables flexible message equality assertions without requiring exact ID matches, essential for testing agent conversations where message IDs are auto-generated and unpredictable.
