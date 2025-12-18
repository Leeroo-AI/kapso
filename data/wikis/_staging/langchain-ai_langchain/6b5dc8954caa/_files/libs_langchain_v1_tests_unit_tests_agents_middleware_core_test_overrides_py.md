# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_overrides.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 378 |
| Classes | `TestModelRequestOverride`, `TestToolCallRequestOverride` |
| Imports | langchain, langchain_core |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests immutable request override functionality

**Mechanism:** Two test classes verify that ModelRequest.override() and ToolCallRequest.override() methods create new request instances with modified attributes while preserving original requests, testing single/multiple attribute changes, message/state modifications, None value handling, object identity preservation for unchanged attributes, and method chaining patterns.

**Significance:** Validates the request override pattern which is crucial for middleware to safely modify requests without side effects, ensuring immutability and proper copy-on-write semantics that prevent bugs from shared state mutations.
