# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_sync_async_wrappers.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 426 |
| Classes | `TestSyncAsyncMiddlewareComposition`, `SyncOnlyMiddleware`, `SyncOnlyMiddleware`, `AsyncOnlyMiddleware`, `AsyncOnlyMiddleware`, `BothSyncAsyncMiddleware`, `BothSyncAsyncMiddleware`, `SyncOnlyMiddleware`, `AsyncOnlyMiddleware`, `SyncOnlyMiddleware`, `AsyncOnlyMiddleware` |
| Functions | `search`, `calculator` |
| Imports | langchain, langchain_core, langgraph, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests sync/async tool call wrapper interoperability

**Mechanism:** Verifies middleware behavior across sync (invoke) and async (ainvoke) execution paths, testing that sync-only middleware raises NotImplementedError on async path, async-only raises on sync path, dual sync/async middleware uses appropriate implementation per path, decorator-created middleware respects sync/async constraints, and mixed middleware compositions fail gracefully when incompatible.

**Significance:** Critical for ensuring middleware correctly declares and enforces its sync/async capabilities, preventing runtime errors from mismatched execution contexts and validating the framework's sync/async delegation strategy for wrap_tool_call hooks.
