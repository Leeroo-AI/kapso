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

**Purpose:** Test sync/async tool call middleware composition behavior and error handling

**Mechanism:** Validates the strict sync/async middleware compatibility rules for wrap_tool_call and awrap_tool_call:

**Sync-Only Middleware (wrap_tool_call only):**
- Works correctly on sync invocation path (.invoke)
- Raises NotImplementedError on async path (.ainvoke)
- Call logging confirms sync method execution

**Async-Only Middleware (awrap_tool_call only):**
- Works correctly on async invocation path (.ainvoke)
- Raises NotImplementedError on sync path (.invoke)
- Call logging confirms async method execution

**Dual Sync/Async Middleware (both wrap_tool_call and awrap_tool_call):**
- Uses sync implementation on sync path
- Uses async implementation on async path
- No cross-contamination between paths

**Mixed Middleware Composition:**
- Multiple middleware on async path fails if any are sync-only
- Multiple middleware on sync path fails if any are async-only
- NotImplementedError identifies incompatible middleware

**Decorator Behavior:**
- @wrap_tool_call with sync function: sync-only, raises on async path
- @wrap_tool_call with async function: async-only, raises on sync path
- Proper error handling for path incompatibility

**Significance:** Critical for preventing runtime errors when middleware doesn't match the invocation mode - unlike model/agent hooks which can delegate between sync/async, tool call wrappers must explicitly support both modes. Tests ensure the framework enforces this requirement and provides clear error messages when middleware is invoked on an unsupported path.
