# File: `packages/@n8n/task-runner-python/src/errors/no_idle_timeout_handler_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `NoIdleTimeoutHandlerError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Missing idle timeout handler error

**Mechanism:** Custom exception raised when idle timeout functionality is requested but no handler has been registered.

**Significance:** Ensures proper idle timeout setup. Catches misconfiguration where timeout events would be silently ignored.
