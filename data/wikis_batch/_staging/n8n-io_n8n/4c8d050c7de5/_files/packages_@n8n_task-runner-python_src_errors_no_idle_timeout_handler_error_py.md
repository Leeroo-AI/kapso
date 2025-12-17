# File: `packages/@n8n/task-runner-python/src/errors/no_idle_timeout_handler_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `NoIdleTimeoutHandlerError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Missing idle timeout handler error.

**Mechanism:** Exception raised when idle timeout is configured but `task_runner.on_idle_timeout` callback is not set before calling `start()`. Includes helpful message with configured timeout and fix instructions.

**Significance:** Internal programming error detection. Ensures the auto-shutdown feature is properly configured before the runner starts. Prevents silent failures where idle timeout expires but nothing happens.
