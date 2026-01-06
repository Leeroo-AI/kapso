# File: `packages/@n8n/task-runner-python/src/errors/task_cancelled_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 5 |
| Classes | `TaskCancelledError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task cancellation signal error

**Mechanism:** Custom exception raised when a running task is cancelled by user request or system intervention via the broker's cancel message.

**Significance:** Enables graceful task termination. Allows proper cleanup and status reporting when tasks are cancelled mid-execution.
