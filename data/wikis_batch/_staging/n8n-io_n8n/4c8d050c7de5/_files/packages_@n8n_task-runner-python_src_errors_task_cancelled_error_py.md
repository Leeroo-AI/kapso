# File: `packages/@n8n/task-runner-python/src/errors/task_cancelled_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 5 |
| Classes | `TaskCancelledError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task cancellation signal error.

**Mechanism:** Simple exception with fixed "Task was cancelled" message. No parameters needed - cancellation is a binary state. Raised when broker sends cancel message or runner shutdown initiates.

**Significance:** Part of task lifecycle management. Enables clean task abortion without treating it as a failure. Filtered from Sentry (expected operational event). Triggers graceful cleanup of running subprocess.
