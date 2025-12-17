# File: `packages/@n8n/task-runner-python/src/errors/task_timeout_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Classes | `TaskTimeoutError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task execution timeout error.

**Mechanism:** Exception with human-friendly message including the timeout duration and proper pluralization ("second" vs "seconds"). Stores `task_timeout` attribute with the configured limit.

**Significance:** User-facing timeout notification. Raised when task execution exceeds the configured timeout. Not sent to Sentry (expected operational behavior). Triggers subprocess termination and cleanup.
