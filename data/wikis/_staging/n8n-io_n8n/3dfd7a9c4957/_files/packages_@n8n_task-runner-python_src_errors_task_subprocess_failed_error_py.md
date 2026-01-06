# File: `packages/@n8n/task-runner-python/src/errors/task_subprocess_failed_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Classes | `TaskSubprocessFailedError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Subprocess failure error

**Mechanism:** Custom exception raised when the task executor subprocess exits with a non-zero status code unexpectedly.

**Significance:** Captures subprocess-level failures separate from code execution errors. Indicates process crashes or system issues.
