# File: `packages/@n8n/task-runner-python/src/errors/task_subprocess_failed_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Classes | `TaskSubprocessFailedError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Non-zero subprocess exit error.

**Mechanism:** Exception for subprocess exits with non-zero codes, excluding SIGTERM (-15) and SIGKILL (-9) which have dedicated error types. Stores `exit_code` and optional `original_error` for diagnostic purposes.

**Significance:** Catches unexpected subprocess failures. Exit codes might indicate segfaults (-11), assertion failures, or other non-Python errors. Helps distinguish between Python exceptions (TaskRuntimeError) and process-level failures.
