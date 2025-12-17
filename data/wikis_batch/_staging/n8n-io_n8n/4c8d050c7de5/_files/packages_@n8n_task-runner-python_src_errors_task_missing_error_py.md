# File: `packages/@n8n/task-runner-python/src/errors/task_missing_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 12 |
| Classes | `TaskMissingError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unknown task ID reference error.

**Mechanism:** Exception raised when a message references a task_id not tracked in the runner's running tasks dictionary. Includes the task_id in the error message and notes it's "likely an internal error."

**Significance:** Internal consistency check. Catches protocol violations or state management bugs. Should never occur in normal operation - indicates broker/runner message ordering issues or duplicate task IDs.
