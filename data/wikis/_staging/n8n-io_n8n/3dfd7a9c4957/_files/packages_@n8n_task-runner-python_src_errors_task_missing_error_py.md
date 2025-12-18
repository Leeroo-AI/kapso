# File: `packages/@n8n/task-runner-python/src/errors/task_missing_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 12 |
| Classes | `TaskMissingError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task not found error

**Mechanism:** Custom exception raised when attempting to operate on a task ID that doesn't exist in the runner's task registry.

**Significance:** Catches race conditions and invalid task references. Prevents operations on completed or non-existent tasks.
