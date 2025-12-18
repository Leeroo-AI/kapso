# File: `packages/@n8n/task-runner-python/src/errors/task_timeout_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Classes | `TaskTimeoutError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task execution timeout error

**Mechanism:** Custom exception raised when a task exceeds its configured maximum execution time and is forcibly terminated.

**Significance:** Enforces resource limits and prevents runaway tasks. Critical for system stability in multi-tenant environments.
