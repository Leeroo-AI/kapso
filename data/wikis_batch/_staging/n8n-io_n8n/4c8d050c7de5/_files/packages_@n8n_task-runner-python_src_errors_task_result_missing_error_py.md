# File: `packages/@n8n/task-runner-python/src/errors/task_result_missing_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 11 |
| Classes | `TaskResultMissingError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Missing subprocess result error.

**Mechanism:** Exception for cases where subprocess exits successfully (exit code 0) but didn't write any result to the communication pipe. No parameters - the situation is self-explanatory.

**Significance:** Internal error detection. Indicates the task_executor wrapper code failed to capture the return value or write it to the pipe. Shouldn't occur in normal operation - suggests a bug in task execution code.
