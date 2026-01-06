# File: `packages/@n8n/task-runner-python/src/errors/task_result_missing_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 11 |
| Classes | `TaskResultMissingError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Missing task result error

**Mechanism:** Custom exception raised when task execution completes but no result data is available, indicating a communication or execution failure.

**Significance:** Detects broken result pipeline. Ensures task completion always produces either a result or explicit error.
