# File: `packages/@n8n/task-runner-python/src/errors/task_result_read_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 4 |
| Classes | `TaskResultReadError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task result read failure error

**Mechanism:** Custom exception raised when the task result cannot be read or deserialized from the subprocess communication channel.

**Significance:** Catches serialization and I/O errors in result retrieval. Separates result availability from result accessibility issues.
