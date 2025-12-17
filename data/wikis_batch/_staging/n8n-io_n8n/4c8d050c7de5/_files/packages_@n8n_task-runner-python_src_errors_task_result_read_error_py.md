# File: `packages/@n8n/task-runner-python/src/errors/task_result_read_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 4 |
| Classes | `TaskResultReadError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pipe read failure error.

**Mechanism:** Exception wrapping the original error when reading from the subprocess pipe fails. Stores `original_error` attribute for debugging. Fixed message "Failed to read result from child process."

**Significance:** IPC error handling. Raised when PipeReader encounters I/O errors, timeouts, or unexpected EOF. The original_error attribute helps diagnose whether it was a timeout, broken pipe, or other issue.
