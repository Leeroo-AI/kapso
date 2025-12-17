# File: `packages/@n8n/task-runner-python/src/errors/task_runtime_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 16 |
| Classes | `TaskRuntimeError` |
| Imports | typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** User code execution error.

**Mechanism:** Exception constructed from TaskErrorInfo typed dict (message, stack, description, stderr). Extracts error details from subprocess and makes them available as attributes: `stack_trace` for traceback and `description` for additional context.

**Significance:** Core user-facing error type. Wraps Python exceptions raised during user code execution. Not sent to Sentry (user error, not infrastructure). Description may contain stderr or additional diagnostic info.
