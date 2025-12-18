# File: `packages/@n8n/task-runner-python/src/errors/task_runtime_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 16 |
| Classes | `TaskRuntimeError` |
| Imports | typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task execution runtime error

**Mechanism:** Custom exception wrapping errors that occur during task code execution. Includes original exception details, traceback, and execution context.

**Significance:** Primary error type for user code failures. Preserves full error context for debugging while maintaining structured error handling.
