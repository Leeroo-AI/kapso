# File: `packages/@n8n/task-runner-python/src/message_types/pipe.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Classes | `TaskErrorInfo`, `PipeResultMessage`, `PipeErrorMessage` |
| Imports | src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Subprocess pipe message definitions

**Mechanism:** Defines message types for communication between runner and executor subprocess: PipeResultMessage for successful task results, PipeErrorMessage for errors with TaskErrorInfo containing exception details.

**Significance:** Inter-process protocol contract. Standardizes result/error reporting from executor subprocesses.
