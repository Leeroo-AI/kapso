# File: `packages/@n8n/task-runner-python/tests/unit/test_task_executor.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 204 |
| Classes | `TestTaskExecutorProcessExitHandling`, `TestTaskExecutorPipeCommunication`, `TestTaskExecutorLowLevelIO` |
| Imports | json, pytest, src, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task execution and IPC mechanism tests

**Mechanism:** Tests TaskExecutor.execute_process() behavior for different subprocess exit scenarios (SIGTERM raises TaskCancelledError, SIGKILL raises TaskKilledError, other non-zero raises TaskSubprocessFailedError, zero with empty pipe raises TaskResultReadError). Tests pipe communication protocol with length-prefixed JSON messages for successful results (PipeResultMessage with result/print_args) and errors (PipeErrorMessage). Tests low-level PipeReader._read_exact_bytes() handling of single/multiple reads, EOF, and write failures.

**Significance:** Validates the critical IPC mechanism between parent task runner and isolated Python execution subprocess. Ensures robust error handling for all subprocess termination scenarios and correct message serialization/deserialization. Core to the multi-process isolation architecture.
