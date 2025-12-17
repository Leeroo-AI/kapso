# File: `packages/@n8n/task-runner-python/src/errors/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 31 |
| Imports | configuration_error, invalid_pipe_msg_content_error, invalid_pipe_msg_length_error, no_idle_timeout_handler_error, security_violation_error, task_cancelled_error, task_killed_error, task_missing_error, task_result_missing_error, task_result_read_error, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Error classes package exports.

**Mechanism:** Re-exports all 14 custom exception classes from submodules via `__all__` list. Provides single import point for ConfigurationError, SecurityViolationError, TaskCancelledError, TaskKilledError, TaskTimeoutError, WebsocketConnectionError, and various pipe/result error types.

**Significance:** Centralizes error hierarchy for the task runner. All error types are organized into categories: configuration errors, security violations, task lifecycle errors (cancelled/killed/timeout), IPC errors (pipe message errors), and result handling errors.
