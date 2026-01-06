# File: `packages/@n8n/task-runner-python/src/errors/task_killed_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 11 |
| Classes | `TaskKilledError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task force-kill notification error

**Mechanism:** Custom exception raised when a task subprocess is forcibly terminated (SIGKILL), typically after timeout or unresponsive behavior.

**Significance:** Distinguishes between graceful cancellation and forced termination. Provides diagnostic info for debugging stuck tasks.
