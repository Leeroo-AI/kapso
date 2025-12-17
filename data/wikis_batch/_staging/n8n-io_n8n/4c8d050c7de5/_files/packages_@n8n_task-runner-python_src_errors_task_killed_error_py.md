# File: `packages/@n8n/task-runner-python/src/errors/task_killed_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 11 |
| Classes | `TaskKilledError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** SIGKILL process termination error.

**Mechanism:** Exception for forceful process termination via SIGKILL. Includes docstring documenting common causes: OOM killer, resource limit exceeded, manual operator intervention.

**Significance:** Distinguished from normal termination (SIGTERM) or user code errors. Indicates infrastructure-level issues rather than code problems. Helps operators diagnose container/system resource issues.
