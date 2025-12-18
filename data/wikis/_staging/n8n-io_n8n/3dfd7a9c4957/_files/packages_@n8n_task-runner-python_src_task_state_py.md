# File: `packages/@n8n/task-runner-python/src/task_state.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Classes | `TaskStatus`, `TaskState` |
| Imports | dataclasses, enum, multiprocessing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task lifecycle state tracking

**Mechanism:** TaskStatus enum defines task states (pending, running, completed, etc.). TaskState dataclass holds task metadata including ID, status, process handle, and result. Used to track multiple concurrent tasks.

**Significance:** State management for task lifecycle. Enables proper tracking and cleanup of running tasks.
