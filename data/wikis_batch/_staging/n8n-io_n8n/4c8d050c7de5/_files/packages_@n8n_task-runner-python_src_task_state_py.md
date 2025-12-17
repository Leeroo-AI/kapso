# File: `packages/@n8n/task-runner-python/src/task_state.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Classes | `TaskStatus`, `TaskState` |
| Imports | dataclasses, enum, multiprocessing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task lifecycle state tracking model

**Mechanism:** Defines task state representation with:
1. TaskStatus enum with three states: WAITING_FOR_SETTINGS, RUNNING, ABORTING
2. TaskState dataclass containing task_id, status, process reference, and workflow/node context
3. Context() method that extracts workflow and node metadata for logging
4. Initializes new tasks in WAITING_FOR_SETTINGS status
5. Stores reference to subprocess (ForkServerProcess) for cancellation

**Significance:** Provides the data model for tracking task execution state throughout the task runner. The status enum enables proper state machine transitions (waiting → running → aborting). The workflow/node context fields enable detailed logging and error reporting. The process reference is critical for task cancellation. This simple model is used extensively by TaskRunner to manage concurrent task execution.
