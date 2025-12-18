# File: `packages/@n8n/task-runner-python/src/errors/invalid_pipe_msg_content_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Classes | `InvalidPipeMsgContentError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Invalid pipe message content error

**Mechanism:** Custom exception for malformed message content received through the subprocess pipe communication channel.

**Significance:** Handles protocol violations in inter-process communication. Ensures message integrity between runner and executor processes.
