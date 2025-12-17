# File: `packages/@n8n/task-runner-python/src/errors/invalid_pipe_msg_content_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 3 |
| Classes | `InvalidPipeMsgContentError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Invalid IPC message content error.

**Mechanism:** Exception raised when pipe message JSON content is malformed or has unexpected structure. Wraps the original message in a descriptive "Invalid pipe message content: {message}" format.

**Significance:** Part of IPC error handling. Raised when subprocess sends data that cannot be parsed or doesn't match expected schema. Helps debug communication issues between runner and subprocess.
