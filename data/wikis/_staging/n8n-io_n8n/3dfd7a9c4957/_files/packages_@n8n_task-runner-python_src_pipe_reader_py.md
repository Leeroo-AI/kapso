# File: `packages/@n8n/task-runner-python/src/pipe_reader.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 85 |
| Classes | `PipeReader` |
| Imports | json, multiprocessing, os, src, threading, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Subprocess pipe communication reader

**Mechanism:** PipeReader class reads length-prefixed JSON messages from subprocess pipes. Uses a background thread to continuously read from pipe, validates message framing, and queues received messages for processing.

**Significance:** Inter-process communication layer. Enables the task runner to receive results and status updates from executor subprocesses.
