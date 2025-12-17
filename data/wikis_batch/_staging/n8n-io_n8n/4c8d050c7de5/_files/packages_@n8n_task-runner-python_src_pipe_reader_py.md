# File: `packages/@n8n/task-runner-python/src/pipe_reader.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 85 |
| Classes | `PipeReader` |
| Imports | json, multiprocessing, os, src, threading, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Background thread for reading subprocess results

**Mechanism:** Reads task results from subprocess via pipe:
1. Extends threading.Thread to run in background
2. Uses os.read() on file descriptor (not Connection.recv() which pickles)
3. Reads fixed 4-byte length prefix first
4. Then reads exactly that many bytes for message payload
5. Preallocates bytearray to avoid repeated reallocation
6. Parses JSON payload and validates structure
7. Validates message has 'print_args' list and either 'result' or 'error' (not both)
8. Stores result in pipe_message and message_size attributes
9. Captures any exceptions in error attribute
10. Closes connection when done

**Significance:** Enables non-blocking result retrieval from subprocesses. Running in a thread allows the main TaskRunner to handle WebSocket messages while waiting for subprocess results. The length-prefixed protocol prevents parsing incomplete messages. Using os.read() instead of multiprocessing's recv() is critical for handling large payloads efficiently. The validation ensures protocol compliance. The error capturing enables proper error handling in TaskExecutor.
