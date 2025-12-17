# File: `packages/@n8n/task-runner-python/src/message_types/pipe.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines messages passed via IPC pipes between processes

**Mechanism:** This module defines TypedDict classes for inter-process communication via pipes (not WebSocket). It defines three structured types: `TaskErrorInfo` (error details with message, description, stack trace, and stderr), `PipeResultMessage` (successful result with items array and print output), and `PipeErrorMessage` (error result with error info and print output). The `PipeMessage` union type represents either success or error outcomes. Both message variants include `print_args` - a list of lists capturing all arguments passed to `print()` calls during task execution. The `PrintArgs` type alias documents this as arguments to all print calls in the task. Items are imported from `broker.py` and represent n8n workflow execution data (`INodeExecutionData[]` in TypeScript).

**Significance:** This defines the data contract for task execution results passed from child processes back to the task runner via pipes. Unlike the broker/runner WebSocket messages, pipe messages are simpler and focused solely on execution outcomes. The inclusion of print arguments enables capturing console output from user Python code for debugging and logging without mixing it with the structured result data. The error structure provides comprehensive diagnostic information including stack traces and stderr. This IPC mechanism allows isolating task execution in separate processes for security and fault tolerance while efficiently returning results.
