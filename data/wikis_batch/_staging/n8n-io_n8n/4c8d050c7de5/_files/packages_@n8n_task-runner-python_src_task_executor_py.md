# File: `packages/@n8n/task-runner-python/src/task_executor.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 506 |
| Classes | `TaskExecutor` |
| Imports | io, json, logging, multiprocessing, os, src, sys, textwrap, traceback |

## Understanding

**Status:** ✅ Explored

**Purpose:** Executes user Python code in isolated subprocesses

**Mechanism:** Provides task execution with security isolation by:
1. Creating fork-server subprocesses with pipe communication channels
2. Supporting two execution modes: all_items (runs once on all data) and per_item (runs once per item)
3. Wrapping user code in a function to capture return values
4. Filtering builtins and sanitizing sys.modules based on security config
5. Executing code with custom print() that captures output for browser console
6. Handling timeouts, cancellations, and subprocess failures
7. Reading results via PipeReader thread from subprocess
8. Applying security restrictions (import validation, denied builtins, environment clearing)
9. Supporting continue-on-fail mode for error resilience

Uses multiprocessing forkserver context for process isolation and pipes for serialized result communication.

**Significance:** This is the security-critical execution layer that runs untrusted user code. It implements defense-in-depth with subprocess isolation, import restrictions, builtin filtering, and environment sanitization. The dual-mode execution (all_items vs per_item) supports different workflow node patterns in n8n. The custom print() capture enables user code debugging via browser console.
