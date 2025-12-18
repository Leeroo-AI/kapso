# File: `packages/@n8n/task-runner-python/src/task_executor.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 506 |
| Classes | `TaskExecutor` |
| Imports | io, json, logging, multiprocessing, os, src, sys, textwrap, traceback |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sandboxed Python code executor

**Mechanism:** TaskExecutor runs user code in isolated subprocess with restricted builtins and controlled imports. Sets up secure execution environment, captures stdout/stderr, handles execution modes (all items vs per item), and communicates results via pipes.

**Significance:** Core execution engine. Implements the sandboxed code execution that makes the task runner secure and functional.
