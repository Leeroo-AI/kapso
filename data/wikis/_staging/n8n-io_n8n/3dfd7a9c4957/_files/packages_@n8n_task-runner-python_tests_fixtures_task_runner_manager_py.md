# File: `packages/@n8n/task-runner-python/tests/fixtures/task_runner_manager.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 127 |
| Classes | `TaskRunnerManager` |
| Imports | asyncio, os, pathlib, re, src, sys, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task runner subprocess manager fixture

**Mechanism:** TaskRunnerManager spawns task runner as subprocess with configurable environment, monitors output, handles startup/shutdown, and provides process lifecycle control for testing.

**Significance:** Test infrastructure for process-level testing. Allows integration tests to run the actual task runner binary.
