# File: `packages/@n8n/task-runner-python/tests/unit/test_task_runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 70 |
| Classes | `TestTaskRunnerConnectionRetry` |
| Imports | pytest, src, unittest, websockets |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task runner connection unit tests

**Mechanism:** Tests WebSocket connection retry logic including exponential backoff, connection failure handling, and reconnection behavior after disconnects.

**Significance:** Validates network resilience. Ensures the runner properly handles broker connection issues and recovers gracefully.
