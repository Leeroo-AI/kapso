# File: `packages/@n8n/task-runner-python/tests/unit/test_task_runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 70 |
| Classes | `TestTaskRunnerConnectionRetry` |
| Imports | pytest, src, unittest, websockets |

## Understanding

**Status:** ✅ Explored

**Purpose:** WebSocket connection resilience tests

**Mechanism:** Tests TaskRunner's WebSocket connection retry logic for broker connectivity. Mocks websockets.connect() to simulate ConnectionRefusedError (logs warnings, retries automatically until shutdown flag set) and InvalidStatus 403 authentication errors (logs error, raises immediately without retry). Verifies logger calls to ensure proper error reporting.

**Significance:** Ensures task runner handles transient network issues gracefully with automatic retries while failing fast on permanent errors like authentication failures. Critical for production reliability in distributed environments where the broker may restart or have temporary network issues.
