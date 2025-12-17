# File: `packages/@n8n/task-runner-python/tests/fixtures/test_constants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralized test configuration constants

**Mechanism:** Defines shared constants for test execution: LOCAL_TASK_BROKER_WS_PATH (WebSocket endpoint path), TASK_RESPONSE_WAIT (3s default timeout for broker responses), TASK_TIMEOUT (2s for task execution), and GRACEFUL_SHUTDOWN_TIMEOUT (1s for subprocess termination).

**Significance:** Provides single source of truth for timing and configuration values used across test fixtures and integration tests, ensuring consistent behavior and making timeout adjustments easier to manage.
