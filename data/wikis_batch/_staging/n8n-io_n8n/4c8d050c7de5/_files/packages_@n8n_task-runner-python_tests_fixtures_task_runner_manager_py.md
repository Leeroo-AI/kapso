# File: `packages/@n8n/task-runner-python/tests/fixtures/task_runner_manager.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 127 |
| Classes | `TaskRunnerManager` |
| Imports | asyncio, os, pathlib, re, src, sys, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manages task runner subprocess lifecycle

**Mechanism:** Spawns the Python task runner as a subprocess (via main.py), configures environment variables (broker URI, timeouts, health check settings, custom options like stdlib allow lists), captures stdout/stderr streams into buffers, and extracts the dynamically assigned health check port via regex parsing of startup logs. Provides clean shutdown with graceful termination and forced kill fallback.

**Significance:** Essential test fixture that enables integration tests to start/stop real task runner instances in controlled environments. Allows tests to verify end-to-end behavior including WebSocket communication, health checks, and proper subprocess lifecycle management.
