# File: `packages/@n8n/task-runner-python/tests/integration/test_health_check.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Functions | `test_health_check_server_responds`, `test_health_check_server_ressponds_mid_execution` |
| Imports | aiohttp, asyncio, pytest, src, tests, textwrap |

## Understanding

**Status:** ✅ Explored

**Purpose:** Health check endpoint validation tests

**Mechanism:** Uses aiohttp client to verify the task runner's health check HTTP endpoint returns "OK" with 200 status. Tests include basic startup health check (with retry loop for server readiness) and health check responsiveness during active task execution (proving the health server remains responsive even while Python code runs).

**Significance:** Validates the health check server functionality that enables orchestration systems (like Kubernetes) to monitor task runner availability. Ensures health checks don't block during task execution, which is critical for production deployments.
