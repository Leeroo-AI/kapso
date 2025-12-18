# File: `packages/@n8n/task-runner-python/tests/integration/test_health_check.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 40 |
| Functions | `test_health_check_server_responds`, `test_health_check_server_ressponds_mid_execution` |
| Imports | aiohttp, asyncio, pytest, src, tests, textwrap |

## Understanding

**Status:** ✅ Explored

**Purpose:** Health check server integration tests

**Mechanism:** Tests that health check HTTP endpoint responds correctly both when idle and during task execution. Validates 200 OK responses from the health check server.

**Significance:** Infrastructure validation. Ensures health checks work correctly for Kubernetes/container orchestration.
