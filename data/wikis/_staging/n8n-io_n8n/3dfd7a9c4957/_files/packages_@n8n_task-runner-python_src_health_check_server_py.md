# File: `packages/@n8n/task-runner-python/src/health_check_server.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 50 |
| Classes | `HealthCheckServer` |
| Imports | asyncio, errno, logging, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** HTTP health check endpoint server

**Mechanism:** HealthCheckServer class creates an async HTTP server responding to health check requests. Returns 200 OK for valid requests, enabling container orchestration liveness/readiness probes.

**Significance:** Production infrastructure component. Enables Kubernetes health monitoring and automatic restart of unhealthy runner instances.
