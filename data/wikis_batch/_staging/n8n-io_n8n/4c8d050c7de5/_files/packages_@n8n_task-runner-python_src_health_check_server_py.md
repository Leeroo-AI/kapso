# File: `packages/@n8n/task-runner-python/src/health_check_server.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 50 |
| Classes | `HealthCheckServer` |
| Imports | asyncio, errno, logging, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Simple HTTP health check endpoint

**Mechanism:** Provides a minimal HTTP server for liveness/readiness checks:
1. Starts asyncio.Server bound to configured host and port
2. Responds to all requests with "HTTP/1.1 200 OK" and body "OK"
3. Uses raw HTTP response bytes (no framework overhead)
4. Handles EADDRINUSE error with helpful message
5. Logs actual port (supports OS-assigned ports for testing)
6. close() and wait_closed() for graceful shutdown
7. Swallows exceptions in request handler to prevent crashes

**Significance:** Enables container orchestration (Kubernetes, Docker Compose) to monitor runner health. The minimal implementation (raw HTTP, no parsing) ensures reliability and zero dependencies. The always-returns-200 behavior is appropriate for a basic liveness check (process is alive). The OS-assigned port support enables parallel testing. This is a common pattern in microservices for container health monitoring without adding framework weight.
