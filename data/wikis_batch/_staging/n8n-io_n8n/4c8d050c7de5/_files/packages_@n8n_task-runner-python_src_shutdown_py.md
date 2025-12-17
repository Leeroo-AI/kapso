# File: `packages/@n8n/task-runner-python/src/shutdown.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 86 |
| Classes | `Shutdown` |
| Imports | asyncio, logging, signal, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Coordinates graceful application shutdown

**Mechanism:** Manages shutdown lifecycle by:
1. Registering signal handlers for SIGINT and SIGTERM at initialization
2. start_shutdown() orchestrates the shutdown sequence with timeout
3. Stops TaskRunner first (waits for running tasks, terminates if needed)
4. Stops HealthCheckServer if present
5. Flushes Sentry error tracking if enabled
6. Uses asyncio.Event (shutdown_complete) for async coordination
7. wait_for_shutdown() allows main() to block until shutdown completes
8. Returns exit code (0 for success, 1 for timeout/error)
9. start_auto_shutdown() variant for idle timeout (3s timeout, no grace period)
10. Prevents duplicate shutdown with is_shutting_down flag

**Significance:** Ensures clean application termination with proper resource cleanup. The signal handler registration enables graceful shutdown on Unix signals (Ctrl+C, container stop). The timeout mechanism prevents indefinite hangs during shutdown. The coordination pattern (shutdown_complete event + exit code) allows the main application to wait synchronously while the shutdown sequence runs asynchronously. The auto-shutdown variant supports the idle timeout feature for cost optimization.
