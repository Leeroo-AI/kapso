# File: `packages/@n8n/task-runner-python/src/shutdown.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 86 |
| Classes | `Shutdown` |
| Imports | asyncio, logging, signal, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Graceful shutdown signal handling

**Mechanism:** Shutdown class registers handlers for SIGTERM and SIGINT signals. Coordinates graceful shutdown by waiting for in-flight tasks to complete, cancelling pending work, and closing connections cleanly.

**Significance:** Essential for production reliability. Ensures clean shutdown without data loss when containers are stopped or scaled down.
