# File: `packages/@n8n/task-runner-python/src/main.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 72 |
| Functions | `main` |
| Imports | asyncio, logging, platform, src, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task runner application entry point

**Mechanism:** main() initializes logging, loads configuration from environment, optionally sets up Sentry, creates TaskRunner instance, and starts the async event loop. Handles graceful shutdown on signals.

**Significance:** Primary entry point for the Python task runner service. Orchestrates all component initialization and lifecycle management.
