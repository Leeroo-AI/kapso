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

**Purpose:** Application entry point for Python task runner

**Mechanism:** Initializes the task runner application by:
1. Setting up logging configuration via `setup_logging()`
2. Optionally initializing Sentry for error tracking if enabled in config
3. Starting health check server if enabled in config
4. Creating and starting TaskRunner instance with configuration from environment
5. Setting up Shutdown coordinator with signal handlers (SIGINT, SIGTERM)
6. Waiting for shutdown completion and exiting with appropriate code
7. Blocking Windows platform at startup (not supported)

**Significance:** This is the main entry point that orchestrates the entire task runner lifecycle. It coordinates all major components (logging, monitoring, health checks, task execution, shutdown) and handles the application bootstrap sequence. The main() function follows a typical async application pattern with proper error handling and graceful shutdown support.
