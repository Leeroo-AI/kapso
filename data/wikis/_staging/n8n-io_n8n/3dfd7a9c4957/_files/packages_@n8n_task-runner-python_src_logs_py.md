# File: `packages/@n8n/task-runner-python/src/logs.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 66 |
| Classes | `ColorFormatter` |
| Functions | `setup_logging` |
| Imports | logging, os, src, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Logging configuration and formatting

**Mechanism:** ColorFormatter adds ANSI color codes to log messages based on level (error=red, warning=yellow, etc.). setup_logging() configures root logger with appropriate format, level, and handlers for console output.

**Significance:** Developer experience improvement. Provides readable, color-coded log output for debugging and monitoring the task runner.
