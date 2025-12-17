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

**Purpose:** Configures structured logging with color support

**Mechanism:** Sets up logging configuration with intelligent formatting:
1. ColorFormatter applies ANSI color codes to log levels (blue=DEBUG, green=INFO, yellow=WARNING, red=ERROR/CRITICAL)
2. Detects NO_COLOR environment variable to disable colors
3. Detects TTY mode: when stdout is not a TTY (e.g., started by launcher), uses short form (message only)
4. Short form mode assumes timestamp and level are added by parent launcher
5. Full form includes timestamp, level, and message (tab-separated for parsing)
6. setup_logging() configures root logger with level from N8N_RUNNERS_LAUNCHER_LOG_LEVEL (default: INFO)
7. Hardcodes websocket loggers to INFO level to reduce verbosity

**Significance:** Provides user-friendly logging for development (with colors and timestamps) while being pipeline-friendly for production (short form). The dual-mode formatting enables the runner to work both standalone and as a subprocess of a launcher. The tab-separated format in full mode enables easy parsing by log aggregation tools. The websocket logger override prevents log spam from the WebSocket library.
