# File: `WebAgent/WebDancer/demos/utils/logs.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 51 |
| Functions | `check_macos`, `setup_logger` |
| Imports | logging, os, platform, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides logging configuration and pre-configured loggers for WebDancer.

**Mechanism:** Key components: (1) `check_macos()` - detects if running on macOS (Darwin), (2) `setup_logger()` - creates configured logger with both console (StreamHandler) and file (FileHandler) outputs, uses standard format with timestamp/filename/line/level/message, respects SEARCH_AGENT_DEBUG env var for debug mode, determines log directory based on OS (macOS uses 'logs', others use AGENT_PATH env var). Creates three pre-configured loggers: `logger` (general), `access_logger` (access logs), `error_logger` (error logs), each writing to separate .log files.

**Significance:** Infrastructure utility for debugging and monitoring WebDancer. Provides structured logging across the application with separate log files for different concerns (access, errors, general).
