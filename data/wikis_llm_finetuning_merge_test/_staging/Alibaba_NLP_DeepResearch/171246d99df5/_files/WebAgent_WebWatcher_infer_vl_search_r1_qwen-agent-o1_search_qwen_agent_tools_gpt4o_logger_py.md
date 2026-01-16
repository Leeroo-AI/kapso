# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/logger.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 178 |
| Classes | `Logger` |
| Functions | `get_logger`, `add_file_handler_if_needed`, `log_with_location` |
| Imports | colorama, functools, inspect, logging, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides a comprehensive logging system with color-coded output and source location tracking for debugging and monitoring.

**Mechanism:** Key components:
- `get_logger()`: Factory function creating configured Python loggers with optional file handlers, tracking initialization state in `init_loggers` dict
- `add_file_handler_if_needed()`: Helper to conditionally add file handlers to existing loggers
- `@log_with_location` decorator: Wraps log methods to automatically prepend file location (filename:line) by inspecting the call stack, applies colorama colors
- `Logger` class: Static utility class providing colored logging methods:
  - Standard levels: `debug`, `info`, `warning`, `error`, `critical`
  - Foreground colors: `white`, `green`, `red`, `blue`, `black`, `cyan`
  - Background colors: `bwhite`, `bgreen`, `bred`, `bblue`, `bblack`
- Uses colorama library (Fore, Style, Back) for cross-platform ANSI color support

**Significance:** Core infrastructure component providing enhanced logging capabilities throughout the gpt4o toolkit. The source location tracking aids debugging by showing exactly where log messages originate, while color coding helps visually distinguish log levels and message types during development.
