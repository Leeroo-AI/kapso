# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/log.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Functions | `setup_logger` |
| Imports | logging, os |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides centralized logging configuration for the qwen_agent package through a `setup_logger()` function and pre-configured `logger` instance.

**Mechanism:** The `setup_logger()` function creates a named logger (`qwen_agent_logger`) with a StreamHandler that outputs to console. Log level defaults to INFO but switches to DEBUG if the `QWEN_AGENT_DEBUG` environment variable is set to '1' or 'true'. The formatter includes timestamp, filename, line number, log level, and message. A global `logger` instance is created at module load time for immediate use throughout the package.

**Significance:** This is a utility component that provides consistent logging across all qwen_agent modules. It enables developers to debug agent behavior by setting the QWEN_AGENT_DEBUG environment variable. The logger is imported and used in agent.py and other modules to log warnings, errors, and debug information during agent execution.
