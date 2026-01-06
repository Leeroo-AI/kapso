# File: `vllm/logger.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 303 |
| Classes | `_VllmLogger` |
| Functions | `init_logger`, `suppress_logging`, `enable_trace_function_call` |
| Imports | collections, contextlib, datetime, functools, json, logging, os, sys, types, typing, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralized logging configuration and custom logging utilities.

**Mechanism:** Configures Python's logging system for vLLM with custom formatters and handlers. The `_VllmLogger` class extends standard Logger with `debug_once()`, `info_once()`, `warning_once()` methods that only log unique messages once (using `@lru_cache`). Supports scoped logging (process/local/global) for distributed settings. `init_logger()` creates patched logger instances. Provides colored output via `ColoredFormatter` (respecting `NO_COLOR`), file info formatting, and configurable log levels. The `enable_trace_function_call()` function enables detailed function call tracing for debugging hangs/crashes. Configuration via `VLLM_LOGGING_*` environment variables.

**Significance:** Provides consistent, high-quality logging across vLLM. The "log once" functionality prevents log spam in loops while ensuring important messages aren't missed. Scoped logging is crucial for distributed inference where logs from all ranks would be overwhelming. The tracing functionality is invaluable for debugging hard-to-reproduce issues. Well-designed logging is essential for a production-grade inference system.
