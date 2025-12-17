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

**Purpose:** Logging infrastructure configuration

**Mechanism:** Sets up vLLM's logging system with custom formatters (NewLineFormatter, ColoredFormatter), configurable log levels, and specialized methods. The _VllmLogger class extends standard Python logging with *_once methods (debug_once, info_once, warning_once) that print messages only once using lru_cache. Supports distributed logging with scope control (process/local/global). Includes function call tracing capability for debugging. Configuration driven by environment variables (VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_LEVEL, etc.).

**Significance:** Provides consistent logging across all vLLM components with features tailored for ML serving (distributed awareness, deduplication, colored output). The *_once methods prevent log spam in distributed settings. Function tracing helps debug hangs and crashes. Critical infrastructure that all other modules depend on for diagnostics and debugging.
