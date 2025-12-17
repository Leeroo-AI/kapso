# File: `tests/test_logger.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 557 |
| Classes | `CustomEnum`, `CustomClass` |
| Functions | `f1`, `f2`, `test_trace_function_call`, `test_default_vllm_root_logger_configuration`, `test_descendent_loggers_depend_on_and_propagate_logs_to_root_logger`, `test_logger_configuring_can_be_disabled`, `test_an_error_is_raised_when_custom_logging_config_file_does_not_exist`, `test_an_error_is_raised_when_custom_logging_config_is_invalid_json`, `... +15 more` |
| Imports | dataclasses, enum, json, logging, os, pytest, sys, tempfile, typing, unittest, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Logging system comprehensive testing

**Mechanism:** Tests trace_function_call, default logger configuration, logger hierarchy, custom logging config loading, VLLM_CONFIGURE_LOGGING and VLLM_LOGGING_CONFIG_PATH handling, prepare_object_to_dump serialization, RequestLogger output logging (streaming/delta/complete modes), truncation, multiprocessing log capture (fork/spawn), and logging in subprocesses.

**Significance:** Ensures logging system correctly configures handlers, propagates messages, handles custom configs, captures subprocess logs, and provides proper request/response logging for debugging and monitoring.
