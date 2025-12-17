# File: `tests/test_envs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 456 |
| Classes | `TestEnvWithChoices`, `TestEnvListWithChoices`, `TestEnvSetWithChoices`, `TestVllmConfigureLogging` |
| Functions | `test_getattr_without_cache`, `test_getattr_with_cache`, `test_getattr_with_reset`, `test_is_envs_cache_enabled` |
| Imports | os, pytest, unittest, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Environment variable system testing

**Mechanism:** Tests envs module functionality including caching behavior, env_with_choices validation, env_list_with_choices parsing, env_set_with_choices deduplication, and VLLM_CONFIGURE_LOGGING handling. Verifies case sensitivity, invalid value handling, callable choices, and cache enable/disable cycles.

**Significance:** Ensures environment variable system correctly validates inputs, handles defaults, supports caching for performance, and provides proper error messages for misconfiguration.
