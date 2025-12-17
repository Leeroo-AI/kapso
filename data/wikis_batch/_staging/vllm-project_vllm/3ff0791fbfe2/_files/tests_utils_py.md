# File: `tests/utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1312 |
| Classes | `RemoteOpenAIServer`, `RemoteOpenAIServerCustom` |
| Functions | `compare_two_settings`, `compare_all_settings`, `init_test_distributed_environment`, `multi_process_parallel`, `error_on_warning`, `get_physical_device_indices`, `wait_for_gpu_memory_to_clear`, `fork_new_process_for_each_test`, `... +15 more` |
| Imports | anthropic, asyncio, cloudpickle, collections, contextlib, copy, functools, httpx, importlib, itertools, ... +20 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive test utilities library

**Mechanism:** Provides RemoteOpenAIServer for integration testing, compare_two_settings/compare_all_settings for configuration validation, multi_process_parallel for distributed testing, fork/spawn_new_process_for_each_test decorators, GPU memory monitoring, large_gpu_test/multi_gpu_test decorators, attention backend selection, prompt generation utilities, and flat_product for test parameterization.

**Significance:** Central testing infrastructure that enables API compatibility testing, multi-GPU scenarios, memory leak detection, and consistent test isolation across the entire test suite.
