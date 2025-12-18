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

**Purpose:** Shared test utilities and helpers

**Mechanism:** Provides comprehensive test utilities including: RemoteOpenAIServer for spawning vLLM servers with OpenAI API, comparison functions for testing different configurations, distributed environment initialization, multiprocess test execution (fork/spawn), GPU memory monitoring (NVML/AMDSMI), process isolation decorators, pytest marks (large_gpu_test, multi_gpu_test), OpenAI API testing helpers, attention backend selection, and prompt generation utilities.

**Significance:** Core test infrastructure providing reusable utilities for integration testing, distributed testing, and performance validation. Essential for comprehensive test coverage across different hardware configurations and deployment scenarios.
