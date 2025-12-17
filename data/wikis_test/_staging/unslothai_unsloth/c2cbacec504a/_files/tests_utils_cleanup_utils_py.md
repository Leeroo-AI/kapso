# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides memory management and cleanup utilities for tests to prevent out-of-memory errors and ensure clean test isolation by clearing caches and freeing GPU/CPU memory.

**Mechanism:** Implements functions to clear Python garbage collection, empty CUDA cache, clear LRU function caches, monitor cache memory usage, and safely remove temporary directories with proper error handling and logging.

**Significance:** Critical for test reliability in GPU-limited environments, preventing memory leaks between tests and enabling long-running test suites to complete successfully without accumulating memory usage from previous test runs.
