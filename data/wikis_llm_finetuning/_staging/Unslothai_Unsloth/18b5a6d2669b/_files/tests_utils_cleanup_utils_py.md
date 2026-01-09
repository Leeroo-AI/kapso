# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive memory management utilities for clearing memory leaks and cleaning up resources after model training and testing. Provides functions for clearing Python variables, LRU caches, CUDA memory, and filesystem directories.

**Mechanism:** clear_memory() performs multi-step cleanup: clears LRU caches across all modules, deletes specified global variables, runs multiple garbage collection passes, and empties CUDA cache with synchronization. clear_all_lru_caches() iterates through all loaded modules to find and clear functions with cache_clear() method, handling problematic modules gracefully. Preserves logging levels during cleanup. safe_remove_directory() provides safe directory deletion with error handling.

**Significance:** Essential for long-running test suites and benchmarks that load multiple large models sequentially. Prevents memory leaks from Python's caching mechanisms and CUDA's memory allocator. Without proper cleanup, tests would run out of GPU memory when evaluating multiple model configurations. Particularly important for vision models and large language models that consume significant GPU memory.
