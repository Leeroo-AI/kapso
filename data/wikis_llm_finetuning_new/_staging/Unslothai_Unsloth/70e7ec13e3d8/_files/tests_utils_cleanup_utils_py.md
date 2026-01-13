# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides comprehensive memory management utilities for cleaning up GPU memory, clearing Python caches, and safely removing temporary directories during and after tests.

**Mechanism:** The module exports five functions: (1) clear_memory handles complete cleanup by deleting specified global variables (model, tokenizer, trainer, etc.), running three garbage collection passes for circular references, clearing CUDA cache with synchronization, resetting memory stats, clearing JIT cache, and preserving logging levels throughout, (2) clear_all_lru_caches iterates through sys.modules to find and clear all functions with cache_clear() method, skipping problematic modules like torch.distributed, with warning suppression for compatibility, (3) clear_specific_lru_cache clears a single function's LRU cache, (4) monitor_cache_sizes inspects cache_info() across modules to report current sizes, hits, and misses sorted by size, (5) safe_remove_directory wraps shutil.rmtree with existence checks and error handling. The memory clearing approach is especially thorough, addressing persistent GPU memory leaks that occur during model loading/unloading cycles.

**Significance:** This module is essential for test stability and preventing out-of-memory errors during test suites that load multiple large models. The comprehensive LRU cache clearing addresses hidden memory retention in transformers and other libraries. The functions are used throughout vision model and benchmark tests to ensure clean state between test phases, and safe_remove_directory provides consistent cleanup of checkpoints and temporary model directories.
