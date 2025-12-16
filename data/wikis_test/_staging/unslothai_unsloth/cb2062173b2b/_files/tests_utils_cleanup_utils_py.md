# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive memory management utilities for test cleanup, addressing persistent memory leaks through aggressive cache clearing, garbage collection, and CUDA memory management.

**Mechanism:** Provides five key utilities: (1) clear_memory orchestrates full cleanup by clearing LRU caches, deleting specified variables (model, tokenizer, trainer, etc.), running 3 passes of garbage collection for circular references, emptying CUDA cache, synchronizing GPU, resetting memory stats, clearing JIT cache, and optionally reporting memory freed - preserves and restores logging levels during cleanup, (2) clear_all_lru_caches iterates through all loaded modules (avoiding problematic ones like torch.distributed, torchaudio) and calls cache_clear on any attribute with lru_cache decorator, also clears known caches in transformers/torch, (3) clear_specific_lru_cache clears single function cache, (4) monitor_cache_sizes reports cache sizes/hits/misses across modules for debugging, and (5) safe_remove_directory wraps shutil.rmtree with error handling for test artifact cleanup.

**Significance:** Critical for preventing memory leaks in long-running test suites. Machine learning tests often accumulate memory through model weights, optimizer states, and various caches. This utility enables tests to run sequentially without OOM errors by performing aggressive cleanup between tests. The LRU cache clearing is particularly important for cached functions in transformers library. Essential infrastructure for CI/CD environments with limited memory resources.
