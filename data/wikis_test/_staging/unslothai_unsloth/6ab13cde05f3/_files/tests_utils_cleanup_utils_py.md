# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Memory and directory cleanup utilities for tests

**Mechanism:** Clears Python GC, CUDA cache, LRU caches, and safely removes directories

**Significance:** Prevents memory leaks and ensures clean test environments between runs
