# File: `tests/utils/cleanup_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 226 |
| Functions | `clear_memory`, `clear_all_lru_caches`, `clear_specific_lru_cache`, `monitor_cache_sizes`, `safe_remove_directory` |
| Imports | gc, logging, os, shutil, sys, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Memory management and cleanup utilities

**Mechanism:** Provides functions for clearing GPU memory, clearing LRU caches across all modules, monitoring cache sizes, and safely removing directories. Critical for preventing memory leaks in tests

**Significance:** Essential utility for test suite hygiene, ensuring clean state between tests and preventing GPU memory accumulation
