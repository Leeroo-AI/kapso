# File: `WebAgent/WebDancer/demos/tools/private/cache_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 57 |
| Classes | `JSONLCache` |
| Imports | fcntl, json, os |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides `JSONLCache` class for persistent key-value caching using JSONL (JSON Lines) file format.

**Mechanism:** The cache stores data as JSONL where each line is `{"key": ..., "value": ...}`. Key methods: (1) `__init__()` - initializes cache file path and loads existing data into memory, (2) `_read_cache()` - reads the JSONL file with shared file lock (LOCK_SH) for concurrent read safety, (3) `_save_cache()` - writes all cached data to file with exclusive lock (LOCK_EX) for write safety, (4) `get(key, default)` - retrieves value from in-memory dict, (5) `set(key, value)` - stores value in memory (note: doesn't auto-persist), (6) `update_cache()` - re-reads and re-saves cache file. Uses fcntl for POSIX file locking.

**Significance:** Utility component for caching search/visit results. Enables persistence of API responses to reduce redundant calls and improve performance across sessions.
