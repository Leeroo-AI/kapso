# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/cache_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 57 |
| Classes | `JSONLCache` |
| Imports | fcntl, json, os |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a thread-safe JSONL-based caching utility for persisting key-value pairs to disk, used to cache web search and page content results.

**Mechanism:** The `JSONLCache` class manages a file-based cache using JSONL (JSON Lines) format. Key methods include:
- `_read_cache()`: Reads cache file with shared file lock (fcntl.LOCK_SH) for concurrent read access
- `_save_cache()`: Writes cache with exclusive file lock (fcntl.LOCK_EX) to prevent data corruption during writes
- `get(key, default)`: Retrieves cached values from in-memory dictionary
- `set(key, value)`: Stores values in in-memory cache
- `update_cache()`: Refreshes cache from disk and saves back, useful for synchronization

**Significance:** Critical utility component that enables persistent caching of expensive operations like web page fetching and search results. The file locking mechanism ensures safe concurrent access in multi-threaded environments, preventing race conditions when multiple processes access the same cache file.
