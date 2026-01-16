# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/storage.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 102 |
| Classes | `KeyNotExistsError`, `Storage` |
| Imports | os, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Implements a simple file-based key-value storage system that enables LLM agents to persist and retrieve data across conversation turns and sessions.

**Mechanism:** The `Storage` class (registered as `'storage'`) provides four operations via the `call()` method: (1) `put(key, value)` - saves string values to files under a configurable root directory, automatically creating subdirectories for hierarchical keys (e.g., `/users/data`); (2) `get(key)` - reads and returns file contents, raising `KeyNotExistsError` if not found; (3) `delete(key)` - removes the file for a given key; (4) `scan(key)` - recursively walks a directory and returns all key-value pairs as formatted strings. Keys are treated as file paths relative to `storage_root_path` (defaults to `DEFAULT_WORKSPACE/tools/storage`). The implementation uses utility functions `read_text_from_file()` and `save_text_to_file()` for I/O operations.

**Significance:** Provides persistent memory for LLM agents, enabling stateful workflows where intermediate results, user preferences, or computed data need to be stored and retrieved later. Used internally by other tools like `SimpleDocParser` for caching parsed documents. The hierarchical key structure supports organizing data by purpose or session.
