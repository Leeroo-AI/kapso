# File: `packages/@n8n/task-runner-python/src/env.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Functions | `read_env`, `read_str_env`, `read_int_env`, `read_bool_env` |
| Imports | os, pathlib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Environment variable utilities

**Mechanism:** Provides typed helper functions for reading environment variables with default values and type conversion. read_str_env for strings, read_int_env for integers, read_bool_env for booleans, and read_env as generic base function.

**Significance:** Centralizes environment configuration reading. Ensures consistent, type-safe configuration loading across all modules.
