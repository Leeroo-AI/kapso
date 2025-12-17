# File: `packages/@n8n/task-runner-python/src/env.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Functions | `read_env`, `read_str_env`, `read_int_env`, `read_bool_env` |
| Imports | os, pathlib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Environment variable reading with file support

**Mechanism:** Provides utilities for reading configuration from environment with fallback patterns:
1. `read_env()`: Base function that checks direct env var first, then tries {VAR}_FILE pattern
2. File pattern enables Docker secrets/Kubernetes mounted files (e.g., N8N_RUNNERS_GRANT_TOKEN_FILE)
3. `read_str_env()`: Returns string with default value
4. `read_int_env()`: Parses integer with validation
5. `read_bool_env()`: Parses boolean (checks for "true" case-insensitive)
6. All functions support defaults when variable is not set
7. Raises ValueError for invalid conversions or file read errors

**Significance:** Implements the common pattern for secure configuration management in containerized environments. The _FILE suffix pattern is widely used in Docker/Kubernetes for mounting secrets without exposing them in environment variables. This abstraction allows config classes to read values uniformly whether from direct env vars or files. The type-safe reading functions prevent configuration errors early in the application lifecycle.
