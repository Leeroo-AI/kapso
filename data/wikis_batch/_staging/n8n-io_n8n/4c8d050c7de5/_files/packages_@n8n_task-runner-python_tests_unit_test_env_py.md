# File: `packages/@n8n/task-runner-python/tests/unit/test_env.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 181 |
| Classes | `TestReadEnv`, `TestReadStrEnv`, `TestReadIntEnv`, `TestReadBoolEnv` |
| Imports | os, pathlib, pytest, src, tempfile, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Environment variable reading utility tests

**Mechanism:** Tests read_env() and type-specific variants (read_str_env, read_int_env, read_bool_env) which support reading from direct env vars or from files (Docker secrets pattern with VAR_FILE suffix). Uses tempfile for file-based tests, unittest.mock.patch for env isolation. Validates precedence (direct over file), whitespace handling, multiline content, Unicode support, error handling (missing files, permission denied, invalid types).

**Significance:** Validates the foundational configuration loading mechanism used throughout the task runner. Docker secrets support is critical for production deployments with sensitive credentials. Ensures configuration loading is robust and handles edge cases properly.
