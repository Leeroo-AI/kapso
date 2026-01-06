# File: `tests/ci_envs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 52 |
| Functions | `is_set` |
| Imports | collections, os, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** CI-specific environment variable management

**Mechanism:** Provides lazy evaluation of CI-specific environment variables (e.g., `VLLM_CI_NO_SKIP`, `VLLM_CI_DTYPE`, `VLLM_CI_ENFORCE_EAGER`) through `__getattr__` and `__dir__` magic methods. Includes an `is_set()` helper to check if variables are explicitly set.

**Significance:** Allows CI tests to customize behavior through environment variables without hardcoding values. Enables testing all models in a family, controlling dtypes, and forcing eager execution mode for testing purposes.
