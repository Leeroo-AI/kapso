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

**Mechanism:** Provides lazy evaluation of CI-specific environment variables through `__getattr__` mechanism. Defines variables like VLLM_CI_NO_SKIP, VLLM_CI_DTYPE, VLLM_CI_HEAD_DTYPE, VLLM_CI_HF_DTYPE, and VLLM_CI_ENFORCE_EAGER. Includes `is_set()` function to check if environment variables are explicitly set.

**Significance:** Allows CI tests to configure model dtype, skip behavior, and eagerness settings without hardcoding values, enabling flexible test configurations across different CI environments.
