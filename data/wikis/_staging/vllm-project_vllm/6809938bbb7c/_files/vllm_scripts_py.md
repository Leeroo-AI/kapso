# File: `vllm/scripts.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Functions | `main` |
| Imports | vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backward compatibility shim for deprecated CLI entry point.

**Mechanism:** Contains a single `main()` function that logs a deprecation warning and delegates to `vllm.entrypoints.cli.main.main()`. The warning informs users that `vllm.scripts.main()` is deprecated and they should either reinstall vLLM or use the new entry point directly. This maintains backward compatibility for scripts/tools that previously used the old import path while guiding users to the new location.

**Significance:** Ensures smooth migration for existing users and tools when vLLM reorganized its CLI structure. Rather than breaking existing workflows immediately, this shim provides a grace period with clear migration guidance. Demonstrates good deprecation practices: maintain old interface, log warnings, point to the new approach. Will likely be removed in a future major version.
