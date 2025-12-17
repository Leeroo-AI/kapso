# File: `vllm/scripts.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 17 |
| Functions | `main` |
| Imports | vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Deprecated CLI entry point

**Mechanism:** Provides backward compatibility shim for the old vllm.scripts.main() entry point. Issues deprecation warning and redirects to the new location at vllm.entrypoints.cli.main.main(). Simple wrapper function that delegates to the new CLI implementation.

**Significance:** Maintains backward compatibility for users who have old installation scripts or documentation referencing the original CLI entry point. Allows smooth migration to the restructured codebase without breaking existing workflows. Part of the package's commitment to stable interfaces.
