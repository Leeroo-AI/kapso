# File: `vllm/_bc_linter.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 54 |
| Functions | `bc_linter_skip`, `bc_linter_skip`, `bc_linter_skip`, `bc_linter_include`, `bc_linter_include`, `bc_linter_include` |
| Imports | collections, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backward compatibility linter decorators for API stability.

**Mechanism:** Provides decorator functions (`bc_linter_skip` and `bc_linter_include`) that mark classes, methods, or functions for backward compatibility checking. The decorators work at multiple levels: class-level, method-level, and function-level. They return a modified version of the decorated object that can be analyzed by compatibility checking tools. The implementation uses a registry pattern to track which APIs should be checked for breaking changes across versions.

**Significance:** Ensures API stability across vLLM releases by providing tooling to detect breaking changes. This is crucial for a library with many downstream dependencies. The decorators allow developers to explicitly mark which parts of the API are stable and should not change in backward-incompatible ways, helping maintain semver compliance.
