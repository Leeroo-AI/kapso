# File: `vllm/_bc_linter.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 54 |
| Functions | `bc_linter_skip`, `bc_linter_skip`, `bc_linter_skip`, `bc_linter_include`, `bc_linter_include`, `bc_linter_include` |
| Imports | collections, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backward compatibility linting decorators

**Mechanism:** Provides decorator functions (bc_linter_skip, bc_linter_include) for controlling which code elements are checked by the backward compatibility linter. These decorators mark functions, methods, and classes as either exempt from BC checks or explicitly included. Works in conjunction with the BCLinter infrastructure to manage API stability enforcement.

**Significance:** Part of the development tooling infrastructure that helps maintain API stability. Allows developers to explicitly opt-in or opt-out specific code elements from backward compatibility checking, providing fine-grained control over what changes are monitored for BC breaks.
