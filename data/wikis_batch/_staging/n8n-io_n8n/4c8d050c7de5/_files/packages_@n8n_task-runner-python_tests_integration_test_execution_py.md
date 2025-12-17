# File: `packages/@n8n/task-runner-python/tests/integration/test_execution.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 437 |
| Functions | `test_all_items_with_success`, `test_all_items_with_error`, `test_all_items_with_continue_on_fail`, `test_per_item_with_success`, `test_per_item_with_explicit_json_and_binary`, `test_per_item_with_binary_only`, `test_per_item_with_filtering`, `test_per_item_with_continue_on_fail`, `... +12 more` |
| Imports | asyncio, pytest, src, tests, textwrap |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end Python code execution tests

**Mechanism:** Tests both node execution modes (all_items processes entire dataset, per_item processes one item at a time). Validates success scenarios, error handling, continue-on-fail behavior, result formatting (json/binary fields, pairedItem tracking), filtering (returning None), and security controls. Tests include timeout handling, task cancellation, multiple import bypass attempts (globals/locals/__builtins__ access), loader/spec exploitation, name-mangled attributes, and environment variable access control.

**Significance:** Most comprehensive test suite validating the core functionality of the Python task runner. Ensures code execution works correctly across modes, security sandboxing prevents malicious code, error handling is robust, and environment access controls are properly enforced. Critical for maintaining security guarantees.
