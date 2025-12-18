# File: `packages/@n8n/task-runner-python/tests/integration/test_execution.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 437 |
| Functions | `test_all_items_with_success`, `test_all_items_with_error`, `test_all_items_with_continue_on_fail`, `test_per_item_with_success`, `test_per_item_with_explicit_json_and_binary`, `test_per_item_with_binary_only`, `test_per_item_with_filtering`, `test_per_item_with_continue_on_fail`, `... +12 more` |
| Imports | asyncio, pytest, src, tests, textwrap |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task execution integration tests

**Mechanism:** Comprehensive tests for code execution modes (all items, per item), error handling (continue on fail), data types (JSON, binary), filtering, imports, and security restrictions. Tests actual code execution through the full runner pipeline.

**Significance:** Critical test coverage. Validates end-to-end task execution behavior in realistic scenarios.
