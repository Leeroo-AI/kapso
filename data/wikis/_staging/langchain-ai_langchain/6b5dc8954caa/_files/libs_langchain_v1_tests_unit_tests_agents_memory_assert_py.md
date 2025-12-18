# File: `libs/langchain_v1/tests/unit_tests/agents/memory_assert.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 56 |
| Classes | `MemorySaverAssertImmutable` |
| Imports | collections, functools, langgraph, os, tempfile |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a memory-based checkpointer that asserts checkpoint immutability for testing.

**Mechanism:** `MemorySaverAssertImmutable` extends `InMemorySaver` and maintains copies of all saved checkpoints in `storage_for_copies`. On each `put()` call, it verifies that previously saved checkpoints haven't been mutated by comparing with stored copies. Uses temporary file-backed `PersistentDict` for storage and supports optional sleep delays for testing timing-related issues.

**Significance:** Critical testing infrastructure that catches bugs where code incorrectly modifies checkpoint objects after saving them, ensuring proper immutability semantics that are essential for reproducible agent execution and debugging.
