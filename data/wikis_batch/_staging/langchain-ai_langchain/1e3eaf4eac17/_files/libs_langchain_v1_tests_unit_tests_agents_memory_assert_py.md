# File: `libs/langchain_v1/tests/unit_tests/agents/memory_assert.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 56 |
| Classes | `MemorySaverAssertImmutable` |
| Imports | collections, functools, langgraph, os, tempfile |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enhanced in-memory checkpoint saver for testing that validates checkpoint immutability.

**Mechanism:** Extends InMemorySaver with additional assertion logic in the put method. Creates deep copies of checkpoints before storage and verifies they haven't been modified when retrieved. Uses temporary file-backed PersistentDict for storage. Optional put_sleep parameter for testing race conditions.

**Significance:** Critical testing utility ensuring agents don't mutate checkpoints after saving, which would break checkpoint integrity. Catches subtle bugs where code incorrectly modifies checkpoint state. Essential for validating correct checkpoint lifecycle management.
