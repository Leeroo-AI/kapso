# File: `libs/langchain_v1/tests/unit_tests/agents/conftest_store.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Imports | contextlib, langgraph |

## Understanding

**Status:** ✅ Explored

**Purpose:** Factory functions for creating store instances across different backends for agent testing.

**Mechanism:** Provides context managers for creating stores: in-memory, and PostgreSQL (sync/async with pipe/pool variants). All PostgreSQL implementations currently fall back to InMemoryStore as placeholders. Uses contextlib decorators for resource management.

**Significance:** Abstracts store creation for test parametrization. Stores provide key-value storage for agent memory and state. Placeholder implementations indicate infrastructure for future database backend testing.
