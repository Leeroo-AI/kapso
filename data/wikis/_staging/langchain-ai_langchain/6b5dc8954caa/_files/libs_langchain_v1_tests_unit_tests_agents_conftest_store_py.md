# File: `libs/langchain_v1/tests/unit_tests/agents/conftest_store.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 58 |
| Imports | contextlib, langgraph |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides factory functions for creating store instances used in agent test fixtures.

**Mechanism:** Defines context manager functions (both sync and async) that yield `InMemoryStore` instances from LangGraph. Currently uses memory stores for all backends (memory, Postgres variants) as placeholder implementations, with fallback to memory when other backends aren't available.

**Significance:** Abstraction layer that allows test fixtures to request different store types while providing consistent interfaces, enabling future implementation of actual Postgres stores without changing test code.
