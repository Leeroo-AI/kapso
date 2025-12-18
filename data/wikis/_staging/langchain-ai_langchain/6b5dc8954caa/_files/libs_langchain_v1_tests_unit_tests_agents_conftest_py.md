# File: `libs/langchain_v1/tests/unit_tests/agents/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 194 |
| Functions | `anyio_backend`, `deterministic_uuids`, `sync_store`, `async_store`, `sync_checkpointer`, `async_checkpointer` |
| Imports | collections, conftest_checkpointer, conftest_store, langgraph, os, pytest, pytest_mock, uuid |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides pytest fixtures for agent tests with configurable checkpointer and store backends.

**Mechanism:** Defines parametrized fixtures (`sync_checkpointer`, `async_checkpointer`, `sync_store`, `async_store`) that yield different backend implementations (memory, SQLite, Postgres with various connection modes). Uses `FAST_MODE` environment variable to skip slow backends during rapid development. Also provides utility fixtures for deterministic UUIDs and asyncio backend selection.

**Significance:** Central test infrastructure that enables agent tests to run against multiple persistence backends, ensuring compatibility and correctness across different storage implementations while maintaining fast test execution in development mode.
