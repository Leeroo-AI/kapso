# File: `libs/langchain_v1/tests/unit_tests/agents/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 194 |
| Functions | `anyio_backend`, `deterministic_uuids`, `sync_store`, `async_store`, `sync_checkpointer`, `async_checkpointer` |
| Imports | collections, conftest_checkpointer, conftest_store, langgraph, os, pytest, pytest_mock, uuid |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest configuration providing fixtures for agent unit tests with multiple checkpoint and store backends.

**Mechanism:** Defines parametrized fixtures for sync/async checkpointers and stores with multiple backends (memory, SQLite, PostgreSQL with different connection modes). Uses FAST_MODE environment variable to skip slow backends during development. Provides deterministic UUIDs and asyncio backend configuration. Imports backend factories from separate conftest files.

**Significance:** Critical test infrastructure enabling comprehensive agent testing across different persistence backends. Parametrization ensures agents work correctly with all storage options. Fast mode accelerates development cycles while full mode validates production configurations.
