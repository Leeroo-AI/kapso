# File: `packages/@n8n/task-runner-python/tests/integration/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Functions | `broker`, `manager`, `manager_with_stdlib_wildcard`, `manager_with_env_access_blocked`, `manager_with_env_access_allowed`, `create_task_settings`, `wait_for_task_done`, `wait_for_task_error`, `... +1 more` |
| Imports | pytest_asyncio, src, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test fixtures and utilities

**Mechanism:** Provides pytest async fixtures that setup and teardown test infrastructure (broker, multiple manager configurations with varying security settings). Includes helper functions to create task settings with proper message format conversion (NODE_MODE_TO_BROKER_STYLE mapping), wait for task completion/error messages with predicates, and extract browser console messages from RPC logs.

**Significance:** Core testing infrastructure that enables comprehensive integration tests. Defines multiple manager fixtures with different security configurations (stdlib wildcards, env access control) to test security boundaries. Centralizes message handling and task lifecycle utilities used across all integration test files.
