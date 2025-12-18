# File: `packages/@n8n/task-runner-python/tests/integration/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Functions | `broker`, `manager`, `manager_with_stdlib_wildcard`, `manager_with_env_access_blocked`, `manager_with_env_access_allowed`, `create_task_settings`, `wait_for_task_done`, `wait_for_task_error`, `... +1 more` |
| Imports | pytest_asyncio, src, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test pytest fixtures

**Mechanism:** Provides pytest fixtures for broker and task runner manager setup with various configurations (stdlib wildcard, env access blocked/allowed). Includes helper functions for task settings creation and waiting for task completion/errors.

**Significance:** Test infrastructure. Enables clean, reusable setup for integration tests with different security configurations.
