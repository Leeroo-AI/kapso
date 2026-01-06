# File: `packages/@n8n/task-runner-python/tests/fixtures/local_task_broker.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 190 |
| Classes | `ActiveTask`, `LocalTaskBroker` |
| Imports | aiohttp, asyncio, collections, dataclasses, json, src, tests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Simulated WebSocket task broker fixture

**Mechanism:** LocalTaskBroker creates a local WebSocket server mimicking broker behavior. Handles runner registration, dispatches task offers, tracks active tasks via ActiveTask, and sends/receives protocol messages for controlled testing.

**Significance:** Core test infrastructure. Enables integration testing without requiring a real n8n instance or broker deployment.
